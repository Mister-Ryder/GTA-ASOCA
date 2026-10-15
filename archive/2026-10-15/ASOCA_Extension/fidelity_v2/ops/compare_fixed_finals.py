"""CPU-only matched GTA/U-Net fixed-final comparison on the internal four.

Explicit local directories relocate downloaded records; recorded remote paths
are checked as strings and never opened. No model/optimizer/CUDA is created.
All four patients remain in the report, including undefined surface distances.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
import tempfile

from verify_postfit import IDS, SUMMARY_FIELDS, EXPECTED_MANIFEST, EXPECTED_SOURCE, finite, require, same, summary_of

SOURCE_ALGORITHM = 'gta_fidelity_selected_python_path_and_bytes_v1'
RECIPE_FIELDS = ('fold', 'epochs', 'steps', 'batch_size', 'patch_size', 'seed', 'split_seed',
                 'lr', 'weight_decay', 'warmup_epochs', 'seg_loss', 'sampling', 'sampling_fov',
                 'no_ds', 'cohort')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    data = Path(path).read_bytes()
    require(not data or data.endswith(b'\n'), 'Incomplete JSONL final line: ' + str(path))
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


def source_identity(code):
    used = {'train.py', 'model.py', 'losses.py', 'classification_data.py', 'checkpoint_utils.py', 'data.py'}
    names = sorted(p.relative_to(code).as_posix() for p in code.rglob('*.py')
                   if '__pycache__' not in p.parts and (p.name in used or 'historical' in p.relative_to(code).parts))
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode()); h.update((code / name).read_bytes())
    return h.hexdigest(), {name: digest(code / name) for name in names}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def geometry(image):
    return {'size_xyz': list(image.GetSize()), 'spacing_xyz_mm': list(image.GetSpacing()),
            'origin_xyz_mm': list(image.GetOrigin()), 'direction': list(image.GetDirection())}


def read_image(path):
    """Work around Windows ITK NRRD readers that cannot open Unicode paths."""
    import SimpleITK as sitk
    path = Path(path)
    if os.name == 'nt' and not str(path).isascii():
        with tempfile.TemporaryDirectory(prefix='asoca_pair_read_') as directory:
            temporary = Path(directory) / 'input.nrrd'
            require(str(temporary).isascii(), 'Windows NRRD reader requires an ASCII temporary folder')
            shutil.copy2(path, temporary)
            return sitk.ReadImage(str(temporary))
    return sitk.ReadImage(str(path))


def check_geometry(actual, expected, case_id):
    require(actual['size_xyz'] == expected['size_xyz'], 'Original-grid size differs: ' + case_id)
    for field in ('spacing_xyz_mm', 'origin_xyz_mm', 'direction'):
        a, b = actual[field], expected[field]
        require(len(a) == len(b) and all(finite(x) and finite(y) and math.isclose(x, y, rel_tol=0, abs_tol=1e-6)
                                       for x, y in zip(a, b)), 'Original-grid geometry differs: ' + case_id + '/' + field)
    require(all(x > 0 for x in actual['spacing_xyz_mm']), 'Nonpositive original spacing')


def primary_from_pixels(prediction, truth, spacing_zyx):
    """Same pooled 6-neighbour surface convention; crop all support plus margin.

    Cropping only accelerates EDT. It includes every foreground voxel and one
    background layer, preserving original boundary erosion and physical units.
    """
    import numpy as np
    from scipy import ndimage as ndi
    require(prediction.shape == truth.shape and truth.ndim == 3, 'Prediction/reference array grid differs')
    require(np.isin(prediction, (0, 1)).all() and np.isin(truth, (0, 1)).all(), 'Nonbinary pixels')
    p, t = prediction.astype(bool), truth.astype(bool)
    pn, tn = int(p.sum()), int(t.sum())
    result = {'dice': 2 * int((p & t).sum()) / max(pn + tn, 1), 'assd_mm': None,
              'empty_prediction': pn == 0}
    if not pn or not tn:
        return result
    if np.array_equal(p, t):
        result['assd_mm'] = 0.; return result
    coordinates = np.nonzero(p | t)
    box = tuple(slice(max(0, int(c.min()) - 1), min(n, int(c.max()) + 2))
                for c, n in zip(coordinates, p.shape))
    p, t = p[box], t[box]
    structure = ndi.generate_binary_structure(3, 1)
    ps = p & ~ndi.binary_erosion(p, structure=structure, border_value=0)
    ts = t & ~ndi.binary_erosion(t, structure=structure, border_value=0)
    a = ndi.distance_transform_edt(~ts, sampling=spacing_zyx)[ps]
    b = ndi.distance_transform_edt(~ps, sampling=spacing_zyx)[ts]
    result['assd_mm'] = float((a.sum() + b.sum()) / (a.size + b.size))
    return result


def inspect_run(method, training, evaluation, manifest_path, raw_root, code, context):
    import SimpleITK as sitk
    training, evaluation = Path(training).resolve(), Path(evaluation).resolve()
    config_path, complete_path = training / 'config.json', training / 'TRAINING_COMPLETE.json'
    config, complete = read(config_path), read(complete_path)
    source, files = context['source_identity']
    expected = {'method': method, 'fold': 0, 'epochs': 200, 'steps': 125, 'batch_size': 2,
                'patch_size': 128, 'no_ds': False, 'smoke': False, 'cohort': 'internal28',
                'backbone_warmup_epochs': 0,
                'seg_loss': 'dice_ce', 'sampling': 'standard', 'sampling_fov': 'image-body',
                'source_hash_algorithm': SOURCE_ALGORITHM, 'source_sha256': source,
                'source_files_sha256': files, 'manifest_sha256': context['manifest_sha256'],
                'checkpoint_selection': 'fixed final', 'internal_selection_valid': True,
                'outer8_used_for_selection': False, 'allow_selection_overlap': False}
    require(all(config.get(k) == v for k, v in expected.items()), method + ': incompatible fixed-final/common recipe/source')
    split = config['internal_split']; train_ids = split['train']; ids = split['val']
    outer = context['manifest']['folds'][0]
    require(ids == IDS and len(train_ids) == len(set(train_ids)) == 28 and set(train_ids).isdisjoint(ids)
            and set(train_ids) | set(ids) == set(outer['train']) and set(train_ids + ids).isdisjoint(outer['val'])
            and config.get('trained_case_ids') == train_ids, method + ': internal28/4 patient isolation differs')
    final_hash = complete.get('final_sha256')
    require(isinstance(final_hash, str) and len(final_hash) == 64 and all(c in '0123456789abcdef' for c in final_hash),
            method + ': invalid immutable final checkpoint SHA')
    marker_expected = {'epoch': 199, 'smoke': False, 'manifest_sha256': context['manifest_sha256'],
                       'source_sha256': source, 'trained_case_ids': train_ids, 'final_sha256': final_hash,
                       'segmentation_updates': 25000, 'internal_selection_valid': True, 'checkpoint_selection': 'fixed final'}
    require(all(complete.get(k) == v for k, v in marker_expected.items()), method + ': training completion identity differs')
    history = lines(training / 'history.jsonl')
    require([row.get('epoch') for row in history] == list(range(200))
            and all(row.get('iterations') == 125 and finite(row.get('loss')) for row in history),
            method + ': complete finite fixed-budget history missing')
    weights = training / 'final.pt'
    if weights.exists():
        require(weights.is_file() and digest(weights) == final_hash, method + ': downloaded final weights differ')

    cases, sources = context['cases'], context['evaluation_sources']
    expected_json = {case_id + '.json' for case_id in ids} | {'summary.json', 'EVALUATION_COMPLETE.json'}
    require({p.name for p in evaluation.glob('*.json')} == expected_json, method + ': unexpected/missing patient JSON')
    require({p.name for p in evaluation.glob('*_prediction.nrrd')} == {i + '_prediction.nrrd' for i in ids},
            method + ': unexpected/missing prediction patients')
    summary = context['trajectory_module'].verify_evaluation(evaluation, ids, final_hash,
                context['manifest_sha256'], config, sources, cases, 199)
    require(summary.get('checkpoint_selection') == 'fixed final' and summary.get('formal_final_test') is False
            and summary.get('internal_selection_valid') is True and summary.get('primary_metrics') == ['dice', 'assd_mm']
            and summary.get('metric_protocol') == context['metric_protocol'], method + ': metric/final role differs')
    rows, pixel_audits = [], []
    for case_id in ids:
        row = read(evaluation / (case_id + '.json'))
        require(row.get('checkpoint_selection') == 'fixed final' and row.get('formal_final_test') is False
                and row.get('internal_selection_valid') is True
                and row.get('metric_protocol_version') == context['metric_protocol']['version']
                and row.get('assd_definition') == context['metric_protocol']['assd_mm']['definition']
                and row.get('assd_directional_mean_definition') == context['metric_protocol']['assd_directional_mean_mm']['definition']
                and row.get('surface_connectivity_neighbors') == 6 and row.get('surface_area_weighted') is False,
                method + ': case metric definition/fixed-final role differs: ' + case_id)
        require(all(key in row for key in SUMMARY_FIELDS) and finite(row['dice']) and 0 <= row['dice'] <= 1,
                method + ': missing/nonfinite metrics: ' + case_id)
        require(row['assd_mm'] is None or finite(row['assd_mm']) and row['assd_mm'] >= 0,
                method + ': invalid ASSD: ' + case_id)
        label_path = (raw_root / cases[case_id]['label_relative_path']).resolve()
        require(label_path.is_relative_to(raw_root), 'Reference path escapes explicit raw root')
        require(digest(label_path) == row['label_sha256'], method + ': raw GT differs: ' + case_id)
        reference = read_image(label_path)
        prediction_path = evaluation / (case_id + '_prediction.nrrd')
        predicted = read_image(prediction_path)
        grid = geometry(reference); check_geometry(grid, cases[case_id]['original_geometry'], case_id)
        check_geometry(geometry(predicted), grid, case_id)
        pixel_metrics = primary_from_pixels(sitk.GetArrayFromImage(predicted), sitk.GetArrayFromImage(reference),
                                            reference.GetSpacing()[::-1])
        for key in ('dice', 'assd_mm'):
            a, b = row[key], pixel_metrics[key]
            require((a is None and b is None) or finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6),
                    method + ': recorded primary metric not reproduced from saved pixels: ' + case_id + '/' + key)
        require(isinstance(row.get('empty_prediction'), bool) and row['empty_prediction'] == pixel_metrics['empty_prediction'],
                method + ': empty-mask flag differs from pixels: ' + case_id)
        rows.append(row)
        pixel_audits.append({'case_id': case_id, 'label_sha256': row['label_sha256'],
                             'prediction_sha256': row['prediction_sha256'], 'original_grid': grid,
                             'recomputed_primary_metrics': pixel_metrics})
        del predicted, reference
    require(lines(evaluation / 'case_metrics.jsonl') == rows, method + ': patient JSONL ledger differs')
    require(same(summary['overall'], summary_of(rows)), method + ': summary/Null policy differs from all four patients')
    for group in ('Normal', 'Diseased'):
        require(same(summary.get(group), summary_of([r for r in rows if r['group'] == group])),
                method + ': subgroup summary differs: ' + group)
    done = read(evaluation / 'EVALUATION_COMPLETE.json')
    expected_done = {'checkpoint_sha256': final_hash, 'manifest_sha256': context['manifest_sha256'],
                     'evaluation_sources': sources, 'case_ids': ids, 'summary_sha256': digest(evaluation / 'summary.json')}
    require(all(done.get(k) == v for k, v in expected_done.items()) and finite(done.get('wall_seconds'))
            and done['wall_seconds'] >= 0, method + ': evaluation completion differs')

    # Bind the final record to the original producer's remote logical paths.
    # Explicit local folders provide relocation; remote strings are never opened.
    trajectory = evaluation.parent
    protocol = read(trajectory / 'trajectory_protocol.json')
    expected_protocol = {'training_config_sha256': digest(config_path), 'manifest_sha256': context['manifest_sha256'],
                         'source_sha256': source, 'evaluation_sources': sources, 'case_ids': ids,
                         'split': 'internal_diagnostic', 'smoke': False, 'formal_final_test': False,
                         'outer8_used_for_selection': False}
    require(all(protocol.get(k) == v for k, v in expected_protocol.items()), method + ': trajectory protocol identity differs')
    remote_training = PurePosixPath(config['output'])
    require(remote_training.is_absolute() and '..' not in remote_training.parts and remote_training.name == 'training',
            method + ': unsafe/ambiguous recorded training path')
    remote_final = str(remote_training / 'final.pt')
    remote_evaluation = str(remote_training.parent / 'internal4_trajectory/final')
    candidates = [x for x in protocol.get('checkpoints', []) if x.get('name') == 'final']
    expected_candidate = {'path': remote_final, 'epoch': 199, 'name': 'final', 'checkpoint_sha256': final_hash}
    require(candidates == [expected_candidate], method + ': protocol final checkpoint differs')
    final_records = [x for x in lines(trajectory / 'validation_history.jsonl') if x.get('name') == 'final']
    expected_record = expected_candidate | {'evaluation_output': remote_evaluation,
        'summary_path': remote_evaluation + '/summary.json', 'split': 'internal_diagnostic',
        'role': summary['evaluation_role'], 'n_cases': 4, 'overall': summary['overall'],
        'seconds': done['wall_seconds'], 'smoke': False, 'checkpoint_selection': 'fixed final', 'formal_final_test': False}
    require(final_records == [expected_record], method + ': trajectory final ledger differs from verified evaluation')
    paths = [config_path, complete_path, training / 'history.jsonl', evaluation / 'summary.json',
             evaluation / 'case_metrics.jsonl', evaluation / 'EVALUATION_COMPLETE.json',
             trajectory / 'trajectory_protocol.json', trajectory / 'validation_history.jsonl']
    paths += [evaluation / (i + suffix) for i in ids for suffix in ('.json', '_prediction.nrrd')]
    paths += [(raw_root / cases[i]['label_relative_path']).resolve() for i in ids]
    if weights.exists(): paths.append(weights)
    expected_bytes = {(raw_root / cases[i]['label_relative_path']).resolve(): cases[i]['source_sha256']['label'] for i in ids}
    expected_bytes.update({evaluation / (row['case_id'] + '_prediction.nrrd'): row['prediction_sha256'] for row in rows})
    for path, expected_hash in expected_bytes.items():
        require(digest(path) == expected_hash, method + ': GT/prediction changed during pixel audit: ' + str(path))
    audit = {'method': method, 'local_training': str(training), 'local_evaluation': str(evaluation),
             'recorded_remote_training': config['output'], 'final_checkpoint_sha256': final_hash,
             'weights_bytes_checked': weights.exists(), 'model_loaded': False, 'pixel_primary_metrics_recomputed': True,
             'full_20point_trajectory_completion_checked': False,
             'source_sha256': source, 'manifest_sha256': context['manifest_sha256'], 'pixel_audits': pixel_audits,
             'inputs': [{'path': str(p), 'sha256': expected_bytes.get(p) or digest(p)} for p in paths]}
    return {'config': config, 'rows': rows, 'summary': summary, 'audit': audit}


def complete_stats(rows, field):
    values = [r[field] for r in rows if finite(r.get(field))]
    undefined = [r['case_id'] for r in rows if not finite(r.get(field))]
    return {'mean': statistics.mean(values) if not undefined else None,
            'sd': statistics.stdev(values) if not undefined and len(values) > 1 else None,
            'n_patients': len(rows), 'n_defined': len(values), 'undefined_case_ids': undefined,
            'reason': 'undefined values retained; no complete-four mean' if undefined else None}


def compare(gta_training, gta_evaluation, unet_training, unet_evaluation, manifest_path, raw_root, code):
    sys.dont_write_bytecode = True
    manifest_path, raw_root, code = map(lambda p: Path(p).resolve(), (manifest_path, raw_root, code))
    manifest = read(manifest_path)
    require(digest(manifest_path) == EXPECTED_MANIFEST and manifest.get('schema_version') == 1
            and manifest.get('dataset') == 'ASOCA' and manifest.get('complete') is True
            and not manifest.get('pilot_only') and len({x['case_id'] for x in manifest['cases']}) == len(manifest['cases']) == 40,
            'Requires the immutable complete ASOCA40 manifest')
    identity = source_identity(code)
    require(identity[0] == EXPECTED_SOURCE, 'Live code differs from the fixed fidelity_v2 experiment source')
    freeze_path = code.parent / 'configs/main_source_freeze.json'
    frozen = read(freeze_path); freeze_hash = digest(freeze_path)
    require(frozen.get('training_source_sha256') == EXPECTED_SOURCE
            and frozen.get('training_dependency_files') == identity[1], 'Source freeze training identity differs')
    for name in ('evaluate.py', 'metrics.py', 'evaluate_trajectory.py'):
        require(digest(code / name) == frozen['code_files'][name]['sha256'], 'Frozen evaluation module differs: ' + name)
    metrics = load_module('paired_frozen_metrics', code / 'metrics.py')
    context = {'manifest': manifest, 'manifest_sha256': digest(manifest_path),
               'cases': {case['case_id']: case for case in manifest['cases']},
               'source_identity': identity, 'metric_protocol': metrics.METRIC_PROTOCOL,
               'evaluation_sources': {name: digest(code / name) for name in ('evaluate.py', 'metrics.py', 'evaluate_trajectory.py')},
               'trajectory_module': load_module('paired_frozen_trajectory', code / 'evaluate_trajectory.py')}
    gta = inspect_run('gta', gta_training, gta_evaluation, manifest_path, raw_root, code, context)
    unet = inspect_run('unet', unet_training, unet_evaluation, manifest_path, raw_root, code, context)
    require(gta['config']['internal_split'] == unet['config']['internal_split'], 'Methods use different28/4 split identities')
    require(all(gta['config'].get(k) == unet['config'].get(k) for k in RECIPE_FIELDS), 'Declared common recipe differs')
    paired = []
    for a, b, ag, bg in zip(gta['rows'], unet['rows'], gta['audit']['pixel_audits'], unet['audit']['pixel_audits']):
        require(a['case_id'] == b['case_id'] and a['label_sha256'] == b['label_sha256'], 'Paired GT/patient differs')
        require(ag['original_grid'] == bg['original_grid'], 'Paired original-grid geometry differs')
        paired.append({'case_id': a['case_id'], 'group': a['group'], 'label_sha256': a['label_sha256'],
                       'gta_dice': a['dice'], 'unet_dice': b['dice'], 'dice_gta_minus_unet': a['dice'] - b['dice'],
                       'gta_assd_mm': a['assd_mm'], 'unet_assd_mm': b['assd_mm'],
                       'assd_gta_minus_unet_mm': a['assd_mm'] - b['assd_mm'] if finite(a['assd_mm']) and finite(b['assd_mm']) else None})
    for run in (gta, unet):
        for item in run['audit']['inputs']:
            require(digest(item['path']) == item['sha256'], 'Input changed during paired comparison: ' + item['path'])
    require(source_identity(code) == context['source_identity'] and digest(manifest_path) == context['manifest_sha256'],
            'Source/manifest changed during paired comparison')
    require(digest(freeze_path) == freeze_hash
            and all(digest(code / name) == value for name, value in context['evaluation_sources'].items()),
            'Frozen evaluation modules changed during paired comparison')
    return {'schema_version': 1, 'status': 'verified_fixed_final_pair', 'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'role': 'four training-internal development patients; previously used for method development, not independent final test',
            'case_ids': IDS, 'n_patients': 4, 'checkpoint_selection': 'fixed final', 'formal_final_test': False,
            'fivefold_result': False, 'significance_test': False, 'metric_protocol': context['metric_protocol'],
            'source_freeze_manifest_sha256': freeze_hash,
            'paired_difference_direction': 'GTA minus U-Net; higher Dice and lower ASSD are better', 'per_case': paired,
            'patient_statistics': {run['config']['method']: {key: complete_stats(run['rows'], key) for key in ('dice', 'assd_mm')}
                                   for run in (gta, unet)},
            'paired_statistics': {key: complete_stats(paired, key) for key in ('dice_gta_minus_unet', 'assd_gta_minus_unet_mm')},
            'sd_definition': 'sample SD across these four patients (ddof=1); not folds, seeds, or uncertainty of model superiority',
            'common_recipe': {k: gta['config'][k] for k in RECIPE_FIELDS},
            'attribution_boundary': 'Comparison includes architecture/capacity, BAS airway-backbone initialization, classifier pretraining, anatomy loss reweighting and auxiliary objectives; it cannot isolate Gaussian/prior effects.',
            'initialization_and_capacity': {run['config']['method']: {k: run['config'].get(k) for k in
                ('parameters', 'backbone_init', 'classifier_init', 'full_init', 'init_audits', 'classifier_pretraining_updates', 'segmentation_updates', 'pretrained_prior', 'disable_anatomy_reweighting')}
                for run in (gta, unet)},
            'audits': {run['config']['method']: run['audit'] for run in (gta, unet)}}


def save_result(result, output):
    output = Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), 'Use a fresh comparison output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'paired_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'paired_cases.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_case'][0])); writer.writeheader(); writer.writerows(result['per_case'])
    rows = []
    for method, stats in result['patient_statistics'].items():
        rows.extend({'method': method, 'metric': key, **value} for key, value in stats.items())
    with (output / 'patient_mean_sd.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = ['固定final：GTA与common-recipe U-Net配对比较；仅4例内部development病例。',
            '差值=GTA−U-Net：Dice正值更好，ASSD负值更好。SD为4患者样本SD，不是fold SD；未做显著性检验。',
            'ASSD未定义病例保留；该方法或配对差缺任一例时，不报告完整4例均值/SD。',
            '已从原GT/保存prediction像素重算Dice及pooled ASSD；未加载模型。权重bytes是否核对见audits。',
            '比较包含容量、BAS初始化、额外5120分类器预训练、解剖重加权及辅助目标差异；不作单模块归因。']
    for method, stats in result['patient_statistics'].items():
        for key, item in stats.items():
            text.append(f"{method} {key}: mean={item['mean']}; patient_SD={item['sd']}; defined={item['n_defined']}/4; undefined={item['undefined_case_ids']}")
    (output / 'comparison_notes.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gta-training', 'gta-evaluation', 'unet-training', 'unet-evaluation', 'manifest', 'raw-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parents[1] / 'code')
    args = parser.parse_args()
    result = compare(args.gta_training, args.gta_evaluation, args.unet_training, args.unet_evaluation,
                     args.manifest, args.raw_root, args.code)
    result['tool_sha256'] = digest(__file__)
    save_result(result, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'n_patients': 4,
                      'development_only': True, 'model_loaded': False}, allow_nan=False))


if __name__ == '__main__':
    main()
