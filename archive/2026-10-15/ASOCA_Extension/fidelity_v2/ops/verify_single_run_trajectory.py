"""Accept one downloaded U-Net fixed-final internal4 trajectory, entirely CPU.

Checks the collector's original files schema,20 checkpoints/80 patients, saved
prediction hashes/grids and final-four pixel Dice/pooled ASSD. Intermediate
checkpoint tensors need not be downloaded; their SHA agreement is explicitly
metadata evidence, not a fresh weight-byte audit. Never selects a best epoch.
"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

from compare_fixed_finals import (SOURCE_ALGORITHM, digest, read, lines, source_identity,
                                 load_module, geometry, check_geometry, inspect_run)
from verify_postfit import (IDS, NAMES, SUMMARY_FIELDS, EXPECTED_MANIFEST, EXPECTED_SOURCE,
                            finite, require, same, summary_of, source_guard)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def collection_files(root):
    """Use files[{relative_path,bytes,sha256}] as written by the real collector."""
    path = root / 'collection_manifest.json'
    manifest = read(path)
    require(isinstance(manifest.get('files'), list) and bool(manifest['files']), 'Collection files manifest missing/incomplete')
    covered = {}
    for item in manifest['files']:
        if item.get('missing'):
            continue
        name = item.get('relative_path')
        require(isinstance(name, str), 'Collection file lacks relative_path')
        relative = PurePosixPath(name.replace('\\', '/'))
        require(not relative.is_absolute() and '..' not in relative.parts and not any(':' in p for p in relative.parts),
                'Unsafe collection relative path')
        target = (root / Path(*relative.parts)).resolve()
        require(target.is_relative_to(root) and target not in covered, 'Duplicate/escaping collection path')
        require(target.is_file() and target.stat().st_size == item['bytes'] and valid_sha(item.get('sha256'))
                and digest(target) == item['sha256'], 'Collected file bytes/hash differ: ' + name)
        covered[target] = item['sha256']
    return manifest, covered


def prediction_header(path):
    import SimpleITK as sitk
    def header(file):
        reader = sitk.ImageFileReader(); reader.SetFileName(str(file)); reader.ReadImageInformation()
        return {'size_xyz': list(reader.GetSize()), 'spacing_xyz_mm': list(reader.GetSpacing()),
                'origin_xyz_mm': list(reader.GetOrigin()), 'direction': list(reader.GetDirection())}
    if os.name == 'nt' and not str(path).isascii():
        with tempfile.TemporaryDirectory(prefix='unet_trajectory_header_') as temporary:
            temporary = Path(temporary) / 'prediction.nrrd'
            require(str(temporary).isascii(), 'ITK needs an ASCII temporary path')
            shutil.copy2(path, temporary)
            return header(temporary)
    return header(path)


def context_for(code, manifest_path):
    manifest = read(manifest_path)
    require(digest(manifest_path) == EXPECTED_MANIFEST and manifest.get('complete') is True
            and not manifest.get('pilot_only') and manifest.get('schema_version') == 1 and manifest.get('dataset') == 'ASOCA'
            and len(manifest['cases']) == len({c['case_id'] for c in manifest['cases']}) == 40, 'Immutable ASOCA40 manifest differs')
    identity = source_identity(code)
    require(identity[0] == EXPECTED_SOURCE, 'Frozen training/model source differs')
    frozen = source_guard(code.parent)
    require(frozen['training_dependency_files'] == identity[1], 'Frozen per-file training identity differs')
    metric_module = load_module('single_run_frozen_metrics', code / 'metrics.py')
    return {'manifest': manifest, 'manifest_sha256': EXPECTED_MANIFEST,
            'cases': {c['case_id']: c for c in manifest['cases']}, 'source_identity': identity,
            'metric_protocol': metric_module.METRIC_PROTOCOL,
            'evaluation_sources': {name: digest(code / name) for name in ('evaluate.py', 'metrics.py', 'evaluate_trajectory.py')},
            'trajectory_module': load_module('single_run_frozen_trajectory', code / 'evaluate_trajectory.py')}


def verify(capture_root, manifest_path, raw_root, code, late_points=5):
    capture_root, manifest_path, raw_root, code = map(lambda p: Path(p).resolve(), (capture_root, manifest_path, raw_root, code))
    sys.dont_write_bytecode = True
    collection, covered = collection_files(capture_root)
    used = set()

    def downloaded(path):
        path = path.resolve()
        require(path in covered, 'Consumed artifact not covered by collection hash: ' + str(path))
        used.add(path)
        return read(path)

    training = capture_root / 'results/unet_recipe200x125/training'
    trajectory = training.parent / 'internal4_trajectory'
    config = downloaded(training / 'config.json')
    done_training = downloaded(training / 'TRAINING_COMPLETE.json')
    for path in (training / 'history.jsonl', training / 'final.pt'):
        require(path.resolve() in covered, 'Training evidence absent from collection'); used.add(path.resolve())
    context = context_for(code, manifest_path)
    source, files = context['source_identity']
    expected = {'method': 'unet', 'fold': 0, 'epochs': 200, 'steps': 125, 'save_every': 10,
                'batch_size': 2, 'patch_size': 128, 'smoke': False, 'no_ds': False, 'cohort': 'internal28',
                'seed': 20261013, 'split_seed': 20261014, 'lr': .0003, 'weight_decay': .0001, 'warmup_epochs': 5,
                'backbone_warmup_epochs': 0, 'seg_loss': 'dice_ce', 'sampling': 'standard', 'sampling_fov': 'image-body',
                'source_hash_algorithm': SOURCE_ALGORITHM, 'source_sha256': source, 'source_files_sha256': files,
                'manifest_sha256': EXPECTED_MANIFEST, 'checkpoint_selection': 'fixed final',
                'internal_selection_valid': True, 'outer8_used_for_selection': False, 'allow_selection_overlap': False,
                'classifier_pretraining_updates': 0, 'init_audits': [], 'backbone_init': None, 'classifier_init': None, 'full_init': None}
    require(all(config.get(k) == v for k, v in expected.items()), 'U-Net fixed common recipe/initialization differs')
    ids = config['internal_split']['val']; train_ids = config['internal_split']['train']
    outer = context['manifest']['folds'][0]
    require(ids == IDS and len(train_ids) == len(set(train_ids)) == 28 and set(train_ids).isdisjoint(ids)
            and set(train_ids) | set(ids) == set(outer['train']) and set(train_ids + ids).isdisjoint(outer['val'])
            and config['trained_case_ids'] == train_ids, 'Internal28/4 patient isolation differs')
    remote_training = PurePosixPath(config['output'])
    require(remote_training.is_absolute() and '..' not in remote_training.parts and remote_training.name == 'training', 'Unsafe remote training path')
    final_hash = digest(training / 'final.pt')
    require(final_hash == done_training.get('final_sha256'), 'Actual downloaded final.pt bytes differ from training marker')
    # The fixed-final inspector below checks complete200-epoch history and final4
    # pixels; precheck its budget before any intermediate prediction work.
    history = lines(training / 'history.jsonl')
    require([r.get('epoch') for r in history] == list(range(200))
            and all(r.get('iterations') == 125 and finite(r.get('loss')) for r in history), 'Incomplete fixed25,000-update history')
    marker_checks = {'epoch': 199, 'smoke': False, 'manifest_sha256': EXPECTED_MANIFEST, 'source_sha256': source,
                     'trained_case_ids': train_ids, 'segmentation_updates': 25000, 'internal_selection_valid': True,
                     'checkpoint_selection': 'fixed final'}
    require(all(done_training.get(k) == v for k, v in marker_checks.items()), 'Training completion identity differs')

    protocol = downloaded(trajectory / 'trajectory_protocol.json')
    complete = downloaded(trajectory / 'TRAJECTORY_COMPLETE.json')
    status = downloaded(trajectory / 'trajectory_status.json')
    require(status == {'status': 'complete', 'completed': 20, 'planned': 20}, 'Trajectory is partial;20 planned points must finish')
    identity = {'training_config_sha256': digest(training / 'config.json'), 'manifest_sha256': EXPECTED_MANIFEST,
                'source_sha256': source, 'evaluation_sources': context['evaluation_sources'], 'case_ids': ids,
                'split': 'internal_diagnostic', 'smoke': False, 'sw_batch_size': 1,
                'checkpoint_selection': 'fixed final remains unchanged; intermediate weights are trajectory diagnostics',
                'formal_final_test': False, 'outer8_used_for_selection': False}
    candidates = protocol.get('checkpoints', [])
    require(len(candidates) == 20 and [r.get('name') for r in candidates] == NAMES, 'Missing/unplanned checkpoint inventory;18/20 is not completion')
    for item in candidates:
        name = item['name']; epoch = 199 if name == 'final' else int(name[5:]) - 1
        path = remote_training / 'final.pt' if name == 'final' else remote_training / 'checkpoints' / (name + '.pt')
        require(set(item) == {'name', 'epoch', 'path', 'checkpoint_sha256'} and item['path'] == str(path)
                and item['epoch'] == epoch and valid_sha(item['checkpoint_sha256']), 'Checkpoint path/epoch/SHA metadata differs: ' + name)
        if name == 'final':
            require(item['checkpoint_sha256'] == final_hash, 'Final trajectory hash differs from actual final weights')
    identity['checkpoints'] = candidates
    require(protocol == identity and all(complete.get(k) == v for k, v in identity.items()) and complete.get('n_evaluations') == 20,
            'Trajectory complete marker/source protocol differs')

    records, case_rows, point_receipts = [], [], []
    metric_protocol = context['metric_protocol']
    for item in candidates:
        directory = trajectory / item['name']; epoch = item['epoch']
        require({p.name for p in directory.glob('*.json')} == {case + '.json' for case in ids} | {'summary.json', 'EVALUATION_COMPLETE.json'},
                'Unexpected/missing patient JSON: ' + item['name'])
        require({p.name for p in directory.glob('*_prediction.nrrd')} == {case + '_prediction.nrrd' for case in ids},
                'Unexpected/missing prediction patients: ' + item['name'])
        summary = downloaded(directory / 'summary.json'); done = downloaded(directory / 'EVALUATION_COMPLETE.json')
        rows = []
        role = 'fixed final' if item['name'] == 'final' else 'periodic internal-four diagnostic; not final selection'
        summary_checks = {'case_ids': ids, 'evaluation_split': 'internal_diagnostic', 'smoke': False, 'fold': 0, 'method': 'unet',
                          'checkpoint_epoch': epoch, 'checkpoint_selection': role, 'formal_final_test': False,
                          'internal_selection_valid': True, 'training_source_verified_against_live_code': True, 'geometry_only': False,
                          'metric_protocol': metric_protocol, 'primary_metrics': ['dice', 'assd_mm']}
        require(all(summary.get(k) == v for k, v in summary_checks.items()), 'Checkpoint summary/metric role differs: ' + item['name'])
        for case_id in ids:
            case = context['cases'][case_id]; row = downloaded(directory / (case_id + '.json'))
            expected_row = {'case_id': case_id, 'group': case['group'], 'fold': 0, 'method': 'unet', 'smoke': False,
                            'checkpoint_epoch': epoch, 'checkpoint_sha256': item['checkpoint_sha256'], 'checkpoint_selection': role,
                            'manifest_sha256': EXPECTED_MANIFEST, 'training_source_sha256': source,
                            'evaluation_source_sha256': context['evaluation_sources']['evaluate.py'],
                            'metrics_source_sha256': context['evaluation_sources']['metrics.py'],
                            'label_sha256': case['source_sha256']['label'], 'metric_grid': 'original CT',
                            'evaluation_split': 'internal_diagnostic', 'formal_final_test': False, 'internal_selection_valid': True,
                            'training_source_verified_against_live_code': True, 'geometry_only': False,
                            'metric_protocol_version': metric_protocol['version'], 'assd_definition': metric_protocol['assd_mm']['definition'],
                            'assd_directional_mean_definition': metric_protocol['assd_directional_mean_mm']['definition'],
                            'surface_connectivity_neighbors': 6, 'surface_area_weighted': False}
            require(all(row.get(k) == v for k, v in expected_row.items()) and set(SUMMARY_FIELDS) <= row.keys()
                    and finite(row['dice']) and 0 <= row['dice'] <= 1 and isinstance(row.get('empty_prediction'), bool)
                    and (row['assd_mm'] is None or finite(row['assd_mm']) and row['assd_mm'] >= 0), 'Patient metric/provenance differs: ' + item['name'] + '/' + case_id)
            require(row.get('evaluation_role') == summary.get('evaluation_role'), 'Patient/summary evaluation role differs')
            prediction = directory / (case_id + '_prediction.nrrd')
            require(prediction.resolve() in covered and valid_sha(row.get('prediction_sha256'))
                    and covered[prediction.resolve()] == row['prediction_sha256'], 'Saved prediction bytes differ: ' + item['name'] + '/' + case_id)
            used.add(prediction.resolve())
            check_geometry(prediction_header(prediction), case['original_geometry'], case_id)
            rows.append(row)
            case_rows.append({'checkpoint': item['name'], 'epoch_1based': epoch + 1, 'case_id': case_id, 'group': row['group'],
                              'dice': row['dice'], 'assd_mm': row['assd_mm'], 'empty_prediction': row['empty_prediction'],
                              'prediction_sha256': row['prediction_sha256'], 'fixed_final': item['name'] == 'final'})
        ledger = directory / 'case_metrics.jsonl'
        require(ledger.resolve() in covered, 'Per-checkpoint JSONL missing from collector'); used.add(ledger.resolve())
        require(lines(ledger) == rows and same(summary['overall'], summary_of(rows)), 'Case ledger/summary Null policy differs: ' + item['name'])
        for group in ('Normal', 'Diseased'):
            require(same(summary.get(group), summary_of([r for r in rows if r['group'] == group])), 'Stratum summary differs: ' + item['name'])
        marker = {'checkpoint_sha256': item['checkpoint_sha256'], 'manifest_sha256': EXPECTED_MANIFEST,
                  'evaluation_sources': context['evaluation_sources'], 'case_ids': ids, 'summary_sha256': digest(directory / 'summary.json')}
        require(all(done.get(k) == v for k, v in marker.items()) and finite(done.get('wall_seconds')) and done['wall_seconds'] >= 0,
                'Evaluation completion marker differs: ' + item['name'])
        records.append(item | {'evaluation_output': str(remote_training.parent / 'internal4_trajectory' / item['name']),
                               'summary_path': str(remote_training.parent / 'internal4_trajectory' / item['name'] / 'summary.json'),
                               'split': 'internal_diagnostic', 'role': summary['evaluation_role'], 'n_cases': 4,
                               'overall': summary['overall'], 'seconds': done['wall_seconds'], 'smoke': False,
                               'checkpoint_selection': role, 'formal_final_test': False})
        point_receipts.append({'name': item['name'], 'epoch_1based': epoch + 1, 'checkpoint_sha256': item['checkpoint_sha256'],
                               'checkpoint_bytes_checked': item['name'] == 'final', 'overall': summary['overall']})
    ledger = trajectory / 'validation_history.jsonl'
    require(ledger.resolve() in covered, 'Trajectory ledger missing from collector'); used.add(ledger.resolve())
    require(lines(ledger) == records and same(complete.get('total_inference_wall_seconds'), sum(r['seconds'] for r in records)),
            'Trajectory observation ledger differs from actual20 verified summaries')
    curve = trajectory / 'validation_curve.csv'
    require(curve.resolve() in covered, 'Producer trajectory CSV missing from collector'); used.add(curve.resolve())
    with curve.open(newline='', encoding='utf-8') as stream:
        curve_rows = list(csv.DictReader(stream))
    require(len(curve_rows) == 20, 'Trajectory CSV is partial')
    for actual, row in zip(curve_rows, records):
        expected_csv = {'epoch': row['epoch'], 'completed_epoch': row['epoch'] + 1, 'name': row['name'],
                        **{k: row['overall'][k]['mean'] for k in ('dice', 'assd_mm', 'cldice')}, 'n_cases': 4, 'seconds': row['seconds']}
        require(actual == {k: '' if v is None else str(v) for k, v in expected_csv.items()}, 'Trajectory CSV differs: ' + row['name'])

    # Reuse the independently reviewed CPU pixel checker, without requiring GTA.
    final = inspect_run('unet', training, trajectory / 'final', manifest_path, raw_root, code, context)
    calibration_protocol = downloaded(capture_root / 'unet_calibration_protocol.json')
    calibration_done = downloaded(capture_root / 'UNET_CALIBRATION_COMPLETE.json')
    calibration_status = downloaded(capture_root / 'unet_calibration_status.json')
    captured_freeze = downloaded(capture_root / 'configs/main_source_freeze.json')
    require(captured_freeze == read(code.parent / 'configs/main_source_freeze.json'), 'Captured/live frozen package manifest differs')
    require(all(calibration_done.get(k) == v for k, v in calibration_protocol.items()), 'Calibration protocol/completion differs')
    checks = {'formal_final_sha256': final_hash, 'trajectory_complete_sha256': digest(trajectory / 'TRAJECTORY_COMPLETE.json'),
              'validation_history_sha256': digest(ledger), 'formal_training_marker_sha256': digest(training / 'TRAINING_COMPLETE.json'),
              'training_source_sha256': source, 'manifest_sha256': EXPECTED_MANIFEST, 'internal_split': config['internal_split'],
              'formal_segmentation_updates': 25000, 'classifier_pretraining_updates': 0, 'outer8_used_for_selection': False, 'fivefold': False}
    require(all(calibration_done.get(k) == v for k, v in checks.items())
            and calibration_status.get('status') == 'complete' and calibration_status.get('phase') == 'complete',
            'Old controller did not complete its sole calibration/20-point trajectory')
    for path, expected_hash in covered.items():
        require(digest(path) == expected_hash, 'Collected artifact changed during verification: ' + str(path))
    require(source_identity(code) == context['source_identity'] and digest(manifest_path) == EXPECTED_MANIFEST,
            'Code/manifest changed during verification')
    source_guard(code.parent)
    late_names = NAMES[-late_points:]
    return {'schema_version': 1, 'status': 'verified_unet_fixed_final_and_full_trajectory',
            'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'role': 'four training-internal development patients; no independent final-test or superiority claim',
            'case_ids': ids, 'training_cases': 28, 'training_updates': 25000, 'checkpoint_selection': 'fixed final',
            'formal_final_test': False, 'best_checkpoint_selected': False, 'fivefold_result': False,
            'verified_checkpoint_evaluations': 20, 'verified_case_evaluations': len(case_rows),
            'prediction_bytes_and_original_grid_headers_checked': 80, 'pixel_primary_metrics_recomputed': 4,
            'intermediate_checkpoint_tensors_downloaded': False, 'intermediate_weight_SHA_evidence': 'protocol/ledger/case metadata agreement only; weight bytes not downloaded',
            'final_checkpoint_bytes_checked': True, 'final_checkpoint_sha256': final_hash,
            'metric_protocol': metric_protocol, 'final_summary': final['summary'], 'final_pixel_audit': final['audit'],
            'trajectory_points': point_receipts, 'per_case_trajectory': case_rows,
            'late_curve': {'checkpoint_names': late_names, 'selection_role': 'observed trajectory only; fixed final unchanged',
                           'rows': [r for r in case_rows if r['checkpoint'] in late_names], 'undefined_ASSD_retained': True},
            'source_sha256': source, 'manifest_sha256': EXPECTED_MANIFEST,
            'collection_manifest_sha256': digest(capture_root / 'collection_manifest.json'),
            'collected_files_verified': len(covered), 'consumed_collected_files': len(used),
            'collection_identity': collection, 'model_loaded': False, 'network_forward': False, 'optimizer_created': False, 'GPU_used': False}


def save_result(result, output):
    (output / 'SINGLE_RUN_VERIFIED.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, rows in [('all80_patient_points.csv', result['per_case_trajectory']), ('last5_patient_points.csv', result['late_curve']['rows'])]:
        with (output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    notes = ['U-Net单run验收：训练200×125与内部4例20点/80病例完整；fixed final保持不变，不作best epoch选择。',
             '80 prediction核对实际SHA与原CT几何header；末轮4例从保存像素重算Dice及6邻域surface pooled ASSD。',
             'final.pt已实际重hash；19个中间权重未下载，其SHA只作protocol/ledger/case元数据交叉核查。',
             '全部病例保留；任一病例ASSD未定义时完整四例cohort ASSD均值保持未定义，不删除病例或插值。',
             '最后5个计划点是观察曲线，不是模型选择。只有内部4例development结果，没有五折、盲测或方法胜负结论。',
             '没有加载模型、网络前向、优化器、训练或GPU工作。']
    (output / 'verification_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parents[1] / 'code')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({'capture_root': str(args.capture_root.resolve()), 'required_checkpoint_names': NAMES,
                          'checkpoint_evaluations': 20, 'case_evaluations': 80, 'prediction_hash_and_header_checks': 80,
                          'final4_pixel_Dice_ASSD_recompute': True, 'final_weight_byte_hash_check': True,
                          'intermediate_weight_bytes_required': False, 'partial18of20_accepted': False,
                          'checkpoint_selection': 'fixed final', 'network_forward': False, 'GPU_used': False}, indent=2)); return
    require(args.output is not None, '--output required for a fresh local verification receipt')
    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), 'Use a fresh verification output; no result overwrites')
    require(not output.is_relative_to(args.capture_root.resolve()) and not output.is_relative_to(args.code.resolve())
            and not output.is_relative_to(args.raw_root.resolve()), 'Verification outputs must be separate from captured inputs/code/raw GT')
    output.mkdir(parents=True, exist_ok=True)
    try:
        verifier_hash = digest(__file__)
        checker_path = Path(__file__).parent / 'compare_fixed_finals.py'
        checker_hash = digest(checker_path)
        result = verify(args.capture_root, args.manifest, args.raw_root, args.code)
        require(digest(__file__) == verifier_hash and digest(checker_path) == checker_hash,
                'Verification/checker implementation changed during audit')
        result['verifier_sha256'] = verifier_hash
        result['fixed_final_checker_sha256'] = checker_hash
        save_result(result, output)
        print(json.dumps({'status': result['status'], 'evaluations': 20, 'case_evaluations': 80, 'n_patients': 4, 'pixel_recomputed_cases': 4,
                          'output': str(output), 'GPU_used': False}, allow_nan=False))
    except Exception as error:
        (output / 'VERIFICATION_FAILED.json').write_text(json.dumps({'status': 'not_accepted', 'reason': str(error),
            'checkpoint_selection': 'fixed final', 'partial_completion_accepted': False}, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
