"""CPU saved-pixel GTA preview/U-Net comparison; never fabricates a trajectory.

Remote paths are lineage strings only. Explicit local paths relocate records.
The GTA branch requires its real preview protocol/completion and six-file byte
view; U-Net uses the existing canonical inspect_run verifier unchanged.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
import compare_fixed_finals as c
from verify_postfit import SUMMARY_FIELDS, finite, require, same, summary_of

GUARD_SHA = '2d7a4f24e4f2bae6fa58f5a023d726e1fb3f91bb0d2b7416a7dfe5bba4147e42'
ROOT = Path(__file__).resolve().parents[1]
ROLE = 'old-server early fixed-final preview; canonical agreement pending'


def file_record(path):
    path = Path(path)
    require(path.is_file(), 'Missing captured file: ' + str(path))
    before = path.stat()
    record = {'bytes': before.st_size, 'sha256': c.digest(path)}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Captured file changed while hashing: ' + str(path))
    return record


def tracked(context, path):
    path = Path(path).resolve()
    record = file_record(path)
    if path in context['input_records']:
        require(context['input_records'][path] == record, 'Captured input changed between checks: ' + str(path))
    else:
        context['input_records'][path] = record
    return record


def recorded_linux_command(guard):
    """Construct exact remote lineage with POSIX types, not host OS Path."""
    old = PurePosixPath(guard.OLD_ROOT.as_posix())
    shared = PurePosixPath(guard.SHARED.as_posix())
    return guard.evaluation_argv(old, shared, old / 'results/gta_fixed_final_preview_20261014')


def context_for(code, manifest_path, guard_path):
    require(c.digest(guard_path) == GUARD_SHA, 'Only reviewed final2d7 preview guard is accepted')
    guard = c.load_module('preview_cpu_reviewed_guard', guard_path)
    frozen = guard.source_guard(code.parent)  # Pure stdlib; no runtime_profile/GPU calls.
    manifest = c.read(manifest_path)
    require(c.digest(manifest_path) == guard.MANIFEST and manifest.get('complete') is True
            and manifest.get('dataset') == 'ASOCA' and manifest.get('schema_version') == 1
            and len(manifest['cases']) == len({x['case_id'] for x in manifest['cases']}) == 40,
            'Requires actual immutable complete ASOCA40 manifest')
    identity = c.source_identity(code)
    require(identity == (guard.SOURCE, frozen['training_dependency_files']), 'Actual BC8 source differs')
    metrics = c.load_module('preview_cpu_frozen_metrics', code / 'metrics.py')
    context = {'manifest': manifest, 'manifest_sha256': guard.MANIFEST,
               'cases': {x['case_id']: x for x in manifest['cases']}, 'source_identity': identity,
               'metric_protocol': metrics.METRIC_PROTOCOL,
               'evaluation_sources': {n: c.digest(code / n) for n in guard.EVALUATORS},
               'trajectory_module': c.load_module('preview_cpu_trajectory_verify', code / 'evaluate_trajectory.py'),
               'input_records': {}}
    tracked(context, guard_path); tracked(context, manifest_path)
    tracked(context, code.parent / 'configs/main_source_freeze.json')
    return guard, context


def inspect_preview(training, evaluation, protocol_path, status_path, mirror, raw_root, context, guard, receipt=None):
    ready_path = mirror / 'READY.json'
    ready_sha = tracked(context, ready_path)['sha256']
    ready = c.read(ready_path); guard.ready_structure(ready)
    require({p.name for p in mirror.iterdir()} == guard.FILES | {'READY.json'},
            'Local byte view requires exactly the real READY plus six payload filenames')
    payload = {name: tracked(context, mirror / name) for name in sorted(guard.FILES)}
    require(payload == ready['files'], 'Local payload byte view differs from actual shared READY')
    for name in ('config.json', 'history.jsonl', 'TRAINING_COMPLETE.json', 'final.pt'):
        require(tracked(context, training / name) == payload[name], 'New GTA training does not close to byte view: ' + name)
    require(payload['manifest.json']['sha256'] == context['manifest_sha256'], 'Mirror manifest differs')
    config = c.read(training / 'config.json')
    train_ids = guard.config_guard(config, context['manifest'])
    require(config.get('source_files_sha256') == context['source_identity'][1], 'GTA per-file source differs')
    marker = c.read(training / 'TRAINING_COMPLETE.json')
    marker_expected = {'epoch': 199, 'smoke': False, 'segmentation_updates': 25000,
                       'classifier_pretraining_updates': 5120, 'checkpoint_selection': 'fixed final',
                       'internal_selection_valid': True, 'source_sha256': guard.SOURCE,
                       'manifest_sha256': guard.MANIFEST, 'trained_case_ids': train_ids,
                       'final_sha256': ready['checkpoint_sha256']}
    require(all(marker.get(k) == v for k, v in marker_expected.items()), 'Actual GTA200 completion differs')
    history = c.lines(training / 'history.jsonl')
    def all_finite(value):
        if isinstance(value, dict): return all(all_finite(v) for v in value.values())
        if isinstance(value, list): return all(all_finite(v) for v in value)
        return math.isfinite(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else True
    require([r.get('epoch') for r in history] == list(range(200))
            and all(r.get('iterations') == 125 and finite(r.get('loss')) and all_finite(r) for r in history)
            and [r.get('stage') for r in history] == ['frozen']*50 + ['partial']*20 + ['joint']*130,
            'Actual GTA history does not close finite200x125/stages')
    canonical_audit = c.read(mirror / 'input_runtime_audit.json')
    require(canonical_audit.get('python', '').split()[0] == '3.11.17'
            and canonical_audit.get('packages') == ready['packages']
            and canonical_audit.get('manifest_sha256') == guard.MANIFEST
            and canonical_audit.get('training_source_sha256') == guard.SOURCE
            and canonical_audit.get('evaluation_sources') == context['evaluation_sources'] == ready['evaluation_sources']
            and canonical_audit.get('cache_CT_inputs') == ready['cache_CT_inputs'], 'Canonical actual runtime input audit differs')

    protocol_sha = tracked(context, protocol_path)['sha256']; protocol = c.read(protocol_path)
    expected_protocol = {'schema_version': 1, 'role': 'early fixed-final preview; no new training',
        'controller_sha256': GUARD_SHA, 'READY_sha256': ready_sha, 'source_root': guard.NEW_ROOT,
        'source_config_sha256': guard.CONFIG, 'training_source_sha256': guard.SOURCE,
        'source_freeze_sha256': guard.FREEZE, 'full_code_package_sha256': guard.PACKAGE,
        'payload_files': payload, 'case_ids': guard.IDS, 'evaluation_sources': context['evaluation_sources'],
        'cache_CT_inputs': ready['cache_CT_inputs'], 'canonical_runtime': canonical_audit,
        'command': recorded_linux_command(guard),
        'split': 'internal_diagnostic', 'checkpoint_selection': 'fixed final', 'checkpoint_epoch': 199,
        'formal_final_test': False, 'sw_batch_size': 1, 'extra_case_inferences': 4,
        'canonical_case_inferences': 80, 'preview_never_replaces_canonical': True}
    require(all(protocol.get(k) == v for k, v in expected_protocol.items()), 'Actual preview protocol/remote lineage differs')
    old_runtime = protocol.get('old_runtime', {})
    require(old_runtime.get('packages') == ready['packages'] == guard.PACKAGES
            and old_runtime.get('python', '').split()[0] == '3.11.17'
            and old_runtime.get('CUDA_context_initialized') is False, 'Actual isolated preview runtime receipt differs')
    gpu = protocol.get('old_GPU_before_inference', {})
    require(gpu.get('all_compute_GPUs_idle') is True and gpu.get('gpu_processes') == '', 'Actual preview idle-GPU receipt differs')

    done_path = evaluation / 'EVAL_COMPLETE.json'
    done_sha = tracked(context, done_path)['sha256']; done = c.read(done_path)
    expected_done = {'schema_version': 1, 'status': 'completed', 'role': 'early fixed-final preview; not canonical trajectory',
        'READY_sha256': ready_sha, 'protocol_sha256': protocol_sha, 'n_cases': 4, 'case_ids': guard.IDS,
        'checkpoint_epoch': 199, 'checkpoint_sha256': ready['checkpoint_sha256'], 'manifest_sha256': guard.MANIFEST,
        'evaluation_sources': context['evaluation_sources'], 'primary_metrics': ['dice', 'assd_mm'],
        'parameter_updates': 0, 'extra_case_inferences': 4, 'formal_final_test': False,
        'canonical_final_unchanged': True, 'canonical20_trajectory_unchanged': True}
    require(all(done.get(k) == v for k, v in expected_done.items())
            and finite(done.get('wall_seconds')) and done['wall_seconds'] >= 0, 'Actual preview EVAL_COMPLETE differs')
    status = c.read(status_path); tracked(context, status_path)
    require(status.get('status') == 'completed' and status.get('controller_source_sha256') == GUARD_SHA
            and status.get('completion_marker_sha256') == done_sha and status.get('READY_sha256') == ready_sha
            and status.get('checkpoint_sha256') == ready['checkpoint_sha256'] and status.get('child_pid') is None
            and status.get('gpu_lock_acquired') is False and status.get('owned_child_exit_confirmed') is True,
            'Actual preview controller has not completed/confirmed owned child exit')
    expected_outputs = {i + suffix for i in guard.IDS for suffix in ('.json', '_prediction.nrrd')}
    expected_outputs |= {'summary.json', 'case_metrics.jsonl', 'EVAL_COMPLETE.json'}
    require({p.name for p in evaluation.iterdir()} == expected_outputs, 'Preview requires its actual11 files; no invented canonical marker/trajectory')
    for name in expected_outputs: tracked(context, evaluation / name)
    require(done.get('summary_sha256') == c.digest(evaluation / 'summary.json')
            and done.get('case_metrics_sha256') == c.digest(evaluation / 'case_metrics.jsonl'), 'Preview completion/summary/ledger byte links differ')
    summary = context['trajectory_module'].verify_evaluation(evaluation, guard.IDS, ready['checkpoint_sha256'],
        guard.MANIFEST, config, context['evaluation_sources'], context['cases'], 199)
    require(summary.get('checkpoint_selection') == 'fixed final' and summary.get('formal_final_test') is False
            and summary.get('internal_selection_valid') is True and summary.get('primary_metrics') == ['dice', 'assd_mm']
            and summary.get('metric_protocol') == context['metric_protocol'], 'Preview primary metric/fixed-final role differs')
    rows = [c.read(evaluation / (i + '.json')) for i in guard.IDS]
    require(c.lines(evaluation / 'case_metrics.jsonl') == rows, 'Preview real patient ledger differs')
    for group, selected in [('overall', rows), ('Normal', [r for r in rows if r['group'] == 'Normal']),
                            ('Diseased', [r for r in rows if r['group'] == 'Diseased'])]:
        require(same(summary.get(group), summary_of(selected)), 'Preview null-aware summary differs: ' + group)
    require(set(done.get('predictions', {})) == set(guard.IDS) and set(protocol.get('raw_GT', {})) == set(guard.IDS),
            'Preview prediction/raw-GT receipt patients differ')
    import numpy as np
    import SimpleITK as sitk
    pixels = []
    for case_id, row in zip(guard.IDS, rows):
        case = context['cases'][case_id]
        mp = context['metric_protocol']
        require(row.get('inference_patch_size') == [128]*3 and row.get('inference_sw_batch_size') == 1
                and row.get('checkpoint_selection') == 'fixed final' and row.get('formal_final_test') is False
                and row.get('internal_selection_valid') is True and row.get('metric_protocol_version') == mp['version']
                and row.get('assd_definition') == mp['assd_mm']['definition']
                and row.get('assd_directional_mean_definition') == mp['assd_directional_mean_mm']['definition']
                and row.get('surface_connectivity_neighbors') == 6 and row.get('surface_area_weighted') is False
                and all(k in row for k in SUMMARY_FIELDS), 'Preview inference/metric definition differs: ' + case_id)
        label = (raw_root / case['label_relative_path']).resolve()
        require(label.is_relative_to(raw_root), 'Raw reference escapes explicit local root')
        label_record = tracked(context, label)
        require(label_record == protocol['raw_GT'][case_id]
                and label_record['sha256'] == case['source_sha256']['label'] == row['label_sha256'] == ready['raw_GT_source_sha256'][case_id],
                'Preview actual raw GT differs: ' + case_id)
        prediction_path = evaluation / (case_id + '_prediction.nrrd')
        reference, prediction = c.read_image(label), c.read_image(prediction_path)
        grid = c.geometry(reference); c.check_geometry(grid, case['original_geometry'], case_id)
        c.check_geometry(c.geometry(prediction), grid, case_id)
        array, truth = sitk.GetArrayFromImage(prediction), sitk.GetArrayFromImage(reference)
        measured = c.primary_from_pixels(array, truth, reference.GetSpacing()[::-1])
        for key in ('dice', 'assd_mm'):
            a, b = row.get(key), measured[key]
            require(a is None and b is None or finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6),
                    'Preview saved-pixel primary metric mismatch: ' + case_id + '/' + key)
        require(isinstance(row.get('empty_prediction'), bool) and row['empty_prediction'] == measured['empty_prediction'],
                'Preview empty-mask flag differs')
        pixel_receipt = {'file_sha256': c.digest(prediction_path), 'decoded_pixel_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            'dtype': str(array.dtype), 'shape_zyx': list(array.shape), **c.geometry(prediction),
            'foreground_voxels': int(np.count_nonzero(array))}
        require(pixel_receipt == done['predictions'][case_id], 'Preview completion decoded-pixel/geometry receipt differs: ' + case_id)
        pixels.append({'case_id': case_id, 'label_sha256': row['label_sha256'], 'prediction_sha256': row['prediction_sha256'],
                       'original_grid': grid, 'recomputed_primary_metrics': measured})
        del array, truth, reference, prediction
    receipt_record = tracked(context, receipt) if receipt else None
    return {'config': config, 'rows': rows, 'summary': summary,
            'audit': {'method': 'gta_early_preview', 'weights_bytes_checked': True, 'model_loaded': False,
                      'pixel_primary_metrics_recomputed': True, 'canonical_trajectory_evidence_created': False,
                      'canonical_agreement_verified': False, 'preview_protocol_sha256': protocol_sha,
                      'preview_complete_sha256': done_sha, 'READY_sha256': ready_sha, 'local_payload_records': payload,
                      'payload_view_origin': 'local byte view assembled from downloaded original source files and checked against actual shared READY and old guard payload fingerprints; not every file claimed re-downloaded from shared',
                      'assembly_receipt': receipt_record, 'pixel_audits': pixels}}


def check_unet_complete(training, evaluation, run, context, guard):
    """Recheck actual20/80 metadata/prediction bytes; recompute pixels only final4."""
    training, evaluation = training.resolve(), evaluation.resolve()
    capture = training.parents[2]
    marker_path = capture / 'UNET_CALIBRATION_COMPLETE.json'
    marker = c.read(marker_path); tracked(context, marker_path)
    trajectory = evaluation.parent
    complete_path, ledger_path = trajectory / 'TRAJECTORY_COMPLETE.json', trajectory / 'validation_history.jsonl'
    tracked(context, complete_path); tracked(context, ledger_path)
    require(marker.get('controller_sha256') == '54fd789b35ad14c56ed74b7fb3f14b1f8a794d1be995fba968da99d1c94bd807'
            and marker.get('root') == str(PurePosixPath(guard.OLD_ROOT.as_posix())) and marker.get('training_source_sha256') == guard.SOURCE
            and marker.get('manifest_sha256') == guard.MANIFEST and marker.get('main_freeze_manifest_sha256') == guard.FREEZE
            and marker.get('formal_segmentation_updates') == 25000 and marker.get('fivefold') is False
            and marker.get('formal_final_sha256') == run['audit']['final_checkpoint_sha256']
            and marker.get('formal_training_marker_sha256') == c.digest(training / 'TRAINING_COMPLETE.json')
            and marker.get('trajectory_complete_sha256') == c.digest(complete_path)
            and marker.get('validation_history_sha256') == c.digest(ledger_path), 'Actual U-Net global completion does not close')
    complete, ledger = c.read(complete_path), c.lines(ledger_path)
    protocol_path = trajectory / 'trajectory_protocol.json'
    protocol = c.read(protocol_path); tracked(context, protocol_path)
    names = [f'epoch{i:03d}' for i in range(10, 200, 10)] + ['final']
    require(complete.get('n_evaluations') == 20 and complete.get('case_ids') == guard.IDS
            and complete.get('sw_batch_size') == 1 and [r.get('name') for r in ledger] == names
            and [r.get('epoch') for r in ledger] == list(range(9, 190, 10)) + [199]
            and all(r.get('n_cases') == 4 and r.get('smoke') is False for r in ledger), 'Actual U-Net20/80 coverage differs')
    require(all(complete.get(k) == v for k, v in protocol.items())
            and len(protocol.get('checkpoints', [])) == 20, 'U-Net full trajectory protocol/completion identity differs')
    remote_base = c.PurePosixPath(run['config']['output']).parent / 'internal4_trajectory'
    remote_training = c.PurePosixPath(run['config']['output'])
    for record, checkpoint in zip(ledger, protocol['checkpoints']):
        remote_weight = remote_training / ('final.pt' if record['name'] == 'final' else 'checkpoints/' + record['name'] + '.pt')
        require(checkpoint == {'path': str(remote_weight), 'epoch': record['epoch'], 'name': record['name'],
                              'checkpoint_sha256': record['checkpoint_sha256']}, 'U-Net checkpoint20 identities differ from ledger')
        directory = trajectory / record['name']
        summary = context['trajectory_module'].verify_evaluation(directory, guard.IDS, record['checkpoint_sha256'],
            guard.MANIFEST, run['config'], context['evaluation_sources'], context['cases'], record['epoch'])
        require(record.get('evaluation_output') == str(remote_base / record['name'])
                and record.get('summary_path') == str(remote_base / record['name'] / 'summary.json')
                and same(record.get('overall'), summary['overall']), 'U-Net trajectory ledger/actual summary differs')
        done = c.read(directory / 'EVALUATION_COMPLETE.json')
        require(done.get('checkpoint_sha256') == record['checkpoint_sha256'] and done.get('manifest_sha256') == guard.MANIFEST
                and done.get('evaluation_sources') == context['evaluation_sources'] and done.get('case_ids') == guard.IDS
                and done.get('summary_sha256') == c.digest(directory / 'summary.json')
                and finite(done.get('wall_seconds')) and done['wall_seconds'] == record.get('seconds'), 'U-Net trajectory completion point differs')
        for filename in ['summary.json', 'EVALUATION_COMPLETE.json'] + [i + suffix for i in guard.IDS for suffix in ('.json', '_prediction.nrrd')]:
            tracked(context, directory / filename)
    run['audit'].update(full_20point_trajectory_completion_checked=True,
                        saved80_case_metadata_and_prediction_bytes_verified=True,
                        primary_pixel_metrics_recomputed_case_evaluations=4,
                        actual_global_completion_sha256=c.digest(marker_path))


def compare(args):
    code, manifest_path, raw_root = args.code.resolve(), args.manifest.resolve(), args.raw_root.resolve()
    guard, context = context_for(code, manifest_path, args.guard_source.resolve())
    preview = inspect_preview(args.gta_training.resolve(), args.gta_evaluation.resolve(), args.preview_protocol.resolve(),
        args.preview_status.resolve(), args.mirror.resolve(), raw_root, context, guard, args.payload_view_receipt)
    unet = c.inspect_run('unet', args.unet_training, args.unet_evaluation, manifest_path, raw_root, code, context)
    check_unet_complete(args.unet_training, args.unet_evaluation, unet, context, guard)
    require(preview['config']['internal_split'] == unet['config']['internal_split']
            and all(preview['config'].get(k) == unet['config'].get(k) for k in c.RECIPE_FIELDS), 'Paired common recipe/internal28+4 differs')
    pairs = []
    for a, b, ap, bp in zip(preview['rows'], unet['rows'], preview['audit']['pixel_audits'], unet['audit']['pixel_audits']):
        require(a['case_id'] == b['case_id'] and a['label_sha256'] == b['label_sha256'] and ap['original_grid'] == bp['original_grid'],
                'Paired actual GT/patient/grid differs')
        pairs.append({'case_id': a['case_id'], 'group': a['group'], 'label_sha256': a['label_sha256'],
            'gta_preview_dice': a['dice'], 'unet_dice': b['dice'], 'dice_preview_minus_unet': a['dice'] - b['dice'],
            'gta_preview_assd_mm': a['assd_mm'], 'unet_assd_mm': b['assd_mm'],
            'assd_preview_minus_unet_mm': a['assd_mm'] - b['assd_mm'] if finite(a['assd_mm']) and finite(b['assd_mm']) else None})
    for path, record in context['input_records'].items(): require(file_record(path) == record, 'Captured input changed during comparison: ' + str(path))
    for item in unet['audit']['inputs']: require(c.digest(item['path']) == item['sha256'], 'U-Net captured input changed during comparison')
    guard.source_guard(code.parent)
    require(c.digest(args.guard_source) == GUARD_SHA and c.source_identity(code) == context['source_identity'], 'Reviewed verifier/source changed')
    return {'schema_version': 1, 'status': 'verified_preview_fixed_final_pair', 'role': ROLE,
        'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'case_ids': guard.IDS, 'n_patients': 4,
        'formal_final_test': False, 'fivefold_result': False, 'significance_test': False, 'checkpoint_selection': 'fixed final',
        'canonical_agreement_verified': False, 'canonical_trajectory_evidence_created': False,
        'paired_difference_direction': 'GTA early preview minus U-Net; higher Dice and lower ASSD are better',
        'metric_protocol': context['metric_protocol'], 'per_case': pairs,
        'patient_statistics': {'gta_early_preview': {k: c.complete_stats(preview['rows'], k) for k in ('dice', 'assd_mm')},
                               'unet': {k: c.complete_stats(unet['rows'], k) for k in ('dice', 'assd_mm')}},
        'paired_statistics': {k: c.complete_stats(pairs, k) for k in ('dice_preview_minus_unet', 'assd_preview_minus_unet_mm')},
        'sd_definition': 'sample SD across four development patients; not fold/seed SD or statistical significance',
        'common_recipe': {k: preview['config'][k] for k in c.RECIPE_FIELDS},
        'attribution_boundary': 'Common segmentation recipe; architecture/capacity, BAS initialization, extra5120 classifier pretraining, priors/reweighting/auxiliary objectives differ. This does not isolate a module effect. Early cross-hardware predictions are not canonical agreement.',
        'initialization_and_capacity': {name: {k: run['config'].get(k) for k in ('parameters', 'backbone_init', 'classifier_init', 'classifier_pretraining_updates', 'segmentation_updates', 'init_audits')}
                                        for name, run in [('gta_early_preview', preview), ('unet', unet)]},
        'CPU_recalculation_packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'SimpleITK')},
        'audits': {'gta_early_preview': preview['audit'], 'unet': unet['audit']},
        'input_records': [{'path': str(path), **record} for path, record in context['input_records'].items()],
        'tool_sha256': c.digest(__file__)}


def save(result, output):
    require(not output.exists(), 'Use a fresh independent comparison output directory')
    output.mkdir(parents=True)
    (output / 'paired_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, rows in [('paired_cases.csv', result['per_case']), ('patient_mean_sd.csv',
        [{'method': method, 'metric': field, **stats} for method, fields in result['patient_statistics'].items() for field, stats in fields.items()])]:
        with (output / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    notes = [ROLE, '仅4例train28内部development病例；GTA为旧服务器提前fixed-final推理，canonical逐像素一致性仍待核对。',
             '差值=GTA preview−U-Net；患者样本SD不是fold SD/显著性。未定义ASSD病例保留，不补0，不报告缺病例的完整均值。',
             '主指标已对同一原GT/保存prediction在原网格CPU重算；没有加载模型、推理或训练，没有伪造canonical轨迹。',
             '相同分割recipe仍存在预训练/初始化、模型容量、先验及辅助目标差异；不能单模块归因。',
             result['audits']['gta_early_preview']['payload_view_origin']]
    for method, fields in result['patient_statistics'].items():
        for field, item in fields.items(): notes.append(f"{method} {field}: mean={item['mean']}; patient_SD={item['sd']}; defined={item['n_defined']}/4")
    (output / 'comparison_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    capture = ROOT / 'server_records/gta_fixed_final_preview_capture_20261014'
    defaults = {'gta-training': ROOT / 'server_records/gta_training_complete_20261014/results/gta_enhanced200x125/training',
        'gta-evaluation': capture / 'results/gta_fixed_final_preview_20261014', 'preview-protocol': capture / 'EARLY_FIXED_FINAL_PROTOCOL.json',
        'preview-status': capture / 'EARLY_FIXED_FINAL_STATUS.json', 'mirror': capture / 'mirror_view',
        'unet-training': ROOT / 'server_records/unet_complete_trajectory_20261014/results/unet_recipe200x125/training',
        'unet-evaluation': ROOT / 'server_records/unet_complete_trajectory_20261014/results/unet_recipe200x125/internal4_trajectory/final',
        'code': ROOT / 'code', 'guard-source': ROOT / 'ops/run_early_fixed_final_eval.py', 'output': capture / 'paired_CPU_comparison'}
    for name, value in defaults.items(): parser.add_argument('--' + name, type=Path, default=value)
    parser.add_argument('--manifest', type=Path); parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--payload-view-receipt', type=Path); parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    args.manifest = args.manifest or args.mirror / 'manifest.json'
    if args.plan:
        print(json.dumps({'role': 'READ_ONLY_PLAN_NOT_MODEL_RESULTS', 'comparison_role': ROLE,
            'paths': {k: str(v) if v else None for k, v in vars(args).items() if k != 'plan'},
            'requires_explicit_raw_root': args.raw_root is None, 'GPU_used': False, 'model_loaded': False,
            'inference_started': False, 'training_started': False, 'outputs_created': False,
            'expected_preview_guard_sha256': GUARD_SHA}, indent=2)); return
    require(args.raw_root is not None, 'Provide explicit local --raw-root; no remote paths are opened')
    result = compare(args); save(result, args.output.resolve())
    print(json.dumps({'status': result['status'], 'role': ROLE, 'n_patients': 4, 'output': str(args.output.resolve()), 'model_loaded': False}))


if __name__ == '__main__': main()
