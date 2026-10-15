"""CPU-only canonical20/80 acceptance + final/OLD-preview decoded-mask check.

Only real captured producer records are read. No checkpoint/trajectory marker
is created or replaced; no Torch, model, GPU, training or inference is used.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ[_name] = '2'
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
sys.dont_write_bytecode = True
import compare_fixed_finals as c
import compare_preview_fixed_final as p
from verify_postfit import finite, require, same, summary_of

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
TRAIN = 'results/gta_enhanced200x125/training'
TRAJECTORY = 'results/gta_enhanced200x125/internal4_trajectory'
CURVES = 'results/gta_enhanced200x125/learning_curves'
FINAL = '2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'
NAMES = [f'epoch{i:03d}' for i in range(10, 200, 10)] + ['final']


def acceptance(capture, gta, context, guard):
    status_path, verified_path, protocol_path = [capture / n for n in ('POSTFIT_STATUS.json', 'POSTFIT_VERIFIED.json', 'POSTFIT_PROTOCOL.json')]
    for path in (status_path, verified_path, protocol_path): p.tracked(context, path)
    status, verified, protocol = [c.read(x) for x in (status_path, verified_path, protocol_path)]
    require(status.get('status') == 'completed' and verified.get('status') == 'postfit_verified'
            and verified.get('verified_checkpoint_evaluations') == 20 and verified.get('verified_case_evaluations') == 80
            and status.get('verified_receipt_sha256') == c.digest(verified_path)
            and status.get('protocol_sha256') == c.digest(protocol_path)
            and protocol.get('source_sha256') == guard.SOURCE and protocol.get('main_freeze_sha256') == guard.FREEZE,
            'Actual canonical postfit20/80 evidence does not close')
    identity = verified['trajectory_identity']
    directory = capture / TRAJECTORY
    actual_protocol = c.read(directory / 'trajectory_protocol.json')
    complete = c.read(directory / 'TRAJECTORY_COMPLETE.json')
    require(actual_protocol == identity and all(complete.get(k) == v for k, v in identity.items())
            and complete.get('n_evaluations') == 20 and identity.get('training_config_sha256') == guard.CONFIG
            and identity.get('manifest_sha256') == guard.MANIFEST and identity.get('source_sha256') == guard.SOURCE
            and identity.get('evaluation_sources') == context['evaluation_sources'] and identity.get('case_ids') == c.IDS
            and identity.get('sw_batch_size') == 1 and identity.get('formal_final_test') is False
            and [x['name'] for x in identity['checkpoints']] == NAMES
            and [x['epoch'] for x in identity['checkpoints']] == list(range(9, 190, 10)) + [199], 'Actual canonical trajectory identity differs')
    require(c.read(directory / 'trajectory_status.json') == {'status': 'complete', 'completed': 20, 'planned': 20}, 'Actual trajectory status differs')
    receipts = verified.get('cases', [])
    require(len(receipts) == 80 and len({(x['checkpoint'], x['case_id']) for x in receipts}) == 80, 'Canonical80 receipts missing/duplicated')
    receipt = {(x['checkpoint'], x['case_id']): x for x in receipts}
    ledger = c.lines(directory / 'validation_history.jsonl')
    require(len(ledger) == 20 and [x['name'] for x in ledger] == NAMES, 'Actual canonical ledger coverage differs')
    for point, recorded in zip(identity['checkpoints'], ledger):
        name, epoch = point['name'], point['epoch']
        weight = PurePosixPath(REMOTE) / TRAIN / ('final.pt' if name == 'final' else 'checkpoints/' + name + '.pt')
        require(point['path'] == str(weight) and point['checkpoint_sha256'] == recorded['checkpoint_sha256']
                and point['epoch'] == recorded['epoch'], 'Actual checkpoint/ledger identity differs')
        output = directory / name
        summary = context['trajectory_module'].verify_evaluation(output, c.IDS, point['checkpoint_sha256'],
            guard.MANIFEST, gta['config'], context['evaluation_sources'], context['cases'], epoch)
        rows = [c.read(output / (i + '.json')) for i in c.IDS]
        require(c.lines(output / 'case_metrics.jsonl') == rows, 'Canonical actual per-case ledger differs')
        for group, selected in [('overall', rows), ('Normal', [r for r in rows if r['group'] == 'Normal']),
                                ('Diseased', [r for r in rows if r['group'] == 'Diseased'])]:
            require(same(summary.get(group), summary_of(selected)), 'Canonical actual summary/null policy differs')
        role = 'fixed final' if name == 'final' else 'periodic internal-four diagnostic; not final selection'
        require(summary.get('checkpoint_selection') == role and all(r.get('checkpoint_selection') == role
                and r.get('formal_final_test') is False for r in rows), 'Canonical role differs')
        done = c.read(output / 'EVALUATION_COMPLETE.json')
        require(done.get('checkpoint_sha256') == point['checkpoint_sha256'] and done.get('manifest_sha256') == guard.MANIFEST
                and done.get('evaluation_sources') == context['evaluation_sources'] and done.get('case_ids') == c.IDS
                and done.get('summary_sha256') == c.digest(output / 'summary.json') and finite(done.get('wall_seconds')), 'Canonical real completion differs')
        remote_output = str(PurePosixPath(REMOTE) / TRAJECTORY / name)
        require(recorded == point | {'evaluation_output': remote_output, 'summary_path': remote_output + '/summary.json',
            'split': 'internal_diagnostic', 'role': summary['evaluation_role'], 'n_cases': 4, 'overall': summary['overall'],
            'seconds': done['wall_seconds'], 'smoke': False, 'checkpoint_selection': role, 'formal_final_test': False}, 'Canonical full ledger record differs')
        for row in rows:
            item = receipt[(name, row['case_id'])]
            require(item['case_json_sha256'] == c.digest(output / (row['case_id'] + '.json'))
                    and item['prediction_sha256'] == row['prediction_sha256'], 'Canonical independently verified80 receipt SHA differs')
        expected_files = {'summary.json', 'case_metrics.jsonl', 'EVALUATION_COMPLETE.json'} | {i + s for i in c.IDS for s in ('.json', '_prediction.nrrd')}
        require({x.name for x in output.iterdir()} == expected_files, 'Canonical point requires actual11file inventory')
        for filename in expected_files: p.tracked(context, output / filename)
    require(identity['checkpoints'][-1]['checkpoint_sha256'] == FINAL
            and c.digest(capture / TRAIN / 'final.pt') == FINAL, 'Canonical actual final bytes differ')
    for name, expected in [('TRAJECTORY_COMPLETE.json', verified['trajectory_complete_sha256']),
                           ('validation_history.jsonl', verified['validation_history_sha256'])]:
        require(p.tracked(context, directory / name)['sha256'] == expected, 'Canonical receipt hash differs')
    for name in ('trajectory_protocol.json', 'trajectory_status.json', 'validation_curve.csv'): p.tracked(context, directory / name)
    curves = capture / CURVES
    for artifact in verified['curve_artifacts']:
        relative = PurePosixPath(artifact['path']).relative_to(PurePosixPath(REMOTE))
        require(p.tracked(context, capture / str(relative))['sha256'] == artifact['sha256'], 'Actual canonical curve artifact differs')
    curve = c.read(curves / 'curve_manifest.json')
    require(curve.get('development_only') is True and curve.get('formal_final_test') is False
            and curve.get('interpolated_validation') is False and curve.get('plotter_sha256') == c.digest(ROOT / 'ops/plot_learning_curves.py')
            and len(curve.get('runs', [])) == 1, 'Actual canonical curve protocol differs')
    run = curve['runs'][0]
    require(run.get('n_recorded_epochs') == 200 and run.get('method') == 'gta' and run.get('selection_case_ids') == c.IDS,
            'Actual canonical curve data scope differs')
    for source in run['sources']:
        relative = PurePosixPath(source['path']).relative_to(PurePosixPath(REMOTE))
        require(p.tracked(context, capture / str(relative))['sha256'] == source['sha256'], 'Actual canonical plot source differs')
    return {'actual_postfit_verified_sha256': c.digest(verified_path), 'actual_POSTFIT_STATUS_completed': True,
            'verified_checkpoint_evaluations': 20, 'verified_case_metadata_and_prediction_SHA': 80,
            'local_fixed_final_primary_pixel_recalculations': 4, 'local_periodic_checkpoint_tensor_bytes_checked': False,
            'actual_server_periodic_tensor_audit_receipt_retained': True, 'curve_source_and_artifact_SHA_verified': True}


def compare(args):
    import numpy as np
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    canonical = args.canonical.resolve(); raw = args.raw_root.resolve()
    guard, context = p.context_for(ROOT / 'code', args.manifest.resolve(), ROOT / 'ops/run_early_fixed_final_eval.py')
    gta = c.inspect_run('gta', canonical / TRAIN, canonical / TRAJECTORY / 'final', args.manifest, raw, ROOT / 'code', context)
    proof = acceptance(canonical, gta, context, guard)
    preview = p.inspect_preview(canonical / TRAIN, args.preview / 'results/gta_fixed_final_preview_20261014',
        args.preview / 'EARLY_FIXED_FINAL_PROTOCOL.json', args.preview / 'EARLY_FIXED_FINAL_STATUS.json', args.preview / 'mirror_view', raw, context, guard)
    unet_training = args.unet / 'results/unet_recipe200x125/training'
    unet_eval = args.unet / 'results/unet_recipe200x125/internal4_trajectory/final'
    unet = c.inspect_run('unet', unet_training, unet_eval, args.manifest, raw, ROOT / 'code', context)
    p.check_unet_complete(unet_training, unet_eval, unet, context, guard)
    require(gta['config']['internal_split'] == unet['config']['internal_split']
            and all(gta['config'].get(k) == unet['config'].get(k) for k in c.RECIPE_FIELDS), 'Canonical/U-Net common recipe differs')
    agreements, pairs = [], []
    for a, b, old in zip(gta['rows'], unet['rows'], preview['rows']):
        case_id = a['case_id']
        require(case_id == b['case_id'] == old['case_id'] and a['label_sha256'] == b['label_sha256'] == old['label_sha256']
                and a['checkpoint_sha256'] == old['checkpoint_sha256'] == FINAL, 'Canonical/preview/pair patient/checkpoint/GT differs')
        paths = [canonical / TRAJECTORY / 'final' / (case_id + '_prediction.nrrd'),
                 args.preview / 'results/gta_fixed_final_preview_20261014' / (case_id + '_prediction.nrrd')]
        images = [c.read_image(x) for x in paths]
        c.check_geometry(c.geometry(images[0]), c.geometry(images[1]), case_id)
        arrays = [sitk.GetArrayFromImage(x) for x in images]
        require(arrays[0].shape == arrays[1].shape and all(np.isin(x, (0, 1)).all() for x in arrays), 'Decoded agreement grid/binary differs')
        masks = [x > 0 for x in arrays]
        diff = int(np.count_nonzero(masks[0] != masks[1])); union = int(np.count_nonzero(masks[0] | masks[1]))
        denominator = int(masks[0].sum()) + int(masks[1].sum())
        agreements.append({'case_id': case_id, 'decoded_mask_voxels_equal': diff == 0,
            'geometry_matches_within1e_minus6': True, 'geometry_exact_dictionary_equal': c.geometry(images[0]) == c.geometry(images[1]),
            'different_voxels': diff, 'different_fraction_of_full_grid': diff / masks[0].size,
            'different_fraction_of_foreground_union': diff / union if union else 0.,
            'mask_to_mask_dice': 2 * int((masks[0] & masks[1]).sum()) / denominator if denominator else 1.,
            'canonical_NRRD_sha256': c.digest(paths[0]), 'preview_NRRD_sha256': c.digest(paths[1]),
            'canonical_decoded_pixel_sha256': hashlib.sha256(arrays[0].tobytes()).hexdigest(),
            'preview_decoded_pixel_sha256': hashlib.sha256(arrays[1].tobytes()).hexdigest(),
            'canonical_dtype': str(arrays[0].dtype), 'preview_dtype': str(arrays[1].dtype), 'shape_zyx': list(arrays[0].shape),
            'canonical_dice': a['dice'], 'preview_dice': old['dice'], 'canonical_minus_preview_dice': a['dice'] - old['dice'],
            'canonical_assd_mm': a['assd_mm'], 'preview_assd_mm': old['assd_mm'],
            'canonical_minus_preview_assd_mm': a['assd_mm'] - old['assd_mm'] if finite(a['assd_mm']) and finite(old['assd_mm']) else None})
        pairs.append({'case_id': case_id, 'group': a['group'], 'label_sha256': a['label_sha256'],
            'gta_dice': a['dice'], 'unet_dice': b['dice'], 'dice_gta_minus_unet': a['dice'] - b['dice'],
            'gta_assd_mm': a['assd_mm'], 'unet_assd_mm': b['assd_mm'],
            'assd_gta_minus_unet_mm': a['assd_mm'] - b['assd_mm'] if finite(a['assd_mm']) and finite(b['assd_mm']) else None})
        del images, arrays, masks
    for path, record in context['input_records'].items(): require(p.file_record(path) == record, 'Captured evidence changed')
    for run in (gta, unet):
        for item in run['audit']['inputs']: require(c.digest(item['path']) == item['sha256'], 'Fixed-final evidence changed')
    guard.source_guard(ROOT)
    preview['audit'].update(canonical_agreement_check_completed=True,
                            canonical_agreement_verified=all(x['decoded_mask_voxels_equal'] for x in agreements))
    return {'schema_version': 1, 'status': 'verified_real_canonical20_80_and_fixed_final_pair', 'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'role': 'four training-internal development patients; no formal test/fivefold/significance', 'case_ids': c.IDS,
        'formal_final_test': False, 'checkpoint_selection': 'fixed final', 'canonical_proof': proof,
        'canonical_preview_all4_decoded_masks_equal': all(x['decoded_mask_voxels_equal'] for x in agreements),
        'canonical_preview_agreement': agreements, 'canonical_unet_per_case': pairs,
        'patient_statistics': {name: {k: c.complete_stats(run['rows'], k) for k in ('dice', 'assd_mm')} for name, run in [('gta_canonical', gta), ('unet', unet)]},
        'sd_definition': 'sample SD of4patients, not folds/seeds/significance; undefinedASSD prevents complete-four mean',
        'attribution_boundary': 'Common segmentation recipe; pretrained donors, extra5120 classifier updates, capacity/prior/auxiliary objectives differ. No module causal attribution or model winner.',
        'agreement_policy': 'Canonical retained even if devices differ; no averaging, voting, checkpoint/device selection or insertion of preview into canonical ledger.',
        'audits': {'gta_canonical': gta['audit'], 'unet': unet['audit'], 'preview': preview['audit']},
        'tool_sha256': c.digest(__file__), 'model_or_forward_or_GPU_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / 'server_records'
    parser.add_argument('--canonical', type=Path, default=base / 'gta_complete_trajectory_20261014')
    parser.add_argument('--preview', type=Path, default=base / 'gta_fixed_final_preview_capture_20261014')
    parser.add_argument('--unet', type=Path, default=base / 'unet_complete_trajectory_20261014')
    parser.add_argument('--manifest', type=Path, default=base / 'gta_fixed_final_preview_capture_20261014/mirror_view/manifest.json')
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--output', type=Path, default=base / 'gta_canonical_acceptance_20261014')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({'role': 'PLAN_ONLY_NOT_EXECUTED', 'actual_records_required': True,
            'canonical': str(args.canonical), 'output': str(args.output), 'GPU_or_model_or_inference': False,
            'preview_decoded_pixels_and_geometry_checked_when_run': True, 'canonical_evidence_created': False}, indent=2)); return
    require(args.raw_root is not None and not args.output.exists(), 'Provide actual --raw-root and fresh output')
    result = compare(args)
    args.output.mkdir(parents=True)
    (args.output / 'canonical_acceptance_and_pair.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for filename, rows in [('canonical_preview_decoded_comparison.csv', result['canonical_preview_agreement']),
                            ('canonical_unet_paired_cases.csv', result['canonical_unet_per_case'])]:
        with (args.output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    notes = ['Actual canonical20/80 source-backed acceptance; fixed-final primary pixels rechecked on4development patients.',
        'All4 canonical/OLD-preview decoded masks equal: ' + str(result['canonical_preview_all4_decoded_masks_equal']),
        result['agreement_policy'], result['attribution_boundary'], result['sd_definition']]
    for method, fields in result['patient_statistics'].items():
        for field, value in fields.items(): notes.append(f"{method} {field}: mean={value['mean']}; SD={value['sd']}; defined={value['n_defined']}/4")
    (args.output / 'acceptance_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'all4_decoded_masks_equal': result['canonical_preview_all4_decoded_masks_equal'], 'output': str(args.output)}))


if __name__ == '__main__': main()
