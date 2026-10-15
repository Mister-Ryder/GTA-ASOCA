"""CPU<=2 saved-mask error accounting; no model/forward/GPU or new tests.

GT Lee skeleton/physical EDT radius follow the frozen metric definition.
Radius groups count skeleton voxels, not clinical branches or mm-weighted
centerline length. FN connected regions receive geometric coordinates only.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ[name] = '2'

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import compare_fixed_finals as c
from verify_postfit import require, finite

ROOT = Path(__file__).resolve().parents[1]
METRICS_SHA = '388353e557d3da61571517803938386fafdfebd0a535fead72579d8920b583f9'
PAIR_TOOL_SHA = 'd4f5c101eccc8d0c47d13b1f492645d491e72f8b7fc4507916de071b3d1d460d'


def close(a, b):
    return a is None and b is None or finite(a) and finite(b) and math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6)


def emit(stage, **extra):
    print(json.dumps({'stage': stage, **extra}), flush=True)


def physical_box(lo_zyx, hi_zyx, grid):
    """Inclusive voxel-index box, image-axis extent, world-axis outer faces."""
    import itertools
    import numpy as np
    lo, hi = np.asarray(lo_zyx), np.asarray(hi_zyx)
    spacing = np.asarray(grid['spacing_xyz_mm'])
    direction = np.asarray(grid['direction']).reshape(3, 3)
    corners = np.asarray(list(itertools.product(*zip(lo[::-1] - .5, hi[::-1] + .5))))
    world = np.asarray(grid['origin_xyz_mm']) + (corners * spacing) @ direction.T
    return {'bbox_zyx_min_inclusive': lo.tolist(), 'bbox_zyx_max_inclusive': hi.tolist(),
            'bbox_size_zyx_voxels': (hi - lo + 1).tolist(),
            'bbox_extent_zyx_mm': ((hi - lo + 1) * spacing[::-1]).tolist(),
            'world_outer_bbox_xyz_min_mm': world.min(axis=0).tolist(),
            'world_outer_bbox_xyz_max_mm': world.max(axis=0).tolist()}


def preflight(args, pair, manifest):
    import numpy as np
    import SimpleITK as sitk
    cases = {x['case_id']: x for x in manifest['cases']}
    audits, fingerprints = {}, {str(args.pair_json): c.digest(args.pair_json), str(args.manifest): c.digest(args.manifest)}
    paired = {x['case_id']: x for x in pair['per_case']}
    for case_id in c.IDS:
        reference = (args.raw_root / cases[case_id]['label_relative_path']).resolve()
        require(reference.is_relative_to(args.raw_root), 'Raw GT escaped explicit root')
        expected_label = paired[case_id]['label_sha256']
        require(c.digest(reference) == expected_label == cases[case_id]['source_sha256']['label'], 'Actual raw label SHA differs')
        image = c.read_image(reference); grid = c.geometry(image)
        c.check_geometry(grid, cases[case_id]['original_geometry'], case_id)
        truth = sitk.GetArrayFromImage(image)
        require(np.isin(truth, (0, 1)).all() and truth.any(), 'GT must be nonempty binary')
        audit = {'label_path': str(reference), 'label_sha256': expected_label, 'original_grid': grid, 'methods': {}}
        fingerprints[str(reference)] = expected_label
        for method, directory, pair_name in [('gta_preview', args.gta_evaluation, 'gta_early_preview'),
                                              ('unet', args.unet_evaluation, 'unet')]:
            row_path, prediction = directory / (case_id + '.json'), directory / (case_id + '_prediction.nrrd')
            row = c.read(row_path)
            source_audit = next(x for x in pair['audits'][pair_name]['pixel_audits'] if x['case_id'] == case_id)
            require(row['label_sha256'] == expected_label and row['checkpoint_epoch'] == 199
                    and row['checkpoint_selection'] == 'fixed final' and row['formal_final_test'] is False
                    and row['manifest_sha256'] == c.EXPECTED_MANIFEST and row['training_source_sha256'] == c.EXPECTED_SOURCE
                    and row['metrics_source_sha256'] == METRICS_SHA
                    and row['prediction_sha256'] == source_audit['prediction_sha256'] == c.digest(prediction), 'Actual prediction/row identity differs')
            for field, pair_field in [('dice', 'gta_preview_dice' if method == 'gta_preview' else 'unet_dice'),
                                       ('assd_mm', 'gta_preview_assd_mm' if method == 'gta_preview' else 'unet_assd_mm')]:
                require(close(row[field], paired[case_id][pair_field]), 'Verified pair/row primary metric differs')
            predicted = c.read_image(prediction); c.check_geometry(c.geometry(predicted), grid, case_id)
            array = sitk.GetArrayFromImage(predicted)
            require(array.shape == truth.shape and np.isin(array, (0, 1)).all(), 'Actual saved prediction grid/binary labels differ')
            fingerprints[str(row_path)] = c.digest(row_path); fingerprints[str(prediction)] = row['prediction_sha256']
            audit['methods'][method] = {'prediction_path': str(prediction), 'prediction_sha256': row['prediction_sha256'],
                                        'row_path': str(row_path), 'row': row}
            del array, predicted
        audits[case_id] = audit
        del image, truth
        emit('original_geometry_and_label_bytes_verified', case_id=case_id)
    return audits, fingerprints


def surfaces(prediction, truth, spacing, metrics):
    """Same6-neighbour surface metric via exact KDTree, at most2 CPU workers."""
    import numpy as np
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    ps = prediction & ~ndi.binary_erosion(prediction, structure=metrics.SURFACE_STRUCTURE, border_value=0)
    ts = truth & ~ndi.binary_erosion(truth, structure=metrics.SURFACE_STRUCTURE, border_value=0)
    pn, tn = int(ps.sum()), int(ts.sum())
    result = {'prediction_surface_voxels': pn, 'truth_surface_voxels': tn,
              'asd_prediction_to_truth_mm': None, 'asd_truth_to_prediction_mm': None, 'assd_mm': None}
    if pn and tn:
        pp, tp = np.argwhere(ps) * spacing, np.argwhere(ts) * spacing
        a = cKDTree(tp).query(pp, workers=2)[0]
        b = cKDTree(pp).query(tp, workers=2)[0]
        result.update(asd_prediction_to_truth_mm=float(a.mean()), asd_truth_to_prediction_mm=float(b.mean()),
                      assd_mm=float((a.sum() + b.sum()) / (len(a) + len(b))),
                      prediction_to_truth_surface_p95_mm=float(np.percentile(a, 95)),
                      truth_to_prediction_surface_p95_mm=float(np.percentile(b, 95)))
    return result


def false_negative_regions(fn, skeleton, thin, radius, offset, grid, case_id, method):
    import numpy as np
    from scipy import ndimage as ndi
    labels, count = ndi.label(fn, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    require(int(sizes.sum()) == int(fn.sum()), 'FN component accounting does not reconcile')
    ids = np.arange(1, count + 1)
    sk_counts = ndi.sum(skeleton, labels, ids).astype(int)
    thin_counts = ndi.sum(thin, labels, ids).astype(int)
    objects = ndi.find_objects(labels)
    order = np.argsort(-sizes, kind='stable')
    volume = float(np.prod(grid['spacing_xyz_mm']))
    regions = []
    for rank, index in enumerate(order, 1):
        box = objects[int(index)]
        lo = np.asarray([x.start for x in box]) + offset
        hi = np.asarray([x.stop - 1 for x in box]) + offset
        row = {'case_id': case_id, 'method': method, 'rank': rank, 'component_id': int(index) + 1,
               'FN_voxels': int(sizes[index]), 'FN_volume_mm3': int(sizes[index]) * volume,
               'fraction_of_case_FN': float(sizes[index] / max(int(fn.sum()), 1)),
               'missed_GT_skeleton_voxels': int(sk_counts[index]), 'missed_GT_skeleton_radius_le1mm': int(thin_counts[index]),
               'missed_GT_skeleton_radius_gt1mm': int(sk_counts[index] - thin_counts[index]), **physical_box(lo, hi, grid)}
        if rank <= 5:
            local = labels[box] == index + 1
            values = radius[box][local]
            row.update(GT_EDT_radius_median_mm=float(np.median(values)), GT_EDT_radius_p90_mm=float(np.percentile(values, 90)),
                       GT_EDT_radius_max_mm=float(values.max()))
        regions.append(row)
    require(sum(x['missed_GT_skeleton_voxels'] for x in regions) == int((skeleton & fn).sum()), 'FN skeleton accounting does not reconcile')
    return regions


def diagnose_case(case_id, audit, metrics):
    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    image = c.read_image(audit['label_path']); truth = sitk.GetArrayFromImage(image).astype(bool)
    spacing = np.asarray(image.GetSpacing()[::-1]); grid = audit['original_grid']
    coords = np.nonzero(truth)
    minimum, maximum = np.asarray([x.min() for x in coords]), np.asarray([x.max() for x in coords])
    require(np.all(minimum > 1) and np.all(maximum < np.asarray(truth.shape) - 2), 'GT touches volume boundary; cropped EDT/Lee equivalence requires full-grid handling')
    offset = minimum - 2
    box = tuple(slice(int(a), int(b + 3)) for a, b in zip(offset, maximum))
    gt = truth[box]
    skeleton = metrics.skeletonize(gt, method='lee').astype(bool)
    radius = ndi.distance_transform_edt(gt, sampling=spacing)
    thin, thick = skeleton & (radius <= 1.), skeleton & (radius > 1.)
    require(int(thin.sum() + thick.sum()) == int(skeleton.sum()), 'Radius strata lost GT skeleton voxels')
    require(int(thin.sum()) > 0 and int(thick.sum()) > 0, 'Both predefined radius groups must be represented')
    voxel_volume = float(np.prod(spacing)); gt_n = int(truth.sum())
    stats, masks, all_regions = [], {}, []
    for method in ('gta_preview', 'unet'):
        item = audit['methods'][method]; row = item['row']
        predicted_image = c.read_image(item['prediction_path']); c.check_geometry(c.geometry(predicted_image), grid, case_id)
        p = sitk.GetArrayFromImage(predicted_image).astype(bool); masks[method] = p
        pn, tp = int(p.sum()), int((p & truth).sum())
        fp, fn = pn - tp, gt_n - tp
        small_hits, large_hits = int((thin & p[box]).sum()), int((thick & p[box]).sum())
        recall_sk = (small_hits + large_hits) / int(skeleton.sum())
        small_recall, large_recall = small_hits / int(thin.sum()), large_hits / int(thick.sum())
        require(int(thin.sum()) == row['thin_centerline_voxels']
                and close(small_recall, row['thin_centerline_recall_radius_le_1mm'])
                and close(recall_sk, row['centerline_recall']), 'Cropped exact Lee/radius coverage does not reproduce frozen full-grid metrics')
        values = {'case_id': case_id, 'method': method, 'GT_voxels': gt_n, 'prediction_voxels': pn,
                  'TP_voxels': tp, 'FP_voxels': fp, 'FN_voxels': fn,
                  'TP_volume_mm3': tp * voxel_volume, 'FP_volume_mm3': fp * voxel_volume, 'FN_volume_mm3': fn * voxel_volume,
                  'precision': tp / max(pn, 1), 'recall': tp / gt_n, 'dice': 2 * tp / max(pn + gt_n, 1),
                  'GT_skeleton_voxels': int(skeleton.sum()), 'GT_skeleton_radius_le1mm_voxels': int(thin.sum()),
                  'GT_skeleton_radius_gt1mm_voxels': int(thick.sum()),
                  'covered_GT_skeleton_radius_le1mm_voxels': small_hits, 'covered_GT_skeleton_radius_gt1mm_voxels': large_hits,
                  'centerline_recall': recall_sk, 'centerline_recall_radius_le1mm': small_recall,
                  'centerline_recall_radius_gt1mm': large_recall,
                  'prediction_components_26_recorded': row['prediction_components']}
        for field in ('precision', 'recall', 'dice'): require(close(values[field], row[field]), 'Pixel count metric differs: ' + field)
        # Include every predicted voxel for the surface check; no FP is cropped out.
        surface = surfaces(p, truth, spacing, metrics)
        for field in ('prediction_surface_voxels', 'truth_surface_voxels', 'assd_mm', 'asd_prediction_to_truth_mm', 'asd_truth_to_prediction_mm'):
            require(close(surface[field], row[field]), 'Saved surface distance/count differs: ' + case_id + '/' + method + '/' + field)
        values.update(surface)
        regions = false_negative_regions(gt & ~p[box], skeleton, thin, radius, offset, grid, case_id, method)
        require(sum(r['FN_voxels'] for r in regions) == fn, 'Global FN differs from complete GT crop')
        values.update(FN_components_26=len(regions), largest_FN_voxels=regions[0]['FN_voxels'] if regions else 0,
                      top5_FN_fraction=sum(r['FN_voxels'] for r in regions[:5]) / max(fn, 1))
        stats.append(values); all_regions.extend(regions)
        del predicted_image
        emit('saved_pixel_error_decomposition_complete', case_id=case_id, method=method)
    a, b = masks['gta_preview'], masks['unet']
    transition = {'case_id': case_id, 'GT_newly_recovered_by_GTA_voxels': int((truth & a & ~b).sum()),
                  'GT_newly_missed_by_GTA_voxels': int((truth & b & ~a).sum()),
                  'FP_newly_added_by_GTA_voxels': int((~truth & a & ~b).sum()),
                  'FP_removed_by_GTA_voxels': int((~truth & b & ~a).sum())}
    for key, selected in [('radius_le1mm', thin), ('radius_gt1mm', thick)]:
        transition['GT_skeleton_newly_recovered_' + key] = int((selected & a[box] & ~b[box]).sum())
        transition['GT_skeleton_newly_missed_' + key] = int((selected & b[box] & ~a[box]).sum())
    transition['TP_GTA_minus_UNet'] = stats[0]['TP_voxels'] - stats[1]['TP_voxels']
    transition['FP_GTA_minus_UNet'] = stats[0]['FP_voxels'] - stats[1]['FP_voxels']
    require(transition['TP_GTA_minus_UNet'] == transition['GT_newly_recovered_by_GTA_voxels'] - transition['GT_newly_missed_by_GTA_voxels']
            and transition['FP_GTA_minus_UNet'] == transition['FP_newly_added_by_GTA_voxels'] - transition['FP_removed_by_GTA_voxels'],
            'Paired voxel transition accounting does not reconcile')
    return {'case_id': case_id, 'original_grid': grid, 'GT_bbox': physical_box(minimum, maximum, grid),
            'GT_skeleton_crop': {'offset_zyx': offset.tolist(), 'shape_zyx': list(gt.shape), 'background_margin_voxels': 2,
                                 'both_model_full_grid_frozen_skeleton_metrics_reproduced': True},
            'methods': stats, 'transitions': transition,
            'largest_FN_regions': {m: [r for r in all_regions if r['method'] == m][:5] for m in ('gta_preview', 'unet')}}, all_regions


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-json', type=Path, default=ROOT / 'server_records/gta_unet_preview_pair_verified_20261014/paired_comparison.json')
    parser.add_argument('--gta-evaluation', type=Path, default=ROOT / 'server_records/gta_fixed_final_preview_capture_20261014/results/gta_fixed_final_preview_20261014')
    parser.add_argument('--unet-evaluation', type=Path, default=ROOT / 'server_records/unet_complete_trajectory_20261014/results/unet_recipe200x125/internal4_trajectory/final')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'server_records/gta_fixed_final_preview_capture_20261014/mirror_view/manifest.json')
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'server_records/pixel_error_diagnosis_20261014')
    args = parser.parse_args()
    for key in vars(args): setattr(args, key, getattr(args, key).resolve())
    require(not args.output.exists(), 'Use a fresh diagnosis output directory')
    pair, manifest = c.read(args.pair_json), c.read(args.manifest)
    require(pair.get('status') == 'verified_preview_fixed_final_pair' and pair.get('case_ids') == c.IDS
            and pair.get('tool_sha256') == PAIR_TOOL_SHA and pair.get('canonical_agreement_verified') is False,
            'Requires actual verified d4f5 preview pair, not an invented canonical result')
    require(c.digest(args.manifest) == c.EXPECTED_MANIFEST and c.digest(ROOT / 'code/metrics.py') == METRICS_SHA,
            'Manifest/frozen metric source identity differs')
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    metrics = c.load_module('saved_error_frozen_metrics', ROOT / 'code/metrics.py')
    audits, inputs = preflight(args, pair, manifest)
    emit('all4_physical_grids_labels_and_predictions_verified_before_diagnosis')
    cases, components = [], []
    for case_id in c.IDS:
        result, regions = diagnose_case(case_id, audits[case_id], metrics)
        cases.append(result); components.extend(regions)
    for path, expected in inputs.items(): require(c.digest(path) == expected, 'Input changed during diagnosis: ' + path)
    require(c.digest(ROOT / 'code/metrics.py') == METRICS_SHA, 'Frozen metrics changed during diagnosis')
    result = {'schema_version': 1, 'role': 'four development patients; old-server GTA early preview; canonical agreement pending',
              'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'case_ids': c.IDS,
              'source_pair_sha256': c.digest(args.pair_json), 'source_pair_tool_sha256': PAIR_TOOL_SHA,
              'tool_sha256': c.digest(__file__), 'frozen_metrics_sha256': METRICS_SHA,
              'CPU_threads_max_requested': 2, 'GPU_used': False, 'model_or_forward_used': False,
              'centerline_definition': 'Lee GT skeleton voxel coverage, no distance tolerance; anisotropic physical EDT radius<=1mm versus>1mm; not mm-weighted skeleton length or clinical branch truth',
              'FN_regions_definition': '26-neighbour connected regions of GT-positive/prediction-negative voxels; geometric boxes and EDT-radius proxies only, no distal/proximal/trunk anatomical attribution',
              'surface_definition': metrics.METRIC_PROTOCOL, 'surface_recalculation': 'exact Euclidean nearest6-neighbour-surface voxel using KDTree workers=2; reproduced frozen EDT row distances',
              'causal_boundary': 'Error accounting does not identify the cause in Gaussian loss/classifier/initialization. No significance test/model winner from4 cases.',
              'cases': cases, 'source_inputs_sha256': inputs}
    args.output.mkdir(parents=True)
    (args.output / 'pixel_error_diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_csv(args.output / 'per_case_error_metrics.csv', [m for case in cases for m in case['methods']])
    write_csv(args.output / 'paired_voxel_transitions.csv', [case['transitions'] for case in cases])
    write_csv(args.output / 'FN_connected_regions.csv', components)
    lines = ['四例原网格保存像素误差诊断；GTA为旧服务器early fixed-final preview，canonical一致性仍待核验。',
             '先核四例实际GT/预测字节与物理网格，再进行CPU<=2计算；没有模型、forward、训练或新测试。',
             '骨架指标计数Lee skeleton体素；半径为原GT物理EDT代理，不是毫米长度加权或临床分支真值。',
             'FN区域只给26邻接几何bbox/体积/半径，不由几何位置命名远端、近端、主干或冠脉解剖段。',
             '四例全部保留，无p值/显著性/模型胜者；错误分解不是Gaussian或分类器的因果证明。']
    for case in cases:
        lines.append('\n' + case['case_id'])
        for row in case['methods']:
            lines.append(f"{row['method']}: TP={row['TP_voxels']}, FP={row['FP_voxels']}, FN={row['FN_voxels']}; precision={row['precision']:.6f}, recall={row['recall']:.6f}; Dice={row['dice']:.6f}, ASSD={row['assd_mm']}; surface pred→GT={row['asd_prediction_to_truth_mm']}, GT→pred={row['asd_truth_to_prediction_mm']}; skeleton≤1mm={row['centerline_recall_radius_le1mm']:.6f}, >1mm={row['centerline_recall_radius_gt1mm']:.6f}")
        lines.append('paired accounting: ' + json.dumps(case['transitions'], ensure_ascii=False))
    (args.output / 'pixel_error_diagnosis.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    emit('diagnosis_complete', output=str(args.output), n_cases=4, n_method_case_rows=8, n_FN_regions=len(components))


if __name__ == '__main__': main()
