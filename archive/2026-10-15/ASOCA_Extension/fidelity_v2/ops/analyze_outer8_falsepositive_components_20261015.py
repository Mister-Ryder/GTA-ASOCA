"""Four saved predictions: 26-connected TP/FP attribution, without inference.

No Dice/ASSD calculation, filtering, threshold search or prediction modification.
The GT-expanded bounding box is a diagnostic reference, never an inference crop.
"""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT/'server_records'
RAW = Path('E:/00-Data/冠状动脉-ASOCA/CompressedVersion')
OUTPUT = RECORDS/'outer8_falsepositive_components_20261015'
CASES = ('Diseased_1', 'Normal_16')


def read_image(path):
    # The bundled Windows ITK reader may not open Unicode filesystem paths.
    if os.name == 'nt' and not str(path).isascii():
        with tempfile.TemporaryDirectory(prefix='asoca_component_read_') as folder:
            copied = Path(folder)/'image.nrrd'
            shutil.copy2(path, copied)
            return sitk.ReadImage(str(copied))
    return sitk.ReadImage(str(path))


def fraction(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def analyze(model, case):
    prediction_path = RECORDS/f'outer8_{model}_report_20261015'/'predictions'/f'{case}_prediction.nrrd'
    gt_path = RAW/case.split('_')[0]/'Annotations'/f'{case}.nrrd'
    predicted, reference = read_image(prediction_path), read_image(gt_path)
    if predicted.GetSize() != reference.GetSize():
        raise ValueError('Prediction/GT array geometry differs: '+model+'/'+case)
    for getter in ('GetSpacing', 'GetOrigin', 'GetDirection'):
        if not np.allclose(getattr(predicted, getter)(), getattr(reference, getter)(), rtol=0, atol=1e-6):
            raise ValueError('Prediction/GT '+getter+' differs: '+model+'/'+case)
    p = sitk.GetArrayFromImage(predicted)
    t = sitk.GetArrayFromImage(reference)
    if not np.isin(p, (0, 1)).all() or not np.isin(t, (0, 1)).all():
        raise ValueError('Expected binary saved masks: '+model+'/'+case)
    p, t = p.astype(bool), t.astype(bool)
    spacing = np.asarray(reference.GetSpacing()[::-1], dtype=float)
    if not (spacing > 0).all() or not t.any():
        raise ValueError('Expected positive spacing and nonempty GT')
    labels, number = ndimage.label(p, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel(), minlength=number+1)
    true_positives = np.bincount(labels[t], minlength=number+1)
    false_positives = counts-true_positives

    # Integer voxel centres within the GT voxel-centre bounding box +10 mm
    # along each image axis. Direction is preserved; no mask is cropped/changed.
    coordinates = np.nonzero(t)
    gt_min = np.asarray([int(c.min()) for c in coordinates])
    gt_max = np.asarray([int(c.max()) for c in coordinates])
    lower = np.maximum(0, np.ceil(gt_min-10./spacing).astype(int))
    upper = np.minimum(t.shape, np.floor(gt_max+10./spacing).astype(int)+1)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    inside = np.bincount(labels[box].ravel(), minlength=number+1)
    outside_false_positives = counts-inside
    order = sorted(range(1, number+1), key=lambda i: (-int(counts[i]), i))
    voxel_mm3 = float(np.prod(spacing))
    components = [dict(rank=rank, component_label=i, voxels=int(counts[i]),
        volume_mm3=float(counts[i]*voxel_mm3), TP_voxels=int(true_positives[i]),
        FP_voxels=int(false_positives[i]),
        FP_outside_GT_bbox_plus10mm_voxels=int(outside_false_positives[i]))
        for rank, i in enumerate(order, 1)]
    largest = order[:2]
    total_tp = int(true_positives[1:].sum())
    total_fp = int(false_positives[1:].sum())
    largest_tp = int(true_positives[largest].sum())
    largest_fp = int(false_positives[largest].sum())
    outside_fp = int(outside_false_positives[1:].sum())
    summary = dict(model=model, case_id=case, components_26=number,
        TP_voxels=total_tp, FP_voxels=total_fp,
        largest2_TP_voxels=largest_tp, largest2_FP_voxels=largest_fp,
        largest2_FP_fraction_of_all_FP=fraction(largest_fp, total_fp),
        other_components_FP_voxels=total_fp-largest_fp,
        other_components_FP_fraction_of_all_FP=fraction(total_fp-largest_fp, total_fp),
        largest2_retained_TP_fraction_of_current_prediction_TP=fraction(largest_tp, total_tp),
        other_components_TP_voxels=total_tp-largest_tp,
        FP_outside_GT_bbox_plus10mm_voxels=outside_fp,
        FP_outside_GT_bbox_plus10mm_fraction_of_all_FP=fraction(outside_fp, total_fp))
    return dict(summary=summary, components_ranked_by_volume=components,
        inputs=dict(prediction=str(prediction_path), GT=str(gt_path)),
        voxel_volume_mm3=voxel_mm3, spacing_zyx_mm=spacing.tolist(),
        GT_bbox_index_zyx_inclusive=dict(minimum=gt_min.tolist(), maximum=gt_max.tolist()),
        padded_bbox_index_zyx_half_open=dict(minimum=lower.tolist(), maximum=upper.tolist()))


def main():
    if OUTPUT.exists():
        raise ValueError('Fresh diagnostic output required')
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    results = [analyze(model, case) for case in CASES for model in ('U', 'B')]
    report = dict(status='actual_saved_mask_connected_component_diagnosis',
        method='26-neighbour components ranked by predicted voxel count; TP/FP from original binary GT',
        scope='Two development patients, four existing saved predictions; no model execution or prediction changes',
        largest2_rule='Descriptive accounting only; no connected-component filter applied or selected',
        GT_bbox_policy='Original GT voxel-centre bounding box expanded by10 physical mm along each image axis; diagnostic only, never inference crop',
        limits=['A26-connected component is not an anatomical coronary branch.',
                'FP within a large component cannot be removed by simply discarding the other components.',
                'The largest-two counts describe only this one prespecified accounting rule, without testing alternative thresholds.',
                'No Dice, ASSD, training, inference, metric recomputation or postprocessing search performed.'],
        results=results)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT/'components.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (OUTPUT/'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]['summary']))
        writer.writeheader(); writer.writerows(item['summary'] for item in results)
    print(json.dumps(dict(status=report['status'], output=str(OUTPUT),
        summaries=[item['summary'] for item in results]), indent=2))


if __name__ == '__main__':
    main()
