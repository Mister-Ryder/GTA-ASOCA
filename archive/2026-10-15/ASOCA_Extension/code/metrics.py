"""Case-wise binary overlap, geometry and centreline evaluation in physical mm."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def binary_metrics(prediction, truth, spacing_zyx, thin_radius_mm=1.):
    p, t = prediction.astype(bool), truth.astype(bool)
    if p.shape != t.shape:
        raise ValueError("Prediction/reference shape mismatch")
    inter = np.count_nonzero(p & t)
    pn, tn = int(p.sum()), int(t.sum())
    skt = skeletonize(t, method="lee").astype(bool)
    skp = skeletonize(p, method="lee").astype(bool) if pn else np.zeros_like(p)
    length_recall = np.count_nonzero(skt & p) / max(int(skt.sum()), 1)
    length_precision = np.count_nonzero(skp & t) / max(int(skp.sum()), 1)
    radius = ndi.distance_transform_edt(t, sampling=spacing_zyx)
    thin = skt & (radius <= thin_radius_mm)
    result = {
        "dice": 2 * inter / max(pn+tn,1), "precision": inter/max(pn,1), "recall": inter/max(tn,1),
        "cldice": 2*length_recall*length_precision/max(length_recall+length_precision,1e-12),
        "centerline_recall": length_recall, "centerline_precision": length_precision,
        "thin_centerline_recall_radius_le_1mm": np.count_nonzero(thin & p)/int(thin.sum()) if thin.any() else None,
        "thin_centerline_voxels": int(thin.sum()), "prediction_components": int(ndi.label(p,np.ones((3,3,3)))[1]),
        "truth_components": int(ndi.label(t,np.ones((3,3,3)))[1]),
        "empty_prediction": not bool(pn), "connectivity_neighbors": 26,
    }
    if pn and tn:
        ps = p & ~ndi.binary_erosion(p)
        ts = t & ~ndi.binary_erosion(t)
        dt_t = ndi.distance_transform_edt(~ts, sampling=spacing_zyx)
        dt_p = ndi.distance_transform_edt(~ps, sampling=spacing_zyx)
        a, b = dt_t[ps], dt_p[ts]
        result["surface_hd95_mm"] = float(max(np.percentile(a,95), np.percentile(b,95)))
        result["assd_mm"] = float((a.sum()+b.sum())/(len(a)+len(b)))
        # ASOCA evaluation.py uses all foreground voxels, not boundary points.
        points_p = np.argwhere(p) * np.asarray(spacing_zyx)
        points_t = np.argwhere(t) * np.asarray(spacing_zyx)
        a = cKDTree(points_t).query(points_p, workers=4)[0]
        b = cKDTree(points_p).query(points_t, workers=4)[0]
        result["asoca_foreground_hd95_mm"] = float(max(np.percentile(a,95), np.percentile(b,95)))
    else:
        result.update(surface_hd95_mm=None,assd_mm=None,asoca_foreground_hd95_mm=None)
    return result
