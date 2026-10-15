"""Original-grid overlap and explicit surface-distance conventions in mm.

The historical assd_mm remains the pooled mean over both lists of boundary
voxel distances. The equal-weight mean of the two directional means is reported
separately; neither value is a connectivity score or surfel-area weighted.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

METRIC_PROTOCOL_VERSION = 2
ASSD_POOLED_DEFINITION = "pooled_bidirectional_surface_voxel_distances"
ASSD_DIRECTIONAL_DEFINITION = "equal_weight_mean_of_two_directional_surface_means"
SURFACE_STRUCTURE = ndi.generate_binary_structure(3, 1)
SURFACE_DISTANCE_FIELDS = (
    "surface_hd95_mm", "assd_mm", "assd_directional_mean_mm",
    "asd_prediction_to_truth_mm", "asd_truth_to_prediction_mm",
    "asoca_foreground_hd95_mm",
)
METRIC_PROTOCOL = {
    "version": METRIC_PROTOCOL_VERSION,
    "primary_metrics": ["dice", "assd_mm"],
    "assd_mm": {
        "definition": ASSD_POOLED_DEFINITION,
        "formula": "(sum(d_prediction_to_truth)+sum(d_truth_to_prediction))/(n_prediction_surface+n_truth_surface)",
        "unit": "mm", "surface_area_weighted": False,
    },
    "assd_directional_mean_mm": {
        "definition": ASSD_DIRECTIONAL_DEFINITION,
        "formula": "(mean(d_prediction_to_truth)+mean(d_truth_to_prediction))/2",
        "unit": "mm", "surface_area_weighted": False,
    },
    "surface_definition": "foreground boundary voxels removed by one 6-neighbour binary erosion, border_value=0",
    "surface_connectivity_neighbors": 6,
    "component_connectivity_neighbors": 26,
    "surface_hd95_mm": "max of the two directional surface-distance 95th percentiles",
    "asoca_foreground_hd95_mm": "max of the two directional ALL-foreground-voxel distance 95th percentiles",
    "evaluation_grid": "original CT", "spacing_axis_order": "z,y,x",
    "empty_distance_policy": "null; cohort distance mean undefined if any required distance is undefined",
    "cldice_role": "auxiliary skeleton overlap; ASSD is a boundary error, not connectivity",
}


def checked_inputs(prediction, truth, spacing_zyx):
    p, t = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    if p.shape != t.shape or p.ndim != 3:
        raise ValueError("Prediction/reference must be same-shape 3D binary volumes")
    spacing = np.asarray(spacing_zyx, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three finite positive physical spacings")
    return p, t, spacing


def surface_distance_metrics(prediction, truth, spacing_zyx):
    """Geometry-only update for already saved predictions; no model inference."""
    p, t, spacing = checked_inputs(prediction, truth, spacing_zyx)
    ps = p & ~ndi.binary_erosion(p, structure=SURFACE_STRUCTURE, border_value=0)
    ts = t & ~ndi.binary_erosion(t, structure=SURFACE_STRUCTURE, border_value=0)
    result = {key: None for key in SURFACE_DISTANCE_FIELDS}
    result.update(metric_protocol_version=METRIC_PROTOCOL_VERSION,
                  assd_definition=ASSD_POOLED_DEFINITION,
                  assd_directional_mean_definition=ASSD_DIRECTIONAL_DEFINITION,
                  surface_connectivity_neighbors=6,
                  surface_area_weighted=False,
                  prediction_surface_voxels=int(ps.sum()), truth_surface_voxels=int(ts.sum()))
    if not ps.any() or not ts.any():
        return result
    dt_t = ndi.distance_transform_edt(~ts, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~ps, sampling=spacing)
    a, b = dt_t[ps], dt_p[ts]  # prediction -> truth, truth -> prediction
    result["surface_hd95_mm"] = float(max(np.percentile(a,95), np.percentile(b,95)))
    # Keep the legacy arithmetic and key; add a distinct directional convention.
    result["assd_mm"] = float((a.sum()+b.sum())/(len(a)+len(b)))
    result["asd_prediction_to_truth_mm"] = float(a.mean())
    result["asd_truth_to_prediction_mm"] = float(b.mean())
    result["assd_directional_mean_mm"] = float((a.mean()+b.mean())/2)
    points_p, points_t = np.argwhere(p) * spacing, np.argwhere(t) * spacing
    a = cKDTree(points_t).query(points_p, workers=4)[0]
    b = cKDTree(points_p).query(points_t, workers=4)[0]
    result["asoca_foreground_hd95_mm"] = float(max(np.percentile(a,95), np.percentile(b,95)))
    return result


def binary_metrics(prediction, truth, spacing_zyx, thin_radius_mm=1.):
    p, t, spacing_zyx = checked_inputs(prediction, truth, spacing_zyx)
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
    result.update(surface_distance_metrics(p, t, spacing_zyx))
    return result


def analytic_geometry_checks():
    """Geometric fixtures with known answers, independent of EDT implementation."""
    cube = np.zeros((9, 9, 9), dtype=np.uint8)
    cube[2:7, 2:7, 2:7] = 1
    same = binary_metrics(cube, cube, (2., 1.5, .5))
    assert same["dice"] == 1 and all(same[key] == 0 for key in SURFACE_DISTANCE_FIELDS)
    one, shifted = np.zeros_like(cube), np.zeros_like(cube)
    one[3, 3, 3], shifted[5, 3, 3] = 1, 1
    moved = binary_metrics(one, shifted, (2., 1.5, .5))
    assert moved["dice"] == 0 and all(np.isclose(moved[key], 4.) for key in SURFACE_DISTANCE_FIELDS)
    two = one.copy()
    two[3, 3, 5] = 1
    unequal = binary_metrics(one, two, (1., 1., 1.))
    assert np.isclose(unequal["dice"], 2/3)
    assert np.isclose(unequal["assd_mm"], 2/3)  # distances [0] and [0,2]
    assert np.isclose(unequal["assd_directional_mean_mm"], .5)
    assert unequal["asd_prediction_to_truth_mm"] == 0 and unequal["asd_truth_to_prediction_mm"] == 1
    reverse = binary_metrics(two, one, (1., 1., 1.))
    assert reverse["assd_mm"] == unequal["assd_mm"]
    assert reverse["assd_directional_mean_mm"] == unequal["assd_directional_mean_mm"]
    empty = binary_metrics(np.zeros_like(cube), cube, (1., 1., 1.))
    assert empty["empty_prediction"] and empty["dice"] == 0
    assert all(empty[key] is None for key in SURFACE_DISTANCE_FIELDS)
    for spacing in ((1., 0., 1.), (1., 1.), (1., np.nan, 1.)):
        try:
            checked_inputs(one, two, spacing)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid physical spacing was accepted")
    return {"identity": "passed", "anisotropic_translation_4mm": "passed",
            "unequal_surface_counts_pooled_2over3_directional_half": "passed",
            "symmetry": "passed", "empty_prediction": "passed", "invalid_spacing": "passed"}


if __name__ == "__main__":
    import json
    print(json.dumps(analytic_geometry_checks()), flush=True)
