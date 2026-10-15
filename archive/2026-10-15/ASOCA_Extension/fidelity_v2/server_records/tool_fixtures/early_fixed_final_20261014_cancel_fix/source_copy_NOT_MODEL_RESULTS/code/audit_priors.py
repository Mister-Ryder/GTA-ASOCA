"""Read-only, deterministic ASOCA prior-label audit; numpy/scipy only, no torch.

Training cases only. Auxiliary labels stay unchanged. Image-only HU centroid
baselines use leave-one-patient-out folds inside the 32 training cases. They are
descriptive probes, not evidence that a neural classifier can learn these labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

from data import PatchDataset, case_directory, extract_patch, load_manifest

HEADS = ("generation", "structure", "location")
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)
HU_BINS = np.array([-200., -100., 0., 100., 200., 400., 600., 800., 1000.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(values):
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.quantile(array, .1)), "p90": float(np.quantile(array, .9))}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "rho": None, "reason": "insufficient observations or constant values"}
    return {"n": len(x), "rho": float(spearmanr(x, y).statistic),
            "inference": "descriptive only; spatial points/patches are correlated, no p-value reported"}


def binary_agreement(predicted, actual):
    predicted, actual = np.asarray(predicted, bool), np.asarray(actual, bool)
    tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
    fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
    return {"n": len(actual), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None}


def location(center_zyx, shape_zyx):
    z = min(2, max(0, int(center_zyx[0] / shape_zyx[0] * 3)))
    x = int(center_zyx[2] >= shape_zyx[2] / 2)
    return 1 + z * 2 + x


def case_geometry(manifest, case, cap):
    folder = case_directory(manifest, case)
    maps = {key: np.load(folder / f"{key}.npy", mmap_mode="r")
            for key in ("radius", "depth", "branch_id")}
    with np.load(folder / "sampling.npz") as archive:
        coords = archive["skeleton"]
    if len(coords) > cap:
        coords = coords[np.linspace(0, len(coords) - 1, cap, dtype=int)]
    index = tuple(coords.T)
    radius, depth, branch = (np.asarray(maps[key][index]) for key in ("radius", "depth", "branch_id"))
    if np.any(radius <= 0) or np.any(branch <= 0):
        raise ValueError("Invalid physical radius or branch assignment at skeleton: " + case["case_id"])
    branch_radius = {int(b): float(np.median(radius[branch == b])) for b in np.unique(branch)}
    corr = correlation(depth, radius)
    row = {"case_id": case["case_id"], "group": case["group"],
           "shape_z": case["shape_zyx"][0], "shape_y": case["shape_zyx"][1], "shape_x": case["shape_zyx"][2],
           "fov_z_mm": float((case["shape_zyx"][0] - 1) * case["spacing_zyx_mm"][0]),
           "fov_x_mm": float((case["shape_zyx"][2] - 1) * case["spacing_zyx_mm"][2]),
           "max_graph_depth": case["auxiliary"]["maximum_graph_depth"],
           "graph_branch_count": case["auxiliary"]["branch_count"],
           "graph_component_roots": len(case["auxiliary"]["graph_roots"]),
           "skeleton_points_audited": len(coords), "depth_radius_spearman": corr["rho"],
           "skeleton_radius_median_mm": float(np.median(radius)),
           "cached_thin_sampling_cutoff_mm": case["auxiliary"]["thin_radius_cutoff_mm"]}
    points = [{"case_id": case["case_id"], "z": int(c[0]), "y": int(c[1]), "x": int(c[2]),
               "branch_id": int(b), "raw_depth": int(d), "radius_mm": float(r)}
              for c, b, d, r in zip(coords, branch, depth, radius)]
    return row, points, branch_radius


def hu_features(image):
    # Existing cached window is (clip(HU,-200,1000)-400)/600. No label mask is
    # used here; foreground contrast would leak target information into a probe.
    hu = np.asarray(image[0], dtype=np.float32) * 600 + 400
    quantiles = np.quantile(hu, [.1, .5, .9])
    histogram = np.histogram(hu, bins=HU_BINS)[0].astype(float) / hu.size
    values = np.r_[hu.mean(), hu.std(), quantiles, histogram]
    names = ["hu_mean", "hu_std", "hu_p10", "hu_p50", "hu_p90"] + [
        f"hu_hist_{int(lo)}_{int(hi)}" for lo, hi in zip(HU_BINS[:-1], HU_BINS[1:])]
    return dict(zip(names, map(float, values)))


def audit_patch(dataset, sample, index, strategy, case, branch_radius, thin_mm, pad_mm):
    arrays = dataset._get_case(sample["case_id"])
    start = sample["start_zyx"].astype(int)
    size = np.asarray(sample["label"].shape)
    shape = np.asarray(arrays["image"].shape)
    foreground = sample["label"] > 0
    branches = extract_patch(arrays["branch_id"], start, tuple(size))
    depths = extract_patch(arrays["depth"], start, tuple(size))
    radii = extract_patch(arrays["radius"], start, tuple(size)).astype(float)
    # Use a one-voxel halo so the degree sees neighbours outside this patch.
    halo = extract_patch(arrays["skeleton"], start - 1, tuple(size + 2)).astype(bool)
    degree = ndi.convolve(halo.astype(np.uint8), CONNECTIVITY).astype(np.int16) - halo
    skeleton = halo[1:-1, 1:-1, 1:-1]
    junction = skeleton & (degree[1:-1, 1:-1, 1:-1] >= 3)
    junction_count = int(ndi.label(junction, structure=CONNECTIVITY)[1])
    ids, counts = np.unique(branches[foreground], return_counts=True)
    missing = int(counts[ids == 0].sum())
    ids, counts = ids[ids > 0], counts[ids > 0]
    dominant = int(ids[counts.argmax()]) if len(ids) else 0
    dominant_depth = None
    if dominant:
        raw = depths[foreground & (branches == dominant)].astype(int)
        dominant_depth = int(np.bincount(raw).argmax())
    local = radii[skeleton & (branches == dominant)] if dominant else np.array([])
    sk_radius = radii[skeleton]
    spacing = np.asarray(case["resampled_geometry"]["spacing_xyz_mm"])[::-1]
    center = start + size / 2
    inside = np.maximum(0, np.minimum(start + size, shape) - np.maximum(start, 0))
    row = {"sample_index": index, "strategy": strategy, "case_id": sample["case_id"], "group": case["group"],
           "sampling_stratum": sample["sampling_stratum"], "generation": int(sample["priors"][0]),
           "structure": int(sample["priors"][1]), "location": int(sample["priors"][2]),
           "start_z": int(start[0]), "start_y": int(start[1]), "start_x": int(start[2]),
           "center_z_fraction": float(center[0] / shape[0]), "center_x_fraction": float(center[2] / shape[2]),
           "foreground_voxels": int(foreground.sum()), "foreground_fraction": float(foreground.mean()),
           "foreground_components_26": int(ndi.label(foreground, structure=CONNECTIVITY)[1]),
           "branch_count_in_patch": len(ids), "foreground_voxels_missing_branch": missing,
           "dominant_branch_id": dominant, "dominant_raw_depth": dominant_depth,
           "dominant_branch_global_radius_mm": branch_radius.get(dominant),
           "dominant_branch_local_radius_mm": float(np.median(local)) if len(local) else None,
           "skeleton_voxels": int(skeleton.sum()), "junction_voxels": int(junction.sum()),
           "junction_clusters_in_patch": junction_count,
           "thin_skeleton_fraction_at_configured_radius": float((sk_radius <= thin_mm).mean()) if len(sk_radius) else None,
           "padding_fraction": float(1 - np.prod(inside) / np.prod(size)),
           **hu_features(sample["image"])}
    if row["generation"] not in range(7) or row["structure"] not in range(7) or row["location"] not in range(7):
        raise ValueError("Prior outside seven-class range")
    if foreground.any():
        if location(center, shape) != row["location"]:
            raise ValueError("Location label no longer matches frozen data definition")
        gt_center = start + np.array(ndi.center_of_mass(foreground))
        row["location_at_foreground_centroid"] = location(gt_center, shape)
        for axis, name in ((0, "z"), (2, "x")):
            amount = int(round(pad_mm / spacing[axis]))
            for side in ("before", "after"):
                new_center, new_shape = center.copy(), shape.copy()
                new_shape[axis] += amount
                if side == "before":
                    new_center[axis] += amount
                row[f"location_if_{pad_mm:g}mm_pad_{side}_{name}"] = location(new_center, new_shape)
    return row


def classification_stats(truth, prediction):
    matrix = np.zeros((7, 7), dtype=int)
    np.add.at(matrix, (truth, prediction), 1)
    support = matrix.sum(1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(7), where=support > 0)
    nonempty = truth > 0
    return {"n": len(truth), "accuracy": float(np.mean(truth == prediction)),
            "balanced_accuracy_observed_classes": float(recall[support > 0].mean()),
            "nonempty_accuracy": float((truth[nonempty] == prediction[nonempty]).mean()) if nonempty.any() else None,
            "support_0_to_6": support.tolist(), "confusion_matrix_true_rows_pred_columns": matrix.tolist()}


def loco_probes(rows):
    features = [key for key in rows[0] if key.startswith("hu_")]
    x = np.asarray([[row[k] for k in features] for row in rows], float)
    cases = np.asarray([row["case_id"] for row in rows])
    predictions, reports = [], {}
    for head in HEADS:
        target = np.asarray([row[head] for row in rows], int)
        majority, centroid = np.zeros(len(rows), int), np.zeros(len(rows), int)
        for case_id in np.unique(cases):
            train, test = cases != case_id, cases == case_id
            if not train.any():
                raise ValueError("Need more than one training case for leave-one-case-out probes")
            majority[test] = int(np.bincount(target[train], minlength=7).argmax())
            mean, scale = x[train].mean(0), x[train].std(0)
            scale[scale < 1e-6] = 1
            standardized = (x - mean) / scale
            classes = np.unique(target[train])
            centers = np.vstack([standardized[train & (target == c)].mean(0) for c in classes])
            distance = ((standardized[test, None, :] - centers[None, :, :]) ** 2).mean(2)
            centroid[test] = classes[distance.argmin(1)]
        counts = np.bincount(target, minlength=7)
        reports[head] = {"observed_global_majority_fraction": float(counts.max() / len(rows)),
                         "loco_majority": classification_stats(target, majority),
                         "loco_image_only_hu_nearest_centroid": classification_stats(target, centroid),
                         "note": "Training-cohort descriptive probe; not neural-classifier learnability or final validation."}
        for index, row in enumerate(rows):
            predictions.append({"sample_index": row["sample_index"], "strategy": row["strategy"],
                                "case_id": row["case_id"], "head": head, "true_class": int(target[index]),
                                "loco_majority_pred": int(majority[index]), "loco_hu_centroid_pred": int(centroid[index])})
    return reports, predictions


def summarize_patches(rows, thin_mm, pad_mm):
    distributions, shape_rows = [], []
    foreground = [r for r in rows if r["foreground_voxels"]]
    for head in HEADS:
        for value in range(7):
            selected = [r for r in rows if r[head] == value]
            distributions.append({"head": head, "class": value, "n_patches": len(selected),
                                  "fraction": len(selected) / len(rows),
                                  "n_cases": len({r["case_id"] for r in selected})})
    for value in range(7):
        selected = [r for r in rows if r["structure"] == value]
        shape_rows.append({"structure_class": value, "n": len(selected),
                           "mean_branch_count": float(np.mean([r["branch_count_in_patch"] for r in selected])) if selected else None,
                           "junction_count": numbers([r["junction_clusters_in_patch"] for r in selected]),
                           "fraction_without_junction": float(np.mean([r["junction_clusters_in_patch"] == 0 for r in selected])) if selected else None,
                           "fraction_exactly_one_junction": float(np.mean([r["junction_clusters_in_patch"] == 1 for r in selected])) if selected else None})
    caliber = [r for r in foreground if r["dominant_branch_global_radius_mm"] is not None]
    report = {"n_patches": len(rows), "n_cases": len({r["case_id"] for r in rows}),
              "nonempty_patches": len(foreground), "sampling_strata": dict(Counter(r["sampling_stratum"] for r in rows)),
              "duplicate_case_start_pairs": len(rows) - len({(r["case_id"], r["start_z"], r["start_y"], r["start_x"]) for r in rows}),
              "foreground_fraction": numbers([r["foreground_fraction"] for r in rows]),
              "padding_fraction": numbers([r["padding_fraction"] for r in rows]),
              "prior_class_distributions": distributions, "structure_class_junction_profile": shape_rows,
              "dominant_raw_depth_vs_global_branch_radius": correlation(
                  [r["dominant_raw_depth"] for r in caliber], [r["dominant_branch_global_radius_mm"] for r in caliber]),
              "head_class_ge5_against_dominant_radius_thin": binary_agreement(
                  [r["generation"] >= 5 for r in caliber], [r["dominant_branch_global_radius_mm"] <= thin_mm for r in caliber]),
              "raw_depth_ge5_against_dominant_radius_thin": binary_agreement(
                  [r["dominant_raw_depth"] >= 5 for r in caliber], [r["dominant_branch_global_radius_mm"] <= thin_mm for r in caliber]),
              "structure_eq3_against_junction_presence": binary_agreement(
                  [r["structure"] == 3 for r in foreground], [r["junction_clusters_in_patch"] > 0 for r in foreground]),
              "structure_ge3_against_junction_presence": binary_agreement(
                  [r["structure"] >= 3 for r in foreground], [r["junction_clusters_in_patch"] > 0 for r in foreground]),
              "generation_class_radius_profile": [{"class": c, **numbers([r["dominant_branch_global_radius_mm"] for r in caliber if r["generation"] == c])} for c in range(1, 7)]}
    fov = {}
    if foreground:
        for key in foreground[0]:
            if key.startswith("location_if_"):
                changed = [r for r in foreground if r[key] != r["location"]]
                fov[key] = {"n_nonempty": len(foreground), "changed_labels": len(changed),
                            "changed_fraction": len(changed) / len(foreground),
                            "cases_with_changed_labels": sorted({r["case_id"] for r in changed})}
    report["fov_only_padding_counterfactual"] = {"pad_mm": pad_mm, "tests": fov,
        "note": "No image content or world-position change; padding before adjusts origin accordingly. This is label-coordinate sensitivity, not an observed scanner intervention."}
    report["patch_center_vs_foreground_centroid_location_disagreement"] = (
        float(np.mean([r["location_at_foreground_centroid"] != r["location"] for r in foreground])) if foreground else None)
    return report, distributions


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--samples", type=int, default=384, help="TOTAL patches; split equally if sampling=both")
    parser.add_argument("--sampling", choices=("balanced", "conventional", "both"), default="balanced")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--thin-radius-mm", type=float, default=1.)
    parser.add_argument("--fov-pad-mm", type=float, default=20.)
    parser.add_argument("--max-centerline-points-per-case", type=int, default=20000)
    args = parser.parse_args()
    if not 250 <= args.samples <= 500 or args.patch_size < 16 or args.thin_radius_mm <= 0 or args.fov_pad_mm <= 0 \
            or args.max_centerline_points_per_case < 32:
        parser.error("Use250..500 patches, patch-size>=16 and positive geometric thresholds")
    started = time.monotonic()
    manifest = load_manifest(args.manifest)
    if not 0 <= args.fold < len(manifest["folds"]):
        raise ValueError("Fold index outside the frozen manifest")
    train_ids = manifest["folds"][args.fold]["train"]
    if len(train_ids) != 32 or len(set(train_ids)) != 32 or set(train_ids) & set(manifest["folds"][args.fold]["val"]):
        raise ValueError("Audit requires the frozen32-case training split, disjoint from development validation")
    if manifest.get("pilot_only") or not manifest.get("complete"):
        raise ValueError("Complete nonpilot cache required")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    strategies = ("balanced", "conventional") if args.sampling == "both" else (args.sampling,)
    geom_rows, points, global_radius = [], [], {}
    for case_id in train_ids:
        row, case_points, radius = case_geometry(manifest, cases[case_id], args.max_centerline_points_per_case)
        geom_rows.append(row)
        points.extend(case_points)
        global_radius[case_id] = radius
    rows = []
    for part, strategy in enumerate(strategies):
        count = args.samples // len(strategies) + int(part < args.samples % len(strategies))
        dataset = PatchDataset(args.manifest, fold=args.fold, split="train", patch_size=args.patch_size,
                               steps=count, seed=args.seed, balanced=strategy == "balanced",
                               augment=False, return_tensors=False)
        dataset.set_epoch(args.epoch)
        for index in range(count):
            sample = dataset[index]
            rows.append(audit_patch(dataset, sample, index, strategy, cases[sample["case_id"]],
                                    global_radius[sample["case_id"]], args.thin_radius_mm, args.fov_pad_mm))
            if (index + 1) % 50 == 0:
                print(f"Audited {strategy}: {index+1}/{count} training patches", flush=True)
    summaries, distributions, predictions = {}, [], []
    groups = [(strategy, [r for r in rows if r["strategy"] == strategy]) for strategy in strategies]
    if len(strategies) > 1:
        groups.append(("pooled_two_sampling_policies", rows))
    for strategy, selected in groups:
        report, head_rows = summarize_patches(selected, args.thin_radius_mm, args.fov_pad_mm)
        report["image_only_probes"], probes = loco_probes(selected)
        summaries[strategy] = report
        distributions.extend({"strategy": strategy, **row} for row in head_rows)
        if strategy != "pooled_two_sampling_policies":
            predictions.extend(probes)
    signature = {"schema_version": 1, "audit": "asoca-priors-training-only-v1", "config": vars(args),
                 "manifest_sha256": digest(manifest["_manifest_path"]),
                 "audit_source_sha256": digest(__file__), "data_source_sha256": digest(Path(__file__).with_name("data.py")),
                 "train_case_ids": train_ids, "excluded_development_case_ids": manifest["folds"][args.fold]["val"],
                 "label_definitions_changed": False, "cache_mutated": False}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "audit_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != signature:
        raise ValueError("Output directory already belongs to a different audit; use a fresh directory")
    identity_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    result = {**signature, "patch_summaries": summaries,
              "all_train_centerline_raw_depth_vs_radius": correlation([p["raw_depth"] for p in points], [p["radius_mm"] for p in points]),
              "per_case_depth_radius_rho": numbers([r["depth_radius_spearman"] for r in geom_rows]),
              "train_case_metadata": geom_rows, "train_cases_without_sampled_patch": sorted(set(train_ids) - {r["case_id"] for r in rows}),
              "seconds": round(time.monotonic() - started, 3),
              "interpretation_limits": [
                  "Branch roots were chosen using radius, so depth-radius association is partially built into label construction.",
                  "Thin means EDT radius<=configured threshold at centreline; not a clinical diagnosis or semantic coronary subclass.",
                  "Junction clusters are26-neighbour skeleton geometry, not named anatomical coronary junction annotations.",
                  "Patch samples overlap and centreline points are spatially correlated; descriptive correlations have no independent-sample p-values.",
                  "HU centroid leave-one-case-out probes use training cases only; they do not establish neural-classifier learnability.",
                  "Dice and ASSD remain primary segmentation endpoints. This audit does not compute segmentation performance; clDice is diagnostic only."]}
    (output / "prior_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(output / "patch_prior_audit.csv", rows)
    write_csv(output / "head_distribution.csv", distributions)
    write_csv(output / "case_geometry_audit.csv", geom_rows)
    case_counts = []
    for strategy, selected in groups:
        for case_id in train_ids:
            case_rows = [row for row in selected if row["case_id"] == case_id]
            for head in HEADS:
                counts = np.bincount([row[head] for row in case_rows], minlength=7)
                for value in range(7):
                    case_counts.append({"strategy": strategy, "case_id": case_id, "head": head, "class": value,
                                        "n": int(counts[value]), "n_case_patches": len(case_rows),
                                        "case_majority_class": int(counts.argmax()) if case_rows else None,
                                        "case_majority_fraction": float(counts.max() / len(case_rows)) if case_rows else None})
    write_csv(output / "case_prior_distribution.csv", case_counts)
    write_csv(output / "train_centerline_depth_radius.csv", points)
    write_csv(output / "hu_loco_predictions.csv", predictions)
    print(json.dumps({"audit_complete": True, "train_cases": len(train_ids), "patches": len(rows),
                      "output": str(output.resolve()), "cache_mutated": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
