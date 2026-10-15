"""CPU-only GT -> 0.5-mm cached label -> original-CT-grid sanity check.

Checks exactly the four development-internal patients, not the outer eight.
Reuses the saved evaluator's original_prediction and preparation resampler
without importing Torch. This is a measured label-pipeline audit, NOT a
mathematical ceiling on attainable model Dice/ASSD.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import SimpleITK as sitk


DEFAULT_RAW = "/root/autodl-fs/datasets/ASOCA_GTA_20261014/raw/CompressedVersion"
NOTE = ("Measured nearest-neighbour label roundtrip on four internal development patients. "
        "Not model inference, independent test, or a mathematical performance ceiling. "
        "Failure/empty distances are retained; no complete-case replacement.")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contained(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audited relative path escapes its data root")
    return path


def source_function(path, name, namespace):
    """Load just a reviewed local pure function, avoiding evaluator Torch imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(selected) != 1 or selected[0].decorator_list:
        raise ValueError("Expected one undecorated geometry function: " + name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def geometry(image):
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz_mm": list(image.GetOrigin()), "direction": list(image.GetDirection())}


def check_geometry(actual, expected, label):
    if actual["size_xyz"] != expected["size_xyz"]:
        raise ValueError(label + ": size differs")
    for key in ("spacing_xyz_mm", "origin_xyz_mm", "direction"):
        if not np.allclose(actual[key], expected[key], rtol=0, atol=1e-6):
            raise ValueError(label + ": " + key + " differs")
    spacing = np.asarray(actual["spacing_xyz_mm"])
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(label + ": invalid physical spacing")


def read_image(path, header_only=False):
    # Windows ITK NRRD fopen cannot reliably read Unicode paths. These copies
    # are read-only, scoped temporary files and are removed on leaving context.
    if os.name == "nt" and not str(path).isascii():
        with tempfile.TemporaryDirectory(prefix="asoca_roundtrip_read_") as directory:
            copy = Path(directory) / "input.nrrd"
            if not str(copy).isascii():
                raise ValueError("An ASCII temporary directory is required by this Windows NRRD reader")
            shutil.copy2(path, copy)
            return read_image(copy, header_only)
    if header_only:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path)); reader.ReadImageInformation()
        return geometry(reader)
    return sitk.ReadImage(str(path))


def physical_volume(array, grid):
    voxel_volume = float(np.prod(grid["spacing_xyz_mm"]) * abs(np.linalg.det(np.asarray(grid["direction"]).reshape(3, 3))))
    voxels = int(np.count_nonzero(array))
    return {"foreground_voxels": voxels, "voxel_volume_mm3": voxel_volume, "foreground_volume_mm3": voxels * voxel_volume}


def frozen_split(manifest, split_document, fold, seed, splitter):
    config = {}
    if split_document:
        config = json.loads(split_document.read_text(encoding="utf-8-sig"))
        if "internal_split" in config:
            if config.get("fold", fold) != fold:
                raise ValueError("Configuration belongs to another fold")
            split = config["internal_split"]
        else:
            split = config
    else:
        split = splitter(manifest, fold, seed)
    train, val = split.get("train", []), split.get("val", [])
    if (len(train) != 28 or len(set(train)) != 28 or len(val) != 4 or len(set(val)) != 4
            or set(train) & set(val) or set(train) | set(val) != set(manifest["folds"][fold]["train"])):
        raise ValueError("Expected frozen internal28/4 within the original outer32")
    lookup = {case["case_id"]: case for case in manifest["cases"]}
    if any(sum(lookup[c]["group"] == group for c in val) != 2 for group in ("Normal", "Diseased")):
        raise ValueError("Internal four must contain two Normal and two Diseased patients")
    if set(val) & set(manifest["folds"][fold]["val"]):
        raise ValueError("Outer-eight patients must not enter this data-pipeline diagnostic")
    return split, config


def audit_case(case, raw_root, cache_root, spacing, helpers, save_to=None):
    began = time.monotonic()
    label_path = contained(raw_root, case["label_relative_path"])
    image_path = contained(raw_root, case["image_relative_path"])
    directory = contained(cache_root, case["cache_subdir"])
    cache_path, metadata_path = directory / "label.npy", directory / "case.json"
    cache_hash, metadata_hash = sha256(cache_path), sha256(metadata_path)
    label_hash, image_hash = sha256(label_path), sha256(image_path)
    if label_hash != case["source_sha256"]["label"] or image_hash != case["source_sha256"]["image"]:
        raise ValueError("Raw case differs from its audited image/label identity")
    cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if cached_metadata.get("case_id") != case["case_id"] or cached_metadata.get("source_sha256") != case["source_sha256"]:
        raise ValueError("Cached completion metadata has another case/source identity")
    if cached_metadata.get("preprocessing_version") != case.get("preprocessing_version"):
        raise ValueError("Cache preprocessing version differs from manifest")
    check_geometry(cached_metadata["original_geometry"], case["original_geometry"], "cache original geometry")
    check_geometry(cached_metadata["resampled_geometry"], case["resampled_geometry"], "cache resampled geometry")
    if not np.allclose(case["resampled_geometry"]["spacing_xyz_mm"], [spacing] * 3, rtol=0, atol=1e-9):
        raise ValueError("Cache is not the requested isotropic spacing")
    raw = read_image(label_path)
    raw_grid = geometry(raw)
    ct_grid = read_image(image_path, header_only=True)
    check_geometry(raw_grid, case["original_geometry"], "raw label vs manifest original CT")
    check_geometry(raw_grid, ct_grid, "raw image/label header")
    raw_array = sitk.GetArrayViewFromImage(raw)
    if not np.isin(np.unique(raw_array), [0, 1]).all():
        raise ValueError("Raw label is not binary")
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or list(cache.shape[::-1]) != case["resampled_geometry"]["size_xyz"]:
        raise ValueError("Cache label dtype/shape differs from its grid")
    if not np.isin(np.unique(cache), [0, 1]).all():
        raise ValueError("Cache label is not binary")
    # This follows the exact preparation order: orientation, full FOV, nearest
    # label resampling. Its header must also reproduce the frozen cache grid.
    oriented = sitk.DICOMOrient(raw, "LPS")
    recomputed = helpers["resample"](oriented, spacing, is_label=True)
    check_geometry(geometry(recomputed), case["resampled_geometry"], "recomputed preprocessing grid")
    recomputed_array = sitk.GetArrayViewFromImage(recomputed)
    mismatch = int(np.count_nonzero(cache != recomputed_array))
    roundtrip = helpers["original_prediction"](cache, case, raw)
    helpers["verify_grid"](roundtrip, raw)
    returned = sitk.GetArrayViewFromImage(roundtrip)
    p, t = returned > 0, raw_array > 0
    pn, tn = int(p.sum()), int(t.sum())
    metrics = helpers["surface_metrics"](p, t, raw.GetSpacing()[::-1])
    metrics["dice"] = 2 * int(np.count_nonzero(p & t)) / max(pn + tn, 1)
    metrics["empty_roundtrip"] = pn == 0
    metrics["empty_raw_truth"] = tn == 0
    volumes = {"raw": physical_volume(raw_array, raw_grid),
               "cache": physical_volume(cache, case["resampled_geometry"]),
               "roundtrip": physical_volume(returned, geometry(roundtrip))}
    raw_volume = volumes["raw"]["foreground_volume_mm3"]
    for grid in ("cache", "roundtrip"):
        volumes[grid]["volume_difference_mm3_vs_raw"] = volumes[grid]["foreground_volume_mm3"] - raw_volume
        volumes[grid]["relative_volume_difference_vs_raw"] = volumes[grid]["volume_difference_mm3_vs_raw"] / raw_volume if raw_volume else None
    saved = None
    if save_to:
        destination = save_to / (case["case_id"] + "_label_roundtrip.nrrd")
        sitk.WriteImage(roundtrip, str(destination), True)
        saved = {"path": str(destination.resolve()), "sha256": sha256(destination)}
    identities = {label_path: label_hash, image_path: image_hash, cache_path: cache_hash, metadata_path: metadata_hash}
    if any(sha256(path) != expected for path, expected in identities.items()):
        raise ValueError("Raw/cache files changed while their roundtrip was being evaluated")
    return {"case_id": case["case_id"], "group": case["group"], "audit_completed": True,
            "pipeline_verified": mismatch == 0, "cache_matches_recomputed_raw_gt": mismatch == 0,
            "cache_mismatched_voxels": mismatch, "seconds": time.monotonic() - began,
            "metrics": metrics, "volumes": volumes,
            "geometry": {"raw_label": raw_grid, "raw_ct_header": ct_grid,
                         "cache": case["resampled_geometry"], "roundtrip": geometry(roundtrip)},
            "sha256": {"raw_label": label_hash, "raw_image": image_hash,
                       "cache_label_npy": cache_hash, "cache_case_json": metadata_hash},
            "saved_roundtrip": saved}


def summarize(rows):
    successful = [row for row in rows if row.get("audit_completed") and row.get("pipeline_verified")]
    output = {"n_expected": 4, "n_audited": sum(row.get("audit_completed", False) for row in rows),
              "n_pipeline_verified": len(successful), "pipeline_verified": len(rows) == len(successful) == 4}
    for metric in ("dice", "assd_mm", "assd_directional_mean_mm"):
        values = [row["metrics"].get(metric) for row in successful]
        valid = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
        complete = len(successful) == len(valid) == 4
        output[metric] = {"mean": float(np.mean(valid)) if complete else None,
                          "patient_sd": float(np.std(valid, ddof=1)) if complete else None,
                          "n_valid_verified_cases": len(valid), "complete_four_case_mean": complete}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Future ROOT/dataset/cache/manifest.json")
    parser.add_argument("--raw-root", type=Path, default=Path(DEFAULT_RAW))
    parser.add_argument("--cache-root", type=Path, help="Override cache root; default manifest parent")
    parser.add_argument("--split-file", type=Path, help="Frozen internal_split.json or training config.json; otherwise reproduce seed20261014")
    parser.add_argument("--fold", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--expected-spacing-mm", type=float, default=.5)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parent.parent / "code")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-roundtrip", action="store_true", help="Optional diagnostic GT NRRDs; not model predictions")
    args = parser.parse_args()
    args.manifest = args.manifest / "manifest.json" if args.manifest.is_dir() else args.manifest
    raw_root = args.raw_root.resolve(); cache_root = (args.cache_root or args.manifest.parent).resolve()
    output, code = args.output.resolve(), args.code_root.resolve()
    if output.is_relative_to(raw_root) or output.is_relative_to(cache_root):
        parser.error("Audit output must be outside the read-only raw/cache roots")
    if not np.isfinite(args.expected_spacing_mm) or args.expected_spacing_mm <= 0:
        parser.error("Expected spacing must be finite and positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("dataset") != "ASOCA" or manifest.get("schema_version") != 1 or not manifest.get("complete") or manifest.get("pilot_only"):
        raise ValueError("Expected complete audited ASOCA schema1 manifest")
    manifest_hash = sha256(args.manifest)
    namespaces = {"np": np, "sitk": sitk}
    evaluate = code / "evaluate.py"; prepare = code / "prepare_data.py"; classification = code / "classification_data.py"
    helpers = {"original_prediction": source_function(evaluate, "original_prediction", namespaces),
               "verify_grid": source_function(evaluate, "verify_grid", namespaces),
               "resample": source_function(prepare, "resample_full_volume", namespaces)}
    splitter = source_function(classification, "internal_split", namespaces)
    split, config = frozen_split(manifest, args.split_file, args.fold, args.seed, splitter)
    if config.get("manifest_sha256") and config["manifest_sha256"] != manifest_hash:
        raise ValueError("Split/training configuration belongs to another manifest")
    metrics_path = code / "metrics.py"
    spec = importlib.util.spec_from_file_location("asoca_roundtrip_metrics", metrics_path)
    metrics_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics_module)
    helpers["surface_metrics"] = metrics_module.surface_distance_metrics
    output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__).resolve(), evaluate, prepare, classification, metrics_path]
    source_hashes = {str(path): sha256(path) for path in source_files}
    lookup = {case["case_id"]: case for case in manifest["cases"]}
    rows = []
    for case_id in sorted(split["val"]):
        try:
            row = audit_case(lookup[case_id], raw_root, cache_root, args.expected_spacing_mm, helpers,
                             output if args.save_roundtrip else None)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            row = {"case_id": case_id, "group": lookup[case_id]["group"], "audit_completed": False,
                   "pipeline_verified": False, "error": str(error)}
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in ("case_id", "audit_completed", "pipeline_verified", "metrics", "error")}), flush=True)
    if any(sha256(path) != value for path, value in source_hashes.items()):
        raise ValueError("Geometry/metric source changed during the audit")
    if sha256(args.manifest) != manifest_hash:
        raise ValueError("Manifest changed during the audit")
    result = {"schema_version": 1, "audit": "internal_four_gt_label_nearest_neighbor_roundtrip", "note": NOTE,
              "development_only": True, "formal_final_test": False, "mathematical_performance_ceiling": False,
              "manifest_sha256": manifest_hash, "case_ids": sorted(split["val"]), "internal_split": split,
              "split_file_sha256": sha256(args.split_file) if args.split_file else None,
              "raw_root": str(raw_root), "cache_root": str(cache_root), "expected_spacing_mm": args.expected_spacing_mm,
              "geometry_function_sources_sha256": source_hashes, "metric_protocol": metrics_module.METRIC_PROTOCOL,
              "overall": summarize(rows), "cases": rows}
    (output / "label_roundtrip.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["case_id", "group", "audit_completed", "pipeline_verified", "cache_mismatched_voxels", "dice", "assd_mm",
              "assd_directional_mean_mm", "raw_foreground_mm3", "cache_foreground_mm3", "roundtrip_foreground_mm3", "error"]
    with (output / "label_roundtrip.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            record = {field: row.get(field) for field in fields}
            record.update({key: row.get("metrics", {}).get(key) for key in ("dice", "assd_mm", "assd_directional_mean_mm")})
            for grid in ("raw", "cache", "roundtrip"):
                record[grid + "_foreground_mm3"] = row.get("volumes", {}).get(grid, {}).get("foreground_volume_mm3")
            writer.writerow(record)
    summary = result["overall"]
    (output / "label_roundtrip_notes.txt").write_text(NOTE + "\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if summary["pipeline_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
