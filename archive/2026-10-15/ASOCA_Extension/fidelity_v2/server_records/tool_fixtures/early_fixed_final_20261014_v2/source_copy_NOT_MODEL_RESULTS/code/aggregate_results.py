"""Aggregate verified patient-level ASOCA predictions without mixing smoke runs.

The queue creates JOB_COMPLETE.json only after checking all expected cases and
checkpoint identity. An unverified or partial fold is listed, not promoted into
a publication result. Distances undefined for an empty prediction remain
undefined at the group level; complete-case distance means are not substituted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from metrics import (METRIC_PROTOCOL, METRIC_PROTOCOL_VERSION, ASSD_POOLED_DEFINITION,
                     ASSD_DIRECTIONAL_DEFINITION, SURFACE_DISTANCE_FIELDS)

MAIN_METHODS = ("gta", "unet", "unet_cldice", "swinunetr", "nnunet")
ABLATION_METHODS = ("gta_no_prior", "gta_no_gaussian", "gta_no_balance", "gta_no_ds", "unet_no_ds")
METRICS = (
    "dice", "assd_mm", "assd_directional_mean_mm", "cldice", "precision", "recall", "centerline_recall", "centerline_precision",
    "thin_centerline_recall_radius_le_1mm", "surface_hd95_mm", "asoca_foreground_hd95_mm",
    "asd_prediction_to_truth_mm", "asd_truth_to_prediction_mm", "prediction_components", "truth_components",
)
REQUIRED_LEGACY_DISTANCES = {"surface_hd95_mm", "asoca_foreground_hd95_mm", "assd_mm"}
DISTANCES = set(SURFACE_DISTANCE_FIELDS)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def load_protocol(manifest_path: Path) -> tuple[dict, str]:
    manifest_path = manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("dataset") != "ASOCA" or manifest.get("schema_version") != 1:
        raise ValueError("Only the audited ASOCA schema_version=1 manifest is supported")
    if manifest.get("pilot_only") or not manifest.get("complete"):
        raise ValueError("Pilot or incomplete manifests cannot produce comparison results")
    ids = [case["case_id"] for case in manifest["cases"]]
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if len(ids) != 40 or len(cases) != 40 or len(manifest["folds"]) != 5:
        raise ValueError("Expected 40 distinct labeled patients and five frozen folds")
    if {g: sum(c["group"] == g for c in cases.values()) for g in ("Normal", "Diseased")} != {"Normal": 20, "Diseased": 20}:
        raise ValueError("Expected 20 Normal and 20 Diseased cases")
    seen = []
    for index, fold in enumerate(manifest["folds"]):
        train, val = fold["train"], fold["val"]
        if fold.get("fold", index) != index or len(train) != 32 or len(val) != 8:
            raise ValueError("Every fold must contain 32 train and 8 held-out patients")
        if len(set(train)) != 32 or len(set(val)) != 8 or set(train) & set(val) or set(train) | set(val) != set(ids):
            raise ValueError("Invalid or leaking patient split")
        if any(sum(cases[c]["group"] == group for c in val) != 4 for group in ("Normal", "Diseased")):
            raise ValueError("Each validation fold must have four patients from each stratum")
        seen.extend(val)
    if sorted(seen) != sorted(ids):
        raise ValueError("Out-of-fold validation must evaluate each patient exactly once")
    return manifest, hashlib.sha256(raw).hexdigest()


def finite_number(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def validate_case_row(row: dict, case: dict, fold: int, method: str,
                      manifest_hash: str, checkpoint_hash: str) -> None:
    expected_method = {"gta_no_ds": "gta", "unet_no_ds": "unet"}.get(method, method)
    checks = {"case_id": case["case_id"], "group": case["group"], "fold": fold,
              "method": expected_method, "manifest_sha256": manifest_hash,
              "checkpoint_sha256": checkpoint_hash, "smoke": False,
              "metric_grid": "original CT", "checkpoint_selection": "fixed final"}
    for key, value in checks.items():
        if key not in row or row[key] != value:
            raise ValueError(f"Case {case['case_id']} has incompatible {key}")
    if row.get("label_sha256") != case["source_sha256"]["label"]:
        raise ValueError(f"Case {case['case_id']} has a different reference-label hash")
    if not isinstance(row.get("empty_prediction"), bool):
        raise ValueError("Missing explicit empty-prediction flag")
    for metric in ("dice", "cldice", "precision", "recall", "centerline_recall", "centerline_precision"):
        if not finite_number(row.get(metric)) or not 0 <= row[metric] <= 1:
            raise ValueError(f"Case {case['case_id']} has invalid {metric}")
    for metric in METRICS:
        value = row.get(metric)
        if value is not None and not finite_number(value):
            raise ValueError(f"Case {case['case_id']} has nonfinite {metric}; record undefined values as null")
    if row["empty_prediction"]:
        if row["dice"] != 0 or row["cldice"] != 0 or any(row.get(k) is not None for k in DISTANCES):
            raise ValueError("Empty predictions must retain zero overlap and undefined distances")
    elif any(row.get(k) is None for k in REQUIRED_LEGACY_DISTANCES):
        raise ValueError("A nonempty prediction unexpectedly lacks a distance metric")
    if row.get("assd_definition",ASSD_POOLED_DEFINITION) != ASSD_POOLED_DEFINITION:
        raise ValueError("assd_mm has a different aggregation definition; do not mix conventions")
    if row.get("metric_protocol_version") == METRIC_PROTOCOL_VERSION:
        if row.get("assd_directional_mean_definition") != ASSD_DIRECTIONAL_DEFINITION:
            raise ValueError("Directional ASSD definition is missing or incompatible")
        if not row["empty_prediction"] and any(row.get(k) is None for k in DISTANCES):
            raise ValueError("Schema2 nonempty predictions require all explicit distance fields")


def metric_summary(rows: list[dict], metric: str) -> dict:
    values = [float(row[metric]) for row in rows if finite_number(row.get(metric))]
    missing_ids = [row["case_id"] for row in rows if not finite_number(row.get(metric))]
    output = {"n_cases": len(rows), "n_valid": len(values), "n_undefined": len(missing_ids),
              "n_not_recorded": sum(metric not in row for row in rows),
              "undefined_case_ids": missing_ids, "mean": None, "sd": None}
    if metric in DISTANCES and missing_ids:
        output["reason"] = "Undefined distances, including empty predictions, prevent a complete-cohort mean"
        return output
    if values:
        output["mean"] = float(np.mean(values))
        output["sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return output


def bootstrap_pair(rows_a: dict[str, dict], rows_b: dict[str, dict], metric: str,
                   seed: int, repetitions: int) -> dict:
    common = sorted(set(rows_a) & set(rows_b))
    valid = [case for case in common if finite_number(rows_a[case].get(metric)) and finite_number(rows_b[case].get(metric))]
    differences = np.asarray([rows_a[case][metric] - rows_b[case][metric] for case in valid], dtype=float)
    result = {"metric": metric, "contrast": "gta minus comparator", "n_paired": len(valid),
              "preferred_direction": "lower" if metric in DISTANCES else "higher",
              "paired_case_ids": valid, "n_common": len(common), "excluded_common_case_ids": sorted(set(common) - set(valid)),
              "gta_without_comparator_case_ids": sorted(set(rows_a) - set(rows_b)),
              "comparator_without_gta_case_ids": sorted(set(rows_b) - set(rows_a)),
              "mean_difference": float(differences.mean()) if len(valid) else None,
              "ci95_low": None, "ci95_high": None, "bootstrap_repetitions": repetitions,
              "bootstrap_seed": seed, "bootstrap_unit": "patient, paired, stratified by Normal/Diseased"}
    if metric in DISTANCES and len(valid) != len(common):
        result.update(mean_difference=None,reason="Undefined distances in paired patients; no complete-cohort contrast or interval")
        return result
    if len(valid) < 2:
        result["reason"] = "At least two paired patients are required for a bootstrap interval"
        return result
    rng = np.random.default_rng(seed)
    sums = np.zeros(repetitions, dtype=float)
    for group in ("Normal", "Diseased"):
        indices = [i for i, case in enumerate(valid) if rows_a[case]["group"] == group]
        if indices:
            group_differences = differences[indices]
            sampled = rng.integers(0, len(indices), size=(repetitions, len(indices)))
            sums += group_differences[sampled].sum(axis=1)
    means = sums / len(valid)
    result["ci95_low"], result["ci95_high"] = map(float, np.quantile(means, [.025, .975]))
    return result


def aggregate_outoffold(root: str | Path, manifest_path: str | Path,
                        output: str | Path | None = None, bootstrap_repetitions: int = 10000,
                        seed: int = 20261013) -> dict:
    if bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    results_root = root / "results"
    output = Path(output).resolve() if output else results_root / "comparison"
    manifest, manifest_hash = load_protocol(manifest_path)
    cases = {case["case_id"]: case for case in manifest["cases"]}
    protocol_id = None
    queue_state = results_root / "queue_status.json"
    if queue_state.exists():
        protocol_id = json.loads(queue_state.read_text(encoding="utf-8")).get("protocol_identity")
    by_method, reports, flattened = {}, {}, []
    for method in MAIN_METHODS + ABLATION_METHODS:
        folds = range(5) if method in MAIN_METHODS else (0,)
        expected_ids = sorted(case for fold in folds for case in manifest["folds"][fold]["val"])
        rows, completed_folds, issues = [], [], []
        for fold in folds:
            folder = results_root / method / f"fold{fold}"
            marker_path = folder / "JOB_COMPLETE.json"
            if not marker_path.exists():
                observed = sum((folder / "evaluation" / (case + ".json")).exists() for case in manifest["folds"][fold]["val"])
                issues.append({"fold": fold, "status": "pending_or_unverified", "observed_case_files": observed,
                               "reason": "No verified JOB_COMPLETE marker"})
                continue
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker.get("status") != "completed" or marker.get("manifest_sha256") != manifest_hash:
                    raise ValueError("Completion marker has incompatible status or manifest")
                if marker.get("method_label") != method or marker.get("fold") != fold or marker.get("smoke") is not False:
                    raise ValueError("Completion marker has incompatible method, fold or smoke status")
                if protocol_id and marker.get("protocol_identity") != protocol_id:
                    raise ValueError("Completion marker belongs to another experiment protocol")
                expected = sorted(manifest["folds"][fold]["val"])
                if sorted(marker.get("case_ids", [])) != expected:
                    raise ValueError("Completion marker does not contain the complete frozen validation fold")
                checkpoint_hash = marker["checkpoint_sha256"]
                if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
                    raise ValueError("Completion marker lacks a checkpoint identity")
                fold_rows = []
                for case_id in expected:
                    row = json.loads((folder / "evaluation" / (case_id + ".json")).read_text(encoding="utf-8"))
                    validate_case_row(row, cases[case_id], fold, method, manifest_hash, checkpoint_hash)
                    prediction = folder / "evaluation" / (case_id + "_prediction.nrrd")
                    if not prediction.is_file() or prediction.stat().st_size == 0:
                        raise ValueError("Missing saved original-grid prediction: " + case_id)
                    fold_rows.append(row)
                rows.extend(fold_rows)
                completed_folds.append(fold)
            except (ValueError, KeyError, OSError, TypeError) as error:
                issues.append({"fold": fold, "status": "invalid", "reason": str(error)})
        row_map = {row["case_id"]: row for row in rows}
        if len(row_map) != len(rows):
            raise ValueError("Duplicate out-of-fold predictions for method " + method)
        by_method[method] = row_map
        report = {"method": method, "comparison_type": "fivefold_cv" if method in MAIN_METHODS else "fold0_ablation",
                  "status": "complete" if len(rows) == len(expected_ids) else "incomplete",
                  "n_cases": len(rows), "n_expected": len(expected_ids), "completed_folds": completed_folds,
                  "missing_case_ids": sorted(set(expected_ids) - set(row_map)),
                  "empty_prediction_case_ids": [row["case_id"] for row in rows if row["empty_prediction"]],
                  "issues": issues, "groups": {}}
        for group in ("overall", "Normal", "Diseased"):
            selected = rows if group == "overall" else [row for row in rows if row["group"] == group]
            report["groups"][group] = {metric: metric_summary(selected, metric) for metric in METRICS}
        reports[method] = report
        for row in rows:
            flattened.append({**row, "training_method": row["method"], "method_label": method,
                              "comparison_type": report["comparison_type"]})
    pairs = []
    for method in MAIN_METHODS[1:] + ABLATION_METHODS:
        for metric in ("dice", "assd_mm", "cldice"):
            contrast_seed = seed + int(hashlib.sha256((method + metric).encode()).hexdigest()[:8], 16)
            pair = bootstrap_pair(by_method["gta"], by_method[method], metric, contrast_seed, bootstrap_repetitions)
            pair.update(comparator=method, n_expected=40 if method in MAIN_METHODS else 8,
                        status="complete" if pair["n_paired"] == (40 if method in MAIN_METHODS else 8) else "incomplete")
            pairs.append(pair)
    notes = [
        "Main indicators are Dice (higher better) and assd_mm (lower better), using the explicitly preserved pooled boundary-voxel definition; clDice is auxiliary.",
        "assd_directional_mean_mm is the equal-weight mean of the two directional boundary means; it is diagnostic and is not substituted for assd_mm.",
        "Neither ASSD convention is a connectivity score or surfel-area weighted. Library versions differ; a paper's ASSD label alone does not establish formula equivalence.",
        "Legacy rows lacking a directional field remain unchanged and report unrecorded counts; directional values cannot be recovered from a pooled scalar alone.",
        "The original fold0 eight were already inspected for development. Fidelity_v2 selection uses four reserved cases inside outer train32; this OOF aggregator never relabels those four as outer/final test cases.",
        "These are patient-level internal cross-validation results on 40 publicly annotated ASOCA cases, not official blind-test scores.",
        "Only verified complete folds are aggregated. Incomplete method or paired results remain explicitly provisional.",
        "Normal/Diseased are patient strata; native segmentation is binary coronary lumen versus background.",
        "Standard deviations describe between-patient variation, not variation across independent random seeds.",
        "Paired intervals are stratified patient bootstrap percentile intervals for GTA minus comparator; they are not multiplicity-adjusted hypothesis tests.",
        "Undefined distance metrics are retained as missing with patient IDs; the mean is undefined if any cohort distance is undefined.",
        "ASOCA foreground H95 uses all foreground voxels. Surface HD95 uses boundary voxels and is a different metric.",
        "The fixed main budget is 25000 optimizer updates: custom 200 x 125; native nnU-Net 100 x 250. FOV, optimizer, automatic plans and batch sizes may differ; wall time is not matched.",
        "Fold 0 ablations evaluate eight cases only and must not be merged with fivefold main-result means.",
        "No clinical, hemodynamic, device-design or manual-correction benefit is established by these segmentation metrics.",
    ]
    summary = {"schema_version": 2, "dataset": "ASOCA", "manifest_sha256": manifest_hash,
               "metric_protocol": METRIC_PROTOCOL, "primary_metrics":["dice","assd_mm"],
               "protocol_identity": protocol_id, "methods": reports, "paired_comparisons": pairs, "notes": notes}
    comparison_rows = []
    for method, report in reports.items():
        for group, metrics in report["groups"].items():
            for metric, stats in metrics.items():
                comparison_rows.append({"method": method, "comparison_type": report["comparison_type"], "status": report["status"],
                                        "group": group, "n_expected": report["n_expected"] if group == "overall" else report["n_expected"] // 2,
                                        "metric": metric, **{key: stats.get(key) for key in ("n_cases", "n_valid", "n_undefined", "n_not_recorded", "mean", "sd", "reason")},
                                        "undefined_case_ids": ";".join(stats["undefined_case_ids"])})
    atomic_json(output / "comparison_summary.json", summary)
    write_csv(output / "comparison.csv", comparison_rows,
              ["method", "comparison_type", "status", "group", "n_expected", "metric", "n_cases", "n_valid", "n_undefined", "n_not_recorded", "mean", "sd", "reason", "undefined_case_ids"])
    write_csv(output / "paired_bootstrap.csv", [{**pair, **{key: ";".join(pair[key]) for key in
              ("paired_case_ids", "excluded_common_case_ids", "gta_without_comparator_case_ids", "comparator_without_gta_case_ids")}} for pair in pairs],
              ["comparator", "metric", "preferred_direction", "status", "n_expected", "n_paired", "n_common", "mean_difference", "ci95_low", "ci95_high",
               "bootstrap_repetitions", "bootstrap_seed", "bootstrap_unit", "paired_case_ids", "excluded_common_case_ids",
               "gta_without_comparator_case_ids", "comparator_without_gta_case_ids", "reason"])
    write_csv(output / "per_case_metrics.csv", flattened,
              ["method_label", "training_method", "comparison_type", "case_id", "group", "fold", "empty_prediction"] + list(METRICS)
              + ["assd_definition", "assd_directional_mean_definition", "metric_protocol_version",
                 "evaluation_split", "formal_final_test", "manifest_sha256", "checkpoint_sha256", "label_sha256", "prediction_sha256"])
    lines = ["ASOCA internal cross-validation comparison", "Manifest SHA256: " + manifest_hash, "", *notes, "", "Available results (mean +/- patient SD):"]
    for method, report in reports.items():
        lines.append(f"{method}: {report['n_cases']}/{report['n_expected']} patients; {report['status']}; completed folds {report['completed_folds']}")
        for group in ("overall", "Normal", "Diseased"):
            chunks = []
            for metric in ("dice", "assd_mm", "assd_directional_mean_mm", "cldice", "centerline_recall"):
                stats = report["groups"][group][metric]
                mean = "undefined" if stats["mean"] is None else f"{stats['mean']:.5f}"
                sd = "n/a" if stats["sd"] is None else f"{stats['sd']:.5f}"
                chunks.append(f"{metric}={mean} +/- {sd} (N={stats['n_cases']}, valid={stats['n_valid']}, undefined={stats['n_undefined']})")
            lines.append("  " + group + ": " + "; ".join(chunks))
        if report["empty_prediction_case_ids"]:
            lines.append("  Empty predictions: " + ", ".join(report["empty_prediction_case_ids"]))
        for issue in report["issues"]:
            lines.append(f"  Fold {issue['fold']}: {issue['status']}: {issue['reason']}")
    lines.extend(["", "Paired differences: GTA minus comparator"])
    for pair in pairs:
        lines.append(f"{pair['comparator']} / {pair['metric']}: N={pair['n_paired']}/{pair['n_expected']}, {pair['status']}, mean difference={pair['mean_difference']}, 95% CI=[{pair['ci95_low']}, {pair['ci95_high']}]")
        lines.append("  Paired patient IDs: " + (", ".join(pair["paired_case_ids"]) or "none"))
    atomic_text(output / "comparison.txt", "\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    summary = aggregate_outoffold(args.root, args.manifest, args.output, args.bootstrap_repetitions, args.seed)
    print(json.dumps({"methods": {key: {"n_cases": value["n_cases"], "n_expected": value["n_expected"], "status": value["status"]}
                                 for key, value in summary["methods"].items()}}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
