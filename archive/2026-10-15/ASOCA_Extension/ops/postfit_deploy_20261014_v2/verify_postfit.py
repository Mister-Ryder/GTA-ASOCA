"""Read-only bounded GTA postfit verification, outside the frozen model code.

Require epoch010..190 plus fixed final, internal4 at each point, and source-
backed curves. Undefined distances are retained with the producer's summary
policy; this checks execution/provenance, not whether model performance is good.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

EXPECTED_MANIFEST = "141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356"
EXPECTED_SOURCE = "bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652"
SOURCE_ALGORITHM = "gta_fidelity_selected_python_path_and_bytes_v1"
IDS = ["Diseased_10", "Diseased_20", "Normal_12", "Normal_3"]
NAMES = [f"epoch{epoch:03d}" for epoch in range(10, 200, 10)] + ["final"]
SUMMARY_FIELDS = ("dice", "assd_mm", "assd_directional_mean_mm", "cldice", "precision", "recall",
                  "centerline_recall", "centerline_precision", "thin_centerline_recall_radius_le_1mm",
                  "surface_hd95_mm", "asoca_foreground_hd95_mm", "asd_prediction_to_truth_mm",
                  "asd_truth_to_prediction_mm", "prediction_components", "truth_components", "seconds")
DISTANCES = {"surface_hd95_mm", "assd_mm", "assd_directional_mean_mm", "asd_prediction_to_truth_mm",
             "asd_truth_to_prediction_mm", "asoca_foreground_hd95_mm"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same(left, right):
    """Compare reproduced floating summaries without requiring bit-identical summation."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if finite(left) and finite(right):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
    return left == right


def summary_of(rows):
    """Same finite/null and complete-cohort distance policy as frozen evaluate.summarize."""
    result = {"n_cases": len(rows), "empty_predictions": sum(bool(row["empty_prediction"]) for row in rows)}
    for field in SUMMARY_FIELDS:
        values = [row[field] for row in rows if finite(row.get(field))]
        undefined = [row["case_id"] for row in rows if not finite(row.get(field))]
        item = {"mean": statistics.mean(values) if values else None,
                "sd": statistics.stdev(values) if len(values) > 1 else None,
                "n_valid": len(values), "n_undefined": len(undefined), "undefined_case_ids": undefined}
        if field in DISTANCES and len(values) != len(rows):
            item.update(mean=None, sd=None, reason="undefined or unrecorded distances; no complete-cohort mean")
        result[field] = item
    return result


def source_guard(root):
    frozen = read(root / "configs/main_source_freeze.json")
    code = root / "code"
    actual = {path.relative_to(code).as_posix(): path for path in code.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts
              and path.suffix not in (".pyc", ".pyo")}
    require(set(actual) == set(frozen["code_files"]), "Frozen code package membership differs")
    for name, record in frozen["code_files"].items():
        require(actual[name].stat().st_size == record["bytes"] and digest(actual[name]) == record["sha256"],
                "Frozen code changed: " + name)
    for name, record in frozen["command_files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and digest(path) == record["sha256"], "Frozen command changed: " + name)
    source = hashlib.sha256()
    for name, expected in sorted(frozen["training_dependency_files"].items()):
        path = code / name
        require(digest(path) == expected, "Training dependency changed: " + name)
        source.update(name.encode("utf-8")); source.update(path.read_bytes())
    require(source.hexdigest() == frozen["training_source_sha256"] == EXPECTED_SOURCE,
            "Live training dependency identity differs")
    return frozen


def checkpoint_audit(path, config, epoch):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint.get("epoch") == epoch and isinstance(checkpoint.get("model"), dict)
            and bool(checkpoint["model"]), "Checkpoint epoch/model payload differs: " + str(path))
    saved = checkpoint.get("config", {})
    require(all(config.get(key) == saved.get(key) for key in (set(config) | set(saved)) - {"resume", "plan"}),
            "Checkpoint config differs from config.json: " + str(path))
    role = checkpoint.get("checkpoint_role")
    require(role == "periodic internal-four diagnostic; not final selection" if Path(path).parent.name == "checkpoints"
            else role in (None, "fixed final"), "Checkpoint role differs: " + str(path))
    for name, value in checkpoint["model"].items():
        require(torch.is_tensor(value), "Non-tensor model state: " + name)
        require(bool(torch.isfinite(value).all()), "Nonfinite model state: " + name)
    del checkpoint


def preflight(root, training, manifest_path, raw_root):
    root, training, manifest_path, raw_root = map(lambda p: Path(p).resolve(), (root, training, manifest_path, raw_root))
    frozen = source_guard(root)
    require(training == (root / "results/gta_enhanced200x125/training").resolve(), "Unplanned training directory")
    manifest_hash = digest(manifest_path)
    require(manifest_hash == EXPECTED_MANIFEST, "Immutable dataset manifest differs")
    manifest, config = read(manifest_path), read(training / "config.json")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    expected = {"method": "gta", "fold": 0, "epochs": 200, "steps": 125, "save_every": 10,
                "batch_size": 2, "patch_size": 128, "smoke": False, "cohort": "internal28",
                "no_ds": False, "checkpoint_selection": "fixed final", "source_hash_algorithm": SOURCE_ALGORITHM,
                "manifest_sha256": manifest_hash, "source_sha256": EXPECTED_SOURCE,
                "internal_selection_valid": True, "outer8_used_for_selection": False}
    require(all(config.get(key) == value for key, value in expected.items()), "Formal GTA recipe/source identity differs")
    split = config["internal_split"]
    require(split.get("val") == IDS and len(split.get("train", [])) == 28
            and len(set(split["train"])) == 28 and set(split["train"]).isdisjoint(IDS)
            and set(split["train"]) | set(IDS) == set(manifest["folds"][0]["train"])
            and config.get("trained_case_ids") == split["train"], "Internal28/4 patient isolation differs")
    require(config.get("source_files_sha256") == frozen["training_dependency_files"], "Saved per-file training source differs")
    for case_id in IDS:
        require(digest(raw_root / cases[case_id]["label_relative_path"]) == cases[case_id]["source_sha256"]["label"],
                "Original-grid reference changed: " + case_id)
    marker = read(training / "TRAINING_COMPLETE.json")
    final_hash = digest(training / "final.pt")
    for key, value in {"epoch": 199, "smoke": False, "manifest_sha256": manifest_hash, "source_sha256": EXPECTED_SOURCE,
                       "trained_case_ids": split["train"], "final_sha256": final_hash, "segmentation_updates": 25000,
                       "internal_selection_valid": True, "checkpoint_selection": "fixed final"}.items():
        require(marker.get(key) == value, "Training completion differs: " + key)
    history = lines(training / "history.jsonl")
    require([item.get("epoch") for item in history] == list(range(200))
            and all(item.get("iterations") == 125 and finite(item.get("loss")) for item in history),
            "History does not demonstrate the complete finite 25,000-update run")
    periodic = training / "checkpoints"
    require({path.name for path in periodic.glob("epoch*.pt")} <= {name + ".pt" for name in NAMES[:-1]} | {"epoch200.pt"},
            "Unexpected periodic checkpoint inventory; stop for inspection")
    candidates = []
    for name in NAMES:
        path = training / "final.pt" if name == "final" else periodic / (name + ".pt")
        require(path.is_file(), "Missing mandatory trajectory checkpoint: " + name)
        epoch = 199 if name == "final" else int(name[5:]) - 1
        checkpoint_hash = digest(path)
        checkpoint_audit(path, config, epoch)
        require(digest(path) == checkpoint_hash, "Checkpoint changed while being audited: " + name)
        candidates.append({"path": str(path), "epoch": epoch, "name": name, "checkpoint_sha256": checkpoint_hash})
    # epoch200 is a redundant periodic copy, excluded by the frozen trajectory.
    if (periodic / "epoch200.pt").is_file():
        checkpoint_audit(periodic / "epoch200.pt", config, 199)
    sources = {name: digest(root / "code" / name) for name in ("evaluate.py", "metrics.py", "evaluate_trajectory.py")}
    identity = {"training_config_sha256": digest(training / "config.json"), "manifest_sha256": manifest_hash,
                "source_sha256": EXPECTED_SOURCE, "evaluation_sources": sources, "case_ids": IDS,
                "split": "internal_diagnostic", "smoke": False, "checkpoints": candidates, "sw_batch_size": 1,
                "checkpoint_selection": "fixed final remains unchanged; intermediate weights are trajectory diagnostics",
                "formal_final_test": False, "outer8_used_for_selection": False}
    return {"schema_version": 1, "status": "preflight_verified", "root": str(root), "training": str(training),
            "manifest": str(manifest_path), "raw_root": str(raw_root), "trajectory_identity": identity,
            "main_freeze_sha256": digest(root / "configs/main_source_freeze.json"),
            "training_complete_sha256": digest(training / "TRAINING_COMPLETE.json"),
            "history_sha256": digest(training / "history.jsonl"), "mandatory_checkpoints": 20,
            "patients_per_checkpoint": 4, "expected_case_evaluations": 80,
            "role": "internal-four development only; no new training, outer-eight selection or fivefold"}


def usable(entry):
    return finite(entry.get("mean")) and entry.get("n_undefined", 0) == 0 and entry.get("n_valid", 4) == 4


def postflight(root, training, manifest_path, raw_root, previous):
    current = preflight(root, training, manifest_path, raw_root)
    require(current == previous, "Pre/post training, source or reference identity changed")
    root, training = Path(current["root"]), Path(current["training"])
    identity = current["trajectory_identity"]
    directory = training.parent / "internal4_trajectory"
    curves = training.parent / "learning_curves"
    manifest = read(manifest_path)
    cases = {case["case_id"]: case for case in manifest["cases"]}
    config = read(training / "config.json")
    sys.path.insert(0, str(root / "code"))
    from evaluate_trajectory import verify_evaluation
    complete = read(directory / "TRAJECTORY_COMPLETE.json")
    require(all(complete.get(key) == value for key, value in identity.items()) and complete.get("n_evaluations") == 20,
            "Trajectory marker identity or20-checkpoint coverage differs")
    require(read(directory / "trajectory_protocol.json") == identity, "Trajectory protocol differs")
    require(read(directory / "trajectory_status.json") == {"status": "complete", "completed": 20, "planned": 20},
            "Trajectory status lacks complete20-checkpoint coverage")
    records, case_receipts, summary_paths = [], [], []
    for item in identity["checkpoints"]:
        child_dir = directory / item["name"]
        require({path.name for path in child_dir.glob("*_prediction.nrrd")} == {case + "_prediction.nrrd" for case in IDS},
                "Unexpected/missing prediction patients: " + item["name"])
        require({path.name for path in child_dir.glob("*.json")} == {case + ".json" for case in IDS} | {"summary.json", "EVALUATION_COMPLETE.json"},
                "Unexpected/missing per-case JSON patients: " + item["name"])
        summary = verify_evaluation(child_dir, IDS, item["checkpoint_sha256"], identity["manifest_sha256"],
                                    config, identity["evaluation_sources"], cases, item["epoch"])
        rows = [read(child_dir / (case + ".json")) for case in IDS]
        require(lines(child_dir / "case_metrics.jsonl") == rows, "Per-case metrics ledger differs: " + item["name"])
        require(same(summary.get("overall"), summary_of(rows)), "Summary values/null policy differ from all four cases: " + item["name"])
        for group in ("Normal", "Diseased"):
            require(same(summary.get(group), summary_of([row for row in rows if row["group"] == group])),
                    "Stratum summary differs: " + item["name"] + "/" + group)
        role = "fixed final" if item["name"] == "final" else "periodic internal-four diagnostic; not final selection"
        require(summary.get("checkpoint_selection") == role and summary.get("formal_final_test") is False,
                "Checkpoint evaluation role differs: " + item["name"])
        require(all(row.get("checkpoint_selection") == role and row.get("formal_final_test") is False for row in rows),
                "Case evaluation role differs: " + item["name"])
        done = read(child_dir / "EVALUATION_COMPLETE.json")
        expected = {"checkpoint_sha256": item["checkpoint_sha256"], "manifest_sha256": identity["manifest_sha256"],
                    "evaluation_sources": identity["evaluation_sources"], "case_ids": IDS, "summary_sha256": digest(child_dir / "summary.json")}
        require(all(done.get(key) == value for key, value in expected.items()) and finite(done.get("wall_seconds"))
                and done["wall_seconds"] >= 0, "Per-checkpoint completion identity differs: " + item["name"])
        records.append(item | {"evaluation_output": str(child_dir), "summary_path": str(child_dir / "summary.json"),
                              "split": "internal_diagnostic", "role": summary["evaluation_role"], "n_cases": 4,
                              "overall": summary["overall"], "seconds": done["wall_seconds"], "smoke": False,
                              "checkpoint_selection": role, "formal_final_test": False})
        summary_paths.append(child_dir / "summary.json")
        case_receipts.extend({"checkpoint": item["name"], "case_id": row["case_id"],
                              "case_json_sha256": digest(child_dir / (row["case_id"] + ".json")),
                              "prediction_sha256": row["prediction_sha256"], "empty_prediction": row["empty_prediction"],
                              "assd_mm_defined": finite(row.get("assd_mm"))} for row in rows)
    require(lines(directory / "validation_history.jsonl") == records, "Validation ledger differs from verified weights/summaries")
    require(same(complete.get("total_inference_wall_seconds"), sum(row["seconds"] for row in records)), "Inference time ledger differs")
    with (directory / "validation_curve.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 20, "Validation CSV does not contain20 checkpoints")
    for actual, row in zip(csv_rows, records):
        expected = {"epoch": row["epoch"], "completed_epoch": row["epoch"] + 1, "name": row["name"],
                    **{key: row["overall"][key]["mean"] for key in ("dice", "assd_mm", "cldice")},
                    "n_cases": 4, "seconds": row["seconds"]}
        require(actual == {key: "" if value is None else str(value) for key, value in expected.items()}, "Validation CSV differs: " + row["name"])
    curve = read(curves / "curve_manifest.json")
    require(curve.get("development_only") is True and curve.get("formal_final_test") is False
            and curve.get("interpolated_validation") is False and curve.get("source_epoch_base") == 0
            and curve.get("plotter_sha256") == digest(root / "ops/plot_learning_curves.py")
            and len(curve.get("runs", [])) == 1, "Curve manifest identity differs")
    run = curve["runs"][0]
    counts = {key: sum(usable(row["overall"][key]) for row in records) for key in ("dice", "assd_mm")}
    require(run.get("name") == "development" and run.get("method") == "gta" and run.get("n_training_cases") == 28
            and run.get("selection_case_ids") == IDS and run.get("n_recorded_epochs") == 200
            and run.get("n_internal_dice_points") == counts["dice"] and run.get("n_internal_assd_points") == counts["assd_mm"],
            "Curve observation counts/scope differ; null distances must remain missing")
    expected_sources = [training / "config.json", training / "history.jsonl", directory / "validation_history.jsonl"] + summary_paths
    if (training / "gradient_probes.jsonl").is_file():
        expected_sources.append(training / "gradient_probes.jsonl")
    expected_sources = [{"path": str(path.resolve()), "sha256": digest(path)} for path in sorted(set(expected_sources), key=str)]
    require(run.get("sources") == expected_sources, "Curve manifest sources differ from verified observations")
    artifacts = [curves / "development_learning_curves.png", curves / "development_learning_curves.svg"]
    require(curve.get("artifacts") == [str(path) for path in artifacts], "Curve artifact paths differ")
    require(artifacts[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and "<svg" in artifacts[1].read_text(encoding="utf-8"),
            "Curve PNG/SVG artifacts are missing or invalid")
    with (curves / "epoch_summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        points = [row for row in csv.DictReader(stream) if row["category"] == "segmentation_validation"]
    expected_points = {(row["epoch"], "validation." + key): row["overall"][key]["mean"]
                       for row in records for key in ("dice", "assd_mm") if usable(row["overall"][key])}
    require(len(points) == len(expected_points), "Curve point CSV count differs")
    seen = set()
    for point in points:
        key = (int(point["epoch_0based"]), point["metric"])
        require(key in expected_points and key not in seen and same(float(point["value"]), expected_points[key])
                and point["run"] == "development" and point["method"] == "gta" and point["n_cases"] == "4"
                and point["evaluation_split"] == "internal_diagnostic"
                and point["source_file"] == str((directory / "validation_history.jsonl").resolve())
                and point["source_kind"] == "recorded_internal4_validation",
                "Curve CSV point differs from verified internal4 summary")
        seen.add(key)
    require((curves / "curve_notes.txt").stat().st_size > 0, "Curve explanation is missing")
    artifacts += [curves / "epoch_summary.csv", curves / "curve_notes.txt", curves / "curve_manifest.json"]
    return current | {"status": "postfit_verified", "verified_checkpoint_evaluations": 20,
                      "verified_case_evaluations": len(case_receipts), "cases": case_receipts,
                      "trajectory_complete_sha256": digest(directory / "TRAJECTORY_COMPLETE.json"),
                      "validation_history_sha256": digest(directory / "validation_history.jsonl"),
                      "curve_manifest_sha256": digest(curves / "curve_manifest.json"), "finite_curve_observations": counts,
                      "curve_artifacts": [{"path": str(path), "sha256": digest(path)} for path in artifacts]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "post" and args.preflight is None:
        parser.error("--preflight is required for post validation")
    value = (preflight(args.root, args.training, args.manifest, args.raw_root) if args.phase == "pre" else
             postflight(args.root, args.training, args.manifest, args.raw_root, read(args.preflight)))
    if args.receipt:
        atomic_json(args.receipt, value)
    print(json.dumps({key: value[key] for key in ("status", "mandatory_checkpoints", "expected_case_evaluations")}
                     | {"receipt": str(args.receipt) if args.receipt else None}), flush=True)


if __name__ == "__main__":
    main()
