"""CPU fixture for postfit hardening; creates no training or experiment result.

Uses the real frozen40-case manifest and original four label bytes, tiny real
torch.save model tensors,80 saved NRRD copies and the unmodified plot producer.
Metric numbers are fixture inputs, never ASOCA model-performance evidence.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
from verify_postfit import (IDS, NAMES, EXPECTED_SOURCE, SOURCE_ALGORITHM, SUMMARY_FIELDS, DISTANCES,
                            atomic_json, digest, preflight, postflight, read, summary_of)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tests = {}
    with tempfile.TemporaryDirectory(prefix="gta_postfit_fixture_") as temporary:
        root = Path(temporary).resolve()
        frozen = read(args.root / "configs/main_source_freeze.json")
        for name in frozen["code_files"]:
            target = root / "code" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.root / "code" / name, target)
        for name in frozen["command_files"]:
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.root / name, target)
        shutil.copy2(args.root / "configs/main_source_freeze.json", root / "configs/main_source_freeze.json")
        manifest_path = root / "dataset/cache/manifest.json"; manifest_path.parent.mkdir(parents=True)
        shutil.copy2(args.manifest, manifest_path)
        manifest = read(manifest_path)
        cases = {case["case_id"]: case for case in manifest["cases"]}
        raw_root = root / "raw"
        for case_id in IDS:
            source = args.raw_root / cases[case_id]["label_relative_path"]
            target = raw_root / cases[case_id]["label_relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        training = root / "results/gta_enhanced200x125/training"
        periodic = training / "checkpoints"; periodic.mkdir(parents=True)
        train_ids = sorted(set(manifest["folds"][0]["train"]) - set(IDS))
        config = {"method": "gta", "fold": 0, "epochs": 200, "steps": 125, "save_every": 10,
                  "batch_size": 2, "patch_size": 128, "smoke": False, "cohort": "internal28", "no_ds": False,
                  "checkpoint_selection": "fixed final", "source_hash_algorithm": SOURCE_ALGORITHM,
                  "source_sha256": EXPECTED_SOURCE, "manifest_sha256": digest(manifest_path),
                  "source_files_sha256": frozen["training_dependency_files"], "internal_selection_valid": True,
                  "outer8_used_for_selection": False, "internal_split": {"train": train_ids, "val": IDS},
                  "trained_case_ids": train_ids}
        atomic_json(training / "config.json", config)
        history = [{"epoch": epoch, "iterations": 125, "loss": .2,
                    "stage": "frozen" if epoch < 50 else "partial" if epoch < 70 else "joint"}
                   for epoch in range(200)]
        (training / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        for name in NAMES:
            epoch = 199 if name == "final" else int(name[5:]) - 1
            payload = {"model": {"fixture.weight": torch.tensor([1., 2.])}, "epoch": epoch, "config": config}
            if name != "final":
                payload["checkpoint_role"] = "periodic internal-four diagnostic; not final selection"
            torch.save(payload, training / "final.pt" if name == "final" else periodic / (name + ".pt"))
        atomic_json(training / "TRAINING_COMPLETE.json", {"epoch": 199, "smoke": False,
                    "manifest_sha256": config["manifest_sha256"], "source_sha256": EXPECTED_SOURCE,
                    "trained_case_ids": train_ids, "final_sha256": digest(training / "final.pt"),
                    "segmentation_updates": 25000, "internal_selection_valid": True, "checkpoint_selection": "fixed final"})
        proof = preflight(root, training, manifest_path, raw_root)
        tests["real_torch_serialization_preflight20"] = "passed"
        missing = periodic / "epoch100.pt"
        hidden = periodic / "epoch100.hidden"; missing.rename(hidden)
        try:
            preflight(root, training, manifest_path, raw_root)
        except ValueError as error:
            assert "Missing mandatory trajectory checkpoint: epoch100" in str(error), str(error)
            tests["missing_middle_checkpoint_rejected_before_inference"] = str(error)
        else:
            raise AssertionError("Missing periodic checkpoint was accepted")
        finally:
            hidden.rename(missing)
        directory = training.parent / "internal4_trajectory"; directory.mkdir()
        identity = proof["trajectory_identity"]
        atomic_json(directory / "trajectory_protocol.json", identity)
        records = []
        for item in identity["checkpoints"]:
            target = directory / item["name"]; target.mkdir()
            role = "fixed final" if item["name"] == "final" else "periodic internal-four diagnostic; not final selection"
            rows = []
            for case_id in IDS:
                prediction = target / (case_id + "_prediction.nrrd")
                shutil.copy2(raw_root / cases[case_id]["label_relative_path"], prediction)
                # Exercise null-distance preservation at one point. This is a
                # synthetic metadata fixture, not a pixel-derived performance row.
                empty = item["name"] == "epoch100" and case_id == IDS[0]
                row = {key: 0. if key in DISTANCES else 1. for key in SUMMARY_FIELDS}
                if empty:
                    row.update({key: None for key in DISTANCES}); row["dice"] = 0.
                row.update(case_id=case_id, group=cases[case_id]["group"], empty_prediction=empty,
                           method="gta", fold=0, smoke=False, checkpoint_epoch=item["epoch"],
                           checkpoint_sha256=item["checkpoint_sha256"], manifest_sha256=config["manifest_sha256"],
                           training_source_sha256=EXPECTED_SOURCE, label_sha256=cases[case_id]["source_sha256"]["label"],
                           prediction_sha256=digest(prediction), evaluation_source_sha256=identity["evaluation_sources"]["evaluate.py"],
                           metrics_source_sha256=identity["evaluation_sources"]["metrics.py"], metric_grid="original CT",
                           training_source_verified_against_live_code=True, geometry_only=False,
                           evaluation_split="internal_diagnostic", formal_final_test=False, checkpoint_selection=role)
                atomic_json(target / (case_id + ".json"), row); rows.append(row)
            (target / "case_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = {"case_ids": IDS, "smoke": False, "fold": 0, "method": "gta", "checkpoint_epoch": item["epoch"],
                       "evaluation_split": "internal_diagnostic", "evaluation_role": "internal diagnostic/model-selection cases; not final independent test",
                       "training_source_verified_against_live_code": True, "geometry_only": False,
                       "checkpoint_selection": role, "formal_final_test": False, "overall": summary_of(rows)}
            for group in ("Normal", "Diseased"):
                summary[group] = summary_of([row for row in rows if row["group"] == group])
            atomic_json(target / "summary.json", summary)
            atomic_json(target / "EVALUATION_COMPLETE.json", {"checkpoint_sha256": item["checkpoint_sha256"],
                        "manifest_sha256": config["manifest_sha256"], "evaluation_sources": identity["evaluation_sources"],
                        "case_ids": IDS, "summary_sha256": digest(target / "summary.json"), "wall_seconds": 1.})
            records.append(item | {"evaluation_output": str(target), "summary_path": str(target / "summary.json"),
                                   "split": "internal_diagnostic", "role": summary["evaluation_role"], "n_cases": 4,
                                   "overall": summary["overall"], "seconds": 1., "smoke": False,
                                   "checkpoint_selection": role, "formal_final_test": False})
        ledger = directory / "validation_history.jsonl"
        ledger.write_text("".join(json.dumps(row) + "\n" for row in records))
        atomic_json(directory / "TRAJECTORY_COMPLETE.json", identity | {"n_evaluations": 20, "total_inference_wall_seconds": 20.})
        atomic_json(directory / "trajectory_status.json", {"status": "complete", "completed": 20, "planned": 20})
        with (directory / "validation_curve.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ("epoch", "completed_epoch", "name", "dice", "assd_mm", "cldice", "n_cases", "seconds")
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row in records:
                writer.writerow({"epoch": row["epoch"], "completed_epoch": row["epoch"] + 1, "name": row["name"],
                                 **{key: row["overall"][key]["mean"] for key in ("dice", "assd_mm", "cldice")},
                                 "n_cases": 4, "seconds": 1.})
        subprocess.run([sys.executable, str(root / "ops/plot_learning_curves.py"), "--run", "development=" + str(training),
                        "--validation", "development=" + str(ledger), "--output", str(training.parent / "learning_curves")], check=True)
        result = postflight(root, training, manifest_path, raw_root, proof)
        assert result["verified_case_evaluations"] == 80 and result["finite_curve_observations"] == {"dice": 20, "assd_mm": 19}
        assert len([item for item in result["cases"] if not item["assd_mm_defined"]]) == 1
        tests["complete80_cases_and_real_plot_artifacts"] = "passed"
        tests["undefined_assd_retained_without_excluding_case"] = {"case_evaluations": 80, "dice_points": 20, "assd_points": 19}
        missing = directory / "epoch050" / (IDS[0] + ".json")
        hidden = missing.with_suffix(".hidden"); missing.rename(hidden)
        try:
            postflight(root, training, manifest_path, raw_root, proof)
        except ValueError as error:
            assert "Unexpected/missing per-case JSON patients: epoch050" in str(error), str(error)
            tests["missing_case_json_rejected"] = str(error)
        else:
            raise AssertionError("Missing case was accepted")
        finally:
            hidden.rename(missing)
        wrong = records.copy(); wrong[5] = wrong[5] | {"epoch": 666}
        ledger.write_text("".join(json.dumps(row) + "\n" for row in wrong))
        try:
            postflight(root, training, manifest_path, raw_root, proof)
        except ValueError as error:
            assert "Validation ledger differs" in str(error), str(error)
            tests["stale_ledger_epoch_rejected"] = str(error)
        else:
            raise AssertionError("Changed ledger epoch was accepted")
        tests["fixture_scope"] = "real40-case manifest and original label SHA; tiny tensor payloads and synthetic metric metadata, no network inference/training"
        report = {"status": "passed", "torch": str(torch.__version__), "python": sys.version,
                  "validator_sha256": digest(Path(__file__).parent / "verify_postfit.py"),
                  "controller_sha256": digest(Path(__file__).parent / "after_gta_training.py"), "tests": tests}
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
