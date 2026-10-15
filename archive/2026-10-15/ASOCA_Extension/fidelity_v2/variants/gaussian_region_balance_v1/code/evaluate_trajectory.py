"""Sequential internal-four evaluation of immutable periodic and final weights.

Run after training completes. This runner never evaluates the original eight,
never changes training weights, and never selects a new final checkpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_evaluation(directory, ids, checkpoint_hash, manifest_hash, config, sources, cases, epoch):
    summary = read_json(directory / "summary.json")
    if (summary.get("case_ids") != ids or summary.get("evaluation_split") != "internal_diagnostic"
            or summary.get("smoke") != config["smoke"] or summary.get("fold") != config["fold"]
            or summary.get("method") != config["method"] or summary.get("overall", {}).get("n_cases") != 4
            or summary.get("checkpoint_epoch") != epoch
            or summary.get("training_source_verified_against_live_code") is not True or summary.get("geometry_only") is not False):
        raise ValueError("Trajectory summary identity or patient coverage differs")
    for case_id in ids:
        row = read_json(directory / (case_id + ".json"))
        if (row.get("case_id") != case_id or row.get("checkpoint_sha256") != checkpoint_hash
                or row.get("manifest_sha256") != manifest_hash or row.get("method") != config["method"]
                or row.get("fold") != config["fold"] or row.get("smoke") != config["smoke"]
                or row.get("checkpoint_epoch") != epoch
                or row.get("training_source_verified_against_live_code") is not True or row.get("geometry_only") is not False
                or row.get("evaluation_split") != "internal_diagnostic" or row.get("metric_grid") != "original CT"
                or row.get("group") != cases[case_id]["group"] or row.get("label_sha256") != cases[case_id]["source_sha256"]["label"]
                or row.get("training_source_sha256") != config["source_sha256"]
                or row.get("evaluation_source_sha256") != sources["evaluate.py"]
                or row.get("metrics_source_sha256") != sources["metrics.py"]):
            raise ValueError("Trajectory case metadata differs: " + case_id)
        prediction = directory / (case_id + "_prediction.nrrd")
        if not prediction.is_file() or digest(prediction) != row.get("prediction_sha256"):
            raise ValueError("Trajectory prediction changed: " + case_id)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    args.manifest = str(Path(args.manifest).resolve())
    args.raw_root = str(Path(args.raw_root).resolve())
    training, output = Path(args.training).resolve(), Path(args.output).resolve()
    code = Path(__file__).resolve().parent
    config = read_json(training / "config.json")
    # Every inference uses these live modules. A hash merely copied from a
    # checkpoint would not prove that those modules are the trained version.
    import torch
    from train import source_hash, SOURCE_HASH_ALGORITHM
    if config.get("source_hash_algorithm") != SOURCE_HASH_ALGORITHM or source_hash() != config["source_sha256"]:
        raise ValueError("Live training/model dependencies differ from this frozen v2 run")
    manifest_hash = digest(args.manifest)
    complete = read_json(training / "TRAINING_COMPLETE.json")
    if args.sw_batch_size < 1 or bool(config["smoke"]) != args.smoke:
        raise ValueError("Inference batch/smoke identity differs")
    if (complete.get("manifest_sha256") != manifest_hash or config["manifest_sha256"] != manifest_hash
            or complete.get("source_sha256") != config["source_sha256"]
            or complete.get("final_sha256") != digest(training / "final.pt")
            or complete.get("epoch") != config["epochs"] - 1 or complete.get("smoke") != args.smoke
            or complete.get("trained_case_ids") != config["trained_case_ids"]):
        raise ValueError("Training is incomplete or its immutable final identity differs")
    ids = list(config["internal_split"]["val"])
    if len(ids) != 4 or len(set(ids)) != 4 or set(ids) & set(config["trained_case_ids"]):
        raise ValueError("Four training-internal diagnostic patients must remain untrained")
    cases = {case["case_id"]: case for case in read_json(args.manifest)["cases"]}
    for case_id in ids:
        if digest(Path(args.raw_root) / cases[case_id]["label_relative_path"]) != cases[case_id]["source_sha256"]["label"]:
            raise ValueError("Raw reference changed: " + case_id)
    candidates = []
    for path in sorted((training / "checkpoints").glob("epoch*.pt")):
        completed_epoch = int(path.stem.removeprefix("epoch"))
        if not 0 < completed_epoch <= config["epochs"]:
            raise ValueError("Periodic checkpoint exceeds the planned training budget")
        candidates.append({"path": str(path), "epoch": completed_epoch - 1, "name": path.stem})
    # The final periodic checkpoint equals final epoch but has a distinct role;
    # evaluating both is redundant. Keep the fixed-final weights at that endpoint.
    candidates = [item for item in candidates if item["epoch"] != config["epochs"] - 1]
    candidates.append({"path": str(training / "final.pt"), "epoch": config["epochs"] - 1, "name": "final"})
    for item in candidates:
        item["checkpoint_sha256"] = digest(item["path"])
        checkpoint = torch.load(item["path"], map_location="cpu", weights_only=True)
        saved_config = checkpoint.get("config", {})
        if checkpoint.get("epoch") != item["epoch"] or "model" not in checkpoint:
            raise ValueError("Checkpoint epoch/format differs from its trajectory position")
        keys = set(config) | set(saved_config)
        if any(config.get(key) != saved_config.get(key) for key in keys - {"resume", "plan"}):
            raise ValueError("Checkpoint config differs from the immutable training protocol")
        del checkpoint
    sources = {name: digest(code / name) for name in ("evaluate.py", "metrics.py", "evaluate_trajectory.py")}
    identity = {"training_config_sha256": digest(training / "config.json"), "manifest_sha256": manifest_hash,
                "source_sha256": config["source_sha256"], "evaluation_sources": sources,
                "case_ids": ids, "split": "internal_diagnostic", "smoke": args.smoke,
                "checkpoints": candidates, "sw_batch_size": args.sw_batch_size,
                "checkpoint_selection": "fixed final remains unchanged; intermediate weights are trajectory diagnostics",
                "formal_final_test": False, "outer8_used_for_selection": False}
    if args.plan:
        print(json.dumps(identity, indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "trajectory_protocol.json"
    if protocol.exists() and read_json(protocol) != identity:
        raise ValueError("Existing trajectory uses another immutable protocol; choose a new output folder")
    atomic_json(protocol, identity)
    records = []
    for item in candidates:
        directory = output / item["name"]
        marker = directory / "EVALUATION_COMPLETE.json"
        began = time.monotonic()
        try:
            if (digest(item["path"]) != item["checkpoint_sha256"] or digest(args.manifest) != manifest_hash
                    or source_hash() != config["source_sha256"]):
                raise ValueError("Checkpoint/manifest/model source changed before inference")
            if not marker.exists():
                command = [args.python, str(code / "evaluate.py"), "--manifest", args.manifest,
                           "--raw-root", args.raw_root, "--checkpoint", item["path"], "--output", str(directory),
                           "--fold", str(config["fold"]), "--split", "internal_diagnostic",
                           "--sw-batch-size", str(args.sw_batch_size)]
                if args.smoke:
                    command.append("--smoke")
                subprocess.run(command, check=True, cwd=code)
            summary = verify_evaluation(directory, ids, item["checkpoint_sha256"], manifest_hash, config, sources, cases, item["epoch"])
            if digest(item["path"]) != item["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during inference")
            expected = {"checkpoint_sha256": item["checkpoint_sha256"], "manifest_sha256": manifest_hash,
                        "evaluation_sources": sources, "case_ids": ids, "summary_sha256": digest(directory / "summary.json")}
            if marker.exists():
                saved = read_json(marker)
                if any(saved.get(key) != value for key, value in expected.items()):
                    raise ValueError("Existing trajectory completion marker differs")
                seconds = saved["wall_seconds"]
            else:
                seconds = time.monotonic() - began
                atomic_json(marker, expected | {"wall_seconds": seconds})
            record = item | {"evaluation_output": str(directory), "summary_path": str(directory / "summary.json"),
                             "split": "internal_diagnostic", "role": summary["evaluation_role"], "n_cases": 4,
                             "overall": summary["overall"], "seconds": seconds, "smoke": args.smoke,
                             "checkpoint_selection": summary["checkpoint_selection"], "formal_final_test": False}
            records.append(record)
            ledger = output / "validation_history.jsonl"
            temporary = ledger.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8")
            temporary.replace(ledger)
            atomic_json(output / "trajectory_status.json", {"status": "running", "completed": len(records), "planned": len(candidates)})
        except Exception as error:
            atomic_json(output / "trajectory_status.json", {"status": "failed_stop", "checkpoint": item,
                                                            "reason": str(error), "completed": len(records)})
            raise
    with (output / "validation_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("epoch", "completed_epoch", "name", "dice", "assd_mm", "cldice", "n_cases", "seconds")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"epoch": row["epoch"], "completed_epoch": row["epoch"] + 1, "name": row["name"],
                             **{metric: row["overall"][metric]["mean"] for metric in ("dice", "assd_mm", "cldice")},
                             "n_cases": row["n_cases"], "seconds": row["seconds"]})
    atomic_json(output / "TRAJECTORY_COMPLETE.json", identity | {"n_evaluations": len(records),
                                                                "total_inference_wall_seconds": sum(row["seconds"] for row in records)})
    atomic_json(output / "trajectory_status.json", {"status": "complete", "completed": len(records), "planned": len(candidates)})
    print("INTERNAL_FOUR_TRAJECTORY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
