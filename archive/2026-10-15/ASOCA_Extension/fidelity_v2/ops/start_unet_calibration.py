"""One old-server U-Net calibration: smoke, fixed formal budget, internal4 only.

Uses the separately deployed frozen fidelity_v2 package and original immutable
cache. No original code/results, external methods, or fivefold queue are changed.
"""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

OLD_ROOT = Path("/root/autodl-tmp/ASOCA_GTAirway_20261013")
EXPECTED_MANIFEST = "141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gpu_idle():
    value = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if value:
        raise RuntimeError("GPU occupied; no calibration child was started: " + value)


@contextmanager
def locked(path):
    import fcntl
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another calibration controller holds this root's lock")
        stream.seek(0); stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "started_at": now(), "controller": str(Path(__file__).resolve())}) + "\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def training_valid(directory, expected, manifest_hash, source_hash, split):
    marker_path = directory / "TRAINING_COMPLETE.json"
    if not marker_path.exists():
        return False
    config, marker = read(directory / "config.json"), read(marker_path)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError("Completed training differs from saved recipe: " + key)
    final_hash = sha(directory / "final.pt")
    if (config.get("source_sha256") != source_hash or config.get("manifest_sha256") != manifest_hash
            or config.get("internal_split") != split or config.get("trained_case_ids") != split["train"]
            or config.get("internal_selection_valid") is not True or config.get("outer8_used_for_selection") is not False
            or marker.get("source_sha256") != source_hash or marker.get("manifest_sha256") != manifest_hash
            or marker.get("smoke") != expected["smoke"] or marker.get("epoch") != expected["epochs"] - 1
            or marker.get("trained_case_ids") != split["train"] or marker.get("final_sha256") != final_hash
            or marker.get("segmentation_updates") != expected["epochs"] * expected["steps"]
            or config.get("classifier_pretraining_updates") != 0 or config.get("init_audits") != []):
        raise ValueError("Completed U-Net has incompatible source/patient/budget/initialization identity")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines() if line.strip()]
    if [row["epoch"] for row in history] != list(range(expected["epochs"])) or any(row["iterations"] != expected["steps"] for row in history):
        raise ValueError("Training history does not demonstrate exactly the planned updates")
    return True


def trajectory_valid(directory, training, config, source_hash, manifest_hash, cases, code):
    complete = directory / "TRAJECTORY_COMPLETE.json"
    if not complete.exists():
        return False
    from evaluate_trajectory import verify_evaluation
    marker = read(complete)
    expected_names = [f"epoch{epoch:03d}" for epoch in range(10, 200, 10)] + ["final"]
    checkpoints = marker.get("checkpoints", [])
    sources = {name: sha(code / name) for name in ("evaluate.py", "metrics.py", "evaluate_trajectory.py")}
    if (marker.get("source_sha256") != source_hash or marker.get("manifest_sha256") != manifest_hash
            or marker.get("training_config_sha256") != sha(training / "config.json")
            or marker.get("evaluation_sources") != sources or marker.get("split") != "internal_diagnostic"
            or marker.get("smoke") is not False or marker.get("case_ids") != config["internal_split"]["val"]
            or marker.get("outer8_used_for_selection") is not False or marker.get("n_evaluations") != 20
            or [item["name"] for item in checkpoints] != expected_names):
        raise ValueError("Trajectory completion identity or20-checkpoint coverage differs")
    expected_records = []
    for item in checkpoints:
        checkpoint = training / "final.pt" if item["name"] == "final" else training / "checkpoints" / (item["name"] + ".pt")
        epoch = 199 if item["name"] == "final" else int(item["name"][5:]) - 1
        if item.get("path") != str(checkpoint) or item.get("epoch") != epoch or item.get("checkpoint_sha256") != sha(checkpoint):
            raise ValueError("Trajectory checkpoint changed or epoch differs")
        child_dir = directory / item["name"]
        summary = verify_evaluation(child_dir, marker["case_ids"], item["checkpoint_sha256"], manifest_hash,
                                    config, sources, cases, epoch)
        done = read(child_dir / "EVALUATION_COMPLETE.json")
        if (done.get("summary_sha256") != sha(child_dir / "summary.json")
                or done.get("checkpoint_sha256") != item["checkpoint_sha256"] or done.get("evaluation_sources") != sources
                or done.get("case_ids") != marker["case_ids"] or summary.get("overall", {}).get("n_cases") != 4):
            raise ValueError("Trajectory checkpoint evaluation completion differs")
        expected_records.append({**item, "evaluation_output": str(child_dir), "summary_path": str(child_dir / "summary.json"),
                                 "split": "internal_diagnostic", "role": summary["evaluation_role"], "n_cases": 4,
                                 "overall": summary["overall"], "seconds": done["wall_seconds"], "smoke": False,
                                 "checkpoint_selection": summary["checkpoint_selection"], "formal_final_test": False})
    rows = [json.loads(line) for line in (directory / "validation_history.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 20 or [row["name"] for row in rows] != expected_names or any(row["split"] != "internal_diagnostic" or row["n_cases"] != 4 for row in rows):
        raise ValueError("Trajectory observation ledger differs")
    for row, expected in zip(rows, expected_records):
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Trajectory ledger differs from its actual checkpoint/verified summary: " + expected["name"])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OLD_ROOT / "fidelity_v2")
    parser.add_argument("--manifest", type=Path, default=OLD_ROOT / "cache/manifest.json")
    parser.add_argument("--raw-root", type=Path, default=OLD_ROOT / "raw/CompressedVersion")
    parser.add_argument("--python", default=str(OLD_ROOT / "env/bin/python"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    root, manifest_path, raw_root = args.root.resolve(), args.manifest.resolve(), args.raw_root.resolve()
    code = root / "code"; outputs = (args.output_root or root / "results").resolve()
    if (outputs == root or outputs.is_relative_to(code)
            or outputs.is_relative_to((OLD_ROOT / "results").resolve())
            or outputs.is_relative_to(manifest_path.parent) or outputs.is_relative_to(raw_root)):
        raise ValueError("Calibration outputs require a separate results directory")
    subprocess.run([args.python, str(root / "configs/verify_main_source.py"), "--root", str(root)], check=True)
    frozen = read(root / "configs/main_source_freeze.json")
    manifest_hash = sha(manifest_path)
    if manifest_hash != EXPECTED_MANIFEST:
        raise ValueError("The immutable40-case patient split/cache manifest differs")
    sys.path.insert(0, str(code))
    from classification_data import internal_split
    from data import load_manifest
    manifest = load_manifest(manifest_path); split = internal_split(manifest, 0, 20261014)
    cases = {case["case_id"]: case for case in manifest["cases"]}
    for case_id in split["val"]:
        if sha(raw_root / cases[case_id]["label_relative_path"]) != cases[case_id]["source_sha256"]["label"]:
            raise ValueError("Original-grid reference differs: " + case_id)
    smoke, formal = outputs / "unet_calibration_smoke128/training", outputs / "unet_recipe200x125/training"
    trajectory = formal.parent / "internal4_trajectory"
    controller_hash = sha(Path(__file__))
    common = {"method": "unet", "fold": 0, "patch_size": 128, "batch_size": 2, "seed": 20261013,
              "split_seed": 20261014, "cohort": "internal28", "lr": .0003, "weight_decay": .0001,
              "backbone_warmup_epochs": 0, "seg_loss": "dice_ce", "sampling": "standard",
              "sampling_fov": "image-body", "no_ds": False, "device": "cuda"}
    specs = [("smoke", smoke, common | {"epochs": 2, "steps": 2, "workers": 2, "warmup_epochs": 0,
                                      "save_every": 1, "gradient_probe_every": 1, "smoke": True}),
             ("formal", formal, common | {"epochs": 200, "steps": 125, "workers": 4, "warmup_epochs": 5,
                                        "save_every": 10, "gradient_probe_every": 250, "smoke": False})]
    protocol = {"schema_version": 1, "controller_sha256": controller_hash,
                "main_freeze_manifest_sha256": sha(root / "configs/main_source_freeze.json"),
                "training_source_sha256": frozen["training_source_sha256"], "manifest_sha256": manifest_hash,
                "root": str(root), "output_root": str(outputs), "manifest": str(manifest_path), "raw_root": str(raw_root),
                "python": args.python, "internal_split": split,
                "stages": [{"name": name, "output": str(directory), "training_config": expected} for name, directory, expected in specs]
                         + [{"name": "internal4_trajectory", "output": str(trajectory), "evaluations": 20, "patients_per_evaluation": 4}],
                "role": "one common-recipe U-Net development calibration; no external-method reproduction",
                "outer8_used_for_selection": False, "fivefold": False, "classifier_pretraining_updates": 0,
                "formal_segmentation_updates": 25000, "smoke_updates": 4,
                "primary_metrics": ["dice", "assd_mm"], "auxiliary_metric": "cldice"}
    if args.plan:
        print(json.dumps(protocol, indent=2)); return
    outputs.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=True); (root / "logs").mkdir(exist_ok=True)
    status_path, protocol_path = root / "unet_calibration_status.json", root / "unet_calibration_protocol.json"
    state = {"status": "running", "pid": os.getpid(), "host": socket.gethostname(), "started_at": now(),
             "role": protocol["role"], "completed_stages": [], "outer8_used_for_selection": False}
    child = None

    def record(phase, **values):
        state.update(phase=phase, updated_at=now(), **values)
        atomic_json(status_path, state); print(json.dumps(state), flush=True)

    def stop(signum, frame):
        raise KeyboardInterrupt("Controller interrupted by signal" + str(signum))

    def guard():
        if (sha(Path(__file__)) != controller_hash or sha(manifest_path) != manifest_hash
                or sha(root / "configs/main_source_freeze.json") != protocol["main_freeze_manifest_sha256"]):
            raise RuntimeError("Controller/source manifest changed during this bounded run")
        subprocess.run([args.python, str(root / "configs/verify_main_source.py"), "--root", str(root)], check=True)

    def run(command, name, use_gpu=True):
        nonlocal child
        guard()
        if use_gpu:
            gpu_idle()
        logfile = root / "logs" / ("unet_calibration_" + name + ".log")
        env = os.environ.copy(); env.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
        with logfile.open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=code, env=env, start_new_session=True)
            record(name, child_pid=child.pid, command=command, logfile=str(logfile))
            while child.poll() is None:
                time.sleep(10)
            exit_code = child.returncode; child = None; state.pop("child_pid", None)
            if exit_code:
                raise RuntimeError(name + " failed with exit" + str(exit_code) + "; inspect" + str(logfile))

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with locked(root / ".unet_calibration.lock"), locked(outputs / ".unet_calibration.lock"):
        if protocol_path.exists():
            if read(protocol_path) != protocol:
                raise ValueError("Existing calibration protocol differs; use a new isolated root/output")
            if not args.resume:
                raise ValueError("Calibration already exists; exact continuation requires --resume")
        elif args.resume:
            raise ValueError("No previous calibration protocol exists to resume")
        atomic_json(protocol_path, protocol)
        try:
            ancestor = outputs
            while not ancestor.exists():
                ancestor = ancestor.parent
            if shutil.disk_usage(ancestor).free < 3 * 1024 ** 3:
                raise RuntimeError("Less than3 GiB free for independent U-Net checkpoints and trajectory outputs")
            for name, directory, expected in specs:
                guard()
                if not training_valid(directory, expected, manifest_hash, frozen["training_source_sha256"], split):
                    command = [args.python, "-u", str(code / "train.py"), "--manifest", str(manifest_path), "--output", str(directory)]
                    for key, value in expected.items():
                        if isinstance(value, bool):
                            if value:
                                command.append("--" + key.replace("_", "-"))
                        else:
                            command.extend(["--" + key.replace("_", "-"), str(value)])
                    if (directory / "config.json").exists():
                        if not args.resume or not (directory / "latest.pt").is_file():
                            raise RuntimeError("Partial stage requires explicit --resume and a valid checkpoint")
                        command.append("--resume")
                    run(command, name)
                    if not training_valid(directory, expected, manifest_hash, frozen["training_source_sha256"], split):
                        raise RuntimeError("Training returned zero without verified completion")
                state["completed_stages"].append(name); record(name + "_verified")
            config = read(formal / "config.json")
            guard()
            if not trajectory_valid(trajectory, formal, config, frozen["training_source_sha256"], manifest_hash, cases, code):
                run([args.python, "-u", str(code / "evaluate_trajectory.py"), "--training", str(formal),
                     "--manifest", str(manifest_path), "--raw-root", str(raw_root), "--output", str(trajectory),
                     "--sw-batch-size", "1"], "internal4_trajectory")
                if not trajectory_valid(trajectory, formal, config, frozen["training_source_sha256"], manifest_hash, cases, code):
                    raise RuntimeError("Trajectory returned zero without verified completion")
            state["completed_stages"].append("internal4_trajectory"); record("internal4_trajectory_verified")
            completion = protocol | {"completed_at": now(), "hostname": socket.gethostname(),
                        "formal_final_sha256": sha(formal / "final.pt"), "trajectory_complete_sha256": sha(trajectory / "TRAJECTORY_COMPLETE.json"),
                        "validation_history_sha256": sha(trajectory / "validation_history.jsonl"),
                        "formal_training_marker_sha256": sha(formal / "TRAINING_COMPLETE.json")}
            final_marker = root / "UNET_CALIBRATION_COMPLETE.json"
            if final_marker.exists():
                old = read(final_marker)
                if any(old.get(key) != value for key, value in completion.items() if key not in ("completed_at", "hostname")):
                    raise ValueError("Calibration completion provenance changed")
            else:
                atomic_json(final_marker, completion)
            state["status"] = "complete"; record("complete")
        except BaseException as error:
            if child is not None and child.poll() is None:
                if os.getpgid(child.pid) != child.pid:
                    raise RuntimeError("Child process-group ownership differs; refusing unrelated process termination") from error
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
            state["status"] = "failed_stop"; state.pop("child_pid", None)
            record("stopped_for_inspection", error=repr(error))
            raise


if __name__ == "__main__":
    main()
