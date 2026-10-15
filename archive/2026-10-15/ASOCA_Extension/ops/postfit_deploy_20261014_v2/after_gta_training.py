"""Finish the existing GTA200 run with verified internal4 trajectory and curves.

No training launcher is called. The only child recipe evaluates immutable
weights on the same four internal patients and plots their observed history.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from verify_postfit import atomic_json, digest, preflight, postflight, read, require, source_guard

DEFAULT_ROOT = Path("/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014")
RAW_READY = Path("/root/autodl-fs/datasets/ASOCA_GTA_20261014/RAW_DIRECTORY_READY.json")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def owned_training_controller(stage, root):
    """Resolve relative command paths through /proc cwd and reject reused/zombie PIDs."""
    pid = stage.get("pid")
    require(isinstance(pid, int) and pid > 1, "Invalid training-controller PID")
    process = Path("/proc") / str(pid)
    require(process.is_dir(), "Training controller disappeared before verified completion")
    status = (process / "status").read_text()
    require(not any(line.startswith("State:") and "Z" in line.split()[1] for line in status.splitlines()),
            "Training controller is a zombie without completed stage; inspect")
    argv = [part.decode() for part in (process / "cmdline").read_bytes().split(b"\0") if part]
    cwd = (process / "cwd").resolve()
    target = (root / "ops/run_saved_stage.py").resolve()
    scripts = [(Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve() for arg in argv if arg.endswith("run_saved_stage.py")]
    require(target in scripts and "--recipe" in argv and argv[argv.index("--recipe") + 1] == "run_gta200.sh"
            and "--tag" in argv and argv[argv.index("--tag") + 1] == "gta200", "Training-controller PID identity differs")
    if "--root" in argv:
        require(Path(argv[argv.index("--root") + 1]).resolve() == root, "Training-controller project differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", action="store_true", help="Explicit continuation of identical postfit protocol")
    parser.add_argument("--plan", action="store_true", help="Read-only source/scope verification; no waiting or child")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "dataset/cache/manifest.json"
    raw_root = Path(read(RAW_READY)["raw_root"]).resolve()
    training = (root / "results/gta_enhanced200x125/training").resolve()
    status = root / "POSTFIT_STATUS.json"
    protocol_path = root / "POSTFIT_PROTOCOL.json"
    pre_path = root / "stage_records/postfit_preflight.json"
    receipt_path = root / "POSTFIT_VERIFIED.json"
    frozen = source_guard(root)
    protocol = {"schema_version": 2, "root": str(root), "training": str(training), "manifest": str(manifest),
                "raw_root": str(raw_root), "manifest_sha256": digest(manifest),
                "main_freeze_sha256": digest(root / "configs/main_source_freeze.json"),
                "source_sha256": frozen["training_source_sha256"],
                "controller_sha256": digest(Path(__file__)), "validator_sha256": digest(root / "ops/verify_postfit.py"),
                "stage_runner_sha256": digest(root / "ops/run_saved_stage.py"),
                "checkpoints": [f"epoch{epoch:03d}" for epoch in range(10, 200, 10)] + ["final"],
                "case_ids": ["Diseased_10", "Diseased_20", "Normal_12", "Normal_3"],
                "expected_evaluations": 20, "expected_case_evaluations": 80,
                "role": "internal-four method development; classifier accuracy diagnostic only",
                "new_training": False, "outer8_used_for_selection": False, "fivefold": False}
    require(protocol["manifest_sha256"] == "141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356", "Manifest differs")
    if args.plan:
        print(json.dumps(protocol, indent=2)); return
    import fcntl
    lock = (root / ".postfit.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child, began = None, False
    state = {"status": "waiting", "pid": os.getpid(), "role": protocol["role"], "started_at": now(),
             "protocol_sha256": None, "new_training": False, "outer8_used_for_selection": False}

    def save(phase, **values):
        state.update(phase=phase, updated_at=now(), **values)
        atomic_json(status, state)

    def guard():
        require(digest(Path(__file__)) == protocol["controller_sha256"]
                and digest(root / "ops/verify_postfit.py") == protocol["validator_sha256"]
                and digest(root / "ops/run_saved_stage.py") == protocol["stage_runner_sha256"]
                and digest(manifest) == protocol["manifest_sha256"]
                and digest(root / "configs/main_source_freeze.json") == protocol["main_freeze_sha256"],
                "Postfit controller/source/manifest identity changed")
        source_guard(root)

    def stop(signum, frame):
        raise KeyboardInterrupt("Signal " + str(signum))

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        if protocol_path.exists():
            require(args.resume and read(protocol_path) == protocol, "Postfit exists; exact continuation requires --resume")
        else:
            require(not status.exists() and not args.resume, "Existing/unversioned postfit state needs explicit inspection/archive")
            atomic_json(protocol_path, protocol)
        state["protocol_sha256"] = digest(protocol_path)
        began = True
        save("waiting_for_fixed_budget_training")
        while True:
            guard()
            stage = read(root / "stage_records/gta200.json")
            require(stage.get("recipe") == "run_gta200.sh" and stage.get("tag") == "gta200", "Formal stage identity differs")
            if stage.get("status") == "completed":
                require(stage.get("return_code") == 0 and stage.get("segmentation_updates") == 25000,
                        "Formal stage did not complete its fixed budget")
                break
            require(stage.get("status") not in ("failed", "interrupted"), "Formal segmentation failed/interrupted; inspect before postfit")
            try:
                owned_training_controller(stage, root)
            except (OSError, ValueError):
                # The producer can finish between our durable-state read and
                # /proc read. Its completed marker is audited on the next loop.
                require(read(root / "stage_records/gta200.json").get("status") == "completed",
                        "Training-controller identity disappeared without completed stage; inspect")
                continue
            time.sleep(30)
        # Missing periodic weights stop BEFORE inference, rather than accepting
        # the smaller inventory discovered by the frozen trajectory glob.
        save("auditing_twenty_immutable_checkpoints")
        proof = preflight(root, training, manifest, raw_root)
        if pre_path.exists():
            require(read(pre_path) == proof, "Previously verified checkpoint inventory changed")
        else:
            atomic_json(pre_path, proof)
        trajectory_stage = root / "stage_records/gta200_internal4.json"
        if receipt_path.exists():
            verified = postflight(root, training, manifest, raw_root, proof)
            require(read(receipt_path) == verified, "Postfit verified receipt changed")
        else:
            if trajectory_stage.exists() and read(trajectory_stage).get("status") == "completed":
                # A completed producer is never silently rerun to repair corrupt
                # provenance. It must already pass the independent post check.
                verified = postflight(root, training, manifest, raw_root, proof)
            else:
                require(not trajectory_stage.exists() or args.resume, "Partial trajectory needs explicit --resume")
                while True:
                    guard()
                    occupants = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
                    if not occupants:
                        break
                    save("waiting_for_idle_gpu", gpu_occupants=occupants); time.sleep(30)
                command = [sys.executable, "-u", str(root / "ops/run_saved_stage.py"), "--root", str(root),
                           "--recipe", "run_internal_trajectory.sh", "--tag", "gta200_internal4", "--training", str(training)]
                if trajectory_stage.exists():
                    command.append("--resume")
                save("internal_four_trajectory", status="running", command=command)
                (root / "logs").mkdir(exist_ok=True)
                with (root / "logs/gta_postfit_controller.log").open("a") as stream:
                    child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    save("internal_four_trajectory", child_pid=child.pid)
                    while child.poll() is None:
                        time.sleep(10)
                    code = child.returncode; child = None; state.pop("child_pid", None)
                require(code == 0, "Postfit evaluation or plotting failed; inspect stage/logs")
                stage = read(trajectory_stage)
                require(stage.get("status") == "completed" and stage.get("return_code") == 0, "Trajectory producer lacks completion")
                guard(); save("verifying_twenty_points_eighty_cases_and_curves")
                verified = postflight(root, training, manifest, raw_root, proof)
            atomic_json(receipt_path, verified)
        guard()
        save("development_curves_ready", status="completed", verified_receipt=str(receipt_path),
             verified_receipt_sha256=digest(receipt_path), verified_checkpoints=20, verified_case_evaluations=80,
             validation_history_sha256=verified["validation_history_sha256"], curve_manifest_sha256=verified["curve_manifest_sha256"],
             finite_curve_observations=verified["finite_curve_observations"],
             note="Dice/ASSD observations retain undefined distances; no performance threshold, new training or final-test claim")
    except BaseException as error:
        if child is not None and child.poll() is None:
            require(os.getpgid(child.pid) == child.pid, "Own child process group differs; refusing unrelated termination")
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        if began:
            state.pop("child_pid", None)
            save("stopped_for_inspection", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == "__main__":
    main()
