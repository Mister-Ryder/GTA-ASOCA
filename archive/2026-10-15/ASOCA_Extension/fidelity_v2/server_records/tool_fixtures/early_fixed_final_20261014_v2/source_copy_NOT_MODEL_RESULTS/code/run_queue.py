"""Sequential single-GPU ASOCA training/evaluation queue with durable provenance.

Five frozen folds of GTA, U-Net, U-Net+clDice, Swin UNETR and native nnU-Net are
followed by fold-0 component/DS controls. Failures are isolated without replacing
the method, changing the budget or promoting smoke runs into the comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time

from aggregate_results import (MAIN_METHODS, ABLATION_METHODS, aggregate_outoffold,
                               atomic_json, load_protocol, validate_case_row)

CUSTOM_EPOCHS = 200
CUSTOM_STEPS = 125
NATIVE_EPOCHS = 100
NATIVE_STEPS = 250


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_hashes(code_root: Path) -> dict[str, str]:
    return {path.relative_to(code_root).as_posix(): sha256_file(path)
            for path in sorted(code_root.rglob("*.py")) if "__pycache__" not in path.parts}


def dependencies() -> dict:
    versions = {}
    for name in ("torch", "monai", "numpy", "scipy", "scikit-image", "SimpleITK", "nnunetv2", "dynamic-network-architectures"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": versions}


def build_jobs(root: Path) -> list[dict]:
    jobs = []
    for fold in range(5):
        for method in MAIN_METHODS:
            jobs.append({"method_label": method, "training_method": method, "fold": fold,
                         "deep_supervision": True, "kind": "native" if method == "nnunet" else "custom",
                         "comparison_type": "fivefold_cv"})
    for label in ABLATION_METHODS:
        method = {"gta_no_ds": "gta", "unet_no_ds": "unet"}.get(label, label)
        jobs.append({"method_label": label, "training_method": method, "fold": 0,
                     "deep_supervision": not label.endswith("_no_ds"), "kind": "custom",
                     "comparison_type": "fold0_ablation"})
    for job in jobs:
        job["job_id"] = f"{job['method_label']}/fold{job['fold']}"
        job["folder"] = str(root / "results" / job["method_label"] / f"fold{job['fold']}")
    return jobs


def expected_config(job: dict, manifest_hash: str) -> dict:
    if job["kind"] == "native":
        return {"method": "nnunet", "fold": job["fold"], "epochs": NATIVE_EPOCHS,
                "manifest_sha256": manifest_hash}
    return {"method": job["training_method"], "fold": job["fold"], "epochs": CUSTOM_EPOCHS,
            "steps": CUSTOM_STEPS, "patch_size": 96, "batch_size": 2, "workers": 4,
            "seed": 20261013, "lr": 2e-4, "no_ds": not job["deep_supervision"],
            "smoke": False, "manifest_sha256": manifest_hash}


def validate_training_config(folder: Path, expected: dict) -> dict | None:
    path = folder / "training" / "config.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Existing training configuration differs at {key}: {config.get(key)!r} != {value!r}")
    if config.get("smoke", False) or config.get("publication_result") is not True:
        raise ValueError("Existing output is not explicitly a publication run")
    return config


def final_checkpoint(job: dict, config: dict | None) -> Path:
    folder = Path(job["folder"])
    if job["kind"] == "custom":
        return folder / "training" / "final.pt"
    if not config or not config.get("checkpoint_path"):
        raise ValueError("Native nnU-Net config lacks its original final checkpoint path")
    return Path(config["checkpoint_path"])


def training_complete(job: dict, expected: dict) -> tuple[bool, dict | None]:
    folder = Path(job["folder"])
    config = validate_training_config(folder, expected)
    marker = folder / "training" / "TRAINING_COMPLETE.json"
    if not marker.exists():
        return False, config
    if config is None:
        raise ValueError("Training-completion marker exists without its configuration")
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if completion.get("smoke", False):
        raise ValueError("Smoke training cannot be resumed as a publication experiment")
    if job["kind"] == "custom" and completion.get("epoch") != CUSTOM_EPOCHS - 1:
        raise ValueError("Custom training did not reach the fixed final epoch")
    if job["kind"] == "native" and completion.get("completed_optimizer_updates") != NATIVE_EPOCHS * NATIVE_STEPS:
        raise ValueError("Native training did not verify its fixed 25000 optimizer updates")
    checkpoint = final_checkpoint(job, config)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError("Training marker exists but its final checkpoint is missing")
    return True, config


def verify_evaluation(job: dict, manifest: dict, manifest_hash: str) -> dict:
    folder = Path(job["folder"])
    complete, config = training_complete(job, expected_config(job, manifest_hash))
    if not complete:
        raise ValueError("Evaluation has no compatible completed training run")
    checkpoint = final_checkpoint(job, config)
    checkpoint_hash = sha256_file(checkpoint)
    expected_ids = sorted(manifest["folds"][job["fold"]]["val"])
    cases = {case["case_id"]: case for case in manifest["cases"]}
    evaluation = folder / "evaluation"
    summary_path = evaluation / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Evaluation did not produce summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("smoke") is not False or summary.get("fold") != job["fold"]:
        raise ValueError("Evaluation summary has incompatible smoke or fold metadata")
    if summary.get("method") != job["training_method"] or summary.get("overall", {}).get("n_cases") != len(expected_ids):
        raise ValueError("Evaluation summary does not describe the complete held-out fold")
    for case_id in expected_ids:
        path = evaluation / (case_id + ".json")
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_case_row(row, cases[case_id], job["fold"], job["method_label"], manifest_hash, checkpoint_hash)
        prediction = evaluation / (case_id + "_prediction.nrrd")
        if not prediction.is_file() or prediction.stat().st_size == 0:
            raise ValueError("Missing original-grid prediction: " + case_id)
    observed_ids = {path.stem for path in evaluation.glob("*.json") if path.stem in cases}
    if observed_ids != set(expected_ids):
        raise ValueError("Evaluation folder contains predictions from another fold")
    return {"case_ids": expected_ids, "checkpoint_sha256": checkpoint_hash,
            "checkpoint_path": str(checkpoint), "n_cases": len(expected_ids)}


def verify_existing_job(job: dict, protocol_id: str, manifest: dict, manifest_hash: str) -> bool:
    marker_path = Path(job["folder"]) / "JOB_COMPLETE.json"
    if not marker_path.exists():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {"protocol_identity": protocol_id, "job_identity": job["job_identity"],
                "method_label": job["method_label"], "fold": job["fold"], "manifest_sha256": manifest_hash,
                "status": "completed", "smoke": False}
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing completion marker belongs to an incompatible experiment")
    verified = verify_evaluation(job, manifest, manifest_hash)
    if marker.get("checkpoint_sha256") != verified["checkpoint_sha256"] or sorted(marker.get("case_ids", [])) != verified["case_ids"]:
        raise ValueError("Completed job files changed after verification")
    return True


class QueueLock:
    def __init__(self, path: Path):
        self.path = path
        self.owned = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                prior = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                raise RuntimeError("Malformed queue lock; inspect it before restarting") from error
            if prior.get("host") != socket.gethostname():
                raise RuntimeError("Queue lock belongs to another host")
            # os.kill(pid, 0) is a read-only liveness probe on POSIX, but can
            # terminate a process on Windows. Never use it there.
            if os.name != "posix":
                raise RuntimeError("Inspect an existing queue lock before restarting on this platform")
            try:
                os.kill(int(prior["pid"]), 0)
            except ProcessLookupError:
                self.path.unlink()
            except PermissionError as error:
                raise RuntimeError("Cannot determine whether the previous queue still runs") from error
            else:
                raise RuntimeError("Another queue process is already running")
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_at": timestamp()}, stream)
        self.owned = True
        return self

    def __exit__(self, *_):
        if self.owned:
            self.path.unlink(missing_ok=True)


def gpu_processes(gpu: str) -> list[dict]:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is required to verify exclusive GPU use")
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader", "-i", gpu]
    check = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if check.returncode:
        raise RuntimeError("Cannot inspect GPU usage: " + check.stderr.strip())
    rows = []
    for line in check.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(",", 1)
        if parts[0].strip().isdigit():
            rows.append({"pid": int(parts[0].strip()), "process_name": parts[1].strip() if len(parts) > 1 else "unknown"})
        elif "no running processes" not in line.lower():
            raise RuntimeError("GPU process visibility is unavailable: " + line.strip())
    return rows


def wait_for_idle_gpu(gpu: str, status: dict, state: dict, status_path: Path, poll: int) -> None:
    while True:
        occupied = gpu_processes(gpu)
        if not occupied:
            return
        status.update(phase="waiting_for_gpu", gpu_occupants=occupied, updated_at=timestamp())
        state["updated_at"] = timestamp()
        atomic_json(status_path, state)
        print(json.dumps({"job": status["job_id"], "phase": "waiting_for_gpu", "occupants": occupied}), flush=True)
        time.sleep(poll)


def run_process(command: list[str], log_path: Path, env: dict, status: dict,
                state: dict, status_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = {"started_at": timestamp(), "command": command, "log": str(log_path)}
    status.setdefault("attempts", []).append(attempt)
    atomic_json(status_path, state)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nQUEUE COMMAND " + json.dumps(attempt) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(Path(__file__).resolve().parent), env=env,
                                   start_new_session=os.name == "posix")
        status["child_pid"] = process.pid
        atomic_json(status_path, state)
        try:
            while True:
                try:
                    return_code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    status["updated_at"] = timestamp()
                    state["updated_at"] = timestamp()
                    atomic_json(status_path, state)
        except BaseException:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            attempt.update(ended_at=timestamp(), return_code=process.returncode, interrupted=True)
            status.pop("child_pid", None)
            atomic_json(status_path, state)
            raise
        attempt.update(ended_at=timestamp(), return_code=return_code)
        status.pop("child_pid", None)
        atomic_json(status_path, state)
        if return_code:
            raise RuntimeError(f"Child command exited with status {return_code}; inspect {log_path}")


def custom_commands(job: dict, code_root: Path, manifest_path: Path, raw_root: Path, resume: bool) -> tuple[list[str], list[str]]:
    folder = Path(job["folder"])
    train = [sys.executable, "-u", str(code_root / "train.py"), "--manifest", str(manifest_path),
             "--output", str(folder / "training"), "--method", job["training_method"], "--fold", str(job["fold"]),
             "--epochs", str(CUSTOM_EPOCHS), "--steps", str(CUSTOM_STEPS), "--patch-size", "96", "--batch-size", "2", "--workers", "4",
             "--seed", "20261013", "--lr", "0.0002"]
    if not job["deep_supervision"]:
        train.append("--no-ds")
    if resume:
        train.append("--resume")
    evaluate = [sys.executable, "-u", str(code_root / "evaluate.py"), "--manifest", str(manifest_path),
                "--raw-root", str(raw_root), "--checkpoint", str(folder / "training" / "final.pt"),
                "--output", str(folder / "evaluation"), "--fold", str(job["fold"])]
    return train, evaluate


def update_counts(state: dict) -> None:
    state["counts"] = {key: sum(job["status"] == key for job in state["jobs"]) for key in ("pending", "running", "completed", "failed")}
    state["updated_at"] = timestamp()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Project root containing code/ and results/")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu", default="0", help="One physical GPU index or UUID; queue waits for it to be idle")
    parser.add_argument("--gpu-idle-poll", type=int, default=30)
    parser.add_argument("--retain-native-preprocessed", action="store_true",
                        help="Retain each native fold's derived preprocessing cache after verified evaluation")
    parser.add_argument("--plan", "--dry-run", dest="plan", action="store_true", help="Print the frozen job plan without launching or modifying results")
    args = parser.parse_args()
    if "," in args.gpu or args.gpu_idle_poll < 1 or args.gpu_idle_poll > 60:
        raise ValueError("Choose one GPU and a polling interval between 1 and 60 seconds")
    root, raw_root = Path(args.root).resolve(), Path(args.raw_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    manifest, manifest_hash = load_protocol(manifest_path)
    code_root = root / "code"
    if code_root.resolve() != Path(__file__).resolve().parent:
        raise ValueError("Run the queue script belonging to the supplied project root")
    hashes = code_hashes(code_root)
    protocol = {"schema_version": 1, "manifest_sha256": manifest_hash, "source_hashes": hashes,
                "main_methods": list(MAIN_METHODS), "ablation_methods": list(ABLATION_METHODS), "folds": 5,
                "custom_epochs": CUSTOM_EPOCHS, "custom_steps_per_epoch": CUSTOM_STEPS,
                "native_epochs": NATIVE_EPOCHS, "native_steps_per_epoch": NATIVE_STEPS,
                "optimizer_updates_per_run": 25000, "custom_patch_size": 96, "custom_batch_size": 2,
                "custom_workers": 4, "custom_seed": 20261013, "custom_lr": 2e-4,
                "checkpoint_selection": "fixed final", "gpu": args.gpu,
                "comparability": "Same frozen patients and 25000 optimizer updates. Native nnU-Net has its own plans, FOV, optimizer and possible batch size; wall time is not matched."}
    protocol_id = identity(protocol)
    jobs = build_jobs(root)
    for job in jobs:
        job["job_identity"] = identity({"protocol_identity": protocol_id, "job_id": job["job_id"], "config": expected_config(job, manifest_hash)})
    if args.plan:
        print(json.dumps({"protocol_identity": protocol_id, "protocol": protocol, "jobs": jobs}, indent=2), flush=True)
        return 0
    for required in ("train.py", "evaluate.py", "nnunet_runner.py", "aggregate_results.py"):
        if not (code_root / required).is_file():
            raise FileNotFoundError("Required queue component is missing: " + required)
    if not raw_root.is_dir():
        raise FileNotFoundError("Raw ASOCA directory is missing")
    results_root = root / "results"
    status_path = results_root / "queue_status.json"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8")
    state = {"schema_version": 1, "protocol_identity": protocol_id, "protocol": protocol,
             "status": "running", "started_at": timestamp(), "updated_at": timestamp(),
             "pid": os.getpid(), "host": socket.gethostname(), "jobs": [{**job, "status": "pending"} for job in jobs],
             "native_cleanup_after_verified_evaluation": not args.retain_native_preprocessed,
             "notes": ["Single GPU; commands run sequentially. Compatible completed results are reverified before skipping.",
                       "Failures are recorded and unrelated jobs continue. No fallback architecture or shorter training budget is substituted.",
                       "Compatible pre-queue checkpoints may be adopted; snapshot records when source provenance begins."]}
    with QueueLock(results_root / ".queue.lock"):
        if status_path.exists():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("protocol_identity") != protocol_id:
                raise ValueError("Existing queue uses different code/data/settings; use a separate project/results directory")
            old = {job["job_id"]: job for job in previous.get("jobs", [])}
            for status in state["jobs"]:
                prior = old.get(status["job_id"], {})
                status["attempts"] = prior.get("attempts", [])
                status["prior_status"] = prior.get("status")
            state["original_started_at"] = previous.get("original_started_at", previous.get("started_at"))
        snapshot = results_root / "protocol_snapshot" / protocol_id
        for relative, digest in hashes.items():
            source = code_root / relative
            target = snapshot / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256_file(target) != digest:
                raise ValueError("Stored source snapshot is inconsistent")
            if not target.exists():
                shutil.copy2(source, target)
        atomic_json(snapshot / "protocol.json", protocol)
        atomic_json(snapshot / "environment.json", dependencies())
        if not (snapshot / "manifest.json").exists():
            shutil.copy2(manifest_path, snapshot / "manifest.json")
        update_counts(state)
        atomic_json(status_path, state)
        active = None
        def stop_queue(signum, _frame):
            raise KeyboardInterrupt(f"Queue interrupted by signal {signum}")
        signal.signal(signal.SIGTERM, stop_queue)
        try:
            for status in state["jobs"]:
                active = status
                # Prevent source/data edits during a long queue from changing the next model silently.
                if code_hashes(code_root) != hashes or sha256_file(manifest_path) != manifest_hash:
                    raise RuntimeError("Source code or frozen manifest changed while the queue was running")
                try:
                    if verify_existing_job(status, protocol_id, manifest, manifest_hash):
                        status.update(status="completed", phase="verified_existing", completed_at=timestamp())
                        update_counts(state)
                        atomic_json(status_path, state)
                        try:
                            aggregate_outoffold(root, manifest_path)
                            status.pop("aggregation_error", None)
                        except Exception as error:
                            status["aggregation_error"] = str(error)
                            atomic_json(status_path, state)
                        print(json.dumps({"job": status["job_id"], "status": "completed", "verified_existing": True}), flush=True)
                        continue
                    folder = Path(status["folder"])
                    folder.mkdir(parents=True, exist_ok=True)
                    expected = expected_config(status, manifest_hash)
                    complete, config = training_complete(status, expected)
                    run_snapshot = {"protocol_identity": protocol_id, "job_identity": status["job_identity"],
                                    "job": status, "expected_training_config": expected, "source_snapshot": str(snapshot),
                                    "source_hashes": hashes, "manifest_sha256": manifest_hash, "created_at": timestamp(),
                                    "adopted_pre_queue_training": config is not None,
                                    "pre_queue_source_provenance": "not retroactively verified; current source snapshot begins at queue adoption" if config else "queue-controlled from launch"}
                    snapshot_path = folder / "run_snapshot.json"
                    if snapshot_path.exists():
                        prior_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                        if prior_snapshot.get("job_identity") != status["job_identity"]:
                            raise ValueError("Existing job snapshot has incompatible identity")
                    else:
                        atomic_json(snapshot_path, run_snapshot)
                    status.update(status="running", phase="starting", started_at=timestamp())
                    update_counts(state)
                    atomic_json(status_path, state)
                    if status["kind"] == "native":
                        free_bytes = shutil.disk_usage(root).free
                        status["native_start_free_bytes"] = free_bytes
                        if free_bytes < 10 * 1024 ** 3:
                            raise RuntimeError("Native preprocessing requires at least 10 GiB free disk space before startup")
                        wait_for_idle_gpu(args.gpu, status, state, status_path, args.gpu_idle_poll)
                        status["phase"] = "native_training_and_evaluation"
                        command = [sys.executable, "-u", str(code_root / "nnunet_runner.py"), "--root", str(root),
                                   "--raw-root", str(raw_root), "--manifest", str(manifest_path), "--fold", str(status["fold"]),
                                   "--epochs", str(NATIVE_EPOCHS)]
                        if not args.retain_native_preprocessed:
                            command.append("--cleanup-preprocessed")
                        run_process(command, folder / "native_runner.log", env, status, state, status_path)
                    else:
                        latest = folder / "training" / "latest.pt"
                        if config is None and latest.exists():
                            raise ValueError("A checkpoint without a verified training config cannot be resumed")
                        if config is not None and not complete and not latest.exists():
                            raise ValueError("Partial custom output lacks latest.pt; inspect before choosing a new output directory")
                        train, evaluate = custom_commands(status, code_root, manifest_path, raw_root, resume=config is not None)
                        if not complete:
                            wait_for_idle_gpu(args.gpu, status, state, status_path, args.gpu_idle_poll)
                            status["phase"] = "training"
                            run_process(train, folder / "training.log", env, status, state, status_path)
                            complete, _ = training_complete(status, expected)
                            if not complete:
                                raise ValueError("Training returned success without its completed final checkpoint")
                        wait_for_idle_gpu(args.gpu, status, state, status_path, args.gpu_idle_poll)
                        status["phase"] = "evaluation"
                        run_process(evaluate, folder / "evaluation.log", env, status, state, status_path)
                    status["phase"] = "verifying"
                    verified = verify_evaluation(status, manifest, manifest_hash)
                    marker = {"schema_version": 1, "status": "completed", "method_label": status["method_label"],
                              "training_method": status["training_method"], "fold": status["fold"], "smoke": False,
                              "protocol_identity": protocol_id, "job_identity": status["job_identity"],
                              "manifest_sha256": manifest_hash, "completed_at": timestamp(), **verified}
                    atomic_json(folder / "JOB_COMPLETE.json", marker)
                    status.update(status="completed", phase="completed", completed_at=timestamp(), **verified)
                    status.pop("error", None)
                    status.pop("gpu_occupants", None)
                    update_counts(state)
                    atomic_json(status_path, state)
                    try:
                        aggregate_outoffold(root, manifest_path)
                        status.pop("aggregation_error", None)
                    except Exception as error:
                        status["aggregation_error"] = str(error)
                        atomic_json(status_path, state)
                    print(json.dumps({"job": status["job_id"], "status": status["status"], "n_cases": verified["n_cases"]}), flush=True)
                except Exception as error:
                    status.update(status="failed", failed_at=timestamp(), error=f"{type(error).__name__}: {error}")
                    update_counts(state)
                    atomic_json(status_path, state)
                    print(json.dumps({"job": status["job_id"], "status": "failed", "error": status["error"]}), flush=True)
            state["status"] = "completed" if all(job["status"] == "completed" for job in state["jobs"]) else "finished_with_failures"
        except KeyboardInterrupt as error:
            if active and active["status"] == "running":
                active.update(status="pending", phase="interrupted", error=str(error))
            state["status"] = "interrupted"
            print(json.dumps({"status": "interrupted", "reason": str(error)}), flush=True)
        except Exception as error:
            if active and active["status"] != "completed":
                active.update(status="failed", phase="queue_preflight_failed", error=str(error))
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(json.dumps({"status": "failed", "error": state["error"]}), flush=True)
        finally:
            update_counts(state)
            state["ended_at"] = timestamp()
            atomic_json(status_path, state)
            try:
                aggregate_outoffold(root, manifest_path)
            except Exception as error:
                state["aggregation_error"] = str(error)
                atomic_json(status_path, state)
    return 0 if state["status"] == "completed" and not state.get("aggregation_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
