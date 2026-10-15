"""Finish the already-running native fold, then run four author-method controls.

This controller imports the frozen main queue's verification/launch helpers but
never invokes its five-fold main(). The eight fold-0 validation cases are already
being used for method development and selection; they are not an independent test.

Server usage, after stopping the old parent and the clDice recovery controller:
  env/bin/python development/coordinator.py --root PROJECT --raw-root RAW \
      --manifest CACHE/manifest.json --old-parent-pid 6398 --native-runner-pid 54628 --plan
  env/bin/python -u development/coordinator.py --root PROJECT --raw-root RAW \
      --manifest CACHE/manifest.json --old-parent-pid 6398 --native-runner-pid 54628

--plan is read-only. Failures stop the reduced queue. Restart skips only verified
completion markers; --resume-after-inspection explicitly permits retrying a failed
author-method job without changing its original source, settings or checkpoint rule.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE))
import run_queue as q

ORDER = ("gta_no_ds", "gta_no_prior", "gta_no_gaussian", "gta_no_balance")
ROLE = "method-development validation used for selection; not an independent test"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def original_file(folder: Path, exact: str, pattern: str) -> Path:
    path = folder / exact
    if path.is_file():
        return path
    candidates = sorted(folder.glob(pattern))
    if len(candidates) != 1:
        raise ValueError(f"Need exactly one original scope record matching {pattern} under {folder}")
    return candidates[0]


def proc(pid: int) -> dict | None:
    """Linux identity, including zombie exit status; existence alone is not liveness."""
    folder = Path("/proc") / str(pid)
    try:
        raw = (folder / "stat").read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(")") + 2:].split()
    value = {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "starttime": fields[19]}
    if value["state"] == "Z" and len(fields) >= 50:
        value["return_code"] = os.waitstatus_to_exitcode(int(fields[49]))
    try:
        value["argv"] = [v.decode(errors="replace") for v in (folder / "cmdline").read_bytes().split(b"\0") if v]
        value["cwd"] = str((folder / "cwd").resolve(strict=True))
    except (OSError, FileNotFoundError):
        value["argv"], value["cwd"] = [], None
    return value


def argv_script(argv: list[str], cwd: str | None, script: Path) -> bool:
    if not cwd:
        return False
    for argument in argv[1:]:
        if argument.endswith(".py"):
            candidate = Path(argument)
            candidate = candidate if candidate.is_absolute() else Path(cwd) / candidate
            return candidate.resolve() == script.resolve()
    return False


def option_matches(argv: list[str], option: str, expected: Path, cwd: str | None) -> bool:
    try:
        candidate = Path(argv[argv.index(option) + 1])
    except (ValueError, IndexError):
        return False
    if not candidate.is_absolute():
        if not cwd:
            return False
        candidate = Path(cwd) / candidate
    return candidate.resolve() == expected.resolve()


def verify_parent(process: dict | None, root: Path, pid: int, prior: dict | None = None) -> None:
    if process is None:
        return
    if process["pid"] != pid or process["state"] not in ("T", "t", "Z"):
        raise RuntimeError("The old queue parent must remain stopped; never resume it during this handoff")
    if prior and any(process[key] != prior[key] for key in ("pid", "ppid", "starttime")):
        raise RuntimeError("The old parent process identity or parent PID changed")
    if process["state"] != "Z" and (not argv_script(process["argv"], process["cwd"], root / "code/run_queue.py")
                                      or not option_matches(process["argv"], "--root", root, process["cwd"])):
        raise RuntimeError("Old PID is not the stopped run_queue.py belonging to this project")


def native_identity(process: dict | None, root: Path, parent_pid: int, native_job: dict,
                    prior: dict | None = None) -> dict | None:
    if process is None:
        return None
    if prior and process["starttime"] != prior["starttime"]:
        raise RuntimeError("Native runner PID has been reused")
    if process["ppid"] != parent_pid:
        raise RuntimeError("Native runner is no longer the old queue parent's direct child")
    if process["state"] == "Z":
        # A stopped parent cannot reap its finished child. cmdline/cwd disappear
        # for zombies; the original queue attempt and direct PPid prove ownership.
        attempts = native_job.get("attempts", [])
        command = attempts[-1].get("command", []) if attempts else []
        if native_job.get("child_pid") != process["pid"] or not argv_script(command, str(root / "code"), root / "code/nnunet_runner.py"):
            raise RuntimeError("Zombie native runner has no matching original queue ownership record")
    elif not argv_script(process["argv"], process["cwd"], root / "code/nnunet_runner.py") \
            or not option_matches(process["argv"], "--root", root, process["cwd"]):
        raise RuntimeError("Native PID does not belong to this project's nnunet_runner.py")
    return process


def counts(state: dict) -> None:
    keys = ("pending", "running", "completed", "failed", "deferred")
    state["counts"] = {key: sum(j.get("status") == key for j in state["jobs"]) for key in keys}
    state["updated_at"] = q.timestamp()


def guards(code_root: Path, hashes: dict, manifest_path: Path, manifest_hash: str,
           scope_path: Path, scope_hash: str) -> None:
    if q.code_hashes(code_root) != hashes or q.sha256_file(manifest_path) != manifest_hash \
            or q.sha256_file(scope_path) != scope_hash:
        raise RuntimeError("Frozen main source, manifest or recorded user scope changed; queue stopped")


def mark_complete(job: dict, base_id: str, manifest: dict, manifest_hash: str,
                  development_id: str) -> dict:
    if q.verify_existing_job(job, base_id, manifest, manifest_hash):
        return q.verify_evaluation(job, manifest, manifest_hash)
    verified = q.verify_evaluation(job, manifest, manifest_hash)
    q.atomic_json(Path(job["folder"]) / "JOB_COMPLETE.json",
                  {"schema_version": 1, "status": "completed", "method_label": job["method_label"],
                   "training_method": job["training_method"], "fold": job["fold"], "smoke": False,
                   "protocol_identity": base_id, "job_identity": job["job_identity"],
                   "manifest_sha256": manifest_hash, "completed_at": q.timestamp(),
                   "execution_stage": "method_development", "development_protocol_identity": development_id,
                   "validation_role": ROLE, **verified})
    return verified


def verify_native_finished(job: dict, manifest: dict, manifest_hash: str) -> None:
    verified = q.verify_evaluation(job, manifest, manifest_hash)
    marker = read_json(Path(job["folder"]) / "EXPERIMENT_COMPLETE.json")
    expected = {"method": "nnunet", "fold": 0, "smoke": False,
                "manifest_sha256": manifest_hash, "checkpoint_sha256": verified["checkpoint_sha256"],
                "evaluation_complete": True}
    if any(marker.get(key) != value for key, value in expected.items()) \
            or sorted(marker.get("heldout_case_ids", [])) != verified["case_ids"]:
        raise ValueError("Native runner's final experiment marker does not match all eight verified cases")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--old-parent-pid", type=int, default=6398)
    parser.add_argument("--native-runner-pid", type=int, default=54628)
    parser.add_argument("--poll", type=int, default=30)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume-after-inspection", action="store_true")
    args = parser.parse_args()
    root, raw_root, manifest_path = args.root.resolve(), args.raw_root.resolve(), args.manifest.resolve()
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    if not 1 <= args.poll <= 60 or "," in args.gpu:
        raise ValueError("Choose one GPU and a polling interval of 1..60 seconds")
    if root / "code" != CODE or not raw_root.is_dir():
        raise ValueError("Use this project's development/coordinator.py with its original raw dataset")
    development = root / "development"
    original_path = original_file(development, "previous_queue_status.json", "previous_queue_status*.json")
    scope_path = original_file(development, "scope_change.json", "scope_change*.json")
    original = read_json(original_path)
    base_id, protocol = original["protocol_identity"], original["protocol"]
    if q.identity(protocol) != base_id or protocol.get("gpu") != args.gpu:
        raise ValueError("Original protocol identity/GPU is incompatible")
    manifest, manifest_hash = q.load_protocol(manifest_path)
    hashes = protocol["source_hashes"]
    if protocol["manifest_sha256"] != manifest_hash:
        raise ValueError("Frozen manifest differs from the original protocol")
    scope_hash = q.sha256_file(scope_path)
    guards(CODE, hashes, manifest_path, manifest_hash, scope_path, scope_hash)
    main_path = root / "results/queue_status.json"
    main_state = read_json(main_path)
    if main_state.get("protocol_identity") != base_id or main_state.get("pid") != args.old_parent_pid:
        raise ValueError("Original queue status identifies a different parent/protocol")
    original_jobs = {j["job_id"]: j for j in original["jobs"]}
    planned = {j["job_id"]: j for j in q.build_jobs(root)}
    for job_id, job in original_jobs.items():
        expected_id = q.identity({"protocol_identity": base_id, "job_id": job_id,
                                  "config": q.expected_config(job, manifest_hash)})
        if job_id not in planned or job.get("job_identity") != expected_id \
                or Path(job["folder"]).resolve() != Path(planned[job_id]["folder"]).resolve():
            raise ValueError("Original job identity/folder is incompatible: " + job_id)
    jobs = [dict(original_jobs[f"{method}/fold0"]) for method in ORDER]
    native = dict(original_jobs["nnunet/fold0"])
    deferred = [j["job_id"] for j in original["jobs"] if j["fold"] > 0 or j["method_label"] == "unet_no_ds"]
    deferred.append("unet_cldice/fold0/retry")
    stage_protocol = {"schema_version": 1, "stage": "scope-reduced method development",
                      "base_protocol_identity": base_id, "manifest_sha256": manifest_hash,
                      "main_source_hashes": hashes, "scope_record": str(scope_path), "scope_sha256": scope_hash,
                      "coordinator_sha256": q.sha256_file(Path(__file__)), "gpu": args.gpu,
                      "adopt_existing_only": "nnunet/fold0", "author_method_order": list(ORDER),
                      "new_jobs": [j["job_id"] for j in jobs], "deferred": deferred,
                      "epochs": 200, "steps_per_epoch": 125, "optimizer_updates": 25000,
                      "checkpoint_selection": "fixed final", "validation_case_ids": manifest["folds"][0]["val"],
                      "validation_role": ROLE, "fivefold_complete_claim_allowed": False,
                      "no_balance_ablation": "combined foreground/geometry sampling and anatomical loss reweighting removal",
                      "external_methods": "retain existing comparisons; no new external-method training or retries",
                      "failure_policy": "stop on first failure for inspection"}
    development_id = q.identity(stage_protocol)
    status_path = root / "results/development_queue_status.json"
    prior_state = read_json(status_path) if status_path.is_file() else None
    if prior_state and prior_state.get("protocol_identity") != development_id:
        raise ValueError("Existing development queue has a different identity")
    parent_now = proc(args.old_parent_pid) if os.name == "posix" else None
    old_parent = (prior_state or {}).get("old_parent", {}).get("identity") or parent_now
    verify_parent(parent_now, root, args.old_parent_pid, old_parent)
    native_now = proc(args.native_runner_pid) if os.name == "posix" else None
    native_prior = (prior_state or {}).get("adopted_native", {}).get("identity")
    if not (prior_state or {}).get("adopted_native", {}).get("completed"):
        native_identity(native_now, root, args.old_parent_pid, native, native_prior)
    if args.plan:
        print(json.dumps({"development_protocol_identity": development_id, "protocol": stage_protocol,
                          "old_parent_observed": parent_now, "native_observed": native_now,
                          "historical_cldice": original_jobs["unet_cldice/fold0"].get("status"),
                          "read_only": True}, indent=2), flush=True)
        return 0
    if os.name != "posix":
        raise RuntimeError("Process handoff is supported only on the Linux experiment server")
    state = prior_state or {"schema_version": 1, "protocol_identity": development_id, "protocol": stage_protocol,
                            "started_at": q.timestamp(), "jobs": [{**j, "status": "pending"} for j in jobs],
                            "adopted_native": {"job_id": native["job_id"], "identity": native_now, "completed": False},
                            "old_parent": {"identity": old_parent, "retired": False},
                            "deferred": deferred, "historical_cldice": original_jobs["unet_cldice/fold0"],
                            "validation_role": ROLE}
    state.update(status="running", pid=os.getpid(), host=socket.gethostname())
    snapshot = root / "results/protocol_snapshot" / base_id
    for relative, digest in hashes.items():
        if q.sha256_file(snapshot / "code" / relative) != digest:
            raise ValueError("Original source snapshot is missing or incompatible: " + relative)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8")
    active = None
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Development coordinator interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with ExitStack() as stack:
        stack.enter_context(q.QueueLock(development / ".coordinator.lock"))
        protocol_path = development / "development_protocol.json"
        if protocol_path.is_file() and read_json(protocol_path) != stage_protocol:
            raise ValueError("Stored development protocol differs")
        q.atomic_json(protocol_path, stage_protocol)
        counts(state)
        q.atomic_json(status_path, state)
        try:
            if not state["adopted_native"].get("completed"):
                state["phase"] = "waiting_for_existing_native_runner"
                while True:
                    guards(CODE, hashes, manifest_path, manifest_hash, scope_path, scope_hash)
                    verify_parent(proc(args.old_parent_pid), root, args.old_parent_pid, old_parent)
                    observed = native_identity(proc(args.native_runner_pid), root, args.old_parent_pid, native,
                                               state["adopted_native"].get("identity"))
                    state["adopted_native"]["last_observed"] = observed
                    counts(state)
                    q.atomic_json(status_path, state)
                    if observed is None or observed["state"] == "Z":
                        if observed and observed.get("return_code") != 0:
                            raise RuntimeError("Existing native runner exited unsuccessfully; no external-method retry is permitted")
                        break
                    time.sleep(args.poll)
                verify_native_finished(native, manifest, manifest_hash)
                verified = mark_complete(native, base_id, manifest, manifest_hash, development_id)
                state["adopted_native"].update(completed=True, completed_at=q.timestamp(), **verified)
                main_state = read_json(main_path)
                status = next(j for j in main_state["jobs"] if j["job_id"] == native["job_id"])
                status.update(status="completed", phase="adopted_verified_completion", completed_at=q.timestamp(), **verified)
                status.pop("child_pid", None)
                if status.get("attempts"):
                    status["attempts"][-1].update(ended_at=q.timestamp(), adopted_by_development_coordinator=True,
                                                return_code=observed.get("return_code") if observed else None,
                                                exit_observation="zombie exit status" if observed else "process gone; artifacts strictly verified")
                main_state["status"] = "scope_reduction_native_verified"
                counts(main_state)
                q.atomic_json(main_path, main_state)
                q.atomic_json(status_path, state)
            else:
                if not q.verify_existing_job(native, base_id, manifest, manifest_hash):
                    raise ValueError("Previously adopted native completion marker is missing; stop for inspection")
                verify_native_finished(native, manifest, manifest_hash)
            if not state["old_parent"].get("retired"):
                observed_parent = proc(args.old_parent_pid)
                verify_parent(observed_parent, root, args.old_parent_pid, old_parent)
                if observed_parent and observed_parent["state"] != "Z":
                    # Kill this verified stopped process only. Never signal its
                    # process group: the adopted native runner is not terminated.
                    os.kill(args.old_parent_pid, signal.SIGKILL)
                deadline = time.monotonic() + 30
                while proc(args.old_parent_pid) is not None and time.monotonic() < deadline:
                    time.sleep(.2)
                remaining = proc(args.old_parent_pid)
                if remaining and remaining["state"] != "Z":
                    raise RuntimeError("Stopped old parent did not retire")
                main_lock = root / "results/.queue.lock"
                if remaining and remaining["state"] == "Z" and main_lock.is_file():
                    lock = read_json(main_lock)
                    if lock.get("pid") != args.old_parent_pid or lock.get("host") != socket.gethostname():
                        raise RuntimeError("Cannot transfer an unknown old queue lock")
                    # A verified zombie cannot execute another job. Its stale
                    # lock is safe to reclaim even though os.kill(pid,0) succeeds.
                    main_lock.unlink()
                state["old_parent"].update(retired=True, retired_at=q.timestamp(),
                                           signal="SIGKILL sent only to verified old parent PID",
                                           final_observation=remaining)
                q.atomic_json(status_path, state)
            stack.enter_context(q.QueueLock(root / "results/.queue.lock"))
            main_state = read_json(main_path)
            for original_job in main_state["jobs"]:
                if original_job["fold"] > 0 or original_job["method_label"] == "unet_no_ds":
                    original_job.setdefault("pre_scope_status", original_job.get("status"))
                    original_job.update(status="deferred", phase="deferred_by_user", deferred_at=q.timestamp(),
                                        reason="method not yet settled; five-fold/external-method work deferred")
                if original_job["job_id"] == "unet_cldice/fold0":
                    original_job["retry_deferred_by_user"] = True
            main_state.update(status="scope_reduced_development", development_protocol_identity=development_id,
                              validation_role=ROLE, deferred_jobs=deferred)
            counts(main_state)
            q.atomic_json(main_path, main_state)
            for active in state["jobs"]:
                guards(CODE, hashes, manifest_path, manifest_hash, scope_path, scope_hash)
                if q.verify_existing_job(active, base_id, manifest, manifest_hash):
                    active.update(status="completed", phase="verified_existing", **q.verify_evaluation(active, manifest, manifest_hash))
                else:
                    if active.get("status") == "failed" and not args.resume_after_inspection:
                        raise RuntimeError("Previously failed author-method job requires inspection before retry")
                    if active.get("child_pid"):
                        old_child = proc(active["child_pid"])
                        if old_child and old_child["state"] != "Z":
                            raise RuntimeError("A previous coordinator's child still runs; do not launch a duplicate")
                    folder = Path(active["folder"])
                    folder.mkdir(parents=True, exist_ok=True)
                    expected = q.expected_config(active, manifest_hash)
                    complete, config = q.training_complete(active, expected)
                    latest = folder / "training/latest.pt"
                    if config is None and latest.exists() or config is not None and not complete and not latest.exists():
                        raise ValueError("Partial author-method output lacks compatible config/checkpoint provenance")
                    run_path = folder / "run_snapshot.json"
                    if run_path.is_file():
                        if read_json(run_path).get("job_identity") != active["job_identity"]:
                            raise ValueError("Existing author-method snapshot has different identity")
                    else:
                        q.atomic_json(run_path, {"protocol_identity": base_id, "job_identity": active["job_identity"],
                                                 "job": active, "expected_training_config": expected,
                                                 "source_snapshot": str(snapshot), "source_hashes": hashes,
                                                 "manifest_sha256": manifest_hash, "created_at": q.timestamp(),
                                                 "execution_stage": "method_development", "validation_role": ROLE,
                                                 "development_protocol_identity": development_id,
                                                 "adopted_pre_queue_training": config is not None})
                    active.update(status="running", phase="starting", started_at=q.timestamp())
                    counts(state)
                    q.atomic_json(status_path, state)
                    train, evaluate = q.custom_commands(active, CODE, manifest_path, raw_root, resume=config is not None)
                    if not complete:
                        q.wait_for_idle_gpu(args.gpu, active, state, status_path, args.poll)
                        guards(CODE, hashes, manifest_path, manifest_hash, scope_path, scope_hash)
                        active["phase"] = "training"
                        q.run_process(train, folder / "training.log", env, active, state, status_path)
                        complete, _ = q.training_complete(active, expected)
                        if not complete:
                            raise ValueError("Training did not produce its compatible fixed final checkpoint")
                    q.wait_for_idle_gpu(args.gpu, active, state, status_path, args.poll)
                    guards(CODE, hashes, manifest_path, manifest_hash, scope_path, scope_hash)
                    active["phase"] = "evaluation"
                    q.run_process(evaluate, folder / "evaluation.log", env, active, state, status_path)
                    active.update(status="completed", phase="completed", completed_at=q.timestamp(),
                                  **mark_complete(active, base_id, manifest, manifest_hash, development_id))
                active.pop("error", None)
                counts(state)
                q.atomic_json(status_path, state)
                matching = next(j for j in main_state["jobs"] if j["job_id"] == active["job_id"])
                matching.update(active)
                counts(main_state)
                q.atomic_json(main_path, main_state)
            state.update(status="completed", phase="method_development_completed", completed_at=q.timestamp())
            state.pop("error", None)
            main_state["status"] = "scope_reduced_development_completed"
        except KeyboardInterrupt as error:
            if active and active.get("status") == "running":
                active.update(status="pending", phase="interrupted", error=str(error))
            state.update(status="interrupted", error=str(error))
        except Exception as error:
            if active and active.get("status") != "completed":
                active.update(status="failed", phase="stopped_for_inspection", error=f"{type(error).__name__}: {error}")
            state.update(status="failed", phase="stopped_for_inspection", error=f"{type(error).__name__}: {error}")
        finally:
            counts(state)
            state["ended_at"] = q.timestamp()
            q.atomic_json(status_path, state)
            if state["old_parent"].get("retired"):
                main_state = read_json(main_path)
                main_state["status"] = "scope_reduced_development_completed" if state["status"] == "completed" else "scope_reduced_development_stopped"
                if active:
                    next(j for j in main_state["jobs"] if j["job_id"] == active["job_id"]).update(active)
                counts(main_state)
                q.atomic_json(main_path, main_state)
            try:
                summary = q.aggregate_outoffold(root, manifest_path, output=root / "results/development_comparison")
                q.atomic_json(root / "results/development_comparison/development_context.json",
                              {"stage": "method_development", "validation_role": ROLE,
                               "development_protocol_identity": development_id, "base_protocol_identity": base_id,
                               "n_validation_cases": 8, "fivefold_complete_claim_allowed": False,
                               "deferred": deferred, "source": "frozen main aggregator; only verified JOB_COMPLETE rows",
                               "methods": {k: {"n_cases": v["n_cases"], "n_expected": v["n_expected"], "status": v["status"]}
                                           for k, v in summary["methods"].items()}})
                state.pop("aggregation_error", None)
                q.atomic_json(status_path, state)
            except Exception as error:
                state["aggregation_error"] = str(error)
                q.atomic_json(status_path, state)
    print(json.dumps({"status": state["status"], "phase": state.get("phase"), "counts": state["counts"],
                      "validation_role": ROLE, "status_file": str(status_path)}), flush=True)
    return 0 if state["status"] == "completed" and not state.get("aggregation_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
