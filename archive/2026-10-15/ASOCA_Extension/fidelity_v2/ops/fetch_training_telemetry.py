"""Read-only SFTP capture of an active frozen training run; password via remote.py.

No remote files are created and no Torch/CUDA workload or controller is started.
Incomplete live trailing JSONL records are kept as bytes and excluded by analysis.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def capture(sftp, root, relative, output, tail=None):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Telemetry path must remain within its selected project")
    source = str(PurePosixPath(root) / rel)
    try:
        before = sftp.stat(source)
    except OSError:
        return {"relative_path": relative, "missing": True, "collected_at_utc": now()}
    limit = tail or 4 * 1024 * 1024
    start = max(0, before.st_size - limit)
    started = now()
    with sftp.open(source, "rb") as stream:
        if start:
            stream.seek(start)
        data = stream.read(min(before.st_size, limit))
    after = sftp.stat(source)
    target = output.joinpath(*rel.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"relative_path": relative, "source_path": source, "local_path": str(target.resolve()),
            "capture_started_at_utc": started, "capture_finished_at_utc": now(),
            "captured_bytes": len(data), "source_bytes_before": before.st_size, "source_bytes_after": after.st_size,
            "source_mtime_epoch_before": before.st_mtime, "source_mtime_epoch_after": after.st_mtime,
            "source_changed_during_capture": (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime),
            "tail_only": start > 0, "tail_start_byte": start,
            "ends_with_newline": data.endswith(b"\n"), "sha256": hashlib.sha256(data).hexdigest()}


def command_capture(client, command, target):
    stdin, stdout, stderr = client.exec_command(command, timeout=30)
    stdin.close()
    body = stdout.read(65536)
    error = stderr.read(2048)
    status = stdout.channel.recv_exit_status()
    target.write_bytes(body)
    return {"command": command, "exit_code": status, "capture_finished_at_utc": now(),
            "sha256": hashlib.sha256(body).hexdigest(), "stderr": error.decode("utf-8", errors="replace")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--training-relative", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh telemetry snapshot folder")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ops"))
    import remote
    client = remote.connect(args.port)
    sftp = None
    manifest = {"capture_started_at_utc": now(), "port": args.port, "remote_root": args.root,
                "training_relative": args.training_relative, "scope": "read-only SFTP and system status queries; no GPU work or remote writes",
                "jsonl_policy": "retain captured bytes; ignore incomplete final line when analysing live JSONL", "files": [], "status_queries": []}
    try:
        sftp = client.open_sftp()
        for name in ("config.json", "history.jsonl", "gradient_probes.jsonl", "TRAINING_COMPLETE.json"):
            manifest["files"].append(capture(sftp, args.root, args.training_relative.rstrip("/") + "/" + name, output))
        for relative in ("unet_calibration_status.json", "unet_calibration_protocol.json", "UNET_CALIBRATION_COMPLETE.json",
                         "GTA_STAGED_CUDA_SMOKE.json", "configs/main_source_freeze.json", "AFTER_RESTART_READY.json"):
            manifest["files"].append(capture(sftp, args.root, relative, output))
        for folder, suffix in (("stage_records", ".json"), ("logs", ".log")):
            try:
                names = sorted(sftp.listdir(str(PurePosixPath(args.root) / folder)))
            except OSError:
                continue
            for name in names:
                if name.endswith(suffix) and (folder == "stage_records" or any(term in name.lower() for term in ("gta", "main", "unet_calibration"))):
                    manifest["files"].append(capture(sftp, args.root, folder + "/" + name, output, tail=40_000 if folder == "logs" else None))
        manifest["status_queries"].append(command_capture(client, "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader", output / "gpu_processes.txt"))
        manifest["status_queries"].append(command_capture(client, "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader", output / "gpu_status.txt"))
        manifest["status_queries"].append(command_capture(client, "ps -eo pid,ppid,pgid,etimes,stat,comm", output / "process_status.txt"))
        manifest.update(status="captured", capture_finished_at_utc=now())
    except Exception as error:
        manifest.update(status="failed_or_partial", error=str(error), capture_finished_at_utc=now())
        raise
    finally:
        if sftp is not None:
            sftp.close()
        client.close()
        (output / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"port": args.port, "output": str(output), "captured_files": sum(not r.get("missing") for r in manifest["files"]),
                      "capture_finished_at_utc": manifest["capture_finished_at_utc"]}), flush=True)


if __name__ == "__main__":
    main()
