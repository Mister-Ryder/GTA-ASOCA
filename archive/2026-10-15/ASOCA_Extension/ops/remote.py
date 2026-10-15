"""Session-only SSH password; never saved in experiment files."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 17552
REMOTE = "/root/autodl-tmp/ASOCA_GTAirway_20261013"


def connect(port=PORT):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=port, username="root", password=getpass.getpass("SSH password: "), timeout=30)
    client.get_transport().set_keepalive(30)
    return client


def run(client, command):
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    for line in stdout:
        print(line, end="", flush=True)
    code = stdout.channel.recv_exit_status()
    if code:
        raise SystemExit(code)


def upload(client, source, destination):
    run(client, "mkdir -p " + shlex.quote(str(Path(destination).parent).replace("\\", "/")))
    sftp = client.open_sftp()
    part = destination + ".part"
    size = os.path.getsize(source)
    try:
        offset = sftp.stat(part).st_size
    except FileNotFoundError:
        offset = 0
    if offset > size:
        raise RuntimeError("Partial remote file is larger than source")
    start, report, initial = time.time(), time.time(), offset
    print(json.dumps({"source": str(source), "destination": destination, "bytes": size, "resume_offset": offset}), flush=True)
    with open(source, "rb") as src, sftp.open(part, "ab" if offset else "wb") as dst:
        src.seek(offset)
        dst.set_pipelined(True)
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            offset += len(chunk)
            now = time.time()
            if now - report >= 15:
                speed = (offset - initial) / max(now - start, 1)
                print(json.dumps({"percent": round(100 * offset / size, 2), "MiB_per_s": round(speed / 2**20, 2), "remaining_minutes": round((size-offset)/max(speed,1)/60, 1)}), flush=True)
                report = now
    print("Upload written; computing source SHA256", flush=True)
    digest = hashlib.sha256()
    with open(source, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    local_sha = digest.hexdigest()
    _, stdout, _ = client.exec_command("sha256sum " + shlex.quote(part))
    remote_sha = stdout.read().decode().split()[0]
    if local_sha != remote_sha:
        raise RuntimeError("SHA256 mismatch; partial upload retained")
    try:
        sftp.posix_rename(part, destination)
    except IOError:
        sftp.rename(part, destination)
    manifest = {"source": str(source), "destination": destination, "bytes": size, "sha256": local_sha, "verified": True}
    Path(__file__).parent.joinpath("last_transfer.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(__file__).parent.joinpath(Path(destination).name + ".transfer.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    sftp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "upload", "exec", "sync", "download", "snapshot"])
    parser.add_argument("--source")
    parser.add_argument("--destination")
    parser.add_argument("--command")
    parser.add_argument("--command-file")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    client = connect(args.port)
    try:
        if args.mode == "inspect":
            run(client, "nvidia-smi; df -h /root/autodl-tmp /; free -h; ps -eo pid,comm,%cpu,%mem,args --sort=-%cpu | head -18; python -c \"import sys,torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())\"; ls -d /root/miniconda3/envs/* 2>/dev/null")
        elif args.mode == "upload":
            upload(client, Path(args.source), args.destination or REMOTE + "/raw/CompressedVersion.zip")
        elif args.mode == "exec":
            command = Path(args.command_file).read_text(encoding="utf-8") if args.command_file else args.command
            run(client, command)
        elif args.mode == "download":
            target = Path(args.destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(args.source, str(target))
            print(json.dumps({"downloaded": args.source, "local": str(target)}))
        elif args.mode == "snapshot":
            command = """/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
import tarfile
root=Path('/root/autodl-tmp/ASOCA_GTAirway_20261013')
patterns=['manifests/*','logs/*.log','cache/manifest.json','results/queue_status.json','results/development_queue_status.json','development/*.json','development/*.log','repair_cldice/repair_status.json','results/*/fold*/JOB_COMPLETE.json','results/*/fold*/training/config.json','results/*/fold*/training/history.jsonl','results/comparison/*.txt','results/comparison/*.json','results/comparison/*.csv','results/smoke96_pinned/*/config.json','results/smoke96_pinned/*/history.jsonl']
with tarfile.open(root/'session_records.tar.gz','w:gz') as tar:
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file(): tar.add(path,arcname=str(path.relative_to(root)))
print('Records packaged')
PY"""
            run(client, command)
            target = Path(args.destination or str(Path(__file__).parent.parent / "server_records" / "session_records.tar.gz"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(REMOTE + "/session_records.tar.gz", str(target))
            import tarfile
            with tarfile.open(target) as tar:
                tar.extractall(target.parent, filter="data")
            print(json.dumps({"records_saved": str(target.parent)}))
        elif args.mode == "sync":
            import tarfile
            source = Path(args.source).resolve()
            archive = source.parent / (source.name + "_code.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                for path in source.rglob("*"):
                    if path.is_file() and not any(x in path.parts for x in [".git", "__pycache__", ".pytest_cache"]) and path.suffix not in [".pt", ".pth"]:
                        tar.add(path, arcname=str(path.relative_to(source)))
            dest = args.destination or REMOTE + "/code"
            upload(client, archive, dest + ".tar.gz")
            run(client, "mkdir -p " + shlex.quote(dest) + " && tar xzf " + shlex.quote(dest + ".tar.gz") + " -C " + shlex.quote(dest))
            if args.command_file or args.command:
                command = Path(args.command_file).read_text(encoding="utf-8") if args.command_file else args.command
                run(client, command)
    finally:
        client.close()


if __name__ == "__main__":
    main()
