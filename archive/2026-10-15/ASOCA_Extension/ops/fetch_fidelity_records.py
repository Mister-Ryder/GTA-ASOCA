"""Fetch bounded experiment RECORDS using remote.connect(port), never weights/data.

Password input remains the existing interactive getpass/Paramiko connection.
The server creates one owned temporary archive below ROOT/records. Its SHA256
and every captured file are verified before safe local extraction. No Torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import tarfile
import tempfile
import uuid


DEFAULT_ROOT = "/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014"
OLD_ROOT = "/root/autodl-tmp/ASOCA_GTAirway_20261013"
MAX_BYTES = 128 * 1024 * 1024
TAIL_BYTES = 20_000

# This program runs in the server's standard-library Python. It reads a strict
# allowlist and writes only its uniquely named archive in project ROOT/records.
COLLECTOR = r"""
import datetime,hashlib,io,json,os,re,stat,tarfile,time
from pathlib import Path

root=Path(CFG['root'])
if not root.is_absolute() or '..' in root.parts or len(root.parts)<4 or not root.is_dir() or root.is_symlink():
    raise ValueError('Expected an existing absolute experiment root, not a symlink/root directory')
root=root.resolve();records=root/'records'
if records.is_symlink():raise ValueError('Records directory must not be a symlink')
records.mkdir(exist_ok=True)
if not records.resolve().is_relative_to(root):raise ValueError('Records path escaped the experiment root')
prefix=CFG['prefix'];limit=CFG['limit'];tail=CFG['tail'];selected={};missing=[];skipped=[]

def add(path,arc,base,log=False):
    path=Path(path);base=Path(base).resolve()
    if not path.exists():missing.append(arc);return
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base):
        skipped.append({'path':arc,'reason':'nonregular/symlink/outside selected base'});return
    if any(x in path.name.lower() for x in ('password','credential','private_key','id_rsa','id_ed25519')):
        skipped.append({'path':arc,'reason':'credential filename'});return
    if path.suffix.lower() not in ('.json','.jsonl','.csv','.txt','.log'):
        raise ValueError('Non-record suffix refused')
    key=prefix+'/'+arc
    if key in selected and selected[key][0]!=path:raise ValueError('Conflicting archive record names')
    selected[key]=(path,log)

def tree(relative,suffixes):
    base=root/relative
    if not base.exists():missing.append(relative);return
    for path in sorted(base.rglob('*')):
        if path.is_file() and path.suffix.lower() in suffixes:
            add(path,path.relative_to(root).as_posix(),base)

if CFG['mode']=='fidelity':
    singles=['ENV_READY.json','CHECKPOINT_CUDA_SMOKE.json','CUDA_READY.json','CUDA_SMOKE.json',
             'LOCAL_CACHE_READY.json','CACHE_READY.json','DATA_READY.json','RAW_DIRECTORY_READY.json',
             'LABEL_ROUNDTRIP_PID.json','MAIN_BACKBONE_TRANSFER_READY.json',
             'classifier_setup_status.json','bootstrap_pid.json','published_readiness.json',
             'baseline_source_provenance.json','classifier_stage/source_provenance.json',
             'prior_audit/source_provenance.json','initialization/initialization_sources.json',
             'configs/main_source_freeze.json','configs/first_stage.json','configs/classifier_quality_review_template.json',
             'manifests/environment_ready.txt','manifests/environment.lock.txt']
    for relative in singles:add(root/relative,relative,root)
    tree('prior_audit/results',{'.json','.csv','.txt'})
    tree('results/label_roundtrip_internal4',{'.json','.csv','.txt'})
    training_names={'config.json','history.jsonl','attribute_statistics.json','internal_split.json',
                    'final_selection_report.json','CLASSIFIER_TRAINING_COMPLETE.json',
                    'CLASSIFIER_TRAINING_COMPLETE_VERIFIED.json','outer_development_diagnostic.json'}
    for relative in ('results/classifier_smoke128/training','results/classifier80x64/training'):
        base=root/relative
        if not base.exists():missing.append(relative);continue
        for path in sorted(base.iterdir()):
            if path.name in training_names or path.suffix=='.json' and ('selection' in path.name.lower() or 'complete' in path.name.lower()):
                add(path,path.relative_to(root).as_posix(),base)
    logs=root/'logs'
    if logs.exists():
        for path in sorted(logs.rglob('*.log')):add(path,path.relative_to(root).as_posix(),logs,log=True)
    shared=Path('/root/autodl-fs/datasets/ASOCA_GTA_20261014')
    for name in ('publish_status.json','packed_publish_status.json','DATA_READY.json','CACHE_ARCHIVE_READY.json',
                 'RAW_ARCHIVE_READY.json','RAW_DIRECTORY_READY.json','raw_extract_status.json','raw_extract_pid.json'):
        add(shared/name,'shared_dataset/'+name,shared)
    for name in ('publish.log','packed_publish.log','raw_extract.log'):
        add(shared/name,'shared_dataset/'+name,shared,log=True)
elif CFG['mode']=='native':
    relative='results/nnunet/fold0';base=root/relative
    if not base.is_dir():raise ValueError('Requested native fold0 directory is missing')
    for path in sorted(base.glob('*COMPLETE*.json')):add(path,path.relative_to(root).as_posix(),base)
    training=base/'training'
    if training.is_dir():
        for path in sorted(training.glob('*.json')):
            if 'complete' in path.name.lower() or 'completion' in path.name.lower():add(path,path.relative_to(root).as_posix(),training)
    evaluation=base/'evaluation'
    if evaluation.is_dir():
        for path in sorted(evaluation.rglob('*.json')):add(path,path.relative_to(root).as_posix(),evaluation)
    add(base/'native_runner.log',relative+'/native_runner.log',base,log=True)
else:raise ValueError('Unknown records collection mode')

secret_keys=re.compile(r'^(?:password|passwd|pwd|api[_-]?key|token|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key|ssh[_-]?password|authorization|secret)$',re.I)
def clean_json(value):
    if isinstance(value,dict):return {k:('[REDACTED]' if secret_keys.fullmatch(str(k)) else clean_json(v)) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(x) for x in value]
    return value
def redact(data,suffix):
    text=data.decode('utf-8',errors='replace');original=text
    if suffix=='.json':
        try:
            value=json.loads(text);clean=clean_json(value)
            if clean!=value:text=json.dumps(clean,indent=2,ensure_ascii=False)+'\n'
        except json.JSONDecodeError:pass # live/incomplete records retain their captured state
    elif suffix=='.jsonl':
        lines=[]
        for line in text.splitlines(keepends=True):
            try:
                value=json.loads(line);clean=clean_json(value)
                lines.append(json.dumps(clean,ensure_ascii=False)+'\n' if clean!=value else line)
            except json.JSONDecodeError:lines.append(line)
        text=''.join(lines)
    text=re.sub(r'(?is)-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----','[REDACTED PRIVATE KEY]',text)
    text=re.sub(r'(?i)(authorization\s*[:=]\s*(?:bearer|basic)\s+)[A-Za-z0-9+/_.=-]+',r'\1[REDACTED]',text)
    text=re.sub(r'(?i)(https?://)[^/\s:@]+:[^/\s@]+@',r'\1[REDACTED]@',text)
    if suffix!='.json':
        text=re.sub(r'''(?i)((?:password|passwd|pwd|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)''',r'\1[REDACTED]',text)
    encoded=text.encode('utf-8')
    return encoded,encoded!=data

archive=records/(CFG['archive_name']);part=archive.with_suffix(archive.suffix+'.part')
if archive.exists() or part.exists():raise ValueError('Owned archive name unexpectedly already exists')
manifest={'mode':CFG['mode'],'root':str(root),'prefix':prefix,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'limit_bytes':limit,'logs_tail_bytes':tail,'scope':'allowlisted text records only; no weights/raw arrays/credentials',
          'missing_paths':sorted(set(missing)),'skipped':skipped,'files':[]}
total=0
try:
    with tarfile.open(part,'w:gz',compresslevel=6) as archive_stream:
        for name,(path,is_log) in sorted(selected.items()):
            before=path.stat()
            if not stat.S_ISREG(before.st_mode):raise ValueError('Record changed into nonregular file')
            count=min(before.st_size,tail) if is_log else before.st_size
            if count>limit or total+count>limit:raise ValueError('Selected records exceed 128 MiB limit; no oversized download')
            with path.open('rb') as stream:
                if is_log:stream.seek(max(0,before.st_size-tail))
                data=stream.read(count)
            captured_hash=hashlib.sha256(data).hexdigest();data,redacted=redact(data,path.suffix.lower());after=path.stat()
            total+=len(data)
            if total>limit:raise ValueError('Sanitized records exceed size limit')
            item=tarfile.TarInfo(name);item.size=len(data);item.mtime=int(time.time());item.mode=0o600
            archive_stream.addfile(item,io.BytesIO(data))
            manifest['files'].append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                  'captured_before_redaction_sha256':captured_hash,'redacted':redacted,'tail_only':is_log,
                  'source_bytes_at_capture':before.st_size,'source_changed_during_capture':(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)})
        manifest['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        data=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
        if total+len(data)>limit:raise ValueError('Record manifest exceeds uncompressed size limit')
        item=tarfile.TarInfo(prefix+'/record_collection_manifest.json');item.size=len(data);item.mtime=int(time.time());item.mode=0o600
        archive_stream.addfile(item,io.BytesIO(data))
    if part.stat().st_size>limit:raise ValueError('Compressed archive exceeds 128 MiB limit')
    part.replace(archive)
    h=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    print(json.dumps({'archive':str(archive),'sha256':h.hexdigest(),'size_bytes':archive.stat().st_size,'uncompressed_bytes':total+len(data),
                      'file_count':len(manifest['files'])+1,'missing_count':len(set(missing)),
                      'redacted_file_count':sum(x['redacted'] for x in manifest['files'])}),flush=True)
except Exception:
    if part.exists() and part.parent.resolve()==records.resolve():part.unlink()
    raise
"""


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_name(name):
    if "\\" in name or any(ord(char) < 32 for char in name):
        raise ValueError("Unsafe archive path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or any(":" in part or part.endswith((" ", ".")) for part in relative.parts):
        raise ValueError("Unsafe archive member path")
    devices = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if any(part.split(".")[0].upper() in devices for part in relative.parts):
        raise ValueError("Archive member is a Windows reserved device")
    return relative


def safe_extract(archive, output, prefix, expected_uncompressed_bytes=None):
    if archive.stat().st_size > MAX_BYTES:
        raise ValueError("Local archive exceeds 128 MiB")
    destination = output / prefix
    if destination.exists() or destination.is_symlink():
        raise ValueError("Use a fresh output directory; existing snapshot will not be overwritten")
    with tempfile.TemporaryDirectory(prefix=".records_extract_", dir=output) as temporary:
        stage = Path(temporary).resolve()
        if not stage.is_relative_to(output.resolve()):
            raise ValueError("Temporary extraction path escapes output")
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
            if len(members) > 4096 or any(item.size < 0 for item in members) or sum(item.size for item in members) > MAX_BYTES:
                raise ValueError("Archive exceeds member/uncompressed limit")
            names = set()
            for member in members:
                relative = safe_name(member.name)
                if not relative.parts or relative.parts[0] != prefix or not (member.isfile() or member.isdir()):
                    raise ValueError("Unexpected prefix or symlink/device/special archive member")
                if member.isfile() and (relative.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt", ".log"}
                        or any(text in relative.name.lower() for text in ("password", "credential", "private_key", "id_rsa", "id_ed25519"))):
                    raise ValueError("Archive contains an excluded raw/weight/credential record")
                identity = member.name.casefold() if os.name == "nt" else member.name
                if identity in names:
                    raise ValueError("Duplicate archive member")
                names.add(identity)
                target = stage.joinpath(*relative.parts).resolve()
                if not target.is_relative_to(stage):
                    raise ValueError("Archive path escaped temporary destination")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError("Missing archive record stream")
                with target.open("xb") as sink:
                    remaining = member.size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError("Truncated archive member")
                        sink.write(block); remaining -= len(block)
        manifest_path = stage / prefix / "record_collection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {entry["path"] for entry in manifest["files"]} | {prefix + "/record_collection_manifest.json"}
        regular = {member.name for member in members if member.isfile()}
        if expected != regular:
            raise ValueError("Archive records disagree with collection manifest")
        for entry in manifest["files"]:
            relative = safe_name(entry["path"])
            target = stage.joinpath(*relative.parts)
            if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
                raise ValueError("Captured record SHA256 mismatch")
        manifest["_verified_uncompressed_archive_bytes"] = sum(member.size for member in members)
        if expected_uncompressed_bytes is not None and manifest["_verified_uncompressed_archive_bytes"] != expected_uncompressed_bytes:
            raise ValueError("Archive's uncompressed bytes disagree with remote metadata")
        source = (stage / prefix).resolve()
        if not source.is_relative_to(output.resolve()) or not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError("Snapshot move would escape the explicitly selected output directory")
        source.rename(destination)
    return manifest


def select_remote_python(client, sftp, remote_root, explicit=None):
    """Prefer existing project/shared 3.11 environments; never use base Python3.8."""
    candidates = [explicit] if explicit else [str(PurePosixPath(remote_root) / "env/bin/python"),
                                             "/root/miniconda3/envs/gta-asoca311/bin/python"]
    failures = []
    probe = "import json,sys; print(json.dumps({'version':list(sys.version_info[:3]),'executable':sys.executable}))"
    for executable in candidates:
        try:
            if not stat.S_ISREG(sftp.stat(executable).st_mode):
                failures.append(executable + ": not a regular executable target")
                continue
        except OSError:
            failures.append(executable + ": unavailable")
            continue
        stdin, stdout, stderr = client.exec_command(shlex.quote(executable) + " -c " + shlex.quote(probe), timeout=30)
        stdin.close()
        response, error = stdout.read(4097), stderr.read(2048)
        code = stdout.channel.recv_exit_status()
        try:
            info = json.loads(response.decode("utf-8")) if code == 0 and len(response) <= 4096 else {}
            version = info.get("version", [])
            supported = len(version) == 3 and tuple(version[:2]) >= (3, 9)
        except (UnicodeDecodeError, ValueError, TypeError):
            supported = False
        if supported:
            return executable, info
        failures.append(executable + ": requires Python>=3.9; prefer the existing 3.11 environment")
    raise RuntimeError("No supported records runtime. Pass --remote-python/--native-python. " + "; ".join(failures))


def fetch_one(port, remote_root, output, mode, prefix, limit_bytes=MAX_BYTES, remote_python=None):
    import remote  # existing getpass/Paramiko helper; no credentials copied here
    name = f"{mode}_records_{uuid.uuid4().hex}.tar.gz"
    config = {"root": remote_root, "mode": mode, "prefix": prefix, "limit": limit_bytes,
              "tail": TAIL_BYTES, "archive_name": name}
    program = "CFG=" + repr(config) + "\n" + COLLECTOR
    client = remote.connect(port)
    sftp = None
    try:
        sftp = client.open_sftp()
        executable, runtime = select_remote_python(client, sftp, remote_root, remote_python)
        command = shlex.quote(executable) + " -c " + shlex.quote(program)
        stdin, stdout, stderr = client.exec_command(command, timeout=300)
        stdin.close()
        response = stdout.read(65537)
        error = stderr.read(8192).decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        if code or len(response) > 65536:
            raise RuntimeError(f"Remote record collection failed (exit {code}): {error[-2000:]}")
        metadata = json.loads(response.decode("utf-8"))
        expected_path = PurePosixPath(remote_root) / "records" / name
        if (PurePosixPath(metadata["archive"]) != expected_path or not 0 < metadata["size_bytes"] <= limit_bytes
                or not 0 < metadata["uncompressed_bytes"] <= limit_bytes):
            raise ValueError("Unexpected remote archive path/size")
        archives = output / "archives"; archives.mkdir(exist_ok=True)
        local = archives / name; partial = archives / (name + ".part")
        if sftp.stat(metadata["archive"]).st_size != metadata["size_bytes"]:
            raise ValueError("Remote archive size changed before download")
        sftp.get(metadata["archive"], str(partial))
        if partial.stat().st_size != metadata["size_bytes"] or digest(partial) != metadata["sha256"]:
            raise ValueError("Downloaded archive SHA256/size mismatch; partial file retained")
        partial.replace(local)
        manifest = safe_extract(local, output, prefix, metadata["uncompressed_bytes"])
        cleanup = True
        try:
            sftp.remove(metadata["archive"])  # exact uniquely owned, verified ROOT/records archive
        except OSError:
            cleanup = False
        return {"port": port, "remote_root": remote_root, "prefix": prefix, "archive": str(local.resolve()),
                "remote_python": executable,"remote_python_version": runtime["version"],
                "sha256": metadata["sha256"], "size_bytes": metadata["size_bytes"],
                "uncompressed_bytes": metadata["uncompressed_bytes"],
                "file_count": metadata["file_count"], "redacted_file_count": metadata["redacted_file_count"],
                "remote_temporary_archive_removed": cleanup,
                "live_files_changed_during_capture": sum(x["source_changed_during_capture"] for x in manifest["files"])}
    finally:
        if sftp is not None:
            sftp.close()
        client.close()


def absolute_project(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 4 or str(path) != value:
        raise argparse.ArgumentTypeError("Use a canonical absolute POSIX experiment root")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=absolute_project, default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=38773)
    parser.add_argument("--remote-python", type=absolute_project, help="Explicit existing remote Python>=3.9; prefer 3.11")
    parser.add_argument("--output", type=Path, required=True, help="Fresh local snapshot directory")
    parser.add_argument("--include-native", action="store_true", help="Also fetch old-server native fold0 records only")
    parser.add_argument("--native-root", type=absolute_project, default=OLD_ROOT)
    parser.add_argument("--native-port", type=int, default=17552)
    parser.add_argument("--native-python", type=absolute_project, help="Optional explicit old-server records runtime")
    args = parser.parse_args()
    if any(not 1 <= port <= 65535 for port in (args.port, args.native_port)):
        parser.error("Port must be between 1 and 65535")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "fidelity").exists() or args.include_native and (output / "native_reference").exists():
        parser.error("Choose a fresh snapshot output; existing records will not be replaced")
    receipt = {"scope": "bounded text records only; no checkpoints/raw data/credentials", "limit_bytes_total": MAX_BYTES,
               "logs_tail_bytes": TAIL_BYTES, "fetcher_sha256": digest(Path(__file__)), "status": "running", "sources": []}
    try:
        receipt["sources"].append(fetch_one(args.port, args.remote_root, output, "fidelity", "fidelity", remote_python=args.remote_python))
        if args.include_native:
            remaining = MAX_BYTES - max(sum(source["size_bytes"] for source in receipt["sources"]),
                                        sum(source["uncompressed_bytes"] for source in receipt["sources"]))
            if remaining <= 0:
                raise ValueError("No remaining 128 MiB budget for requested native records")
            receipt["sources"].append(fetch_one(args.native_port, args.native_root, output, "native", "native_reference", remaining, args.native_python))
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed_or_partial", error=str(error))
        raise
    finally:
        (output / "fetch_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": receipt["status"],
                      "sources": [{key: source[key] for key in ("port", "file_count", "size_bytes", "sha256")} for source in receipt["sources"]]}), flush=True)


if __name__ == "__main__":
    main()
