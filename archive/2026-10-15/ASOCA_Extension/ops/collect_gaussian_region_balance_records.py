"""Read-only SFTP collector; no remote commands, writes, GPU or model imports.

Immutable receipts are added once and must retain the same SHA on every retry.
Mutable log/status observations go into a fresh timestamped capture. Passwords
are interactive by default; --password-stdin reads NEW then OLD, one line each.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import uuid

import remote

NEW = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
OLD = '/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2'
DONORS = '/root/autodl-fs/experiments/GTA_ASOCA_Fidelity_20261014/gaussian_control_donors_20261014'
RESULT = 'results/gaussian_region_balance_final200_shadow_20261014'
ARTIFACTS = {'analytic_fixture.json', 'fixed_sampling_bank.json', 'diagnostic_plan.json', 'patch_metrics.jsonl', 'summary.json'}
DONOR_FILES = {'airway_backbone_epoch599.pkl': {'bytes': 43313082, 'sha256': '29b6977b74e547735717fda0785a9d532397df364e1f641bed0e6682db0384d3'},
               'classifier80_final.pt': {'bytes': 49148853, 'sha256': '308ebc33d55766458ebd3241c8422dbadb9259b929570bc5f03afd51fbec31a4'}}
DEFAULT = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records/gaussian_region_balance_actual_20261014'
MAX_BYTES = 128 * 1024**2


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def json_bytes(value): return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def save_fresh(base, relative, data):
    """Publish a new local file atomically without ever replacing a destination."""
    path = base / relative
    require(path.resolve().is_relative_to(base.resolve()), 'Local capture path escaped output')
    expected = digest(data)
    if path.exists():
        require(path.is_file() and file_sha(path) == expected and path.stat().st_size == len(data),
                'Existing captured bytes differ; never overwrite: ' + str(path))
        return 'existing_sha_verified'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.incoming.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        require(file_sha(temporary) == expected, 'Local incoming bytes differ')
        try: os.link(temporary, path)  # No-replace publication on Windows/Linux.
        except FileExistsError:
            require(file_sha(path) == expected, 'Concurrent destination differs; never overwrite')
    finally:
        if temporary.exists(): temporary.unlink()  # Only this unique owned file.
    require(file_sha(path) == expected, 'Published local bytes differ')
    return 'added'


def read_remote(sftp, path, tail=None, optional=True):
    try: before = sftp.lstat(path)
    except FileNotFoundError:
        if optional: return None
        raise
    require(stat.S_ISREG(before.st_mode), 'Remote record is not a regular file: ' + path)
    count = min(before.st_size, tail) if tail else before.st_size
    require(count <= MAX_BYTES and count >= 0, 'Record exceeds128MiB bound')
    with sftp.open(path, 'rb') as stream:
        if tail: stream.seek(max(0, before.st_size - tail))
        data = stream.read(count)
    after = sftp.stat(path)
    require(len(data) == count, 'Short remote read: ' + path)
    changed = (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime)
    if not tail: require(not changed, 'Immutable remote record changed while capturing: ' + path)
    return data, {'remote_path': path, 'remote_bytes_at_start': before.st_size, 'captured_bytes': len(data),
                  'sha256': digest(data), 'remote_mtime_at_start': before.st_mtime, 'changed_during_capture': changed,
                  'tail_bytes_requested': tail}


def client_for(port, password_stdin):
    if not password_stdin: return remote.connect(port)
    import paramiko
    print('SSH password stdin for port ' + str(port) + ':', file=sys.stderr, flush=True)
    password = sys.stdin.readline()
    require(bool(password), 'Missing stdin password for port ' + str(port))
    client = paramiko.SSHClient(); client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(remote.HOST, port=port, username='root', password=password.rstrip('\r\n'), timeout=30)
    finally: password = None
    client.get_transport().set_keepalive(30)
    return client


def collect_host(base, capture, tag, port, password_stdin, include_weights):
    records, missing = [], []
    client = client_for(port, password_stdin)
    try:
        with client.open_sftp() as sftp:
            def fetch(path, relative, tail=None, optional=True, expected=None, mutable=False):
                read = read_remote(sftp, path, tail, optional)
                if read is None:
                    missing.append(path); return None
                data, entry = read
                if expected is not None: require(digest(data) == expected, 'Remote artifact differs from completed SHA: ' + path)
                location = ('captures/' + capture + '/' + tag + '/' if mutable else 'records/' + tag + '/') + relative
                entry['local_relative_path'] = location
                entry['action'] = save_fresh(base, location, data)
                records.append(entry)
                return data

            donor_bytes = fetch(DONORS + '/DONORS_READY.json', 'shared_donors/DONORS_READY.json', optional=False)
            ready = json.loads(donor_bytes)
            require(ready.get('schema_version') == 1 and ready.get('role') == 'PRIVATE_IMMUTABLE_GTA_CONTROL_DONORS'
                    and ready.get('main_config_sha256') == 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
                    and set(ready.get('files', {})) == set(DONOR_FILES) and ready.get('weights_public') is False
                    and ready.get('model_forward') is False and ready.get('CUDA_context') is False, 'Actual donor READY identity differs')
            for name, expected in DONOR_FILES.items():
                require(all(ready['files'][name].get(k) == v for k, v in expected.items()), 'Actual donor record differs: ' + name)
                info = sftp.stat(DONORS + '/' + name)
                require(info.st_size == expected['bytes'], 'Actual shared donor size differs: ' + name)
                if include_weights:
                    fetch(DONORS + '/' + name, 'shared_donors/' + name, optional=False, expected=expected['sha256'])
            state = {'donor_READY_sha256': digest(donor_bytes), 'donor_sizes_checked': True,
                     'donor_weight_bytes_downloaded_and_SHA_checked': include_weights,
                     'remote_read_only': True, 'shadow_completion': 'not_applicable_on_NEW'}
            if tag.startswith('old'):
                for relative in ('GAUSSIAN_REGION_BALANCE_SHADOW_LAUNCH_20261014.json', 'SHADOW_PLAN_EXECUTED_20261014.json',
                                 'SHADOW_RESOURCE_PREFLIGHT_20261014.json', 'results/gaussian_region_balance_torch_fixture_20261014.json'):
                    fetch(OLD + '/' + relative, relative)
                for relative in ('GAUSSIAN_REGION_BALANCE_SHADOW_STATUS_20261014.json',
                                 'GAUSSIAN_REGION_BALANCE_SHADOW_LIVE_CHECK_20261014.json'):
                    fetch(OLD + '/' + relative, relative, mutable=True)
                fetch(OLD + '/GAUSSIAN_REGION_BALANCE_SHADOW_20261014.log', 'GAUSSIAN_REGION_BALANCE_SHADOW_20261014.log.tail',
                      tail=20_000, mutable=True)
                done_raw = read_remote(sftp, OLD + '/' + RESULT + '/DIAGNOSTIC_COMPLETE.json')
                if done_raw is None:
                    state['shadow_completion'] = 'marker_absent; partial artifacts deliberately not collected'
                    missing.append(OLD + '/' + RESULT + '/DIAGNOSTIC_COMPLETE.json')
                else:
                    done = json.loads(done_raw[0])
                    require(done.get('status') == 'completed' and done.get('n_patches') == 56
                            and done.get('state_unchanged') is True and done.get('parameter_updates') is False
                            and done.get('held_out_GT_read') is False and set(done.get('artifacts_sha256', {})) == ARTIFACTS,
                            'Actual shadow completion/5artifact set differs')
                    for name in sorted(ARTIFACTS):
                        fetch(OLD + '/' + RESULT + '/' + name, RESULT + '/' + name, optional=False,
                              expected=done['artifacts_sha256'][name])
                    fetch(OLD + '/' + RESULT + '/DIAGNOSTIC_COMPLETE.json', RESULT + '/DIAGNOSTIC_COMPLETE.json',
                          optional=False, expected=digest(done_raw[0]))
                    state['shadow_completion'] = 'real_DIAGNOSTIC_COMPLETE_and5artifact_SHA_verified'
            state.update(port=port, captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         files=records, missing_paths=missing)
            save_fresh(base, 'captures/' + capture + '/' + tag + '/collection_manifest.json', json_bytes(state))
            return state
    finally: client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--new-port', type=int, default=38773); parser.add_argument('--old-port', type=int, default=17552)
    parser.add_argument('--password-stdin', action='store_true')
    parser.add_argument('--include-donor-weights', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({'role': 'PLAN_ONLY_NOT_EXECUTED', 'output': str(args.output), 'ports': [args.new_port, args.old_port],
            'remote_writes_or_commands': False, 'default_weights_downloaded': False,
            'password_stdin_order': ['NEW port38773', 'OLD port17552'], 'snapshot_outputs_created': False}, indent=2)); return
    base = args.output.resolve(); base.mkdir(parents=True, exist_ok=True)
    capture = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:6]
    results = []
    for tag, port in [('new_38773', args.new_port), ('old_17552', args.old_port)]:
        results.append(collect_host(base, capture, tag, port, args.password_stdin, args.include_donor_weights))
    require(results[0]['donor_READY_sha256'] == results[1]['donor_READY_sha256'], 'Two server views of shared donor READY differ')
    summary = {'schema_version': 1, 'capture': capture, 'collector_sha256': file_sha(Path(__file__)),
               'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'capture_completed',
               'scope': 'read-only source-backed records, not training/evaluation results',
               'remote_writes_or_commands': False, 'weights_downloaded': args.include_donor_weights,
               'hosts': [{'port': x['port'], 'shadow_completion': x['shadow_completion'], 'files': len(x['files'])} for x in results]}
    save_fresh(base, 'captures/' + capture + '/capture_complete.json', json_bytes(summary))
    print(json.dumps(summary))


if __name__ == '__main__': main()
