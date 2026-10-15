"""Read-only one-shot B/lambda0 progress snapshots; never start/restart work.

Passwords come only from remote.connect(port)'s getpass. No PTY, remote writes,
model imports or training waits. Default order: NEW38773, then OLD17552.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
SPECS = {
    38773: ('B', 'status_B_formal_38773.txt', 'ad0ee77f6b491ee1960cdf82849bd3f2ee1e97e4109095bbcb90070836b1cd81',
            '231e191e77e5f492b2602df19ce29046bb78dcb582dd8b03f1e982f21493828a', 10., 'gta_region_balance200x125_20261014'),
    17552: ('lambda0', 'status_gaussian_lambda0_17552.txt', '1e70915479984ea821054ded502fa6cbce74950be966834b6b20781f4b27bcf7',
            'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652', 0., 'gta_gaussian_lambda0_200x125_20261014')}
INITIAL = 'd84df10851c2532e9b70311eba2efe07049f5348e56fe4e76daf7856e6427d79'


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(value): return hashlib.sha256(value).hexdigest()


def finite_tree(value):
    if isinstance(value, dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value, list): return all(finite_tree(v) for v in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


def validate(port, status, config):
    name, _, _, source, weight, run = SPECS[port]
    expected_output = ROOT+'/results/'+run+'/training'
    need(finite_tree(status) and finite_tree(config), 'Nonfinite JSON numeric value')
    need(config.get('method') == 'gta' and config.get('source_sha256') == source, 'Wrong run/source')
    need(config.get('output') == expected_output and config.get('epochs') == 200 and config.get('steps') == 125,
         'Wrong output/budget')
    need(config.get('gaussian_weight') == weight and config.get('gaussian_late_weight') is None
         and config.get('gaussian_schedule', {}).get('base') == weight
         and config.get('gaussian_schedule', {}).get('late') is None and config.get('no_ds') is False,
         'Wrong Gaussian coefficient/schedule/DS')
    count, latest = status.get('epochs_completed'), status.get('latest_epoch')
    need(type(count) is int and 0 <= count <= 200, 'Invalid completed-epoch count')
    if count:
        need(isinstance(latest, dict) and type(latest.get('epoch')) is int and latest['epoch'] == count-1,
             'Completed count and zero-based latest epoch disagree')
        epoch = latest['epoch']
        need(latest.get('iterations') == 125 and isinstance(latest.get('loss'), (int, float))
             and not isinstance(latest['loss'], bool), 'Invalid latest iteration/loss record')
        need(latest.get('gaussian_lambda') == weight and latest.get('gaussian_lambda_configured') == weight,
             'Latest effective/configured lambda differs')
        need(latest.get('stage') == ('frozen' if epoch < 50 else 'partial' if epoch < 70 else 'joint'), 'Wrong classifier stage')
        theta = status.get('actual_theta0_brief') or {}
        need(theta.get('state_sha256') == INITIAL and theta.get('tensor_count') == 327
             and theta.get('model_state_and_RNG_unchanged') is True, 'Initial-state receipt differs')
        original_key = 'historical_main_theta0_verified' if port == 38773 else 'actual_original_main_epoch0_state_verified'
        need(theta.get(original_key) is False, 'Historical main theta0 must remain unverified')
    else:
        need(latest is None, 'Zero completed epochs must have no latest row')
    brief = status.get('actual_config_brief')
    if brief is not None:
        need(all(config.get(k) == v for k, v in brief.items()), 'Command brief and SFTP config disagree')
    state = status.get('status') or {}
    return {'port': port, 'run': name, 'status': state.get('status'), 'epochs_completed': count,
            'latest_epoch_0based': latest['epoch'] if count else None,
            'loss': latest['loss'] if count else None, 'lambda': weight,
            'parent_alive': (status.get('parent_process') or {}).get('alive'),
            'child_alive': (status.get('child_process') or {}).get('alive'), 'error': state.get('error'),
            'source_sha256': source, 'captured_at_utc': status.get('captured_at_utc')}


def collect(port, directory):
    from remote import connect
    name, filename, expected_sha, _, _, run = SPECS[port]
    command_bytes = (HERE/filename).read_bytes()
    need(sha(command_bytes) == expected_sha, 'Read-only status command changed: '+filename)
    print('Connecting '+str(port)+' ('+name+'); password is session-only.', flush=True)
    client = connect(port)
    try:
        stdin, stdout, stderr = client.exec_command(command_bytes.decode('utf-8').replace('\r\n', '\n'), get_pty=False, timeout=60)
        stdin.close()
        raw, errors = stdout.read(), stderr.read()
        exit_code = stdout.channel.recv_exit_status()
        (directory/(str(port)+'.stdout.txt')).write_bytes(raw)
        (directory/(str(port)+'.stderr.txt')).write_bytes(errors)
        need(exit_code == 0, 'Remote read-only status command failed; see saved stderr')
        status = json.loads(raw.decode('utf-8'))
        config_path = ROOT+'/results/'+run+'/training/config.json'
        with client.open_sftp() as sftp:
            with sftp.open(config_path, 'rb') as stream: config_bytes = stream.read()
        (directory/(str(port)+'.config.json')).write_bytes(config_bytes)
        config = json.loads(config_bytes.decode('utf-8'))
        summary = validate(port, status, config)
        packet = {'role': 'READ_ONLY_PROGRESS_NOT_COMPLETION_VERIFICATION', 'port': port,
                  'status_command_sha256': expected_sha, 'raw_stdout_sha256': sha(raw),
                  'actual_config_path': config_path, 'actual_config_sha256': sha(config_bytes),
                  'summary': summary, 'raw_status': status, 'actual_config': config,
                  'snapshot_scope': 'Latest finished epoch only; history may continue growing between reads.'}
        (directory/(str(port)+'.json')).write_text(json.dumps(packet, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return {'port': port, 'status': 'validated', 'packet_sha256': sha((directory/(str(port)+'.json')).read_bytes())}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, choices=tuple(SPECS), help='Omit to ask sequentially for both servers')
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = BASE/'fidelity_v2/server_records/dual_control_progress_snapshots'/stamp
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for port in ([args.port] if args.port else [38773, 17552]):
        try: results.append(collect(port, directory))
        except Exception as error:
            item = {'port': port, 'status': 'read_or_validation_failed', 'error': str(error)}
            results.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    manifest = {'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'observer_sha256': sha(Path(__file__).read_bytes()), 'remote_mutations': False,
                'password_saved': False, 'results': results}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Saved snapshot: '+str(directory), flush=True)
    raise SystemExit(0 if all(r['status'] == 'validated' for r in results) else 1)


if __name__ == '__main__': main()
