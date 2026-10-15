"""Fresh-only, read-only SFTP collection of actual completed GTA20/80+curves.

No remote commands/writes, model, GPU or partial trajectory capture. The actual
producer uses POSTFIT_VERIFIED + completed POSTFIT_STATUS, not an invented
POSTFIT_COMPLETE. Reuse downloaded training bytes including final69MiB locally.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time

from remote import connect

ROOT = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
TRAIN = 'results/gta_enhanced200x125/training'
TRAJECTORY = 'results/gta_enhanced200x125/internal4_trajectory'
CURVES = 'results/gta_enhanced200x125/learning_curves'
IDS = ('Diseased_10', 'Diseased_20', 'Normal_12', 'Normal_3')
NAMES = tuple(f'epoch{i:03d}' for i in range(10, 200, 10)) + ('final',)
FINAL = '2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'
CONFIG = 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
SOURCE = 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
FREEZE = '9b294024e33b0d0aaa12b473be53dd3a81dfcd9dd020446948fe667405c7ff46'
BASE = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records'
LIMIT = 128 * 1024**2


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_remote(sftp, relative, optional=False):
    pure = PurePosixPath(relative)
    need(not pure.is_absolute() and '..' not in pure.parts, 'Unsafe remote relative path')
    path = ROOT + '/' + str(pure)
    try: before = sftp.lstat(path)
    except FileNotFoundError:
        if optional: return None
        raise
    need(stat.S_ISREG(before.st_mode) and before.st_size <= LIMIT, 'Expected bounded regular evidence file: ' + relative)
    with sftp.open(path, 'rb') as stream: data = stream.read(before.st_size)
    after = sftp.stat(path)
    need(len(data) == before.st_size and (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime),
         'Evidence changed while capturing: ' + relative)
    return data, {'relative_path': relative, 'remote_path': path, 'bytes': len(data), 'sha256': sha(data), 'source_mtime_epoch': before.st_mtime}


def gate(sftp):
    small = {}
    for name in ('POSTFIT_STATUS.json', 'POSTFIT_VERIFIED.json', 'POSTFIT_PROTOCOL.json', TRAJECTORY + '/TRAJECTORY_COMPLETE.json'):
        value = read_remote(sftp, name, optional=True)
        if value is None: return None, {'reason': 'actual_completion_marker_not_present', 'missing': name}
        small[name] = value
    status, verified, protocol, complete = [json.loads(small[name][0]) for name in small]
    if status.get('status') != 'completed':
        return None, {'reason': 'existing_postfit_not_completed', 'actual_status': status.get('status'), 'phase': status.get('phase')}
    need(verified.get('status') == 'postfit_verified' and verified.get('verified_checkpoint_evaluations') == 20
         and verified.get('verified_case_evaluations') == 80 and verified.get('root') == ROOT
         and status.get('verified_checkpoints') == 20 and status.get('verified_case_evaluations') == 80
         and status.get('verified_receipt_sha256') == sha(small['POSTFIT_VERIFIED.json'][0])
         and status.get('protocol_sha256') == sha(small['POSTFIT_PROTOCOL.json'][0]), 'Actual POSTFIT20/80 receipts do not close')
    identity = verified['trajectory_identity']
    need(protocol.get('source_sha256') == SOURCE and protocol.get('main_freeze_sha256') == FREEZE
         and protocol.get('expected_evaluations') == 20 and protocol.get('expected_case_evaluations') == 80
         and protocol.get('case_ids') == list(IDS) and protocol.get('new_training') is False,
         'Actual postfit protocol identity differs')
    need(complete.get('n_evaluations') == 20 and all(complete.get(k) == v for k, v in identity.items())
         and verified.get('trajectory_complete_sha256') == sha(small[TRAJECTORY + '/TRAJECTORY_COMPLETE.json'][0])
         and identity.get('training_config_sha256') == CONFIG and identity.get('source_sha256') == SOURCE
         and identity.get('case_ids') == list(IDS) and identity.get('sw_batch_size') == 1
         and [r.get('name') for r in identity.get('checkpoints', [])] == list(NAMES)
         and [r.get('epoch') for r in identity['checkpoints']] == list(range(9, 190, 10)) + [199]
         and identity['checkpoints'][-1]['checkpoint_sha256'] == FINAL, 'Actual completed trajectory identity differs')
    cases = verified.get('cases', [])
    need(len(cases) == 80 and {(r['checkpoint'], r['case_id']) for r in cases} == {(n, i) for n in NAMES for i in IDS},
         'Actual independently verified80 case receipts missing/duplicated')
    return {'small': small, 'verified': verified, 'identity': identity}, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE / 'gta_complete_trajectory_20261014')
    p.add_argument('--reuse-training', type=Path, default=BASE / 'gta_training_complete_20261014' / TRAIN)
    p.add_argument('--wait-seconds', type=int, default=0)
    p.add_argument('--plan', action='store_true')
    a = p.parse_args()
    need(0 <= a.wait_seconds <= 1800, 'Wait must be0..1800seconds')
    if a.plan:
        print(json.dumps({'role': 'PLAN_ONLY_NOT_EXECUTED', 'port': 38773, 'source_root': ROOT,
            'output': str(a.output), 'reuse_training': str(a.reuse_training), 'partial_trajectory_downloaded': False,
            'actual_completion_contract': 'completed POSTFIT_STATUS + POSTFIT_VERIFIED20/80 + actual TRAJECTORY_COMPLETE20',
            'expected_core_files': 234, 'additional_curve_files': 5, 'additional_gradient_probe_if_present': 1,
            'network_weight_download': False, 'server_connections': 0}, indent=2)); return
    output = a.output.resolve(); reuse = a.reuse_training.resolve()
    need(not output.exists(), 'Fresh evidence directory required; never overwrite or merge existing records')
    client = connect(38773)
    capture = {'source_root': ROOT, 'port': 38773, 'read_only_server_access': True, 'started_at_utc': now(), 'files': []}
    began = False
    try:
        with client.open_sftp() as sftp:
            deadline = time.monotonic() + a.wait_seconds
            while True:
                evidence, waiting = gate(sftp)
                if evidence: break
                if time.monotonic() >= deadline:
                    print(json.dumps({'status': 'waiting_actual_canonical_completion', **waiting,
                                      'partial_files_downloaded': 0, 'local_target_created': False})); return
                time.sleep(min(15, max(.1, deadline - time.monotonic())))
            verified = evidence['verified']
            expected = {TRAIN + '/config.json': CONFIG, TRAIN + '/final.pt': FINAL,
                TRAIN + '/history.jsonl': verified['history_sha256'], TRAIN + '/TRAINING_COMPLETE.json': verified['training_complete_sha256'],
                'configs/main_source_freeze.json': FREEZE, TRAJECTORY + '/TRAJECTORY_COMPLETE.json': verified['trajectory_complete_sha256'],
                TRAJECTORY + '/validation_history.jsonl': verified['validation_history_sha256']}
            for item in verified['cases']:
                base = TRAJECTORY + '/' + item['checkpoint'] + '/' + item['case_id']
                expected[base + '.json'] = item['case_json_sha256']; expected[base + '_prediction.nrrd'] = item['prediction_sha256']
            curve_names = []
            for item in verified['curve_artifacts']:
                relative = str(PurePosixPath(item['path']).relative_to(PurePosixPath(ROOT)))
                need(relative.startswith(CURVES + '/'), 'Curve artifact outside exact canonical curve directory')
                curve_names.append(relative); expected[relative] = item['sha256']
            need(len(set(curve_names)) == 5, 'Actual canonical curve package requires5files')
            train_names = ['config.json', 'history.jsonl', 'initialization_audit.json', 'TRAINING_COMPLETE.json', 'final.pt']
            for name in train_names:
                need((reuse / name).is_file(), 'Previously downloaded complete training record missing: ' + name)
                if TRAIN + '/' + name in expected: need(file_sha(reuse / name) == expected[TRAIN + '/' + name], 'Reused actual training bytes differ: ' + name)
            names = [TRAIN + '/' + name for name in train_names]
            names += [TRAJECTORY + '/' + name for name in ('trajectory_protocol.json', 'trajectory_status.json', 'validation_history.jsonl', 'validation_curve.csv', 'TRAJECTORY_COMPLETE.json')]
            names += ['POSTFIT_STATUS.json', 'POSTFIT_PROTOCOL.json', 'POSTFIT_VERIFIED.json', 'configs/main_source_freeze.json']
            files = ['summary.json', 'case_metrics.jsonl', 'EVALUATION_COMPLETE.json'] + [i + suffix for i in IDS for suffix in ('.json', '_prediction.nrrd')]
            names += [TRAJECTORY + '/' + n + '/' + f for n in NAMES for f in files]
            need(len(names) == 234, 'Core evidence inventory count differs')
            names += curve_names
            probe = read_remote(sftp, TRAIN + '/gradient_probes.jsonl', optional=True)
            if probe: names.append(TRAIN + '/gradient_probes.jsonl')
            output.mkdir(parents=True); began = True
            for index, relative in enumerate(names, 1):
                target = output / relative; target.parent.mkdir(parents=True, exist_ok=True)
                if relative == TRAIN + '/final.pt':
                    before = sftp.stat(ROOT + '/' + relative)
                    need(before.st_size == (reuse / 'final.pt').stat().st_size, 'Actual remote final size differs from reused weight')
                    os.link(reuse / 'final.pt', target)
                    record = {'relative_path': relative, 'remote_path': ROOT + '/' + relative, 'bytes': target.stat().st_size,
                              'sha256': file_sha(target), 'transfer': 'local_hardlink_from_actual_downloaded_training; no network weight download', 'reused_source': str(reuse / 'final.pt')}
                else:
                    data, record = read_remote(sftp, relative)
                    if relative.startswith(TRAIN + '/') and PurePosixPath(relative).name in train_names:
                        need(data == (reuse / PurePosixPath(relative).name).read_bytes(), 'Actual remote/reused training record bytes differ')
                    with target.open('xb') as stream: stream.write(data)
                    need(file_sha(target) == record['sha256'], 'Downloaded bytes differ')
                    record['transfer'] = 'actual_remote_SFTP_bytes'
                if relative in expected: need(record['sha256'] == expected[relative], 'Actual completed receipt SHA differs: ' + relative)
                capture['files'].append(record)
                if index % 40 == 0: print(json.dumps({'phase': 'capture_actual_completed_canonical', 'files': index, 'planned': len(names)}), flush=True)
            # Close the actual producer/marker identities again after all transfers.
            final_gate, _ = gate(sftp)
            need(final_gate is not None and {k: sha(v[0]) for k, v in final_gate['small'].items()}
                 == {k: sha(v[0]) for k, v in evidence['small'].items()}, 'Actual canonical completion changed during transfer')
            capture.update(status='captured_completed_canonical_pending_independent_CPU_acceptance', ended_at_utc=now(),
                core_files=234, curve_files=5, total_files=len(names), actual_final_weights_bytes_hash_verified=True,
                network_weight_download=False, model_or_inference_or_training_started=False, collector_sha256=file_sha(Path(__file__)))
            print(json.dumps({'status': capture['status'], 'files': len(names), 'output': str(output), 'network_weight_download': False}))
    except BaseException as error:
        if began: capture.update(status='failed_or_partial_capture_not_completed', error=repr(error), ended_at_utc=now())
        raise
    finally:
        client.close()
        if began:
            with (output / 'collection_manifest.json').open('x', encoding='utf-8') as stream:
                json.dump(capture, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__': main()
