"""Bounded read-only wait/capture of the existing U-Net calibration controller.

Wait on its known process and completion marker; never start/restart a job.
Transfer saved original-grid predictions, final weights and source-backed logs.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

from remote import connect

ROOT = '/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2'
TRAIN = 'results/unet_recipe200x125/training'
TRAJECTORY = 'results/unet_recipe200x125/internal4_trajectory'
IDS = ('Diseased_10', 'Diseased_20', 'Normal_12', 'Normal_3')
NAMES = tuple(f'epoch{n:03d}' for n in range(10, 200, 10)) + ('final',)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(sftp, path):
    with sftp.open(path, 'rb') as stream:
        return stream.read()


def exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


def observed_process(sftp, pid, required):
    base = '/proc/' + str(pid)
    try:
        command = read(sftp, base + '/cmdline').replace(b'\0', b' ').decode()
        state = read(sftp, base + '/stat').decode().split(') ')[1].split()[0]
    except FileNotFoundError:
        return {'pid': pid, 'alive': False}
    return {'pid': pid, 'alive': state not in ('Z', 'X') and required in command,
            'state': state, 'command': command, 'identity_matches': required in command}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=900)
    a = p.parse_args()
    if not 0 <= a.wait_seconds <= 1800:
        p.error('Wait is bounded to 0..1800 seconds')
    output = a.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Fresh evidence directory required')
    output.mkdir(parents=True, exist_ok=True)
    capture = {'started_at_utc': now(), 'server_port': 17552, 'source_root': ROOT,
               'scope': 'read-only wait on existing known controller; no training or remote mutations',
               'wait_observations': [], 'files': []}
    client = connect(17552)
    try:
        with client.open_sftp() as sftp:
            deadline = time.monotonic() + a.wait_seconds
            last_progress = None
            while not exists(sftp, ROOT + '/UNET_CALIBRATION_COMPLETE.json'):
                controller = observed_process(sftp, 108861, ROOT + '/ops/start_unet_calibration.py')
                evaluator = observed_process(sftp, 120722, ROOT + '/code/evaluate_trajectory.py')
                status_path = ROOT + '/' + TRAJECTORY + '/trajectory_status.json'
                status = json.loads(read(sftp, status_path)) if exists(sftp, status_path) else None
                observation = {'captured_at_utc': now(), 'controller': controller,
                               'evaluator': evaluator, 'trajectory_status': status}
                capture['wait_observations'].append(observation)
                progress = (status or {}).get('completed')
                if progress != last_progress:
                    print(json.dumps({'phase': 'verified_wait', 'completed_checkpoints': progress,
                                      'controller_live': controller['alive'], 'evaluator_live': evaluator['alive']}), flush=True)
                    last_progress = progress
                if not controller['alive']:
                    # Completion may appear between the first marker read and process exit.
                    if exists(sftp, ROOT + '/UNET_CALIBRATION_COMPLETE.json'):
                        break
                    capture.update(status='owned_controller_missing_without_completion', ended_at_utc=now())
                    raise RuntimeError('Known controller stopped without completion; inspect, never restart automatically')
                if (status or {}).get('status') == 'failed_stop':
                    capture.update(status='observed_failed_stop', ended_at_utc=now())
                    raise RuntimeError('Existing trajectory reports failed_stop')
                if time.monotonic() >= deadline:
                    capture.update(status='observation_timeout_existing_job_not_restarted', ended_at_utc=now())
                    print(json.dumps({'status': capture['status'], 'controller_live': controller['alive'],
                                      'evaluator_live': evaluator['alive']}), flush=True)
                    return
                time.sleep(min(15, max(.1, deadline-time.monotonic())))
            names = [TRAIN + '/' + n for n in ('config.json', 'history.jsonl', 'initialization_audit.json',
                                               'TRAINING_COMPLETE.json', 'final.pt')]
            names += [TRAJECTORY + '/' + n for n in ('trajectory_protocol.json', 'trajectory_status.json',
                      'validation_history.jsonl', 'validation_curve.csv', 'TRAJECTORY_COMPLETE.json')]
            names += ['UNET_CALIBRATION_COMPLETE.json', 'unet_calibration_status.json', 'unet_calibration_protocol.json',
                      'configs/main_source_freeze.json']
            case_files = ['summary.json', 'case_metrics.jsonl', 'EVALUATION_COMPLETE.json']
            case_files += [i + extension for i in IDS for extension in ('.json', '_prediction.nrrd')]
            names += [TRAJECTORY + '/' + checkpoint + '/' + name for checkpoint in NAMES for name in case_files]
            for index, relative in enumerate(names):
                remote_path = ROOT + '/' + relative
                before = sftp.stat(remote_path)
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_path, str(target))
                after = sftp.stat(remote_path)
                if (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime):
                    raise RuntimeError('Completed evidence changed during capture: ' + relative)
                h = hashlib.sha256()
                with target.open('rb') as stream:
                    for block in iter(lambda: stream.read(8*1024**2), b''):
                        h.update(block)
                capture['files'].append({'relative_path': relative, 'bytes': target.stat().st_size,
                                         'sha256': h.hexdigest(), 'source_mtime_epoch': before.st_mtime})
                if (index+1) % 40 == 0:
                    print(json.dumps({'phase': 'capture_completed_evidence', 'files_copied': index+1,
                                      'files_planned': len(names)}), flush=True)
            config = json.loads((output / TRAIN / 'config.json').read_text(encoding='utf-8'))
            training_complete = json.loads((output / TRAIN / 'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
            final_record = next(r for r in capture['files'] if r['relative_path'] == TRAIN + '/final.pt')
            assert final_record['sha256'] == training_complete['final_sha256']
            assert config['method'] == 'unet' and training_complete['epoch'] == 199
            trajectory_complete = json.loads((output / TRAJECTORY / 'TRAJECTORY_COMPLETE.json').read_text(encoding='utf-8'))
            assert trajectory_complete['n_evaluations'] == 20
            capture.update(status='captured_completed_records_pending_independent_verification', ended_at_utc=now(),
                           final_weights_bytes_hash_verified=True, model_inference_or_training_started=False)
            print(json.dumps({'status': capture['status'], 'output': str(output), 'files': len(names),
                              'final_weight_hash_verified': True}), flush=True)
    except Exception as error:
        capture.update(status='capture_failed_or_partial', reason=str(error), ended_at_utc=now())
        raise
    finally:
        client.close()
        (output / 'collection_manifest.json').write_text(json.dumps(capture, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
