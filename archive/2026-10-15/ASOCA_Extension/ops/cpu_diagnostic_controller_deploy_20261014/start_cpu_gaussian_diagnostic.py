"""One bounded CPU-only diagnostic of an immutable GTA70 checkpoint."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

r = Path('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014')
output = r / 'results/gaussian_epoch070_cpu_bank56'
status = r / 'CPU_GAUSSIAN70_STATUS.json'
script = r / 'ops/diagnose_gaussian.py'
expected = '62daf89c7c49d23f3cb96a32a1970b2bb5fd5c68311aa197705850a5f7dba361'
lock = (r / '.cpu_gaussian_diagnostic.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
child = None
began = False
state = {'status': 'starting', 'pid': os.getpid(), 'new_training': False,
         'device': 'cpu', 'threads': 2, 'role': 'fixed training28 diagnostic; not validation or checkpoint selection'}

def save(**items):
    state.update(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **items)
    tmp = status.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(status)

def stop(signum, frame):
    raise KeyboardInterrupt('Signal ' + str(signum))

signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
try:
    if status.exists() or output.exists():
        raise RuntimeError('CPU diagnostic state/output already exists; inspect before any restart')
    if hashlib.sha256(script.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Diagnostic source differs from independently reviewed version')
    ready = json.loads((r / 'GAUSSIAN_DIAGNOSTIC_CPU_READY.json').read_text())
    if ready['status'] != 'passed_header_and_parameter_readiness' or ready['cuda_initialized']:
        raise RuntimeError('Actual CPU readiness does not match the intended scope')
    available = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    cg = Path('/sys/fs/cgroup')
    maximum = (cg / 'memory.max').read_text().strip()
    if maximum != 'max':
        available = min(available, int(maximum) - int((cg / 'memory.current').read_text()))
    quota, period = (cg / 'cpu.max').read_text().split()
    cores = len(os.sched_getaffinity(0)) if quota == 'max' else int(quota) / int(period)
    def usage():
        return int(dict(s.split() for s in (cg / 'cpu.stat').read_text().splitlines())['usage_usec'])
    start_time, first = time.monotonic(), usage()
    time.sleep(1)
    used_cores = (usage() - first) / (1e6 * (time.monotonic() - start_time))
    if available < 8 * 2**30 or cores - used_cores < 3:
        raise RuntimeError('Insufficient measured spare CPU/memory; no diagnostic started')
    weights = r / 'results/gta_enhanced200x125/training/checkpoints/epoch070.pt'
    if not weights.is_file():
        raise RuntimeError('Immutable epoch070 is missing')
    proof = ready['readiness_plan']
    if (proof['diagnostic_script_sha256'] != expected or proof['checkpoint_epoch'] != 69
            or hashlib.sha256(weights.read_bytes()).hexdigest() != proof['input_sha256']['checkpoint']):
        raise RuntimeError('CPU-ready immutable weight/source identity changed')
    command = [sys.executable, '-u', str(script), '--checkpoint', str(weights), '--manifest',
               str(r / 'dataset/cache/manifest.json'), '--device', 'cpu', '--output', str(output)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    save(status='running', phase='cpu_only_fixed_bank_diagnostic', command=command,
         source_sha256=expected, available_GiB=available / 2**30, effective_cpu_cores=cores,
         measured_existing_cpu_cores=used_cores, wall_budget_seconds=1800)
    began = True
    with (r / 'logs/gaussian_epoch070_cpu.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=r, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True, preexec_fn=lambda: os.nice(10))
        save(child_pid=child.pid)
        deadline = time.monotonic() + 1800
        while child.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError('CPU diagnostic exceeded its bounded wall budget; inspect partial outputs')
            metrics = output / 'patch_metrics.jsonl'
            complete = sum(1 for line in metrics.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')) if metrics.exists() else 0
            save(completed_patches=complete)
            time.sleep(10)
        result = child.returncode; child = None
    if result:
        raise RuntimeError('CPU diagnostic failed with exit ' + str(result))
    marker = json.loads((output / 'DIAGNOSTIC_COMPLETE.json').read_text())
    summary_path = output / 'summary.json'
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    if (marker.get('n_patches') != 56 or marker.get('summary_sha256') != digest
            or marker.get('parameter_updates') is not False or marker.get('outer8_used') is not False):
        raise RuntimeError('Diagnostic completion/source scope differs')
    state.pop('child_pid', None)
    save(status='completed', phase='cpu_diagnostic_ready', completed_patches=56,
         summary_sha256=digest, output=str(output), note='No GPU or model updates; interpret only this fixed training bank')
except BaseException as error:
    if child is not None and child.poll() is None:
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError('Child group ownership changed; refusing unrelated termination') from error
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
    if began:
        save(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed_stop',
             phase='stopped_for_inspection', error=repr(error))
    raise
finally:
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
