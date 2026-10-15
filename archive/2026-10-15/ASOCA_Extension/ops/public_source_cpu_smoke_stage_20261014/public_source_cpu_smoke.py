"""Bounded source-only CPU smoke for a fresh anonymous GTA-ASOCA checkout.

NOT_MODEL_RESULTS. No real data, checkpoint loading, optimizer, training, CUDA
context or private warm-start reproduction. This helper stays OUTSIDE checkout.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

COMMIT = '7a7e7faafef9ee99f5b714d6e808d239b79d95a0'
URL = 'https://github.com/Mister-Ryder/GTA-ASOCA.git'
MAIN_SHA = 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
CLASS_SHA = '61d752700a57dc02c6c1d7cdbcc59d1171957bf072f4fd46394af3980eff0bb1'
PACKAGE_SHA = 'fa3fad9af0fae985f79efc7cf8cab6bbe251bf42e67ae75678fef6c1c2539480'
HELP = ['code/prepare_data.py', 'code/audit_priors.py',
        'classifier_pretraining/code/pretrain_classifier.py', 'code/train.py',
        'code/evaluate.py', 'code/evaluate_trajectory.py']
SOURCE_USED = {'train.py', 'model.py', 'losses.py', 'classification_data.py', 'checkpoint_utils.py', 'data.py'}
PACKAGES = {'torch':'2.7.1+cu128','monai':'1.5.1','numpy':'1.26.4','scipy':'1.15.3',
            'scikit-image':'0.25.2','SimpleITK':'2.5.2','scikit-learn':'1.7.2','pandas':'2.3.3','tqdm':'4.67.1'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def git(root, *args, deadline=None):
    remaining = 30 if deadline is None else min(30,deadline-time.monotonic())
    require(remaining > 0,'Source audit deadline exceeded')
    return subprocess.check_output(['git', '-c', 'credential.helper=', '-c', 'http.extraHeader=',
                                    '-C', str(root), *args], timeout=remaining)


def combined(root, names):
    result = hashlib.sha256()
    for name in sorted(names):
        result.update(name.encode()); result.update((root/name).read_bytes())
    return result.hexdigest()


def source_identity(root, deadline=None):
    root = root.resolve()
    require((root/'.git').exists(), 'Use a fresh Git checkout of the anonymous public source, not a private source folder')
    require(git(root, 'rev-parse', 'HEAD',deadline=deadline).decode().strip() == COMMIT, 'Public commit differs')
    origin = git(root, 'config', '--get', 'remote.origin.url',deadline=deadline).decode().strip()
    require(origin in (URL, URL.removesuffix('.git')), 'Checkout origin must be the public HTTPS repository')
    require(not git(root, 'status', '--porcelain', '--untracked-files=all',deadline=deadline).strip(), 'Checkout has source changes or untracked files')
    tree = git(root, 'ls-tree', '-rz', COMMIT,deadline=deadline)
    files = {}
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        descriptor, path_bytes = entry.split(b'\t', 1)
        mode, kind, oid = descriptor.decode().split()
        name = path_bytes.decode('utf-8')
        require(kind == 'blob' and mode == '100644', 'Only regular source files are expected')
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Checkout member escapes/missing: '+name)
        data = path.read_bytes()
        expected = git(root, 'cat-file', 'blob', oid,deadline=deadline)
        require(data == expected, 'Checkout bytes differ from published commit: '+name)
        files[name] = {'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}
    require(len(files) == 48, 'Pinned public source must have exactly48 regular files')
    actual = set()
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in names:
            if Path(current) == root and name == '.git':
                continue
            actual.add((Path(current)/name).relative_to(root).as_posix())
    require(actual == set(files), 'Fresh source tree contains extra ignored/private files')
    provenance = json.loads((root/'PUBLIC_SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
    require(set(provenance['public_file_allowlist']) | {'PUBLIC_SOURCE_PROVENANCE.json'} == set(files), 'Public whitelist differs')
    require({k:files[k] for k in provenance['public_files']} == provenance['public_files'], 'Public blob provenance differs')
    require(combined(root, [n for n in files if n != 'PUBLIC_SOURCE_PROVENANCE.json']) == PACKAGE_SHA, 'Public package digest differs')
    main = root/'code'
    main_names = sorted(p.relative_to(main).as_posix() for p in main.rglob('*.py')
                        if p.name in SOURCE_USED or 'historical' in p.relative_to(main).parts)
    classifier = root/'classifier_pretraining/code'
    class_names = sorted(p.relative_to(classifier).as_posix() for p in classifier.rglob('*.py'))
    require(len(main_names) == 12 and combined(main, main_names) == MAIN_SHA, 'Main12 BC8 source differs')
    require(len(class_names) == 6 and combined(classifier, class_names) == CLASS_SHA, 'Correct pretraining6 61d source differs')
    return {'commit':COMMIT, 'origin':origin, 'source_files':files, 'source_file_count':48,
            'main12_source_sha256':MAIN_SHA, 'classifier6_source_sha256':CLASS_SHA,
            'public_package_sha256_excluding_manifest':PACKAGE_SHA,
            'README_sha256':files['README.md']['sha256'],
            'anonymous_download_authentication_scope':'Origin/commit/blob identity verified here; root records the separate fresh anonymous clone command.'}


def cpu_limits():
    result = {'torch_threads':2, 'torch_interop_threads':1, 'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
              'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'), 'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        require(bool(allowed), 'No allowed CPU cores')
        os.sched_setaffinity(0, set(allowed[:2]))
        result['CPU_affinity'] = sorted(os.sched_getaffinity(0))
        require(len(result['CPU_affinity']) <= 2, 'CPU affinity exceeded2 cores')
    if hasattr(os, 'getpriority'):
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if current < 10:
            os.setpriority(os.PRIO_PROCESS, 0, 10)
        result['nice'] = os.getpriority(os.PRIO_PROCESS, 0)
        require(result['nice'] >= 10, 'CPU smoke requires low priority')
    return result


def internal_module_origins(base):
    result = {}
    names = ('train','model','checkpoint_utils','classification_data','data','losses','pretrain_classifier')
    for name, module in list(sys.modules.items()):
        if name in names or name.startswith('historical.'):
            file = getattr(module, '__file__', None)
            if file:
                path = Path(file).resolve()
                require(path.is_relative_to(base), 'Internal module imported outside public source: '+name)
                result[name] = str(path)
    return result


def state_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        require(str(value.device) == 'cpu', 'Model state left CPU')
        array = value.detach().contiguous().numpy()
        result.update(name.encode()); result.update(str(array.dtype).encode()); result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def worker(args):
    sys.dont_write_bytecode = True
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA devices must be hidden for source-only CPU smoke')
    limits = cpu_limits()
    identity = source_identity(args.checkout)
    packages = {name:importlib.metadata.version(name) for name in PACKAGES}
    require(packages == PACKAGES and sys.version.split()[0] == '3.11.17','Use the tested nine-package Python3.11.17 runtime')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'Torch import initialized CUDA')
    def forbidden(*_args, **_kwargs):
        raise RuntimeError('NOT_MODEL_RESULTS CPU smoke forbids CUDA/checkpoint/optimizer operations')
    torch.cuda._lazy_init = forbidden
    torch.load = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    import SimpleITK
    SimpleITK.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    torch.random.default_generator.manual_seed(20261014)
    root = args.checkout.resolve()
    result = {'role':'NOT_MODEL_RESULTS', 'mode':args.worker_mode, 'commit':COMMIT,
              'untrained':True, 'real_dataset_used':False, 'checkpoint_loaded':False,
              'optimizer_created':False, 'parameter_updates':0, 'CPU_limits':limits,
              'torch':str(torch.__version__), 'python':sys.version, 'packages':packages, 'started_at_utc':now()}
    if args.worker_mode == 'help':
        require(args.entrypoint in HELP, 'Only README current entrypoints may run --help')
        entry = root/args.entrypoint; base = entry.parent
        sys.path.insert(0, str(base)); sys.argv = [str(entry), '--help']
        try:
            runpy.run_path(str(entry), run_name='__main__')
        except SystemExit as exited:
            require(exited.code in (None,0), 'CLI --help failed')
        else:
            raise ValueError('CLI --help did not exit via argparse before execution')
        result.update(entrypoint=args.entrypoint, help_exit_code=0, internal_module_origins=internal_module_origins(base))
    else:
        base = root/('code' if args.worker_mode == 'main_forward' else 'classifier_pretraining/code')
        sys.path.insert(0, str(base))
        if args.worker_mode == 'main_forward':
            train = importlib.import_module('train')
            require(train.source_hash() == MAIN_SHA, 'Imported main training source differs')
            model = train.make_model('gta', True).cpu().eval()
        else:
            pretrainer = importlib.import_module('pretrain_classifier')
            require(pretrainer.source_hash() == CLASS_SHA, 'Imported classifier-stage source differs')
            model = pretrainer.AirwayAnatomicalClassifier(in_channels=1, generation_classes=7, structure_classes=7, location_classes=7).cpu().eval()
        require(not model.training, 'Smoke must use eval mode, with no BN/dropout training updates')
        before = state_hash(model)
        x = torch.randn((1,1,32,32,32), device='cpu')
        with torch.inference_mode():
            prediction = model(x)
        after = state_hash(model)
        require(before == after, 'Untrained CPU forward changed model parameters or buffers')
        if args.worker_mode == 'main_forward':
            require(list(prediction['logits'].shape) == [1,2,32,32,32], 'Main logits shape differs')
            require([list(t.shape) for t in prediction['aux_logits']] == [[1,2,16,16,16],[1,2,8,8,8]], 'DS shapes differ')
            require(list(prediction['gaussian'].shape) == [1,1,32,32,32], 'Gaussian output shape differs')
            classifier = prediction['classifier']
            tensors = [prediction['logits'], prediction['gaussian'], *prediction['aux_logits']]
            result.update(logits_shape=list(prediction['logits'].shape), gaussian_shape=list(prediction['gaussian'].shape),
                          decoder_DS_shapes=[list(t.shape) for t in prediction['aux_logits']])
        else:
            classifier = prediction; tensors = []
        require(all(list(classifier[n].shape) == [1,7] for n in ('generation','structure','location')), 'Three classifier head shapes differ')
        require(list(classifier['guidance_map'].shape) == [1,16,2,2,2], '32cube guidance shape differs')
        require([list(t.shape) for t in classifier['encoder_features']] == [[1,64,8,8,8],[1,128,4,4,4],[1,256,2,2,2]], 'Classifier spatial features differ')
        tensors += [classifier[n] for n in ('generation','structure','location','guidance_map')]+classifier['encoder_features']
        require(all(str(t.device) == 'cpu' and bool(torch.isfinite(t).all()) for t in tensors), 'Nonfinite or non-CPU output')
        require(all(p.grad is None for p in model.parameters()), 'Forward unexpectedly produced parameter gradients')
        result.update(input_shape=list(x.shape), classifier_head_shapes={n:list(classifier[n].shape) for n in ('generation','structure','location')},
                      guidance_shape=list(classifier['guidance_map'].shape), classifier_spatial_shapes=[list(t.shape) for t in classifier['encoder_features']],
                      all_outputs_finite=True, all_parameter_gradients_none=True, model_state_SHA_before=before, model_state_SHA_after=after,
                      state_unchanged=True, internal_module_origins=internal_module_origins(base))
    require(not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    require(source_identity(root) == identity, 'Fresh public source changed during CPU smoke')
    result.update(status='passed', CUDA_context_initialized=False, completed_at_utc=now())
    atomic(args.worker_receipt, result)


class Cancellation:
    def __init__(self):
        self.launching = False; self.cleaning = False; self.pending = []
    def request(self, signum, _frame=None):
        self.pending.append(int(signum)); self.check()
    def check(self):
        if self.pending and not self.launching and not self.cleaning:
            self.cleaning = True
            raise KeyboardInterrupt('Owned source CPU smoke cancelled: '+str(self.pending))


def stop_owned(child):
    if child is None or child.poll() is not None:
        return
    try:
        require(os.getpgid(child.pid) == child.pid, 'Refusing signal to unowned child group')
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    except Exception:
        # Do not signal an unverified group; retain ownership until this exact
        # Popen child exits rather than orphaning it during cancellation.
        pass
    while child.poll() is None:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def supervisor(args):
    require(sys.platform == 'linux', 'Actual bounded CPU smoke runs on OLD Linux; --plan is read-only locally')
    root = args.checkout.resolve(); output = args.output.resolve()
    require(not Path(__file__).resolve().is_relative_to(root) and not output.is_relative_to(root), 'Keep helper/receipts outside fresh public source')
    require(not output.exists(), 'Use a fresh receipt directory')
    started = time.monotonic(); deadline = started+args.timeout_seconds
    cpu_limits(); identity = source_identity(root,deadline=deadline)
    output.mkdir(parents=True)
    sys.dont_write_bytecode = True
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GTA_','ASOCA_')) and k not in ('PYTHONPATH','PYTHONHOME','CLS_CHECKPOINT','OUTPUT_ROOT')}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', GIT_TERMINAL_PROMPT='0')
    cancellation = Cancellation(); child = None
    signal.signal(signal.SIGTERM, cancellation.request); signal.signal(signal.SIGINT, cancellation.request)
    records = []
    jobs = [('help',entry) for entry in HELP]+[('main_forward',None),('classifier_forward',None)]
    try:
        for index,(mode,entry) in enumerate(jobs):
            remaining = deadline-time.monotonic(); require(remaining > 0, 'Global source CPU smoke deadline exceeded')
            receipt = output/f'worker{index:02d}_{mode}.json'; log = output/f'worker{index:02d}_{mode}.log'
            command = [sys.executable, '-I', '-u', str(Path(__file__).resolve()), '--checkout', str(root),
                       '--worker-mode', mode, '--worker-receipt', str(receipt)]
            if entry:
                command += ['--entrypoint', entry]
            with log.open('x',encoding='utf-8') as stream:
                cancellation.launching = True
                try:
                    child = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    require(os.getpgid(child.pid) == child.pid, 'Worker ownership PGID differs')
                finally:
                    cancellation.launching = False
                cancellation.check()
                child_pid = child.pid
                code = child.wait(timeout=max(.1, deadline-time.monotonic()))
            child = None
            require(code == 0 and receipt.is_file(), 'CPU source worker failed; inspect '+str(log))
            row = json.loads(receipt.read_text(encoding='utf-8'))
            require(row['status'] == 'passed' and row['role'] == 'NOT_MODEL_RESULTS' and row['CUDA_context_initialized'] is False, 'Worker receipt scope differs')
            records.append({'mode':mode,'entrypoint':entry,'child_pid':child_pid,'command':command,
                            'receipt':str(receipt),'receipt_sha256':sha(receipt),'log_sha256':sha(log)})
        require(source_identity(root,deadline=deadline) == identity, 'Checkout changed during source CPU smoke')
        result = {'status':'passed', 'role':'NOT_MODEL_RESULTS', 'utility_sha256':sha(__file__), 'checkout':str(root),
                  'source_identity':identity,'worker_records':records,'wall_seconds':time.monotonic()-started,
                  'timeout_seconds':args.timeout_seconds,'CPU_only':True,'max_torch_threads':2,'nice_minimum':10,
                  'CUDA_context_initialized':False,'private_checkpoint_or_real_data_used':False,'optimizer_created':False,'parameter_updates':0,
                  'private_warmstart_reproduction':False,'clinical_segmentation_result':False,
                  'limitations':['32cube random untrained eval-mode forward only; not128cube staged training or full-volume inference.',
                                 'Uses the existing matching dependency environment; not a from-empty-environment install test.',
                                 'No published/private warm-start weights, real dataset, losses/optimization or Dice/ASSD evaluation.',
                                 'Anonymous clone creation is independently recorded by root; this helper verifies its public origin/commit/48 blobs.']}
        atomic(output/'PUBLIC_SOURCE_CPU_SMOKE_COMPLETE.json',result)
        print(json.dumps({'status':'passed','role':'NOT_MODEL_RESULTS','commit':COMMIT,'n_help':len(HELP),'n_untrained_CPU_forwards':2,'receipt':str(output/'PUBLIC_SOURCE_CPU_SMOKE_COMPLETE.json')}))
    except BaseException as error:
        cancellation.cleaning = True
        stop_owned(child)
        child = None
        atomic(output/'PUBLIC_SOURCE_CPU_SMOKE_FAILED.json',{'status':'failed_stop','role':'NOT_MODEL_RESULTS','error':repr(error),
                'worker_records':records,'utility_sha256':sha(__file__),'wall_seconds':time.monotonic()-started,'model_results':False})
        raise
    finally:
        cancellation.cleaning = True; stop_owned(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--timeout-seconds',type=int,default=300)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--worker-mode',choices=('help','main_forward','classifier_forward'))
    parser.add_argument('--entrypoint')
    parser.add_argument('--worker-receipt',type=Path)
    args = parser.parse_args()
    require(1 <= args.timeout_seconds <= 300,'CPU smoke timeout must be1..300seconds')
    if args.plan:
        identity = source_identity(args.checkout)
        print(json.dumps({'role':'READ_ONLY_PLAN_NOT_MODEL_RESULTS','source_identity':identity,'help_entrypoints':HELP,
                          'CPU_forward_input':[1,1,32,32,32],'main_logits':[1,2,32,32,32],
                          'main_DS':[[1,2,16,16,16],[1,2,8,8,8]],'Gaussian':[1,1,32,32,32],
                          'classifier_heads_each':[1,7],'classifier_guidance':[1,16,2,2,2],
                          'timeout_seconds':args.timeout_seconds,'GPU_context_or_private_weights_or_training':False},indent=2)); return
    if args.worker_mode:
        require(args.worker_receipt is not None,'Worker receipt required'); worker(args); return
    require(args.output is not None,'Fresh output directory required'); supervisor(args)


if __name__ == '__main__':
    main()
