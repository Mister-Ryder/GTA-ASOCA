"""Owned OLD-server lambda0 recipe control. --plan never launches anything.

The frozen model/package is copied byte-for-byte to its original logical ROOT.
Only Gaussian weight/base and the fresh training output differ from main.
Hardware differs: this is a matched-recipe control, not strict loss-only causality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

OLD = Path('/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2')
LOGICAL = Path('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014')
SHARED = Path('/root/autodl-fs/experiments/GTA_ASOCA_Fidelity_20261014')
MIRROR = SHARED/'early_final_mirror_20261014'
DONORS = SHARED/'gaussian_control_donors_20261014'
RUN = 'gta_gaussian_lambda0_200x125_20261014'
RECORDS = OLD/'results/gaussian_lambda0_controller_20261014'
SHADOW = OLD/'results/gaussian_region_balance_final200_shadow_20261014'
HELPERS = {
    'run_early_fixed_final_eval.py': '2d7a4f24e4f2bae6fa58f5a023d726e1fb3f91bb0d2b7416a7dfe5bba4147e42',
    'prepare_gaussian_control.py': '0145556d123356956bd80546fb7f2eda0c44bd8761dac2c6ae1f6ae44c894a17',
    'diagnose_gaussian.py': '62daf89c7c49d23f3cb96a32a1970b2bb5fd5c68311aa197705850a5f7dba361',
    'verify_postfit.py': '698b349799cff927c9c8be2bf6330e0b2cbf4c1b9198ed8f4a303c6075b78e51',
}
DONOR_SHA = {
    'airway_backbone_epoch599.pkl': '29b6977b74e547735717fda0785a9d532397df364e1f641bed0e6682db0384d3',
    'classifier80_final.pt': '308ebc33d55766458ebd3241c8422dbadb9259b929570bc5f03afd51fbec31a4',
}
MAIN_FINAL = '2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def dependencies():
    directory = Path(__file__).resolve().parent
    for name, expected in HELPERS.items():
        need(sha(directory/name) == expected, 'Independent helper changed/missing: '+name)
    sys.path.insert(0, str(directory))
    guard = importlib.import_module('run_early_fixed_final_eval')
    prepare = importlib.import_module('prepare_gaussian_control')
    verify = importlib.import_module('verify_postfit')
    return guard, prepare, verify


def shadow_guard(guard):
    path = SHADOW/'DIAGNOSTIC_COMPLETE.json'
    record = read(path)
    expected = {'status': 'completed', 'n_patches': 56, 'state_unchanged': True,
                'parameter_updates': False, 'held_out_GT_read': False}
    need(all(record.get(k) == v for k, v in expected.items()), 'Actual final200 shadow is incomplete or has another scope')
    artifacts = record.get('artifacts_sha256', {})
    need(set(artifacts) == {'analytic_fixture.json', 'fixed_sampling_bank.json', 'diagnostic_plan.json', 'patch_metrics.jsonl', 'summary.json'},
         'Shadow requires the exact five artifact hashes')
    for name, expected_sha in artifacts.items():
        path = guard.safe_member(SHADOW, name)
        need(sha(path) == expected_sha, 'Shadow artifact changed: '+name)
    diagnostic_plan = read(SHADOW/'diagnostic_plan.json')
    need(diagnostic_plan.get('checkpoint_sha256') == MAIN_FINAL
         and diagnostic_plan.get('config_sha256') == guard.CONFIG
         and diagnostic_plan.get('manifest_sha256') == guard.MANIFEST
         and diagnostic_plan.get('source_sha256') == guard.SOURCE
         and diagnostic_plan.get('n_patches') == 56, 'Shadow is not the immutable final200/full-bank diagnostic')
    summary = read(SHADOW/'summary.json')
    need(summary.get('role') == 'ACTUAL_FINAL200_TRAIN28_OFFLINE_AB_DIAGNOSTIC_NOT_MODEL_RESULTS'
         and summary.get('state_sha256_before') == summary.get('state_sha256_after')
         and summary.get('parameter_updates') is False and summary.get('held_out_GT_read') is False, 'Shadow scope/state summary differs')
    return {'marker_sha256': sha(SHADOW/'DIAGNOSTIC_COMPLETE.json'), 'artifacts_sha256': artifacts}


def donors_guard(guard, main):
    marker = DONORS/'DONORS_READY.json'
    identity = sha(marker)
    ready = read(marker)
    need(ready.get('schema_version') == 1 and ready.get('role') == 'PRIVATE_IMMUTABLE_GTA_CONTROL_DONORS'
         and ready.get('main_config_sha256') == guard.CONFIG, 'Private donor READY schema/role/main differs')
    need(set(ready.get('files', {})) == set(DONOR_SHA), 'Private donor inventory differs')
    targets = {'airway_backbone_epoch599.pkl': main['backbone_init'], 'classifier80_final.pt': main['classifier_init']}
    for name, expected_sha in DONOR_SHA.items():
        item = ready['files'][name]
        need(item.get('sha256') == expected_sha and item.get('recorded_main_path') == targets[name], 'Donor identity/path differs: '+name)
        need(guard.file_record(guard.safe_member(DONORS, name)) == {'bytes': item.get('bytes'), 'sha256': expected_sha}, 'Donor actual bytes differ: '+name)
    need(sha(marker) == identity, 'Donor READY changed during validation')
    return {'READY_sha256': identity, 'payload': ready}


def preview_guard(guard, payload):
    directory = OLD/'results/gta_fixed_final_preview_20261014'
    marker = read(directory/'EVAL_COMPLETE.json')
    verified = guard.verify_result(OLD, directory, payload)
    need(all(marker.get(k) == v for k, v in verified.items()) and marker.get('status') == 'completed'
         and marker.get('checkpoint_epoch') == 199 and marker.get('parameter_updates') == 0
         and marker.get('formal_final_test') is False and marker.get('READY_sha256') == payload['READY_sha256']
         and marker.get('protocol_sha256') == sha(OLD/'EARLY_FIXED_FINAL_PROTOCOL.json'), 'Actual fixed-final internal4 preview is not completely verified')
    return {'marker_sha256': sha(directory/'EVAL_COMPLETE.json'), 'result': verified}


def copy_exact(source, target, expected):
    need(not target.exists() and not target.is_symlink(), 'Refusing existing copy target: '+str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    need(target.stat().st_size == expected['bytes'] and sha(target) == expected['sha256'], 'Copied bytes differ: '+str(target))


def prepare_logical(guard, main, donor_record):
    need(not LOGICAL.exists() and not LOGICAL.is_symlink(), 'Logical ROOT must be fresh; inspect previous state instead of resuming')
    frozen = guard.source_guard(OLD)
    LOGICAL.mkdir()
    for name, record in frozen['code_files'].items():
        copy_exact(OLD/'code'/name, LOGICAL/'code'/name, record)
    for name, record in frozen['command_files'].items():
        copy_exact(OLD/name, LOGICAL/name, record)
    copy_exact(OLD/'configs/main_source_freeze.json', LOGICAL/'configs/main_source_freeze.json',
               guard.file_record(OLD/'configs/main_source_freeze.json'))
    (LOGICAL/'dataset').mkdir()
    (LOGICAL/'dataset/cache').symlink_to(OLD.parent/'cache', target_is_directory=True)
    need((LOGICAL/'dataset/cache/manifest.json').resolve() == (OLD.parent/'cache/manifest.json').resolve(), 'Cache symlink target differs')
    # Reference evidence only: do not create a historical-main training folder.
    copy_exact(MIRROR/'config.json', LOGICAL/'configs/reference_main_actual_config.json', guard.file_record(MIRROR/'config.json'))
    for name, item in donor_record['payload']['files'].items():
        target = Path(item['recorded_main_path'])
        need(target.is_relative_to(LOGICAL), 'Donor target escapes logical ROOT')
        copy_exact(DONORS/name, target, {'bytes': item['bytes'], 'sha256': item['sha256']})
    guard.source_guard(LOGICAL)


def dataset_guard(guard, main):
    """Hash only train28 cached supervision plus internal4 image/raw-reference."""
    import numpy as np
    cache = OLD.parent/'cache'
    need(sha(cache/'manifest.json') == guard.MANIFEST, 'Immutable cache manifest differs')
    manifest = read(cache/'manifest.json')
    cases = {c['case_id']: c for c in manifest['cases']}
    result = {}
    for case_id in main['trained_case_ids']+guard.IDS:
        case = cases[case_id]
        folder = guard.safe_member(cache, case['cache_subdir']+'/image.npy').parent
        names = ['image.npy']
        if case_id in main['trained_case_ids']:
            names += [x+'.npy' for x in ('label', 'gaussian', 'skeleton', 'radius', 'branch_id', 'depth')]+['sampling.npz']
        for name in names:
            path = guard.safe_member(folder, name)
            result[case_id+'/'+name] = guard.file_record(path)
            if name.endswith('.npy'):
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                need(list(array.shape) == case['shape_zyx'], 'Cached supervision grid differs: '+case_id+'/'+name)
                if name == 'image.npy':
                    need(array.dtype == np.dtype('float16'), 'Cached image dtype differs')
        if case_id in guard.IDS:
            path = guard.safe_member(OLD.parent/'raw/CompressedVersion', case['label_relative_path'])
            need(sha(path) == case['source_sha256']['label'], 'Internal-four original GT changed')
            result[case_id+'/raw_GT'] = guard.file_record(path)
    return result


def training_preflight(guard, prepare, verify, expected):
    training = Path(expected['output'])
    guard.source_guard(LOGICAL)
    need(read(training/'config.json') == expected, 'Actual control configuration has unplanned differences')
    main = read(MIRROR/'config.json')
    need(sha(MIRROR/'config.json') == guard.CONFIG, 'Canonical main config changed')
    prepare.check_candidate(main, expected, str(training))
    need(sha(LOGICAL/'dataset/cache/manifest.json') == guard.MANIFEST, 'Manifest changed')
    marker = read(training/'TRAINING_COMPLETE.json')
    final_sha = sha(training/'final.pt')
    for k, v in {'epoch': 199, 'smoke': False, 'source_sha256': guard.SOURCE, 'manifest_sha256': guard.MANIFEST,
                 'trained_case_ids': expected['trained_case_ids'], 'final_sha256': final_sha, 'segmentation_updates': 25000,
                 'classifier_pretraining_updates': 5120, 'checkpoint_selection': 'fixed final', 'internal_selection_valid': True}.items():
        need(marker.get(k) == v, 'Control training completion differs: '+k)
    rows = guard.lines(training/'history.jsonl')
    need([r.get('epoch') for r in rows] == list(range(200))
         and [r.get('stage') for r in rows] == ['frozen']*50+['partial']*20+['joint']*130
         and all(r.get('iterations') == 125 and verify.finite(r.get('loss')) for r in rows), 'Incomplete/nonfinite control training history')
    periodic = training/'checkpoints'
    need({p.name for p in periodic.glob('epoch*.pt')} == {f'epoch{i:03d}.pt' for i in range(10, 201, 10)}, 'Control periodic checkpoint inventory differs')
    points = []
    for name in verify.NAMES:
        path = training/'final.pt' if name == 'final' else periodic/(name+'.pt')
        epoch = 199 if name == 'final' else int(name[5:])-1
        value = sha(path)
        verify.checkpoint_audit(path, expected, epoch)
        need(sha(path) == value, 'Control checkpoint changed during CPU audit')
        points.append({'path': str(path), 'epoch': epoch, 'name': name, 'checkpoint_sha256': value})
    verify.checkpoint_audit(periodic/'epoch200.pt', expected, 199)
    return {'schema_version': 1, 'status': 'preflight_verified', 'root': str(LOGICAL), 'training': str(training),
            'manifest': str((OLD.parent/'cache/manifest.json').resolve()), 'raw_root': str((OLD.parent/'raw/CompressedVersion').resolve()),
            'trajectory_identity': {'training_config_sha256': sha(training/'config.json'), 'manifest_sha256': guard.MANIFEST,
                'source_sha256': guard.SOURCE, 'evaluation_sources': {n: sha(LOGICAL/'code'/n) for n in sorted(guard.EVALUATORS)},
                'case_ids': guard.IDS, 'split': 'internal_diagnostic', 'smoke': False, 'checkpoints': points, 'sw_batch_size': 1,
                'checkpoint_selection': 'fixed final remains unchanged; intermediate weights are trajectory diagnostics',
                'formal_final_test': False, 'outer8_used_for_selection': False},
            'main_freeze_sha256': sha(LOGICAL/'configs/main_source_freeze.json'),
            'training_complete_sha256': sha(training/'TRAINING_COMPLETE.json'), 'history_sha256': sha(training/'history.jsonl'),
            'mandatory_checkpoints': 20, 'patients_per_checkpoint': 4, 'expected_case_evaluations': 80,
            'role': 'internal-four lambda0 development control; no original-eight selection/fivefold'}


def worker(args):
    """Only read the actual control model/RNG immediately before first forward."""
    guard, prepare, _ = dependencies()
    need(args.worker and args.logical_root == LOGICAL and args.records == RECORDS, 'Worker scope differs')
    owner = read(RECORDS/'STATUS.json')
    protocol = read(RECORDS/'PROTOCOL.json')
    need(owner.get('controller_pid') == os.getppid() and owner.get('controller_sha256') == sha(__file__)
         and protocol.get('status') == 'formal_control_authorized_recipe'
         and protocol.get('controller_sha256') == sha(__file__), 'Private worker requires its actual owning controller/protocol')
    guard.source_guard(LOGICAL)
    expected = read(RECORDS/'expected_lambda0_config.json')
    train_argv = read(RECORDS/'train_argv.json')['argv']
    need(train_argv[:3] == [str(OLD/'early_final_env311/bin/python'), '-u', str(LOGICAL/'code/train.py')], 'Worker frozen entrypoint differs')
    sys.path.insert(0, str(LOGICAL/'code'))
    import torch
    import train
    factory = train.make_model
    called = []

    def instrumented_factory(*factory_args, **factory_kwargs):
        need(not called, 'Unexpected second model construction')
        model = factory(*factory_args, **factory_kwargs)
        called.append(True)

        def before_first_forward(actual, inputs):
            handle.remove()
            need(read(Path(expected['output'])/'config.json') == expected, 'Actual first-forward config differs')
            need(torch.cuda.is_initialized(), 'Expected actual CUDA training model before snapshot')
            need(tuple(inputs[0].shape) == (2, 1, 128, 128, 128), 'Actual first-forward input differs')
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
            state = {k: v.detach().cpu().clone() for k, v in actual.state_dict().items()}
            class Snapshot:
                def state_dict(self): return state
                def parameters(self): return actual.parameters()
            identity = prepare.state_identity(Snapshot())
            need(identity['tensor_count'] == 327 and identity['parameter_count'] == expected['parameters'], 'Actual control initial-state capacity differs')
            snapshot = RECORDS/'actual_control_theta0_before_first_forward.pt'
            need(not snapshot.exists(), 'Actual control theta0 already exists')
            temporary = snapshot.with_suffix('.pt.tmp')
            torch.save({'model': state, 'rng_cpu': cpu_rng, 'rng_cuda': cuda_rng,
                        'role': 'actual control model after donors, before any forward/optimizer update'}, temporary)
            temporary.replace(snapshot)
            need(torch.equal(cpu_rng, torch.get_rng_state()) and all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all())), 'Readonly snapshot changed RNG')
            # Reading CPU clones and saving tensors do not mutate live params/BN.
            need(all(torch.equal(state[k], v.detach().cpu()) for k, v in actual.state_dict().items()), 'Readonly snapshot changed model state')
            atomic(RECORDS/'ACTUAL_CONTROL_THETA0.json', {
                'schema_version': 1, 'role': 'actual control pre-first-forward state; private artifact', 'state': identity,
                'snapshot': str(snapshot), 'snapshot_sha256': sha(snapshot), 'captured_at_utc': guard.now(),
                'config_sha256': sha(Path(expected['output'])/'config.json'), 'controller_sha256': sha(__file__),
                'source_sha256': guard.SOURCE, 'input_shape': list(inputs[0].shape), 'optimizer_updates': 0,
                'network_forwards_before_capture': 0, 'donors_loaded': True, 'model_state_and_RNG_unchanged': True,
                'CPU_RNG_sha256': hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
                'CUDA_RNG_sha256': [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in cuda_rng],
                'actual_original_main_epoch0_state_verified': False,
                'note': 'Historical main theta0 was not saved. This new snapshot cannot certify that historical state or its RNG.'})
        handle = model.register_forward_pre_hook(before_first_forward)
        return model

    train.make_model = instrumented_factory
    sys.argv = [train_argv[2]]+train_argv[3:]
    train.main()
    need((RECORDS/'ACTUAL_CONTROL_THETA0.json').is_file(), 'Training did not capture actual pre-forward state')


def plan():
    return {'schema_version': 1, 'status': 'NOT_MODEL_RESULTS_PLAN_ONLY', 'server_operations_performed': False,
            'old_root': OLD.as_posix(), 'fresh_logical_root': LOGICAL.as_posix(), 'records': RECORDS.as_posix(),
            'training': (LOGICAL/'results'/RUN/'training').as_posix(), 'main_mirror': MIRROR.as_posix(), 'private_donors': DONORS.as_posix(),
            'shadow_completion': (SHADOW/'DIAGNOSTIC_COMPLETE.json').as_posix(), 'python': (OLD/'early_final_env311/bin/python').as_posix(),
            'helper_sha256': HELPERS, 'segmentation_updates': 25000, 'epochs': 200, 'iterations_per_epoch': 125,
            'allowed_config_changes': ['gaussian_weight:10->0', 'gaussian_schedule.base:10->0', 'output:fresh control'],
            'source_files_mutated': False, 'classifier_pretraining_updates_reused': 5120,
            'internal4_checkpoint_evaluations': 20, 'case_inferences': 80, 'checkpoint_selection': 'fixed final',
            'fivefold': False, 'outer8_used': False, 'historical_main_theta0_known': False,
            'interpretation': 'Matched recipe on different OLD hardware/driver; not strict loss-only causality. No averaging or checkpoint/device selection.'}


def run(args):
    import fcntl
    guard, prepare, verify = dependencies()
    need(Path(__file__).resolve().parent == (OLD/'ops').resolve(), 'Deploy only this independent controller to OLD ROOT/ops')
    need(Path(sys.executable).absolute() == OLD/'early_final_env311/bin/python', 'Use matching isolated Python')
    need(not RECORDS.exists() and not LOGICAL.exists() and not LOGICAL.is_symlink(), 'Fresh-only controller: previous state requires explicit review')
    own = (OLD/'.gaussian_lambda0_controller.lock').open('a+')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logical_lock = (LOGICAL.parent/'.GTA_ASOCA_Fidelity_20261014.lambda0.lock').open('a+')
    fcntl.flock(logical_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    RECORDS.mkdir(parents=True)
    state = plan() | {'status': 'waiting_shadow_and_donors', 'server_operations_performed': True,
                      'controller_pid': os.getpid(), 'controller_sha256': sha(__file__),
                      'started_at_utc': guard.now(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      'gpu_lock_acquired': False, 'children': []}
    def save(**value):
        state.update(value, updated_at_utc=guard.now())
        atomic(RECORDS/'STATUS.json', state)
    cancel = guard.OwnedCancellation()
    signal.signal(signal.SIGTERM, cancel.request); signal.signal(signal.SIGINT, cancel.request)
    child = None
    gpu_lock = None
    env = {k: v for k, v in os.environ.items() if not k.startswith('GTA_') and k not in ('OUTPUT_ROOT', 'CLS_CHECKPOINT', 'PYTHONPATH')}
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')

    def execute(label, command, limit, gpu=False):
        nonlocal child
        log = RECORDS/(label+'.log')
        began = time.monotonic()
        record = {'stage': label, 'argv': command, 'started_at_utc': guard.now(), 'log': str(log), 'GPU_stage': gpu}
        state['children'].append(record)
        with log.open('x', encoding='utf-8') as stream:
            cancel.launching = True
            try:
                child = subprocess.Popen(command, cwd=LOGICAL/'code', env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record.update(pid=child.pid, pgid=os.getpgid(child.pid))
                need(record['pgid'] == child.pid, 'Owned child process group differs')
            finally:
                cancel.launching = False
            cancel.raise_pending()
            while child.poll() is None:
                need(time.monotonic()-began < limit, 'Stage time limit exceeded: '+label)
                save(status=label, child_pid=child.pid, stage_wall_seconds=time.monotonic()-began, gpu_lock_acquired=gpu_lock is not None)
                time.sleep(args.poll_seconds)
        record.update(returncode=child.returncode, ended_at_utc=guard.now(), wall_seconds=time.monotonic()-began)
        child = None
        need(record['returncode'] == 0, 'Owned stage failed: '+label)
        save(status=label+'_exited', child_pid=None)

    try:
        deadline = time.monotonic()+args.max_wait_seconds
        while not (SHADOW/'DIAGNOSTIC_COMPLETE.json').is_file() or not (DONORS/'DONORS_READY.json').is_file():
            need(time.monotonic() < deadline, 'Timed out waiting for actual shadow completion/donors')
            save(status='waiting_shadow_and_donors', gpu_lock_acquired=False)
            time.sleep(args.poll_seconds)
        baseline = guard.unet_guard(OLD)
        shadow = shadow_guard(guard)
        runtime = guard.runtime_profile()
        import torch
        torch.set_num_threads(8)
        CPU_policy = {'controller_affinity': sorted(os.sched_getaffinity(0)), 'controller_nice': os.getpriority(os.PRIO_PROCESS, 0),
                      'train_torch_threads': 8, 'train_environment_OMP_threads': 8, 'train_environment_MKL_threads': 8}
        need(len(CPU_policy['controller_affinity']) >= 8 and CPU_policy['controller_nice'] == 0,
             'Formal controller must not inherit diagnostic two-CPU affinity/nice10')
        payload = guard.payload_guard(OLD, MIRROR, check_header=False)
        need(payload['ready']['checkpoint_sha256'] == MAIN_FINAL, 'Actual main immutable final200 identity differs')
        preview = preview_guard(guard, payload)
        main = payload['config']
        donor = donors_guard(guard, main)
        prepare_logical(guard, main, donor)
        expected = prepare.expected_candidate(main, str(LOGICAL/'results'/RUN/'training'))
        differences = prepare.check_candidate(main, expected, expected['output'])
        atomic(RECORDS/'expected_lambda0_config.json', expected)
        train_argv = prepare.cli_argv(expected, LOGICAL/'code', str(OLD/'early_final_env311/bin/python'), LOGICAL/'code')
        atomic(RECORDS/'train_argv.json', {'argv': train_argv})
        save(status='CPU_preflight', main_config_sha256=guard.CONFIG, donor_receipt=donor, shadow_receipt=shadow,
             baseline_receipt=baseline, fixed_final_preview_receipt=preview, runtime=runtime,
             canonical_runtime=payload['runtime_audit'], CPU_policy=CPU_policy, allowed_actual_differences=differences)
        preflight_cmd = [sys.executable, '-u', str(OLD/'ops/prepare_gaussian_control.py'), '--main-config', str(LOGICAL/'configs/reference_main_actual_config.json'),
                         '--manifest', expected['manifest'], '--candidate-output', expected['output'], '--proposal-output', str(RECORDS/'cpu_preflight'),
                         '--candidate-config', str(RECORDS/'expected_lambda0_config.json'), '--python', sys.executable, '--cpu-check']
        execute('CPU_preflight', preflight_cmd, 600)
        pre = read(RECORDS/'cpu_preflight/CONTROL_PREFLIGHT.json')
        need(pre.get('CPU_rebuilt_main_and_lambda0_equal') is True and pre.get('matched_actual_historical_main_initialization') is False, 'CPU recipe reconstruction evidence differs')
        cache_identity = dataset_guard(guard, main)
        atomic(RECORDS/'cache_input_identity.json', cache_identity)
        save(status='waiting_idle_GPU', CPU_preflight_sha256=sha(RECORDS/'cpu_preflight/CONTROL_PREFLIGHT.json'))
        while True:
            need(time.monotonic() < deadline, 'Timed out waiting for OLD segmentation mutex/idle GPU')
            first = guard.gpu_snapshot()
            if first['all_compute_GPUs_idle']:
                candidate = (OLD/'.segmentation_stage.lock').open('a+')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                else:
                    gpu_lock = candidate
                    time.sleep(args.idle_confirmation_seconds)
                    second = guard.gpu_snapshot()
                    if second['all_compute_GPUs_idle']:
                        break
                    fcntl.flock(candidate, fcntl.LOCK_UN); candidate.close()
                    gpu_lock = None
            save(status='waiting_idle_GPU', GPU=first, gpu_lock_acquired=False)
            time.sleep(args.poll_seconds)
        need(shadow_guard(guard) == shadow and guard.unet_guard(OLD) == baseline
             and preview_guard(guard, payload) == preview, 'Baseline/shadow/actual fixed-final4 changed before launch')
        need(donors_guard(guard, main) == donor and sha(MIRROR/'config.json') == guard.CONFIG, 'Donor/main evidence changed before launch')
        guard.source_guard(OLD); guard.source_guard(LOGICAL)
        need(guard.runtime_profile() == runtime and guard.gpu_snapshot()['all_compute_GPUs_idle'], 'Runtime or GPU changed immediately before launch')
        need(not Path(expected['output']).exists(), 'Control training output is not fresh')
        protocol = plan() | {'status': 'formal_control_authorized_recipe', 'server_operations_performed': True,
                    'controller_sha256': state['controller_sha256'],
                    'main_config_sha256': guard.CONFIG, 'expected_control_config_sha256': sha(RECORDS/'expected_lambda0_config.json'),
                    'source_sha256': guard.SOURCE, 'source_freeze_sha256': guard.FREEZE, 'package_sha256': guard.PACKAGE,
                    'manifest_sha256': guard.MANIFEST, 'allowed_actual_differences': differences, 'donors': donor,
                    'shadow': shadow, 'baseline': baseline, 'fixed_final_preview': preview,
                    'reference_main_config': {'path': str(LOGICAL/'configs/reference_main_actual_config.json'),
                         'sha256': guard.CONFIG, 'source': str(MIRROR/'config.json'), 'role': 'reference evidence only; no historical-main training on OLD'},
                    'runtime': runtime, 'canonical_runtime': payload['runtime_audit'], 'CPU_policy': CPU_policy,
                    'GPU_idle_observations': [first, second], 'actual_theta0_capture': 'readonly first model forward pre-hook; no training source edit',
                    'CPU_preflight_sha256': sha(RECORDS/'cpu_preflight/CONTROL_PREFLIGHT.json'),
                    'cache_input_identity_sha256': sha(RECORDS/'cache_input_identity.json'), 'training_argv': train_argv}
        atomic(RECORDS/'PROTOCOL.json', protocol)
        execute('formal_training', [sys.executable, '-u', str(Path(__file__).absolute()), '--worker', '--logical-root', str(LOGICAL), '--records', str(RECORDS)], args.max_training_seconds, True)
        initial = read(RECORDS/'ACTUAL_CONTROL_THETA0.json')
        need(initial.get('model_state_and_RNG_unchanged') is True and initial.get('actual_original_main_epoch0_state_verified') is False
             and sha(initial['snapshot']) == initial['snapshot_sha256'], 'Actual control theta0 evidence differs')
        training = Path(expected['output'])
        before = training_preflight(guard, prepare, verify, expected)
        atomic(RECORDS/'POSTTRAIN_PREFLIGHT.json', before)
        trajectory = training.parent/'internal4_trajectory'
        execute('internal4_trajectory', [sys.executable, '-u', str(LOGICAL/'code/evaluate_trajectory.py'), '--training', str(training),
                '--manifest', expected['manifest'], '--raw-root', str(OLD.parent/'raw/CompressedVersion'), '--output', str(trajectory),
                '--python', sys.executable, '--sw-batch-size', '1'], args.max_trajectory_seconds, True)
        curves = training.parent/'learning_curves'
        execute('plot_curves', [sys.executable, str(LOGICAL/'ops/plot_learning_curves.py'), '--run', 'development='+str(training),
                '--validation', 'development='+str(trajectory/'validation_history.jsonl'), '--output', str(curves)], 600)
        # The independently hashed verifier's sole main-output path restriction
        # is replaced by our stricter lambda0 preflight; its entire 80-prediction,
        # metric/ledger/curve verification body is reused without editing bytes.
        verify.preflight = lambda root, train, manifest, raw: training_preflight(guard, prepare, verify, expected)
        verified = verify.postflight(LOGICAL, training, OLD.parent/'cache/manifest.json', OLD.parent/'raw/CompressedVersion', before)
        atomic(RECORDS/'POSTFIT_VERIFIED.json', verified)
        need(dataset_guard(guard, main) == cache_identity, 'Cached training/reference data changed during control')
        need(donors_guard(guard, main) == donor and shadow_guard(guard) == shadow and guard.unet_guard(OLD) == baseline, 'External frozen evidence changed during control')
        guard.source_guard(OLD); guard.source_guard(LOGICAL)
        need(sha(__file__) == state['controller_sha256'], 'Controller source changed')
        need(preview_guard(guard, payload) == preview, 'Actual main fixed-final4 evidence changed during control')
        post_payload = guard.payload_guard(OLD, MIRROR, check_header=False)
        need(post_payload['READY_sha256'] == payload['READY_sha256'] and post_payload['files'] == payload['files'],
             'Canonical main mirror evidence changed during control')
        complete = protocol | {'status': 'completed', 'completed_at_utc': guard.now(), 'children': state['children'],
                    'formal_training_complete_sha256': sha(training/'TRAINING_COMPLETE.json'), 'fixed_final_sha256': sha(training/'final.pt'),
                    'actual_control_theta0_receipt_sha256': sha(RECORDS/'ACTUAL_CONTROL_THETA0.json'),
                    'postfit_verified_sha256': sha(RECORDS/'POSTFIT_VERIFIED.json'), 'verified_case_inferences': 80,
                    'trajectory_complete_sha256': sha(trajectory/'TRAJECTORY_COMPLETE.json'), 'curve_manifest_sha256': sha(curves/'curve_manifest.json')}
        atomic(RECORDS/'CONTROL_COMPLETE.json', complete)
        save(status='completed', completion_sha256=sha(RECORDS/'CONTROL_COMPLETE.json'), child_pid=None)
    except BaseException as error:
        cancel.cleaning = True
        try:
            save(status='stopping_owned_child' if child is not None else 'failed_stop', error_type=type(error).__name__, error=str(error), cancellation_signals=cancel.pending)
        finally:
            guard.cleanup_owned_child(child)
            child = None
        save(status='failed_stop', child_pid=None, owned_child_exit_confirmed=True)
        raise
    finally:
        cancel.cleaning = True
        guard.cleanup_owned_child(child)
        if gpu_lock is not None:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN); gpu_lock.close()
        fcntl.flock(logical_lock, fcntl.LOCK_UN); logical_lock.close()
        fcntl.flock(own, fcntl.LOCK_UN); own.close()
        try:
            save(gpu_lock_acquired=False, owned_child_exit_confirmed=True, locks_released_at_utc=guard.now())
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--logical-root', type=Path, default=LOGICAL, help=argparse.SUPPRESS)
    p.add_argument('--records', type=Path, default=RECORDS, help=argparse.SUPPRESS)
    p.add_argument('--max-wait-seconds', type=float, default=7200)
    p.add_argument('--max-training-seconds', type=float, default=43200)
    p.add_argument('--max-trajectory-seconds', type=float, default=43200)
    p.add_argument('--poll-seconds', type=float, default=20)
    p.add_argument('--idle-confirmation-seconds', type=float, default=3)
    args = p.parse_args()
    need(min(args.max_wait_seconds, args.max_training_seconds, args.max_trajectory_seconds, args.poll_seconds, args.idle_confirmation_seconds) > 0
         and args.poll_seconds <= 60 and args.idle_confirmation_seconds <= 30, 'Invalid bounded timing arguments')
    if args.plan:
        need(not args.worker, 'Plan and private worker are mutually exclusive')
        print(json.dumps(plan(), indent=2)); return
    if args.worker:
        worker(args); return
    need(sys.platform == 'linux', 'Execution is restricted to the named Linux OLD host; use --plan locally')
    run(args)


if __name__ == '__main__':
    main()
