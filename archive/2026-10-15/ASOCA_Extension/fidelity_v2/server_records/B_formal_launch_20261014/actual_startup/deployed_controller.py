"""Owned NEW-server sealed-B formal control, requiring ROOT's actual decision.

No decision is inferred from finite gradients. --plan is read-only and carries
no authorization. Frozen A, sealed B and legacy main/public artifacts are never
edited. The only recipe changes are output and B's actual source identities.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014')
PYTHON = '/root/miniconda3/envs/gta-asoca311/bin/python'
RECORDS = ROOT/'results/gaussian_region_balance_controller_20261014'
TRAINING = ROOT/'results/gta_region_balance200x125_20261014/training'
DIAGNOSTIC = ROOT/'results/initial_gaussian_parameter_gradients_amp_20261014'
DECISION = ROOT/'ops_inputs/B_TRAIN_DECISION_20261014.json'
ROOT_REVIEW = ROOT/'ops_inputs/region_balance_v1_root_review_20261014.txt'
DIAG_SOURCE = '18683db9538eaf9edf2b04b4f9c9b9dffb9913c5800741d0acd780d2a9d56c3e'
VERIFY_SOURCE = '698b349799cff927c9c8be2bf6330e0b2cbf4c1b9198ed8f4a303c6075b78e51'
VARIANT_PROTOCOL = 'a2905169196c75625772621b1f66f65e493f8bd90d02d98bf31e24547cef462a'
ARTIFACTS = {'diagnostic_plan.json', 'fixed_sampling_bank.json', 'reconstructed_initial_state.json', 'batch_metrics.jsonl', 'summary.json'}


def need(ok, reason):
    if not ok: raise ValueError(reason)


def sha(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): result.update(block)
    return result.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8'); temporary.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def dependencies():
    folder = Path(__file__).resolve().parent
    need(sha(folder/'diagnose_initial_gaussian_parameter_gradients.py') == DIAG_SOURCE, 'Actual initial diagnostic source changed')
    need(sha(folder/'verify_postfit.py') == VERIFY_SOURCE, 'Independent postfit verifier changed')
    diagnostic = load(folder/'diagnose_initial_gaussian_parameter_gradients.py', 'B_control_diagnostic')
    guard, prepare, _, _ = diagnostic.helpers()
    verify = load(folder/'verify_postfit.py', 'B_control_postfit_verifier')
    return diagnostic, guard, prepare, verify


def expected_config(main, diagnostic, inspection):
    result = copy.deepcopy(main)
    result.update(output=str(TRAINING), source_sha256=diagnostic.B_SOURCE,
                  source_files_sha256=inspection['source_files_sha256'])
    need(result['gaussian_weight'] == result['gaussian_schedule']['base'] == 10.
         and result['gaussian_late_weight'] is None and result['epochs'] == 200 and result['steps'] == 125
         and result['freeze_epochs'] == 50 and result['partial_until'] == 70 and result['no_ds'] is False,
         'Original recipe is not constant lambda10/200x125/50-20-130/DS')
    return result


def evidence(diagnostic, guard):
    guard.source_guard(ROOT)
    need(sha(ROOT/'dataset/cache/manifest.json') == guard.MANIFEST, 'Actual immutable dataset manifest changed')
    checker, protocol, inspection = diagnostic.variant_guard()
    need(sha(diagnostic.VARIANT/'protocol.json') == VARIANT_PROTOCOL, 'Sealed B protocol differs')
    canonical = diagnostic.canonical_guard(guard)
    main = read(ROOT/'results/gta_enhanced200x125/training/config.json')
    guard.config_guard(main, read(ROOT/'dataset/cache/manifest.json'))
    marker = read(DIAGNOSTIC/'DIAGNOSTIC_COMPLETE.json')
    status = read(DIAGNOSTIC/'CONTROLLER_STATUS.json')
    need(marker.get('status') == 'completed' and marker.get('n_batches') == 28 and marker.get('n_patches') == 56
         and marker.get('parameter_updates') is False and marker.get('held_out_GT_read') is False
         and marker.get('buffer_state_restored_at_end') is True, 'Complete actual initial AMP28-batch diagnostic is required')
    need(status.get('status') == 'completed' and status.get('gpu_lock_acquired') is False
         and status.get('child_pid') is None and status.get('script_sha256') == DIAG_SOURCE,
         'Initial diagnostic controller has not completed/released its GPU lease')
    need(set(marker.get('artifacts_sha256', {})) == ARTIFACTS, 'Initial diagnostic requires exact five artifacts')
    for name, expected in marker['artifacts_sha256'].items():
        need(sha(guard.safe_member(DIAGNOSTIC, name)) == expected, 'Initial diagnostic artifact changed: '+name)
    summary = read(DIAGNOSTIC/'summary.json'); plan = read(DIAGNOSTIC/'diagnostic_plan.json')
    need(summary.get('precision') == 'amp' and summary.get('n_batches') == 28 and summary.get('n_patches') == 56
         and summary.get('initial_state_sha256') == diagnostic.INITIAL and summary.get('historical_main_theta0_verified') is False
         and summary.get('parameter_updates') is False and summary.get('parameter_grad_populated') is False
         and plan.get('canonical') == canonical and plan['source_inspection']['source_sha256'] == diagnostic.B_SOURCE,
         'Initial diagnostic precision/source/initialization/canonical scope differs')
    fixture = read(DIAGNOSTIC/'cpu_fixture.json')
    need(fixture.get('status') == 'passed' and fixture.get('actual_candidate_torch_verified') is True
         and 'parameter_sum_and_group_metric_fixture' in fixture, 'Actual CPU B/parameter fixture is required')
    need(ROOT_REVIEW.is_file(), 'ROOT independent source/formula review must be uploaded')
    rows = guard.lines(DIAGNOSTIC/'batch_metrics.jsonl')
    need([row.get('batch') for row in rows] == list(range(28)), 'Initial gradient ledger is incomplete')
    expected = expected_config(main, diagnostic, inspection)
    donor_records = {}
    for component, expected_sha in diagnostic.DONORS.items():
        path = Path(main[component+'_init'])
        need(sha(path) == expected_sha, 'Original immutable donor differs: '+component)
        donor_records[component] = {'path': str(path), 'sha256': expected_sha, 'bytes': path.stat().st_size}
    proof = {'canonical': canonical, 'B_protocol_sha256': VARIANT_PROTOCOL, 'B_READY_sha256': sha(diagnostic.VARIANT/'VARIANT_READY.json'),
             'B_source_sha256': diagnostic.B_SOURCE, 'B_package_sha256': inspection['code_package_sha256'],
             'ROOT_variant_review_sha256': sha(ROOT_REVIEW), 'main_config_sha256': guard.CONFIG,
             'initial_gradient_complete_sha256': sha(DIAGNOSTIC/'DIAGNOSTIC_COMPLETE.json'),
             'initial_gradient_artifacts_sha256': marker['artifacts_sha256'], 'actual_CPU_fixture_sha256': sha(DIAGNOSTIC/'cpu_fixture.json'),
             'initial_gradient_controller_protocol_sha256': sha(DIAGNOSTIC/'CONTROLLER_PROTOCOL.json'),
             'initial_gradient_controller_status_sha256': sha(DIAGNOSTIC/'CONTROLLER_STATUS.json'), 'donors': donor_records}
    return checker, inspection, main, expected, summary, proof


def decision_guard(diagnostic, guard, fresh=True):
    checker, inspection, main, expected, summary, proof = evidence(diagnostic, guard)
    decision_sha = sha(DECISION); decision = read(DECISION)
    need(decision.get('schema_version') == 1 and decision.get('action') == 'authorize_one_B_formal_run'
         and decision.get('decision_author') == 'ROOT' and decision.get('output') == str(TRAINING), 'No explicit ROOT authorization for this one B run')
    need(decision.get('evidence') == proof, 'ROOT decision does not bind the exact actual canonical/gradient/fixture/source evidence')
    canonical_review = decision.get('canonical_review', {})
    need(canonical_review.get('all20_checkpoints_reviewed') is True
         and canonical_review.get('late_checkpoint_names') == ['epoch160', 'epoch170', 'epoch180', 'epoch190', 'final']
         and isinstance(canonical_review.get('rationale'), str) and bool(canonical_review['rationale'].strip()), 'ROOT must assess complete trajectory and late curves')
    gradient_review = decision.get('gradient_review', {})
    need(gradient_review.get('all28_batches_reviewed') is True
         and gradient_review.get('nonfinite_batches') == summary['nonfinite_batches']
         and gradient_review.get('batches_implying_clip12') == summary['batches_implying_clip12']
         and gradient_review.get('proceed_with_fixed_lambda10_after_risk_review') is True
         and isinstance(gradient_review.get('rationale'), str) and bool(gradient_review['rationale'].strip()),
         'ROOT must explicitly interpret nonfinite/clipping observations; finite gradients alone never authorize training')
    created = dt.datetime.fromisoformat(decision['created_at_utc'])
    need(created.tzinfo is not None, 'Decision time must include UTC/timezone')
    if fresh:
        age = (dt.datetime.now(dt.timezone.utc)-created).total_seconds()
        need(-300 <= age <= 21600, 'Decision must be freshly published within six hours')
        need(created.timestamp()+5 >= (DIAGNOSTIC/'CONTROLLER_STATUS.json').stat().st_mtime, 'Decision predates actual diagnostic completion/release')
    need(sha(DECISION) == decision_sha, 'Decision changed while being read')
    return checker, inspection, main, expected, proof, {'path': str(DECISION), 'sha256': decision_sha, 'decision': decision}


def actual_config_guard(checker, inspection, expected):
    need(read(TRAINING/'config.json') == expected, 'Actual B config differs beyond output/source identities')
    return checker.actual_config(read(checker.VARIANT/'protocol.json'), inspection,
        ROOT/'results/gta_enhanced200x125/training/config.json', TRAINING/'config.json', str(TRAINING))


def training_preflight(diagnostic, guard, verify, checker, inspection, expected):
    guard.source_guard(ROOT); diagnostic.variant_guard()
    bound = actual_config_guard(checker, inspection, expected)
    complete = read(TRAINING/'TRAINING_COMPLETE.json'); final_sha = sha(TRAINING/'final.pt')
    for key, value in {'epoch': 199, 'smoke': False, 'source_sha256': diagnostic.B_SOURCE, 'manifest_sha256': guard.MANIFEST,
        'trained_case_ids': expected['trained_case_ids'], 'final_sha256': final_sha, 'segmentation_updates': 25000,
        'classifier_pretraining_updates': 5120, 'checkpoint_selection': 'fixed final', 'internal_selection_valid': True}.items():
        need(complete.get(key) == value, 'B training completion differs: '+key)
    history = guard.lines(TRAINING/'history.jsonl')
    need([row.get('epoch') for row in history] == list(range(200)) and all(row.get('iterations') == 125 and verify.finite(row.get('loss'))
         and row.get('gaussian_lambda') == 10. for row in history)
         and [row.get('stage') for row in history] == ['frozen']*50+['partial']*20+['joint']*130, 'B history does not demonstrate fixed finite200x125/lambda10/stages')
    periodic = TRAINING/'checkpoints'
    need({path.name for path in periodic.glob('epoch*.pt')} == {f'epoch{i:03d}.pt' for i in range(10, 201, 10)}, 'B mandatory periodic inventory differs')
    checkpoints = []
    for name in verify.NAMES:
        path = TRAINING/'final.pt' if name == 'final' else periodic/(name+'.pt')
        epoch = 199 if name == 'final' else int(name[5:])-1
        value = sha(path); verify.checkpoint_audit(path, expected, epoch); need(sha(path) == value, 'B checkpoint changed during CPU audit')
        checkpoints.append({'path': str(path), 'epoch': epoch, 'name': name, 'checkpoint_sha256': value})
    verify.checkpoint_audit(periodic/'epoch200.pt', expected, 199)
    canonical = read(ROOT/'POSTFIT_VERIFIED.json')
    return {'schema_version': 1, 'status': 'preflight_verified', 'root': str(ROOT), 'training': str(TRAINING),
        'manifest': str((ROOT/'dataset/cache/manifest.json').resolve()), 'raw_root': canonical['raw_root'],
        'trajectory_identity': {'training_config_sha256': bound['actual_config_sha256'], 'manifest_sha256': guard.MANIFEST,
            'source_sha256': diagnostic.B_SOURCE, 'evaluation_sources': {name: sha(diagnostic.VARIANT/'code'/name) for name in sorted(guard.EVALUATORS)},
            'case_ids': guard.IDS, 'split': 'internal_diagnostic', 'smoke': False, 'checkpoints': checkpoints, 'sw_batch_size': 1,
            'checkpoint_selection': 'fixed final remains unchanged; intermediate weights are trajectory diagnostics',
            'formal_final_test': False, 'outer8_used_for_selection': False},
        'main_freeze_sha256': guard.FREEZE, 'B_protocol_sha256': VARIANT_PROTOCOL,
        'training_complete_sha256': sha(TRAINING/'TRAINING_COMPLETE.json'), 'history_sha256': sha(TRAINING/'history.jsonl'),
        'mandatory_checkpoints': 20, 'patients_per_checkpoint': 4, 'expected_case_evaluations': 80,
        'role': 'sealed B fixed-final/internal-four development; no new classifier pretraining or final-test claim'}


def worker():
    diagnostic, guard, prepare, _ = dependencies()
    need(read(RECORDS/'STATUS.json')['controller_pid'] == os.getppid()
         and read(RECORDS/'STATUS.json')['controller_sha256'] == sha(__file__), 'Worker requires its owning live controller')
    checker, inspection, main, expected, proof, decision = decision_guard(diagnostic, guard)
    need(read(RECORDS/'PROTOCOL.json')['decision_sha256'] == decision['sha256'], 'Worker decision differs from owned launch protocol')
    sys.path.insert(0, str(diagnostic.VARIANT/'code'))
    import torch
    import train
    need(train.source_hash() == diagnostic.B_SOURCE, 'Worker imported another training source')
    factory = train.make_model; calls = []
    def instrumented_factory(*args, **kwargs):
        need(not calls, 'Unexpected multiple model construction'); calls.append(True)
        model = factory(*args, **kwargs)
        def before_first_forward(actual, inputs):
            handle.remove()
            bound = actual_config_guard(checker, inspection, expected)
            need(tuple(inputs[0].shape) == (2, 1, 128, 128, 128), 'Actual first-forward batch/patch differs')
            cpu_rng = torch.get_rng_state().clone(); cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
            state = {name: value.detach().cpu().clone() for name, value in actual.state_dict().items()}
            class Snapshot:
                def state_dict(self): return state
                def parameters(self): return actual.parameters()
            identity = prepare.state_identity(Snapshot())
            need(identity['state_sha256'] == diagnostic.INITIAL and identity['tensor_count'] == 327, 'Actual B initial327 state differs from audited reconstruction')
            path = RECORDS/'actual_B_theta0_before_first_forward.pt'; need(not path.exists(), 'B theta0 snapshot already exists')
            temporary = path.with_suffix('.pt.tmp'); torch.save({'model': state, 'rng_cpu': cpu_rng, 'rng_cuda': cuda_rng}, temporary); temporary.replace(path)
            need(torch.equal(cpu_rng, torch.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
                 and all(torch.equal(state[name], value.detach().cpu()) for name, value in actual.state_dict().items()), 'Readonly initial snapshot changed model/RNG')
            write(RECORDS/'ACTUAL_B_THETA0.json', {'role': 'actual B model after donors before any forward/update; private state',
                'captured_at_utc': guard.now(), 'state': identity, 'snapshot': str(path), 'snapshot_sha256': sha(path),
                'actual_config_binding': bound, 'source_sha256': diagnostic.B_SOURCE, 'controller_sha256': sha(__file__),
                'model_state_and_RNG_unchanged': True, 'network_forwards_before_capture': 0, 'optimizer_updates': 0,
                'historical_main_theta0_verified': False, 'CPU_RNG_sha256': __import__('hashlib').sha256(cpu_rng.numpy().tobytes()).hexdigest(),
                'CUDA_RNG_sha256': [__import__('hashlib').sha256(x.numpy().tobytes()).hexdigest() for x in cuda_rng],
                'note': 'Initial state matches audited reconstruction. Actual historical main theta0/RNG was not saved; DataLoader may consume CPU RNG before this capture.'})
        handle = model.register_forward_pre_hook(before_first_forward); return model
    train.make_model = instrumented_factory
    argv = read(RECORDS/'train_argv.json')['argv']
    need(argv == prepare.cli_argv(expected, diagnostic.VARIANT/'code', PYTHON, diagnostic.VARIANT/'code'), 'Worker training argv differs from the exact authorized B recipe')
    sys.argv = [argv[2]]+argv[3:]; train.main()
    need((RECORDS/'ACTUAL_B_THETA0.json').is_file(), 'No actual pre-forward B state captured')


def plan():
    return {'schema_version': 1, 'status': 'PLAN_ONLY_NOT_AUTHORIZATION_NOT_MODEL_RESULTS', 'ROOT': ROOT.as_posix(),
        'decision': DECISION.as_posix(), 'records': RECORDS.as_posix(), 'training': TRAINING.as_posix(), 'Python': PYTHON,
        'required_helper_sha256': {'diagnose_initial_gaussian_parameter_gradients.py': DIAG_SOURCE, 'verify_postfit.py': VERIFY_SOURCE},
        'B_protocol_sha256': VARIANT_PROTOCOL, 'allowed_config_changes': ['output', 'source_sha256', 'source_files_sha256'],
        'Gaussian_lambda': 10, 'epochs': 200, 'updates_per_epoch': 125, 'classifier_pretraining_updates_reused': 5120,
        'internal4_checkpoint_evaluations': 20, 'case_inferences': 80, 'checkpoint_selection': 'fixed final',
        'historical_main_theta0_verified': False, 'no_decision_no_training': True,
        'decision_contract': {'schema_version': 1, 'action': 'authorize_one_B_formal_run', 'decision_author': 'ROOT',
            'created_at_utc': 'fresh ISO8601 timezone-aware, after actual diagnostic release, within six hours',
            'output': TRAINING.as_posix(), 'evidence': 'exact evidence JSON printed by --decision-evidence after actual completion',
            'canonical_review': {'all20_checkpoints_reviewed': True, 'late_checkpoint_names': ['epoch160', 'epoch170', 'epoch180', 'epoch190', 'final'], 'rationale': 'ROOT actual complete/late curve interpretation'},
            'gradient_review': {'all28_batches_reviewed': True, 'nonfinite_batches': 'exact actual summary list',
                'batches_implying_clip12': 'exact actual summary dict', 'proceed_with_fixed_lambda10_after_risk_review': True,
                'rationale': 'ROOT explicit nonfinite/clipping assessment, not merely gradient finite'}}}


def run(args):
    import fcntl
    need(Path(__file__).resolve().parents[1] == ROOT and Path(sys.executable).absolute() == Path(PYTHON), 'Use actual named NEW ROOT/Python')
    need(not RECORDS.exists() and not TRAINING.exists(), 'Fresh-only B run; inspect any existing controller/output before continuation')
    diagnostic, guard, prepare, verify = dependencies()
    own = (ROOT/'.gaussian_region_balance_formal.lock').open('a+'); fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    RECORDS.mkdir(parents=True)
    state = {'status': 'waiting_ROOT_decision', 'controller_pid': os.getpid(), 'controller_sha256': sha(__file__),
             'started_at_utc': guard.now(), 'gpu_lock_acquired': False, 'children': []}
    def save(**value): state.update(value, updated_at_utc=guard.now()); write(RECORDS/'STATUS.json', state)
    cancellation = guard.OwnedCancellation(); child = None; lease = None
    signal.signal(signal.SIGTERM, cancellation.request); signal.signal(signal.SIGINT, cancellation.request)
    env = {key: value for key, value in os.environ.items() if not key.startswith('GTA_') and key not in ('OUTPUT_ROOT', 'CLS_CHECKPOINT', 'PYTHONPATH')}
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    def execute(label, argv, limit):
        nonlocal child
        record = {'stage': label, 'argv': argv, 'started_at_utc': guard.now(), 'log': str(RECORDS/(label+'.log'))}; state['children'].append(record)
        began = time.monotonic()
        with (RECORDS/(label+'.log')).open('x', encoding='utf-8') as log:
            cancellation.launching = True
            try:
                child = subprocess.Popen(argv, cwd=diagnostic.VARIANT/'code', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record.update(pid=child.pid, pgid=os.getpgid(child.pid)); need(record['pgid'] == child.pid, 'Owned B child group differs')
            finally: cancellation.launching = False
            cancellation.raise_pending()
            while child.poll() is None:
                need(time.monotonic()-began < limit, 'B stage deadline exceeded: '+label)
                save(status=label, child_pid=child.pid, stage_wall_seconds=time.monotonic()-began, gpu_lock_acquired=lease is not None); time.sleep(20)
        record.update(returncode=child.returncode, wall_seconds=time.monotonic()-began, completed_at_utc=guard.now()); child = None
        need(record['returncode'] == 0, 'Owned B stage failed: '+label); save(status=label+'_exited', child_pid=None)
    try:
        save()
        deadline = time.monotonic()+args.max_wait_seconds
        while not DECISION.is_file():
            need(time.monotonic() < deadline, 'Timed out waiting for explicit ROOT decision'); save(); time.sleep(15)
        checker, inspection, main, expected, proof, decision = decision_guard(diagnostic, guard)
        write(RECORDS/'expected_B_config.json', expected)
        argv = prepare.cli_argv(expected, diagnostic.VARIANT/'code', PYTHON, diagnostic.VARIANT/'code')
        write(RECORDS/'train_argv.json', {'argv': argv})
        runtime = guard.runtime_profile()
        need(os.getpriority(os.PRIO_PROCESS, 0) == 0 and len(os.sched_getaffinity(0)) >= 8, 'Formal B must not inherit two-CPU/nice10 diagnostic limits')
        while True:
            need(time.monotonic() < deadline, 'Timed out waiting for released diagnostic/idle GPU')
            first = guard.gpu_snapshot()
            if first['all_compute_GPUs_idle']:
                candidate = (ROOT/'.segmentation_stage.lock').open('a+')
                try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: candidate.close()
                else:
                    lease = candidate
                    diagnostic_lock = (ROOT/'.initial_gaussian_parameter_diagnostic.lock').open('a+')
                    try: fcntl.flock(diagnostic_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        diagnostic_lock.close(); fcntl.flock(lease, fcntl.LOCK_UN); lease.close(); lease = None
                    else:
                        fcntl.flock(diagnostic_lock, fcntl.LOCK_UN); diagnostic_lock.close(); time.sleep(3)
                        second = guard.gpu_snapshot()
                        if second['all_compute_GPUs_idle']: break
                        fcntl.flock(lease, fcntl.LOCK_UN); lease.close(); lease = None
            save(status='waiting_released_diagnostic_and_idle_GPU', GPU=first, gpu_lock_acquired=False); time.sleep(15)
        recheck = decision_guard(diagnostic, guard)
        need(recheck[4:] == (proof, decision) and recheck[3] == expected, 'ROOT decision/evidence changed before B launch')
        prior_GPU = read(DIAGNOSTIC/'CONTROLLER_PROTOCOL.json')['GPU_idle_observations'][-1]['gpu']
        need(first['gpu'] == second['gpu'] == prior_GPU, 'NEW diagnostic and formal B hardware/driver differ; require a revised plan')
        need(guard.runtime_profile() == runtime and not TRAINING.exists(), 'Runtime changed or B output is not fresh')
        protocol = plan() | {'status': 'actual_ROOT_decision_authorized_formal_B', 'decision_sha256': decision['sha256'],
            'decision': decision, 'evidence': proof, 'controller_sha256': state['controller_sha256'], 'runtime': runtime,
            'GPU_idle_observations': [first, second], 'training_argv': argv, 'expected_B_config_sha256': sha(RECORDS/'expected_B_config.json'),
            'actual_theta0_capture': 'readonly pre-first-forward all327state+CPU/CUDA RNG; historical main remains unknown'}
        write(RECORDS/'PROTOCOL.json', protocol)
        save(status='ready_to_launch_formal_B', gpu_lock_acquired=True, decision_sha256=decision['sha256'])
        execute('formal_training', [PYTHON, '-u', str(Path(__file__).absolute()), '--worker'], args.max_training_seconds)
        initial = read(RECORDS/'ACTUAL_B_THETA0.json')
        need(initial['state']['state_sha256'] == diagnostic.INITIAL and initial['model_state_and_RNG_unchanged'] is True
             and initial['historical_main_theta0_verified'] is False and sha(initial['snapshot']) == initial['snapshot_sha256'], 'Actual B theta0 snapshot evidence differs')
        bound = actual_config_guard(checker, inspection, expected); write(RECORDS/'ACTUAL_B_CONFIG_BINDING.json', bound)
        before = training_preflight(diagnostic, guard, verify, checker, inspection, expected); write(RECORDS/'POSTTRAIN_PREFLIGHT.json', before)
        trajectory = TRAINING.parent/'internal4_trajectory'; curves = TRAINING.parent/'learning_curves'
        execute('internal4_trajectory', [PYTHON, '-u', str(diagnostic.VARIANT/'code/evaluate_trajectory.py'), '--training', str(TRAINING),
            '--manifest', expected['manifest'], '--raw-root', before['raw_root'], '--output', str(trajectory), '--python', PYTHON, '--sw-batch-size', '1'], args.max_trajectory_seconds)
        execute('plot_curves', [PYTHON, str(ROOT/'ops/plot_learning_curves.py'), '--run', 'development='+str(TRAINING),
            '--validation', 'development='+str(trajectory/'validation_history.jsonl'), '--output', str(curves)], 600)
        # Reuse the complete698 verifier body. Only its hardcoded A path/source
        # preflight is replaced in RAM by our exact sealed-B preflight; no bytes
        # are edited and actual evaluator/metric source files remain identical.
        verify.preflight = lambda root, train, manifest, raw: training_preflight(diagnostic, guard, verify, checker, inspection, expected)
        verified = verify.postflight(ROOT, TRAINING, ROOT/'dataset/cache/manifest.json', Path(before['raw_root']), before)
        write(RECORDS/'POSTFIT_VERIFIED.json', verified)
        final_check = decision_guard(diagnostic, guard, fresh=False)
        need(final_check[4:] == (proof, decision) and sha(__file__) == state['controller_sha256'], 'Immutable authorization/source/evidence changed during B run')
        complete = protocol | {'status': 'completed', 'completed_at_utc': guard.now(), 'children': state['children'],
            'actual_B_config_binding_sha256': sha(RECORDS/'ACTUAL_B_CONFIG_BINDING.json'), 'actual_B_theta0_receipt_sha256': sha(RECORDS/'ACTUAL_B_THETA0.json'),
            'training_complete_sha256': sha(TRAINING/'TRAINING_COMPLETE.json'), 'fixed_final_sha256': sha(TRAINING/'final.pt'),
            'B_postfit_verified_sha256': sha(RECORDS/'POSTFIT_VERIFIED.json'), 'trajectory_complete_sha256': sha(trajectory/'TRAJECTORY_COMPLETE.json'),
            'verified_checkpoint_evaluations': 20, 'verified_case_evaluations': 80, 'A_canonical_unchanged': True}
        write(RECORDS/'CONTROL_COMPLETE.json', complete); save(status='completed', child_pid=None, completion_sha256=sha(RECORDS/'CONTROL_COMPLETE.json'))
    except BaseException as error:
        cancellation.cleaning = True; guard.cleanup_owned_child(child); child = None
        save(status='failed_stop', child_pid=None, error=repr(error), owned_child_exit_confirmed=True); raise
    finally:
        cancellation.cleaning = True; guard.cleanup_owned_child(child)
        if lease is not None: fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
        fcntl.flock(own, fcntl.LOCK_UN); own.close()
        try: save(gpu_lock_acquired=False, owned_child_exit_confirmed=True, locks_released_at_utc=guard.now())
        except Exception: pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', action='store_true'); p.add_argument('--decision-evidence', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--max-wait-seconds', type=int, default=7200)
    p.add_argument('--max-training-seconds', type=int, default=43200); p.add_argument('--max-trajectory-seconds', type=int, default=43200)
    args = p.parse_args()
    need(min(args.max_wait_seconds, args.max_training_seconds, args.max_trajectory_seconds) > 0, 'Budgets must be positive')
    if args.plan: print(json.dumps(plan(), indent=2)); return
    need(sys.platform == 'linux', 'Only named NEW Linux host is executable; --plan is local/read-only')
    if args.decision_evidence:
        diagnostic, guard, _, _ = dependencies()
        print(json.dumps({'role': 'ACTUAL_EVIDENCE_ONLY_NOT_TRAIN_AUTHORIZATION', 'evidence': evidence(diagnostic, guard)[-1]}, indent=2)); return
    if args.worker: worker()
    else: run(args)


if __name__ == '__main__': main()
