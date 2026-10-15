"""Bounded, owned OLD-server fixed-final preview; never trains/selects a model.

Waiting owns only its preview lock. The shared segmentation/GPU lock is acquired
non-blockingly immediately before frozen evaluate.py, then held until postchecks.
This utility is outside the frozen code/command package. --plan is read-only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OLD_ROOT = Path('/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2')
NEW_ROOT = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
SHARED = Path('/root/autodl-fs/experiments/GTA_ASOCA_Fidelity_20261014/early_final_mirror_20261014')
SOURCE = 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
CONFIG = 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
MANIFEST = '141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
FREEZE = '9b294024e33b0d0aaa12b473be53dd3a81dfcd9dd020446948fe667405c7ff46'
PACKAGE = '0d14f942a7d87c81d272cca0a1b3b55c458aa0da0e0da01f6b15456107e7c48b'
SOURCE_ALGORITHM = 'gta_fidelity_selected_python_path_and_bytes_v1'
IDS = ['Diseased_10', 'Diseased_20', 'Normal_12', 'Normal_3']
FILES = {'final.pt', 'config.json', 'history.jsonl', 'TRAINING_COMPLETE.json',
         'manifest.json', 'input_runtime_audit.json'}
EVALUATORS = {'evaluate.py', 'metrics.py', 'evaluate_trajectory.py'}
PACKAGES = {'torch': '2.7.1+cu128', 'monai': '1.5.1', 'numpy': '1.26.4', 'scipy': '1.15.3',
            'scikit-image': '0.25.2', 'SimpleITK': '2.5.2', 'scikit-learn': '1.7.2',
            'pandas': '2.3.3', 'tqdm': '4.67.1'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def safe_member(base, name):
    path = (base/name).resolve()
    require(path.is_relative_to(base.resolve()) and path.is_file(), 'File escapes or is missing: ' + name)
    return path


def source_guard(root):
    freeze_path = root/'configs/main_source_freeze.json'
    require(sha(freeze_path) == FREEZE, 'Main freeze identity differs')
    freeze = read(freeze_path)
    code = root/'code'
    paths = {p.relative_to(code).as_posix(): p for p in code.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and '.git' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    require(len(paths) == 28 and set(paths) == set(freeze['code_files']), 'Full code package membership differs')
    full = hashlib.sha256()
    for name in sorted(paths):
        path = safe_member(code, name)
        require(file_record(path) == freeze['code_files'][name], 'Frozen code changed: ' + name)
        full.update(name.encode()); full.update(path.read_bytes())
    require(full.hexdigest() == freeze['code_package_sha256'] == PACKAGE, 'Full code package digest differs')
    for name, expected in freeze['command_files'].items():
        require(file_record(safe_member(root, name)) == expected, 'Frozen command changed: ' + name)
    training = hashlib.sha256()
    for name, expected in sorted(freeze['training_dependency_files'].items()):
        path = safe_member(code, name)
        require(sha(path) == expected, 'Training dependency changed: ' + name)
        training.update(name.encode()); training.update(path.read_bytes())
    require(training.hexdigest() == freeze['training_source_sha256'] == SOURCE, 'BC8 training source differs')
    return freeze


def ready_structure(ready):
    require(ready.get('schema_version') == 1 and ready.get('status') == 'fixed_final_payload_ready', 'READY status/schema differs')
    require(ready.get('role') == 'early fixed-final inference mirror; no new training', 'READY role differs')
    require(ready.get('source_root') == NEW_ROOT and ready.get('case_ids') == IDS, 'READY origin/patients differ')
    require(set(ready.get('files', {})) == FILES, 'READY requires exactly six filenames-only payload files')
    for name, item in ready['files'].items():
        require(isinstance(item, dict) and isinstance(item.get('bytes'), int) and not isinstance(item['bytes'], bool)
                and item['bytes'] > 0 and isinstance(item.get('sha256'), str) and len(item['sha256']) == 64
                and all(c in '0123456789abcdef' for c in item['sha256']), 'Invalid READY file record: ' + name)
    require(ready.get('config_sha256') == CONFIG and ready['files']['config.json']['sha256'] == CONFIG, 'Canonical config digest differs')
    require(ready.get('manifest_sha256') == MANIFEST and ready['files']['manifest.json']['sha256'] == MANIFEST, 'Canonical manifest digest differs')
    require(ready.get('source_sha256') == SOURCE and ready.get('source_freeze_sha256') == FREEZE
            and ready.get('source_guard_full_package_verified') is True, 'READY source guard identity differs')
    require(ready.get('checkpoint_epoch') == 199 and ready.get('checkpoint_sha256') == ready['files']['final.pt']['sha256'], 'READY final identity differs')
    require(set(ready.get('evaluation_sources', {})) == EVALUATORS, 'READY evaluator set differs')
    require(ready.get('packages') == PACKAGES, 'READY actual runtime nine versions differ')
    require(set(ready.get('cache_CT_inputs', {})) == set(IDS) and set(ready.get('raw_GT_source_sha256', {})) == set(IDS), 'READY CT/GT cohort differs')


def config_guard(config, manifest):
    require(config.get('method') == 'gta' and config.get('smoke') is False and config.get('fold') == 0, 'Not formal GTA fold0')
    require(config.get('source_hash_algorithm') == SOURCE_ALGORITHM and config.get('source_sha256') == SOURCE
            and config.get('manifest_sha256') == MANIFEST, 'GTA source/manifest config differs')
    require(config.get('epochs') == 200 and config.get('steps') == 125 and config.get('segmentation_updates') == 25000
            and config.get('patch_size') == 128 and config.get('no_ds') is False, 'Final budget/patch/DS differs')
    require(config.get('output') == NEW_ROOT+'/results/gta_enhanced200x125/training', 'Original GTA training output lineage differs')
    require(config.get('manifest') == NEW_ROOT+'/dataset/cache/manifest.json', 'Original GTA manifest lineage differs')
    require(config.get('checkpoint_selection') == 'fixed final', 'Final selection role differs')
    require(config.get('cohort') == 'internal28' and config.get('internal_selection_valid') is True
            and config.get('outer8_used_for_selection') is False and config.get('allow_selection_overlap') is False, 'Patient isolation declarations differ')
    split = config.get('internal_split', {})
    train, val = split.get('train', []), split.get('val', [])
    require(len(train) == len(set(train)) == 28 and val == IDS and config.get('trained_case_ids') == train, 'GTA trained28/internal4 identity differs')
    outer = manifest['folds'][0]
    require(not set(train) & set(val) and set(train) | set(val) == set(outer['train'])
            and not set(train) & set(outer['val']), 'GTA patient-domain partition differs')
    return train


def final_header(saved, config, torch_module):
    require(isinstance(saved, dict) and saved.get('epoch') == 199 and saved.get('config') == config, 'Actual final checkpoint epoch/config differs')
    require(saved.get('checkpoint_role') in (None, 'fixed final'), 'Actual checkpoint role differs')
    state = saved.get('model')
    require(isinstance(state, dict) and bool(state), 'Actual final lacks model tensors')
    for name, value in state.items():
        require(torch_module.is_tensor(value), 'Non-tensor model state: ' + name)
        require(str(value.device) == 'cpu' and bool(torch_module.isfinite(value).all()), 'Nonfinite/non-CPU model state: ' + name)
    return {'epoch': 199, 'model_state_tensor_count': len(state), 'all_model_tensors_finite': True,
            'load_mode': 'torch.load(map_location=cpu, weights_only=True)', 'CUDA_context_initialized': torch_module.cuda.is_initialized()}


def unet_guard(root):
    marker = read(root/'UNET_CALIBRATION_COMPLETE.json')
    require(marker.get('training_source_sha256') == SOURCE and marker.get('manifest_sha256') == MANIFEST
            and marker.get('main_freeze_manifest_sha256') == FREEZE and marker.get('root') == str(root)
            and marker.get('controller_sha256') == '54fd789b35ad14c56ed74b7fb3f14b1f8a794d1be995fba968da99d1c94bd807'
            and marker.get('formal_segmentation_updates') == 25000 and marker.get('fivefold') is False, 'U-Net complete marker identity differs')
    training = root/'results/unet_recipe200x125/training'
    trajectory = root/'results/unet_recipe200x125/internal4_trajectory'
    require(sha(training/'TRAINING_COMPLETE.json') == marker['formal_training_marker_sha256']
            and sha(training/'final.pt') == marker['formal_final_sha256']
            and sha(trajectory/'TRAJECTORY_COMPLETE.json') == marker['trajectory_complete_sha256']
            and sha(trajectory/'validation_history.jsonl') == marker['validation_history_sha256'], 'U-Net final/complete/trajectory bytes changed')
    train_marker = read(training/'TRAINING_COMPLETE.json')
    require(train_marker.get('epoch') == 199 and train_marker.get('smoke') is False
            and train_marker.get('segmentation_updates') == 25000 and train_marker.get('final_sha256') == marker['formal_final_sha256'], 'U-Net formal budget/completion differs')
    complete = read(trajectory/'TRAJECTORY_COMPLETE.json')
    rows = lines(trajectory/'validation_history.jsonl')
    names = [f'epoch{i:03d}' for i in range(10, 200, 10)]+['final']
    require(complete.get('n_evaluations') == 20 and complete.get('case_ids') == IDS and complete.get('sw_batch_size') == 1
            and [r.get('name') for r in rows] == names
            and [r.get('epoch') for r in rows] == list(range(9, 190, 10))+[199]
            and all(r.get('n_cases') == 4 and r.get('smoke') is False for r in rows), 'U-Net complete20 trajectory coverage differs')
    return {'complete_marker_sha256': sha(root/'UNET_CALIBRATION_COMPLETE.json'),
            'trajectory_complete_sha256': marker['trajectory_complete_sha256'], 'n_evaluations': 20}


def runtime_profile():
    require(sys.version.split()[0] == '3.11.17', 'Requires Python3.11.17')
    actual = {name: metadata.version(name) for name in PACKAGES}
    require(actual == PACKAGES, 'Isolated runtime package versions differ')
    import torch, monai, numpy, scipy, skimage, SimpleITK, sklearn, pandas, tqdm
    modules = {'torch': torch, 'monai': monai, 'numpy': numpy, 'scipy': scipy, 'scikit-image': skimage,
               'scikit-learn': sklearn, 'pandas': pandas, 'tqdm': tqdm}
    for name, module in modules.items():
        require(module.__version__ == PACKAGES[name], 'Imported package differs from metadata: ' + name)
    require(SimpleITK.Version_VersionString() == PACKAGES['SimpleITK'], 'Imported SimpleITK differs')
    require(not torch.cuda.is_initialized(), 'Guard imports unexpectedly initialized CUDA')
    return {'python': sys.version, 'python_executable': sys.executable, 'packages': actual,
            'torch_cuda_runtime': torch.version.cuda, 'torch_cudnn_version': torch.backends.cudnn.version(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'CUDA_context_initialized': torch.cuda.is_initialized()}


def gpu_snapshot():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True, timeout=30).strip()
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader'], text=True, timeout=30).strip()
    require(bool(hardware), 'No observable GPU hardware')
    return {'captured_at_utc': now(), 'gpu': hardware, 'gpu_processes': processes, 'all_compute_GPUs_idle': not processes}


def payload_guard(root, shared, check_header=True):
    ready_path = shared/'READY.json'
    ready_sha = sha(ready_path)
    ready = read(ready_path); ready_structure(ready)
    actual_files = {}
    for name in sorted(FILES):
        actual_files[name] = file_record(safe_member(shared, name))
        require(actual_files[name] == ready['files'][name], 'Shared payload SHA/size differs: ' + name)
    freeze = source_guard(root)
    eval_sources = {name: sha(root/'code'/name) for name in sorted(EVALUATORS)}
    require(eval_sources == ready['evaluation_sources'], 'Local evaluators differ from actual canonical evaluator')
    require(sha(root.parent/'cache/manifest.json') == MANIFEST, 'OLD manifest differs')
    manifest = read(shared/'manifest.json'); config = read(shared/'config.json')
    train_ids = config_guard(config, manifest)
    require(config.get('source_files_sha256') == freeze['training_dependency_files'], 'Config dependency identities differ')
    marker = read(shared/'TRAINING_COMPLETE.json')
    require(marker.get('epoch') == 199 and marker.get('smoke') is False and marker.get('segmentation_updates') == 25000
            and marker.get('classifier_pretraining_updates') == 5120 and marker.get('checkpoint_selection') == 'fixed final'
            and marker.get('internal_selection_valid') is True and marker.get('source_sha256') == SOURCE
            and marker.get('manifest_sha256') == MANIFEST and marker.get('trained_case_ids') == train_ids
            and marker.get('final_sha256') == ready['checkpoint_sha256'], 'Canonical TRAINING_COMPLETE differs')
    history = lines(shared/'history.jsonl')
    require([r.get('epoch') for r in history] == list(range(200))
            and all(r.get('iterations') == 125 and isinstance(r.get('loss'), (int, float))
                    and math.isfinite(r['loss']) for r in history), 'Canonical history does not complete200x125')
    require([r.get('stage') for r in history] == ['frozen']*50+['partial']*20+['joint']*130, 'Canonical training stages differ')
    audit = read(shared/'input_runtime_audit.json')
    require(audit.get('python', '').split()[0] == '3.11.17' and audit.get('packages') == ready['packages']
            and audit.get('manifest_sha256') == MANIFEST and audit.get('training_source_sha256') == SOURCE
            and audit.get('evaluation_sources') == eval_sources and audit.get('cache_CT_inputs') == ready['cache_CT_inputs'], 'Canonical runtime audit differs from READY')
    cases = {case['case_id']: case for case in manifest['cases']}
    ct_records = {}; label_records = {}
    import numpy as np
    for case_id in IDS:
        case = cases[case_id]
        image = safe_member(root.parent/'cache', case['cache_subdir']+'/image.npy')
        ct_records[case_id] = file_record(image)
        require(ct_records[case_id] == ready['cache_CT_inputs'][case_id], 'OLD complete cached CT differs: ' + case_id)
        array = np.load(image, mmap_mode='r')
        require(list(array.shape) == case['shape_zyx'] and array.dtype == np.dtype('float16'), 'Cached CT shape/dtype differs: ' + case_id)
        require(all(math.isclose(x, .5, abs_tol=1e-12) for x in case['spacing_zyx_mm']), 'Cache spacing differs')
        label = safe_member(root.parent/'raw/CompressedVersion', case['label_relative_path'])
        label_records[case_id] = file_record(label)
        require(label_records[case_id]['sha256'] == case['source_sha256']['label'] == ready['raw_GT_source_sha256'][case_id], 'Original raw GT differs: ' + case_id)
    header = None
    if check_header:
        import torch
        checkpoint = torch.load(shared/'final.pt', map_location='cpu', weights_only=True)
        header = final_header(checkpoint, config, torch)
        require(header['CUDA_context_initialized'] is False, 'CPU checkpoint guard initialized CUDA')
        del checkpoint
    require(sha(ready_path) == ready_sha, 'READY changed while auditing')
    for name in FILES:
        require(file_record(shared/name) == actual_files[name], 'Shared file changed while auditing: ' + name)
    return {'READY_sha256': ready_sha, 'files': actual_files, 'ready': ready, 'config': config,
            'manifest': manifest, 'runtime_audit': audit, 'evaluation_sources': eval_sources,
            'cache_CT_inputs': ct_records, 'raw_GT': label_records, 'actual_final_CPU_header': header}


def evaluation_argv(root, shared, output):
    return [str(root/'early_final_env311/bin/python'), '-u', str(root/'code/evaluate.py'),
            '--manifest', str(root.parent/'cache/manifest.json'), '--raw-root', str(root.parent/'raw/CompressedVersion'),
            '--checkpoint', str(shared/'final.pt'), '--output', str(output), '--fold', '0',
            '--split', 'internal_diagnostic', '--sw-batch-size', '1']


def verify_result(root, output, audit):
    sys.path.insert(0, str(root/'code'))
    from evaluate_trajectory import verify_evaluation
    from evaluate import summarize, verify_grid
    import numpy as np
    import SimpleITK as sitk
    ready, config, manifest = audit['ready'], audit['config'], audit['manifest']
    cases = {c['case_id']: c for c in manifest['cases']}
    summary = verify_evaluation(output, IDS, ready['checkpoint_sha256'], MANIFEST, config,
                                audit['evaluation_sources'], cases, 199)
    require(summary.get('checkpoint_selection') == 'fixed final' and summary.get('formal_final_test') is False
            and summary.get('internal_selection_valid') is True, 'Preview final/diagnostic role differs')
    rows = [read(output/(case_id+'.json')) for case_id in IDS]
    ledger = lines(output/'case_metrics.jsonl')
    require(ledger == rows, 'Preview ledger differs from case files')
    for group, selected in [('overall', rows), ('Normal', [r for r in rows if r['group'] == 'Normal']),
                            ('Diseased', [r for r in rows if r['group'] == 'Diseased'])]:
        require(summary[group] == summarize(selected), 'Preview summary does not reproduce: ' + group)
    predictions = {}
    from metrics import ASSD_POOLED_DEFINITION, ASSD_DIRECTIONAL_DEFINITION, METRIC_PROTOCOL_VERSION
    for case_id, row in zip(IDS, rows):
        require(row.get('inference_patch_size') == [128]*3 and row.get('inference_sw_batch_size') == 1
                and row.get('checkpoint_selection') == 'fixed final' and row.get('formal_final_test') is False
                and row.get('internal_selection_valid') is True and row.get('metric_protocol_version') == METRIC_PROTOCOL_VERSION
                and row.get('assd_definition') == ASSD_POOLED_DEFINITION
                and row.get('assd_directional_mean_definition') == ASSD_DIRECTIONAL_DEFINITION, 'Preview inference/metric role differs')
        prediction_path = output/(case_id+'_prediction.nrrd')
        predicted = sitk.ReadImage(str(prediction_path))
        reference = sitk.ReadImage(str(root.parent/'raw/CompressedVersion'/cases[case_id]['label_relative_path']))
        verify_grid(predicted, reference)
        array = sitk.GetArrayFromImage(predicted)
        require(set(np.unique(array).tolist()) <= {0, 1}, 'Prediction is not binary')
        truth = sitk.GetArrayFromImage(reference) > 0
        positive = array > 0
        pn, tn = int(positive.sum()), int(truth.sum())
        inter = int(np.count_nonzero(positive & truth))
        require(row.get('empty_prediction') == (pn == 0), 'Empty-prediction metadata differs from pixels')
        for name, expected in {'dice': 2*inter/max(pn+tn, 1), 'precision': inter/max(pn, 1), 'recall': inter/max(tn, 1)}.items():
            require(isinstance(row.get(name), (int, float)) and math.isclose(row[name], expected, rel_tol=1e-12, abs_tol=1e-12),
                    'Saved-pixel overlap differs: ' + name)
        if pn and tn:
            require(isinstance(row.get('assd_mm'), (int, float)) and math.isfinite(row['assd_mm']) and row['assd_mm'] >= 0,
                    'Nonempty prediction lacks valid ASSD')
        else:
            require(row.get('assd_mm') is None, 'Empty-surface ASSD must remain undefined')
        predictions[case_id] = {'file_sha256': sha(prediction_path), 'decoded_pixel_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
                                'dtype': str(array.dtype), 'shape_zyx': list(array.shape),
                                'size_xyz': list(predicted.GetSize()), 'spacing_xyz_mm': list(predicted.GetSpacing()),
                                'origin_xyz_mm': list(predicted.GetOrigin()), 'direction': list(predicted.GetDirection()),
                                'foreground_voxels': int(np.count_nonzero(array))}
    return {'summary_sha256': sha(output/'summary.json'), 'case_metrics_sha256': sha(output/'case_metrics.jsonl'),
            'predictions': predictions, 'n_cases': 4, 'case_ids': IDS, 'checkpoint_epoch': 199,
            'checkpoint_sha256': ready['checkpoint_sha256'], 'manifest_sha256': MANIFEST,
            'evaluation_sources': audit['evaluation_sources'], 'primary_metrics': ['dice', 'assd_mm']}


class OwnedCancellation:
    """Defer Python cancellation while acquiring or cleaning child ownership."""
    def __init__(self):
        self.launching = False
        self.cleaning = False
        self.pending = []

    def request(self, signum, _frame=None):
        self.pending.append(int(signum))
        if not self.launching and not self.cleaning:
            self.raise_pending()

    def raise_pending(self):
        if self.pending and not self.launching and not self.cleaning:
            # Make repeated signal delivery non-raising before unwinding into
            # the outer except/finally cleanup, including that entry window.
            self.cleaning = True
            raise KeyboardInterrupt('Owned preview cancellation signals ' + str(self.pending))


def stop_child(child, on_wait=None):
    if child is None or child.poll() is not None:
        return
    # Popen owns this still-live child; start_new_session made its PID the PGID.
    try:
        require(os.getpgid(child.pid) == child.pid, 'Refusing signal: own child process group differs')
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=15)
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            while child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if on_wait:
                        on_wait()
    require(child.poll() is not None, 'Owned evaluator exit is not confirmed')


def cleanup_owned_child(child, on_wait=None):
    """Never let a failed signal/timeout release the GPU lease with a live child."""
    if child is None:
        return
    try:
        stop_child(child, on_wait)
    except Exception:
        # Signal delivery/PGID inspection failures cannot authorize releasing
        # the lease. Keep ownership until Popen confirms its own child exited.
        while child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if on_wait:
                    on_wait()
    require(child.poll() is not None, 'Refusing to release GPU lock before owned evaluator exit')


def run(args):
    import fcntl
    # AutoDL's /root/autodl-fs commonly aliases /autodl-fs/data. Keep the
    # requested logical paths for lineage/argv, and compare actual directories.
    root = args.root.absolute(); shared = args.shared.absolute()
    require(root.resolve() == OLD_ROOT.resolve() and shared.resolve() == SHARED.resolve(),
            'This saved preview is restricted to the named OLD ROOT/shared payload')
    require(Path(sys.executable).absolute() == root/'early_final_env311/bin/python', 'Use the isolated preview environment executable')
    output = root/'results/gta_fixed_final_preview_20261014'
    own = (root/'.early_fixed_final_eval.lock').open('a+')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = root/'EARLY_FIXED_FINAL_STATUS.json'
    require(not status.exists() and not output.exists(), 'Preview has existing durable state/output; inspect it before any explicit continuation')
    child = None; gpu_lock = None
    state = {'schema_version': 1, 'role': 'early fixed-final preview only; no training or checkpoint selection',
             'status': 'starting', 'controller_pid': os.getpid(), 'controller_source_sha256': sha(Path(__file__)),
             'python_executable': sys.executable, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'root': str(root), 'shared': str(shared), 'output': str(output), 'started_at_utc': now(),
             'max_wait_seconds': args.max_wait_seconds, 'max_evaluation_seconds': args.max_evaluation_seconds,
             'canonical_final_unchanged': True, 'canonical20_trajectory_unchanged': True, 'fivefold': False,
             'formal_final_test': False, 'gpu_lock_acquired': False}

    def save(**values):
        state.update(values, updated_at_utc=now()); atomic(status, state)

    cancellation = OwnedCancellation()
    signal.signal(signal.SIGTERM, cancellation.request); signal.signal(signal.SIGINT, cancellation.request)

    def cleanup_progress():
        try:
            save(status='waiting_owned_child_exit', child_pid=child.pid if child is not None else None,
                 gpu_lock_acquired=gpu_lock is not None, cancellation_signals=cancellation.pending,
                 release_GPU_lock_requires_owned_child_exit=True)
        except Exception:
            # A status write failure must not interrupt reaping or release the
            # lease. The original exception and existing durable state remain.
            pass
    deadline = time.monotonic()+args.max_wait_seconds
    try:
        save(status='waiting_unet_and_ready')
        while not (root/'UNET_CALIBRATION_COMPLETE.json').is_file() or not (shared/'READY.json').is_file():
            require(time.monotonic() < deadline, 'Timed out waiting for U-Net complete and atomic READY')
            save(status='waiting_unet_and_ready', unet_marker_present=(root/'UNET_CALIBRATION_COMPLETE.json').is_file(),
                 ready_present=(shared/'READY.json').is_file(), gpu_lock_acquired=False)
            time.sleep(min(args.poll_seconds, max(.01, deadline-time.monotonic())))
        baseline = unet_guard(root)
        runtime = runtime_profile()
        audit = payload_guard(root, shared)
        command = evaluation_argv(root, shared, output)
        save(status='waiting_idle_GPU', unet_completion=baseline, runtime=runtime, READY_sha256=audit['READY_sha256'],
             checkpoint_sha256=audit['ready']['checkpoint_sha256'], actual_final_CPU_header=audit['actual_final_CPU_header'],
             command=command, canonical_runtime=audit['runtime_audit'], gpu_lock_acquired=False)
        while True:
            require(time.monotonic() < deadline, 'Timed out waiting for shared GPU-stage ownership')
            gpu = gpu_snapshot()
            if gpu['all_compute_GPUs_idle']:
                candidate = (root/'.segmentation_stage.lock').open('a+')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                else:
                    gpu = gpu_snapshot()
                    if gpu['all_compute_GPUs_idle']:
                        gpu_lock = candidate
                        break
                    fcntl.flock(candidate, fcntl.LOCK_UN); candidate.close()
            save(status='waiting_idle_GPU', GPU=gpu, gpu_lock_acquired=False)
            time.sleep(min(args.poll_seconds, max(.01, deadline-time.monotonic())))
        require(sha(shared/'READY.json') == audit['READY_sha256'], 'READY changed before inference')
        require(unet_guard(root) == baseline, 'U-Net completion changed before inference')
        locked_audit = payload_guard(root, shared, check_header=False)
        require(locked_audit['files'] == audit['files'] and locked_audit['cache_CT_inputs'] == audit['cache_CT_inputs'], 'Payload/input changed before inference')
        require(not output.exists(), 'Fresh preview output was created by another process')
        require(runtime_profile() == runtime, 'Runtime changed before evaluator')
        require(gpu_snapshot()['all_compute_GPUs_idle'], 'GPU became occupied immediately before evaluator')
        log = root/'logs/early_fixed_final_eval.log'; log.parent.mkdir(exist_ok=True)
        env = {k:v for k,v in os.environ.items() if not k.startswith('GTA_') and k not in ('OUTPUT_ROOT', 'CLS_CHECKPOINT')}
        env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        protocol = {'schema_version': 1, 'role': 'early fixed-final preview; no new training',
                    'controller_sha256': state['controller_source_sha256'], 'READY_sha256': audit['READY_sha256'],
                    'source_root': NEW_ROOT, 'source_config_sha256': CONFIG, 'training_source_sha256': SOURCE,
                    'source_freeze_sha256': FREEZE, 'full_code_package_sha256': PACKAGE,
                    'payload_files': audit['files'], 'case_ids': IDS, 'evaluation_sources': audit['evaluation_sources'],
                    'cache_CT_inputs': audit['cache_CT_inputs'], 'raw_GT': audit['raw_GT'],
                    'old_runtime': runtime, 'canonical_runtime': audit['runtime_audit'], 'old_GPU_before_inference': gpu,
                    'command': command, 'split': 'internal_diagnostic', 'checkpoint_selection': 'fixed final',
                    'checkpoint_epoch': 199, 'formal_final_test': False, 'sw_batch_size': 1,
                    'extra_case_inferences': 4, 'canonical_case_inferences': 80, 'preview_never_replaces_canonical': True,
                    'comparison_rule': 'compare decoded pixels/geometry/per-case metrics later; no averaging or device-based selection'}
        atomic(root/'EARLY_FIXED_FINAL_PROTOCOL.json', protocol)
        started = time.monotonic(); started_at = now()
        with log.open('x', encoding='utf-8') as stream:
            cancellation.launching = True
            try:
                child = subprocess.Popen(command, cwd=root/'code', env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                child_pgid = os.getpgid(child.pid)
                require(child_pgid == child.pid, 'Owned evaluator did not create its own process group')
                child_start_ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            finally:
                cancellation.launching = False
            # Signals received before Popen returned can only raise now that
            # the controller owns the child/PID/PGID needed for cleanup.
            cancellation.raise_pending()
            save(status='evaluating', child_pid=child.pid, child_pgid=child_pgid, child_proc_start_ticks=child_start_ticks,
                 evaluation_started_at_utc=started_at, GPU=gpu, log=str(log), gpu_lock_acquired=True)
            while child.poll() is None:
                require(time.monotonic()-started < args.max_evaluation_seconds, 'Owned evaluator exceeded wall-time budget')
                save(status='evaluating', child_pid=child.pid, evaluation_wall_seconds=time.monotonic()-started,
                     GPU=gpu_snapshot(), gpu_lock_acquired=True)
                time.sleep(args.poll_seconds)
        returncode = child.returncode; child = None
        require(returncode == 0, 'Frozen evaluator failed with exit ' + str(returncode))
        save(status='verifying', evaluator_return_code=returncode, gpu_lock_acquired=True)
        result = verify_result(root, output, audit)
        post = payload_guard(root, shared, check_header=False)
        require(post['READY_sha256'] == audit['READY_sha256'] and post['files'] == audit['files']
                and post['cache_CT_inputs'] == audit['cache_CT_inputs'] and post['raw_GT'] == audit['raw_GT'], 'Weights/source/data changed during preview')
        require(unet_guard(root) == baseline and sha(Path(__file__)) == state['controller_source_sha256'], 'Baseline or controller changed during preview')
        require(runtime_profile() == runtime, 'Runtime changed during preview')
        completed = result | {'schema_version': 1, 'status': 'completed', 'role': 'early fixed-final preview; not canonical trajectory',
                              'READY_sha256': audit['READY_sha256'], 'protocol_sha256': sha(root/'EARLY_FIXED_FINAL_PROTOCOL.json'),
                              'started_at_utc': started_at, 'completed_at_utc': now(), 'wall_seconds': time.monotonic()-started,
                              'parameter_updates': 0, 'extra_case_inferences': 4, 'formal_final_test': False,
                              'canonical_final_unchanged': True, 'canonical20_trajectory_unchanged': True}
        atomic(output/'EVAL_COMPLETE.json', completed)
        save(status='completed', completion_marker_sha256=sha(output/'EVAL_COMPLETE.json'),
             n_cases=4, evaluation_wall_seconds=completed['wall_seconds'], child_pid=None, gpu_lock_acquired=False)
        print(json.dumps({'status':'completed', 'role':completed['role'], 'output':str(output), 'n_cases':4}))
    except BaseException as error:
        cancellation.cleaning = True
        cleanup_owned_child(child, cleanup_progress)
        child = None
        save(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed_stop', error=repr(error),
             child_pid=None, gpu_lock_acquired=gpu_lock is not None, owned_child_exit_confirmed=True,
             cancellation_signals=cancellation.pending,
             completion_marker_written=(output/'EVAL_COMPLETE.json').is_file())
        raise
    finally:
        # Repeated normal cancellation signals are recorded through reaping
        # and lock release; they cannot abort ownership cleanup.
        cancellation.cleaning = True
        cleanup_owned_child(child, cleanup_progress)
        if gpu_lock is not None:
            fcntl.flock(gpu_lock, fcntl.LOCK_UN); gpu_lock.close()
        fcntl.flock(own, fcntl.LOCK_UN); own.close()
        try:
            save(gpu_lock_acquired=False, owned_child_exit_confirmed=True, locks_released_at_utc=now(),
                 cancellation_signals=cancellation.pending)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OLD_ROOT)
    parser.add_argument('--shared', type=Path, default=SHARED)
    parser.add_argument('--max-wait-seconds', type=int, default=7200)
    parser.add_argument('--max-evaluation-seconds', type=int, default=7200)
    parser.add_argument('--poll-seconds', type=int, default=10)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    require(1 <= args.max_wait_seconds <= 7200 and 1 <= args.max_evaluation_seconds <= 7200
            and 1 <= args.poll_seconds <= 30, 'Invalid bounded time settings')
    if args.plan:
        print(json.dumps({'role': 'READ_ONLY_PLAN_NOT_MODEL_RESULTS', 'server_connections': 0,
                          'inference_or_training_started': False, 'root': str(args.root), 'shared': str(args.shared),
                          'files': sorted(FILES), 'expected_config_sha256': CONFIG, 'expected_manifest_sha256': MANIFEST,
                          'expected_source_sha256': SOURCE, 'expected_freeze_sha256': FREEZE,
                          'packages': PACKAGES, 'python': '3.11.17', 'case_ids': IDS,
                          'max_wait_seconds': args.max_wait_seconds, 'max_evaluation_seconds': args.max_evaluation_seconds,
                          'wait_holds_gpu_lock': False, 'command': evaluation_argv(args.root, args.shared, args.root/'results/gta_fixed_final_preview_20261014'),
                          'canonical20_trajectory_changed': False}, indent=2)); return
    require(sys.platform == 'linux', 'Execution requires the named Linux OLD server; use --plan locally')
    run(args)


if __name__ == '__main__':
    main()
