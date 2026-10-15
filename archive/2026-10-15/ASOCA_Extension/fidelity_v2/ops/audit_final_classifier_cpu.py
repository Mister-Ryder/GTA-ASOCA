"""CPU-only final-classifier distributions on the immutable56 training bank.

No segmentation forward, CUDA, optimizer, parameter gradient, GPU query or
segmentation lock. May coexist with OLD GPU training. --plan is stdlib-only.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
OLD = Path('/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2')
SHADOW_SHA = 'a8c1c6b9726aebf1fbeb7fff4a05d46b59bd4d02660d14eaf98028c4844e7137'
DONOR_SHA = '308ebc33d55766458ebd3241c8422dbadb9259b929570bc5f03afd51fbec31a4'
HEADS = ('generation', 'structure', 'location')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def helper():
    path = Path(__file__).resolve().parent/'shadow_gaussian_region_balance.py'
    require(sha(path) == SHADOW_SHA, 'Reviewed read-only source/bank guard changed')
    spec = importlib.util.spec_from_file_location('classifier_audit_guard', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@contextmanager
def cpu_lease():
    import fcntl
    with (OLD/'.classifier_final_cpu_audit.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def rng_identity(torch, np):
    return {'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
            'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def summarize(rows, model_name, nonempty):
    selected = [r for r in rows if not nonempty or r['nonempty']]
    n = len(selected); heads = {}
    for i, name in enumerate(HEADS):
        matrix = [[0]*7 for _ in range(7)]; confidence = []
        for row in selected:
            prediction = row['models'][model_name][name]
            matrix[row['priors'][i]][prediction['prediction']] += 1
            confidence.append(prediction['max_probability'])
        gt = [sum(v) for v in matrix]; pred = [sum(matrix[j][i] for j in range(7)) for i in range(7)]
        mean = sum(confidence)/n if n else None
        heads[name] = {'n_patches': n, 'confusion_matrix_GT_rows_prediction_columns': matrix,
                       'GT_counts': gt, 'prediction_counts': pred,
                       'prediction_frequency': [v/n for v in pred] if n else [None]*7,
                       'predicted0_fraction': pred[0]/n if n else None,
                       'accuracy_analysis_only': sum(matrix[j][j] for j in range(7))/n if n else None,
                       'confidence_max_probability': {'n': n, 'mean': mean,
                         'population_variance': sum((v-mean)**2 for v in confidence)/n if n else None,
                         'minimum': min(confidence) if n else None, 'maximum': max(confidence) if n else None}}
    triggers = {'generation_ge5': sum(r['models'][model_name]['generation']['prediction'] >= 5 for r in selected),
                'structure_eq3': sum(r['models'][model_name]['structure']['prediction'] == 3 for r in selected),
                'location_nonzero': sum(r['models'][model_name]['location']['prediction'] != 0 for r in selected)}
    return {'n_patches': n, 'heads': heads, 'conditional_trigger_counts': triggers,
            'conditional_trigger_fractions': {k: v/n if n else None for k, v in triggers.items()}}


def run(args):
    h = helper()
    args.device = 'cpu'
    diagnostic, guard, config, manifest_path, reference, plan = h.config_inputs(args)
    require(args.classifier_donor is not None and sha(args.classifier_donor) == DONOR_SHA, 'Actual fixed classifier80 donor308 required')
    plan['tracked_input_sha256'][str(Path(__file__).resolve())] = sha(__file__)
    plan['tracked_input_sha256'][str(args.classifier_donor.resolve())] = DONOR_SHA
    for key in ('A', 'B', 'probe', 'gaussian_lambda', 'main_DS_coefficient', 'GPU_idle_check_1', 'GPU_idle_check_2'):
        plan.pop(key, None)
    plan.update(role='CPU_CLASSIFIER_DISTRIBUTIONS_TRAIN28_ONLY_NOT_SELECTION',
                model_mode='direct classifier.eval forward only, CPU FP32 batch1 CT128',
                classifier_donor_sha256=DONOR_SHA, CUDA_allowed=False,
                donor_logical_relocation={'recorded_main_donor_path': config['classifier_init'],
                                          'actual_CPU_donor_path': str(args.classifier_donor.resolve()),
                                          'identity_basis': 'same308 byte SHA and exact original load audit except physical path'},
                GPU_query_or_idle_required=False, segmentation_lock_used=False,
                independent_lock=str(OLD/'.classifier_final_cpu_audit.lock'),
                accuracy_is_gate=False, classifier_selection_performed=False,
                scope='all56 and nonempty49 standard fixed bank; never mix geometry512 pretraining bank frequencies')
    if args.plan:
        print(json.dumps(plan, indent=2)); return
    require(Path(sys.executable).absolute() == OLD/'early_final_env311/bin/python', 'Use isolated OLD early_final_env311')
    require(sys.version.split()[0] == '3.11.17', 'Isolated Python version differs')
    versions = {name: importlib.metadata.version(name) for name in guard.PACKAGES}
    require(versions == guard.PACKAGES, 'Actual isolated runtime packages differ')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment must hide all CUDA devices')
    with cpu_lease():
        sys.path.insert(0, str(OLD/'code'))
        import numpy as np
        import torch
        from train import make_model
        from checkpoint_utils import read_checkpoint, load_audited
        from classification_data import CohortDataset, internal_split
        from data import load_manifest, case_directory
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        require(str(torch.__version__) == config['torch'] and not torch.cuda.is_initialized(), 'Actual Torch/CPU state differs')
        require(len(os.sched_getaffinity(0)) <= 2 and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'CPU resource limits differ')
        manifest = load_manifest(manifest_path)
        require(internal_split(manifest, 0, config['split_seed']) == config['internal_split'], 'Training split differs')
        train_ids = config['trained_case_ids']; cases = {c['case_id']: c for c in manifest['cases']}
        headers = {}
        for case_id in train_ids:
            headers[case_id] = {'CT_full_cache_shape_zyx': cases[case_id]['shape_zyx'],
                                'CT_original_size_xyz': cases[case_id]['original_geometry']['size_xyz'],
                                'CT_original_spacing_xyz_mm': cases[case_id]['original_geometry']['spacing_xyz_mm'],
                                'source_CT_SHA': cases[case_id]['source_sha256']['image'], 'patch_CT_shape': [1, 128, 128, 128],
                                'patch_priors_shape': [3]}
            cache = case_directory(manifest, case_id)
            for name in ('image', 'label', 'gaussian', 'skeleton', 'radius', 'branch_id', 'depth'):
                path = guard.safe_member(OLD.parent/'cache', cases[case_id]['cache_subdir']+'/'+name+'.npy')
                a = np.load(path, mmap_mode='r')
                require(list(a.shape) == cases[case_id]['shape_zyx'], 'Training cache dimensions differ')
                headers[case_id][name] = {'dtype': a.dtype.str, 'shape': list(a.shape)}
                del a
            metadata = h.read(cache/'case.json')
            for key in ('case_id', 'shape_zyx', 'spacing_zyx_mm', 'source_sha256', 'preprocessing_version'):
                require(metadata.get(key) == cases[case_id].get(key), 'Training metadata differs: '+case_id+'/'+key)
        saved = torch.load(h.SHARED/'final.pt', map_location='cpu', weights_only=True)
        header = guard.final_header(saved, config, torch)
        require(header['model_state_tensor_count'] == 327, 'Main final must have327 tensors')
        model = make_model('gta', True); model.load_state_dict(saved['model'], strict=True); del saved
        model.requires_grad_(False); model.eval()
        donor_checkpoint, donor_state, donor_format = read_checkpoint(args.classifier_donor)
        cfg = donor_checkpoint.get('config', {})
        require(donor_checkpoint.get('epoch') == 79 and len(donor_state) == 90
                and cfg.get('pretrained_classifier') is True and cfg.get('smoke') is False
                and cfg.get('epochs') == 80 and cfg.get('steps') == 64
                and cfg.get('additional_pretraining_updates') == 5120
                and cfg.get('manifest_sha256') == h.MANIFEST and cfg.get('trained_case_ids') == train_ids,
                'Donor is not actual fixed80x64 train28 classifier final')
        require(all(bool(torch.isfinite(v).all()) for v in donor_state.values()), 'Nonfinite classifier80 donor')
        classifier80 = copy.deepcopy(model.classifier)
        donor_audit = load_audited(classifier80, args.classifier_donor, 'classifier', 'asoca', 'load')
        require(donor_audit['loaded_key_count'] == 90 and not donor_audit['missing_keys']
                and not donor_audit['unexpected_keys'] and not donor_audit['shape_mismatches'], 'Not all90 donor tensors loaded')
        recorded = next(x for x in config['init_audits'] if x['component'] == 'classifier')
        require({k: v for k, v in donor_audit.items() if k != 'path'} == {k: v for k, v in recorded.items() if k != 'path'},
                'Actual donor transfer differs from original main initialization')
        classifier80.requires_grad_(False); classifier80.eval()
        del donor_checkpoint, donor_state
        classifiers = {'main200': model.classifier, 'classifier80_fixedfinal': classifier80}
        before = {'main327': h.state_hash(model), **{k: h.state_hash(v) for k, v in classifiers.items()}}
        before_flags = {k: {n: m.training for n, m in v.named_modules()} for k, v in classifiers.items()}
        require(all(not flag for flags in before_flags.values() for flag in flags.values()), 'Classifier must stay eval')
        before_rng = rng_identity(torch, np)
        dataset = CohortDataset(manifest, train_ids, fold=0, role='train', sampling='standard', sampling_fov='image-body',
                                steps=reference['samples'][-1]['index']+1, patch_size=128, seed=20261013,
                                augment=False, max_cached_cases=1)
        dataset.set_epoch(0)
        bank = []; counts = {x: 0 for x in train_ids}
        for index in range(5600):
            rng = np.random.default_rng(np.random.SeedSequence([20261013, 0, index]))
            case_id = train_ids[int(rng.integers(len(train_ids)))]
            if counts[case_id] < 2: bank.append({'index': index, 'case_id': case_id}); counts[case_id] += 1
            if len(bank) == 56: break
        actual_bank = {k: v for k, v in reference.items() if k != 'samples'}
        # Reconstruct protocol fields independently: copying reference values cannot certify them.
        expected_protocol = {'schema_version': 1, 'source_sha256': h.SOURCE, 'manifest_sha256': h.MANIFEST,
                             'training_patients': train_ids, 'excluded_internal4': config['internal_split']['val'],
                             'fold': 0, 'seed': 20261013, 'bank_epoch': 0, 'patches_per_training_case': 2,
                             'patch_size': 128, 'sampling': 'standard', 'sampling_fov': 'image-body', 'augmentation': False,
                             'role': 'training28 only; no checkpoint selection or independent validation'}
        require(actual_bank == expected_protocol and 'run_root' not in reference, 'Original bank protocol differs')
        actual_bank = {**expected_protocol, 'samples': [diagnostic.sample_identity(item, dataset[item['index']]) for item in bank]}
        diagnostic.compare_bank(actual_bank, reference)
        h.verify_inputs(guard, plan['tracked_input_sha256'])
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output/'fixed_sampling_bank.json', actual_bank)
        plan.update(actual_main_header=header, donor_load_audit=donor_audit, cache_headers_training28_only=headers,
                    runtime_versions=versions,
                    state_sha256_before=before, rng_sha256_before=before_rng, eval_flags_before=before_flags,
                    n_threads=torch.get_num_threads(), n_interop_threads=torch.get_num_interop_threads(),
                    affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0))
        write(args.output/'audit_plan.json', plan)
        rows = []; started = time.monotonic()
        with (args.output/'patch_predictions.jsonl').open('w', encoding='utf-8') as stream:
            for item, identity in zip(bank, actual_bank['samples']):
                sample = dataset[item['index']]
                diagnostic.compare_bank(diagnostic.sample_identity(item, sample), identity)
                image = sample['image'].unsqueeze(0).float()
                require(tuple(image.shape) == (1, 1, 128, 128, 128) and image.device.type == 'cpu', 'Expected CPU CT128 batch1')
                row = {**item, 'sample_identity': identity, 'priors': sample['priors'].tolist(),
                       'nonempty': bool(sample['label'].any()), 'models': {}}
                with torch.inference_mode():
                    for name, classifier in classifiers.items():
                        prediction = classifier(image)  # Deliberately never call the full GTA model.
                        outputs = {}
                        for head in HEADS:
                            logits = prediction[head].float()
                            require(tuple(logits.shape) == (1, 7) and bool(torch.isfinite(logits).all()), 'Invalid classifier logits')
                            probability = logits.softmax(1)[0]
                            outputs[head] = {'prediction': int(probability.argmax()), 'max_probability': float(probability.max()),
                                             'probabilities': probability.tolist()}
                        guidance = prediction['guidance_map'].float()
                        require(bool(torch.isfinite(guidance).all()), 'Nonfinite guidance map')
                        outputs['guidance_map'] = {'shape': list(guidance.shape), 'mean': float(guidance.mean()),
                                                   'population_variance': float(guidance.var(unbiased=False))}
                        row['models'][name] = outputs
                require(row['nonempty'] == bool(row['priors'][0] > 0), 'Nonempty mask/prior disagreement')
                require(not torch.cuda.is_initialized(), 'CPU classifier audit initialized CUDA')
                rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                print(json.dumps({'classifier_patches_complete': len(rows), 'total': 56, 'seconds': time.monotonic()-started}), flush=True)
        for item, identity in zip(bank, reference['samples']):
            diagnostic.compare_bank(diagnostic.sample_identity(item, dataset[item['index']]), identity)
        after = {'main327': h.state_hash(model), **{k: h.state_hash(v) for k, v in classifiers.items()}}
        after_flags = {k: {n: m.training for n, m in v.named_modules()} for k, v in classifiers.items()}
        after_rng = rng_identity(torch, np)
        require(before == after and before_rng == after_rng and before_flags == after_flags, 'Model state/RNG/eval mode changed')
        require(all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in classifier80.parameters()), 'Unexpected parameter gradient')
        require(len(rows) == 56 and sum(r['nonempty'] for r in rows) == 49, 'Must report full56/nonempty49')
        h.verify_inputs(guard, plan['tracked_input_sha256'])
        require(h.read(args.output/'fixed_sampling_bank.json') == reference, 'Persisted fixed bank identity changed')
        summary = {'role': 'ACTUAL_CPU_CLASSIFIER_DISTRIBUTIONS_TRAIN28_NOT_SELECTION', 'n_patches': 56,
                   'reference_bank_sha256': h.BANK, 'n_nonempty': 49,
                   'models': {name: {'all56': summarize(rows, name, False), 'nonempty49': summarize(rows, name, True)} for name in classifiers},
                   'state_sha256_before': before, 'state_sha256_after': after, 'rng_sha256_before': before_rng, 'rng_sha256_after': after_rng,
                   'states_RNG_eval_unchanged': True, 'CUDA_initialized': torch.cuda.is_initialized(),
                   'parameter_gradients_absent': True, 'optimizer_created': False, 'heldout_GT_read': False,
                   'accuracy_used_as_gate_or_checkpoint_selection': False,
                   'limitations': ['Standard training56 frequency is not the earlier geometry512 pretraining evaluation frequency.',
                                   'Two patches per training patient are a fixed diagnostic bank, not natural patch frequency or independent validation.',
                                   'Whole-patch proxy prediction audit cannot establish clinical anatomy accuracy or segmentation benefit.',
                                   'Conditional trigger fractions are whole-patch argmax switches, not spatial guidance maps.',
                                   'Only direct classifier eval forwards; no full GTA segmentation forward or training updates.']}
        write(args.output/'summary.json', summary)
        artifacts = {name: sha(args.output/name) for name in ('fixed_sampling_bank.json', 'audit_plan.json', 'patch_predictions.jsonl', 'summary.json')}
        write(args.output/'AUDIT_COMPLETE.json', {'status': 'completed', 'n_patches': 56, 'n_nonempty': 49,
                                                 'artifact_sha256': artifacts, 'states_RNG_eval_unchanged': True,
                                                 'CUDA_initialized': False, 'parameter_updates': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true'); mode.add_argument('--audit', action='store_true')
    parser.add_argument('--reference-bank', type=Path, required=True)
    parser.add_argument('--classifier-donor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=int, default=900)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); require(1 <= args.deadline <= 900, 'Deadline must be1..900 seconds')
    if args.plan: run(args); return
    require(sys.platform == 'linux', 'Actual execution is OLD Linux CPU-only')
    if args._worker:
        signal.signal(signal.SIGALRM, signal.SIG_DFL); signal.alarm(args.deadline)
        require(int(os.environ.get('GTA_CLASSIFIER_AUDIT_PARENT', '-1')) == os.getppid(), 'Only bounded owner may start worker')
        run(args); return
    h = helper(); h.resources()
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', GTA_CLASSIFIER_AUDIT_PARENT=str(os.getpid()))
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker']
    worker = subprocess.Popen(command, env=env, start_new_session=True)
    try: worker.wait(timeout=args.deadline)
    except BaseException:
        if worker.poll() is None: os.killpg(worker.pid, signal.SIGKILL); worker.wait()
        raise
    raise SystemExit(worker.returncode)


if __name__ == '__main__': main()
