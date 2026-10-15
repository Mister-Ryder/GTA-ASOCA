"""Owned OLD-only A/B Gaussian diagnostics; no training or model selection.

--plan uses stdlib. --fixture checks hand-calculated volumes (Torch if installed).
--diagnose is explicit, guarded, bounded eval-forward on immutable final200/train28.
Frozen code, existing diagnostics, config, targets and bank identity are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
OLD = Path('/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2')
SHARED = Path('/root/autodl-fs/experiments/GTA_ASOCA_Fidelity_20261014/early_final_mirror_20261014')
SOURCE = 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
FINAL = '2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'
CONFIG = 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
MANIFEST = '141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
BANK = '6d58425c5e057739c38c6734f37d3f0c4a79933544f0547705de81bf3853ab14'
HELPERS = {'diagnose_gaussian.py': '62daf89c7c49d23f3cb96a32a1970b2bb5fd5c68311aa197705850a5f7dba361',
           'run_early_fixed_final_eval.py': '2d7a4f24e4f2bae6fa58f5a023d726e1fb3f91bb0d2b7416a7dfe5bba4147e42'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_helpers():
    root = Path(__file__).resolve().parent
    loaded = []
    for name, expected in HELPERS.items():
        path = root/name
        require(sha(path) == expected, 'Pinned read-only helper differs: '+name)
        spec = importlib.util.spec_from_file_location('shadow_'+path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def region_losses(q, target, foreground):
    """Per-patch A and B plus FG/BG-originating terms; missing region renormalized."""
    error = (q-target).square()
    terms = {name: [] for name in ('A_FG', 'A_BG', 'B_FG', 'B_BG')}
    for e, mask in zip(error, foreground):
        n, nf = e.numel(), int(mask.sum())
        nb = n-nf
        zero = e.sum()*0.
        terms['A_FG'].append(e[mask].sum()/n)
        terms['A_BG'].append(e[~mask].sum()/n)
        present = int(nf > 0)+int(nb > 0)
        terms['B_FG'].append(e[mask].mean()/present if nf else zero)
        terms['B_BG'].append(e[~mask].mean()/present if nb else zero)
    result = {name: sum(values)/len(values) for name, values in terms.items()}
    result.update(A=result['A_FG']+result['A_BG'], B=result['B_FG']+result['B_BG'])
    return result


def fixture(torch_module=None):
    """Hand fractions first; optional actual CPU Torch autograd, never real-model data."""
    q4 = [Fraction(3, 5), Fraction(7, 10), Fraction(9, 10), Fraction(1, 2)]
    q8 = [Fraction(3, 4)]+[Fraction(1, 2)]*7
    cases = [('uneven_fg', [q4], [[1, 0, 0, 0]], Fraction(21, 400), Fraction(23, 600)),
             ('all_empty', [q4], [[0]*4], Fraction(21, 400), Fraction(21, 400)),
             ('all_fg', [q4], [[1]*4], Fraction(21, 400), Fraction(21, 400)),
             ('one_voxel', [q8], [[1]+[0]*7], Fraction(1, 128), Fraction(1, 32)),
             ('batchavg', [q8, [Fraction(7, 10)]*8], [[1]+[0]*7, [0]*8],
              Fraction(153, 6400), Fraction(57, 1600))]
    rows = []
    for name, qs, masks, expected_a, expected_b in cases:
        aa, bb, ga, gb = [], [], [], []
        for values, mask in zip(qs, masks):
            n, nf = len(values), sum(mask); nb = n-nf; present = int(nf > 0)+int(nb > 0)
            errors = [(v-Fraction(1, 2))**2 for v in values]
            aa.append(sum(errors)/n)
            bb.append(sum(sum(errors[i] for i in range(n) if mask[i] == r)/nr
                          for r, nr in ((1, nf), (0, nb)) if nr)/present)
            ga.append([2*(v-Fraction(1, 2))/n/len(qs) for v in values])
            gb.append([2*(v-Fraction(1, 2))/(nf if mask[i] else nb)/present/len(qs)
                       for i, v in enumerate(values)])
        require(sum(aa)/len(aa) == expected_a and sum(bb)/len(bb) == expected_b,
                'Hand calculation differs: '+name)
        row = {'name': name, 'hand_A': str(expected_a), 'hand_B': str(expected_b),
               'hand_calculation_verified': True, 'actual_torch_autograd_verified': False}
        if torch_module is not None:
            t = torch_module
            q = t.tensor([[float(v) for v in values] for values in qs], dtype=t.float64).requires_grad_(True)
            target = t.full_like(q, .5); fg = t.tensor(masks, dtype=t.bool)
            losses = region_losses(q, target, fg)
            t.testing.assert_close(losses['A'], t.tensor(float(expected_a), dtype=t.float64), rtol=0, atol=1e-12)
            t.testing.assert_close(losses['B'], t.tensor(float(expected_b), dtype=t.float64), rtol=0, atol=1e-12)
            for key, expected in (('A', ga), ('B', gb)):
                gradient = t.autograd.grad(losses[key], q, retain_graph=True)[0]
                t.testing.assert_close(gradient, t.tensor([[float(v) for v in x] for x in expected], dtype=t.float64), rtol=0, atol=1e-12)
            row['actual_torch_autograd_verified'] = True
        rows.append(row)
    return {'schema_version': 1, 'role': 'ANALYTIC_FIXTURE_NOT_ACTUAL_MODEL_DIAGNOSTIC',
            'status': 'passed' if torch_module is not None else 'hand_verified_torch_unavailable',
            'torch_available': torch_module is not None, 'cases': rows,
            'empty_policy': 'renormalize over present regions; all-empty/all-FG B equals A',
            'batch_policy': 'mean of per-patch losses; never pool batch region denominators'}


def config_inputs(args):
    require(OLD == Path(__file__).resolve().parents[1], 'Diagnostic must live in OLDROOT/ops')
    require(args.reference_bank.resolve().is_relative_to(OLD/'ops_inputs'), 'Reference must be uploaded under private OLD ops_inputs')
    require(args.output.resolve().is_relative_to(OLD/'results') and not args.output.exists(), 'Fresh OLD/results output required')
    diagnostic, guard = load_helpers()
    freeze = guard.source_guard(OLD)
    ready = read(SHARED/'READY.json'); guard.ready_structure(ready)
    require(ready['checkpoint_sha256'] == FINAL, 'Immutable final200 differs')
    tracked = {str(Path(__file__).resolve()): sha(__file__),
               str(OLD/'configs/main_source_freeze.json'): sha(OLD/'configs/main_source_freeze.json')}
    for name, expected in HELPERS.items(): tracked[str(OLD/'ops'/name)] = expected
    for name, record in ready['files'].items():
        path = guard.safe_member(SHARED, name)
        require(guard.file_record(path) == record, 'Shared payload changed: '+name)
        tracked[str(path)] = record['sha256']
    tracked[str(SHARED/'READY.json')] = sha(SHARED/'READY.json')
    manifest_path = OLD.parent/'cache/manifest.json'
    require(sha(manifest_path) == MANIFEST, 'OLD BASE/cache manifest differs')
    tracked[str(manifest_path)] = MANIFEST
    require(sha(args.reference_bank) == BANK, 'Original epoch070 fixed bank bytes differ')
    tracked[str(args.reference_bank.resolve())] = BANK
    config = read(SHARED/'config.json'); manifest = read(manifest_path)
    require(sha(SHARED/'config.json') == CONFIG, 'Actual final config bytes differ')
    train_ids = guard.config_guard(config, manifest)
    require(config['source_files_sha256'] == freeze['training_dependency_files'], 'Actual config training dependencies differ')
    require(config['sampling'] == 'standard' and config['sampling_fov'] == 'image-body'
            and config['seg_loss'] == 'dice_ce' and config['gaussian_weight'] == 10.
            and config['gaussian_late_weight'] is None and config['backbone_warmup_epochs'] == 0,
            'Actual recipe differs from predeclared diagnostic')
    evaluators = {name: sha(OLD/'code'/name) for name in guard.EVALUATORS}
    require(evaluators == ready['evaluation_sources'], 'Full evaluator guard differs')
    marker = read(SHARED/'TRAINING_COMPLETE.json')
    require(marker['epoch'] == 199 and marker['final_sha256'] == FINAL
            and marker['segmentation_updates'] == 25000 and marker['source_sha256'] == SOURCE
            and marker['manifest_sha256'] == MANIFEST and marker['trained_case_ids'] == train_ids,
            'Actual training completion differs')
    reference = read(args.reference_bank)
    require(len(reference['samples']) == 56 and Counter(x['case_id'] for x in reference['samples']) == Counter({x: 2 for x in train_ids}), 'Not the full 28x2 reference bank')
    require('run_root' not in reference, 'Original reference bank unexpectedly gained run_root')
    plan = {'schema_version': 1, 'role': 'FINAL200_TRAIN28_OFFLINE_AB_SHADOW_NOT_MODEL_RESULTS',
            'root': str(OLD), 'device': args.device, 'deadline_seconds': args.deadline,
            'threads_max': 2, 'nice_min': 10, 'checkpoint_sha256': FINAL,
            'config_sha256': CONFIG, 'manifest_sha256': MANIFEST, 'source_sha256': SOURCE,
            'full_source_package_sha256': freeze['code_package_sha256'],
            'evaluation_sources': evaluators, 'reference_bank_sha256': BANK,
            'reference_bank_has_run_root': False, 'n_patches': 56, 'last_sampling_index': reference['samples'][-1]['index'],
            'logical_relocation': {'config_untouched': True, 'recorded_output': config['output'],
                                   'actual_checkpoint': str(SHARED/'final.pt'), 'recorded_manifest': config['manifest'],
                                   'actual_manifest': str(manifest_path), 'basis': 'exact config/SHA, full source/evaluator guards and exact sampled bank'},
            'A': 'per-patch full-voxel mean MSE', 'B': 'equal FG/BG conditional means, present-region renormalization, then batch mean',
            'gaussian_lambda': 10., 'main_DS_coefficient': 4/7, 'model_mode': 'eval, batch1, FP32, no augmentation',
            'probe': 'detached logits; segmentation/A/B and regional-origin losses; no parameter gradients',
            'training_patients': train_ids, 'excluded_internal4': config['internal_split']['val'],
            'held_out_GT_read': False, 'parameter_updates': False, 'optimizer_created': False,
            'tracked_input_sha256': tracked, 'output': str(args.output)}
    return diagnostic, guard, config, manifest_path, reference, plan


def verify_inputs(guard, tracked):
    guard.source_guard(OLD)
    for path, expected in tracked.items(): require(sha(path) == expected, 'Immutable input changed: '+path)


def resources():
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      NUMEXPR_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    if hasattr(os, 'sched_getaffinity'):
        available = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, available[:2])
    if hasattr(os, 'nice'):
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if current < 10: os.nice(10-current)


def gpu_snapshot(allow_own=False):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=30)
    pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    require(not (set(pids)-({os.getpid()} if allow_own else set())), 'OLD GPU occupied by another process; no operation against that process')
    return {'time_utc_unix': time.time(), 'compute_pids': pids, 'own_pid': os.getpid(), 'all_idle': not pids}


@contextmanager
def lease():
    import fcntl
    with (OLD/'.segmentation_stage.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(json.dumps(list(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def energy(g, seg, mask):
    import numpy as np
    gc, gm = (g[0]+g[1])/math.sqrt(2), (g[1]-g[0])/math.sqrt(2)
    sc, sm = (seg[0]+seg[1])/math.sqrt(2), (seg[1]-seg[0])/math.sqrt(2)
    return [float(np.square(v[mask].astype(np.float64)).sum()) for v in (gc, gm, sc, sm)]+[
        float((gc[mask].astype(np.float64)*sc[mask]+gm[mask].astype(np.float64)*sm[mask]).sum()),
        float((gm[mask].astype(np.float64)*sm[mask]).sum())]


def energy_report(v):
    c, m, sc, sm, dot, mdot = v; total, stotal = c+m, sc+sm
    return {'raw_additive_statistics': v, 'common_energy': c, 'margin_energy': m, 'total_energy': total,
            'common_norm': math.sqrt(c), 'margin_norm': math.sqrt(m), 'total_norm': math.sqrt(total),
            'common_fraction': c/total if total else None,
            'norm_ratio_to_segmentation': math.sqrt(total/stotal) if stotal else None,
            'margin_norm_ratio_to_segmentation': math.sqrt(m/sm) if sm else None,
            'full_cosine_to_segmentation': max(-1., min(1., dot/math.sqrt(total*stotal))) if total*stotal else None,
            'margin_cosine_to_segmentation': max(-1., min(1., mdot/math.sqrt(m*sm))) if m*sm else None}


def diagnose(args):
    diagnostic, guard, config, manifest_path, reference, plan = config_inputs(args)
    require(Path(sys.executable).absolute() == OLD/'early_final_env311/bin/python', 'Use isolated early_final_env311')
    require(sys.version.split()[0] == '3.11.17', 'Actual isolated Python version differs')
    versions = {name: importlib.metadata.version(name) for name in guard.PACKAGES}
    require(versions == guard.PACKAGES, 'Actual nine runtime packages differ')
    with lease():
        plan['GPU_idle_check_1'] = gpu_snapshot()
        sys.path.insert(0, str(OLD/'code'))
        import numpy as np
        import torch
        from classification_data import CohortDataset, internal_split
        from data import load_manifest, case_directory
        from losses import objective
        from train import make_model
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        require(torch.get_num_threads() <= 2 and torch.get_num_interop_threads() <= 2
                and len(os.sched_getaffinity(0)) <= 2 and os.getpriority(os.PRIO_PROCESS, 0) >= 10,
                'Actual process resource limits differ')
        analytic = fixture(torch); require(analytic['status'] == 'passed', 'Analytic Torch fixture failed')
        manifest = load_manifest(manifest_path)
        require(internal_split(manifest, 0, config['split_seed']) == config['internal_split'], 'Deterministic training28 split differs')
        train_ids = plan['training_patients']; cases = {c['case_id']: c for c in manifest['cases']}
        cache_headers = {}
        for case_id in train_ids:
            headers = {}; cache = case_directory(manifest, case_id)
            for name in ('image', 'label', 'gaussian', 'skeleton', 'radius', 'branch_id', 'depth'):
                path = guard.safe_member(OLD.parent/'cache', cases[case_id]['cache_subdir']+'/'+name+'.npy')
                array = np.load(path, mmap_mode='r')
                require(list(array.shape) == cases[case_id]['shape_zyx'], 'Training cache shape differs: '+case_id+'/'+name)
                headers[name] = {'dtype': array.dtype.str, 'shape': list(array.shape), 'bytes': path.stat().st_size}
                del array
            with np.load(cache/'sampling.npz') as coords:
                require({'foreground', 'skeleton', 'thin', 'junction'}.issubset(coords.files), 'Training sampling keys missing')
            metadata = read(cache/'case.json')
            for key in ('case_id', 'shape_zyx', 'spacing_zyx_mm', 'source_sha256', 'preprocessing_version'):
                require(metadata.get(key) == cases[case_id].get(key), 'Training metadata differs: '+case_id+'/'+key)
            cache_headers[case_id] = headers
        saved = torch.load(SHARED/'final.pt', map_location='cpu', weights_only=True)
        header = guard.final_header(saved, config, torch)
        require(header['model_state_tensor_count'] == 327 and not header['CUDA_context_initialized'], 'Final327 CPU header differs')
        model = make_model(config['method'], not config['no_ds'])
        model.load_state_dict(saved['model'], strict=True); del saved
        model.requires_grad_(False); model.eval()
        before_state = state_hash(model)
        dataset = CohortDataset(manifest, train_ids, fold=0, role='train', sampling='standard',
                                sampling_fov='image-body', steps=reference['samples'][-1]['index']+1,
                                patch_size=128, seed=20261013, augment=False, max_cached_cases=1)
        dataset.set_epoch(0)
        counts = {x: 0 for x in train_ids}; bank = []
        for index in range(5600):
            rng = np.random.default_rng(np.random.SeedSequence([20261013, 0, index]))
            case_id = train_ids[int(rng.integers(len(train_ids)))]
            if counts[case_id] < 2: bank.append({'index': index, 'case_id': case_id}); counts[case_id] += 1
            if len(bank) == 56: break
        actual_bank = {'schema_version': 1, 'source_sha256': SOURCE, 'manifest_sha256': MANIFEST,
                       'training_patients': train_ids, 'excluded_internal4': config['internal_split']['val'],
                       'fold': 0, 'seed': 20261013, 'bank_epoch': 0, 'patches_per_training_case': 2,
                       'patch_size': 128, 'sampling': 'standard', 'sampling_fov': 'image-body', 'augmentation': False,
                       'role': 'training28 only; no checkpoint selection or independent validation',
                       'samples': [diagnostic.sample_identity(item, dataset[item['index']]) for item in bank]}
        diagnostic.compare_bank(actual_bank, reference)
        verify_inputs(guard, plan['tracked_input_sha256'])
        plan['GPU_idle_check_2'] = gpu_snapshot()  # Immediately before device transfer/first CUDA allocation.
        if args.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        model.to(args.device)
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output/'analytic_fixture.json', analytic)
        write(args.output/'fixed_sampling_bank.json', actual_bank)
        plan.update(runtime_versions=versions, actual_header=header, state_sha256_before=before_state,
                    cache_headers_training28_only=cache_headers, bank_exactly_verified=True,
                    torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                    affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0), own_pid=os.getpid())
        write(args.output/'diagnostic_plan.json', plan)
        totals = {}; q_totals = {}; origin_totals = {}; loss_totals = Counter(); started = time.monotonic()
        with (args.output/'patch_metrics.jsonl').open('w', encoding='utf-8') as stream:
            for ordinal, (item, expected_sample) in enumerate(zip(bank, actual_bank['samples'])):
                sample = dataset[item['index']]
                diagnostic.compare_bank(diagnostic.sample_identity(item, sample), expected_sample)
                batch = {k: v.unsqueeze(0).to(args.device) for k, v in sample.items() if torch.is_tensor(v)}
                with torch.no_grad(): prediction = model(batch['image'])
                require(len(prediction['aux_logits']) == 2, 'Main DS coefficient is not4/7')
                z = prediction['logits'].detach().float().clone().requires_grad_(True)
                require(z.dtype == torch.float32 and z.shape == (1, 2, 128, 128, 128), 'Expected batch1 FP32 logits')
                q = model.gau_activation(model.gau_final(z)); target = .5+.5*batch['gaussian'].float()
                torch.testing.assert_close(q.detach(), prediction['gaussian'], rtol=1e-6, atol=1e-7)
                fg_t = batch['label'][:, None] > 0
                terms = region_losses(q, target, fg_t)
                prediction.update(logits=z, gaussian=q)
                _, _, old_terms = objective(prediction, batch, 'gta', segmentation=config['seg_loss'], classifier_active=False,
                                             gaussian_weight=10., anatomy_reweighting=not config['disable_anatomy_reweighting'], return_terms=True)
                torch.testing.assert_close(10*terms['A'], old_terms['gaussian'], rtol=1e-6, atol=1e-9)
                seg_loss = old_terms['segmentation']
                losses = {'segmentation': seg_loss, **{k: 10*v for k, v in terms.items()}}
                gradients = {k: torch.autograd.grad(v, z, retain_graph=True)[0].detach().cpu().numpy()[0]
                             for k, v in losses.items()}
                require(all(np.isfinite(g).all() for g in gradients.values()), 'Nonfinite detached-logit gradient')
                closure = {}
                for key in ('A', 'B'):
                    summed = gradients[key+'_FG']+gradients[key+'_BG']
                    residual = gradients[key]-summed
                    cross_dot = float((gradients[key+'_FG'].astype(np.float64)*gradients[key+'_BG']).sum())
                    origin_energies = [float(np.square(gradients[key+s].astype(np.float64)).sum()) for s in ('_FG', '_BG')]
                    total_energy = float(np.square(gradients[key].astype(np.float64)).sum())
                    raw_origin = origin_energies+[cross_dot, total_energy]
                    origin_totals[key] = [a+b for a, b in zip(origin_totals.get(key, [0.]*4), raw_origin)]
                    closure[key] = {'max_abs_error': float(np.abs(residual).max()),
                                    'residual_norm': float(np.linalg.norm(residual.astype(np.float64))),
                                    'total_norm': math.sqrt(total_energy),
                                    'FG_origin_energy': origin_energies[0], 'BG_origin_energy': origin_energies[1],
                                    'FG_BG_origin_dot': cross_dot, 'total_energy': total_energy,
                                    'energy_closure_residual': total_energy-sum(origin_energies)-2*cross_dot}
                    closure[key]['relative_residual_to_origin_norm_sum'] = closure[key]['residual_norm']/max(
                        math.sqrt(origin_energies[0])+math.sqrt(origin_energies[1]), 1e-30)
                    require(np.allclose(gradients[key], summed, rtol=1e-4, atol=1e-9), 'Originating gradient closure failed')
                    require(closure[key]['relative_residual_to_origin_norm_sum'] <= 1e-5, 'Originating gradient relative closure failed')
                qa, qb = [torch.autograd.grad(10*terms[k], q, retain_graph=True)[0].detach() for k in ('A', 'B')]
                nf = int(fg_t.sum()); n = fg_t.numel(); nb = n-nf; present = int(nf > 0)+int(nb > 0)
                multiplier = torch.where(fg_t, n/(present*max(1, nf)), n/(present*max(1, nb)))
                torch.testing.assert_close(qb, qa*multiplier, rtol=1e-5, atol=1e-8)
                fg = fg_t.detach().cpu().numpy()[0, 0]
                masks = {'all': np.ones(fg.shape, dtype=bool), 'FG_coordinates': fg, 'BG_coordinates': ~fg}
                row = {**item, 'ordinal': ordinal, 'sample_identity': expected_sample,
                       'n_voxels': n, 'foreground_voxels': nf, 'background_voxels': nb, 'empty_FG': nf == 0,
                       'losses_lambda10': {k: float(v.detach()) for k, v in losses.items()},
                       'originating_gradient_closure': closure, 'q_derivative': {}, 'gradient_by_loss_origin_and_logit_coordinate': {}}
                for region, mask_t in (('FG', fg_t), ('BG', ~fg_t)):
                    a, b = qa[mask_t].double(), qb[mask_t].double(); ea, eb = float(a.square().sum()), float(b.square().sum())
                    row['q_derivative'][region] = {'n_voxels': a.numel(), 'A_energy': ea, 'B_energy': eb,
                                                   'expected_B_over_A_multiplier': n/(present*(nf if region == 'FG' else nb)) if a.numel() else None,
                                                   'actual_norm_multiplier': math.sqrt(eb/ea) if ea else None,
                                                   'exact_multiplier_relation_verified': True}
                    q_totals[region] = [x+y for x, y in zip(q_totals.get(region, [0., 0., 0.]), [a.numel(), ea, eb])]
                for origin, g in gradients.items():
                    for region, mask in masks.items():
                        name = origin+'.'+region; raw = energy(g, gradients['segmentation'], mask)
                        totals[name] = [a+b for a, b in zip(totals.get(name, [0.]*6), raw)]
                        row['gradient_by_loss_origin_and_logit_coordinate'][name] = energy_report(raw)
                for key, val in row['losses_lambda10'].items(): loss_totals[key] += val
                row['B_over_A_logit_norm'] = math.sqrt(row['gradient_by_loss_origin_and_logit_coordinate']['B.all']['total_energy']/row['gradient_by_loss_origin_and_logit_coordinate']['A.all']['total_energy']) if row['gradient_by_loss_origin_and_logit_coordinate']['A.all']['total_energy'] else None
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                print(json.dumps({'completed': ordinal+1, 'total': 56, 'elapsed_seconds': time.monotonic()-started}), flush=True)
                if args.device == 'cuda': gpu_snapshot(allow_own=True)
                else: require(not torch.cuda.is_initialized(), 'CPU diagnostic initialized CUDA')
                del prediction, z, q, terms, old_terms, losses, gradients, batch, qa, qb
        after_state = state_hash(model)
        require(before_state == after_state and all(p.grad is None for p in model.parameters()), 'Model state or parameter gradients changed')
        for item, expected_sample in zip(bank, reference['samples']):
            diagnostic.compare_bank(diagnostic.sample_identity(item, dataset[item['index']]), expected_sample)
        verify_inputs(guard, plan['tracked_input_sha256'])
        require(read(args.output/'fixed_sampling_bank.json') == reference, 'Persisted bank identity changed')
        pooled = {k: energy_report(v) for k, v in totals.items()}
        summary = {'schema_version': 1, 'role': 'ACTUAL_FINAL200_TRAIN28_OFFLINE_AB_DIAGNOSTIC_NOT_MODEL_RESULTS',
                   'n_patches': 56, 'mean_lambda10_losses': {k: v/56 for k, v in loss_totals.items()},
                   'aggregated_gradient_by_loss_origin_and_logit_coordinate': pooled,
                   'aggregate_B_over_A_logit_norm': math.sqrt(pooled['B.all']['total_energy']/pooled['A.all']['total_energy']) if pooled['A.all']['total_energy'] else None,
                   'aggregation': 'sum squared norms and dot products over bank before ratios/cosines; never mean per-patch ratios',
                   'aggregate_q_derivative': {k: {'n_voxels': int(v[0]), 'A_energy': v[1], 'B_energy': v[2],
                                                 'norm_multiplier_B_over_A': math.sqrt(v[2]/v[1]) if v[1] else None}
                                              for k, v in q_totals.items()},
                   'aggregate_originating_gradient_energy': {k: {'FG_energy': v[0], 'BG_energy': v[1], 'FG_BG_dot': v[2],
                                                                  'total_energy': v[3], 'closure_residual': v[3]-v[0]-v[1]-2*v[2],
                                                                  'energies_do_not_add_without_cross_term': True}
                                                               for k, v in origin_totals.items()},
                   'state_sha256_before': before_state, 'state_sha256_after': after_state,
                   'source_and_inputs_before_after_verified': True, 'parameter_gradients_absent': True,
                   'optimizer_created': False, 'parameter_updates': False, 'held_out_GT_read': False,
                   'limitations': ['Eval-mode batch1 FP32 detached-logit derivatives exclude DS sibling/parameter Jacobian paths.',
                                   'Regional-origin losses and logit-coordinate masks are different axes; convolution crosses GT boundary.',
                                   'FG/BG balancing changes gradient scale, preserves empty patches, and does not guarantee lower common energy.',
                                   'Training bank is not validation; no Dice/ASSD or connectivity claim; no thin-centreline supervision added.']}
        write(args.output/'summary.json', summary)
        artifacts = {name: sha(args.output/name) for name in ('analytic_fixture.json', 'fixed_sampling_bank.json', 'diagnostic_plan.json', 'patch_metrics.jsonl', 'summary.json')}
        write(args.output/'DIAGNOSTIC_COMPLETE.json', {'status': 'completed', 'n_patches': 56, 'artifacts_sha256': artifacts,
                                                     'state_unchanged': True, 'parameter_updates': False, 'held_out_GT_read': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true'); mode.add_argument('--fixture', action='store_true')
    mode.add_argument('--diagnose', action='store_true')
    parser.add_argument('--reference-bank', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--deadline', type=int, default=1800)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(1 <= args.deadline <= 1800, 'Hard deadline must be1..1800seconds')
    if args.fixture:
        torch_module = None
        if importlib.util.find_spec('torch') is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''; resources()
            import torch
            torch.set_num_threads(2); torch.set_num_interop_threads(1); torch_module = torch
        result = fixture(torch_module)
        if args.output is not None:
            require(not args.output.exists(), 'Fixture output must be fresh'); write(args.output, result)
        print(json.dumps(result, indent=2)); return
    require(args.reference_bank is not None and args.output is not None, '--reference-bank/--output required')
    if args.plan:
        *_, plan = config_inputs(args); print(json.dumps(plan, indent=2)); return
    require(sys.platform == 'linux', 'Actual diagnostic is Linux OLD-only')
    if args._worker:
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(args.deadline)  # OS termination even if the bounded owning parent dies.
        require(int(os.environ.get('GTA_SHADOW_OWN_PARENT', '-1')) == os.getppid(), 'Worker must be started by its bounded owning parent')
        diagnose(args); return
    resources()
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.device == 'cpu': env['CUDA_VISIBLE_DEVICES'] = ''
    env['GTA_SHADOW_OWN_PARENT'] = str(os.getpid())
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker']
    worker = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        worker.wait(timeout=args.deadline)
    except subprocess.TimeoutExpired:
        os.killpg(worker.pid, signal.SIGKILL); worker.wait()
        raise SystemExit('Hard diagnostic deadline reached; only owned worker process group terminated')
    except BaseException:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            try: worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL); worker.wait()
        raise
    raise SystemExit(worker.returncode)


if __name__ == '__main__':
    main()
