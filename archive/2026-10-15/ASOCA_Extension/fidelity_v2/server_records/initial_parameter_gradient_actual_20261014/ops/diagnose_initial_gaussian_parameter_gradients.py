"""NEW-only initial-model train-mode parameter-gradient diagnostic, never training.

AMP is the primary main-recipe precision. FP32 is a separate explicit run; an
OOM never changes patch/batch/precision automatically. Existing final results,
frozen A, sealed B, donor files and held-out GT remain untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014')
VARIANT = ROOT/'variants/gaussian_region_balance_v1'
B_SOURCE = '231e191e77e5f492b2602df19ce29046bb78dcb582dd8b03f1e982f21493828a'
MAIN_FINAL = '2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'
INITIAL = 'd84df10851c2532e9b70311eba2efe07049f5348e56fe4e76daf7856e6427d79'
BANK = '6d58425c5e057739c38c6734f37d3f0c4a79933544f0547705de81bf3853ab14'
HELPERS = {'run_early_fixed_final_eval.py': '2d7a4f24e4f2bae6fa58f5a023d726e1fb3f91bb0d2b7416a7dfe5bba4147e42',
           'prepare_gaussian_control.py': '0145556d123356956bd80546fb7f2eda0c44bd8761dac2c6ae1f6ae44c894a17',
           'diagnose_gaussian.py': '62daf89c7c49d23f3cb96a32a1970b2bb5fd5c68311aa197705850a5f7dba361',
           'shadow_gaussian_region_balance.py': 'a8c1c6b9726aebf1fbeb7fff4a05d46b59bd4d02660d14eaf98028c4844e7137'}
CHECKER = 'dc785e52f628859613bbe34594ef3711621583cbd5eee17ef25910cabc0135da'
DONORS = {'backbone': '29b6977b74e547735717fda0785a9d532397df364e1f641bed0e6682db0384d3',
          'classifier': '308ebc33d55766458ebd3241c8422dbadb9259b929570bc5f03afd51fbec31a4'}
LOSSES = ('segmentation', 'A', 'B', 'totalA', 'totalB')


def need(value, reason):
    if not value: raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def helpers():
    folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(folder))
    for name, expected in HELPERS.items(): need(sha(folder/name) == expected, 'Pinned helper changed: '+name)
    return tuple(load(folder/name, 'initial_'+Path(name).stem) for name in HELPERS)


def variant_guard():
    path = Path(__file__).resolve().parents[1]/'variants/gaussian_region_balance_v1/ops/check_variant.py'
    need(sha(path) == CHECKER, 'Variant checker changed')
    checker = load(path, 'initial_variant_checker')
    protocol, inspection = checker.inspect()
    need(inspection['source_sha256'] == B_SOURCE and inspection['changed_code_files'] == ['losses.py'], 'Not the sealed B-only reduction variant')
    return checker, protocol, inspection


def canonical_guard(guard):
    """Read existing receipts, never held-out images, GT or predictions."""
    done = read(ROOT/'POSTFIT_VERIFIED.json'); status = read(ROOT/'POSTFIT_STATUS.json')
    train = ROOT/'results/gta_enhanced200x125/training'
    trajectory = train.parent/'internal4_trajectory'
    need(done.get('status') == 'postfit_verified' and done.get('verified_checkpoint_evaluations') == 20
         and done.get('verified_case_evaluations') == 80 and status.get('status') == 'completed'
         and status.get('verified_checkpoints') == 20 and status.get('verified_case_evaluations') == 80,
         'Canonical20/80 has not completed and passed independent verification')
    need(status.get('verified_receipt_sha256') == sha(ROOT/'POSTFIT_VERIFIED.json'), 'Canonical completion receipt changed')
    complete = read(trajectory/'TRAJECTORY_COMPLETE.json')
    need(done['trajectory_complete_sha256'] == sha(trajectory/'TRAJECTORY_COMPLETE.json')
         and complete.get('n_evaluations') == 20 and complete.get('case_ids') == guard.IDS
         and complete['checkpoints'][-1]['checkpoint_sha256'] == MAIN_FINAL
         and done['curve_manifest_sha256'] == sha(train.parent/'learning_curves/curve_manifest.json'), 'Canonical complete/curve identity differs')
    need(sha(train/'config.json') == guard.CONFIG and sha(train/'final.pt') == MAIN_FINAL, 'Actual canonical config/final changed')
    return {'postfit_receipt_sha256': sha(ROOT/'POSTFIT_VERIFIED.json'), 'postfit_status_sha256': sha(ROOT/'POSTFIT_STATUS.json'),
            'trajectory_complete_sha256': sha(trajectory/'TRAJECTORY_COMPLETE.json'), 'main_config_sha256': guard.CONFIG,
            'main_final_sha256': MAIN_FINAL, 'held_out_GT_or_images_read': False}


def plan(args):
    return {'role': 'INITIAL_TRAINMODE_PARAMETER_GRADIENT_DIAGNOSTIC_NOT_MODEL_RESULTS', 'status': 'PLAN_ONLY',
            'root': ROOT.as_posix(), 'B_source_sha256': B_SOURCE, 'precision': args.precision,
            'batch_size': 2, 'patch_size': [128]*3, 'stage': 'frozen', 'augmentation': False,
            'bank_patches': 56, 'standalone_batches': 28, 'paired_by': 'adjacent samples in original fixed bank order',
            'reset_before_each_batch': 'all327 initial CPU state tensors and identical CPU/CUDA RNG; set frozen/train',
            'expected_reconstructed_initial_state_sha256': INITIAL, 'historical_main_theta0_verified': False,
            'losses': list(LOSSES), 'Gaussian_lambda': 10, 'DS_weights': [4/7, 2/7, 1/7],
            'AMP_derivative_scale': 1024 if args.precision == 'amp' else 1,
            'optimizer_created': False, 'parameter_updates': False, 'parameter_grad_populated': False,
            'held_out_GT_read': False, 'new_validation_or_performance_claim': False,
            'BN_policy': 'Diagnostic copy buffers may update during train-mode forward; restored before each batch/end. Parameters never update.',
            'clip_policy': 'Report implied multiplier min(1,12/(direct-combined FP32 global L2 norm+1e-6)); no clipping applied.',
            'FP32_policy': 'Explicit separate run. Failure/OOM stops with retained logs; never silently lower precision/batch/patch.',
            'later_stages': 'Not assessed: initial frozen stage only; partial/joint need their own trained state and auxiliary objectives.',
            'helpers': HELPERS, 'max_wait_seconds': args.max_wait_seconds, 'diagnostic_deadline_seconds': args.deadline}


def vector_metrics(grads, indices):
    """CPU FP64 energies/dots; None means a genuine unused/frozen path."""
    import torch
    result = {}
    finite = all(g is None or bool(torch.isfinite(g).all()) for values in grads.values() for i, g in enumerate(values) if i in indices)
    if not finite: return {'finite': False, 'norms': {key: None for key in grads}, 'cos_A_seg': None, 'cos_B_seg': None}
    energy = {key: sum(float(g.double().square().sum()) for i, g in enumerate(values) if i in indices and g is not None) for key, values in grads.items()}
    dots = {key: sum(float(a.double().mul(b.double()).sum()) for i, (a, b) in enumerate(zip(grads['segmentation'], grads[key]))
                     if i in indices and a is not None and b is not None) for key in ('A', 'B')}
    norms = {key: math.sqrt(value) for key, value in energy.items()}
    result.update(finite=True, energy=energy, norms=norms, gaussian_dot_seg=dots,
                  tensor_counts={key: {'unused_None': sum(values[i] is None for i in indices),
                       'nonzero': sum(values[i] is not None and bool(torch.count_nonzero(values[i])) for i in indices)} for key, values in grads.items()})
    for key in ('A', 'B'):
        denom = norms[key]*norms['segmentation']
        result['cos_'+key+'_seg'] = dots[key]/denom if denom else None
        result[key+'_over_seg_norm'] = norms[key]/norms['segmentation'] if norms['segmentation'] else None
        residual = 0.
        for i in indices:
            values = [grads[k][i] for k in ('segmentation', key, 'total'+key)]
            present = next((x for x in values if x is not None), None)
            if present is None: continue
            a, b, total = [x.double() if x is not None else torch.zeros_like(present, dtype=torch.float64) for x in values]
            residual += float((total-a-b).square().sum())
        result[key+'_direct_vs_sum_relative_residual'] = math.sqrt(residual)/(norms['total'+key]+norms['segmentation']+norms[key]+1e-30)
    return result


def fixture(args):
    guard, prepare, diagnostic, shadow = helpers()
    checker, _, _ = variant_guard()
    report = checker.fixture()
    report.update(role='SYNTHETIC_PARAMETER_GRADIENT_FIXTURE_NOT_MODEL_RESULTS', actual_model_forward_executed=False)
    if importlib.util.find_spec('torch') is not None:
        import torch
        torch.set_num_threads(2)
        x = torch.tensor([1., -2., 3.], requires_grad=True)
        s = (x.square()).sum(); a = (x*torch.tensor([2., 1., -1.])).sum(); b = a*3
        terms = {'segmentation': s, 'A': a, 'B': b, 'totalA': s+a, 'totalB': s+b}
        gradients = {k: [torch.autograd.grad(value, x, retain_graph=True)[0]] for k, value in terms.items()}
        measured = vector_metrics(gradients, {0})
        assert measured['A_direct_vs_sum_relative_residual'] == measured['B_direct_vs_sum_relative_residual'] == 0
        assert measured['norms']['totalB'] > measured['norms']['totalA']
        report['parameter_sum_and_group_metric_fixture'] = measured
        need(not torch.cuda.is_initialized(), 'CPU fixture initialized CUDA')
    if args.output:
        need(not args.output.exists(), 'Use a fresh fixture output file'); write(args.output, report)
    print(json.dumps(report, indent=2)); return


def worker(args):
    signal.signal(signal.SIGALRM, signal.SIG_DFL); signal.alarm(args.deadline)
    guard, prepare, diagnostic, shadow = helpers()
    _, _, inspection = variant_guard()
    guard.source_guard(ROOT)
    canonical = canonical_guard(guard)
    output = args.output.absolute()
    owner = read(output/'CONTROLLER_STATUS.json')
    need(owner.get('controller_pid') == os.getppid() and owner.get('script_sha256') == sha(__file__), 'Worker requires its owning controller')
    config_path = ROOT/'results/gta_enhanced200x125/training/config.json'
    config = read(config_path)
    manifest_path = ROOT/'dataset/cache/manifest.json'
    need(sha(manifest_path) == guard.MANIFEST and sha(args.reference_bank) == BANK, 'Actual manifest/bank identity differs')
    manifest = read(manifest_path); train_ids = guard.config_guard(config, manifest)
    reference = read(args.reference_bank)
    need(len(reference['samples']) == 56 and Counter(x['case_id'] for x in reference['samples']) == Counter({x: 2 for x in train_ids})
         and reference['augmentation'] is False and reference['bank_epoch'] == 0, 'Not the fixed train28/56 noaug bank')
    tracked = {str(config_path): sha(config_path), str(manifest_path): sha(manifest_path), str(args.reference_bank): sha(args.reference_bank),
               str(Path(__file__).resolve()): sha(__file__)}
    for component, expected in DONORS.items():
        path = Path(config[component+'_init']); need(sha(path) == expected, 'Actual initial donor differs: '+component)
        tracked[str(path)] = expected
    sys.path.insert(0, str(VARIANT/'code'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import default_collate
    from checkpoint_utils import load_audited
    from classification_data import CohortDataset, internal_split
    from data import load_manifest
    from train import make_model, source_hash
    from losses import objective, gaussian_region_balanced_mse
    need(source_hash() == B_SOURCE and internal_split(manifest, 0, config['split_seed']) == config['internal_split'], 'Live B or patient split differs')
    runtime = guard.runtime_profile(); torch.set_num_threads(8)
    seed = config['seed']+config['fold']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = make_model('gta', True)
    for component, module, policy in [('backbone', model.base_model, 'reinit'), ('classifier', model.classifier, 'load')]:
        audit = load_audited(module, config[component+'_init'], component, config[component+'_domain'], policy)
        need(audit == next(x for x in config['init_audits'] if x['component'] == component), 'Actual initial transfer audit differs')
    model.set_classifier_stage('frozen'); model.train()
    identity = prepare.state_identity(model)
    need(identity['state_sha256'] == INITIAL and identity['tensor_count'] == 327, 'Actual reconstructed initial state differs')
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    write(output/'reconstructed_initial_state.json', identity)
    need(not torch.cuda.is_initialized(), 'CPU initialization unexpectedly created a CUDA context')
    pre_CUDA_GPU = guard.gpu_snapshot()
    need(pre_CUDA_GPU['all_compute_GPUs_idle'], 'GPU became occupied before first diagnostic CUDA allocation')
    model.to('cuda'); model.set_classifier_stage('frozen'); model.train()
    rng_cpu = torch.get_rng_state().clone(); rng_cuda = [x.clone() for x in torch.cuda.get_rng_state_all()]
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names = [name for name, _ in params]
    groups = {'segmentation_network': {i for i, name in enumerate(names) if not name.startswith(('classifier.', 'gau_final.'))},
              'classifier': {i for i, name in enumerate(names) if name.startswith('classifier.')},
              'Gaussian_head': {i for i, name in enumerate(names) if name.startswith('gau_final.')},
              'main_segmentation_optimizer_group': {i for i, name in enumerate(names) if not name.startswith('classifier.')},
              'all_trainable': set(range(len(params)))}
    need(not groups['classifier'], 'Frozen classifier unexpectedly has trainable parameters')
    dataset = CohortDataset(load_manifest(manifest_path), train_ids, fold=0, role='train', sampling=config['sampling'],
                            sampling_fov=config['sampling_fov'], steps=reference['samples'][-1]['index']+1,
                            patch_size=128, seed=seed, augment=False)
    dataset.set_epoch(0)
    scale = 1024. if args.precision == 'amp' else 1.
    protocol = plan(args) | {'status': 'actual_diagnostic_running', 'canonical': canonical, 'runtime': runtime,
                            'source_inspection': inspection, 'tracked_input_sha256': tracked,
                            'initial_state_sha256': identity['state_sha256'], 'historical_main_theta0_verified': False,
                            'GPU_immediately_before_CUDA_allocation': pre_CUDA_GPU,
                            'gradient_scaling': 'autograd.grad(loss*1024) then FP32/1024 matches initial main GradScaler unscaled-gradient meaning; scale1 in separate FP32 mode',
                            'frozen_classifier_parameters': {'n_tensors': sum(name.startswith('classifier.') for name, p in model.named_parameters()),
                                'n_elements': sum(p.numel() for name, p in model.named_parameters() if name.startswith('classifier.'))},
                            'active_parameters': {name: {'n_tensors': len(indices), 'n_elements': sum(params[i][1].numel() for i in indices)} for name, indices in groups.items()}}
    write(output/'diagnostic_plan.json', protocol)
    write(output/'fixed_sampling_bank.json', reference)
    rows = []; patch_rows = []; started = time.monotonic()
    with (output/'batch_metrics.jsonl').open('x', encoding='utf-8') as ledger:
        for ordinal in range(28):
            samples = []; items = reference['samples'][ordinal*2:ordinal*2+2]
            for item in items:
                sample = dataset[item['index']]
                diagnostic.compare_bank(diagnostic.sample_identity({'index': item['index'], 'case_id': item['case_id']}, sample), item)
                samples.append(sample)
            batch = {key: value.to('cuda') if torch.is_tensor(value) else value for key, value in default_collate(samples).items()}
            model.load_state_dict(initial, strict=True); model.set_classifier_stage('frozen'); model.train()
            torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state_all(rng_cuda)
            need(all(p.grad is None for p in model.parameters()), 'A parameter .grad was populated')
            torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast('cuda', enabled=args.precision == 'amp', dtype=torch.float16):
                prediction = model(batch['image'])
                totalB, details, terms = objective(prediction, batch, 'gta', epoch=0, segmentation=config['seg_loss'], classifier_active=False,
                    classifier_weight=.1, gaussian_weight=10., guidance_weight=.1,
                    anatomy_reweighting=not config['disable_anatomy_reweighting'], return_terms=True)
                s = terms['segmentation']; target = .5+.5*batch['gaussian'].float()
                a = 10*F.mse_loss(prediction['gaussian'].float(), target)
                b = terms['gaussian']
                torch.testing.assert_close(b, 10*shadow.region_losses(prediction['gaussian'].float(), target, batch['label'][:, None] > 0)['B'], rtol=1e-5, atol=1e-7)
                objectives = {'segmentation': s, 'A': a, 'B': b, 'totalA': s+a, 'totalB': totalB}
            gradients = {}; direct_norms = {}; nonfinite = {}
            for j, (key, loss) in enumerate(objectives.items()):
                need(bool(torch.isfinite(loss)), 'Nonfinite actual diagnostic loss: '+key)
                raw = torch.autograd.grad(loss*scale, [p for _, p in params], allow_unused=True, retain_graph=j < 4)
                values = [value.detach().float()/scale if value is not None else None for value in raw]
                nonfinite[key] = [names[i] for i, value in enumerate(values) if value is not None and not bool(torch.isfinite(value).all())]
                if nonfinite[key]: direct_norms[key] = None
                else:
                    norms = [torch.linalg.vector_norm(value, 2.) for value in values if value is not None]
                    value_norm = float(torch.linalg.vector_norm(torch.stack(norms), 2.)) if norms else 0.
                    direct_norms[key] = value_norm if math.isfinite(value_norm) else None
                gradients[key] = [value.cpu() if value is not None else None for value in values]
                del raw, values
            foreground = [int((sample['label'] > 0).sum()) for sample in samples]
            for i, name in enumerate(names):
                if name.startswith('gau_final.'):
                    need(gradients['segmentation'][i] is None, 'Segmentation objective unexpectedly reaches Gaussian head')
                if name.startswith('ds_heads.'):
                    need(gradients['A'][i] is None and gradients['B'][i] is None, 'Gaussian objective unexpectedly reaches DS sibling heads')
            empty_checks = []
            for i, n in enumerate(foreground):
                if n == 0 or n == 128**3:
                    q = prediction['gaussian'][i:i+1].detach().float(); t = target[i:i+1]
                    torch.testing.assert_close(gaussian_region_balanced_mse(q, t, batch['label'][i:i+1]), F.mse_loss(q, t))
                    empty_checks.append({'patch_in_batch': i, 'foreground_voxels': n, 'A_equals_B_when_one_region_present': True})
                patch_rows.append({'batch': ordinal, 'patch_in_batch': i, 'case_id': items[i]['case_id'], 'index': items[i]['index'], 'foreground_voxels': n})
            changed_buffers = [name for name, value in model.named_buffers() if not torch.equal(value.detach().cpu(), initial[name])]
            row = {'batch': ordinal, 'samples': [{'index': x['index'], 'case_id': x['case_id']} for x in items],
                   'foreground_voxels': foreground, 'losses': {k: float(value.detach()) for k, value in objectives.items()},
                   'parameter_group_metrics': {name: vector_metrics(gradients, indices) for name, indices in groups.items()},
                   'FP32_global_direct_norms': direct_norms, 'nonfinite_parameter_names': nonfinite,
                   'nonfinite_global_norm_objectives': [k for k, v in direct_norms.items() if v is None],
                   'implied_clip12_factors': {k: min(1., 12/(direct_norms[k]+1e-6)) if direct_norms[k] is not None else None for k in ('totalA', 'totalB')},
                   'clip_applied': False, 'AMP_derivative_scale_removed': scale, 'BN_or_other_buffers_changed_in_diagnostic_copy': changed_buffers,
                   'one_region_patch_checks': empty_checks, 'peak_GPU_GiB': torch.cuda.max_memory_allocated()/2**30}
            row['Gaussian_head_energy_share_and_clipped_shared_norm'] = {}
            for key in ('A', 'B'):
                head = row['parameter_group_metrics']['Gaussian_head']; all_group = row['parameter_group_metrics']['all_trainable']
                shared = row['parameter_group_metrics']['segmentation_network']; factor = row['implied_clip12_factors']['total'+key]
                valid = head['finite'] and all_group['finite'] and shared['finite']
                denom = all_group['energy']['total'+key] if valid else None
                row['Gaussian_head_energy_share_and_clipped_shared_norm'][key] = {
                    'Gaussian_head_direct_total_energy_share': head['energy']['total'+key]/denom if denom else None,
                    'shared_segmentation_direct_total_norm': shared['norms']['total'+key] if valid else None,
                    'implied_clipped_shared_segmentation_norm': shared['norms']['total'+key]*factor if valid and factor is not None else None}
            need(all(torch.equal(parameter.detach().cpu(), initial[name]) for name, parameter in model.named_parameters()), 'Diagnostic parameter changed')
            need(all(p.grad is None for p in model.parameters()), 'autograd.grad unexpectedly populated parameter .grad')
            ledger.write(json.dumps(row, allow_nan=False)+'\n'); ledger.flush(); rows.append(row)
            print(json.dumps({'completed_batches': ordinal+1, 'total_batches': 28, 'precision': args.precision,
                              'clip12': row['implied_clip12_factors'], 'nonfinite_gradient_tensors': {k: len(v) for k, v in nonfinite.items()},
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
            del gradients, prediction, objectives, batch, s, a, b, totalB, terms
        model.load_state_dict(initial, strict=True)
    need(prepare.state_identity(model)['state_sha256'] == INITIAL, 'Diagnostic copy could not restore all327 initial state tensors')
    for path, expected in tracked.items(): need(sha(path) == expected, 'Tracked source/input changed during diagnostic')
    guard.source_guard(ROOT); variant_guard()
    need(canonical_guard(guard) == canonical, 'Canonical immutable evidence changed during diagnostic')
    small = sorted((x for x in patch_rows if x['foreground_voxels'] > 0), key=lambda x: (x['foreground_voxels'], x['batch'], x['patch_in_batch']))[:5]
    summary = {'role': 'ACTUAL_INITIAL_TRAINMODE_PARAMETER_DIAGNOSTIC_NOT_MODEL_RESULTS', 'precision': args.precision,
               'n_batches': 28, 'n_patches': 56, 'initial_state_sha256': INITIAL, 'historical_main_theta0_verified': False,
               'parameter_updates': False, 'optimizer_created': False, 'parameter_grad_populated': False, 'held_out_GT_read': False,
               'diagnostic_copy_buffers_may_change_during_forward': True, 'diagnostic_copy_restored_to_initial_at_end': True,
               'nonfinite_batches': [r['batch'] for r in rows if any(r['nonfinite_parameter_names'].values()) or any(v is None for v in r['FP32_global_direct_norms'].values())],
               'batches_implying_clip12': {k: [r['batch'] for r in rows if r['implied_clip12_factors'][k] is not None and r['implied_clip12_factors'][k] < 1] for k in ('totalA', 'totalB')},
               'top5_small_positive_FG_patches': small, 'top5_batch_observations': [rows[x['batch']] for x in small],
               'empty_patch_count': sum(x['foreground_voxels'] == 0 for x in patch_rows), 'wall_seconds': time.monotonic()-started,
               'limitations': ['No optimizer/GradScaler update or training trajectory; clipping factors are implied, not applied.',
                   'Only initial frozen stage; classification/guidance auxiliary losses inactive, frozen classifier gradients zero by design while its prior features remain active.',
                   'Nonfinite scaled gradients flag AMP overflow. Main also raises if clip_norm is nonfinite before scaler.step; this tool neither steps nor adapts scale.',
                   'FP32 mode disables autocast but retains/records actual runtime TF32 flags; it is not an asserted full IEEE FP32-kernel run.',
                   'No augmentation; stratified train28 bank is not natural sampler frequency or independent validation.',
                   'Each batch resets initial weights/buffers/RNG, so this is standalone minibatch sensitivity, not28 updates.',
                   'A/B share one forward/dropout/BN outcome; AMP separate backward rounding is reported via direct-vs-sum residual.',
                   'Small-FG patch entries reference their B2 batch gradients, not an isolated per-patch parameter gradient.',
                   'Gradient magnitude/clipping cannot establish Dice/ASSD, connectivity or benefit.']}
    def quantiles(values):
        values = [x for x in values if x is not None and math.isfinite(x)]
        return {'n_finite': len(values), 'min': min(values) if values else None, 'median': float(np.median(values)) if values else None,
                'p90': float(np.quantile(values, .9)) if values else None, 'max': max(values) if values else None}
    summary['per_batch_distribution_not_pooled_cosine'] = {key: {
        'direct_total_global_norm': quantiles([r['FP32_global_direct_norms']['total'+key] for r in rows]),
        'implied_clip12_factor': quantiles([r['implied_clip12_factors']['total'+key] for r in rows]),
        'shared_gaussian_over_segmentation_norm': quantiles([r['parameter_group_metrics']['segmentation_network'].get(key+'_over_seg_norm') for r in rows]),
        'shared_cosine_with_segmentation': quantiles([r['parameter_group_metrics']['segmentation_network'].get('cos_'+key+'_seg') for r in rows])} for key in ('A', 'B')}
    write(output/'summary.json', summary)
    artifacts = {name: sha(output/name) for name in ('diagnostic_plan.json', 'fixed_sampling_bank.json', 'reconstructed_initial_state.json', 'batch_metrics.jsonl', 'summary.json')}
    write(output/'DIAGNOSTIC_COMPLETE.json', {'status': 'completed', 'n_batches': 28, 'n_patches': 56, 'artifacts_sha256': artifacts,
                                            'parameter_updates': False, 'held_out_GT_read': False, 'buffer_state_restored_at_end': True,
                                            'nonfinite_batches': summary['nonfinite_batches']})


def run(args):
    import fcntl
    need(ROOT == Path(__file__).resolve().parents[1], 'Run this tool only in named NEW ROOT/ops')
    need(Path(sys.executable).absolute() == Path('/root/miniconda3/envs/gta-asoca311/bin/python'), 'Use actual main Python environment')
    output = args.output.absolute()
    need(output.is_relative_to(ROOT/'results') and not output.exists(), 'Use fresh NEW/results diagnostic output')
    need(args.reference_bank.resolve().is_relative_to(ROOT/'ops_inputs') and sha(args.reference_bank) == BANK, 'Use uploaded immutable reference bank under ops_inputs')
    guard, _, _, _ = helpers(); variant_guard()
    own = (ROOT/'.initial_gaussian_parameter_diagnostic.lock').open('a+')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True)
    state = {'status': 'waiting_canonical20_and_idle_GPU', 'controller_pid': os.getpid(), 'script_sha256': sha(__file__),
             'started_at_utc': guard.now(), 'gpu_lock_acquired': False}
    def save(**value):
        state.update(value, updated_at_utc=guard.now()); write(output/'CONTROLLER_STATUS.json', state)
    cancellation = guard.OwnedCancellation(); child = None; lease = None
    signal.signal(signal.SIGTERM, cancellation.request); signal.signal(signal.SIGINT, cancellation.request)
    try:
        fixture_args = argparse.Namespace(**(vars(args) | {'output': output/'cpu_fixture.json'}))
        fixture(fixture_args)
        checked = read(output/'cpu_fixture.json')
        need(checked.get('status') == 'passed' and checked.get('actual_candidate_torch_verified') is True
             and 'parameter_sum_and_group_metric_fixture' in checked, 'Actual CPU candidate/parameter fixture must pass before any GPU diagnostic')
        deadline = time.monotonic()+args.max_wait_seconds
        while True:
            need(time.monotonic() < deadline, 'Timed out waiting for canonical20/80 completion and idle GPU')
            ready = (ROOT/'POSTFIT_VERIFIED.json').exists() and read(ROOT/'POSTFIT_STATUS.json').get('status') == 'completed'
            if ready:
                canonical = canonical_guard(guard)
                first = guard.gpu_snapshot()
                if first['all_compute_GPUs_idle']:
                    candidate = (ROOT/'.segmentation_stage.lock').open('a+')
                    try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: candidate.close()
                    else:
                        lease = candidate; time.sleep(3); second = guard.gpu_snapshot()
                        if second['all_compute_GPUs_idle']: break
                        fcntl.flock(candidate, fcntl.LOCK_UN); candidate.close(); lease = None
            save(); time.sleep(15)
        guard.source_guard(ROOT); canonical_guard(guard); variant_guard()
        command = [sys.executable, '-u', str(Path(__file__).absolute()), '--worker', '--precision', args.precision,
                   '--reference-bank', str(args.reference_bank.absolute()), '--output', str(output), '--deadline', str(args.deadline)]
        write(output/'CONTROLLER_PROTOCOL.json', plan(args) | {'status': 'actual_diagnostic_authorized', 'command': command,
              'script_sha256': sha(__file__), 'canonical': canonical, 'GPU_idle_observations': [first, second],
              'actual_cpu_fixture_sha256': sha(output/'cpu_fixture.json')})
        env = dict(os.environ); env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        began = time.monotonic()
        with (output/'worker.log').open('x', encoding='utf-8') as log:
            cancellation.launching = True
            try:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                need(os.getpgid(child.pid) == child.pid, 'Owned diagnostic process group differs')
            finally: cancellation.launching = False
            cancellation.raise_pending()
            save(status='diagnosing', child_pid=child.pid, gpu_lock_acquired=True)
            while child.poll() is None:
                need(time.monotonic()-began < args.deadline+60, 'Owned diagnostic exceeded bounded deadline')
                time.sleep(10)
        returncode = child.returncode; child = None
        need(returncode == 0, 'Diagnostic failed/OOM/timeout; inspect retained log; no automatic precision change')
        completed = read(output/'DIAGNOSTIC_COMPLETE.json')
        need(completed.get('status') == 'completed' and completed.get('n_batches') == 28 and completed.get('n_patches') == 56
             and completed.get('parameter_updates') is False and completed.get('held_out_GT_read') is False, 'Diagnostic completion scope differs')
        for name, expected in completed['artifacts_sha256'].items(): need(sha(guard.safe_member(output, name)) == expected, 'Diagnostic artifact changed')
        need(canonical_guard(guard) == canonical, 'Canonical evidence changed')
        save(status='completed', child_pid=None, completion_sha256=sha(output/'DIAGNOSTIC_COMPLETE.json'))
    except BaseException as error:
        cancellation.cleaning = True
        guard.cleanup_owned_child(child); child = None
        save(status='failed_stop', child_pid=None, error=repr(error), owned_child_exit_confirmed=True)
        raise
    finally:
        cancellation.cleaning = True; guard.cleanup_owned_child(child)
        if lease is not None: fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
        fcntl.flock(own, fcntl.LOCK_UN); own.close()
        try: save(gpu_lock_acquired=False, locks_released_at_utc=guard.now())
        except Exception: pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', action='store_true'); modes.add_argument('--fixture', action='store_true')
    modes.add_argument('--diagnose', action='store_true'); modes.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--precision', choices=('amp', 'fp32'), default='amp')
    p.add_argument('--reference-bank', type=Path, default=ROOT/'ops_inputs/gaussian_epoch070_fixed_sampling_bank.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--max-wait-seconds', type=int, default=7200); p.add_argument('--deadline', type=int, default=2400)
    args = p.parse_args()
    need(1 <= args.max_wait_seconds <= 43200 and 60 <= args.deadline <= 7200, 'Invalid bounded time settings')
    if args.plan: print(json.dumps(plan(args), indent=2)); return
    if args.fixture: fixture(args); return
    need(sys.platform == 'linux' and args.output is not None, 'Actual diagnostic needs named Linux host and fresh output')
    if args.worker: worker(args)
    else: run(args)


if __name__ == '__main__': main()
