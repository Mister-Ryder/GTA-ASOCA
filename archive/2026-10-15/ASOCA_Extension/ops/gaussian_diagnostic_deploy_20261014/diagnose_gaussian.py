"""Offline Gaussian diagnostics; no optimizer, training or checkpoint selection.

--plan is stdlib-only; --cpu-readiness validates inputs without a forward/CUDA.
Diagnostics use eval-mode, batch-one FP32 output derivatives, not parameter updates.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def source_identity(code):
    used = {'train.py', 'model.py', 'losses.py', 'classification_data.py', 'checkpoint_utils.py', 'data.py'}
    paths = [p for p in sorted(code.rglob('*.py'), key=lambda p: p.relative_to(code).as_posix())
             if '__pycache__' not in p.parts and (p.name in used or 'historical' in p.relative_to(code).parts)]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.relative_to(code).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest(), {p.relative_to(code).as_posix(): sha(p) for p in paths}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@contextmanager
def gpu_lease(run_root):
    """Share the formal controller's nonblocking kernel lock; never wait for GPU."""
    import fcntl
    with (run_root / '.segmentation_stage.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Segmentation stage is busy; no GPU diagnostic started') from error
        try:
            gpu_idle()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_idle():
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout.strip()
    if active:
        raise RuntimeError('GPU occupied; no diagnostic forward or CUDA allocation started: ' + active)


def sample_identity(item, sample):
    """Hash actual CPU patch bytes without persisting CT or supervision arrays."""
    import numpy as np
    if sample['case_id'] != item['case_id']:
        raise ValueError('Frozen sampler no longer matches the planned training patient')
    arrays = {}
    for key in ('image', 'label', 'gaussian', 'priors', 'start_zyx'):
        value = sample[key]
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite sampled bank input: ' + key)
        if key == 'label' and not np.isin(value, (0, 1)).all():
            raise ValueError('Sampled coronary label must remain binary')
        if key == 'gaussian' and (value.min() < 0 or value.max() > 1):
            raise ValueError('Sampled Gaussian field outside [0,1]')
        if key == 'priors' and (value.min() < 0 or value.max() > 6):
            raise ValueError('Sampled auxiliary targets outside0..6')
        arrays[key] = {'dtype': value.dtype.str, 'shape': list(value.shape),
                       'content_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
    stratum = sample['sampling_stratum']
    return {**item, 'arrays': arrays,
            'sampling_stratum': {'value': stratum, 'dtype': 'utf8', 'shape': [],
                                 'content_sha256': hashlib.sha256(stratum.encode('utf-8')).hexdigest()}}


def compare_bank(actual, reference):
    if actual != reference:
        raise ValueError('Fixed sampling bank differs in patient/index, CT/label/field/priors/start bytes, '
                         'sampling stratum, source, or sampling protocol; no cross-checkpoint comparison is valid')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--training-config', type=Path)
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parents[1] / 'code')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference-bank', type=Path,
                        help='Previous fixed_sampling_bank.json; exact patch-content identity required')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--bank-epoch', type=int, default=0)
    parser.add_argument('--patches-per-case', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--cpu-readiness', action='store_true')
    args = parser.parse_args()
    if args.patches_per_case < 1 or args.bank_epoch < 0:
        parser.error('Positive patches-per-case and nonnegative bank epoch required')
    code, checkpoint, manifest_path = args.code.resolve(), args.checkpoint.resolve(), args.manifest.resolve()
    run_root = code.parent
    if code.name != 'code' or Path(__file__).resolve().parents[1] != run_root:
        parser.error('--code must be this diagnostic script\'s own ROOT/code')
    folder = checkpoint.parent.parent if checkpoint.parent.name == 'checkpoints' else checkpoint.parent
    config_path = (args.training_config or folder / 'config.json').resolve()
    if (not checkpoint.is_relative_to(run_root / 'results') or config_path != folder / 'config.json'
            or manifest_path != run_root / 'dataset/cache/manifest.json'
            or (args.output is not None and not args.output.resolve().is_relative_to(run_root))):
        parser.error('Checkpoint/config/cache manifest/output must belong to the same code.parent run ROOT')
    if not code.is_dir():
        parser.error('Frozen code directory is missing')
    code_hash, source_files = source_identity(code)
    inputs = {'checkpoint': checkpoint, 'manifest': manifest_path, 'training_config': config_path}
    if args.reference_bank is not None:
        inputs['reference_bank'] = args.reference_bank.resolve()
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else None
    plan = {'schema_version': 1, 'role': 'train28 offline Gaussian diagnostic; not final test or model selection',
            'run_root': str(run_root), 'code': str(code), 'source_sha256': code_hash, 'source_files_sha256': source_files,
            'source_hash_algorithm': 'gta_fidelity_selected_python_path_and_bytes_v1',
            'diagnostic_script_sha256': sha(__file__), 'inputs': {k: str(v) for k, v in inputs.items()},
            'input_sha256': {k: sha(v) if v.is_file() else None for k, v in inputs.items()},
            'missing_inputs': missing, 'seed': args.seed, 'bank_epoch': args.bank_epoch,
            'patches_per_training_case': args.patches_per_case, 'bank_size': 28 * args.patches_per_case,
            'device_for_explicit_diagnostic': args.device, 'model_mode': 'eval, no CT augmentation',
            'probe_space': 'detached FP32 main logits, batch one; DS sibling derivatives excluded',
            'parameter_updates': False, 'optimizer_created': False, 'outer8_used': False, 'formal_final_test': False,
            'bank_population': 'patient-stratified original training-sampler indices; not natural patch-frequency estimate',
            'readiness_scope': 'cache array headers/sampling keys and finite model parameters only; '
                               'no full cache-content hashes, patch traversal, forward, or CUDA',
            'fixed_bank_scope': 'explicit diagnostics persist dtype/shape/content SHA of actual CT/label/field/priors/start '
                                'and sampling stratum; CPU readiness does not certify patch-content identity'}
    if config is not None and manifest is not None:
        split = config['internal_split']; train_ids = split['train']; val_ids = split['val']; fold = config['fold']
        outer = manifest['folds'][fold]
        if (len(train_ids) != 28 or len(set(train_ids)) != 28 or len(val_ids) != 4 or len(set(val_ids)) != 4
                or set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(outer['train'])
                or set(train_ids) & set(outer['val']) or config.get('trained_case_ids') != train_ids):
            raise ValueError('Expected isolated training28/internal4 in the frozen outer32')
        if config.get('smoke') is not False or config['method'] not in ('gta', 'gta_no_prior', 'gta_no_balance'):
            raise ValueError('Requires a non-smoke GTA checkpoint with the Gaussian branch')
        if (config.get('cohort') != 'internal28' or config.get('internal_selection_valid') is not True
                or config.get('outer8_used_for_selection') is not False or config.get('allow_selection_overlap') is not False
                or Path(config['output']).resolve() != folder or Path(config['manifest']).resolve() != manifest_path):
            raise ValueError('Recorded run ROOT or isolated training28 protocol markers differ')
        if (config.get('source_hash_algorithm') != 'gta_fidelity_selected_python_path_and_bytes_v1'
                or config['source_sha256'] != code_hash
                or config['manifest_sha256'] != plan['input_sha256']['manifest']
                or config.get('source_files_sha256') != source_files):
            raise ValueError('Source or manifest differs from the checkpoint training protocol')
        plan.update(training_patients=train_ids, excluded_internal4=val_ids, fold=fold,
                    patch_size=config['patch_size'], sampling=config['sampling'], sampling_fov=config['sampling_fov'],
                    method=config['method'], segmentation_loss=config['seg_loss'], decoder_DS=not config['no_ds'])
    if args.plan:
        print(json.dumps(plan, indent=2, allow_nan=False)); return
    if missing:
        raise FileNotFoundError('Missing inputs: ' + ', '.join(missing))
    if not args.cpu_readiness and args.output is None:
        parser.error('--output is required for an explicit diagnostic')

    def verify_identity():
        if (source_identity(code) != (code_hash, source_files)
                or sha(__file__) != plan['diagnostic_script_sha256']
                or any(sha(v) != plan['input_sha256'][k] for k, v in inputs.items())):
            raise RuntimeError('Source/checkpoint/config/manifest/reference-bank changed during diagnostics/readiness')

    # Do not create bytecode under the frozen source tree, even during readiness.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    import numpy as np
    import torch
    from scipy import ndimage as ndi
    from classification_data import CohortDataset, internal_split
    from data import load_manifest, case_directory
    from losses import objective
    from train import make_model
    manifest = load_manifest(manifest_path)
    if internal_split(manifest, fold, config['split_seed']) != config['internal_split']:
        raise ValueError('Recorded28/4 differs from the frozen deterministic internal split')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    saved_config = saved.get('config', {})
    if ('model' not in saved or any(config.get(k) != saved_config.get(k)
                                  for k in (set(config) | set(saved_config)) - {'resume', 'plan'})
            or not isinstance(saved.get('epoch'), int) or not 0 <= saved['epoch'] < config['epochs']):
        raise ValueError('Checkpoint configuration/epoch differs from recorded training')
    if saved['epoch'] < config.get('backbone_warmup_epochs', 0):
        raise ValueError('Backbone warm-up checkpoint did not train the enhanced Gaussian branch')
    cases = {case['case_id']: case for case in manifest['cases']}
    cache_headers = {}
    for case_id in train_ids:
        cache = case_directory(manifest, case_id)
        names = ('image', 'label', 'gaussian', 'skeleton', 'radius', 'branch_id', 'depth')
        headers = {}
        for name in names:
            path = cache / (name + '.npy')
            array = np.load(path, mmap_mode='r')
            headers[name] = {'shape': list(array.shape), 'dtype': array.dtype.str,
                             'file_size': path.stat().st_size}
            del array
        if any(value['shape'] != cases[case_id]['shape_zyx'] for value in headers.values()):
            raise ValueError('Cache grid differs: ' + case_id)
        metadata_path = cache / 'case.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        for key in ('case_id', 'shape_zyx', 'spacing_zyx_mm', 'source_sha256', 'preprocessing_version'):
            if metadata.get(key) != cases[case_id].get(key):
                raise ValueError('Cache metadata differs from manifest: ' + case_id + '/' + key)
        cache_headers[case_id] = {'arrays': headers, 'case_metadata_sha256': sha(metadata_path)}
        with np.load(cache / 'sampling.npz') as coordinates:
            if not {'foreground', 'skeleton', 'thin', 'junction'}.issubset(coordinates.files):
                raise ValueError('Sampling coordinates missing: ' + case_id)
    # Match frozen data.py:174-175 case choice; select indices before accessing GT.
    counts = {case_id: 0 for case_id in train_ids}; bank = []
    for index in range(28 * args.patches_per_case * 100):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.bank_epoch, index]))
        case_id = train_ids[int(rng.integers(len(train_ids)))]
        if counts[case_id] < args.patches_per_case:
            bank.append({'index': index, 'case_id': case_id}); counts[case_id] += 1
        if len(bank) == plan['bank_size']:
            break
    if len(bank) != plan['bank_size']:
        raise ValueError('Deterministic bank did not cover all training patients')
    plan.update(checkpoint_epoch=saved['epoch'], bank=bank, patches_by_patient=counts, cache_headers=cache_headers,
                readiness='Cache headers/sampling keys and finite model parameters verified; '
                          'cache values and actual bank content are not verified; no forward or CUDA operation',
                torch=str(torch.__version__), numpy=np.__version__)
    model = make_model(config['method'], not config['no_ds'])
    model.load_state_dict(saved['model'], strict=True); del saved
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise FloatingPointError('Checkpoint contains nonfinite model tensors')
    model.requires_grad_(False); model.eval()
    verify_identity()
    if args.cpu_readiness:
        print(json.dumps(plan, indent=2, allow_nan=False)); return
    with gpu_lease(run_root) if args.device == 'cuda' else nullcontext():
        if args.output.exists() and any(args.output.iterdir()):
            raise RuntimeError('Use a fresh output directory; diagnostics are not overwritten')
        output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
        if args.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        dataset = CohortDataset(manifest, train_ids, fold=fold, role='train', sampling=config['sampling'],
                               sampling_fov=config['sampling_fov'], steps=bank[-1]['index'] + 1,
                               patch_size=config['patch_size'], seed=args.seed, augment=False)
        dataset.set_epoch(args.bank_epoch)
        fixed_bank = {'schema_version': 1, 'source_sha256': code_hash,
                      'manifest_sha256': plan['input_sha256']['manifest'],
                      'training_patients': train_ids, 'excluded_internal4': val_ids, 'fold': fold,
                      'seed': args.seed, 'bank_epoch': args.bank_epoch,
                      'patches_per_training_case': args.patches_per_case,
                      'patch_size': config['patch_size'], 'sampling': config['sampling'],
                      'sampling_fov': config['sampling_fov'], 'augmentation': False,
                      'role': 'training28 only; no checkpoint selection or independent validation',
                      'samples': [sample_identity(item, dataset[item['index']]) for item in bank]}
        if args.reference_bank is not None:
            compare_bank(fixed_bank, json.loads(args.reference_bank.read_text(encoding='utf-8')))
        write_json(output / 'fixed_sampling_bank.json', fixed_bank)
        plan['fixed_sampling_bank_sha256'] = sha(output / 'fixed_sampling_bank.json')
        plan['reference_bank_verified'] = args.reference_bank is not None
        verify_identity()
        if args.device == 'cuda':
            gpu_idle()  # Recheck immediately before the first CUDA allocation.
        model.to(torch.device(args.device))
        base, late = config['gaussian_weight'], config.get('gaussian_late_weight')
        ramp = min(1., max(0., (plan['checkpoint_epoch'] + 1 - config['gaussian_ramp_start']) /
                               max(1, config['gaussian_ramp_end'] - config['gaussian_ramp_start'])))
        weight = base if late is None else base + ramp * (late - base)
        plan['gaussian_lambda'] = weight
        plan['readiness'] = 'explicit offline diagnostic; no optimizer or parameter update'
        write_json(output / 'diagnostic_plan.json', plan)
        totals = {}; correlation_totals = {}; gradient_totals = {}

        def accumulate(target, name, values):
            target[name] = np.asarray(values, dtype=np.float64) + target.get(name, 0.)

        def moments(a, b):
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            return np.array([a.size, a.sum(), b.sum(), np.square(a).sum(), np.square(b).sum(), (a * b).sum()])

        def correlation(v):
            n, a, b, aa, bb, ab = v
            va, vb = aa - a*a/max(n, 1), bb - b*b/max(n, 1)
            return float(np.clip((ab-a*b/n) / math.sqrt(va*vb), -1., 1.)) if n and va > 1e-20 and vb > 1e-20 else None

        def gradients(v):
            sc, sm, gc, gm, dot, margin_dot = v
            return {'segmentation_common_energy': float(sc), 'segmentation_margin_energy': float(sm),
                    'gaussian_common_energy': float(gc), 'gaussian_margin_energy': float(gm),
                    'gaussian_common_fraction': float(gc/(gc+gm)) if gc+gm else None,
                    'gaussian_to_segmentation_norm_ratio': float(math.sqrt((gc+gm)/(sc+sm))) if sc+sm else None,
                    'gaussian_to_segmentation_margin_norm_ratio': float(math.sqrt(gm/sm)) if sm else None,
                    'full_cosine': float(np.clip(dot/math.sqrt((sc+sm)*(gc+gm)), -1., 1.)) if (sc+sm)*(gc+gm) else None,
                    'margin_cosine': float(np.clip(margin_dot/math.sqrt(sm*gm), -1., 1.)) if sm*gm else None}

        with (output / 'patch_metrics.jsonl').open('w', encoding='utf-8') as stream:
            for item, expected_sample in zip(bank, fixed_bank['samples']):
                sample = dataset[item['index']]
                compare_bank(sample_identity(item, sample), expected_sample)
                batch = {k: v.unsqueeze(0).to(args.device) for k, v in sample.items() if torch.is_tensor(v)}
                with torch.no_grad():
                    prediction = model(batch['image'])  # CT is the only model input.
                z = prediction['logits'].detach().float().clone().requires_grad_(True)
                q = model.gau_activation(model.gau_final(z))
                prediction.update(logits=z, gaussian=q)
                _, _, terms = objective(prediction, batch, config['method'], segmentation=config['seg_loss'],
                                        classifier_active=False, gaussian_weight=weight,
                                        anatomy_reweighting=not config['disable_anatomy_reweighting'], return_terms=True)
                s, g = [torch.autograd.grad(terms[k], z, retain_graph=True)[0].detach().cpu().numpy()[0]
                        for k in ('segmentation', 'gaussian')]
                fg = batch['label'].detach().cpu().numpy()[0] > 0
                field = batch['gaussian'].detach().cpu().numpy()[0, 0].astype(np.float64)
                predicted = q.detach().cpu().numpy()[0, 0].astype(np.float64)
                probability = z.detach().softmax(1).cpu().numpy()[0, 1].astype(np.float64)
                if not all(np.isfinite(v).all() for v in (s, g, field, predicted, probability)) or field.min() < 0 or field.max() > 1:
                    raise ValueError('Nonfinite output/derivative or field outside [0,1]')
                spacing = cases[item['case_id']]['spacing_zyx_mm']
                distance = ndi.distance_transform_edt(~fg, sampling=spacing) if fg.any() else np.full(fg.shape, np.inf)
                regions = {'all': np.ones(fg.shape, bool), 'foreground': fg, 'background': ~fg,
                           'outside_0_1mm': (~fg) & (distance <= 1),
                           'outside_1_3mm': (~fg) & (distance > 1) & (distance <= 3),
                           'background_gt3mm': (~fg) & (distance > 3)}
                target = .5 + .5 * field
                sc, sm = (s[0]+s[1])/math.sqrt(2), (s[1]-s[0])/math.sqrt(2)
                gc, gm = (g[0]+g[1])/math.sqrt(2), (g[1]-g[0])/math.sqrt(2)
                row = item | {'start_zyx': sample['start_zyx'].tolist(), 'sampling_stratum': sample['sampling_stratum'],
                              'foreground_fraction': float(fg.mean()), 'priors': sample['priors'].tolist(),
                              'field': {'mean': float(field.mean()), 'mean_squared': float(np.square(field).mean()),
                                        'minimum': float(field.min()), 'maximum': float(field.max()),
                                        'positive_fraction': float((field > 0).mean())},
                              'sample_identity': expected_sample,
                              'regions': {}, 'gradient_regions': {}, 'correlations': {}}
                for name, mask in regions.items():
                    t, p = target[mask], predicted[mask]; n = t.size
                    stats = [n, np.square(p-t).sum(), np.square(.5-t).sum(), t.sum(), np.square(t).sum()]
                    accumulate(totals, name, stats)
                    row['regions'][name] = {'n_voxels': n, 'mse': float(stats[1]/n) if n else None,
                                            'mse_constant05': float(stats[2]/n) if n else None}
                    energies = [np.square(v[mask].astype(np.float64)).sum() for v in (sc, sm, gc, gm)]
                    energies += [float((sc[mask]*gc[mask]+sm[mask]*gm[mask]).sum(dtype=np.float64)),
                                 float((sm[mask]*gm[mask]).sum(dtype=np.float64))]
                    accumulate(gradient_totals, name, energies); row['gradient_regions'][name] = gradients(energies)
                for region in ('all', 'foreground'):
                    for label, values in (('seg_probability_vs_field', probability), ('gaussian_vs_field', predicted)):
                        name = region + '.' + label; stats = moments(values[regions[region]], field[regions[region]])
                        accumulate(correlation_totals, name, stats); row['correlations'][name] = correlation(stats)
                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                del prediction, z, q, terms, batch
        mean_target = totals['all'][3] / totals['all'][0]
        summary = {'provenance': plan, 'bank_optimal_constant_target': float(mean_target), 'regions': {},
                   'gradient_regions': {k: gradients(v) for k, v in gradient_totals.items()},
                   'correlations': {k: correlation(v) for k, v in correlation_totals.items()},
                   'limitations': ['Patient-stratified training patches; not independent validation or natural patch frequencies.',
                                   'Eval-mode batch-one FP32 logits derivatives; excludes DS sibling paths and parameter updates.',
                                   'Regions overlap; each denominator is explicit. Shell distances are patch-local.',
                                   'Gradient regions partition logit coordinates, not originating regional losses; Gaussian3x3x3 convolutions cross region boundaries.',
                                   'Common is a local softmax-null direction; parameter Jacobians can couple common and margin, so common energy is not proof of ineffective training.',
                                   'Bank-optimal constant uses this bank GT; descriptive diagnostic, not an independent fitted model.',
                                   'MSE, correlations and gradient energy cannot establish connectivity or Dice/ASSD benefit.']}
        for name, (n, error, constant05, target_sum, target_squared) in totals.items():
            summary['regions'][name] = {'n_voxels': int(n), 'mse': float(error/n) if n else None,
                                       'weighted_mse': float(weight*error/n) if n else None,
                                       'mse_constant05': float(constant05/n) if n else None,
                                       'field_mean': float(2*target_sum/n-1) if n else None,
                                       'field_mean_squared': float(max(0., 4*target_squared/n-4*target_sum/n+1)) if n else None,
                                       'mse_bank_optimal_constant': float(max(0., (target_squared-2*mean_target*target_sum+n*mean_target**2)/n)) if n else None}
        verify_identity()
        if sha(output / 'fixed_sampling_bank.json') != plan['fixed_sampling_bank_sha256']:
            raise RuntimeError('Recorded fixed bank changed during diagnostics')
        write_json(output / 'summary.json', summary)
        write_json(output / 'DIAGNOSTIC_COMPLETE.json', {'status': 'completed', 'n_patches': len(bank),
                                                       'summary_sha256': sha(output / 'summary.json'),
                                                       'fixed_sampling_bank_sha256': plan['fixed_sampling_bank_sha256'],
                                                       'patch_metrics_sha256': sha(output / 'patch_metrics.jsonl'),
                                                       'parameter_updates': False, 'outer8_used': False})
        print(json.dumps({'output': str(output), 'n_patches': len(bank), 'parameter_updates': False}))


if __name__ == '__main__':
    main()
