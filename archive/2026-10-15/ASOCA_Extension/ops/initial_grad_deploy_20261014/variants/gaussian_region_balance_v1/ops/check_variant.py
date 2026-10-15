"""Inspect B-only source/actual config or run small math fixtures; never train.

All writes require a fresh --report path. Actual config is read-only: the
unchanged trainer records new source identity, and this audit binds its exact
config bytes to the separately sealed Gaussian reduction protocol.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
VARIANT = Path(__file__).resolve().parents[1]
BASE = VARIANT.parents[1]
BASE_CONFIG = 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
SHADOW = 'a8c1c6b9726aebf1fbeb7fff4a05d46b59bd4d02660d14eaf98028c4844e7137'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inventory(folder):
    return {p.relative_to(folder).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and p.suffix not in ('.pyc', '.pyo')}


def combined(folder, names):
    h = hashlib.sha256()
    for name in sorted(names):
        h.update(name.encode()); h.update((folder/name).read_bytes())
    return h.hexdigest()


def inspect():
    protocol = read(VARIANT/'protocol.json'); ready = read(VARIANT/'VARIANT_READY.json')
    require(sha(VARIANT/'protocol.json') == ready['protocol_sha256'], 'Immutable B protocol changed')
    for name, expected in ready['auxiliary_files_sha256'].items():
        require(sha(VARIANT/name) == expected, 'Variant auxiliary file changed: '+name)
    freeze = read(BASE/'configs/main_source_freeze.json')
    require(sha(BASE/'configs/main_source_freeze.json') == protocol['base_freeze_sha256'], 'Original freeze changed')
    base_files, new_files = inventory(BASE/'code'), inventory(VARIANT/'code')
    require(base_files == freeze['code_files'], 'Original frozen28 code package changed')
    require(len(new_files) == 28 and set(new_files) == set(base_files), 'Variant must retain exact28-file membership')
    changes = sorted(name for name in base_files if base_files[name] != new_files[name])
    require(changes == ['losses.py'], 'B must change only losses.py')
    require(new_files == protocol['variant_code_files'], 'Sealed variant bytes changed')
    dependencies = {name: new_files[name]['sha256'] for name in freeze['training_dependency_files']}
    source = combined(VARIANT/'code', dependencies)
    package = combined(VARIANT/'code', new_files)
    require(source == protocol['variant_training_source_sha256'] and package == protocol['variant_code_package_sha256'], 'Variant source/package identity differs')
    return protocol, {'status': 'source_inspection_passed_not_training', 'source_sha256': source,
                      'code_package_sha256': package, 'source_files_sha256': dependencies,
                      'protocol_sha256': sha(VARIANT/'protocol.json'), 'changed_code_files': changes,
                      'gaussian_reduction': protocol['gaussian_reduction'],
                      'actual_model_forward_executed': False, 'training_started': False}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture():
    require(sha(BASE/'ops/shadow_gaussian_region_balance.py') == SHADOW, 'Reviewed shadow source differs')
    shadow = load(BASE/'ops/shadow_gaussian_region_balance.py', 'reviewed_shadow')
    hand = shadow.fixture(None)
    report = {'role': 'SMALL_MATH_FIXTURE_NOT_ACTUAL_MODEL_RESULTS', 'hand_fixture': hand,
              'actual_candidate_torch_verified': False, 'actual_model_forward_executed': False}
    if importlib.util.find_spec('torch') is None:
        report['status'] = 'hand_verified_torch_unavailable'; return report
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    loss_module = load(VARIANT/'code/losses.py', 'candidate_losses')
    require(loss_module.GAUSSIAN_REDUCTION == 'per_patch_equal_fg_bg_present_mean_v1', 'Candidate reduction marker differs')
    qs = [[.6, .7, .9, .5], [.75]+[.5]*7]
    cases = [('uneven_fg', [qs[0]], [[1, 0, 0, 0]]),
             ('all_empty', [qs[0]], [[0]*4]), ('all_fg', [qs[0]], [[1]*4]),
             ('one_voxel', [qs[1]], [[1]+[0]*7]),
             ('batchavg', [qs[1], [.7]*8], [[1]+[0]*7, [0]*8])]
    actual = []
    for (name, values, masks), expected in zip(cases, hand['cases']):
        require(name == expected['name'], 'Fixture order differs')
        batch, n = len(values), len(values[0])
        q = torch.tensor(values, dtype=torch.float32).reshape(batch, 1, 1, 1, n).requires_grad_(True)
        t = torch.full_like(q, .5); label = torch.tensor(masks, dtype=torch.int64).reshape(batch, 1, 1, n)
        b = loss_module.gaussian_region_balanced_mse(q, t, label)
        shadow_b = shadow.region_losses(q, t, label[:, None] > 0)['B']
        torch.testing.assert_close(b, torch.tensor(float(Fraction(expected['hand_B']))), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(b, shadow_b, rtol=1e-6, atol=1e-7)
        g = torch.autograd.grad(b, q, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(shadow_b, q, retain_graph=True)[0]
        torch.testing.assert_close(g, reference_gradient, rtol=1e-5, atol=1e-7)
        if name in ('all_empty', 'all_fg'):
            a = torch.nn.functional.mse_loss(q, t)
            torch.testing.assert_close(b, a, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(g, torch.autograd.grad(a, q)[0], rtol=1e-5, atol=1e-7)
        if batch == 2:
            separate = sum(loss_module.gaussian_region_balanced_mse(q[i:i+1], t[i:i+1], label[i:i+1]) for i in range(2))/2
            torch.testing.assert_close(b, separate, rtol=1e-6, atol=1e-7)
        require(bool(torch.isfinite(b)) and bool(torch.isfinite(g).all()), 'Nonfinite candidate fixture')
        actual.append({'name': name, 'candidate_loss': float(b.detach()), 'shadow_B_and_hand_expected_verified': True,
                       'candidate_q_gradient_matches_reviewed_shadow': True})
    require(not torch.cuda.is_initialized(), 'Small CPU fixture initialized CUDA')
    report.update(status='passed', actual_candidate_torch_verified=True, actual_cases=actual,
                  actual_shadow_torch_fixture=shadow.fixture(torch), torch_version=str(torch.__version__))
    return report


def actual_config(protocol, inspection, main_path, candidate_path, expected_output):
    require(main_path is not None and candidate_path is not None and expected_output is not None,
            '--main-config/--actual-config/--expected-output required')
    require(sha(main_path) == BASE_CONFIG, 'Actual A reference config bytes differ')
    candidate_sha = sha(candidate_path)
    main, candidate = read(main_path), read(candidate_path)
    require(expected_output.startswith('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014/results/')
            and expected_output.endswith('/training') and '..' not in Path(expected_output).parts
            and expected_output != main['output'], 'Use a distinct NEW-root B training output')
    expected = deepcopy(main)
    expected.update(output=expected_output, source_sha256=inspection['source_sha256'],
                    source_files_sha256=inspection['source_files_sha256'])
    differences = {key: {'expected': expected.get(key), 'actual': candidate.get(key),
                         'missing_from_expected': key not in expected, 'missing_from_actual': key not in candidate}
                   for key in set(expected) | set(candidate)
                   if key not in expected or key not in candidate or expected[key] != candidate[key]}
    require(not differences, 'Actual B config differs beyond output/new source identity: '+json.dumps(differences))
    require(candidate['gaussian_weight'] == 10. and candidate['gaussian_late_weight'] is None
            and candidate['parameters'] == main['parameters'], 'B lambda or architecture capacity changed')
    require(sha(main_path) == BASE_CONFIG and sha(candidate_path) == candidate_sha, 'Config changed during read-only audit')
    return {'status': 'actual_config_bound_to_sealed_B_protocol', 'actual_config_sha256': candidate_sha,
            'actual_config_source_sha256': candidate['source_sha256'], 'B_protocol_sha256': inspection['protocol_sha256'],
            'gaussian_reduction': protocol['gaussian_reduction'],
            'actual_config_has_gaussian_reduction_field': 'gaussian_reduction' in candidate,
            'binding': 'Exact actual config bytes plus source/file SHA map bind the separate immutable loss protocol; no config/checkpoint edits.',
            'allowed_actual_config_differences_from_A': ['output', 'source_sha256', 'source_files_sha256'],
            'same_25000_updates': True, 'training_started_by_auditor': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--inspect', action='store_true'); modes.add_argument('--fixture', action='store_true')
    modes.add_argument('--audit-config', action='store_true')
    parser.add_argument('--main-config', type=Path); parser.add_argument('--actual-config', type=Path)
    parser.add_argument('--expected-output'); parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    protocol, inspection = inspect()
    result = dict(inspection) if args.inspect else fixture() if args.fixture else actual_config(
        protocol, inspection, args.main_config, args.actual_config, args.expected_output)
    result['source_inspection'] = inspection
    require(inspect()[1] == inspection, 'Source/protocol changed during inspection/fixture/audit')
    if args.report is not None:
        require(not args.report.exists(), 'Use a fresh report file')
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
