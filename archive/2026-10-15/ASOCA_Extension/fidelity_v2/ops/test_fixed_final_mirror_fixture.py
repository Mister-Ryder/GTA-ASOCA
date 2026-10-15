"""Small standard-library parser/copy fixture. NOT_MODEL_RESULTS.

This creates text bytes named final.pt only inside a temporary directory.
It never produces production READY.json, loads a model, or contacts a server.
The fixture's synthetic losses/markers are not experimental measurements.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import publish_fixed_final_mirror as p


def rejected(call, fragment):
    try:
        call()
    except (ValueError, OSError) as error:
        assert fragment in str(error), str(error)
        return str(error)
    raise AssertionError('Invalid fixture accepted: ' + fragment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    evidence = {'role': 'NOT_MODEL_RESULTS', 'training_started': False,
                'GPU_used': False, 'model_loaded': False, 'server_access': False, 'checks': {}}
    checks = evidence['checks']
    with tempfile.TemporaryDirectory(prefix='gta_mirror_parser_fixture_') as temporary:
        root = Path(temporary).resolve()
        frozen = p.read(args.root / 'configs/main_source_freeze.json')
        for name in frozen['code_files']:
            target = root / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.root / 'code' / name, target)
        for name in frozen['command_files']:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.root / name, target)
        shutil.copyfile(args.root / 'configs/main_source_freeze.json', root / 'configs/main_source_freeze.json')
        helper = root / 'ops/verify_postfit.py'
        helper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.root / 'ops/verify_postfit.py', helper)
        training = root / 'results/gta_enhanced200x125/training'
        training.mkdir(parents=True)
        shutil.copyfile(args.config, training / 'config.json')
        manifest = root / 'dataset/cache/manifest.json'
        manifest.parent.mkdir(parents=True)
        shutil.copyfile(args.manifest, manifest)
        config, manifest_info = p.static_guard(root, p.load_source_guard(root))
        checks['real_frozen_package_config_manifest_guard'] = 'passed; no weights or data arrays read'
        plan = subprocess.run([sys.executable, str(Path(p.__file__).resolve()), '--root', str(root), '--plan'],
                              check=True, capture_output=True, text=True)
        planned = json.loads(plan.stdout)
        assert planned['status'] == 'plan_only' and planned['READY_created'] is False and planned['copy_started'] is False
        assert not (root / 'EARLY_FINAL_MIRROR_STATUS.json').exists()
        assert not list(root.rglob('READY.json'))
        checks['plan_creates_no_READY_state_or_copy'] = 'passed'

        final = training / 'final.pt'
        final.write_bytes(b'NOT_MODEL_RESULTS: parser fixture, not a torch checkpoint\n')
        history = [{'epoch': epoch, 'iterations': 125, 'loss': 0.2,
                    'component_means': {'gaussian_weighted': 0.001}} for epoch in range(200)]
        history_path = training / 'history.jsonl'
        def save_history():
            history_path.write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
        save_history()
        marker = {'epoch': 199, 'smoke': False, 'manifest_sha256': p.EXPECTED_MANIFEST,
                  'source_sha256': p.EXPECTED_SOURCE, 'trained_case_ids': config['trained_case_ids'],
                  'final_sha256': p.digest(final), 'segmentation_updates': 25000,
                  'internal_selection_valid': True, 'checkpoint_selection': 'fixed final',
                  'fixture_role': 'NOT_MODEL_RESULTS'}
        marker_path = training / 'TRAINING_COMPLETE.json'
        p.atomic_json(marker_path, marker)
        p.completed_payload(root, config)
        checks['synthetic_complete200x125_parser'] = 'passed; NOT_MODEL_RESULTS text checkpoint'
        removed = history.pop()
        save_history()
        checks['missing_epoch_rejected'] = rejected(lambda: p.completed_payload(root, config), 'complete finite200x125')
        history.append(removed)
        history[90]['component_means']['gaussian_weighted'] = float('inf')
        save_history()
        checks['nonfinite_component_rejected'] = rejected(lambda: p.completed_payload(root, config), 'Nonfinite numeric')
        history[90]['component_means']['gaussian_weighted'] = 0.001
        save_history()
        history_path.write_bytes(history_path.read_bytes().rstrip(b'\n'))
        checks['unfinished_final_line_rejected'] = rejected(lambda: p.completed_payload(root, config), 'unfinished final line')
        save_history()
        marker['final_sha256'] = '0' * 64
        p.atomic_json(marker_path, marker)
        checks['wrong_actual_checkpoint_hash_rejected'] = rejected(lambda: p.completed_payload(root, config), 'completion/final identity')
        marker['final_sha256'] = p.digest(final)
        p.atomic_json(marker_path, marker)

        # Exercise only byte/metadata audit parsing. These are small text files,
        # not NPY volumes or measurements, and metadata.version is mocked here.
        case_info = {case['case_id']: case for case in manifest_info['cases']}
        ct_inputs = {}
        for case_id in p.IDS:
            image = root / 'dataset/cache' / case_info[case_id]['cache_subdir'] / 'image.npy'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(('NOT_MODEL_RESULTS_CT_BYTES_' + case_id).encode('utf-8'))
            ct_inputs[case_id] = p.fingerprint(image)
        runtime = {'python': sys.version, 'manifest_sha256': p.EXPECTED_MANIFEST,
                   'training_source_sha256': p.EXPECTED_SOURCE, 'evaluation_sources': p.EVALUATION_SOURCES,
                   'cache_CT_inputs': ct_inputs, 'packages': {name: 'NOT_MODEL_RESULTS' for name in sorted(p.PACKAGE_NAMES)}}
        runtime_path = root / 'EARLY_FINAL_INPUT_RUNTIME_AUDIT_20261014.json'
        p.atomic_json(runtime_path, runtime)
        original_version = p.importlib.metadata.version
        try:
            p.importlib.metadata.version = lambda name: 'NOT_MODEL_RESULTS'
            p.runtime_audit(root, manifest_info)
            checks['runtime_fixed9_names_and_actual_CT_bytes_parser'] = 'passed; fixture text CT and mocked version metadata'
            runtime['packages']['unrelated_package'] = runtime['packages'].pop('torch')
            p.atomic_json(runtime_path, runtime)
            checks['unrelated9_package_names_rejected'] = rejected(lambda: p.runtime_audit(root, manifest_info), 'fixed9 audited runtime package names')
            runtime['packages']['torch'] = runtime['packages'].pop('unrelated_package')
            p.atomic_json(runtime_path, runtime)
            p.importlib.metadata.version = lambda name: 'WRONG_FIXTURE_VERSION'
            checks['changed_installed_runtime_version_rejected'] = rejected(lambda: p.runtime_audit(root, manifest_info), 'Installed runtime package versions differ')
            p.importlib.metadata.version = lambda name: 'NOT_MODEL_RESULTS'
            ct_file = root / 'dataset/cache' / case_info[p.IDS[0]]['cache_subdir'] / 'image.npy'
            ct_file.write_bytes(b'NOT_MODEL_RESULTS_changed_CT')
            checks['changed_actual_cached_CT_rejected'] = rejected(lambda: p.runtime_audit(root, manifest_info), 'Actual internal4 CT bytes differ')
        finally:
            p.importlib.metadata.version = original_version

        audit = root / 'NOT_MODEL_RESULTS_input_runtime_audit.json'
        p.atomic_json(audit, {'role': 'NOT_MODEL_RESULTS'})
        sources = {'final.pt': final, 'config.json': training / 'config.json', 'history.jsonl': history_path,
                   'TRAINING_COMPLETE.json': marker_path, 'manifest.json': manifest, 'input_runtime_audit.json': audit}
        ready = {'status': 'NOT_MODEL_RESULTS', 'files': {name: p.fingerprint(path) for name, path in sources.items()}}
        destination = root / 'NOT_MODEL_RESULTS_copy'
        marker_name = 'NOT_MODEL_RESULTS_READY.json'
        p.copy_atomic(sources, destination, ready, ready_name=marker_name)
        assert p.read(destination / marker_name) == ready
        assert not (destination / 'READY.json').exists()
        assert all(p.fingerprint(destination / name) == expected for name, expected in ready['files'].items())
        checks['six_file_copy_actual_bytes_match'] = 'passed; NOT_MODEL_RESULTS_READY.json only'
        checks['existing_target_never_overwritten'] = rejected(lambda: p.copy_atomic(sources, destination, ready, ready_name=marker_name),
                                                              'Fresh mirror required')
        original = final.read_bytes()
        final.write_bytes(original + b'changed fixture\n')
        changed_destination = root / 'NOT_MODEL_RESULTS_changed_source'
        checks['source_mutation_rejected_before_copy'] = rejected(lambda: p.copy_atomic(sources, changed_destination, ready, ready_name=marker_name),
                                                                'Source file changed before copying')
        assert not changed_destination.exists()
        final.write_bytes(original)
        race_source, race_destination = root / 'NOT_MODEL_RESULTS_rename_source', root / 'NOT_MODEL_RESULTS_race_target'
        race_source.mkdir(); race_destination.mkdir()
        (race_source / 'fixture.txt').write_text('NOT_MODEL_RESULTS', encoding='utf-8')
        try:
            p.rename_fresh(race_source, race_destination)
        except OSError:
            assert race_source.is_dir() and race_destination.is_dir() and not list(race_destination.iterdir())
            checks['empty_target_race_never_replaced'] = 'passed'
        else:
            raise AssertionError('Existing empty directory replaced')
        assert not list(root.rglob('READY.json'))
    evidence['publisher_sha256'] = p.digest(p.__file__)
    evidence['fixture_sha256'] = p.digest(__file__)
    evidence['production_READY_created'] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    p.atomic_json(args.report, evidence)
    print(json.dumps({'role': 'NOT_MODEL_RESULTS', 'checks': len(checks), 'report': str(args.report)}))


if __name__ == '__main__':
    main()
