"""NOT_MODEL_RESULTS: local metadata/source guard fixtures, no server/GPU/model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import run_early_fixed_final_eval as guard


class MockTensor:
    def __init__(self, device='cpu', finite=True):
        self.device = device
        self.finite = finite


class MockTorch:
    class cuda:
        @staticmethod
        def is_initialized():
            return False

    @staticmethod
    def is_tensor(x):
        return isinstance(x, MockTensor)

    @staticmethod
    def isfinite(x):
        class All:
            def all(self):
                return x.finite
        return All()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parent.parent
    assert output.is_relative_to((root/'server_records/tool_fixtures').resolve())
    assert not output.exists(), 'Use a fresh fixture output'
    output.mkdir(parents=True)
    passed = []

    def rejects(name, function):
        try:
            function()
        except (ValueError, FileNotFoundError):
            passed.append(name)
        else:
            raise AssertionError('Negative fixture accepted: ' + name)

    # These placeholder digests are only metadata fixtures, never weight bytes.
    ready = {'schema_version':1, 'status':'fixed_final_payload_ready', 'role':'early fixed-final inference mirror; no new training',
             'source_root':guard.NEW_ROOT, 'case_ids':guard.IDS.copy(),
             'files':{name:{'bytes':1, 'sha256':'1'*64} for name in guard.FILES},
             'config_sha256':guard.CONFIG, 'manifest_sha256':guard.MANIFEST, 'source_sha256':guard.SOURCE,
             'source_freeze_sha256':guard.FREEZE, 'source_guard_full_package_verified':True,
             'checkpoint_epoch':199, 'checkpoint_sha256':'1'*64,
             'evaluation_sources':{name:'2'*64 for name in guard.EVALUATORS}, 'packages':guard.PACKAGES.copy(),
             'cache_CT_inputs':{case:{'bytes':1, 'sha256':'3'*64} for case in guard.IDS},
             'raw_GT_source_sha256':{case:'4'*64 for case in guard.IDS}}
    ready['files']['config.json']['sha256'] = guard.CONFIG
    ready['files']['manifest.json']['sha256'] = guard.MANIFEST
    guard.ready_structure(ready); passed.append('READY_schema_positive_metadata_only')
    mutations = [
        ('READY_wrong_status', lambda r:r.update(status='training_running')),
        ('READY_wrong_origin', lambda r:r.update(source_root=str(guard.OLD_ROOT))),
        ('READY_wrong_config', lambda r:r.update(config_sha256='0'*64)),
        ('READY_wrong_manifest', lambda r:r.update(manifest_sha256='0'*64)),
        ('READY_wrong_source', lambda r:r.update(source_sha256='0'*64)),
        ('READY_not_verified_package', lambda r:r.update(source_guard_full_package_verified=False)),
        ('READY_epoch198', lambda r:r.update(checkpoint_epoch=198)),
        ('READY_reordered_cases', lambda r:r.update(case_ids=list(reversed(guard.IDS)))),
        ('READY_wrong_runtime', lambda r:r['packages'].update(scipy='1.14.1')),
        ('READY_missing_evaluator', lambda r:r['evaluation_sources'].pop('metrics.py')),
        ('READY_missing_CT_case', lambda r:r['cache_CT_inputs'].pop(guard.IDS[0])),
        ('READY_missing_GT_case', lambda r:r['raw_GT_source_sha256'].pop(guard.IDS[0])),
        ('READY_path_traversal_extra_file', lambda r:r['files'].update({'../final.pt':{'bytes':1,'sha256':'1'*64}})),
        ('READY_wrong_final_digest', lambda r:r.update(checkpoint_sha256='0'*64)),
        ('READY_bool_byte_count', lambda r:r['files']['final.pt'].update(bytes=True)),
        ('READY_non_hex_digest', lambda r:r['files']['history.jsonl'].update(sha256='g'*64)),
    ]
    for name, mutate in mutations:
        value = copy.deepcopy(ready); mutate(value)
        rejects(name, lambda value=value:guard.ready_structure(value))
    config_path = root/'server_records/snapshot_training_20261014_rootrecheck/gta_38773/results/gta_enhanced200x125/training/config.json'
    manifest_path = root/'server_records/classifier_scope_20261014/dataset/cache/manifest.json'
    assert guard.sha(config_path) == guard.CONFIG and guard.sha(manifest_path) == guard.MANIFEST
    config, manifest = guard.read(config_path), guard.read(manifest_path)
    guard.config_guard(config, manifest); passed.append('real_config_manifest_readonly_positive')
    mutations = [
        ('config_invalid_selection', lambda c:c.update(internal_selection_valid=False)),
        ('config_wrong_output_lineage', lambda c:c.update(output=str(guard.OLD_ROOT/'results/pretend_training'))),
        ('config_wrong_epochs', lambda c:c.update(epochs=201)),
        ('config_wrong_method', lambda c:c.update(method='unet')),
        ('config_no_DS', lambda c:c.update(no_ds=True)),
        ('config_periodic_selection', lambda c:c.update(checkpoint_selection='periodic')),
        ('config_wrong_trained_ids', lambda c:c['trained_case_ids'].append(guard.IDS[0])),
    ]
    for name, mutate in mutations:
        value = copy.deepcopy(config); mutate(value)
        rejects(name, lambda value=value:guard.config_guard(value, manifest))
    saved = {'epoch':199, 'config':config, 'model':{'NOT_MODEL_RESULTS_tensor':MockTensor()}}
    assert guard.final_header(saved, config, MockTorch)['CUDA_context_initialized'] is False
    passed.append('mock_CPU_header_positive_no_checkpoint_load')
    for name, value in [
        ('header_wrong_epoch', saved | {'epoch':198}),
        ('header_wrong_config', saved | {'config':config | {'epochs':201}}),
        ('header_no_model', saved | {'model':{}}),
        ('header_non_tensor', saved | {'model':{'x':1}}),
        ('header_nonfinite_tensor', saved | {'model':{'x':MockTensor(finite=False)}}),
        ('header_CUDA_tensor', saved | {'model':{'x':MockTensor(device='cuda:0')}}),
        ('header_periodic_role', saved | {'checkpoint_role':'periodic internal-four diagnostic; not final selection'}),
    ]:
        rejects(name, lambda value=value:guard.final_header(value, config, MockTorch))
    freeze = guard.source_guard(root); passed.append('actual_local_frozen28_commands_BC8_positive')
    fixture = output/'source_copy_NOT_MODEL_RESULTS'
    for name in freeze['code_files']:
        destination = fixture/'code'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/'code'/name, destination)
    for name in list(freeze['command_files'])+['configs/main_source_freeze.json']:
        destination = fixture/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, destination)
    guard.source_guard(fixture); passed.append('copied_frozen_source_positive_no_execution')
    source = fixture/'code/evaluate.py'; original = source.read_bytes()
    source.write_bytes(original+b'\n# NOT_MODEL_RESULTS fixture mutation\n')
    rejects('source_mutated_evaluator_outside_BC8_rejected', lambda:guard.source_guard(fixture))
    source.write_bytes(original)
    command = fixture/'configs/run_internal_trajectory.sh'; original = command.read_bytes()
    command.write_bytes(original+b'\n# NOT_MODEL_RESULTS fixture mutation\n')
    rejects('source_mutated_command_rejected', lambda:guard.source_guard(fixture))
    command.write_bytes(original)
    extra = fixture/'code/NOT_MODEL_RESULTS_extra.txt'; extra.write_text('metadata fixture')
    rejects('source_untracked_package_member_rejected', lambda:guard.source_guard(fixture))
    # Keep evidence instead of deleting any directory; restore only this owned file
    # membership via its suffix so the source guard ignores it as bytecode.
    extra.rename(extra.with_suffix('.pyc'))
    outside = output/'outside.txt'; outside.write_text('NOT_MODEL_RESULTS')
    rejects('path_escape_rejected', lambda:guard.safe_member(fixture, '../outside.txt'))
    argv = guard.evaluation_argv(guard.OLD_ROOT, guard.SHARED, guard.OLD_ROOT/'results/gta_fixed_final_preview_20261014')
    assert argv[argv.index('--split')+1] == 'internal_diagnostic' and argv[argv.index('--sw-batch-size')+1] == '1'
    assert not ({'--smoke','--geometry-only','--max-cases'} & set(argv))
    passed.append('exact_argv_internal4_batch1_no_smoke_geometry_only_subset')
    report = {'status':'passed', 'role':'NOT_MODEL_RESULTS', 'n_checks':len(passed), 'checks':passed,
              'controller_source_sha256':guard.sha(Path(guard.__file__)), 'fixture_script_sha256':guard.sha(Path(__file__)),
              'source_root_readonly':str(root), 'server_connections':0, 'inference_or_training_started':False,
              'scope':'READY structure/config isolation/mock CPU header/full source membership/argv guards only',
              'not_executed':['Actual torch checkpoint deserialization/tensor inspection', 'Server payload files/cache/GT checks',
                              'Linux flock/GPU-idle/owned signals and time budgets', 'Actual runtime nine-package verification',
                              'Frozen evaluation/inference/NRRD result verification']}
    (output/'fixture_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
