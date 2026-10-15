"""Completed-only, fresh-only SFTP evidence collector; no remote commands or weights.

Default authentication is interactive getpass through remote.connect. Diagnostic
nonfinite-gradient observations are retained, not treated as performance gates.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat

from remote import connect

NEW = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
OLD = '/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2'
BASE = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records'
LIMIT = 128 * 1024**2
FINAL = '2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'
CONFIG = 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
MANIFEST = '141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
SOURCE = 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
B_SOURCE = '231e191e77e5f492b2602df19ce29046bb78dcb582dd8b03f1e982f21493828a'
BANK = '6d58425c5e057739c38c6734f37d3f0c4a79933544f0547705de81bf3853ab14'
INITIAL = 'd84df10851c2532e9b70311eba2efe07049f5348e56fe4e76daf7856e6427d79'
DONOR = '308ebc33d55766458ebd3241c8422dbadb9259b929570bc5f03afd51fbec31a4'
SPECS = {
    'initial': dict(root=NEW, port=38773, result='results/initial_gaussian_parameter_gradients_amp_20261014',
        marker='DIAGNOSTIC_COMPLETE.json', artifacts=('diagnostic_plan.json', 'fixed_sampling_bank.json',
        'reconstructed_initial_state.json', 'batch_metrics.jsonl', 'summary.json'), artifact_key='artifacts_sha256',
        tool='ops/diagnose_initial_gaussian_parameter_gradients.py',
        tool_sha='18683db9538eaf9edf2b04b4f9c9b9dffb9913c5800741d0acd780d2a9d56c3e',
        launch='INITIAL_PARAMETER_GRADIENT_LAUNCH_20261014.json',
        log='INITIAL_PARAMETER_GRADIENT_CONTROLLER_20261014.log',
        output='initial_parameter_gradient_actual_20261014'),
    'classifier': dict(root=OLD, port=17552, result='results/classifier_final_cpu_audit_20261014',
        marker='AUDIT_COMPLETE.json', artifacts=('fixed_sampling_bank.json', 'audit_plan.json',
        'patch_predictions.jsonl', 'summary.json'), artifact_key='artifact_sha256',
        tool='ops/audit_final_classifier_cpu.py',
        tool_sha='b321207067fb81536e0c70b45e83cd48280a9a6e2fce93589b9fa10da6aaaf3d',
        launch='CLASSIFIER_FINAL_CPU_AUDIT_LAUNCH_20261014.json',
        log='CLASSIFIER_FINAL_CPU_AUDIT_20261014.log', output='classifier_final_cpu_actual_20261014'),
}


def need(value, message):
    if not value: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def decoded(data): return json.loads(data.decode('utf-8-sig'))


def remote_bytes(sftp, root, relative, optional=False):
    pure = PurePosixPath(relative)
    need(not pure.is_absolute() and '..' not in pure.parts, 'Unsafe evidence path')
    remote = root + '/' + str(pure)
    try: before = sftp.lstat(remote)
    except OSError as error:
        if optional and getattr(error, 'errno', None) == 2: return None
        raise
    need(stat.S_ISREG(before.st_mode) and before.st_size <= LIMIT, 'Evidence must be bounded regular file: '+relative)
    with sftp.open(remote, 'rb') as stream: data = stream.read(before.st_size + 1)
    after = sftp.lstat(remote)
    need(len(data) == before.st_size and (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime),
         'Evidence changed while reading: '+relative)
    return data, dict(remote_path=remote, bytes=len(data), sha256=sha(data), remote_mtime=before.st_mtime)


def validate(kind, spec, marker, values):
    result = spec['result']+'/'
    summary = decoded(values[result+'summary.json'][0])
    plan_name = 'diagnostic_plan.json' if kind == 'initial' else 'audit_plan.json'
    plan = decoded(values[result+plan_name][0])
    need(marker.get('status') == 'completed' and marker.get('n_patches') == 56
         and marker.get('parameter_updates') is False, 'Wrong completed diagnostic scope')
    need(set(marker.get(spec['artifact_key'], {})) == set(spec['artifacts']), 'Artifact map differs')
    for name, expected in marker[spec['artifact_key']].items():
        need(sha(values[result+name][0]) == expected, 'Artifact SHA mismatch: '+name)
    need(sha(values[result+'fixed_sampling_bank.json'][0]) == BANK, 'Actual standard bank differs')
    need(sha(values[spec['tool']][0]) == spec['tool_sha'], 'Deployed diagnostic tool differs')
    need(plan['tracked_input_sha256'].get(spec['root']+'/'+spec['tool']) == spec['tool_sha'], 'Plan source identity differs')
    launch = decoded(values[spec['launch']][0])
    need(launch.get('output') == spec['root']+'/'+spec['result'], 'Launch output differs')
    if kind == 'initial':
        need(marker.get('n_batches') == 28 and marker.get('held_out_GT_read') is False
             and marker.get('buffer_state_restored_at_end') is True, 'Initial diagnostic completion differs')
        need(summary.get('n_batches') == 28 and summary.get('n_patches') == 56
             and summary.get('initial_state_sha256') == INITIAL and summary.get('precision') == 'amp', 'Initial summary identity differs')
        for key in ('parameter_updates', 'optimizer_created', 'parameter_grad_populated', 'held_out_GT_read', 'historical_main_theta0_verified'):
            need(summary.get(key) is False, 'Unexpected initial diagnostic flag: '+key)
        need(summary.get('diagnostic_copy_restored_to_initial_at_end') is True, 'Model state restoration missing')
        state = decoded(values[result+'reconstructed_initial_state.json'][0])
        need(state.get('state_sha256') == INITIAL and state.get('tensor_count') == 327, 'Reconstructed state identity differs')
        need(plan.get('B_source_sha256') == B_SOURCE and plan.get('initial_state_sha256') == INITIAL,
             'Initial diagnostic source/state differs')
        need(MANIFEST in plan['tracked_input_sha256'].values() and CONFIG in plan['tracked_input_sha256'].values(),
             'Initial diagnostic manifest/config input identities missing')
        canonical = plan['canonical']
        need(canonical.get('main_config_sha256') == CONFIG and canonical.get('main_final_sha256') == FINAL,
             'Canonical main identity differs')
        status = decoded(values[result+'CONTROLLER_STATUS.json'][0])
        protocol = decoded(values[result+'CONTROLLER_PROTOCOL.json'][0])
        fixture = decoded(values[result+'cpu_fixture.json'][0])
        need(status.get('status') == 'completed' and status.get('child_pid') is None
             and status.get('gpu_lock_acquired') is False
             and status.get('script_sha256') == spec['tool_sha']
             and status.get('completion_sha256') == sha(values[result+spec['marker']][0]), 'Controller not fully completed/released')
        need(protocol.get('script_sha256') == spec['tool_sha'] and protocol.get('canonical') == canonical
             and protocol.get('actual_cpu_fixture_sha256') == sha(values[result+'cpu_fixture.json'][0]), 'Controller protocol differs')
        need(fixture.get('status') == 'passed' and fixture.get('actual_candidate_torch_verified') is True
             and 'parameter_sum_and_group_metric_fixture' in fixture, 'Actual CPU fixture not passed')
        need(launch.get('tool_sha256') == spec['tool_sha'], 'Launch tool differs')
        rows = [json.loads(x) for x in values[result+'batch_metrics.jsonl'][0].decode().splitlines() if x.strip()]
        need(len(rows) == 28 and [r['batch'] for r in rows] == list(range(28)), 'Batch ledger incomplete')
        need(marker.get('nonfinite_batches') == summary.get('nonfinite_batches'), 'Nonfinite observations disagree')
    else:
        need(marker.get('n_nonempty') == 49 and marker.get('states_RNG_eval_unchanged') is True
             and marker.get('CUDA_initialized') is False, 'CPU audit completion differs')
        need(summary.get('n_patches') == 56 and summary.get('n_nonempty') == 49
             and summary.get('reference_bank_sha256') == BANK, 'CPU audit summary scope differs')
        need(summary.get('state_sha256_before') == summary.get('state_sha256_after')
             and set(summary.get('state_sha256_before', {})) == {'main327', 'main200', 'classifier80_fixedfinal'}
             and set(summary.get('rng_sha256_before', {})) == {'python', 'numpy', 'torch_cpu'}
             and summary.get('rng_sha256_before') == summary.get('rng_sha256_after'), 'CPU audit mutated model/RNG')
        for key in ('CUDA_initialized', 'optimizer_created', 'heldout_GT_read', 'accuracy_used_as_gate_or_checkpoint_selection'):
            need(summary.get(key) is False, 'Unexpected CPU audit flag: '+key)
        need(summary.get('states_RNG_eval_unchanged') is True and summary.get('parameter_gradients_absent') is True,
             'No-update evidence missing')
        for key, expected in [('checkpoint_sha256', FINAL), ('config_sha256', CONFIG), ('manifest_sha256', MANIFEST),
                              ('source_sha256', SOURCE), ('reference_bank_sha256', BANK), ('classifier_donor_sha256', DONOR)]:
            need(plan.get(key) == expected, 'CPU audit identity differs: '+key)
        need(plan.get('accuracy_is_gate') is False and plan.get('classifier_selection_performed') is False
             and plan.get('CUDA_allowed') is False, 'CPU audit role differs')
        need(launch.get('source_sha256') == spec['tool_sha'], 'Launch source differs')
        rows = [json.loads(x) for x in values[result+'patch_predictions.jsonl'][0].decode().splitlines() if x.strip()]
        need(len(rows) == 56 and sum(r['nonempty'] for r in rows) == 49, 'CPU patch ledger incomplete')


def collect(kind, output):
    spec = SPECS[kind]
    need(not output.exists(), 'Fresh output required; existing records are never overwritten: '+str(output))
    client = connect(spec['port'])
    try:
        with client.open_sftp() as sftp:
            prefix = spec['result']+'/'
            marker_path = prefix+spec['marker']
            first = remote_bytes(sftp, spec['root'], marker_path, optional=True)
            if first is None:
                print(json.dumps(dict(kind=kind, status='waiting', reason='completion marker absent', output_created=False)))
                return
            marker = decoded(first[0])
            need(marker.get('status') == 'completed', 'Existing completion marker is not completed')
            if kind == 'initial':
                status = remote_bytes(sftp, spec['root'], prefix+'CONTROLLER_STATUS.json', optional=True)
                if status is None or decoded(status[0]).get('status') != 'completed' or decoded(status[0]).get('gpu_lock_acquired') is not False:
                    print(json.dumps(dict(kind=kind, status='waiting', reason='controller has not finished/released lease', output_created=False)))
                    return
            paths = [prefix+x for x in spec['artifacts']] + [marker_path, spec['tool'], spec['launch'], spec['log']]
            if kind == 'initial': paths += [prefix+x for x in ('cpu_fixture.json', 'CONTROLLER_STATUS.json', 'CONTROLLER_PROTOCOL.json', 'worker.log')]
            values = {path: remote_bytes(sftp, spec['root'], path) for path in paths}
            need(values[marker_path][0] == first[0], 'Completion marker changed during capture')
            validate(kind, spec, marker, values)
            for path in paths:
                need(remote_bytes(sftp, spec['root'], path)[0] == values[path][0], 'Completed evidence changed before publication: '+path)
        output.mkdir(parents=True, exist_ok=False)
        for path, (data, metadata) in values.items():
            local = output.joinpath(*PurePosixPath(path).parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open('xb') as stream: stream.write(data)
            need(sha(local.read_bytes()) == metadata['sha256'], 'Local evidence checksum differs')
        receipt = dict(schema_version=1, captured_at_utc=now(), status='completed_evidence_collected', kind=kind,
                       role='read-only actual diagnostic evidence; not model performance or accuracy gate',
                       remote_root=spec['root'], port=spec['port'], deployed_tool_sha256=spec['tool_sha'],
                       collector_sha256=sha(Path(__file__).read_bytes()), model_weights_downloaded=False,
                       remote_commands_or_mutations=False, files={p: m for p, (_, m) in values.items()})
        with (output/'collection_manifest.json').open('x', encoding='utf-8') as stream: json.dump(receipt, stream, indent=2)
        print(json.dumps(dict(kind=kind, status=receipt['status'], n_files=len(values), output=str(output))))
    finally: client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('initial', 'classifier', 'both'), required=True)
    parser.add_argument('--output', type=Path, help='Fresh target override; single kind only')
    parser.add_argument('--plan', action='store_true', help='Print fixed paths; no connection or writes')
    args = parser.parse_args()
    need(args.output is None or args.kind != 'both', '--output requires one kind')
    kinds = ('initial', 'classifier') if args.kind == 'both' else (args.kind,)
    for kind in kinds:
        target = args.output or BASE/SPECS[kind]['output']
        if args.plan: print(json.dumps(dict(kind=kind, **SPECS[kind], local_output=str(target), SSH_performed=False), indent=2))
        else: collect(kind, target)


if __name__ == '__main__': main()
