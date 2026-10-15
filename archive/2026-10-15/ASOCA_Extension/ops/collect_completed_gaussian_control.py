"""Fresh-only SFTP capture of a genuinely completed lambda0 or sealed-B control.

CPU/stdlib+Paramiko only. --plan never connects. Actual use waits for each
controller's real CONTROL_COMPLETE/STATUS/POSTFIT_VERIFIED contract, then copies
actual final weights once, predictions, curves and provenance; no remote writes.
"""
import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath
import stat
import time

from remote import connect
from collect_gta_completed_trajectory import IDS, NAMES, BASE, CONFIG, SOURCE, FREEZE, sha, file_sha, now, need

LOGICAL = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
OLD = '/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2'
MANIFEST = '141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
B_SOURCE = '231e191e77e5f492b2602df19ce29046bb78dcb582dd8b03f1e982f21493828a'
B_PROTOCOL = 'a2905169196c75625772621b1f66f65e493f8bd90d02d98bf31e24547cef462a'
INITIAL = 'd84df10851c2532e9b70311eba2efe07049f5348e56fe4e76daf7856e6427d79'
LIMIT = 128 * 1024**2
EVALUATORS = {'evaluate.py': 'd913495c36a245611fde854134e6dfe8ec420dd1bc76ca03cc0f8fbf5bc670bb',
    'metrics.py': '388353e557d3da61571517803938386fafdfebd0a535fead72579d8920b583f9',
    'evaluate_trajectory.py': '08850729e46bc8d5e0450b282f88e10d93851a9a055fc7ffb69249a11b21ee63'}
SPECS = {
    'lambda0': dict(port=17552, owner=OLD, records=OLD+'/results/gaussian_lambda0_controller_20261014',
        run='gta_gaussian_lambda0_200x125_20261014', source=SOURCE, weight=0.,
        controller='run_gaussian_lambda0_control.py', controller_sha='31777dcd11f7d08579038a322a39874eb4b4244db3aba18fb78f01cecd4e5040',
        theta='ACTUAL_CONTROL_THETA0.json', expected='expected_lambda0_config.json',
        output='gaussian_lambda0_complete_trajectory_20261014'),
    'B': dict(port=38773, owner=LOGICAL, records=LOGICAL+'/results/gaussian_region_balance_controller_20261014',
        run='gta_region_balance200x125_20261014', source=B_SOURCE, weight=10.,
        controller='run_gaussian_region_balance_control.py', controller_sha='ace0561942714916fe35e4717a1c3df2736aa85b44e63bf978ea9b9fa8da1f1b',
        theta='ACTUAL_B_THETA0.json', expected='expected_B_config.json',
        output='gaussian_region_balance_complete_trajectory_20261014'),
}


def decoded(value): return json.loads(value[0].decode('utf-8-sig'))


def finite_tree(value):
    """None denotes missing diagnostics; every present numeric value must be finite."""
    if isinstance(value, dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value, list): return all(finite_tree(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool): return math.isfinite(value)
    return True


def fetch(sftp, remote, optional=False):
    need(remote.startswith('/root/') and '..' not in PurePosixPath(remote).parts, 'Unsafe fixed evidence path')
    try: before = sftp.lstat(remote)
    except OSError as error:
        if optional and getattr(error, 'errno', None) == 2: return None
        raise
    need(stat.S_ISREG(before.st_mode) and before.st_size <= LIMIT, 'Expected bounded regular evidence: '+remote)
    with sftp.open(remote, 'rb') as stream: data = stream.read(before.st_size+1)
    after = sftp.lstat(remote)
    need(len(data) == before.st_size and (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime), 'Evidence changed during capture: '+remote)
    return data, dict(remote_path=remote, bytes=len(data), sha256=sha(data), source_mtime_epoch=before.st_mtime)


def gate(sftp, kind, main):
    spec = SPECS[kind]; rec = spec['records']; train = LOGICAL+'/results/'+spec['run']+'/training'
    trajectory = str(PurePosixPath(train).parent/'internal4_trajectory')
    curves = str(PurePosixPath(train).parent/'learning_curves')
    cache = {}
    def get(remote, optional=False):
        value = fetch(sftp, remote, optional)
        if value is not None: cache[remote] = value
        return value
    marker = get(rec+'/CONTROL_COMPLETE.json', True); status = get(rec+'/STATUS.json', True)
    if marker is None or status is None: return None, 'actual control completion not yet present'
    complete, state = decoded(marker), decoded(status)
    if state.get('status') != 'completed' or state.get('gpu_lock_acquired') is not False or state.get('owned_child_exit_confirmed') is not True:
        return None, 'controller not completed and fully released'
    need(complete.get('status') == 'completed' and state.get('child_pid') is None
         and state.get('completion_sha256') == sha(marker[0])
         and state.get('controller_sha256') == complete.get('controller_sha256') == spec['controller_sha'], 'Control completion/state/source does not close')
    protocol = decoded(get(rec+'/PROTOCOL.json'))
    # Both pinned producers overwrite only launch status when extending their
    # protocol into CONTROL_COMPLETE. Completion timestamps/children/SHAs are
    # newly added keys; every other launch key must remain present and equal.
    launch_status = {'lambda0': 'formal_control_authorized_recipe',
                     'B': 'actual_ROOT_decision_authorized_formal_B'}[kind]
    need(protocol.get('status') == launch_status and complete.get('status') == 'completed',
         'Control launch-to-completion status transition differs from pinned producer')
    need(all(k in complete and complete[k] == ('completed' if k == 'status' else v)
             for k,v in protocol.items()), 'Completion no longer extends immutable actual launch protocol')
    need(protocol.get('training') == train and protocol.get('records') == rec, 'Control paths differ')
    need(sha(get(spec['owner']+'/ops/'+spec['controller'])[0]) == spec['controller_sha'], 'Actual deployed controller changed')
    verified_data = get(rec+'/POSTFIT_VERIFIED.json'); verified = decoded(verified_data)
    postfit_key = 'postfit_verified_sha256' if kind == 'lambda0' else 'B_postfit_verified_sha256'
    need(complete.get(postfit_key) == sha(verified_data[0]) and verified.get('status') == 'postfit_verified'
         and verified.get('root') == LOGICAL and verified.get('training') == train
         and verified.get('verified_checkpoint_evaluations') == 20 and verified.get('verified_case_evaluations') == 80,
         'Actual independent postfit20/80 receipt differs')
    need((complete.get('verified_case_inferences') == 80) if kind == 'lambda0' else
         (complete.get('verified_checkpoint_evaluations') == 20 and complete.get('verified_case_evaluations') == 80
          and complete.get('A_canonical_unchanged') is True), 'Completed formal control scope differs')
    config_data = get(train+'/config.json'); config = decoded(config_data)
    # Protocol hashes the controller's expected file (with trailing newline).
    # Actual training config has its own bytes; require equal parsed objects,
    # and bind actual bytes through theta0, postfit identity and B's receipt.
    expected_data = get(rec+'/'+spec['expected']); need(decoded(expected_data) == config, 'Actual config differs from launched expected recipe')
    expected = copy.deepcopy(main); expected['output'] = train
    if kind == 'lambda0':
        expected['gaussian_weight'] = 0.; expected['gaussian_schedule']['base'] = 0.
        need(protocol.get('expected_control_config_sha256') == sha(expected_data[0]), 'lambda0 expected config file SHA differs')
        need(sha(get(rec+'/cpu_preflight/CONTROL_PREFLIGHT.json')[0]) == protocol.get('CPU_preflight_sha256'),
             'Actual lambda0 CPU preflight receipt differs')
    else:
        expected['source_sha256'] = B_SOURCE; expected['source_files_sha256'] = config['source_files_sha256']
        amap, bmap = main['source_files_sha256'], config['source_files_sha256']
        need(set(amap) == set(bmap) and [k for k in amap if amap[k] != bmap[k]] == ['losses.py'], 'B changes more than loss source')
        need(protocol.get('expected_B_config_sha256') == sha(expected_data[0]) and protocol.get('B_protocol_sha256') == B_PROTOCOL, 'B launched identity differs')
        binding_data = get(rec+'/ACTUAL_B_CONFIG_BINDING.json'); binding = decoded(binding_data)
        need(complete.get('actual_B_config_binding_sha256') == sha(binding_data[0])
             and binding.get('status') == 'actual_config_bound_to_sealed_B_protocol'
             and binding.get('actual_config_sha256') == sha(config_data[0])
             and binding.get('actual_config_source_sha256') == B_SOURCE and binding.get('B_protocol_sha256') == B_PROTOCOL
             and binding.get('same_25000_updates') is True and binding.get('actual_config_has_gaussian_reduction_field') is False,
             'Sealed-B actual config binding differs')
        variant = LOGICAL+'/variants/gaussian_region_balance_v1'
        need(sha(get(variant+'/protocol.json')[0]) == B_PROTOCOL, 'Actual sealed-B mathematical protocol differs')
        get(variant+'/VARIANT_READY.json')
        need(sha(get(variant+'/code/losses.py')[0]) == bmap['losses.py'], 'Actual B loss bytes differ')
    need(config == expected and config.get('epochs') == 200 and config.get('steps') == 125
         and config.get('patch_size') == 128 and config.get('batch_size') == 2 and config.get('no_ds') is False
         and config.get('smoke') is False and config.get('gaussian_weight') == spec['weight']
         and config.get('gaussian_late_weight') is None and config['gaussian_schedule']['base'] == spec['weight'],
         'Unplanned recipe/schedule change')
    training_data = get(train+'/TRAINING_COMPLETE.json'); training = decoded(training_data)
    training_key = 'formal_training_complete_sha256' if kind == 'lambda0' else 'training_complete_sha256'
    need(complete.get(training_key) == verified.get('training_complete_sha256') == sha(training_data[0]), 'Training marker SHA differs')
    for key, value in dict(epoch=199, smoke=False, source_sha256=spec['source'], manifest_sha256=MANIFEST,
        trained_case_ids=main['trained_case_ids'], segmentation_updates=25000, classifier_pretraining_updates=5120,
        checkpoint_selection='fixed final', internal_selection_valid=True).items(): need(training.get(key) == value, 'Formal training completion differs: '+key)
    need(training.get('final_sha256') == complete.get('fixed_final_sha256'), 'Fixed-final weight receipt differs')
    history_data = get(train+'/history.jsonl'); history = [json.loads(x) for x in history_data[0].decode().splitlines() if x.strip()]
    need(sha(history_data[0]) == verified.get('history_sha256') and [x.get('epoch') for x in history] == list(range(200))
         and [x.get('stage') for x in history] == ['frozen']*50+['partial']*20+['joint']*130
         and all(finite_tree(x) for x in history)
         and all(x.get('iterations') == 125 and isinstance(x.get('loss'), (float,int)) and math.isfinite(x['loss'])
                 and x.get('gaussian_lambda') == spec['weight'] for x in history), 'Actual200x125 history/stages/lambda/finite loss incomplete')
    theta_data = get(rec+'/'+spec['theta']); theta = decoded(theta_data)
    theta_key = 'actual_control_theta0_receipt_sha256' if kind == 'lambda0' else 'actual_B_theta0_receipt_sha256'
    need(complete.get(theta_key) == sha(theta_data[0]) and theta.get('model_state_and_RNG_unchanged') is True
         and theta.get('state', {}).get('tensor_count') == 327 and theta['state'].get('state_sha256') == INITIAL
         and theta.get('source_sha256') == spec['source']
         and theta.get('controller_sha256') == spec['controller_sha'] and theta.get('optimizer_updates') == 0
         and theta.get('network_forwards_before_capture') == 0, 'Actual readonly theta0 capture differs')
    if kind == 'lambda0':
        need(theta.get('actual_original_main_epoch0_state_verified') is False
             and theta.get('config_sha256') == sha(config_data[0]), 'Historical theta0/config binding differs')
    else:
        need(theta.get('historical_main_theta0_verified') is False
             and theta.get('actual_config_binding') == binding, 'B historical state/config binding differs')
    identity = verified['trajectory_identity']; tdata = get(trajectory+'/TRAJECTORY_COMPLETE.json'); tcomplete = decoded(tdata)
    need(tcomplete.get('n_evaluations') == 20 and complete.get('trajectory_complete_sha256') == verified.get('trajectory_complete_sha256') == sha(tdata[0])
         and all(tcomplete.get(k) == v for k,v in identity.items()) and identity.get('training_config_sha256') == sha(config_data[0])
         and identity.get('source_sha256') == spec['source'] and identity.get('manifest_sha256') == MANIFEST
         and identity.get('evaluation_sources') == EVALUATORS and identity.get('case_ids') == list(IDS)
         and identity.get('sw_batch_size') == 1 and identity.get('smoke') is False and identity.get('formal_final_test') is False
         and identity.get('split') == 'internal_diagnostic' and identity.get('outer8_used_for_selection') is False
         and [r['name'] for r in identity['checkpoints']] == list(NAMES)
         and [r['epoch'] for r in identity['checkpoints']] == list(range(9,190,10))+[199]
         and identity['checkpoints'][-1]['checkpoint_sha256'] == training['final_sha256'], 'Actual completed20-point trajectory identity differs')
    cases = verified.get('cases', [])
    need(len(cases) == 80 and {(r['checkpoint'],r['case_id']) for r in cases} == {(n,i) for n in NAMES for i in IDS}, 'Independent80-case evidence incomplete')
    curve_data = get(curves+'/curve_manifest.json')
    need(sha(curve_data[0]) == verified.get('curve_manifest_sha256') and (kind != 'lambda0' or complete.get('curve_manifest_sha256') == sha(curve_data[0])), 'Complete curves manifest differs')
    artifacts = verified.get('curve_artifacts', [])
    need(len(artifacts) == 5 and {PurePosixPath(r['path']).name for r in artifacts} == {'development_learning_curves.png','development_learning_curves.svg','epoch_summary.csv','curve_notes.txt','curve_manifest.json'}
         and all(str(PurePosixPath(r['path']).parent) == curves for r in artifacts), 'Full actual curve package missing/outside control')
    manifest = get(LOGICAL+'/dataset/cache/manifest.json'); need(sha(manifest[0]) == MANIFEST, 'Actual manifest differs')
    freeze = get(LOGICAL+'/configs/main_source_freeze.json'); need(sha(freeze[0]) == FREEZE, 'Actual main source freeze differs')
    for name, digest in EVALUATORS.items(): need(sha(get(LOGICAL+'/code/'+name)[0]) == digest, 'Actual evaluator differs')
    get(rec+'/POSTTRAIN_PREFLIGHT.json')
    return dict(cache=cache, complete=complete, verified=verified, config=config, training=training,
                train=train, trajectory=trajectory, curves=curves, records=rec), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('lambda0','B'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference-main', type=Path, default=BASE/'gta_training_complete_20261014/results/gta_enhanced200x125/training/config.json')
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(); spec = SPECS[args.kind]
    output = (args.output or BASE/spec['output']).resolve()
    need(0 <= args.wait_seconds <= 1800, 'Bounded wait0..1800seconds')
    if args.plan:
        print(json.dumps(dict(spec, role='PLAN_ONLY_NO_SERVER_ACCESS',kind=args.kind, logical_training_root=LOGICAL,
            output=str(output), expected_checkpoint_evaluations=20, expected_case_evaluations=80,
            completion_contract='own CONTROL_COMPLETE + fully released completed STATUS + own POSTFIT_VERIFIED; no POSTFIT_STATUS exists',
            final_weights='download actual bytes once after completion; SHA against actual training/postfit receipt',
            private_initial_snapshot_downloaded=False), indent=2)); return
    need(not output.exists(), 'Fresh output required; no merge/overwrite/resume of records')
    need(file_sha(args.reference_main) == CONFIG, 'Actual downloaded main config reference differs')
    main_config = json.loads(args.reference_main.read_text(encoding='utf-8'))
    client = connect(spec['port']); capture = dict(started_at_utc=now(),kind=args.kind,port=spec['port'],read_only_SFTP=True, files=[])
    began = False
    try:
        with client.open_sftp() as sftp:
            deadline = time.monotonic()+args.wait_seconds
            while True:
                evidence, reason = gate(sftp,args.kind,main_config)
                if evidence is not None: break
                if time.monotonic() >= deadline:
                    print(json.dumps(dict(status='waiting_actual_control_completion',reason=reason,local_target_created=False,partial_files_downloaded=0))); return
                time.sleep(min(15,max(.1,deadline-time.monotonic())))
            cache=evidence['cache']; train=evidence['train']; trajectory=evidence['trajectory']; curves=evidence['curves']; rec=evidence['records']; verified=evidence['verified']
            expected={train+'/config.json':sha(cache[train+'/config.json'][0]), train+'/final.pt':evidence['training']['final_sha256'],
                      train+'/history.jsonl':verified['history_sha256'],train+'/TRAINING_COMPLETE.json':verified['training_complete_sha256'],
                      trajectory+'/validation_history.jsonl':verified['validation_history_sha256']}
            paths=set(cache)
            paths.update(train+'/'+n for n in ('initialization_audit.json','final.pt'))
            paths.update(trajectory+'/'+n for n in ('trajectory_protocol.json','trajectory_status.json','validation_history.jsonl','validation_curve.csv'))
            for item in verified['cases']:
                base=trajectory+'/'+item['checkpoint']+'/'+item['case_id']
                expected[base+'.json']=item['case_json_sha256'];expected[base+'_prediction.nrrd']=item['prediction_sha256']
            filenames=['summary.json','case_metrics.jsonl','EVALUATION_COMPLETE.json']+[i+s for i in IDS for s in ('.json','_prediction.nrrd')]
            paths.update(trajectory+'/'+n+'/'+f for n in NAMES for f in filenames)
            for item in verified['curve_artifacts']: paths.add(item['path']);expected[item['path']]=item['sha256']
            curve_manifest=decoded(cache[curves+'/curve_manifest.json'])
            need(curve_manifest.get('formal_final_test') is False and curve_manifest.get('interpolated_validation') is False,
                 'Curve manifest changed development/no-interpolation role')
            curve_runs=curve_manifest.get('runs',[])
            need(len(curve_runs)==1 and curve_runs[0].get('n_recorded_epochs')==200
                 and curve_runs[0].get('n_training_cases')==28 and curve_runs[0].get('selection_case_ids')==list(IDS),
                 'Actual curve training/case coverage differs')
            for item in curve_runs[0].get('sources',[]):
                remote=item['path'];pure=PurePosixPath(remote)
                allowed=remote in {train+'/'+n for n in ('config.json','history.jsonl','gradient_probes.jsonl')} or remote==trajectory+'/validation_history.jsonl' or (
                    pure.name=='summary.json' and str(pure.parent.parent)==trajectory and pure.parent.name in NAMES)
                need(allowed,'Curve source outside exact control metric/history inventory')
                paths.add(remote);expected[remote]=item['sha256']
            probe=fetch(sftp,train+'/gradient_probes.jsonl',True)
            if probe: cache[train+'/gradient_probes.jsonl']=probe;paths.add(train+'/gradient_probes.jsonl')
            output.mkdir(parents=True,exist_ok=False);began=True
            for index,remote in enumerate(sorted(paths),1):
                if remote.startswith(rec+'/'): relative='controller/'+str(PurePosixPath(remote).relative_to(rec))
                elif remote==spec['owner']+'/ops/'+spec['controller']: relative='controller/source/'+spec['controller']
                else: relative=str(PurePosixPath(remote).relative_to(LOGICAL))
                need(not PurePosixPath(relative).is_absolute() and '..' not in PurePosixPath(relative).parts, 'Unsafe capture mapping')
                data,record=cache[remote] if remote in cache else fetch(sftp,remote)
                if remote in expected: need(record['sha256']==expected[remote],'Actual receipt SHA differs: '+remote)
                target=output.joinpath(*PurePosixPath(relative).parts);target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(data)
                need(file_sha(target)==record['sha256'],'Downloaded evidence bytes differ')
                capture['files'].append(dict(relative_path=relative,**record))
                if index%50==0:print(json.dumps(dict(phase='capture_completed_control',files=index,planned=len(paths))),flush=True)
            # Recheck released controller and all immutable completion inputs.
            end,_=gate(sftp,args.kind,main_config)
            need(end is not None and {p:sha(v[0]) for p,v in end['cache'].items()}==
                 {p:sha(v[0]) for p,v in evidence['cache'].items() if p!=train+'/gradient_probes.jsonl'}, 'Completed producer evidence changed during capture')
            final_record=next(r for r in capture['files'] if r['remote_path']==train+'/final.pt')
            final_stat=sftp.lstat(train+'/final.pt')
            need((final_stat.st_size,final_stat.st_mtime)==(final_record['bytes'],final_record['source_mtime_epoch']),
                 'Actual final weight size/mtime changed after its one byte download')
            capture.update(status='captured_completed_control_pending_independent_CPU_acceptance',ended_at_utc=now(),
                actual_final_weight_bytes_downloaded=True,actual_final_weight_sha256=evidence['training']['final_sha256'],
                private_theta0_tensor_downloaded=False,actual_server_periodic_tensor_audit_receipt_retained=True,
                local_intermediate_checkpoint_tensor_bytes_checked=False,controller_sha256=spec['controller_sha'],
                training_source_sha256=spec['source'],model_inference_training_or_server_mutation=False,collector_sha256=file_sha(Path(__file__)))
            print(json.dumps(dict(status=capture['status'],kind=args.kind,files=len(paths),output=str(output))))
    except BaseException as error:
        if began:capture.update(status='failed_or_partial_capture_not_completed',ended_at_utc=now(),error=repr(error))
        raise
    finally:
        client.close()
        if began:
            with (output/'collection_manifest.json').open('x',encoding='utf-8') as stream:json.dump(capture,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
