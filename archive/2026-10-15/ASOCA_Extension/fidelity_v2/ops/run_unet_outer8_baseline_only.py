"""Independent, quarantined OLD-U fixed-final outer8 acquisition; --plan is CPU-only.

No training, GTA outer8, tuning, unsealing, resuming or self-restart. Actual
execution requires a later explicit --run on the named OLD U environment.
"""
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
LOCAL=Path(__file__).resolve().parents[1]
ROOT=Path('/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2')
LOGICAL=Path('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014')  # same OLD host
PLAN_SHA='27ae698cf9faa357b84408058455ea6c15bcc2dca11c185a649f643f9b4b93e3'
PLAN_REL='docs/unet_outer8_fixed_final_baseline_only_plan_20261014.json'
DEPENDENCY=ROOT/'results/gaussian_lambda0_controller_20261014'
LAMBDA_TRAIN=LOGICAL/'results/gta_gaussian_lambda0_200x125_20261014/training'
QUARANTINE=ROOT/'results/unet_outer8_baseline_only_quarantine_20261014'
STATUS=ROOT/'UNET_OUTER8_BASELINE_ONLY_STATUS.json'
PINS={'run_early_fixed_final_eval.py':'2d7a4f24e4f2bae6fa58f5a023d726e1fb3f91bb0d2b7416a7dfe5bba4147e42',
      'compare_fixed_finals.py':'857c6f90b6e5283547c7d88390792897176fa37a06589f8cb16e412b42abe2c8',
      'verify_postfit.py':'698b349799cff927c9c8be2bf6330e0b2cbf4c1b9198ed8f4a303c6075b78e51',
      'run_gaussian_lambda0_control.py':'31777dcd11f7d08579038a322a39874eb4b4244db3aba18fb78f01cecd4e5040'}
LAMBDA_SHA=PINS['run_gaussian_lambda0_control.py']


def require(ok,message):
    if not ok:raise RuntimeError(message)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def record(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def lines(path):
    data=Path(path).read_bytes();require(data.endswith(b'\n'),'Incomplete JSONL')
    return [json.loads(x) for x in data.decode().splitlines() if x.strip()]
def atomic(path,value):
    temporary=Path(path).with_suffix(Path(path).suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.chmod(temporary,0o600);temporary.replace(path)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def finite_tree(value):
    import math
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,list):return all(finite_tree(v) for v in value)
    return math.isfinite(value) if isinstance(value,(int,float)) and not isinstance(value,bool) else True


def plan():
    path=LOCAL/PLAN_REL;require(sha(path)==PLAN_SHA,'Reviewed prospective plan changed')
    result=read(path)
    return dict(role='PREPARED_BASELINE_ONLY_CONTROLLER_NOT_EXECUTED',reviewed_plan_sha256=PLAN_SHA,
        planned_argv=result['planned_argv'],case_count=8,execution_authorized_by_plan=False,
        future_execution_requires='explicit --run after ROOT review/deployment',wait_cap_seconds=28800,
        evaluator_cap_seconds=2700,static_plan_evaluator_cap_seconds=7200,
        deadline_revision='Latest ROOT instruction narrows7200 to2700; safety bound, not runtime estimate',
        quarantine_metadata=QUARANTINE.as_posix(),quarantine_evaluation=result['planned_output'],
        public_status=STATUS.as_posix(),public_status_excludes='metrics, predictions, detailed errors, outcome hashes',
        dual_OLD_host_stage_locks=[(ROOT/'.segmentation_stage.lock').as_posix(),(LOGICAL/'.segmentation_stage.lock').as_posix()],
        source_or_training_or_server_operations=False)


def dependencies():
    for name,expected in PINS.items():require(sha(ROOT/'ops'/name)==expected,'Reviewed helper changed')
    guard=load(ROOT/'ops/run_early_fixed_final_eval.py','outer8_owned_guard')
    compare=load(ROOT/'ops/compare_fixed_finals.py','outer8_pixel_guard')
    verifier=load(ROOT/'ops/verify_postfit.py','outer8_summary_guard')
    return guard,compare,verifier


def dependency_ready(guard):
    """A true failure/disappearance stops; incomplete live work only waits."""
    state=read(DEPENDENCY/'STATUS.json')
    require(state.get('controller_sha256')==LAMBDA_SHA,'Dependency controller identity changed')
    if state.get('status')!='completed' or state.get('gpu_lock_acquired') is not False or state.get('owned_child_exit_confirmed') is not True:
        require(state.get('status') not in ('failed_stop','stopping_owned_child'),'Dependency failed/stopping')
        pid=state.get('controller_pid');require(isinstance(pid,int) and pid>1,'Dependency PID missing')
        proc=Path('/proc')/str(pid)
        argv=(proc/'cmdline').read_bytes().split(b'\0')
        require(str(ROOT/'ops/run_gaussian_lambda0_control.py').encode() in argv and b'--worker' not in argv,
            'Dependency owner process disappeared or changed')
        require(state.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'Dependency boot identity differs')
        return None
    complete=read(DEPENDENCY/'CONTROL_COMPLETE.json');protocol=read(DEPENDENCY/'PROTOCOL.json')
    verified=read(DEPENDENCY/'POSTFIT_VERIFIED.json')
    require(complete.get('status')=='completed' and state.get('child_pid') is None
        and state.get('completion_sha256')==sha(DEPENDENCY/'CONTROL_COMPLETE.json')
        and protocol.get('status')=='formal_control_authorized_recipe'
        and all(complete.get(k)==v for k,v in protocol.items() if k!='status') and complete.get('controller_sha256')==LAMBDA_SHA
        and complete.get('verified_case_inferences')==80 and complete.get('postfit_verified_sha256')==sha(DEPENDENCY/'POSTFIT_VERIFIED.json'),
        'Dependency actual completion does not close')
    require(verified.get('status')=='postfit_verified' and verified.get('verified_checkpoint_evaluations')==20
        and verified.get('verified_case_evaluations')==80 and verified.get('training')==str(LAMBDA_TRAIN),'Dependency postfit20/80 missing')
    trajectory=LAMBDA_TRAIN.parent/'internal4_trajectory';identity=verified['trajectory_identity']
    done=read(trajectory/'TRAJECTORY_COMPLETE.json');train=read(LAMBDA_TRAIN/'TRAINING_COMPLETE.json')
    require(done.get('n_evaluations')==20 and all(done.get(k)==v for k,v in identity.items())
        and identity.get('source_sha256')==guard.SOURCE and identity.get('manifest_sha256')==guard.MANIFEST
        and identity.get('case_ids')==guard.IDS and identity.get('smoke') is False
        and [x['name'] for x in identity['checkpoints']]==[f'epoch{i:03d}' for i in range(10,200,10)]+['final']
        and train.get('epoch')==199 and train.get('segmentation_updates')==25000 and train.get('smoke') is False,
        'Dependency fixed-budget trajectory scope differs')
    require(sha(LAMBDA_TRAIN/'TRAINING_COMPLETE.json')==complete.get('formal_training_complete_sha256')==verified.get('training_complete_sha256')
        and sha(LAMBDA_TRAIN/'final.pt')==complete.get('fixed_final_sha256')==train.get('final_sha256')==identity['checkpoints'][-1]['checkpoint_sha256']
        and sha(trajectory/'TRAJECTORY_COMPLETE.json')==complete.get('trajectory_complete_sha256')==verified.get('trajectory_complete_sha256')
        and sha(trajectory/'validation_history.jsonl')==verified.get('validation_history_sha256')
        and sha(LAMBDA_TRAIN/'history.jsonl')==verified.get('history_sha256'),'Dependency actual final/training/ledger hashes differ')
    case_receipts=verified.get('cases',[]);names=[x['name'] for x in identity['checkpoints']]
    require(len(case_receipts)==80 and {(x['checkpoint'],x['case_id']) for x in case_receipts}=={(n,i) for n in names for i in guard.IDS},'Dependency80 receipt coverage differs')
    for item in case_receipts:
        base=trajectory/item['checkpoint']/item['case_id']
        require(sha(base.with_suffix('.json'))==item['case_json_sha256'] and sha(base.with_name(base.name+'_prediction.nrrd'))==item['prediction_sha256'],
            'Dependency actual80 bytes changed')
    curves=LAMBDA_TRAIN.parent/'learning_curves'
    require(sha(curves/'curve_manifest.json')==complete.get('curve_manifest_sha256')==verified.get('curve_manifest_sha256'),'Dependency curves missing')
    for item in verified.get('curve_artifacts',[]):
        path=Path(item['path']).resolve();require(path.is_relative_to(curves.resolve()) and sha(path)==item['sha256'],'Dependency curve bytes changed')
    guard.source_guard(LOGICAL)
    return dict(completion_sha256=sha(DEPENDENCY/'CONTROL_COMPLETE.json'),postfit_sha256=sha(DEPENDENCY/'POSTFIT_VERIFIED.json'),
        status_sha256=sha(DEPENDENCY/'STATUS.json'),trajectory_sha256=sha(trajectory/'TRAJECTORY_COMPLETE.json'))


def runtime(config):
    """Original U environment, not NEW/isolated-preview version assumptions."""
    require(Path(sys.executable).absolute()==ROOT.parent/'env/bin/python' and sys.version_info[:2]==(3,11),'Use the original verified U Python3.11 environment')
    import torch,numpy,monai,scipy,skimage,SimpleITK,sklearn,pandas,tqdm
    modules={'torch':torch,'numpy':numpy,'monai':monai,'scipy':scipy,'scikit-image':skimage,'SimpleITK':SimpleITK,'scikit-learn':sklearn,'pandas':pandas,'tqdm':tqdm}
    require(str(torch.__version__)==config['torch'] and torch.version.cuda is not None,'Original U checkpoint/runtime Torch differs or lacks CUDA build')
    for module in modules.values():require(Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'Runtime imported outside original U environment')
    require(callable(monai.inferers.sliding_window_inference) and not torch.cuda.is_initialized(),'CUDA initialized during CPU readiness or missing inferer')
    torch.set_num_threads(2)
    return dict(python=sys.version,executable=sys.executable,packages={n:importlib.metadata.version(n) for n in modules},
        CUDA_context_initialized=False,evaluator_actual_Torch_CPU_threads=8,CUDA_inference=True,not_NEW_runtime_equivalence=True)


def preflight(p,guard,c,v):
    import numpy as np
    import SimpleITK as sitk
    source=guard.source_guard(ROOT);immutable=p['immutable_inputs'];remote=p['remote_inputs']
    training=Path(remote['training']);manifest_path=Path(remote['manifest']);manifest=read(manifest_path);config=read(training/'config.json')
    for path,key in [(training/'config.json','training_config_sha256'),(training/'TRAINING_COMPLETE.json','training_complete_sha256'),
        (training/'final.pt','fixed_final_sha256'),(ROOT/'UNET_CALIBRATION_COMPLETE.json','U_global_complete_sha256'),
        (manifest_path,'manifest_sha256'),(ROOT/'configs/main_source_freeze.json','main_source_freeze_sha256'),
        (ROOT/'code/evaluate.py','evaluate_py_sha256'),(ROOT/'code/metrics.py','metrics_py_sha256')]:require(sha(path)==immutable[key],'Immutable U input identity differs')
    ids=p['evaluation']['case_ids'];internal=config['internal_split'];outer=manifest['folds'][0]
    require(outer['val']==ids and len(ids)==len(set(ids))==8 and len(config['trained_case_ids'])==28
        and config['trained_case_ids']==internal['train'] and internal['val']==guard.IDS
        and not set(ids)&set(internal['train']+internal['val']) and set(internal['train']+internal['val'])==set(outer['train']),
        'Outer8/trained28/internal4 patient isolation differs')
    require(config.get('method')=='unet' and config.get('epochs')==200 and config.get('steps')==125 and config.get('smoke') is False
        and config.get('source_sha256')==guard.SOURCE and config.get('source_files_sha256')==source['training_dependency_files']
        and config.get('manifest_sha256')==guard.MANIFEST and config.get('no_ds') is False
        and config.get('checkpoint_selection')=='fixed final' and config.get('patch_size')==128,'Not immutable formal U-Net recipe')
    guard.unet_guard(ROOT)
    history=lines(training/'history.jsonl');require([x['epoch'] for x in history]==list(range(200))
        and all(x.get('iterations')==125 and finite_tree(x) for x in history),'U200x125 finite history differs')
    trajectory=training.parent/'internal4_trajectory';protocol=read(trajectory/'trajectory_protocol.json');ledger=lines(trajectory/'validation_history.jsonl')
    evaluator=load(ROOT/'code/evaluate_trajectory.py','outer8_existing_internal_verifier')
    cases={x['case_id']:x for x in manifest['cases']};eval_sources={n:sha(ROOT/'code'/n) for n in guard.EVALUATORS}
    require(protocol['case_ids']==guard.IDS and len(ledger)==len(protocol['checkpoints'])==20,'Accepted U20/80 evidence missing')
    evidence_paths=[training/n for n in ('config.json','TRAINING_COMPLETE.json','history.jsonl','initialization_audit.json','final.pt')]
    evidence_paths += [ROOT/'UNET_CALIBRATION_COMPLETE.json',manifest_path,ROOT/'configs/main_source_freeze.json']
    evidence_paths += [trajectory/n for n in ('trajectory_protocol.json','TRAJECTORY_COMPLETE.json','trajectory_status.json','validation_history.jsonl','validation_curve.csv')]
    for item,point in zip(ledger,protocol['checkpoints']):
        directory=trajectory/point['name']
        summary=evaluator.verify_evaluation(directory,guard.IDS,point['checkpoint_sha256'],guard.MANIFEST,config,eval_sources,cases,point['epoch'])
        require(v.same(item['overall'],summary['overall']) and item['name']==point['name'],'U internal trajectory ledger changed')
        evidence_paths += list(directory.iterdir())
    records={str(path):record(path) for path in evidence_paths}
    inputs={}
    for case_id in ids:
        case=cases[case_id];ct=(manifest_path.parent/case['cache_subdir']/'image.npy').resolve();truth=(Path(remote['raw_root'])/case['label_relative_path']).resolve()
        require(ct.is_relative_to(manifest_path.parent.resolve()) and truth.is_relative_to(Path(remote['raw_root']).resolve()),'Outer input escaped audited roots')
        require(sha(truth)==case['source_sha256']['label'],'Original outer GT SHA differs')
        array=np.load(ct,mmap_mode='r',allow_pickle=False)
        require(array.dtype==np.dtype('float16') and list(array.shape)==case['shape_zyx'] and np.isfinite(array).all(),'Outer CT cache dtype/shape/finite differs')
        reference=c.read_image(truth);c.check_geometry(c.geometry(reference),case['original_geometry'],case_id);del reference,array
        inputs[case_id]=dict(CT=record(ct),CT_path=str(ct),GT=record(truth),GT_path=str(truth),
            cached_CT_SHA_basis='first observed here and rechecked after; manifest does not provide an expected cache SHA')
    import torch
    saved=torch.load(training/'final.pt',map_location='cpu',weights_only=True)
    header=guard.final_header(saved,config,torch);del saved
    require(header['CUDA_context_initialized'] is False,'CPU checkpoint audit created CUDA context')
    return dict(config=config,cases=cases,records=records,outer_inputs=inputs,checkpoint_header=header,evaluation_sources=eval_sources)


def postflight(p,before,c,v):
    import SimpleITK as sitk
    output=Path(p['planned_output']);ids=p['evaluation']['case_ids'];config=before['config'];rows=[];pixels=[]
    expected={i+s for i in ids for s in ('.json','_prediction.nrrd')}|{'summary.json','case_metrics.jsonl'}
    require({x.name for x in output.iterdir()}==expected,'Standalone outer evaluator must write exactly18 original files')
    metrics=load(ROOT/'code/metrics.py','outer8_frozen_metric_protocol')
    for case_id in ids:
        row=read(output/(case_id+'.json'));case=before['cases'][case_id]
        identity=dict(case_id=case_id,group=case['group'],fold=0,method='unet',checkpoint_sha256=p['immutable_inputs']['fixed_final_sha256'],
            manifest_sha256=p['immutable_inputs']['manifest_sha256'],label_sha256=case['source_sha256']['label'],checkpoint_epoch=199,
            smoke=False,metric_grid='original CT',checkpoint_selection='fixed final',evaluation_split='outer_development',
            evaluation_role=p['evaluation']['role'],formal_final_test=False,geometry_only=False,internal_selection_valid=True,
            training_source_verified_against_live_code=True,training_source_sha256=p['immutable_inputs']['training_source_sha256'],
            evaluation_source_sha256=p['immutable_inputs']['evaluate_py_sha256'],metrics_source_sha256=p['immutable_inputs']['metrics_py_sha256'],
            metric_protocol_version=2,surface_connectivity_neighbors=6,surface_area_weighted=False,inference_patch_size=[128]*3,inference_sw_batch_size=1)
        require(all(row.get(k)==value for k,value in identity.items()) and finite_tree(row),'Outer case/source/role/finite metadata differs')
        require(row.get('assd_definition')==metrics.ASSD_POOLED_DEFINITION and row.get('assd_directional_mean_definition')==metrics.ASSD_DIRECTIONAL_DEFINITION,'Outer metric convention differs')
        require(all(k in row for k in v.SUMMARY_FIELDS) and isinstance(row.get('empty_prediction'),bool),'Outer metric fields/empty flag missing')
        prediction=output/(case_id+'_prediction.nrrd');truth=Path(before['outer_inputs'][case_id]['GT_path'])
        require(sha(prediction)==row.get('prediction_sha256'),'Outer prediction bytes differ')
        reference=c.read_image(truth);predicted=c.read_image(prediction)
        c.check_geometry(c.geometry(reference),case['original_geometry'],case_id);c.check_geometry(c.geometry(predicted),c.geometry(reference),case_id)
        primary=c.primary_from_pixels(sitk.GetArrayFromImage(predicted),sitk.GetArrayFromImage(reference),reference.GetSpacing()[::-1])
        require(v.same(primary['dice'],row['dice']) and v.same(primary['assd_mm'],row['assd_mm']) and primary['empty_prediction']==row['empty_prediction'],
            'Outer recorded Dice/ASSD/empty flag differs from actual pixels')
        require(all((row[field] is None)==(primary['assd_mm'] is None) for field in v.DISTANCES), 'Outer distance-null policy differs')
        require(all(v.finite(row[field]) for field in v.SUMMARY_FIELDS if field not in v.DISTANCES and field!='thin_centerline_recall_radius_le_1mm')
            and row['seconds']>=0,'Outer required non-distance metric missing or invalid')
        pixels.append(dict(case_id=case_id,original_grid=c.geometry(reference),primary_from_pixels=primary,prediction_sha256=row['prediction_sha256']))
        rows.append(row);del reference,predicted
    require(lines(output/'case_metrics.jsonl')==rows,'Outer patient ledger differs')
    summary=read(output/'summary.json')
    for group,selected in [('overall',rows),('Normal',[r for r in rows if r['group']=='Normal']),('Diseased',[r for r in rows if r['group']=='Diseased'])]:
        require(v.same(summary.get(group),v.summary_of(selected)),'Outer complete-eight/subgroup/null policy differs')
    summary_identity=dict(case_ids=ids,metric_protocol=metrics.METRIC_PROTOCOL,primary_metrics=['dice','assd_mm'],fold=0,method='unet',
        evaluation_split='outer_development',evaluation_role=p['evaluation']['role'],checkpoint_epoch=199,checkpoint_selection='fixed final',
        formal_final_test=False,geometry_only=False,smoke=False,internal_selection_valid=True,training_source_verified_against_live_code=True)
    require(all(summary.get(key)==value for key,value in summary_identity.items()),'Outer summary identity differs')
    return dict(case_count=8,pixel_audits=pixels,summary_sha256=sha(output/'summary.json'),
        artifacts={x.name:record(x) for x in output.iterdir()},independent_executor_receipt=True,
        evaluator_EVALUATION_COMPLETE_marker_created=False,quarantine_remains=True)


def run(args):
    import fcntl
    require(sys.platform=='linux' and LOCAL==ROOT and Path(sys.executable).absolute()==ROOT.parent/'env/bin/python','Only named OLD/root original U environment may execute')
    p=read(LOCAL/PLAN_REL);require(sha(LOCAL/PLAN_REL)==PLAN_SHA,'Reviewed static plan changed')
    own=(ROOT/'.unet_outer8_fixed_final_eval.lock').open('a+')
    try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:own.close();raise
    leases=[];child=None;guard=None;cancel=None;started=time.monotonic()
    try:
        require(not QUARANTINE.exists() and not Path(p['planned_output']).exists() and not STATUS.exists(),'Fresh-only acquisition; existing state needs ROOT review')
        os.umask(0o077);QUARANTINE.mkdir(mode=0o700,parents=True)
        state=dict(case_count=8,controller_pid=os.getpid(),evaluator_pid=None,completed=False,attention=False,state='waiting_dependency')
        def public(**changes):state.update(changes,updated_at_utc=now());atomic(STATUS,state)
        with (QUARANTINE/'controller.log').open('x',encoding='utf-8') as controller_log,contextlib.redirect_stdout(controller_log),contextlib.redirect_stderr(controller_log):
            try:
                guard,c,v=dependencies();cancel=guard.OwnedCancellation()
                signal.signal(signal.SIGTERM,cancel.request);signal.signal(signal.SIGINT,cancel.request)
                public();deadline=time.monotonic()+args.max_wait_seconds
                while True:
                    require(time.monotonic()<deadline,'Bounded wait expired')
                    dependency=dependency_ready(guard)
                    if dependency is None:time.sleep(args.poll_seconds);continue
                    public(state='waiting_idle')
                    for lockpath in (ROOT/'.segmentation_stage.lock',LOGICAL/'.segmentation_stage.lock'):
                        candidate=lockpath.open('a+')
                        leases.append(candidate)
                        try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:leases.pop();candidate.close();break
                    if len(leases)==2:
                        first=guard.gpu_snapshot()
                        if first['all_compute_GPUs_idle']:
                            time.sleep(5);second=guard.gpu_snapshot()
                            if second['all_compute_GPUs_idle']:break
                    for lease in reversed(leases):fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
                    leases=[];time.sleep(args.poll_seconds)
                guard.source_guard(ROOT);guard.source_guard(LOGICAL)
                # U config pins its own actual Torch version; never call guard.runtime_profile().
                config=read(Path(p['remote_inputs']['training'])/'config.json');profile=runtime(config)
                before=preflight(p,guard,c,v)
                require(dependency_ready(guard)==dependency and guard.gpu_snapshot()['all_compute_GPUs_idle'],'Dependency/source/resource changed before owned evaluation')
                protocol=plan()|dict(status='actual_baseline_only_authorized',started_at_utc=now(),controller_sha256=sha(__file__),
                    runtime=profile,dependency=dependency,GPU_idle_observations=[first,second],preflight=before,
                    actual_max_wait_seconds=args.max_wait_seconds,actual_evaluator_deadline_seconds=args.max_eval_seconds,
                    quarantine_held_until='ROOT freezes GTA using internal4 and explicitly releases; never automatic',
                    classifier_forward=False,optimizer_updates=0,new_training=False)
                atomic(QUARANTINE/'PROTOCOL.json',protocol)
                require(not Path(p['planned_output']).exists(),'Evaluator destination is not fresh')
                env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
                command=p['planned_argv'];evaluation_started=time.monotonic()
                with (QUARANTINE/'evaluator.log').open('x',encoding='utf-8') as log:
                    cancel.launching=True
                    try:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,umask=0o077)
                        require(os.getpgid(child.pid)==child.pid,'Owned evaluator group differs')
                    finally:cancel.launching=False
                    cancel.raise_pending();public(state='evaluating',evaluator_pid=child.pid)
                    owned=dict(pid=child.pid,PGID=os.getpgid(child.pid),argv=command,started_at_utc=now())
                    atomic(QUARANTINE/'OWNED_EVALUATOR.json',owned)
                    while child.poll() is None:
                        require(time.monotonic()-evaluation_started<args.max_eval_seconds,'Owned evaluation exceeded45-minute cap')
                        time.sleep(5)
                returncode=child.returncode;child=None
                require(returncode==0,'Owned evaluator failed; no retry')
                os.chmod(Path(p['planned_output']),0o700);public(state='postchecking',evaluator_pid=None)
                result=postflight(p,before,c,v)
                guard.source_guard(ROOT);guard.source_guard(LOGICAL)
                require(dependency_ready(guard)==dependency,'Completed lambda0 evidence changed')
                for path,expected in before['records'].items():require(record(path)==expected,'Original accepted U input changed')
                for entry in before['outer_inputs'].values():
                    require(record(entry['CT_path'])==entry['CT'] and record(entry['GT_path'])==entry['GT'],'Original outer CT/GT input changed')
                for name,expected in PINS.items():require(sha(ROOT/'ops'/name)==expected,'Reviewed helper changed')
                require(sha(__file__)==protocol['controller_sha256'] and sha(LOCAL/PLAN_REL)==PLAN_SHA and runtime(config)==profile,'Controller/plan/runtime drifted')
                complete=dict(schema_version=1,status='quarantined_baseline_only_complete',case_count=8,completed_at_utc=now(),
                    protocol_sha256=sha(QUARANTINE/'PROTOCOL.json'),input_source_prepost_unchanged=True,
                    fixed_final_sha256=p['immutable_inputs']['fixed_final_sha256'],evaluation_seconds=time.monotonic()-evaluation_started,
                    total_controller_seconds=time.monotonic()-started,owned_evaluator=owned,postflight=result,
                    GTA_outer8_authorized=False,quarantine_remains=True,automatic_unseal=False)
                atomic(QUARANTINE/'BASELINE_ONLY_COMPLETE.json',complete);public(state='completed',completed=True)
            except BaseException as error:
                if cancel is not None:cancel.cleaning=True
                if guard is not None:guard.cleanup_owned_child(child);child=None
                atomic(QUARANTINE/'ATTENTION_PRIVATE.json',dict(error=repr(error),traceback=traceback.format_exc(),at_utc=now(),quarantine_remains=True))
                public(state='attention_stop',evaluator_pid=None,attention=True,completed=False)
                raise SystemExit(1)
            finally:
                if cancel is not None:cancel.cleaning=True
                if guard is not None:guard.cleanup_owned_child(child);child=None
    finally:
        for lease in reversed(leases):fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
        fcntl.flock(own,fcntl.LOCK_UN);own.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--max-wait-seconds',type=int,default=28800);parser.add_argument('--max-eval-seconds',type=int,default=2700)
    parser.add_argument('--poll-seconds',type=int,default=15);args=parser.parse_args()
    require(1<=args.max_wait_seconds<=28800 and 1<=args.max_eval_seconds<=2700 and 5<=args.poll_seconds<=60,'Invalid bounded budgets')
    if args.plan:print(json.dumps(plan(),indent=2));return
    try:run(args)
    except BaseException:
        # No exception type/message, data path, log tail, metrics or outcome hash.
        print(json.dumps(dict(case_count=8,completed=False,attention=True,state='attention_stop')));raise SystemExit(1)


if __name__=='__main__':main()
