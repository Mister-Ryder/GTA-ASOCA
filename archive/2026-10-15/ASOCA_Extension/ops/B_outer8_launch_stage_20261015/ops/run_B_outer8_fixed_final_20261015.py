"""One B fixed-final outer-eight development evaluation, using accepted setup.

No training, retuning, restart, repeated historical audit or pixel recomputation.
Frozen evaluate.py retains its existing source/manifest and metric safeguards.
"""
import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
ROOT=Path('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014')
PYTHON='/root/miniconda3/envs/gta-asoca311/bin/python'
CODE=ROOT/'variants/gaussian_region_balance_v1/code'
TRAIN=ROOT/'results/gta_region_balance200x125_20261014/training'
OUTPUT=TRAIN.parent/'outer8_fixed_final_development_20261015'
RECORDS=ROOT/'results/B_outer8_fixed_final_controller_20261015'
FREEZE=ROOT/'ops_inputs/B_OUTER8_FREEZE_20261015.json'
STATUS=ROOT/'B_OUTER8_FIXED_FINAL_STATUS_20261015.json'
MANIFEST=ROOT/'dataset/cache/manifest.json'
SOURCE='231e191e77e5f492b2602df19ce29046bb78dcb582dd8b03f1e982f21493828a'
IDS=['Diseased_1','Diseased_15','Diseased_18','Diseased_7','Normal_16','Normal_19','Normal_2','Normal_8']


def need(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)
def argv(raw):return [PYTHON,'-u',(CODE/'evaluate.py').as_posix(),'--checkpoint',(TRAIN/'final.pt').as_posix(),'--manifest',MANIFEST.as_posix(),
    '--raw-root',Path(raw).as_posix(),'--output',OUTPUT.as_posix(),'--fold','0','--split','outer_development','--sw-batch-size','1']
def plan():return dict(status='PLAN_ONLY_NOT_EXECUTED',host_port=38773,freeze=FREEZE.as_posix(),requires_exact_freeze_sha256=True,
    method='gta',source_sha256=SOURCE,case_ids=IDS,output=OUTPUT.as_posix(),records=RECORDS.as_posix(),training=False,
    argv=argv('/root/autodl-fs/datasets/ASOCA_GTA_20261014/raw'),max_evaluator_seconds=2700,
    checks='One upfront identity/split/freshness/idle check; evaluator existing safeguards; exit and eight-case summary.',
    repeats_historical_audit_or_pixel_metrics=False,automatic_next_action=False)


def run(args):
    import fcntl
    need(sys.platform=='linux' and Path(__file__).resolve().parents[1]==ROOT and sys.executable==PYTHON,'Use named NEW root/environment')
    own=(ROOT/'.B_outer8_fixed_final_20261015.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=None;child=None;guard=None;cancel=None;state=None;started=time.monotonic()
    try:
        need(not OUTPUT.exists() and not RECORDS.exists() and not STATUS.exists(),'Fresh-only; existing output/state needs ROOT review')
        need(sha(FREEZE)==args.freeze_sha256,'ROOT freeze SHA differs');freeze=read(FREEZE)
        need(freeze.get('status')=='frozen_B_for_outer8' and freeze.get('decision_author')=='ROOT' and freeze.get('case_ids')==IDS
            and freeze.get('source_sha256')==SOURCE,'ROOT frozen B8 identity differs')
        config=read(TRAIN/'config.json');manifest=read(MANIFEST);complete=read(TRAIN/'TRAINING_COMPLETE.json')
        need(sha(TRAIN/'config.json')==freeze['config_sha256'] and config['method']=='gta' and config['source_sha256']==SOURCE
            and config['manifest_sha256']==freeze['manifest_sha256'] and config['smoke'] is False
            and complete['epoch']==199 and complete['final_sha256']==freeze['checkpoint_sha256'],'Already accepted B fixed-final identity differs')
        split=config['internal_split'];need(manifest['folds'][0]['val']==IDS and len(set(IDS))==8 and config['trained_case_ids']==split['train']
            and len(set(split['train']))==28 and len(set(split['val']))==4 and not set(IDS)&set(split['train']+split['val']),'Outer8/train28/internal4 overlap or identity differs')
        dependency=read(ROOT/'results/gaussian_region_balance_controller_20261014/STATUS.json')
        need(dependency.get('status')=='completed' and dependency.get('gpu_lock_acquired') is False and dependency.get('child_pid') is None
            and dependency.get('owned_child_exit_confirmed') is True,'Accepted B controller has not released its resources')
        raw=read(ROOT/'results/gaussian_region_balance_controller_20261014/POSTFIT_VERIFIED.json')['raw_root']
        # This reviewed helper supplies idle observation and owned-child cleanup only.
        spec=importlib.util.spec_from_file_location('B8_owned_guard',ROOT/'ops/run_early_fixed_final_eval.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        cancel=guard.OwnedCancellation();signal.signal(signal.SIGTERM,cancel.request);signal.signal(signal.SIGINT,cancel.request)
        os.umask(0o077);RECORDS.mkdir(mode=0o700,parents=True)
        state=dict(status='waiting_idle',controller_pid=os.getpid(),controller_sha256=sha(__file__),child_pid=None,
            case_count=8,completed=False,attention=False,gpu_lock_acquired=False)
        def public(**changes):state.update(changes,updated_at_utc=now());atomic(STATUS,state)
        with (RECORDS/'controller.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                public();deadline=time.monotonic()+args.max_wait_seconds
                while True:
                    need(time.monotonic()<deadline,'GPU wait expired');candidate=(ROOT/'.segmentation_stage.lock').open('a+')
                    try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:candidate.close();time.sleep(15);continue
                    lease=candidate;first=guard.gpu_snapshot()
                    if first['all_compute_GPUs_idle']:
                        time.sleep(5);second=guard.gpu_snapshot()
                        if second['all_compute_GPUs_idle']:break
                    fcntl.flock(lease,fcntl.LOCK_UN);lease.close();lease=None;time.sleep(15)
                command=argv(raw);atomic(RECORDS/'PROTOCOL.json',dict(started_at_utc=now(),freeze_sha256=args.freeze_sha256,freeze=freeze,
                    argv=command,GPU_idle_observations=[first,second],max_evaluator_seconds=args.max_eval_seconds,training=False,
                    existing_acceptance_reused=True,repeated_historical_audit_or_pixel_metrics=False))
                env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
                inference_started=time.monotonic()
                with (RECORDS/'evaluator.log').open('x',encoding='utf-8') as child_log:
                    cancel.launching=True
                    try:
                        child=subprocess.Popen(command,cwd=CODE,env=env,stdout=child_log,stderr=subprocess.STDOUT,start_new_session=True,umask=0o077)
                        need(os.getpgid(child.pid)==child.pid,'Owned child group differs')
                    finally:cancel.launching=False
                    cancel.raise_pending();public(status='evaluating',child_pid=child.pid,gpu_lock_acquired=True)
                    atomic(RECORDS/'OWNED_EVALUATOR.json',dict(pid=child.pid,PGID=os.getpgid(child.pid),argv=command))
                    while child.poll() is None:need(time.monotonic()-inference_started<args.max_eval_seconds,'Owned evaluator timed out');time.sleep(5)
                returncode=child.returncode;child=None;need(returncode==0,'Evaluator failed; no retry')
                summary=read(OUTPUT/'summary.json');need(summary.get('case_ids')==IDS and summary.get('overall',{}).get('n_cases')==8
                    and summary.get('method')=='gta' and summary.get('checkpoint_epoch')==199 and summary.get('evaluation_split')=='outer_development'
                    and summary.get('smoke') is False and summary.get('geometry_only') is False,'Eight-case evaluation incomplete')
                atomic(RECORDS/'OUTER8_COMPLETE.json',dict(status='completed_fixed_B_outer8_development',case_count=8,completed_at_utc=now(),
                    freeze_sha256=args.freeze_sha256,checkpoint_sha256=freeze['checkpoint_sha256'],summary_sha256=sha(OUTPUT/'summary.json'),
                    evaluator_exit_code=returncode,evaluation_seconds=time.monotonic()-inference_started,controller_seconds=time.monotonic()-started,
                    training=False,independent_test=False,repeated_historical_audit_or_pixel_metrics=False,automatic_next_action=False))
                public(status='completed',child_pid=None,completed=True)
            except BaseException as error:
                cancel.cleaning=True;guard.cleanup_owned_child(child);child=None
                atomic(RECORDS/'ATTENTION_PRIVATE.json',dict(error=repr(error),traceback=traceback.format_exc(),at_utc=now()))
                public(status='attention_stop',child_pid=None,attention=True,completed=False);raise SystemExit(1)
            finally:
                cancel.cleaning=True;guard.cleanup_owned_child(child);child=None
    finally:
        if lease is not None:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
        fcntl.flock(own,fcntl.LOCK_UN);own.close()
        if state is not None:
            state.update(gpu_lock_acquired=False,child_pid=None,owned_child_exit_confirmed=True,locks_released_at_utc=now());atomic(STATUS,state)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--freeze-sha256',default='');p.add_argument('--max-wait-seconds',type=int,default=1800);p.add_argument('--max-eval-seconds',type=int,default=2700);args=p.parse_args()
    need(1<=args.max_wait_seconds<=1800 and 1<=args.max_eval_seconds<=2700,'Invalid budgets')
    if args.plan:print(json.dumps(plan(),indent=2));return
    try:run(args)
    except BaseException:print(json.dumps(dict(status='attention_stop',completed=False,attention=True,case_count=8)));raise SystemExit(1)


if __name__=='__main__':main()
