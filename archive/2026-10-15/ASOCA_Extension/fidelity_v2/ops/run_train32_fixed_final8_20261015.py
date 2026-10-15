"""Thin two-server train32 -> fixed-final outer8 runner; no new model source.

OLD PlainUNet can start immediately. NEW B waits only for its existing B8 job
to complete/release. Reuses already checked recipes; no historical re-audit.
"""
import argparse
import contextlib
import datetime
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
LOCAL=Path(__file__).resolve().parents[1]


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def run(args,recipe):
    import fcntl
    root=Path(recipe['root']);training=Path(recipe['training_args']['output']);output=Path(recipe['evaluation_argv'][recipe['evaluation_argv'].index('--output')+1])
    records=training.parent/'controller';status=training.parent/'STATUS.json'
    need(sys.platform=='linux' and LOCAL==root and sys.executable==recipe['python'],'Wrong named host/root/Python')
    own=(root/('.train32_'+args.job+'_20261015.lock')).open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=None;child=None;guard=None;cancel=None;state=None;began=time.monotonic()
    try:
        need(not training.parent.exists(),'Fresh-only; do not resume/adopt an existing job')
        cfg=recipe['training_args'];manifest=read(cfg['manifest']);ids=recipe['evaluation_case_ids'];train_ids=recipe['trained_case_ids']
        need(cfg['cohort']=='train32' and cfg['epochs']==200 and cfg['steps']==125 and cfg['save_every']==50
            and len(set(train_ids))==32 and train_ids==manifest['folds'][cfg['fold']]['train']
            and len(set(ids))==8 and ids==manifest['folds'][cfg['fold']]['val'] and not set(ids)&set(train_ids),'Wrong32/8 split or budget')
        spec=importlib.util.spec_from_file_location('train32_owned_guard',root/'ops/run_early_fixed_final_eval.py')
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);cancel=guard.OwnedCancellation()
        signal.signal(signal.SIGTERM,cancel.request);signal.signal(signal.SIGINT,cancel.request)
        os.umask(0o077);records.mkdir(mode=0o700,parents=True)
        state=dict(status='waiting_idle',controller_pid=os.getpid(),child_pid=None,job=args.job,training32=True,completed=False,gpu_lock_acquired=False)
        def public(**changes):state.update(changes,updated_at_utc=now());atomic(status,state)
        with (records/'controller.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                public();deadline=time.monotonic()+args.max_wait_seconds
                while True:
                    need(time.monotonic()<deadline,'Bounded prerequisite/GPU wait expired')
                    if args.job=='B':
                        previous=read(root/'B_OUTER8_FIXED_FINAL_STATUS_20261015.json')
                        need(previous.get('status')!='attention_stop','Existing B8 needs attention; do not start B32')
                        if not (previous.get('completed') is True and previous.get('gpu_lock_acquired') is False and previous.get('owned_child_exit_confirmed') is True):
                            pid=previous.get('controller_pid');need(isinstance(pid,int) and (Path('/proc')/str(pid)).exists(),'Existing B8 owner disappeared');time.sleep(15);continue
                    candidate=(root/'.segmentation_stage.lock').open('a+')
                    try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:candidate.close();time.sleep(15);continue
                    lease=candidate;first=guard.gpu_snapshot()
                    if first['all_compute_GPUs_idle']:
                        time.sleep(5);second=guard.gpu_snapshot()
                        if second['all_compute_GPUs_idle']:break
                    fcntl.flock(lease,fcntl.LOCK_UN);lease.close();lease=None;time.sleep(15)
                atomic(records/'RECIPE.json',recipe);atomic(records/'PROTOCOL.json',dict(started_at_utc=now(),GPU_idle_observations=[first,second],training_cap_seconds=args.max_training_seconds,evaluation_cap_seconds=args.max_eval_seconds,internal4_evaluation=False))
                env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
                def execute(label,command,cap):
                    nonlocal child
                    started=time.monotonic()
                    with (records/(label+'.log')).open('x',encoding='utf-8') as child_log:
                        cancel.launching=True
                        try:
                            child=subprocess.Popen(command,cwd=recipe['code'],env=env,stdout=child_log,stderr=subprocess.STDOUT,start_new_session=True,umask=0o077)
                            need(os.getpgid(child.pid)==child.pid,'Owned process group differs')
                        finally:cancel.launching=False
                        cancel.raise_pending();public(status=label,child_pid=child.pid,gpu_lock_acquired=True)
                        while child.poll() is None:need(time.monotonic()-started<cap,'Owned '+label+' timed out');time.sleep(5)
                    code=child.returncode;child=None;need(code==0,label+' failed; no retry');public(child_pid=None)
                    return dict(exit_code=code,seconds=time.monotonic()-started)
                trained=execute('training',recipe['training_argv'],args.max_training_seconds)
                marker=read(training/'TRAINING_COMPLETE.json');actual=read(training/'config.json')
                need(marker.get('epoch')==199 and marker.get('trained_case_ids')==train_ids and marker.get('segmentation_updates')==25000
                    and actual.get('cohort')=='train32' and actual.get('method')==cfg['method'] and actual.get('source_sha256')==recipe['source_sha256'],'Train32 fixed-budget completion differs')
                evaluated=execute('outer8_evaluation',recipe['evaluation_argv'],args.max_eval_seconds)
                summary=read(output/'summary.json');need(summary.get('case_ids')==ids and summary.get('overall',{}).get('n_cases')==8
                    and summary.get('method')==cfg['method'] and summary.get('checkpoint_epoch')==199 and summary.get('evaluation_split')=='outer_development'
                    and summary.get('smoke') is False and summary.get('geometry_only') is False,'Eight-case final evaluation incomplete')
                atomic(records/'TRAIN32_FINAL8_COMPLETE.json',dict(status='completed_train32_fixed_final8_development',completed_at_utc=now(),job=args.job,
                    training=trained,evaluation=evaluated,total_seconds=time.monotonic()-began,trained_case_count=32,evaluated_case_count=8,
                    independent_test=False,internal4_evaluation=False,automatic_next_action=False))
                public(status='completed',child_pid=None,completed=True)
            except BaseException as error:
                cancel.cleaning=True;guard.cleanup_owned_child(child);child=None
                atomic(records/'ATTENTION_PRIVATE.json',dict(error=repr(error),traceback=traceback.format_exc(),at_utc=now()));public(status='attention_stop',child_pid=None,completed=False);raise SystemExit(1)
            finally:cancel.cleaning=True;guard.cleanup_owned_child(child);child=None
    finally:
        if lease is not None:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
        fcntl.flock(own,fcntl.LOCK_UN);own.close()
        if state is not None:state.update(gpu_lock_acquired=False,child_pid=None,owned_child_exit_confirmed=True,locks_released_at_utc=now());atomic(status,state)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',choices=('B','unet'),required=True);mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--recipe',type=Path);p.add_argument('--max-wait-seconds',type=int,default=7200);p.add_argument('--max-training-seconds',type=int,default=21600);p.add_argument('--max-eval-seconds',type=int,default=2700);args=p.parse_args()
    need(1<=args.max_wait_seconds<=7200 and 1<=args.max_training_seconds<=21600 and 1<=args.max_eval_seconds<=2700,'Invalid bounded supervision')
    recipe=read(args.recipe or LOCAL/'ops_inputs'/('train32_'+args.job+'_recipe_20261015.json'));need(recipe['job']==args.job,'Wrong named recipe')
    if args.plan:print(json.dumps(dict(status='PLAN_ONLY_NOT_EXECUTED',recipe=recipe,training_cap_seconds=args.max_training_seconds,evaluation_cap_seconds=args.max_eval_seconds),indent=2));return
    try:run(args,recipe)
    except BaseException:print(json.dumps(dict(status='attention_stop',completed=False,job=args.job)));raise SystemExit(1)


if __name__=='__main__':main()
