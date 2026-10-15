"""Run one frozen recipe, with durable ownership and no score-based startup gate."""
import argparse,datetime,fcntl,hashlib,json,math,os,re,signal,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--recipe',choices=['run_main_smoke.sh','run_gta200.sh','run_internal_trajectory.sh'],required=True)
p.add_argument('--tag',required=True)
p.add_argument('--training')
p.add_argument('--resume',action='store_true')
a=p.parse_args();r=Path(a.root).resolve()
if not re.fullmatch('[a-z0-9_]+',a.tag):raise ValueError('Unsafe stage tag')
record=r/'stage_records'/ (a.tag+'.json');record.parent.mkdir(exist_ok=True)
log=r/'logs'/ (a.tag+'.log');log.parent.mkdir(exist_ok=True)
lock=(r/'.segmentation_stage.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if record.exists():
 if not a.resume:raise RuntimeError('Stage has durable state; inspect before an explicit continuation')
 previous=json.loads(record.read_text())
 requested=str(Path(a.training).resolve()) if a.training else None
 if (previous.get('recipe')!=a.recipe or previous.get('tag')!=a.tag
     or previous.get('output_root')!=str(r/'results')
     or (a.recipe=='run_internal_trajectory.sh' and previous.get('requested_training')!=requested)):
  raise RuntimeError('Continuation identity differs from the recorded recipe/output/training')
child=None
state={'status':'starting','pid':os.getpid(),'recipe':a.recipe,'tag':a.tag,
 'role':'single-fold method development; classification accuracy diagnostic only',
 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'output_root':str(r/'results'),'requested_training':str(Path(a.training).resolve()) if a.training else None,
 'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save(**kw):
 state.update(kw,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 temp=record.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(record)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def stop(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
signal.signal(signal.SIGTERM,stop)
try:
 if not (r/'AFTER_RESTART_READY.json').exists():raise RuntimeError('Post-restart data/weights/CUDA check missing')
 ready=json.loads((r/'AFTER_RESTART_READY.json').read_text())
 if ready['boot_id']!=state['boot_id'] or ready['status']!='verified':raise RuntimeError('Readiness belongs to another boot')
 env={k:v for k,v in os.environ.items() if not k.startswith('GTA_') and k not in ['OUTPUT_ROOT','CLS_CHECKPOINT']}
 raw_ready=json.loads(Path('/root/autodl-fs/datasets/ASOCA_GTA_20261014/RAW_DIRECTORY_READY.json').read_text())
 env.update(GTA_ROOT=str(r),GTA_PYTHON=sys.executable,OUTPUT_ROOT=str(r/'results'),GTA_RESUME='1' if a.resume else '0',
            GTA_MANIFEST=str(r/'dataset/cache/manifest.json'),GTA_RAW_ROOT=raw_ready['raw_root'])
 if a.training:
  training=Path(a.training).resolve()
  if not training.is_relative_to((r/'results').resolve()):raise RuntimeError('Trajectory training path escapes this project')
  env['GTA_TRAINING']=str(training)
 command=['bash',str(r/'configs'/a.recipe)]
 save(status='running',command=command,log=str(log))
 with log.open('a') as stream:
  child=subprocess.Popen(command,cwd=r,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  save(child_pid=child.pid)
  while child.poll() is None:time.sleep(5)
 code=child.returncode;child=None
 state.pop('child_pid',None)
 if code:raise RuntimeError('Recipe exit '+str(code)+'; inspect '+str(log))
 if a.recipe in ['run_main_smoke.sh','run_gta200.sh']:
  is_smoke=a.recipe=='run_main_smoke.sh'
  folder=r/'results'/ ('gta_mechanical_smoke128' if is_smoke else 'gta_enhanced200x125')/'training'
  marker=json.loads((folder/'TRAINING_COMPLETE.json').read_text())
  frozen=json.loads((r/'configs/main_source_freeze.json').read_text())
  expected_epochs,expected_steps=(3,2) if is_smoke else (200,125)
  rows=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines() if line.strip()]
  if (marker.get('smoke')!=is_smoke or marker.get('epoch')!=expected_epochs-1
      or marker.get('segmentation_updates')!=expected_epochs*expected_steps
      or marker.get('source_sha256')!=frozen['training_source_sha256']
      or marker.get('final_sha256')!=sha(folder/'final.pt')
      or [x.get('epoch') for x in rows]!=list(range(expected_epochs))
      or any(not math.isfinite(x['loss']) for x in rows)):
   raise RuntimeError('Recipe completion identity, budget or finite history differs')
  if is_smoke:
   if [x.get('stage') for x in rows]!=['frozen','partial','joint']:raise RuntimeError('Three stages were not actually executed')
   norms=[x['component_means']['classifier_gradient_norm'] for x in rows]
   if norms[0]!=0 or any(not math.isfinite(v) or v<=0 for v in norms[1:]):raise RuntimeError('Actual classifier gradients do not match stage policy')
   receipt={'status':'passed','role':'mechanical three-stage test only; not segmentation performance',
    'training_source_sha256':marker['source_sha256'],'stage_classifier_gradient_norms':norms,
    'max_gpu_peak_GiB':max(x['gpu_peak_GiB'] for x in rows),'segmentation_updates':6,
    'boot_id':state['boot_id'],'classifier_initializer':'separate smoke fixed-final; formal uses independent80-final'}
   (r/'GTA_STAGED_CUDA_SMOKE.json').write_text(json.dumps(receipt,indent=2)+'\n')
  save(training=str(folder),final_sha256=marker['final_sha256'],segmentation_updates=marker['segmentation_updates'])
 save(status='completed',return_code=0)
except BaseException as e:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 save(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e))
 raise
finally:
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
