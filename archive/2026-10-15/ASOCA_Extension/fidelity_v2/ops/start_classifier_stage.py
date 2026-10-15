"""One-machine setup, data validation and bounded classifier diagnosis queue."""
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

P=argparse.ArgumentParser();P.add_argument('--root',default='/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014');A=P.parse_args()
R=Path(A.root);S=Path('/root/autodl-fs/datasets/ASOCA_GTA_20261014')
PY=Path('/root/miniconda3/envs/gta-asoca311/bin/python')
STATUS=R/'classifier_setup_status.json'
state={'status':'running','pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'role':'method development; classifier quality remains unverified until actual reports',
       'new_segmentation_training_started':False}
child=None

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def record(phase,**kwargs):
 state.update(phase=phase,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
 tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATUS)
 print(json.dumps({'phase':phase,'status':state['status']}),flush=True)

def idle_gpu():
 while True:
  p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
  if not p.stdout.strip():return
  record('waiting_for_idle_gpu',gpu_occupants=p.stdout.strip());time.sleep(30)

def run(command,logfile,phase):
 global child
 record(phase,command=[str(x) for x in command])
 logfile.parent.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
 with logfile.open('a') as stream:
  child=subprocess.Popen([str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True)
  record(phase,child_pid=child.pid,log=str(logfile))
  while child.poll() is None:time.sleep(10)
  code=child.returncode;child=None
  state.pop('child_pid',None)
  if code:raise RuntimeError('Stage exited with '+str(code)+'; inspect '+str(logfile))

def stop(signum,frame):raise KeyboardInterrupt('Interrupted by signal '+str(signum))
signal.signal(signal.SIGTERM,stop)
R.mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True)
lock=(R/'.classifier_setup.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 record('waiting_for_environment')
 while not (R/'ENV_READY.json').exists():time.sleep(30)
 env=json.loads((R/'ENV_READY.json').read_text())
 if env.get('status')!='ready' or not env.get('cuda_forward_backward_finite'):raise RuntimeError('Environment marker is invalid')
 idle_gpu()
 if not (R/'CHECKPOINT_CUDA_SMOKE.json').exists():
  run([PY,R/'ops/checkpoint_smoke.py','--root',R],R/'logs/checkpoint_cuda_smoke.log','checking_transfer_and_cuda')
 mechanical=json.loads((R/'CHECKPOINT_CUDA_SMOKE.json').read_text())
 if mechanical.get('status')!='passed':raise RuntimeError('CUDA checkpoint smoke was not verified')
 record('waiting_for_packed_cache',cuda_smoke_peak_GiB=mechanical['gpu_peak_GiB'])
 while not (S/'CACHE_ARCHIVE_READY.json').exists():
  p=S/'packed_publish_status.json'
  if p.exists() and json.loads(p.read_text()).get('status')=='failed':raise RuntimeError('Shared cache publication failed')
  time.sleep(30)
 packed=json.loads((S/'CACHE_ARCHIVE_READY.json').read_text());archive=Path(packed['cache_archive'])
 dataset=R/'dataset';dataset.mkdir(exist_ok=True)
 marker=R/'LOCAL_CACHE_READY.json'
 if not marker.exists():
  record('verifying_cache_archive')
  if sha(archive)!=packed['cache_archive_sha256']:raise RuntimeError('Shared cache archive checksum differs')
  import shutil
  if shutil.disk_usage(dataset).free<12*1024**3:raise RuntimeError('Local SSD needs 12 GiB free to materialize the sparse cache')
  run(['tar','--sparse','-xzf',archive,'-C',dataset],R/'logs/materialize_cache.log','materializing_local_cache')
  manifest=dataset/'cache/manifest.json'
  if sha(manifest)!=packed['manifest_sha256']:raise RuntimeError('Materialized manifest differs')
  metadata=json.loads(manifest.read_text())
  if len(metadata['cases'])!=40 or metadata.get('complete') is not True:raise RuntimeError('Cache is not the complete 40-case dataset')
  for case in metadata['cases']:
   for key in ['image','label','gaussian','skeleton','radius','branch_id','depth']:
    if not (dataset/'cache'/case['cache_subdir']/(key+'.npy')).is_file():raise RuntimeError('Cache array is missing')
  marker.write_text(json.dumps({'status':'ready','cache_root':str(dataset/'cache'),'manifest_sha256':packed['manifest_sha256'],
       'cache_archive_sha256':packed['cache_archive_sha256'],'case_count':40,'original_folds_preserved':True},indent=2)+'\n')
 manifest=dataset/'cache/manifest.json'
 if sha(manifest)!=packed['manifest_sha256']:raise RuntimeError('Local frozen manifest changed')
 if not (R/'prior_audit/results/prior_audit.json').exists():
  run([PY,R/'prior_audit/code/audit_priors.py','--manifest',manifest,'--output',R/'prior_audit/results','--fold','0',
       '--samples','384','--sampling','both','--seed','20261013','--epoch','0','--patch-size','96'],R/'logs/prior_audit.log','auditing_prior_labels')
 code=R/'classifier_stage/code/pretrain_classifier.py'
 common=[PY,'-u',code,'--manifest',manifest,'--fold','0','--patch-size','128','--batch-size','2','--workers','4',
         '--init-checkpoint',R/'initialization/airway_classifier_improved_epoch100.pkl','--init-domain','airway',
         '--semantic-head-policy','reinit','--sampling','geometry','--sampling-fov','image-body','--guidance-weight','0.1']
 smoke=R/'results/classifier_smoke128/training'
 idle_gpu()
 if not (smoke/'CLASSIFIER_TRAINING_COMPLETE.json').exists():
  run(common+['--output',smoke,'--epochs','2','--steps','2','--stats-samples','32','--validation-patches','64','--eval-every','1','--smoke'],
      R/'logs/classifier_smoke128.log','classifier_true_data_smoke')
 sm=json.loads((smoke/'CLASSIFIER_TRAINING_COMPLETE.json').read_text())
 if sm.get('smoke') is not True:raise RuntimeError('Smoke marker role differs')
 actual=R/'results/classifier80x64/training'
 idle_gpu()
 if not (actual/'CLASSIFIER_TRAINING_COMPLETE.json').exists():
  run(common+['--output',actual,'--epochs','80','--steps','64','--stats-samples','2048','--validation-patches','512','--eval-every','5'],
      R/'logs/classifier80x64.log','training_and_diagnosing_classifier')
 done=json.loads((actual/'CLASSIFIER_TRAINING_COMPLETE.json').read_text())
 if done.get('smoke') is not False or not done.get('selection_positive_coverage_valid'):raise RuntimeError('Classifier diagnosis lacks formal positive coverage')
 state.update(status='completed',classifier_report=str(actual/'final_selection_report.json'),classifier_checkpoint=str(actual/'best_selection.pt'))
 record('classifier_diagnostics_ready',note='Inspect label audit and positive classification reliability before choosing staged GTA configuration')
except KeyboardInterrupt as e:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
 state['status']='interrupted';record('stopped',error=str(e))
except Exception as e:
 state['status']='failed';record('stopped_for_inspection',error=repr(e));raise
finally:
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
