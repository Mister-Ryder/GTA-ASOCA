"""Wait/copy one immutable fixed-final payload; no GPU, model or training.

Publication is a fresh sibling temporary directory renamed atomically after
all six files and READY.json pass byte/size/identity checks. Existing targets
are verified for an identical retry, never overwritten. Signals affect self.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import uuid

DEFAULT_ROOT=Path('/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014')
DEFAULT_DESTINATION=Path('/root/autodl-fs/experiments/GTA_ASOCA_Fidelity_20261014/early_final_mirror_20261014')
MAIN_PID=1943
EXPECTED_CONFIG='b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
EXPECTED_MANIFEST='141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
EXPECTED_SOURCE='bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
EXPECTED_FREEZE='9b294024e33b0d0aaa12b473be53dd3a81dfcd9dd020446948fe667405c7ff46'
EXPECTED_HELPER='698b349799cff927c9c8be2bf6330e0b2cbf4c1b9198ed8f4a303c6075b78e51'
EVALUATION_SOURCES={'evaluate.py':'d913495c36a245611fde854134e6dfe8ec420dd1bc76ca03cc0f8fbf5bc670bb',
    'metrics.py':'388353e557d3da61571517803938386fafdfebd0a535fead72579d8920b583f9',
    'evaluate_trajectory.py':'08850729e46bc8d5e0450b282f88e10d93851a9a055fc7ffb69249a11b21ee63'}
IDS=['Diseased_10','Diseased_20','Normal_12','Normal_3']
PAYLOAD_NAMES={'final.pt','config.json','history.jsonl','TRAINING_COMPLETE.json','manifest.json','input_runtime_audit.json'}
PACKAGE_NAMES={'torch','monai','numpy','scipy','scikit-image','SimpleITK','scikit-learn','pandas','tqdm'}


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def fingerprint(path):
    path=Path(path); require(path.is_file(),'Missing file: '+str(path))
    before=path.stat(); value={'bytes':before.st_size,'sha256':digest(path)}; after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'File changed during hashing: '+str(path))
    return value


def load_source_guard(root):
    helper=root/'ops/verify_postfit.py'
    require(digest(helper)==EXPECTED_HELPER,'Pinned source_guard helper changed')
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('early_mirror_pinned_postfit',helper)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(digest(helper)==EXPECTED_HELPER,'Pinned helper changed while loading')
    return module.source_guard


def static_guard(root,source_guard):
    config_path=root/'results/gta_enhanced200x125/training/config.json'
    manifest_path=root/'dataset/cache/manifest.json'
    require(digest(config_path)==EXPECTED_CONFIG,'Main config SHA differs')
    require(digest(manifest_path)==EXPECTED_MANIFEST,'Dataset manifest SHA differs')
    require(digest(root/'configs/main_source_freeze.json')==EXPECTED_FREEZE,'Main full source-freeze identity differs')
    require(digest(root/'ops/verify_postfit.py')==EXPECTED_HELPER,'Pinned helper changed')
    frozen=source_guard(root)
    require(len(frozen['code_files'])==28 and len(frozen['command_files'])==12,'Frozen package/commands membership differs')
    for name,value in EVALUATION_SOURCES.items(): require(digest(root/'code'/name)==value,'Evaluator source changed: '+name)
    config,manifest=read(config_path),read(manifest_path)
    expected={'method':'gta','fold':0,'epochs':200,'steps':125,'batch_size':2,'patch_size':128,'smoke':False,
              'no_ds':False,'cohort':'internal28','checkpoint_selection':'fixed final','source_sha256':EXPECTED_SOURCE,
              'manifest_sha256':EXPECTED_MANIFEST,'source_hash_algorithm':'gta_fidelity_selected_python_path_and_bytes_v1',
              'internal_selection_valid':True,'outer8_used_for_selection':False,'segmentation_updates':25000}
    require(all(config.get(k)==v for k,v in expected.items()),'Formal fixed-final configuration differs')
    require(config.get('source_files_sha256')==frozen['training_dependency_files'],'Per-file training source differs')
    split=config['internal_split']; train=split['train']; val=split['val']; outer=manifest['folds'][0]
    require(val==IDS and len(train)==len(set(train))==28 and not set(train)&set(val)
            and set(train)|set(val)==set(outer['train']) and not set(train+val)&set(outer['val'])
            and config.get('trained_case_ids')==train,'Training28/internal4 isolation differs')
    return config,manifest


def completed_payload(root,config):
    training=root/'results/gta_enhanced200x125/training'
    marker_path=training/'TRAINING_COMPLETE.json'; marker_before=fingerprint(marker_path)
    marker_raw=marker_path.read_bytes(); marker=json.loads(marker_raw.decode('utf-8'))
    require({'bytes':len(marker_raw),'sha256':hashlib.sha256(marker_raw).hexdigest()}==marker_before
            and fingerprint(marker_path)==marker_before,'Training completion changed during parsing')
    actual=fingerprint(training/'final.pt')
    expected={'epoch':199,'smoke':False,'manifest_sha256':EXPECTED_MANIFEST,'source_sha256':EXPECTED_SOURCE,
              'trained_case_ids':config['trained_case_ids'],'final_sha256':actual['sha256'],'segmentation_updates':25000,
              'internal_selection_valid':True,'checkpoint_selection':'fixed final'}
    require(all(marker.get(k)==v for k,v in expected.items()),'Training completion/final identity differs')
    history_path=training/'history.jsonl'; raw=history_path.read_bytes()
    history_verified={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    require(raw.endswith(b'\n'),'History has an unfinished final line')
    history=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    def finite_tree(value):
        if isinstance(value,dict): return all(finite_tree(v) for v in value.values())
        if isinstance(value,list): return all(finite_tree(v) for v in value)
        return math.isfinite(value) if isinstance(value,(int,float)) and not isinstance(value,bool) else True
    require([r.get('epoch') for r in history]==list(range(200))
            and all(r.get('iterations')==125 and isinstance(r.get('loss'),(int,float))
                    and not isinstance(r['loss'],bool) and math.isfinite(r['loss']) for r in history),
            'History lacks the complete finite200x125 run')
    require(all(finite_tree(row) for row in history),'Nonfinite numeric training telemetry')
    require(fingerprint(history_path)==history_verified and fingerprint(marker_path)==marker_before,
            'Verified history/completion bytes changed during parsing')
    require(fingerprint(training/'final.pt')==actual and digest(training/'config.json')==EXPECTED_CONFIG,
            'Actual final/config changed across completed-payload checks')
    return actual,marker,{'TRAINING_COMPLETE.json':marker_before,'history.jsonl':history_verified}


def process_identity(root,proc_root=Path('/proc')):
    process=proc_root/str(MAIN_PID)
    require(process.is_dir(),'Confirmed main PID disappeared before a completion marker')
    argv=[p.decode('utf-8') for p in (process/'cmdline').read_bytes().split(b'\0') if p]
    cwd=(process/'cwd').resolve()
    scripts=[(Path(x) if Path(x).is_absolute() else cwd/x).resolve() for x in argv if x.endswith('train.py')]
    require(scripts==[(root/'code/train.py').resolve()],'PID1943 is not this fixed main training script')
    def option(name):
        require(argv.count(name)==1,'Main process argument missing/duplicated: '+name)
        index=argv.index(name); require(index+1<len(argv),'Main process argument value missing')
        return argv[index+1]
    require(option('--method')=='gta' and option('--epochs')=='200' and option('--steps')=='125'
            and (Path(option('--output')) if Path(option('--output')).is_absolute() else cwd/option('--output')).resolve()
                ==(root/'results/gta_enhanced200x125/training').resolve()
            and (Path(option('--manifest')) if Path(option('--manifest')).is_absolute() else cwd/option('--manifest')).resolve()
                ==(root/'dataset/cache/manifest.json').resolve(), 'Main PID argument identity differs')
    stat=(process/'stat').read_text(); tail=stat[stat.rfind(')')+2:].split()
    require(len(tail)>19 and tail[0]!='Z','Main PID is a zombie/invalid stat')
    return {'pid':MAIN_PID,'starttime_ticks':tail[19],'boot_id':(proc_root/'sys/kernel/random/boot_id').read_text().strip(),
            'script':str((root/'code/train.py').resolve())}


def runtime_audit(root,manifest):
    path=root/'EARLY_FINAL_INPUT_RUNTIME_AUDIT_20261014.json'; expected=fingerprint(path); audit=read(path)
    require(audit.get('training_source_sha256')==EXPECTED_SOURCE and audit.get('manifest_sha256')==EXPECTED_MANIFEST
            and audit.get('evaluation_sources')==EVALUATION_SOURCES,'Input runtime source/evaluator audit differs')
    require(audit.get('python')==sys.version,'Current Python differs from original runtime input audit')
    recorded=audit.get('cache_CT_inputs'); require(isinstance(recorded,dict) and set(recorded)==set(IDS),'Runtime CT audit is not exact internal4')
    cases={c['case_id']:c for c in manifest['cases']}; actual={}
    for case_id in IDS:
        ct=(root/'dataset/cache'/cases[case_id]['cache_subdir']/'image.npy').resolve()
        require(ct.is_relative_to((root/'dataset/cache').resolve()),'Cached CT escaped frozen cache root')
        actual[case_id]=fingerprint(ct)
    require(actual==recorded,'Actual internal4 CT bytes differ from prior input-runtime audit')
    packages=audit.get('packages'); require(isinstance(packages,dict) and set(packages)==PACKAGE_NAMES,
                                          'Expected the fixed9 audited runtime package names')
    actual_packages={}
    for name,version in packages.items():
        require(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name)
                and isinstance(version,str) and bool(version),'Invalid audited package/version')
        actual_packages[name]=importlib.metadata.version(name)
    require(actual_packages==packages,'Installed runtime package versions differ from prior input audit')
    require(fingerprint(path)==expected,'Input runtime audit changed while verifying CT/runtime')
    return path,audit,actual,actual_packages,expected


def verify_existing(destination,ready):
    require(destination.is_dir() and read(destination/'READY.json')==ready,'Existing mirror identity differs; never overwrite')
    require({p.name for p in destination.iterdir()}==PAYLOAD_NAMES|{'READY.json'},'Existing mirror file inventory differs')
    for name,expected in ready['files'].items(): require(fingerprint(destination/name)==expected,'Existing mirror file changed: '+name)


def rename_fresh(source,destination):
    """Never replace even an empty destination directory (including a race)."""
    if os.name=='nt':
        Path(source).rename(destination)  # Windows rename refuses existing targets.
        return
    import ctypes
    library=ctypes.CDLL(None,use_errno=True)
    require(hasattr(library,'renameat2'),'Filesystem-safe no-replace directory rename unavailable; no fallback overwrite')
    rename=library.renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]; rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(destination),1):
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error),str(destination))


def copy_atomic(sources,destination,ready,ready_name='READY.json',progress=None,before_publish=None):
    """Low-level byte copy primitive; fixtures use NOT_MODEL_RESULTS_READY.json."""
    require(set(sources)==PAYLOAD_NAMES and set(ready['files'])==PAYLOAD_NAMES,'Payload must contain the fixed six filenames')
    destination=Path(destination)
    require(not destination.exists(),'Fresh mirror required; existing destination will not be overwritten')
    destination.parent.mkdir(parents=True,exist_ok=True)
    total=sum(v['bytes'] for v in ready['files'].values())
    require(shutil.disk_usage(destination.parent).free>=total+512*1024**2,'Insufficient shared free space for copy plus512MiB headroom')
    temporary=destination.with_name('.'+destination.name+'.pending.'+uuid.uuid4().hex); temporary.mkdir()
    if progress: progress(temporary)
    for name,path in sources.items():
        require(name==Path(name).name and fingerprint(path)==ready['files'][name],'Source file changed before copying: '+name)
        copied=temporary/name
        with Path(path).open('rb') as source,copied.open('xb') as target:
            shutil.copyfileobj(source,target,length=8*1024**2); target.flush(); os.fsync(target.fileno())
        require(fingerprint(path)==ready['files'][name] and fingerprint(copied)==ready['files'][name],
                'Source/copy differs after copying: '+name)
    atomic_json(temporary/ready_name,ready)
    require(read(temporary/ready_name)==ready,'Temporary readiness record differs')
    if before_publish: before_publish()
    for name,path in sources.items(): require(fingerprint(path)==ready['files'][name],'Source changed before publication: '+name)
    require(not destination.exists(),'Destination appeared; refuse to replace it')
    rename_fresh(temporary,destination)
    for name,expected in ready['files'].items(): require(fingerprint(destination/name)==expected,'Published copy changed: '+name)
    require(read(destination/ready_name)==ready,'Published readiness record changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--destination',type=Path,default=DEFAULT_DESTINATION)
    parser.add_argument('--wait-seconds',type=int,default=7200); parser.add_argument('--plan',action='store_true')
    args=parser.parse_args(); root=args.root.resolve(); destination=args.destination.resolve()
    require(0<=args.wait_seconds<=7200,'Bounded waiting must be0..7200 seconds')
    require(destination==DEFAULT_DESTINATION.resolve(),'Only the fixed shared mirror destination is authorized')
    require(not args.destination.is_symlink(),'Mirror destination must not be a symlink')
    helper=root/'ops/verify_postfit.py'; source_guard=load_source_guard(root)
    config,manifest=static_guard(root,source_guard)
    protocol={'schema_version':1,'source_root':str(root),'destination':str(destination),'main_pid':MAIN_PID,
        'config_sha256':EXPECTED_CONFIG,'manifest_sha256':EXPECTED_MANIFEST,'source_sha256':EXPECTED_SOURCE,
        'source_freeze_sha256':EXPECTED_FREEZE,'evaluation_sources':EVALUATION_SOURCES,'source_guard_helper_sha256':EXPECTED_HELPER,
        'publisher_sha256':digest(__file__),'wait_seconds':args.wait_seconds,
        'role':'early fixed-final inference mirror; no new training','GPU_used':False,'model_loaded':False,'training_started':False}
    if args.plan:
        print(json.dumps(protocol|{'status':'plan_only','READY_created':False,'copy_started':False},indent=2)); return
    require(root==DEFAULT_ROOT.resolve() and Path(__file__).resolve().parents[1]==root,'Publisher must run in the actual NEW_ROOT')
    import fcntl
    lock=(root/'.early_final_mirror.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    destination.parent.mkdir(parents=True,exist_ok=True)
    target_lock=(destination.parent/'.early_final_mirror_20261014.lock').open('a+')
    try: fcntl.flock(target_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException: lock.close(); target_lock.close(); raise
    status=root/'EARLY_FINAL_MIRROR_STATUS.json'; state={}; began=False
    def save(phase,**values):
        state.update(phase=phase,updated_at=now(),**values); atomic_json(status,state)
    def guard():
        require(digest(__file__)==protocol['publisher_sha256'] and digest(helper)==EXPECTED_HELPER,'Publisher/helper source changed')
        static_guard(root,source_guard)
    def stop(signum,frame): raise KeyboardInterrupt('Publisher signal '+str(signum)+'; main/postfit untouched')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        if status.exists():
            prior=read(status); require(prior.get('protocol')==protocol,'Existing publisher protocol differs; inspect without overwriting')
            deadline=prior['deadline_unix']; pinned=prior.get('confirmed_main_process')
        else: deadline=time.time()+args.wait_seconds; pinned=None
        state={'protocol':protocol,'pid':os.getpid(),'started_at':now(),'deadline_unix':deadline,
               'confirmed_main_process':pinned,'status':'waiting','main_or_postfit_signals_sent':False}
        began=True; save('waiting_for_verified_fixed_final')
        marker=root/'results/gta_enhanced200x125/training/TRAINING_COMPLETE.json'
        while not marker.is_file():
            guard()
            try: identity=process_identity(root)
            except (OSError,ValueError):
                if marker.is_file(): break
                raise
            if pinned is None: pinned=identity
            require(identity==pinned,'Main PID was reused/restarted; no unconfirmed waiting')
            state['confirmed_main_process']=pinned
            require(time.time()<deadline,'7200s-bounded wait expired without completion')
            save('waiting_for_verified_fixed_final',remaining_seconds=max(0,deadline-time.time()))
            time.sleep(min(30,max(0,deadline-time.time())))
        guard(); final,complete,verified_training_files=completed_payload(root,config); save('verifying_input_runtime_audit',status='checking')
        audit_path,audit,ct_inputs,packages,verified_runtime_audit=runtime_audit(root,manifest)
        training=root/'results/gta_enhanced200x125/training'
        sources={'final.pt':training/'final.pt','config.json':training/'config.json','history.jsonl':training/'history.jsonl',
                 'TRAINING_COMPLETE.json':training/'TRAINING_COMPLETE.json','manifest.json':root/'dataset/cache/manifest.json',
                 'input_runtime_audit.json':audit_path}
        files={name:fingerprint(path) for name,path in sources.items()}
        require(files['final.pt']==final and files['config.json']['sha256']==EXPECTED_CONFIG,'Completed actual final/config moved')
        require(all(files[name]==expected for name,expected in verified_training_files.items())
                and files['input_runtime_audit.json']==verified_runtime_audit,
                'Semantically verified completion/history/runtime audit bytes changed before copy')
        ready={'schema_version':1,'status':'fixed_final_payload_ready','role':'early fixed-final inference mirror; no new training',
            'source_root':str(root),'files':files,'config_sha256':EXPECTED_CONFIG,'manifest_sha256':EXPECTED_MANIFEST,
            'source_sha256':EXPECTED_SOURCE,'evaluation_sources':EVALUATION_SOURCES,'case_ids':IDS,
            'checkpoint_sha256':complete['final_sha256'],'checkpoint_epoch':199,'cache_CT_inputs':ct_inputs,'packages':packages,
            'raw_GT_source_sha256':{i:next(c for c in manifest['cases'] if c['case_id']==i)['source_sha256']['label'] for i in IDS},
            'source_freeze_sha256':EXPECTED_FREEZE,'source_guard_full_package_verified':True,
            'publisher_sha256':protocol['publisher_sha256'],'source_guard_helper_sha256':EXPECTED_HELPER,
            'training_updates':25000,'diagnostic_only':True,'formal_final_test':False,'new_training':False,'canonical80_case_evaluation_unchanged':True,
            'captured_at_utc':now(),'model_loaded':False,'checkpoint_payload_deserialized':False}
        guard()
        if destination.exists():
            saved=read(destination/'READY.json')
            # Capture time is fixed by the first successful publication.
            ready['captured_at_utc']=saved.get('captured_at_utc')
            verify_existing(destination,ready)
            save('existing_identical_mirror_verified',status='completed',READY=str(destination/'READY.json'))
        else:
            save('copying_verified_payload',status='copying')
            copy_atomic(sources,destination,ready,progress=lambda path:save('copying_verified_payload',temporary_directory=str(path)),before_publish=guard)
            guard(); require(fingerprint(training/'final.pt')==final and digest(training/'config.json')==EXPECTED_CONFIG,
                             'Actual final/config changed after publication')
            save('fixed_final_mirror_published',status='completed',READY=str(destination/'READY.json'),READY_sha256=digest(destination/'READY.json'))
        print(json.dumps({'status':'completed','destination':str(destination),'main_or_postfit_touched':False,'training_started':False}))
    except BaseException as error:
        if began: save('stopped_without_touching_main',status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        raise
    finally:
        fcntl.flock(target_lock,fcntl.LOCK_UN); target_lock.close(); fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__': main()
