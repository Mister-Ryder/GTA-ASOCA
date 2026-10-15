"""Bounded read-only capture on the known preview process; no job launches."""
import datetime,hashlib,json,os,time
from pathlib import Path
from remote import connect

root='/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2'
relative_eval='results/gta_fixed_final_preview_20261014'
base=Path(__file__).resolve().parents[1]/'fidelity_v2/server_records'
output=base/'gta_fixed_final_preview_capture_20261014'
capture={'server_port':17552,'source_root':root,'scope':'known controller136666 read-only capture; no launches/restarts','files':[],'wait_observations':[]}
if output.exists():
    prior=json.loads((output/'collection_manifest.json').read_text())
    assert prior['status']=='partial_capture_retained','Only the inspected partial capture may continue'
    for row in prior['files']:
        local=output/row['relative_path']
        assert local.stat().st_size==row['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==row['sha256']
    capture['files']=prior['files']
    capture['continued_from_inspected_partial']=True
    capture['prior_capture_failure']=prior.get('error')
else:output.mkdir(parents=True)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(sftp,path):
    with sftp.open(path,'rb') as stream:return stream.read()
def exists(sftp,path):
    try:sftp.stat(path);return True
    except FileNotFoundError:return False
client=connect(17552)
try:
    with client.open_sftp() as sftp:
        deadline=time.monotonic()+180
        while not exists(sftp,root+'/'+relative_eval+'/EVAL_COMPLETE.json'):
            state=json.loads(read(sftp,root+'/EARLY_FIXED_FINAL_STATUS.json'))
            assert state['controller_pid']==136666
            try:
                command=read(sftp,'/proc/136666/cmdline').decode()
                live_state=read(sftp,'/proc/136666/stat').decode().rsplit(')',1)[1].split()[0]
                alive=root+'/ops/run_early_fixed_final_eval.py' in command and live_state not in ('Z','X')
            except FileNotFoundError:alive=False
            observation={'captured_at_utc':now(),'controller_pid':136666,'actual_command_identity_live':alive,'state':state}
            capture['wait_observations'].append(observation)
            if not alive:
                if exists(sftp,root+'/'+relative_eval+'/EVAL_COMPLETE.json'):break
                raise RuntimeError('Known controller missing without completion; inspect its terminal state/log, never restart solely on timeout')
            assert state['status'] not in ('failed_stop','interrupted')
            if time.monotonic()>=deadline:
                capture.update(status='verified_wait_observation_timeout_no_restart',captured_at_utc=now())
                print(json.dumps({'status':capture['status'],'actual_controller_live':alive,'phase':state['status']}),flush=True)
                raise TimeoutError('Existing preview still live; observation ended without restart')
            time.sleep(min(15,deadline-time.monotonic()))
        # evaluate.py writes the summary/cases. The preview controller owns
        # EVAL_COMPLETE; EVALUATION_COMPLETE belongs to the canonical trajectory
        # wrapper and must never be invented for a preview.
        names=[relative_eval+'/'+name for name in ('summary.json','case_metrics.jsonl','EVAL_COMPLETE.json')]
        names += [relative_eval+'/'+case+suffix for case in ('Diseased_10','Diseased_20','Normal_12','Normal_3') for suffix in ('.json','_prediction.nrrd')]
        names += ['EARLY_FIXED_FINAL_PROTOCOL.json','EARLY_FIXED_FINAL_STATUS.json','EARLY_FIXED_FINAL_LAUNCH_20261014.json','EARLY_FINAL_LIVE_CHECK_20261014.json','logs/early_fixed_final_controller.log','logs/early_fixed_final_eval.log','results/early_preview_controller_attempts/81785_waiting_stop_20261014/STOP_RECEIPT.json']
        for name in names:
            source=root+'/'+name;before=sftp.stat(source)
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
            saved=next((row for row in capture['files'] if row['relative_path']==name),None)
            if saved:
                assert target.stat().st_size==saved['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==saved['sha256']
                assert before.st_size==saved['bytes'] and hashlib.sha256(read(sftp,source)).hexdigest()==saved['sha256'],'Retained evidence differs from current completed remote bytes'
                continue
            assert not target.exists(),'Unrecorded partial file needs explicit inspection: '+name
            sftp.get(source,str(target));after=sftp.stat(source)
            assert (before.st_size,before.st_mtime)==(after.st_size,after.st_mtime),'Completed file changed during capture: '+name
            capture['files'].append({'relative_path':name,'remote_path':source,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    # Assemble content-identical local payload blobs from independently downloaded
    # source bytes; all are checked against the actual shared READY receipt.
    training=base/'gta_training_complete_20261014/results/gta_enhanced200x125/training'
    ready_source=base/'gta_training_complete_20261014/shared_mirror/READY.json'
    ready=json.loads(ready_source.read_text())
    view=output/'mirror_view';view.mkdir()
    sources={name:training/name for name in ('final.pt','config.json','history.jsonl','TRAINING_COMPLETE.json')}
    sources['manifest.json']=base/'classifier_scope_20261014/dataset/cache/manifest.json'
    sources['input_runtime_audit.json']=base/'early_fixed_final_preflight_20261014/38773_EARLY_FINAL_INPUT_RUNTIME_AUDIT_20261014.json'
    mapped=[]
    for name,source in sources.items():
        record={'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        assert record==ready['files'][name],name+' source blob differs from actual READY'
        os.link(source,view/name)
        assert hashlib.sha256((view/name).read_bytes()).hexdigest()==record['sha256']
        mapped.append({'name':name,'downloaded_source_path':str(source),'local_payload_path':str(view/name),**record})
    os.link(ready_source,view/'READY.json')
    receipt={'role':'local payload byte view assembled from actual downloaded source blobs; not a claim each blob was downloaded again from shared storage','READY_source_path':str(ready_source),'READY_sha256':hashlib.sha256(ready_source.read_bytes()).hexdigest(),'files':mapped,'training_or_inference_started':False}
    (output/'payload_view_assembly_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    capture.update(status='captured_completed_preview_pending_independent_pixel_review',captured_at_utc=now(),model_forward_or_training_started=False,payload_view_receipt_sha256=hashlib.sha256((output/'payload_view_assembly_receipt.json').read_bytes()).hexdigest())
    print(json.dumps({'status':capture['status'],'output':str(output),'files':len(capture['files'])}),flush=True)
except Exception as error:
    capture.update(status=capture.get('status','partial_capture_retained'),error=repr(error))
    raise
finally:
    client.close()
    (output/'collection_manifest.json').write_text(json.dumps(capture,indent=2)+'\n')
