"""Capture immutable completed training bytes; no training or remote mutation."""
import datetime,hashlib,json
from pathlib import Path
from remote import connect

root='/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
training='results/gta_enhanced200x125/training'
shared='/root/autodl-fs/experiments/GTA_ASOCA_Fidelity_20261014/early_final_mirror_20261014'
output=Path(__file__).resolve().parents[1]/'fidelity_v2/server_records/gta_training_complete_20261014'
assert not output.exists(),'Use a fresh evidence directory; never overwrite captured bytes'
output.mkdir(parents=True)
paths=[(root+'/'+training+'/'+name,training+'/'+name) for name in ('config.json','history.jsonl','initialization_audit.json','TRAINING_COMPLETE.json','final.pt')]
paths += [(root+'/'+name,name) for name in ('stage_records/gta200.json','configs/main_source_freeze.json','LIVE_MAIN_CHECK_20261014.json','FIXED_FINAL_FOLLOWUP_CHECK_20261014.json','EARLY_FINAL_MIRROR_STATUS.json','EARLY_FINAL_MIRROR_LAUNCH_20261014.json')]
paths += [(shared+'/READY.json','shared_mirror/READY.json')]
capture={'source_root':root,'server_port':38773,'read_only_server_access':True,'files':[]}
client=connect(38773)
try:
    with client.open_sftp() as sftp:
        for source,relative in paths:
            before=sftp.stat(source)
            target=output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            sftp.get(source,str(target))
            after=sftp.stat(source)
            assert (before.st_size,before.st_mtime)==(after.st_size,after.st_mtime),'File changed while captured: '+source
            capture['files'].append({'relative_path':relative,'remote_path':source,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    complete=json.loads((output/training/'TRAINING_COMPLETE.json').read_text())
    ready=json.loads((output/'shared_mirror/READY.json').read_text())
    config_record=next(r for r in capture['files'] if r['relative_path']==training+'/config.json')
    final_record=next(r for r in capture['files'] if r['relative_path']==training+'/final.pt')
    assert final_record['sha256']==complete['final_sha256']==ready['checkpoint_sha256']
    assert final_record['bytes']==ready['files']['final.pt']['bytes']
    assert config_record['sha256']==ready['config_sha256']=='b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
    assert complete['epoch']==199 and complete['segmentation_updates']==25000 and not complete['smoke']
    capture.update(status='captured_actual_completed_training_pending_independent_review',captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_final_weights_bytes_hash_verified=True,model_forward_or_training_started=False)
    print(json.dumps({'output':str(output),'files':len(capture['files']),'final_sha256':final_record['sha256'],'status':capture['status']}),flush=True)
except Exception as error:
    capture.update(status='partial_capture_retained',error=repr(error))
    raise
finally:
    client.close()
    (output/'collection_manifest.json').write_text(json.dumps(capture,indent=2)+'\n')
