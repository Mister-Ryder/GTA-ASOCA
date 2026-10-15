"""Read-only retrieval of actual source-only CPU smoke receipts/logs."""
import datetime,hashlib,json
from pathlib import Path
from remote import connect
root='/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2/public_source_cpu_smoke_20261014'
output=Path(__file__).resolve().parents[1]/'fidelity_v2/server_records/public_source_cpu_smoke_actual_20261014'
assert not output.exists(),'Never overwrite captured source-test evidence'
output.mkdir(parents=True)
files=['ANONYMOUS_CLONE_RECEIPT.json','anonymous_clone.log','receipts/PUBLIC_SOURCE_CPU_SMOKE_COMPLETE.json']
files += ['receipts/worker'+f'{index:02d}'+'_'+('help' if index<6 else 'main_forward' if index==6 else 'classifier_forward')+suffix for index in range(8) for suffix in ('.json','.log')]
capture={'source_root':root,'server_port':17552,'read_only_server_access':True,'role':'NOT_MODEL_RESULTS','files':[]}
client=connect(17552)
try:
    with client.open_sftp() as sftp:
        for name in files:
            before=sftp.stat(root+'/'+name)
            dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True)
            sftp.get(root+'/'+name,str(dest));after=sftp.stat(root+'/'+name)
            assert (before.st_size,before.st_mtime)==(after.st_size,after.st_mtime)
            capture['files'].append({'relative_path':name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    complete=json.loads((output/'receipts/PUBLIC_SOURCE_CPU_SMOKE_COMPLETE.json').read_text())
    assert complete['status']=='passed' and complete['role']=='NOT_MODEL_RESULTS' and len(complete['worker_records'])==8
    assert not complete['CUDA_context_initialized'] and not complete['private_checkpoint_or_real_data_used'] and complete['parameter_updates']==0
    for index,row in enumerate(complete['worker_records']):
        name='worker'+f'{index:02d}'+'_'+row['mode']
        local=output/'receipts'/name
        assert hashlib.sha256(local.with_suffix('.json').read_bytes()).hexdigest()==row['receipt_sha256']
        assert hashlib.sha256(local.with_suffix('.log').read_bytes()).hexdigest()==row['log_sha256']
        worker=json.loads(local.with_suffix('.json').read_text())
        assert worker['status']=='passed' and worker['role']=='NOT_MODEL_RESULTS' and not worker['CUDA_context_initialized'] and worker['parameter_updates']==0
        if row['mode']!='help':assert worker['state_unchanged'] and worker['all_parameter_gradients_none'] and worker['all_outputs_finite']
    capture.update(status='actual19files_captured_and_worker_hashes_verified',captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    print(json.dumps({'status':capture['status'],'files':len(files),'output':str(output),'clinical_metrics_or_training':False}))
finally:
    client.close()
    (output/'collection_manifest.json').write_text(json.dumps(capture,indent=2)+'\n')
