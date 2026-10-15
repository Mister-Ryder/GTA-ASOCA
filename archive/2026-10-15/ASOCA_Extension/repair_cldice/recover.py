"""Rebuild the author-formula clDice baseline after the immutable main queue.

Original runs and their source remain unchanged. Corrected comparison explicitly
selects the retrained baseline and retains per-method implementation provenance.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/ASOCA_GTAirway_20261013')
REPAIR = ROOT / 'repair_cldice'
CODE = REPAIR / 'code'
sys.path.insert(0, str(CODE))
from run_queue import verify_evaluation
from aggregate_results import atomic_json, load_protocol, aggregate_outoffold, MAIN_METHODS, ABLATION_METHODS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, logfile):
    with logfile.open('a') as stream:
        stream.write('\nREPAIR COMMAND '+json.dumps(command)+'\n')
        stream.flush()
        subprocess.run(command, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    REPAIR.mkdir(parents=True, exist_ok=True)
    statefile = REPAIR / 'repair_status.json'
    state = {'status':'waiting_for_main_queue','pid':os.getpid(),'reason':'Correct author-formula soft-clDice; original failed attempt excluded',
             'training':'fresh initialization for every fold; no resume across changed objectives', 'folds':[]}
    atomic_json(statefile, state)
    while True:
        parent = json.loads((ROOT/'results/queue_status.json').read_text())
        if parent['status'] in ['completed','finished_with_failures']:
            break
        if parent['status'] not in ['running']:
            raise RuntimeError('Main queue stopped unexpectedly; do not overlap an unverified run')
        time.sleep(60)
    if not (REPAIR/'tests_passed.json').exists():
        raise RuntimeError('Corrected loss has no passed-test marker')
    while True:
        occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not occupied or 'No running processes' in occupied:
            break
        state.update(status='waiting_for_idle_gpu',gpu_processes=occupied)
        atomic_json(statefile,state)
        time.sleep(60)
    manifest_path = ROOT/'cache/manifest.json'
    manifest, manifest_hash = load_protocol(manifest_path)
    protocol = {'manifest_sha256':manifest_hash,'implementation':'author-formula soft-clDice; smooth=1; global reduction',
                'source_hashes':{p.relative_to(CODE).as_posix():sha(p) for p in CODE.rglob('*.py')},
                'epochs':200,'steps_per_epoch':125,'patch_size':96,'batch_size':2,'deep_supervision':True,
                'source_reference':'https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py',
                'base_protocol_identity':parent['protocol_identity'],'original_outputs':'Retained unchanged under original results/',
                'fixed_final_checkpoint':True}
    protocol_id = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    atomic_json(REPAIR/'repair_protocol.json',protocol)
    state.update(status='running',protocol_identity=protocol_id)
    atomic_json(statefile,state)
    for fold in range(5):
        folder = REPAIR/'results/unet_cldice'/f'fold{fold}'
        folder.mkdir(parents=True,exist_ok=True)
        current = {'fold':fold,'status':'running'}
        state['folds'].append(current)
        atomic_json(statefile,state)
        try:
            train = [sys.executable,'-u',str(CODE/'train.py'),'--manifest',str(manifest_path),'--output',str(folder/'training'),
                     '--method','unet_cldice','--fold',str(fold),'--epochs','200','--steps','125','--patch-size','96',
                     '--batch-size','2','--workers','4']
            complete = folder/'training/TRAINING_COMPLETE.json'
            if not complete.exists():
                if (folder/'training/latest.pt').exists(): train.append('--resume')
                run(train,folder/'training.log')
            evaluate = [sys.executable,'-u',str(CODE/'evaluate.py'),'--manifest',str(manifest_path),
                        '--raw-root',str(ROOT/'raw/CompressedVersion'),'--checkpoint',str(folder/'training/final.pt'),
                        '--output',str(folder/'evaluation'),'--fold',str(fold)]
            run(evaluate,folder/'evaluation.log')
            job = {'folder':str(folder),'kind':'custom','fold':fold,'method_label':'unet_cldice',
                   'training_method':'unet_cldice','deep_supervision':True}
            verified = verify_evaluation(job,manifest,manifest_hash)
            marker = {'schema_version':1,'status':'completed','smoke':False,'method_label':'unet_cldice','fold':fold,
                      'manifest_sha256':manifest_hash,'protocol_identity':protocol_id,
                      'implementation':'author clDice correction, fresh training',**verified}
            atomic_json(folder/'JOB_COMPLETE.json',marker)
            current.update(status='completed',**verified)
        except Exception as error:
            current.update(status='failed',error=repr(error))
        atomic_json(statefile,state)
    # A separate report bundle permits explicit, source-backed implementations.
    # All method rows still undergo original patient/label/checkpoint validation.
    bundle = REPAIR/'report_bundle'
    (bundle/'results').mkdir(parents=True,exist_ok=True)
    selected = {}
    for method in MAIN_METHODS+ABLATION_METHODS:
        source = (REPAIR if method=='unet_cldice' else ROOT)/'results'/method
        target = bundle/'results'/method
        if source.exists() and not target.exists(): target.symlink_to(source,target_is_directory=True)
        selected[method] = {'source':str(source),'implementation_protocol':protocol_id if method=='unet_cldice' else parent['protocol_identity']}
    output = ROOT/'results/comparison_corrected'
    atomic_json(output/'implementation_selection.json',{'selected_methods':selected,'correction_protocol':protocol})
    aggregate_outoffold(bundle,manifest_path,output=output)
    state['status'] = 'completed' if all(x['status']=='completed' for x in state['folds']) else 'finished_with_failures'
    state['corrected_comparison'] = str(output)
    atomic_json(statefile,state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
