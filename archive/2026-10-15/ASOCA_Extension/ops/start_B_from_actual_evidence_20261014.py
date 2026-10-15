"""Publish ROOT's reviewed one-candidate decision and launch the reviewed controller."""
import datetime
import hashlib
import json
from pathlib import Path
from remote import connect

LOCAL = Path(__file__).resolve().parents[1]/'fidelity_v2'
REMOTE = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
OUT = LOCAL/'server_records/B_formal_launch_20261014'
DIAG = LOCAL/'server_records/initial_parameter_gradient_actual_20261014/results/initial_gaussian_parameter_gradients_amp_20261014'
CANON = LOCAL/'server_records/gta_complete_trajectory_20261014'
VARIANT = LOCAL/'variants/gaussian_region_balance_v1'


def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(path.read_bytes())
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not OUT.exists(), 'Preserve existing launch records; inspect before retrying'
    summary = read(DIAG/'summary.json')
    numerical_review = read(LOCAL/'server_records/initial_parameter_gradient_independent_review_20261014/review.json')
    acceptance = read(LOCAL/'server_records/gta_canonical_acceptance_20261014/canonical_acceptance_and_pair.json')
    assert numerical_review['status']=='passed' and numerical_review['n_batches']==28
    assert acceptance['status']=='verified_real_canonical20_80_and_fixed_final_pair'
    assert acceptance['canonical_preview_all4_decoded_masks_equal'] is True
    assert summary['nonfinite_batches']==[] and summary['batches_implying_clip12']=={'totalA':[],'totalB':[8,12,19,21,23]}
    client = connect(38773)
    try:
        with client.open_sftp() as sftp:
            with sftp.open(REMOTE+'/ops_inputs/B_FORMAL_EVIDENCE_20261014.json','rb') as f: evidence_bytes=f.read()
            assert sha(evidence_bytes)=='b24d957ef7db1eac4d6f9091b303bb6d943a9f6384af9f38ed79a8caec2364a4'
            proof = json.loads(evidence_bytes)['evidence']
            assert proof['canonical']==read(DIAG/'diagnostic_plan.json')['canonical']
            for key,name in [('postfit_receipt_sha256','POSTFIT_VERIFIED.json'),('postfit_status_sha256','POSTFIT_STATUS.json'),('trajectory_complete_sha256','results/gta_enhanced200x125/internal4_trajectory/TRAJECTORY_COMPLETE.json')]:
                assert proof['canonical'][key]==file_sha(CANON/name)
            assert proof['B_protocol_sha256']==file_sha(VARIANT/'protocol.json')
            assert proof['B_READY_sha256']==file_sha(VARIANT/'VARIANT_READY.json')
            assert proof['ROOT_variant_review_sha256']==file_sha(LOCAL/'docs/region_balance_v1_root_review_20261014.txt')
            assert proof['initial_gradient_complete_sha256']==file_sha(DIAG/'DIAGNOSTIC_COMPLETE.json')
            for name,digest in proof['initial_gradient_artifacts_sha256'].items():assert digest==file_sha(DIAG/name)
            for key,name in [('actual_CPU_fixture_sha256','cpu_fixture.json'),('initial_gradient_controller_protocol_sha256','CONTROLLER_PROTOCOL.json'),('initial_gradient_controller_status_sha256','CONTROLLER_STATUS.json')]:
                assert proof[key]==file_sha(DIAG/name)
            config=read(CANON/'results/gta_enhanced200x125/training/config.json')
            donor_ready=read(LOCAL/'server_records/gaussian_region_balance_actual_20261014/records/new_38773/shared_donors/DONORS_READY.json')
            for component,name in [('backbone','airway_backbone_epoch599.pkl'),('classifier','classifier80_final.pt')]:
                donor=donor_ready['files'][name]
                assert proof['donors'][component]=={'path':config[component+'_init'],'sha256':donor['sha256'],'bytes':donor['bytes']}
            decision={'schema_version':1,'action':'authorize_one_B_formal_run','decision_author':'ROOT',
                'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'output':REMOTE+'/results/gta_region_balance200x125_20261014/training','evidence':proof,
                'canonical_review':{'all20_checkpoints_reviewed':True,'late_checkpoint_names':['epoch160','epoch170','epoch180','epoch190','final'],
                    'rationale':'All20 actual observations and individual final patients were reviewed; local CPU acceptance verified80case metadata/prediction SHA and4final raw-grid primary metrics; OLD and NEW final masks match exactly. Late Dice=.829948/.830684/.828404/.829990/.831854 and ASSD=.656950/.587544/.562398/.627177/.567825 fluctuate; epoch140 mean .832446/.534270 also exceeds final. This does not justify extending A or selecting a different endpoint. Retain fixed final and test one B loss-measure change at the same25000-update budget.'},
                'gradient_review':{'all28_batches_reviewed':True,'nonfinite_batches':summary['nonfinite_batches'],
                    'batches_implying_clip12':summary['batches_implying_clip12'],'proceed_with_fixed_lambda10_after_risk_review':True,
                    'rationale':'Independent13file/5artifact and all28ledger review passed. No nonfinite gradients. A total-norm median/max3.30274/3.89479; B8.61351/19.09710. B implies existing global clip12 on5/28 standalone batches, min multiplier.628368, which also scales the segmentation component. Shared Gaussian/seg norm median rises.61025->3.24517; B cosine remains positive but weaker (median.08345 vs A.29349), and Gaussian-head share median2.94percent shows the stronger gradient mostly reaches shared segmentation parameters. Accept this explicit optimization risk for one unchanged-lambda10 mechanism trial; no coefficient/cap/architecture/budget adjustment. These initial noaug frozen-stage batches are not natural training frequencies or proof of later stability or benefit. Preserve loss/LR/group-gradient telemetry and stop on actual nonfinite failure; judge outcome by fixed-final Dice/ASSD and complete curves. Do not compare these parameter gradients directly with the distinct final-model detached-logit probe.'},
                'independent_numeric_review_sha256':file_sha(LOCAL/'server_records/initial_parameter_gradient_independent_review_20261014/review.json'),
                'local_canonical_acceptance_sha256':file_sha(LOCAL/'server_records/gta_canonical_acceptance_20261014/canonical_acceptance_and_pair.json'),
                'historical_main_theta0_verified':False,'new_classifier_pretraining':False,'new_segmentation_updates':25000}
            encoded=(json.dumps(decision,indent=2,allow_nan=False)+'\n').encode()
            OUT.mkdir()
            (OUT/'actual_evidence.json').write_bytes(evidence_bytes)
            (OUT/'B_TRAIN_DECISION_20261014.json').write_bytes(encoded)
            with sftp.open(REMOTE+'/ops_inputs/B_TRAIN_DECISION_20261014.json','wx') as f:f.write(encoded)
            with sftp.open(REMOTE+'/ops_inputs/B_TRAIN_DECISION_20261014.json','rb') as f:assert f.read()==encoded
        command=Path(__file__).with_name('launch_B_38773.txt').read_text(encoding='utf-8')
        _,stdout,stderr=client.exec_command(command)
        data=stdout.read();errors=stderr.read();code=stdout.channel.recv_exit_status()
        (OUT/'launch_stdout.txt').write_bytes(data);(OUT/'launch_stderr.txt').write_bytes(errors)
        assert code==0,errors.decode(errors='replace')
        launch=json.loads(data)
        assert launch['decision_sha256']==sha(encoded)
        (OUT/'launch_receipt.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(launch,indent=2))
    finally:client.close()


if __name__=='__main__':main()
