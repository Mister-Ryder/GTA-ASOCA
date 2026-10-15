"""Capture actual B config/owner/initial-state receipts, no remote mutation."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from remote import connect

LOCAL=Path(__file__).resolve().parents[1]/'fidelity_v2'
REMOTE='/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
RECORDS=REMOTE+'/results/gaussian_region_balance_controller_20261014'
TRAIN=REMOTE+'/results/gta_region_balance200x125_20261014/training'
OUT=LOCAL/'server_records/B_formal_launch_20261014/actual_startup'


def main():
    assert not OUT.exists(), 'Keep earlier actual startup captures immutable'
    paths={'deployed_controller.py':REMOTE+'/ops/run_gaussian_region_balance_control.py',
        'launch_receipt.json':REMOTE+'/GAUSSIAN_REGION_BALANCE_CONTROL_LAUNCH_20261014.json',
        'protocol.json':RECORDS+'/PROTOCOL.json','status_at_capture.json':RECORDS+'/STATUS.json',
        'expected_B_config.json':RECORDS+'/expected_B_config.json','train_argv.json':RECORDS+'/train_argv.json',
        'ACTUAL_B_THETA0.json':RECORDS+'/ACTUAL_B_THETA0.json',
        'actual_training_config.json':TRAIN+'/config.json','history_at_capture.jsonl':TRAIN+'/history.jsonl'}
    client=connect(38773)
    try:
        with client.open_sftp() as sftp:
            values={}
            for name,source in paths.items():
                with sftp.open(source,'rb') as f:values[name]=f.read()
            sha=lambda b:hashlib.sha256(b).hexdigest()
            assert sha(values['deployed_controller.py'])=='ace0561942714916fe35e4717a1c3df2736aa85b44e63bf978ea9b9fa8da1f1b'
            theta=json.loads(values['ACTUAL_B_THETA0.json'])
            config=json.loads(values['actual_training_config.json'])
            assert config==json.loads(values['expected_B_config.json'])
            assert theta['state']['tensor_count']==327 and theta['state']['state_sha256']=='d84df10851c2532e9b70311eba2efe07049f5348e56fe4e76daf7856e6427d79'
            assert theta['network_forwards_before_capture']==theta['optimizer_updates']==0
            assert theta['model_state_and_RNG_unchanged'] and not theta['historical_main_theta0_verified']
            assert theta['actual_config_binding']['actual_config_sha256']==sha(values['actual_training_config.json'])
            spec=importlib.util.spec_from_file_location('B_startup_checker',LOCAL/'variants/gaussian_region_balance_v1/ops/check_variant.py')
            checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
            protocol,inspection=checker.inspect()
            OUT.mkdir()
            for name,data in values.items():(OUT/name).write_bytes(data)
            bound=checker.actual_config(protocol,inspection,
                LOCAL/'server_records/gta_complete_trajectory_20261014/results/gta_enhanced200x125/training/config.json',
                OUT/'actual_training_config.json',TRAIN)
            assert bound==theta['actual_config_binding']
            receipt={'status':'actual_B_config_and_initial_state_verified_not_final_results',
                'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'files':{name:{'remote':paths[name],'bytes':len(data),'sha256':sha(data)} for name,data in values.items()},
                'independent_local_actual_config_binding':bound,'actual_historical_main_theta0_verified':False,
                'model_weights_downloaded':False,'remote_commands_or_writes':False}
            (OUT/'capture_complete.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'status':receipt['status'],'files':len(values),'actual_config_sha256':bound['actual_config_sha256'],'actual_initial_state_sha256':theta['state']['state_sha256']}))
    finally:client.close()


if __name__=='__main__':main()
