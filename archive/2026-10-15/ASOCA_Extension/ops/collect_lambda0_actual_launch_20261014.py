"""Read actual lambda0 launch/config/initial-state receipts; no remote commands."""
import datetime
import hashlib
import json
from pathlib import Path
from remote import connect

OLD = '/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2'
LOGICAL = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
RECORDS = OLD+'/results/gaussian_lambda0_controller_20261014'
OUT = Path(__file__).resolve().parents[1]/'fidelity_v2/server_records/lambda0_actual_launch_20261014'


def main():
    assert not OUT.exists(), 'Fresh capture required; preserve existing actual bytes'
    client = connect(17552)
    try:
        with client.open_sftp() as sftp:
            paths = {
                'launch.json': OLD+'/GAUSSIAN_LAMBDA0_CONTROL_LAUNCH_20261014.json',
                'deployed_controller.py': OLD+'/ops/run_gaussian_lambda0_control.py',
                'protocol.json': RECORDS+'/PROTOCOL.json',
                'expected_lambda0_config.json': RECORDS+'/expected_lambda0_config.json',
                'train_argv.json': RECORDS+'/train_argv.json',
                'CPU_PREFLIGHT.json': RECORDS+'/cpu_preflight/CONTROL_PREFLIGHT.json',
                'cache_input_identity.json': RECORDS+'/cache_input_identity.json',
                'ACTUAL_CONTROL_THETA0.json': RECORDS+'/ACTUAL_CONTROL_THETA0.json',
                'actual_training_config.json': LOGICAL+'/results/gta_gaussian_lambda0_200x125_20261014/training/config.json',
                'status_at_capture.json': RECORDS+'/STATUS.json',
                'history_at_capture.jsonl': LOGICAL+'/results/gta_gaussian_lambda0_200x125_20261014/training/history.jsonl',
            }
            values = {}
            for name, source in paths.items():
                with sftp.open(source, 'rb') as f: values[name] = f.read()
            digest = lambda b: hashlib.sha256(b).hexdigest()
            assert digest(values['deployed_controller.py']) == '31777dcd11f7d08579038a322a39874eb4b4244db3aba18fb78f01cecd4e5040'
            protocol = json.loads(values['protocol.json'])
            config = json.loads(values['actual_training_config.json'])
            theta = json.loads(values['ACTUAL_CONTROL_THETA0.json'])
            pre = json.loads(values['CPU_PREFLIGHT.json'])
            assert config == json.loads(values['expected_lambda0_config.json'])
            assert protocol['controller_sha256'] == digest(values['deployed_controller.py'])
            assert protocol['expected_control_config_sha256'] == digest(values['expected_lambda0_config.json'])
            assert theta['config_sha256'] == digest(values['actual_training_config.json'])
            assert theta['state']['tensor_count'] == 327 and theta['optimizer_updates'] == 0
            assert theta['model_state_and_RNG_unchanged'] and not theta['actual_original_main_epoch0_state_verified']
            assert pre['CPU_rebuilt_main_and_lambda0_equal'] and not pre['matched_actual_historical_main_initialization']
            assert config['gaussian_weight'] == config['gaussian_schedule']['base'] == 0
            assert config['epochs'] == 200 and config['steps'] == 125 and config['source_sha256'] == 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
            assert theta['state']['state_sha256'] == 'd84df10851c2532e9b70311eba2efe07049f5348e56fe4e76daf7856e6427d79'
            OUT.mkdir()
            for name, value in values.items():
                with (OUT/name).open('xb') as f: f.write(value)
            receipt = {'role': 'ACTUAL_LAMBDA0_LAUNCH_AND_THETA0_NOT_COMPLETED_RESULTS',
                       'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'files': {n: {'remote': paths[n], 'bytes': len(b), 'sha256': digest(b)} for n, b in values.items()},
                       'actual_initial_model_matches_expected_CPU_rebuild': True,
                       'actual_original_main_initial_model_verified': False,
                       'remote_commands_or_writes': False, 'model_weight_snapshots_downloaded': False}
            (OUT/'capture_complete.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
            print(json.dumps({'captured': str(OUT), 'files': len(values), 'actual_theta0_state_sha256': theta['state']['state_sha256'], 'formal_training_complete': False}))
    finally:
        client.close()


if __name__ == '__main__': main()
