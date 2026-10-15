"""Read-only SFTP copy of actual CPU-only control-preflight records."""
import datetime
import hashlib
import json
from pathlib import Path

from remote import connect

source = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014/results/gaussian_lambda0_cpu_preflight_20261014'
output = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records/gaussian_lambda0_cpu_preflight_20261014'
if output.exists() and any(output.iterdir()):
    raise RuntimeError('Evidence directory must be fresh')
output.mkdir(parents=True, exist_ok=True)
names = ('CONTROL_PREFLIGHT.json', 'expected_lambda0_config.json', 'proposed_train_argv_NOT_EXECUTED.json',
         'cpu_reconstructed_main_initial_state.json', 'cpu_reconstructed_lambda0_initial_state.json')
client = connect(38773)
records = []
try:
    with client.open_sftp() as sftp:
        for name in names:
            sftp.get(source + '/' + name, str(output / name))
            records.append({'name': name, 'bytes': (output / name).stat().st_size,
                            'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()})
    report = json.loads((output / names[0]).read_text(encoding='utf-8'))
    main = json.loads((output / names[3]).read_text(encoding='utf-8'))
    control = json.loads((output / names[4]).read_text(encoding='utf-8'))
    assert report['cpu_reconstruction_performed'] and report['CPU_rebuilt_main_and_lambda0_equal']
    assert report['actual_original_main_epoch0_state_verified'] is False
    assert not report['GPU_used'] and not report['network_forward'] and not report['optimizer_created']
    assert main['state_sha256'] == control['state_sha256'] == report['cpu_reconstructed_state_sha256']
    assert main['per_tensor'] == control['per_tensor'] and main['torch_cpu_rng_sha256'] == control['torch_cpu_rng_sha256']
    assert not main['CUDA_context_initialized'] and not control['CUDA_context_initialized']
    result = {'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_root': source,
              'files': records, 'recorded_reconstructed_tensor_identity_verified': True,
              'actual_historical_main_initial_state_verified': False, 'training_started': False}
    (output / 'collection_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'files': len(records), 'state_sha256': main['state_sha256'],
                      'tensor_count': main['tensor_count'], 'historical_initial_state_verified': False, 'training_started': False}))
finally:
    client.close()
