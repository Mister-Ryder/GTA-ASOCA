"""Read-only SFTP collection of completed diagnostic and current scope evidence."""
import datetime
import hashlib
import json
from pathlib import Path

from remote import connect

root = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
output = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records/classifier_scope_20261014'
output.mkdir(parents=True, exist_ok=True)
paths = [
    'CPU_GAUSSIAN70_STATUS.json', 'CPU_GAUSSIAN70_LIVE_CHECK.json',
    'LIVE_MAIN_CHECK_20261014.json', 'dataset/cache/manifest.json',
    'results/gaussian_epoch070_cpu_bank56/diagnostic_plan.json',
    'results/gaussian_epoch070_cpu_bank56/fixed_sampling_bank.json',
    'results/gaussian_epoch070_cpu_bank56/patch_metrics.jsonl',
    'results/gaussian_epoch070_cpu_bank56/summary.json',
    'results/gaussian_epoch070_cpu_bank56/DIAGNOSTIC_COMPLETE.json',
]
client = connect(38773)
records = []
try:
    with client.open_sftp() as sftp:
        for relative in paths:
            target = output / relative
            if target.exists():
                raise RuntimeError(f'Existing evidence is not overwritten: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(root + '/' + relative, str(target))
            data = target.read_bytes()
            records.append({'path': relative, 'bytes': len(data),
                            'sha256': hashlib.sha256(data).hexdigest()})
    result = {'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_root': root, 'records': records, 'read_only_server_access': True}
    (output / 'collection_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(records), 'output': str(output)}))
finally:
    client.close()
