"""Read-only collection of completed CPU scale audit; no raw arrays transferred."""
import datetime
import hashlib
import json
from pathlib import Path

from remote import connect

remote_root = '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014/results/label_scale_audit_v1'
output = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records/label_scale_audit_20261014'
if output.exists() and any(output.iterdir()):
    raise RuntimeError('Use a fresh evidence directory; no overwrite')
output.mkdir(parents=True, exist_ok=True)
names = ('audit_plan.json', 'fixed_sampling_bank.json', 'training_tree_extents.json',
         'roi_patch_metrics.jsonl', 'roi_patch_metrics.csv', 'summary.json', 'LABEL_SCALE_AUDIT_COMPLETE.json')
client = connect(38773)
records = []
try:
    with client.open_sftp() as sftp:
        for name in names:
            path = output / name
            sftp.get(remote_root + '/' + name, str(path))
            records.append({'name': name, 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    marker = json.loads((output / names[-1]).read_text(encoding='utf-8'))
    assert marker['status'] == 'completed' and marker['input_patches'] == 56 and marker['roi_observations'] == 168
    for name, expected in marker['artifacts_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    reference = output.parent / 'classifier_scope_20261014/results/gaussian_epoch070_cpu_bank56/fixed_sampling_bank.json'
    assert json.loads((output / 'fixed_sampling_bank.json').read_text(encoding='utf-8')) == json.loads(reference.read_text(encoding='utf-8'))
    record = {'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_root': remote_root,
              'records': records, 'complete_artifact_hashes_verified': True, 'actual_Gaussian56_bank_identical': True,
              'read_only_server_access': True}
    (output / 'collection_manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'files': len(records), 'roi_observations': 168, 'bank_identical': True}))
finally:
    client.close()
