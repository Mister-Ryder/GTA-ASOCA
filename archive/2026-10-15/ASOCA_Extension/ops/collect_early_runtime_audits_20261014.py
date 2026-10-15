"""Read-only two-server audit receipt download; credentials never persisted."""
import hashlib
import json
from pathlib import Path
from remote import connect

output = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records/early_fixed_final_preflight_20261014'
output.mkdir(parents=True, exist_ok=True)
records = []
for port, root, names in [
    (38773, '/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014', ['EARLY_FINAL_INPUT_RUNTIME_AUDIT_20261014.json']),
    (17552, '/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2', ['EARLY_FINAL_INPUT_RUNTIME_AUDIT_20261014.json', 'EARLY_FINAL_ENV_READY_20261014.json'])]:
    print('Read-only audit connection: ' + str(port), flush=True)
    client = connect(port)
    try:
        with client.open_sftp() as sftp:
            for name in names:
                target = output / (str(port) + '_' + name)
                if target.exists():
                    raise RuntimeError('Existing receipts are immutable: ' + str(target))
                sftp.get(root + '/' + name, str(target))
                records.append({'port': port, 'name': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    finally:
        client.close()
(output / 'collection_manifest.json').write_text(json.dumps({'files': records, 'read_only_server_access': True}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(output), 'files': len(records)}))
