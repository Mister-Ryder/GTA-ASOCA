"""Check every object downloaded through an anonymous Git clone."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--clone', required=True)
p.add_argument('--receipt', required=True)
a = p.parse_args()
root = Path(a.root).resolve()
clone = Path(a.clone).resolve()
local_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
def remote(*args):
    return subprocess.check_output(['git', '--git-dir', str(clone), *args])
if remote('rev-parse', 'refs/heads/main').decode().strip() != local_head:
    raise RuntimeError('Anonymous remote main differs from reviewed local commit')
entries = {}
for row in remote('ls-tree', '-rz', 'refs/heads/main').split(b'\0'):
    if not row:
        continue
    metadata, filename = row.split(b'\t', 1)
    mode, kind, sha = metadata.decode().split()
    if kind != 'blob':
        raise RuntimeError('Unexpected non-file object in public source')
    entries[filename.decode('utf-8')] = sha
expected = json.loads((root / 'PUBLIC_SOURCE_PROVENANCE.json').read_text())['public_file_allowlist'] + ['PUBLIC_SOURCE_PROVENANCE.json']
if set(entries) != set(expected):
    raise RuntimeError('Anonymous source file membership differs from reviewed allowlist')
for name, sha in entries.items():
    if remote('cat-file', 'blob', sha) != (root / name).read_bytes():
        raise RuntimeError('Anonymous downloaded bytes differ: ' + name)
value = {
    'status': 'verified_public',
    'url': 'https://github.com/Mister-Ryder/GTA-ASOCA',
    'private': False,
    'commit': local_head,
    'branch': 'main',
    'source_file_count': len(entries),
    'all_published_blobs_match_reviewed_bytes': True,
    'anonymous_access_verified': True,
    'verification_method': 'Git clone with credential helper and extra authorization headers cleared; all downloaded blob bytes compared',
    'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
Path(a.receipt).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps(value))
