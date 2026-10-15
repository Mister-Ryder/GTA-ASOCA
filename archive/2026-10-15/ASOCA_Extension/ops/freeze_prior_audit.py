from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent.parent/'fidelity_v2'
source=r/'code';target=r/'prior_audit/code';target.mkdir(parents=True,exist_ok=True)
rows={}
for relative in ['audit_priors.py','data.py']:
 p=source/relative;t=target/relative;h=hashlib.sha256(p.read_bytes()).hexdigest()
 if t.exists() and hashlib.sha256(t.read_bytes()).hexdigest()!=h:raise RuntimeError('Prior audit snapshot already differs')
 if not t.exists():shutil.copy2(p,t)
 rows[relative]=h
(target.parent/'source_provenance.json').write_text(json.dumps({'source_hashes':rows,'role':'train32 prior diagnostics only'},indent=2)+'\n')
print(json.dumps({'prior_audit_frozen':True,'files':len(rows)}))
