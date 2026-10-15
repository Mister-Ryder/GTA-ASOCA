from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent.parent/'fidelity_v2'
source=r/'code';target=r/'classifier_stage/code'
files=['pretrain_classifier.py','classification_data.py','checkpoint_utils.py','losses.py','data.py','historical/func/AnatomicalClassifier.py']
target.mkdir(parents=True,exist_ok=True)
rows={}
for relative in files:
 p=source/relative;t=target/relative;h=hashlib.sha256(p.read_bytes()).hexdigest()
 if t.exists() and hashlib.sha256(t.read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen stage already exists with different contents: '+relative)
 t.parent.mkdir(parents=True,exist_ok=True)
 if not t.exists():shutil.copy2(p,t)
 rows[relative]=h
(target.parent/'source_provenance.json').write_text(json.dumps({'source':'fidelity_v2/code','frozen_classifier_stage':True,'source_hashes':rows},indent=2)+'\n')
print(json.dumps({'files':len(rows),'path':str(target)}))
