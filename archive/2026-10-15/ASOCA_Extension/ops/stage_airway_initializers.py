"""Preserve verified historical weights for explicit, recorded component transfer."""
from pathlib import Path
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parent.parent / 'fidelity_v2'
DEST = ROOT / 'initialization'
SOURCES = [
    ('airway_backbone_epoch599.pkl',
     Path(r'E:\01-投稿备份\投稿备份2604-NeurIPS 2026\V100X1-ICME-GTAirway-main-BAS\V100X1-ICME-GTAirway-main-BAS\checkpoint\BAS_model10-200epoch_-ronghe-xiuzheng_599.pkl'),
     '29b6977b74e547735717fda0785a9d532397df364e1f641bed0e6682db0384d3'),
    ('airway_classifier_improved_epoch100.pkl',
     Path(r'E:\01-投稿备份\投稿备份2506-AAAI 2026\30417\model_attention_vision\checkpoint\airway_anatomical_classifier_improved_epoch100.pkl'),
     'cc3ba8b4c82361a0ce18751ee94879caf2cd9e00f69a2c3da542c8641c09ab87'),
    ('airway_gta_epoch350.pkl',
     Path(r'E:\01-投稿备份\投稿备份2506-AAAI 2026\30417\model_attention_vision\checkpoint\154913gau+_enhanced_airway_segmentation_epoch350.pkl'),
     '93f3184fe81366fe38978928bc42705b9e314862bd333461a4b61a6f277a1711'),
]

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

DEST.mkdir(parents=True, exist_ok=True)
rows = []
for name, source, expected in SOURCES:
    if sha(source) != expected:
        raise RuntimeError('Historical source differs: '+str(source))
    target = DEST/name
    if target.exists() and sha(target) != expected:
        raise RuntimeError('Target exists with different contents: '+str(target))
    if not target.exists():
        shutil.copy2(source, target)
    rows.append({'file':name, 'source':str(source),'sha256':expected,'bytes':target.stat().st_size,
                 'training_domain':'pulmonary airway; coronary semantic heads require separate audit/adaptation',
                 'status':'candidate exists; tensor compatibility and original best-run identity not yet established'})
(DEST/'initialization_sources.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'staged':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'folder':str(DEST)},ensure_ascii=False))
