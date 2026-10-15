"""Read existing evaluator summaries; no inference, hashes or metric recalculation."""
import argparse
import json
import sys
from pathlib import Path
from remote import connect

p=argparse.ArgumentParser();p.add_argument('--kind',choices=('U','B'),required=True)
p.add_argument('--prediction-cases',nargs='+');a=p.parse_args()
old='/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2'
new='/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014'
port,remote={
 'U':(17552,old+'/results/unet_recipe200x125/outer8_fixed_final_development_20261014'),
 'B':(38773,new+'/results/gta_region_balance200x125_20261014/outer8_fixed_final_development_20261015')
}[a.kind]
local=Path(__file__).resolve().parent.parent/'fidelity_v2/server_records'/('outer8_'+a.kind+'_report_20261015')
client=connect(port)
try:
 with client.open_sftp() as sftp:
  if a.prediction_cases:
   target=local/'predictions';target.mkdir(parents=True,exist_ok=True)
   for case in a.prediction_cases:
    assert case in ('Diseased_1','Normal_16')
    sftp.get(remote+'/'+case+'_prediction.nrrd',str(target/(case+'_prediction.nrrd')))
   print(json.dumps(dict(existing_prediction_files=a.prediction_cases,kind=a.kind,local=str(target),new_inference=False)))
   sys.exit(0)
  with sftp.open(remote+'/summary.json','rb') as f:summary_bytes=f.read()
  with sftp.open(remote+'/case_metrics.jsonl','rb') as f:rows_bytes=f.read()
 summary=json.loads(summary_bytes);rows=[json.loads(x)for x in rows_bytes.splitlines()if x.strip()]
 assert summary['overall']['n_cases']==len(rows)==8
 assert summary['checkpoint_epoch']==199 and summary['evaluation_split']=='outer_development'
 assert summary['case_ids']==[x['case_id']for x in rows]
 local.mkdir(parents=True,exist_ok=True)
 (local/'summary.json').write_bytes(summary_bytes);(local/'case_metrics.jsonl').write_bytes(rows_bytes)
 print(json.dumps(dict(kind=a.kind,role=summary['evaluation_role'],overall={k:summary['overall'][k]for k in ('dice','assd_mm','empty_predictions')},
  per_case=[{k:r[k]for k in ('case_id','dice','assd_mm')}for r in rows],local=str(local)),indent=2))
finally:client.close()
