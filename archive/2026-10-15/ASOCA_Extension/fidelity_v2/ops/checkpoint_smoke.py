"""Verify historical classifier transfer and actual 128-cubed CUDA gradients."""
import argparse,json,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
r=Path(a.root)
sys.path.insert(0,str(r/'classifier_stage/code'))
import torch
from checkpoint_utils import load_audited,sha256_file
from historical.func.AnatomicalClassifier import AirwayAnatomicalClassifier
from losses import focal_loss

sources=json.loads((r/'initialization/initialization_sources.json').read_text())
for source in sources:
 path=r/'initialization'/source['file']
 if sha256_file(path)!=source['sha256']:raise RuntimeError('Historical weight checksum differs: '+source['file'])
model=AirwayAnatomicalClassifier()
audit=load_audited(model,r/'initialization/airway_classifier_improved_epoch100.pkl','classifier','airway','reinit')
assert len(audit['excluded_semantic_keys'])==12
model.cuda().train();torch.manual_seed(20261014)
torch.cuda.reset_peak_memory_stats()
x=torch.randn(2,1,128,128,128,device='cuda')
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
scaler=torch.amp.GradScaler('cuda',init_scale=1024,growth_interval=1000000)
with torch.amp.autocast('cuda',dtype=torch.float16):
 out=model(x)
 loss=sum(focal_loss(out[name],torch.tensor([1,3],device='cuda')) for name in ['generation','structure','location'])+.1*out['guidance_map'].float().square().mean()
assert torch.isfinite(loss)
scaler.scale(loss).backward();scaler.unscale_(optimizer)
assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
assert all(model.state_dict()[key].shape for key in audit['loaded_keys'] if model.state_dict()[key].ndim)
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
scaler.step(optimizer);scaler.update()
result={'status':'passed','role':'synthetic mechanical smoke only; not prior-quality evidence','initialization_audit':audit,
        'loss':float(loss.detach()),'gradient_norm':float(norm),'patch_size':128,'batch_size':2,
        'gpu_peak_GiB':float(torch.cuda.max_memory_allocated()/2**30),'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda}
(r/'CHECKPOINT_CUDA_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='initialization_audit'}),flush=True)
