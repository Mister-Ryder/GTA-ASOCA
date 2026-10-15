"""Four selected patches: frozen-model context and rotation sensitivity only.

No optimization, whole-case evaluation, Dice/ASSD or saved probability volumes.
"""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
SPECS={'U':dict(root='/root/autodl-tmp/ASOCA_GTAirway_20261013/fidelity_v2',python='/root/autodl-tmp/ASOCA_GTAirway_20261013/env/bin/python',
    code='code',run='unet_recipe200x125',method='unet',manifest='/root/autodl-tmp/ASOCA_GTAirway_20261013/cache/manifest.json'),
    'B':dict(root='/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014',python='/root/miniconda3/envs/gta-asoca311/bin/python',
    code='variants/gaussian_region_balance_v1/code',run='gta_region_balance200x125_20261014',method='gta',manifest='/root/autodl-tmp/GTA_ASOCA_Fidelity_20261014/dataset/cache/manifest.json')}
BASE=(128,128,128);CONTEXT=(128,160,160)


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def stop(signum,_frame):raise TimeoutError('Probe stopped by signal '+str(signum))


def run(args):
    import fcntl
    spec=SPECS[args.kind];root=Path(spec['root']);code=root/spec['code'];training=root/'results'/spec['run']/'training'
    output=root/'results/context_rotation_probe_20261015';protocol=read(args.protocol);samples=protocol['samples']
    need(sys.platform=='linux' and Path(__file__).resolve().parents[1]==root and sys.executable==spec['python'],'Wrong named host/environment')
    need(not output.exists() and len(samples)==4 and collections.Counter(x['case_id'] for x in samples)=={'Diseased_1':2,'Normal_16':2},'Require fresh output and exactly four specified patches')
    need(tuple(protocol.get('base_roi_zyx',BASE))==BASE and tuple(protocol.get('context_roi_zyx',CONTEXT))==CONTEXT,'Unexpected ROI protocol')
    manifest=read(spec['manifest']);cases={x['case_id']:x for x in manifest['cases']};config=read(training/'config.json');done=read(training/'TRAINING_COMPLETE.json')
    need(done['epoch']==199 and config['method']==spec['method'] and len(config['trained_case_ids'])==28 and not config['smoke'],'Require existing trained final200/28 model')
    lease=(root/'.segmentation_stage.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=None;torch=None;began=time.monotonic()
    try:
        idle=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip()
        need(not idle,'GPU is occupied; no model loaded');os.environ['CUDA_VISIBLE_DEVICES']='0'
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(300)
        import numpy as np
        import torch
        torch.set_num_threads(4);sys.path.insert(0,str(code))
        from data import extract_patch
        from train import make_model
        saved=torch.load(training/'final.pt',map_location='cpu',weights_only=True)
        need(saved['epoch']==199 and saved['config']==config,'Checkpoint header differs')
        model=make_model(config['method'],not config['no_ds']);model.load_state_dict(saved['model']);del saved
        model=model.cuda().eval();rows=[];cache={}
        def predict(image):
            x=torch.from_numpy(np.ascontiguousarray(image,dtype=np.float32))[None,None].cuda()
            with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=model(x);logits=out['logits'].float()
            attributes={name:int(out['classifier'][name].argmax(1)[0]) for name in ('generation','structure','location') if name in out.get('classifier',{})}
            p=logits.softmax(1)[0,1].cpu().numpy();hard=logits.argmax(1)[0].cpu().numpy()>0;del x,logits,out
            return p,hard,attributes
        for sample in samples:
            case_id=sample['case_id'];center=np.asarray(sample['center_cache_zyx'],dtype=np.int64)
            if case_id not in cache:
                folder=Path(spec['manifest']).parent/cases[case_id]['cache_subdir']
                cache[case_id]=(np.load(folder/'image.npy',mmap_mode='r'),np.load(folder/'label.npy',mmap_mode='r'))
            image,label=cache[case_id]
            need(center.shape==(3,) and np.all(center>=0) and np.all(center<image.shape),'Invalid cached voxel center')
            base_start=center-np.asarray(BASE)//2;context_start=center-np.asarray(CONTEXT)//2
            base=extract_patch(image,base_start,BASE,fill=-1,dtype=np.float32);truth=extract_patch(label,base_start,BASE,fill=0)>0
            context=extract_patch(image,context_start,CONTEXT,fill=-1,dtype=np.float32)
            offset=base_start-context_start;crop=tuple(slice(int(o),int(o+s)) for o,s in zip(offset,BASE))
            need(np.array_equal(context[crop],base),'Centered context crop alignment differs')
            base_result=predict(base);wide_result=predict(context);rot_result=predict(np.rot90(base,k=1,axes=(1,2)))
            probabilities={'base':base_result[:2],'context':tuple(x[crop] for x in wide_result[:2]),
                'rotation':tuple(np.rot90(x,k=-1,axes=(1,2)).copy() for x in rot_result[:2])}
            reference,hard=probabilities['base'];base_fp=hard&~truth;base_tp=hard&truth
            variants={}
            for name,(p,pred) in probabilities.items():
                need(p.shape==BASE and np.isfinite(p).all(),'Invalid output probabilities')
                selected=p[base_fp];delta=(p-reference)[base_fp]
                variants[name]=dict(TP=int((pred&truth).sum()),FP=int((pred&~truth).sum()),FN=int((~pred&truth).sum()),
                    on_base_FP=dict(count=int(base_fp.sum()),mean_confidence=float(selected.mean()) if selected.size else None,
                        fraction_confidence_ge_0_9=float((selected>=.9).mean()) if selected.size else None,
                        mean_probability_delta=float(delta.mean()) if delta.size else None,
                        suppressed_count=int((base_fp&~pred).sum())),
                    base_TP_retained=int((base_tp&pred).sum()),new_FN_from_base_TP=int((base_tp&~pred).sum()),
                    hard_disagreement_count=int((pred!=hard).sum()),hard_disagreement_fraction=float((pred!=hard).mean()),
                    mean_absolute_probability_difference=float(np.abs(p-reference).mean()))
                if name=='rotation':variants[name]['mean_absolute_probability_equivariance_error']=variants[name]['mean_absolute_probability_difference']
            rows.append(dict(sample=sample,base_start_cache_zyx=base_start.tolist(),context_start_cache_zyx=context_start.tolist(),
                context_to_base_crop_start_zyx=offset.tolist(),cached_GT_foreground=int(truth.sum()),variants=variants,
                classifier_argmax=dict(zip(('base','context','rotation'),(x[2] for x in (base_result,wide_result,rot_result))))))
            del probabilities,base_result,wide_result,rot_result,base,context,truth,reference,hard,base_fp,base_tp,p,pred,selected,delta
        result=dict(status='completed_frozen_model_patch_probe',kind=args.kind,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            seconds=time.monotonic()-began,sample_count=4,protocol=protocol,checkpoint=str(training/'final.pt'),checkpoint_epoch=199,
            base_roi_zyx=BASE,context_roi_zyx=CONTEXT,rotation='np.rot90 k=1 axes(Y,X), prediction inverse k=-1 before comparison',
            precision='Existing evaluator CUDA autocast float16; logits then softmax in float32',training=False,optimizer_updates=0,
            whole_case_metrics_computed=False,scope='Selected 128-cube patch sensitivity; context difference is not whole-case improvement and GT is used only for analysis.',rows=rows)
        output.mkdir(mode=0o700,parents=True);(output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(dict(status=result['status'],kind=args.kind,samples=4,seconds=result['seconds'],output=str(output/'results.json'))))
    finally:
        signal.alarm(0)
        if model is not None:del model
        if torch is not None:torch.cuda.empty_cache()
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('U','B'),required=True);p.add_argument('--protocol',type=Path);mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--run',action='store_true');mode.add_argument('--plan',action='store_true');args=p.parse_args()
    if args.plan:print(json.dumps(dict(status='PLAN_ONLY',kind=args.kind,spec=SPECS[args.kind],base_roi_zyx=BASE,context_roi_zyx=CONTEXT,max_seconds=300,training=False),indent=2));return
    need(args.protocol is not None,'Protocol is required');run(args)


if __name__=='__main__':main()
