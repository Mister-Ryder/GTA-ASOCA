"""Prepare/check one matched Gaussian lambda0 control; never start training.

The frozen GTA model retains its Gaussian module. No lambda10 rerun is planned.
CPU reconstruction can verify two proposed starts, but cannot recover the actual
historical main start: no epoch0 tensor snapshot was saved by the main trainer.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import sys

from diagnose_gaussian import sha, source_identity

EXPECTED_MANIFEST='141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
EXPECTED_SOURCE='bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
PATH_FIELDS=('manifest','backbone_init','classifier_init','full_init')


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def difference(left,right,prefix=''):
    if isinstance(left,dict) and isinstance(right,dict):
        result=[]
        for key in sorted(set(left)|set(right)):
            name=prefix+'.'+key if prefix else key
            if key not in left or key not in right: result.append({'field':name,'main':left.get(key),'candidate':right.get(key),'missing_key':True})
            else: result+=difference(left[key],right[key],name)
        return result
    if isinstance(left,list) and isinstance(right,list) and len(left)==len(right):
        return [row for i,(a,b) in enumerate(zip(left,right)) for row in difference(a,b,prefix+f'[{i}]')]
    return [] if canonical(left)==canonical(right) else [{'field':prefix,'main':left,'candidate':right}]


def rebased(value,rebase):
    if value is None: return None
    path=PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts: raise ValueError('Expected safe absolute recorded Linux path: '+str(value))
    for old,new in sorted(rebase,key=lambda item:len(item[0].parts),reverse=True):
        try: suffix=path.relative_to(old)
        except ValueError: continue
        return str(new/suffix)
    return value


def expected_candidate(main,candidate_output,rebase=()):
    output=PurePosixPath(candidate_output)
    if not output.is_absolute() or '..' in output.parts or output.name!='training':
        raise ValueError('Proposed output must be a safe absolute Linux training directory')
    if str(output)==rebased(main['output'],rebase): raise ValueError('Candidate must not overwrite the main training output')
    result=copy.deepcopy(main)
    result['gaussian_weight']=0.; result['gaussian_schedule']['base']=0.; result['output']=str(output)
    for key in PATH_FIELDS: result[key]=rebased(main[key],rebase)
    for audit in result['init_audits']: audit['path']=rebased(audit['path'],rebase)
    return result


def check_candidate(main,candidate,candidate_output,rebase=()):
    if main.get('gaussian_weight')!=10 or main.get('gaussian_late_weight') is not None:
        raise ValueError('Original main must use constant Gaussian lambda10, with no late schedule')
    if (main.get('method')!='gta' or main.get('no_ds') is not False or main.get('resume') is not False
        or main.get('smoke') is not False or main.get('plan') is not False or main.get('full_init') is not None):
        raise ValueError('Requires the actual fresh GTA/DS formal recipe with separate backbone/classifier donors')
    schedule=main.get('gaussian_schedule',{})
    if (schedule.get('base')!=10 or schedule.get('late') is not None
        or schedule.get('start_epoch')!=main['gaussian_ramp_start'] or schedule.get('end_epoch')!=main['gaussian_ramp_end']):
        raise ValueError('Original declared schedule disagrees with its actual Gaussian arguments')
    expected=expected_candidate(main,candidate_output,rebase)
    if canonical(candidate)!=canonical(expected):
        raise ValueError('Unexpected lambda0 configuration differences: '+canonical(difference(expected,candidate)))
    if candidate['gaussian_late_weight'] is not None or candidate['gaussian_schedule']['late'] is not None:
        raise ValueError('Lambda0 must not have an extra Gaussian schedule')
    return difference(main,candidate)


def cli_argv(config,code,python,remote_code):
    """Recover only the real frozen train CLI fields, preserving every argument."""
    fields=[]
    for node in ast.walk(ast.parse((code/'train.py').read_text(encoding='utf-8'))):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='p'):
            option=ast.literal_eval(node.args[0])
            if not option.startswith('--'): continue
            fields.append((node.lineno,option,option[2:].replace('-','_')))
    argv=[python,'-u',str(remote_code/'train.py')]
    for _,option,key in sorted(fields):
        if key not in config: raise ValueError('Actual saved configuration lacks frozen CLI field: '+key)
        value=config[key]
        if isinstance(value,bool):
            if value: argv.append(option)
        elif value is not None: argv.extend([option,str(value)])
    return argv


def state_identity(model):
    import torch
    h=hashlib.sha256(); tensors={}
    for name,value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()): raise ValueError('Invalid model tensor: '+name)
        value=value.detach().cpu().contiguous()
        raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
        header={'name':name,'dtype':str(value.dtype),'shape':list(value.shape)}
        metadata=canonical(header).encode(); h.update(len(metadata).to_bytes(8,'little')); h.update(metadata)
        h.update(len(raw).to_bytes(8,'little')); h.update(raw)
        tensors[name]={**header,'content_sha256':hashlib.sha256(raw).hexdigest()}
    return {'algorithm':'sorted_tensor_name_dtype_shape_and_contiguous_raw_bytes_v1','state_sha256':h.hexdigest(),
            'tensor_count':len(tensors),'per_tensor':tensors,'parameter_count':sum(p.numel() for p in model.parameters())}


def reconstruct(config,donors,code):
    """Mirror train.py factory then backbone/classifier transfer, entirely on CPU."""
    import numpy as np
    import torch
    from checkpoint_utils import load_audited,read_checkpoint
    from train import make_model
    if str(torch.__version__)!=config['torch']:
        raise ValueError('CPU reconstruction needs the recorded Torch build; no cross-build byte-equivalence claim')
    if torch.cuda.is_initialized(): raise ValueError('CPU checker requires a process without initialized CUDA context')
    seed=config['seed']+config['fold']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(8)
    model=make_model(config['method'],not config['no_ds'])
    if not getattr(model,'use_gaussian',False) or not getattr(model,'use_prior',False):
        raise ValueError('Gaussian/prior modules must remain in both matched architectures')
    audits=[]
    for component,module,domain,policy in [('backbone',model.base_model,config['backbone_domain'],'reinit'),
                                          ('classifier',model.classifier,config['classifier_domain'],config['classifier_head_policy'])]:
        recorded=next(a for a in config['init_audits'] if a['component']==component)
        path=donors[component]
        if sha(path)!=recorded['sha256']: raise ValueError('Donor bytes differ: '+component)
        donor,_,_=read_checkpoint(path)
        if domain=='asoca':
            cfg=donor.get('config',{}); ids=cfg.get('trained_case_ids',cfg.get('internal_split',{}).get('train'))
            if not ids or not set(ids).issubset(config['trained_case_ids']): raise ValueError('ASOCA donor leaks or lacks training28 provenance')
            if (cfg.get('manifest_sha256')!=config['manifest_sha256'] or cfg.get('smoke') is not False
                or cfg.get('pretrained_classifier') is not True or policy!='load'
                or cfg.get('additional_pretraining_updates')!=config['classifier_pretraining_updates']):
                raise ValueError('Classifier80 fixed donor pretraining identity differs')
            weights={k:cfg['attribute_statistics']['heads'][k]['focal_weights'] for k in ('generation','structure','location')}
            if weights!=config['attribute_weights']: raise ValueError('Classifier focal weights differ from actual main config')
        audit=load_audited(module,path,component,domain,policy)
        if canonical({k:v for k,v in audit.items() if k!='path'})!=canonical({k:v for k,v in recorded.items() if k!='path'}):
            raise ValueError('Reconstructed donor transfer differs from original audit: '+component)
        audits.append(audit)
    model.to(torch.device('cpu'))
    # These operations are the recorded first-epoch stage setting; no forward.
    model.set_classifier_stage('frozen'); model.train()
    identity=state_identity(model)
    if identity['parameter_count']!=config['parameters']: raise ValueError('Parameter capacity differs from main')
    identity.update(torch=str(torch.__version__),numpy=np.__version__,effective_seed=seed,
                    torch_cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    classifier_stage='frozen',donor_audits=audits,CUDA_context_initialized=torch.cuda.is_initialized())
    if identity['CUDA_context_initialized']: raise RuntimeError('CPU reconstruction unexpectedly initialized CUDA')
    del model
    return identity


def main():
    root=Path(__file__).resolve().parents[1]; code=root/'code'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-config',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--candidate-output',required=True); p.add_argument('--proposal-output',type=Path,required=True)
    p.add_argument('--candidate-config',type=Path); p.add_argument('--rebase',action='append',default=[],metavar='OLD=NEW')
    p.add_argument('--python',default='/root/miniconda3/envs/gta-asoca311/bin/python')
    p.add_argument('--cpu-check',action='store_true',help='Explicit donor-byte/model reconstruction check; no optimizer/forward')
    p.add_argument('--backbone-donor',type=Path); p.add_argument('--classifier-donor',type=Path)
    args=p.parse_args(); rebase=[]
    for value in args.rebase:
        old,new=value.split('=',1); pair=(PurePosixPath(old),PurePosixPath(new))
        if any(not x.is_absolute() or '..' in x.parts for x in pair): p.error('Rebase prefixes must be safe absolute Linux paths')
        rebase.append(pair)
    if len({str(a) for a,b in rebase})!=len(rebase): p.error('Duplicate rebase prefix')
    main_config=json.loads(args.main_config.read_text(encoding='utf-8')); manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    source,files=source_identity(code)
    if (source!=EXPECTED_SOURCE or main_config.get('source_sha256')!=source or main_config.get('source_files_sha256')!=files
        or sha(args.manifest)!=EXPECTED_MANIFEST or main_config.get('manifest_sha256')!=EXPECTED_MANIFEST):
        raise ValueError('Actual main source/manifest identities differ from frozen fidelity experiment')
    required={'method':'gta','fold':0,'epochs':200,'steps':125,'patch_size':128,'batch_size':2,'no_ds':False,
        'sampling':'standard','sampling_fov':'image-body','cohort':'internal28','backbone_warmup_epochs':0,
        'freeze_epochs':50,'partial_until':70,'classifier_weight':.1,'guidance_weight':.1,
        'classifier_pretraining_updates':5120,'segmentation_updates':25000,'pretrained_prior':True,
        'internal_selection_valid':True,'outer8_used_for_selection':False,'allow_selection_overlap':False,
        'backbone_domain':'airway','classifier_domain':'asoca','classifier_head_policy':'load','device':'cuda',
        'source_hash_algorithm':'gta_fidelity_selected_python_path_and_bytes_v1'}
    if any(main_config.get(k)!=v for k,v in required.items()): raise ValueError('Actual main is not the intended fixed25000-update/classifier80/staged recipe')
    split=main_config['internal_split']; train_ids=split['train']; val_ids=split['val']
    if (len(set(train_ids))!=28 or len(train_ids)!=28 or len(set(val_ids))!=4 or len(val_ids)!=4
        or set(train_ids)&set(val_ids) or set(train_ids)|set(val_ids)!=set(manifest['folds'][0]['train'])
        or set(train_ids+val_ids)&set(manifest['folds'][0]['val']) or main_config['trained_case_ids']!=train_ids):
        raise ValueError('Main training28/internal4 partition leaks or differs')
    if [a['component'] for a in main_config['init_audits']]!=['backbone','classifier']:
        raise ValueError('Expected only ordered backbone and classifier donors')
    candidate=expected_candidate(main_config,args.candidate_output,rebase)
    if args.candidate_config: candidate=json.loads(args.candidate_config.read_text(encoding='utf-8'))
    changes=check_candidate(main_config,candidate,args.candidate_output,rebase)
    proposal=args.proposal_output.resolve()
    if proposal.exists() and any(proposal.iterdir()): raise ValueError('Use a fresh proposal output; never overwrite an active run')
    proposal.mkdir(parents=True,exist_ok=True)
    inputs={args.main_config.resolve():sha(args.main_config),args.manifest.resolve():sha(args.manifest),
            Path(__file__).resolve():sha(__file__),root/'ops/diagnose_gaussian.py':sha(root/'ops/diagnose_gaussian.py')}
    if args.candidate_config: inputs[args.candidate_config.resolve()]=sha(args.candidate_config)
    donors={'backbone':args.backbone_donor or Path(candidate['backbone_init']),
            'classifier':args.classifier_donor or Path(candidate['classifier_init'])}
    donor_records={}
    for component,path in donors.items():
        audit=next(a for a in main_config['init_audits'] if a['component']==component)
        exists=path.is_file(); actual=sha(path) if exists else None
        if exists and actual!=audit['sha256']: raise ValueError('Donor file differs from original recorded hash: '+component)
        donor_records[component]={'explicit_local_path':str(path.resolve()),'recorded_main_path':audit['path'],
            'expected_sha256':audit['sha256'],'bytes_checked':exists,'actual_sha256':actual}
        if exists: inputs[path.resolve()]=actual
    report={'schema_version':1,'status':'prepared_not_launched','source_sha256':source,'manifest_sha256':EXPECTED_MANIFEST,
        'scope':'one lambda0 matched control versus existing lambda10 main; no lambda10 rerun/fivefold',
        'main_output':main_config['output'],'candidate_output':candidate['output'],'allowed_actual_differences':changes,
        'constant_gaussian_coefficients':{'existing_main':10.,'candidate':0.,'extra_schedule':False},
        'method':'gta','Gaussian_module_retained':True,'same_25000_segmentation_updates':True,'same_classifier_pretraining_updates':5120,
        'same_internal28_4_split':True,'same_declared_DS_sampler_donors_stages_seeds':True,'donors':donor_records,
        'actual_original_main_epoch0_state_verified':False,
        'historical_initialization_evidence':'Main saved no pre-update epoch0 tensor snapshot. Seed/source/donor agreement supports reconstruction but cannot certify actual historical random tensors byte-for-byte.',
        'cpu_reconstruction_performed':False,'network_forward':False,'optimizer_created':False,'GPU_used':False,
        'launch_decision':'deferred; review existing fixed-final internal4 Dice/ASSD before deciding on one lambda0 run',
        'classification_accuracy_gate':False}
    if args.cpu_check:
        if not all(x['bytes_checked'] for x in donor_records.values()): raise FileNotFoundError('Both actual donor files are required for CPU check')
        if os.name=='posix': os.nice(10)
        sys.dont_write_bytecode=True; sys.path.insert(0,str(code))
        original=reconstruct(main_config,donors,code); control=reconstruct(candidate,donors,code)
        same_state=original['per_tensor']==control['per_tensor'] and original['state_sha256']==control['state_sha256']
        if not same_state or original['torch_cpu_rng_sha256']!=control['torch_cpu_rng_sha256']:
            raise ValueError('Two CPU reconstructed starts or post-initialization RNG states differ')
        write(proposal/'cpu_reconstructed_main_initial_state.json',original)
        write(proposal/'cpu_reconstructed_lambda0_initial_state.json',control)
        report.update(cpu_reconstruction_performed=True,CPU_rebuilt_main_and_lambda0_equal=True,
            cpu_reconstructed_state_sha256=original['state_sha256'],Torch_build=original['torch'],
            matched_actual_historical_main_initialization=False)
    if source_identity(code)!=(source,files) or any(sha(path)!=value for path,value in inputs.items()):
        raise RuntimeError('Source/main/manifest/donor inputs changed during preparation')
    write(proposal/'expected_lambda0_config.json',candidate)
    manifest_remote=PurePosixPath(candidate['manifest'])
    if manifest_remote.parts[-3:]!=('dataset','cache','manifest.json'):
        raise ValueError('Cannot infer this frozen run remote ROOT/code from the proposed manifest; use dataset/cache/manifest.json')
    argv=cli_argv(candidate,code,args.python,manifest_remote.parents[2]/'code')
    write(proposal/'proposed_train_argv_NOT_EXECUTED.json',{'argv':argv,'executed':False,
        'note':'Run only after a separate launch decision and resource/frozen-source/readiness checks; this helper never starts it.'})
    report['inputs']=[{'path':str(path),'sha256':value} for path,value in inputs.items()]
    write(proposal/'CONTROL_PREFLIGHT.json',report)
    print(json.dumps({'proposal':str(proposal),'status':report['status'],'cpu_reconstruction_performed':report['cpu_reconstruction_performed'],
        'actual_original_main_epoch0_state_verified':False,'training_started':False},allow_nan=False))


if __name__=='__main__': main()
