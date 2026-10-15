"""CPU-only thin acceptance of actual completed Gaussian controls and final4 pairs.

Reuses the collector's completion gate through a strictly LOCAL read adapter,
and existing inspect_run/primary_from_pixels. No SSH, Torch/model or inference.
No control receipt is created until all actual evidence and pixels pass.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import datetime
import json
from pathlib import Path, PurePosixPath
import sys
sys.dont_write_bytecode=True
import compare_fixed_finals as c
import compare_preview_fixed_final as p
import verify_canonical_and_preview as canonical
from verify_postfit import finite, require, same, summary_of

ROOT=Path(__file__).resolve().parents[1]
RECORDS=ROOT/'server_records'
OUTER_OPS=ROOT.parent/'ops'
COLLECTOR_SHA='ec612fd2e1ca017dbca81cbbafa48df136617d91a1d9c7147aa96db1ef24735f'
PINS={ROOT/'ops/compare_fixed_finals.py':'857c6f90b6e5283547c7d88390792897176fa37a06589f8cb16e412b42abe2c8',
      ROOT/'ops/compare_preview_fixed_final.py':'d4f5c101eccc8d0c47d13b1f492645d491e72f8b7fc4507916de071b3d1d460d',
      ROOT/'ops/verify_canonical_and_preview.py':'7f75c7abc9bd3bfda0ceaca5bf2de8539890fe4d7cb3502906a540b3d49ac026',
      ROOT/'ops/verify_postfit.py':'698b349799cff927c9c8be2bf6330e0b2cbf4c1b9198ed8f4a303c6075b78e51',
      OUTER_OPS/'collect_completed_gaussian_control.py':COLLECTOR_SHA,
      OUTER_OPS/'collect_gta_completed_trajectory.py':'0b1f86a83e75aa139597365afb197c5990c4a591c336790e3de75cf4273854de'}
DEFAULTS={'lambda0':'gaussian_lambda0_complete_trajectory_20261014','B':'gaussian_region_balance_complete_trajectory_20261014'}


def local_file(folder, relative):
    pure=PurePosixPath(relative)
    require(not pure.is_absolute() and '..' not in pure.parts, 'Unsafe captured relative path')
    path=folder.joinpath(*pure.parts).resolve()
    require(path.is_relative_to(folder) and path.is_file(), 'Captured evidence missing/escaped: '+relative)
    return path


def capture_bytes(folder, expected_status):
    manifest_path=folder/'collection_manifest.json'; manifest=c.read(manifest_path)
    require(manifest.get('status')==expected_status and isinstance(manifest.get('files'),list), 'Incomplete capture cannot be accepted')
    paths={}; remote={}
    for record in manifest['files']:
        relative=record['relative_path']; path=local_file(folder,relative)
        require(relative not in paths and path.stat().st_size==record['bytes'] and c.digest(path)==record['sha256'], 'Captured bytes differ: '+relative)
        paths[relative]=dict(path=str(path),bytes=record['bytes'],sha256=record['sha256'])
        if 'remote_path' in record:
            require(record['remote_path'] not in remote, 'Duplicate remote identity in capture')
            remote[record['remote_path']]=path
    require(paths, 'Empty capture inventory')
    return manifest,paths,remote,dict(path=str(manifest_path),sha256=c.digest(manifest_path))


class LocalEvidence:
    """Implements only the read calls used by collector.gate; never networking."""
    def __init__(self,mapping):self.mapping=mapping
    def path(self,remote):
        if remote not in self.mapping:raise FileNotFoundError(2,'Required actual captured evidence missing',remote)
        return self.mapping[remote]
    def lstat(self,remote):return self.path(remote).stat()
    def open(self,remote,mode):
        require(mode=='rb','LOCAL adapter is read-only')
        return self.path(remote).open('rb')


def verify_control_trajectory(capture,evidence,run,context):
    directory=capture/'results'/PurePosixPath(evidence['train']).parent.name/'internal4_trajectory'
    identity=evidence['verified']['trajectory_identity']
    require(c.read(directory/'trajectory_protocol.json')==identity and c.read(directory/'trajectory_status.json')==
            {'status':'complete','completed':20,'planned':20}, 'Control full trajectory protocol/status differs')
    ledger=c.lines(directory/'validation_history.jsonl')
    require([r['name'] for r in ledger]==canonical.NAMES and c.digest(directory/'validation_history.jsonl')==
            evidence['verified']['validation_history_sha256'],'Control trajectory needs original20-point ledger bytes')
    receipts={(r['checkpoint'],r['case_id']):r for r in evidence['verified']['cases']}
    for point,record in zip(identity['checkpoints'],ledger):
        name=point['name']; output=directory/name
        remote_weight=str(PurePosixPath(evidence['train'])/('final.pt' if name=='final' else 'checkpoints/'+name+'.pt'))
        require(point['path']==remote_weight,'Control periodic checkpoint remote lineage differs')
        summary=context['trajectory_module'].verify_evaluation(output,c.IDS,point['checkpoint_sha256'],
            context['manifest_sha256'],run['config'],context['evaluation_sources'],context['cases'],point['epoch'])
        rows=[c.read(output/(i+'.json')) for i in c.IDS]
        require(c.lines(output/'case_metrics.jsonl')==rows,'Control patient ledger differs')
        for group,selected in [('overall',rows),('Normal',[r for r in rows if r['group']=='Normal']),('Diseased',[r for r in rows if r['group']=='Diseased'])]:
            require(same(summary.get(group),summary_of(selected)),'Control summary/null policy differs')
        done=c.read(output/'EVALUATION_COMPLETE.json')
        role='fixed final' if name=='final' else 'periodic internal-four diagnostic; not final selection'
        require(summary.get('checkpoint_selection')==role and finite(done.get('wall_seconds')) and done['wall_seconds']>=0,
                'Control summary role/evaluation duration differs')
        remote=str(PurePosixPath(evidence['trajectory'])/name)
        require(record==point|dict(evaluation_output=remote,summary_path=remote+'/summary.json',split='internal_diagnostic',
            role=summary['evaluation_role'],n_cases=4,overall=summary['overall'],seconds=done['wall_seconds'],smoke=False,
            checkpoint_selection=role,formal_final_test=False),'Control full20 ledger record differs')
        require(done.get('checkpoint_sha256')==point['checkpoint_sha256'] and done.get('case_ids')==c.IDS
            and done.get('manifest_sha256')==context['manifest_sha256'] and done.get('evaluation_sources')==context['evaluation_sources']
            and done.get('summary_sha256')==c.digest(output/'summary.json'),'Control actual evaluation completion differs')
        for row in rows:
            item=receipts[(name,row['case_id'])]
            require(item['case_json_sha256']==c.digest(output/(row['case_id']+'.json')) and item['prediction_sha256']==row['prediction_sha256']
                and row['checkpoint_selection']==role and row['formal_final_test'] is False,'Control80-case receipt/role differs')
        required={'summary.json','case_metrics.jsonl','EVALUATION_COMPLETE.json'}|{i+s for i in c.IDS for s in ('.json','_prediction.nrrd')}
        require({x.name for x in output.iterdir()}==required,'Control requires original11 files per point')
    for item in evidence['verified']['curve_artifacts']:
        relative=str(PurePosixPath(item['path']).relative_to(PurePosixPath(canonical.REMOTE)))
        require(c.digest(local_file(capture,relative))==item['sha256'],'Actual curve bytes differ from independent postfit receipt')
    curve_path=local_file(capture,str(PurePosixPath(evidence['curves']).relative_to(PurePosixPath(canonical.REMOTE)))+'/curve_manifest.json')
    curve=c.read(curve_path)
    require(curve.get('development_only') is True and curve.get('formal_final_test') is False and curve.get('interpolated_validation') is False
        and curve.get('plotter_sha256')==c.digest(ROOT/'ops/plot_learning_curves.py') and len(curve.get('runs',[]))==1,
        'Control actual curve protocol differs')
    curve_run=curve['runs'][0]
    require(curve_run.get('n_recorded_epochs')==200 and curve_run.get('n_training_cases')==28
        and curve_run.get('selection_case_ids')==c.IDS,'Control curve scope differs')
    for item in curve_run.get('sources',[]):
        relative=str(PurePosixPath(item['path']).relative_to(PurePosixPath(canonical.REMOTE)))
        require(c.digest(local_file(capture,relative))==item['sha256'],'Actual curve source bytes differ')
    return dict(verified_checkpoint_evaluations=20,verified_case_evaluations=80,local_final_pixel_recalculations=4,
        local_periodic_checkpoint_tensor_bytes_checked=False,actual_server_periodic_tensor_audit_receipt_retained=True,
        actual_postfit_sha256=c.digest(capture/'controller/POSTFIT_VERIFIED.json'))


def verify(args):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    for path,expected in PINS.items():require(c.digest(path)==expected,'Reviewed helper changed: '+str(path))
    sys.path.insert(0,str(OUTER_OPS))
    collector=c.load_module('completed_control_local_gate',OUTER_OPS/'collect_completed_gaussian_control.py')
    capture=args.capture.resolve(); raw=args.raw_root.resolve()
    receipt,inventory,remote,receipt_identity=capture_bytes(capture,'captured_completed_control_pending_independent_CPU_acceptance')
    require(receipt.get('kind')==args.kind and receipt.get('port')==collector.SPECS[args.kind]['port']
        and receipt.get('collector_sha256')==COLLECTOR_SHA and receipt.get('actual_final_weight_bytes_downloaded') is True,
        'Actual control collection identity differs')
    require(c.digest(args.reference_main)==collector.CONFIG,'Actual A reference config changed')
    evidence,waiting=collector.gate(LocalEvidence(remote),args.kind,c.read(args.reference_main))
    require(evidence is not None,'Actual completed control gate failed: '+str(waiting))
    manifest=local_file(capture,'dataset/cache/manifest.json')
    guard,context=p.context_for(ROOT/'code',manifest,ROOT/'ops/run_early_fixed_final_eval.py')
    control_context=context; code=ROOT/'code'; sealed=None
    if args.kind=='B':
        variant=ROOT/'variants/gaussian_region_balance_v1'
        require(c.digest(variant/'ops/check_variant.py')=='dc785e52f628859613bbe34594ef3711621583cbd5eee17ef25910cabc0135da','Reviewed full-B checker changed')
        checker=c.load_module('completed_control_full_B_checker',variant/'ops/check_variant.py')
        protocol,inspection=checker.inspect()
        require(inspection['source_sha256']==collector.B_SOURCE and inspection['protocol_sha256']==collector.B_PROTOCOL
            and inspection['code_package_sha256']=='52493f28ee8760d7c7d940cd26db84f440ac6c35009e344810a0284b5360a4ff','Full sealed28 B source identity differs')
        require(c.digest(capture/'variants/gaussian_region_balance_v1/protocol.json')==inspection['protocol_sha256']
            and c.digest(capture/'variants/gaussian_region_balance_v1/VARIANT_READY.json')==c.digest(variant/'VARIANT_READY.json'),
            'Captured B protocol/READY differs from full local package')
        code=variant/'code'
        require(all(c.digest(code/n)==context['evaluation_sources'][n] for n in context['evaluation_sources']), 'B evaluation source differs')
        control_context=dict(context,source_identity=(inspection['source_sha256'],inspection['source_files_sha256']))
        training=capture/'results'/collector.SPECS[args.kind]['run']/'training'
        bound=checker.actual_config(protocol,inspection,args.reference_main,training/'config.json',evidence['train'])
        require(bound==c.read(capture/'controller/ACTUAL_B_CONFIG_BINDING.json'),'Local full-B binding does not match actual run')
        sealed=dict(inspection=inspection,actual_binding=bound)
    control_training=capture/'results'/collector.SPECS[args.kind]['run']/'training'
    control_eval=control_training.parent/'internal4_trajectory/final'
    control=c.inspect_run('gta',control_training,control_eval,manifest,raw,code,control_context)
    require(control['audit']['weights_bytes_checked'] and control['audit']['final_checkpoint_sha256']==evidence['training']['final_sha256'], 'Actual captured final weight bytes not checked')
    control_proof=verify_control_trajectory(capture,evidence,control,control_context)
    # Existing mature A and U checks are reused; labels stay outside config.method.
    acap=args.canonical.resolve(); ucap=args.unet.resolve()
    a_inventory=capture_bytes(acap,'captured_completed_canonical_pending_independent_CPU_acceptance')
    u_inventory=capture_bytes(ucap,'captured_completed_records_pending_independent_verification')
    a=c.inspect_run('gta',acap/canonical.TRAIN,acap/canonical.TRAJECTORY/'final',manifest,raw,ROOT/'code',context)
    a_proof=canonical.acceptance(acap,a,context,guard)
    ut=ucap/'results/unet_recipe200x125/training';ue=ut.parent/'internal4_trajectory/final'
    u=c.inspect_run('unet',ut,ue,manifest,raw,ROOT/'code',context)
    p.check_unet_complete(ut,ue,u,context,guard)
    runs={'A':a,args.kind:control,'U':u}
    require(len(runs)==3,'Run labels collided')
    for label,run in runs.items():
        require(run['config']['internal_split']==a['config']['internal_split'] and
            all(run['config'].get(k)==a['config'].get(k) for k in c.RECIPE_FIELDS),'Internal split/common recipe differs: '+label)
    by_case={label:{row['case_id']:row for row in run['rows']} for label,run in runs.items()}
    rows=[];pairs=[]
    for case_id in c.IDS:
        require(len({by_case[label][case_id]['label_sha256'] for label in runs})==1,'Paired GT source differs')
        for label in runs:
            row=by_case[label][case_id]
            rows.append(dict(run_label=label,case_id=case_id,group=row['group'],label_sha256=row['label_sha256'],
                             dice=row['dice'],assd_mm=row['assd_mm'],empty_prediction=row['empty_prediction']))
        for right in ('A','U'):
            left_row=by_case[args.kind][case_id];right_row=by_case[right][case_id]
            pairs.append(dict(left_run=args.kind,right_run=right,case_id=case_id,label_sha256=left_row['label_sha256'],
                left_dice=left_row['dice'],right_dice=right_row['dice'],dice_left_minus_right=left_row['dice']-right_row['dice'],
                left_assd_mm=left_row['assd_mm'],right_assd_mm=right_row['assd_mm'],
                assd_left_minus_right_mm=left_row['assd_mm']-right_row['assd_mm'] if finite(left_row['assd_mm']) and finite(right_row['assd_mm']) else None))
    # Recheck all evidence after pixel calculations; no source is rewritten.
    for label,run in runs.items():
        for item in run['audit']['inputs']:require(c.digest(item['path'])==item['sha256'],'Evidence changed during pixels: '+label)
    for directory,expected_status,original,original_receipt in [(capture,receipt['status'],inventory,receipt_identity),
            (acap,a_inventory[0]['status'],a_inventory[1],a_inventory[3]),(ucap,u_inventory[0]['status'],u_inventory[1],u_inventory[3])]:
        rechecked=capture_bytes(directory,expected_status)
        require(rechecked[1]==original and rechecked[3]==original_receipt,'Capture/collection manifest changed during acceptance')
    guard.source_guard(ROOT)
    if args.kind=='B':require(checker.inspect()[1]==inspection,'Full B source changed during acceptance')
    for path,expected in PINS.items():require(c.digest(path)==expected,'Reviewed helper changed during acceptance')
    require('torch' not in sys.modules,'Acceptance unexpectedly imported Torch')
    return dict(schema_version=1,status='verified_actual_completed_control20_80_and_fixed_final4',kind=args.kind,
        verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),role='four internal development patients; no independent test/significance',
        case_ids=c.IDS,run_labels=list(runs),checkpoint_selection='fixed final',formal_final_test=False,
        control_completion=control_proof,canonical_A_completion=a_proof,sealed_B_source=sealed,
        control_collection_manifest=receipt_identity,control_collection_bytes_checked=len(inventory),
        patient_statistics={label:{metric:c.complete_stats(run['rows'],metric) for metric in ('dice','assd_mm')} for label,run in runs.items()},
        per_case=rows,paired_case_differences=pairs,audits={label:run['audit'] for label,run in runs.items()},
        metric_policy='Dice and pooled physical-mm ASSD; undefined distances remain null; no complete-four mean when any undefined; patient sample SD, not fold/seed SD',
        attribution_boundary='A/U differ in initialization/pretraining/capacity/objectives; lambda0 OLD hardware/driver differs; B changes loss reduction. Historical A theta0 was unsaved. No module-causal or winner claim.',
        other_control_not_checked='B' if args.kind=='lambda0' else 'lambda0',
        model_forward_Torch_GPU_SSH_used=False,helper_sha256={str(k):v for k,v in PINS.items()},tool_sha256=c.digest(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('lambda0','B'),required=True)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--canonical',type=Path,default=RECORDS/'gta_complete_trajectory_20261014')
    parser.add_argument('--unet',type=Path,default=RECORDS/'unet_complete_trajectory_20261014')
    parser.add_argument('--reference-main',type=Path,default=RECORDS/'gta_training_complete_20261014/results/gta_enhanced200x125/training/config.json')
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();args.capture=args.capture or RECORDS/DEFAULTS[args.kind]
    args.output=args.output or RECORDS/(args.kind+'_fixed_final4_acceptance_20261014')
    if args.plan:
        print(json.dumps(dict(role='PLAN_ONLY_NO_ACCEPTANCE_CREATED',kind=args.kind,capture=str(args.capture),output=str(args.output),
            paired_labels=['A',args.kind,'U'],actual_completed_capture_required=True,fixed_final_pixel_recalculations_when_run=12,
            CPU_only=True,network_or_model_operations=False),indent=2));return
    require(args.raw_root is not None and not args.output.exists(),'Provide actual --raw-root and fresh output')
    result=verify(args)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for filename,rows in [('fixed_final_patient_metrics.csv',result['per_case']),('paired_case_differences.csv',result['paired_case_differences'])]:
        with (args.output/filename).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    notes=[result['status'],result['role'],result['metric_policy'],result['attribution_boundary'],
           'Other control not checked: '+result['other_control_not_checked']]
    for label,fields in result['patient_statistics'].items():
        for field,stats in fields.items():notes.append(f"{label} {field}: mean={stats['mean']}; patientSD={stats['sd']}; defined={stats['n_defined']}/4")
    (args.output/'acceptance_notes.txt').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],kind=args.kind,output=str(args.output))))


if __name__=='__main__':main()
