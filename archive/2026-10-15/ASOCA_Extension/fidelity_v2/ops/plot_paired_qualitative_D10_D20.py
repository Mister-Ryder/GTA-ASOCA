"""Private CPU-only original-mask projections for two accepted development cases.

No inference, smoothing, resampling, component filtering or metric selection.
The same full union-of-GT/U/A crop preserves every foreground voxel per case.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ[_name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import compare_fixed_finals as c

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_records'
CASES=('Diseased_10','Diseased_20')
COMPARE_SHA='857c6f90b6e5283547c7d88390792897176fa37a06589f8cb16e412b42abe2c8'
FINAL_A='2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'
FINAL_U='a730d5747f3f0f05abf35c4b80578a62ced33ae904221411f6265c7a2c3b875c'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=BASE/'paired_qualitative_D10_D20_20261014')
    args=parser.parse_args();raw=args.raw_root.resolve();out=args.output.resolve()
    c.require(not out.exists(),'Fresh private output required')
    c.require(c.digest(ROOT/'ops/compare_fixed_finals.py')==COMPARE_SHA,'Reviewed image/geometry reader changed')
    import numpy as np
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    manifest_path=BASE/'gta_fixed_final_preview_capture_20261014/mirror_view/manifest.json'
    pair_path=BASE/'gta_canonical_acceptance_20261014/canonical_acceptance_and_pair.json'
    manifest=c.read(manifest_path);pair=c.read(pair_path)
    c.require(c.digest(manifest_path)==c.EXPECTED_MANIFEST and pair.get('status')=='verified_real_canonical20_80_and_fixed_final_pair'
        and pair.get('checkpoint_selection')=='fixed final' and pair.get('formal_final_test') is False,'Actual canonical pair/manifest required')
    c.require(pair.get('canonical_preview_all4_decoded_masks_equal') is True,'Canonical/preview agreement receipt changed')
    cases={x['case_id']:x for x in manifest['cases']}
    pair_rows={r['case_id']:r for r in pair['canonical_unet_per_case']}
    evaluations={'GT':None,
        'U':BASE/'unet_complete_trajectory_20261014/results/unet_recipe200x125/internal4_trajectory/final',
        'A':BASE/'gta_complete_trajectory_20261014/results/gta_enhanced200x125/internal4_trajectory/final'}
    inputs={str(path):dict(bytes=path.stat().st_size,sha256=c.digest(path)) for path in (manifest_path,pair_path)}
    prepared=[]
    for case_id in CASES:
        case=cases[case_id];truth_path=(raw/case['label_relative_path']).resolve()
        c.require(truth_path.is_relative_to(raw),'GT outside explicit raw root')
        truth_sha=c.digest(truth_path)
        c.require(truth_sha==case['source_sha256']['label']==pair_rows[case_id]['label_sha256'],'Actual original GT source differs')
        reference=c.read_image(truth_path);grid=c.geometry(reference)
        c.check_geometry(grid,case['original_geometry'],case_id)
        c.require(np.allclose(np.asarray(grid['direction']).reshape(3,3),np.eye(3),rtol=0,atol=1e-10),
            'This no-resampling plot requires identity LPS direction; stop rather than relabel unknown orientation')
        arrays={'GT':sitk.GetArrayFromImage(reference)};del reference
        inputs[str(truth_path)]=dict(bytes=truth_path.stat().st_size,sha256=truth_sha)
        metrics={};audit={}
        for label in ('U','A'):
            directory=evaluations[label];row_path=directory/(case_id+'.json');prediction=directory/(case_id+'_prediction.nrrd')
            row=c.read(row_path);name='gta_canonical' if label=='A' else 'unet'
            proof=next(x for x in pair['audits'][name]['pixel_audits'] if x['case_id']==case_id)
            c.require(row['case_id']==case_id and row['label_sha256']==truth_sha and row['checkpoint_epoch']==199
                and row['checkpoint_selection']=='fixed final' and row['formal_final_test'] is False
                and row['manifest_sha256']==c.EXPECTED_MANIFEST and row['training_source_sha256']==c.EXPECTED_SOURCE
                and row['checkpoint_sha256']==(FINAL_A if label=='A' else FINAL_U)
                and row['prediction_sha256']==proof['prediction_sha256']==c.digest(prediction),'Accepted fixed-final prediction identity differs')
            for field,prefix in [('dice','gta_dice' if label=='A' else 'unet_dice'),('assd_mm','gta_assd_mm' if label=='A' else 'unet_assd_mm')]:
                c.require(row[field]==pair_rows[case_id][prefix] and math.isclose(row[field],proof['recomputed_primary_metrics'][field],abs_tol=1e-6,rel_tol=1e-6),
                    'Previously verified full3D metric differs')
            c.check_geometry(proof['original_grid'],grid,case_id)
            image=c.read_image(prediction);c.check_geometry(c.geometry(image),grid,case_id)
            arrays[label]=sitk.GetArrayFromImage(image);del image
            metrics[label]={k:row[k] for k in ('dice','assd_mm')}
            for path in (row_path,prediction):inputs[str(path)]=dict(bytes=path.stat().st_size,sha256=c.digest(path))
            audit[label]=dict(prediction_path=str(prediction),prediction_sha256=row['prediction_sha256'],
                             metadata_path=str(row_path),metrics_source='already accepted complete-volume original-grid pixel calculation')
        for label,array in arrays.items():
            c.require(array.shape==arrays['GT'].shape and np.isin(array,(0,1)).all() and array.any(),'Nonbinary/empty/mismatched source mask: '+label)
        union=arrays['GT'].astype(bool)
        for label in ('U','A'):np.logical_or(union,arrays[label],out=union)
        coordinates=np.nonzero(union);spacing=np.asarray(grid['spacing_xyz_mm'])[::-1]
        padding=np.ceil(5./spacing).astype(int)
        low=np.maximum(0,[x.min() for x in coordinates]-padding)
        high=np.minimum(np.asarray(union.shape)-1,[x.max() for x in coordinates]+padding)
        crop=tuple(slice(int(a),int(b)+1) for a,b in zip(low,high))
        original_counts={label:int(array.sum()) for label,array in arrays.items()}
        projected={}
        for label,array in arrays.items():
            c.require(int(array[crop].sum())==original_counts[label],'Crop discarded foreground; stop')
            projected[label]={'coronal':array[crop].max(axis=1),'sagittal':array[crop].max(axis=2)}
        corners_low=np.asarray(grid['origin_xyz_mm'])+(low[::-1]-.5)*np.asarray(grid['spacing_xyz_mm'])
        corners_high=np.asarray(grid['origin_xyz_mm'])+(high[::-1]+.5)*np.asarray(grid['spacing_xyz_mm'])
        record=dict(case_id=case_id,original_grid=grid,label_sha256=truth_sha,models=audit,full3D_metrics=metrics,
            crop=dict(definition='all GT/U/A foreground union + at least5mm padding, clipped only at original volume boundary',
                low_zyx_inclusive=low.tolist(),high_zyx_inclusive=high.tolist(),world_outer_min_xyz_mm=corners_low.tolist(),
                world_outer_max_xyz_mm=corners_high.tolist(),foreground_counts_preserved=original_counts),
            projections={label:{view:dict(shape=list(array.shape),dtype=str(array.dtype),pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest())
                for view,array in views.items()} for label,views in projected.items()})
        prepared.append((record,projected,corners_low,corners_high))
        del arrays,union,coordinates
        print(json.dumps(dict(stage='actual_mask_bytes_geometry_and_crop_verified',case_id=case_id)),flush=True)
    # All actual source bytes/geometry pass before any diagnostic artifact exists.
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':300})
    colors={'GT':'#202020','U':'#D55E00','A':'#0072B2'};names={'GT':'Reference GT','U':'U: plain U-Net','A':'A: GTA'}
    for record,projections,low,high in prepared:
        fig,axes=plt.subplots(2,3,figsize=(10.5,8.0),layout='constrained')
        for column,label in enumerate(('GT','U','A')):
            for row,view in enumerate(('coronal','sagittal')):
                ax=axes[row,column]
                horizontal=0 if view=='coronal' else 1
                extent=(low[horizontal],high[horizontal],low[2],high[2])
                ax.imshow(projections[label][view],origin='lower',extent=extent,interpolation='nearest',aspect='equal',
                    cmap=ListedColormap(['white',colors[label]]),vmin=0,vmax=1)
                ax.set_xlabel('LPS x (mm; +L)' if view=='coronal' else 'LPS y (mm; +P)')
                ax.set_ylabel('LPS z (mm; +S)' if column==0 else '')
                title=names[label]
                if label!='GT':
                    m=record['full3D_metrics'][label];title+=f"\n3D Dice {100*m['dice']:.2f}% | ASSD {m['assd_mm']:.3f} mm"
                if row==1:title=('Sagittal: max over x' if column==0 else '')
                elif column==0:title+='\nCoronal: max over y'
                ax.set_title(title)
                ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:])
        fig.suptitle(record['case_id']+' | fixed-final binary-mask projections',fontsize=15)
        fig.supxlabel('Same full-foreground crop and physical scale across methods within each view. Projection can hide depth errors.\n'
            'Metrics are from accepted full 3D volumes; four-patient development study, no independent-test or causal claim.',fontsize=9)
        for suffix in ('png','svg'):fig.savefig(out/(record['case_id']+'_GT_U_A_projections.'+suffix))
        plt.close(fig)
    for path,record in inputs.items():c.require(Path(path).stat().st_size==record['bytes'] and c.digest(path)==record['sha256'],'Actual source changed while plotting')
    caption=('真实固定final预测的定性投影诊断，仅供本地研究，不上传公开仓库。\n'
        '两例预先由已验收四例配对结果指定：D10为最大Dice改善例，D20为最大Dice退化例；不是总体无偏样本。\n'
        '每例按GT、U-Net、GTA A三列展示相同原网格二值mask的冠状及矢状最大占据投影；未平滑、重采样、删连通域或修改预测。\n'
        '同一病例的裁剪框为三种mask全部前景的并集加5mm余量（只在原图边界截断）；没有排除远处FP。所有前景体素计数在crop前后完全保留。\n'
        '同一病例的同一投影视图内三列显示尺度相同；冠状和矢状两视图不强制使用相同屏幕缩放，应以各自mm坐标轴阅读。\n'
        '轴为原header的LPS物理坐标mm（已核identity方向矩阵）；x向左、y向后、z向上。二图不是CT强度MIP或重新推理。\n'
        '投影可使不同深度结构重叠、遮盖FP/FN或形成貌似连接，不能由2D外观推断三维连通、细分支完整性、临床病因或模块因果收益。\n'
        '图上Dice/pooled ASSD引用canonical A及已验收U的完整3D原网格像素核验值，不是二维新指标。\n'
        'A与U还存在预训练、初始化、容量、均衡和辅助目标差异；A不是由这两例指定的方法赢家。完整N4结果包括另两例。\n')
    (out/'figure_caption.txt').write_text(caption,encoding='utf-8')
    (out/'case_geometry_and_projection_audit.json').write_text(json.dumps([x[0] for x in prepared],indent=2,allow_nan=False)+'\n',encoding='utf-8')
    metadata=dict(schema_version=1,role='PRIVATE_ACTUAL_BINARY_MASK_PROJECTIONS_NOT_NEW_METRICS',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_script=str(Path(__file__).resolve()),source_script_sha256=c.digest(__file__),reader_sha256=COMPARE_SHA,
        case_ids=list(CASES),checkpoint_selection='fixed final',source_inputs=inputs,formal_final_test=False,
        visualization='binary maximum occupancy along original physical axes; nearest display; no smoothing/resampling/filtering',
        CPU_threads_limit=2,model_GPU_SSH_or_public_upload=False,artifacts={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in out.iterdir() if p.is_file()})
    (out/'figure_manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='private_actual_qualitative_masks_rendered_pending_visual_review',output=str(out),cases=list(CASES))))


if __name__=='__main__':main()
