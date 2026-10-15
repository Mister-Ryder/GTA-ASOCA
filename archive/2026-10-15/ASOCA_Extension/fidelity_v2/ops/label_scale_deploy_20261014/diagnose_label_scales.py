"""Read-only training28 local-label scale audit; NumPy/CPU, no network forward.

Keep the same128^3 CT input and predeclare concentric64/96/128 voxel ROIs.
Global cached topology is cropped, never rerooted/reskeletonized in each ROI.
No validation/outer GT, label mutation, classifier update or accuracy gate.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from diagnose_gaussian import sha, source_identity, sample_identity, compare_bank

ROI_SIZES = (64, 96, 128)
EXPECTED_MANIFEST = '141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
EXPECTED_SOURCE = 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def array_identity(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return {'dtype': value.dtype.str, 'shape': list(value.shape), 'content_sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def entropy(counts):
    total = sum(counts.values())
    return -sum((n/total) * math.log2(n/total) for n in counts.values() if n) if total else None


def hist(values):
    return {str(int(k)): int(v) for k, v in sorted(Counter(int(x) for x in values).items())}


def mode(counts):
    return min(counts, key=lambda k: (-counts[k], int(k))) if counts else None


def stats(values):
    values = [float(x) for x in values if x is not None]
    return {'n_defined': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'observation_sd': statistics.stdev(values) if len(values)>1 else None}


def sample_weight(generation, structure, disabled=False):
    if disabled: return 1.
    return (1.5 if generation>=5 else 1.2 if generation>=3 else 1.) * (1.3 if structure==3 else 1.)


def complexity_structure(branch_count, volume):
    if branch_count<=1: return None, None
    value = math.log(branch_count+1)/math.log(10) / max(.1, math.log(volume+1)/math.log(10000))
    return 6 if value>1.5 else 5 if value>1 else 4 if value>.8 else 3, value


def roi_metrics(label, branches, depths, skeleton, radius, start, full_shape, patch_priors,
                generation_bin, parent_volume, parent_branches, parent_structure):
    import numpy as np
    foreground = label>0; assigned = foreground & (branches>0); sk = (skeleton>0) & assigned
    priors = patch_priors(label, branches, depths, start, full_shape)
    ids, counts = np.unique(branches[assigned], return_counts=True)
    ids = [int(x) for x in ids]; counts = [int(x) for x in counts]
    volume_depth = hist(depths[assigned]); centerline_depth = hist(depths[sk]); branch_details=[]
    for branch, count in zip(ids, counts):
        depth_counts = hist(depths[foreground & (branches==branch)])
        representative_depth = int(mode(depth_counts))
        branch_details.append({'branch_id':branch,'foreground_voxels':count,
                               'representative_global_depth':representative_depth,
                               'generation_bin':generation_bin(representative_depth)})
    def bin_hist(depth_hist):
        output=Counter()
        for depth, n in depth_hist.items(): output[generation_bin(int(depth))]+=n
        return {str(k):int(v) for k,v in sorted(output.items())}
    distributions = {'volume':bin_hist(volume_depth), 'equal_branch':hist(x['generation_bin'] for x in branch_details),
                     'centerline_voxel':bin_hist(centerline_depth)}
    aggregation = {k:int(mode(v)) if v else None for k,v in distributions.items()}
    dominant = int(priors[0]); high_volume=int(sum(n for k,n in distributions['volume'].items() if int(k)>dominant))
    high_sk=int(sum(n for k,n in distributions['centerline_voxel'].items() if int(k)>dominant))
    tile_labels=[]; tile_nonempty=[]; size=label.shape[0]; half=size//2
    for z in (0,half):
        for y in (0,half):
            for x in (0,half):
                region=(slice(z,z+half),slice(y,y+half),slice(x,x+half))
                child=patch_priors(label[region],branches[region],depths[region],start+np.array([z,y,x]),full_shape)
                tile_labels.append(int(child[1]))
                if child[1]>0: tile_nonempty.append(int(child[1]))
    struct_formula, complexity=complexity_structure(len(ids),int(foreground.sum()))
    fixed_struct,fixed_complexity=complexity_structure(len(ids),parent_volume)
    same_branches=ids==parent_branches
    assigned_n=int(assigned.sum()); sk_n=int(sk.sum()); foreground_n=int(foreground.sum())
    return {'priors':[int(x) for x in priors], 'foreground_voxels':foreground_n,
            'foreground_fraction':float(foreground.mean()),'empty_roi':not bool(foreground_n),
            'assigned_foreground_voxels':assigned_n,'unassigned_foreground_voxels':foreground_n-assigned_n,
            'branch_count':len(ids),'visible_global_branch_ids':ids,'branch_details':branch_details,
            'dominant_branch_volume_fraction':max(counts)/foreground_n if foreground_n and counts else None,
            'volume_depth_histogram':volume_depth,'centerline_depth_histogram':centerline_depth,
            'generation_histograms':distributions,'aggregation_modal_generation':aggregation,
            'generation_histogram_entropy_bits':{k:entropy(v) for k,v in distributions.items()},
            'multiple_generation_bins_present':len(distributions['volume'])>1,
            'higher_bin_than_dominant_volume_fraction':high_volume/assigned_n if assigned_n else None,
            'higher_bin_than_dominant_centerline_fraction':high_sk/sk_n if sk_n else None,
            'cached_centerline_voxels':sk_n,'thin_centerline_radius_le1mm_fraction':float((radius[sk]<=1).mean()) if sk_n else None,
            'fixed_2x2x2_structure_labels':tile_labels,'nonempty_subtiles':len(tile_nonempty),
            'nonempty_subtile_structure_classes':sorted(set(tile_nonempty)),
            'mixed_local_structure_proxy':len(set(tile_nonempty))>1,
            'subtile_structure_entropy_bits':entropy(Counter(tile_nonempty)),
            'natural_complexity':complexity,'structure_by_complexity':struct_formula,
            'counterfactual_complexity_fixed128_volume':fixed_complexity,
            'counterfactual_structure_fixed128_volume':fixed_struct,
            'natural_vs_fixed128_volume_structure_disagreement':struct_formula is not None and struct_formula!=fixed_struct,
            'same_visible_branches_as128':same_branches,
            'same_branch_set_structure_change_volume_threshold_only':len(ids)>1 and same_branches and int(priors[1])!=parent_structure,
            'counterfactual_roi_sample_weight_NOT_APPLIED':sample_weight(int(priors[0]),int(priors[1]))}


def full_tree_extent(label, spacing):
    """Whole cached GT scan for measurement only; never a sampling/input ROI."""
    import numpy as np
    low=np.array(label.shape); high=np.zeros(3,dtype=int); count=0; content=hashlib.sha256()
    for z in range(0,label.shape[0],16):
        chunk=np.asarray(label[z:z+16]); content.update(np.ascontiguousarray(chunk).tobytes())
        if not np.isin(chunk,(0,1)).all(): raise ValueError('Nonbinary cached GT')
        coordinates=np.nonzero(chunk); count+=len(coordinates[0])
        if len(coordinates[0]):
            offset=np.array([z,0,0]); lo=np.array([c.min() for c in coordinates])+offset
            hi=np.array([c.max()+1 for c in coordinates])+offset; low=np.minimum(low,lo); high=np.maximum(high,hi)
    extent=(high-low)*np.asarray(spacing) if count else None
    return {'grid':'full .5mm nearest-neighbour cached GT, not native annotation grid','foreground_voxels':count,
            'bbox_zyx_half_open':[low.tolist(),high.tolist()] if count else None,
            'extent_mm_in_cache_axis_order_zyx':extent.tolist() if count else None,
            'cached_label_content_sha256':content.hexdigest(),'dtype':label.dtype.str,'shape_zyx':list(label.shape),
            'roi_side_mm_over_bbox_extent':{str(n):[n*.5/float(v) for v in extent] if count else None for n in ROI_SIZES},
            'used_for_sampling_or_input_crop':False}


def summarize(rows, train_ids):
    output={}
    for size in ROI_SIZES:
        selected=[r for r in rows if r['roi_size_vox']==size]; positive=[r for r in selected if not r['empty_roi']]
        eligible=[r for r in positive if r['nonempty_subtiles']>=2]; by_case={}
        for case_id in train_ids:
            case_rows=[r for r in selected if r['case_id']==case_id]
            classes=hist(r['priors'][1] for r in case_rows)
            by_case[case_id]={'n_patches':len(case_rows),'structure_histogram':classes,
                             'structure_entropy_bits':entropy(classes),'structure_varies_across_fixed_patches':len(classes)>1}
        def rate(name, population):
            return {'count':sum(bool(r[name]) for r in population),'denominator':len(population),
                    'fraction':sum(bool(r[name]) for r in population)/len(population) if population else None}
        disagreements={}
        for name in ('volume','equal_branch','centerline_voxel'):
            available=[r for r in positive if r['aggregation_modal_generation'][name] is not None]
            differences=sum(r['priors'][0]!=r['aggregation_modal_generation'][name] for r in available)
            disagreements[name]={'different_from_current_dominant_branch_label':differences,'denominator':len(available),
                                  'fraction':differences/len(available) if available else None}
        labels=hist(r['priors'][1] for r in selected); positive_labels=hist(r['priors'][1] for r in positive)
        output[str(size)]={'physical_side_mm':size*.5,'n_fixed_patches':len(selected),'n_nonempty':len(positive),
            'n_empty':len(selected)-len(positive),'structure_histogram_including_empty0':labels,
            'structure_histogram_positive':positive_labels,'structure_entropy_all_bits':entropy(labels),
            'structure_entropy_positive_bits':entropy(positive_labels),'generation_histogram_including_empty0':hist(r['priors'][0] for r in selected),
            'branch_count_nonempty':stats(r['branch_count'] for r in positive),
            'dominant_branch_volume_fraction':stats(r['dominant_branch_volume_fraction'] for r in positive),
            'higher_bin_than_dominant_volume_fraction':stats(r['higher_bin_than_dominant_volume_fraction'] for r in positive),
            'higher_bin_than_dominant_centerline_fraction':stats(r['higher_bin_than_dominant_centerline_fraction'] for r in positive),
            'multiple_generation_bins_present':rate('multiple_generation_bins_present',positive),
            'mixed_local_structure_proxy_all_nonempty':rate('mixed_local_structure_proxy',positive),
            'mixed_local_structure_proxy_eligible_ge2_nonempty_subtiles':rate('mixed_local_structure_proxy',eligible),
            'fixed128_denominator_structure_sensitivity':rate('natural_vs_fixed128_volume_structure_disagreement',positive),
            'unchanged_branch_set_volume_threshold_only_structure_change':rate('same_branch_set_structure_change_volume_threshold_only',positive),
            'aggregation_label_disagreement':disagreements,'within_case_structure_variation':by_case,
            'structure_differs_from_whole128_among_both_nonempty':{
                'count':sum(r['priors'][1]!=r['whole128_priors'][1] for r in positive if r['whole128_priors'][1]>0),
                'denominator':sum(r['whole128_priors'][1]>0 for r in positive)},
            'generation_differs_from_whole128_among_both_nonempty':{
                'count':sum(r['priors'][0]!=r['whole128_priors'][0] for r in positive if r['whole128_priors'][0]>0),
                'denominator':sum(r['whole128_priors'][0]>0 for r in positive)},
            'whole128_positive_but_central_roi_empty':sum(r['whole128_priors'][0]>0 for r in selected if r['empty_roi']),
            'cases_with_structure_variation':sum(x['structure_varies_across_fixed_patches'] for x in by_case.values()),
            'n_cases':len(train_ids)}
    return output


def main():
    root=Path(__file__).resolve().parents[1]; code=root/'code'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-config',type=Path,default=root/'results/gta_enhanced200x125/training/config.json')
    p.add_argument('--manifest',type=Path,default=root/'dataset/cache/manifest.json')
    p.add_argument('--output',type=Path); p.add_argument('--reference-bank',type=Path)
    p.add_argument('--classifier-statistics',type=Path,default=root/'results/classifier80x64/training/attribute_statistics.json')
    p.add_argument('--seed',type=int,default=20261013); p.add_argument('--bank-epoch',type=int,default=0)
    p.add_argument('--patches-per-case',type=int,default=2); p.add_argument('--plan',action='store_true')
    p.add_argument('--plan-output',type=Path)
    a=p.parse_args()
    if a.bank_epoch<0 or a.patches_per_case<1: p.error('Nonnegative bank epoch and positive patches-per-case required')
    if not a.plan and (a.output is None or not a.output.resolve().is_relative_to(root/'results')):
        p.error('Explicit CPU audit needs a fresh output inside this ROOT/results')
    os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
    if not a.plan and os.name=='posix': os.nice(10)
    sys.dont_write_bytecode=True; sys.path.insert(0,str(code))
    import numpy as np
    from classification_data import CohortDataset, internal_split
    from data import load_manifest, case_directory, extract_patch, patch_priors, generation_bin
    config_path=a.training_config.resolve(); manifest_path=a.manifest.resolve()
    config=json.loads(config_path.read_text(encoding='utf-8')); manifest=load_manifest(manifest_path)
    source,files=source_identity(code); train_ids=config['internal_split']['train']; val_ids=config['internal_split']['val']
    if (sha(manifest_path)!=EXPECTED_MANIFEST or source!=EXPECTED_SOURCE or config.get('source_sha256')!=source
        or config.get('source_files_sha256')!=files or config.get('manifest_sha256')!=EXPECTED_MANIFEST
        or config.get('smoke') is not False or config.get('method')!='gta' or config.get('patch_size')!=128
        or config.get('cohort')!='internal28' or config.get('internal_selection_valid') is not True
        or config.get('allow_selection_overlap') is not False or config.get('fold')!=0
        or config.get('outer8_used_for_selection') is not False or config.get('trained_case_ids')!=train_ids
        or internal_split(manifest,config['fold'],config['split_seed'])!=config['internal_split']):
        raise ValueError('Requires exact frozen main128, training28/internal4 source/manifest/split identity')
    reference=json.loads(a.reference_bank.read_text(encoding='utf-8')) if a.reference_bank else None
    bank=[]; counts={case:0 for case in train_ids}
    for index in range(28*a.patches_per_case*100):
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,a.bank_epoch,index])); case=train_ids[int(rng.integers(28))]
        if counts[case]<a.patches_per_case: bank.append({'index':index,'case_id':case}); counts[case]+=1
        if len(bank)==28*a.patches_per_case: break
    if len(bank)!=28*a.patches_per_case: raise ValueError('Fixed bank did not cover every training patient')
    inputs={config_path:sha(config_path),manifest_path:sha(manifest_path),Path(__file__).resolve():sha(__file__),
            root/'ops/diagnose_gaussian.py':sha(root/'ops/diagnose_gaussian.py')}
    if a.reference_bank: inputs[a.reference_bank.resolve()]=sha(a.reference_bank)
    classifier_stats=None
    if a.classifier_statistics.is_file():
        raw=json.loads(a.classifier_statistics.read_text(encoding='utf-8'))
        if raw.get('patients')!=train_ids: raise ValueError('Classifier-statistics patients differ from training28')
        classifier_stats={'file':str(a.classifier_statistics.resolve()),'sha256':sha(a.classifier_statistics),
                          'n_fixed_pretraining_patches':raw['samples'],'heads':{}}
        for name in ('generation','structure','location'):
            c=raw['heads'][name]['counts']; total=sum(c[1:]); classifier_stats['heads'][name]={
                'counts':c,'positive_patches':total,'positive_fractions':[n/total if total else None for n in c[1:]]}
        classifier_stats['interpretation']='Pretraining geometry-sampler frequency evidence; not accuracy and not this standard-sampler bank prevalence'
        inputs[a.classifier_statistics.resolve()]=sha(a.classifier_statistics)
    cases={x['case_id']:x for x in manifest['cases']}
    if any(cases[c]['spacing_zyx_mm']!=[.5,.5,.5] for c in train_ids): raise ValueError('Audit scales require .5mm cache')
    plan={'schema_version':1,'role':'training28 label-scale diagnosis only; no network forward/training/selection',
          'root':str(root),'source_sha256':source,'manifest_sha256':EXPECTED_MANIFEST,'training_patients':train_ids,
          'excluded_internal4':val_ids,'outer8_read':False,'validation_GT_read':False,'input_size_vox':128,
          'input_side_mm':64.,'roi_sizes_vox':list(ROI_SIZES),'roi_sides_mm':[n*.5 for n in ROI_SIZES],
          'same_center':True,'seed':a.seed,'bank_epoch':a.bank_epoch,'patches_per_case':a.patches_per_case,'bank':bank,
          'sampling':config['sampling'],'sampling_fov':config['sampling_fov'],'augmentation':False,
          'bank_population':'patient-stratified original train-time sampler indices; not natural patch frequency',
          'tree_bbox_role':'full cached training GT physical extent for scale context only; never sampler/image ROI',
          'aggregation':'current dominant-volume branch vs volume/equal-visible-branch/cached-centerline-voxel mode',
          'local_mixture_definition':'at least2 distinct positive structure proxy labels across fixed2x2x2 subtiles',
          'scale_sensitivity':'same branch set but changed structure isolates old log(volume) threshold; fixed128-volume counterfactual also reported',
          'fused_readout_candidate_only':{'64':'central fused[2:6]^3 avg+max','96':'central fused[1:7]^3 avg+max','128':'fused[0:8]^3 avg+max',
                                         'head_input_dim':512,'heads_unchanged':3,'classes_per_head':7,'active_network_changed':False},
          'classification_accuracy_gate':False,'parameter_updates':False,'GPU_used':False,'classifier_pretraining_frequency_evidence':classifier_stats,
          'inputs':[{'path':str(k),'sha256':v} for k,v in inputs.items()]}
    if a.plan:
        if a.plan_output: write(a.plan_output,plan)
        print(json.dumps(plan,indent=2,allow_nan=False)); return
    if (manifest_path!=root/'dataset/cache/manifest.json' or not config_path.is_relative_to(root/'results')
        or Path(config['output']).resolve()!=config_path.parent): raise ValueError('Execution requires this run ROOT; relocation is plan-only')
    output=a.output.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Fresh output required; no overwrite of active or diagnostic results')
    output.mkdir(parents=True,exist_ok=True); write(output/'audit_plan.json',plan)
    dataset=CohortDataset(manifest,train_ids,fold=config['fold'],role='train',sampling=config['sampling'],
        sampling_fov=config['sampling_fov'],steps=bank[-1]['index']+1,patch_size=128,seed=a.seed,augment=False,return_tensors=False)
    dataset.set_epoch(a.bank_epoch)
    fixed_bank={'schema_version':1,'source_sha256':source,'manifest_sha256':EXPECTED_MANIFEST,'training_patients':train_ids,
        'excluded_internal4':val_ids,'fold':config['fold'],'seed':a.seed,'bank_epoch':a.bank_epoch,
        'patches_per_training_case':a.patches_per_case,'patch_size':128,'sampling':config['sampling'],
        'sampling_fov':config['sampling_fov'],'augmentation':False,'role':'training28 only; no checkpoint selection or independent validation','samples':[]}
    rows=[]; tree_extents={}; cache_stamps={}
    for n,case_id in enumerate(train_ids):
        folder=case_directory(manifest,case_id); label=np.load(folder/'label.npy',mmap_mode='r')
        metadata=json.loads((folder/'case.json').read_text(encoding='utf-8'))
        for key in ('case_id','shape_zyx','spacing_zyx_mm','source_sha256','preprocessing_version'):
            if metadata.get(key)!=cases[case_id].get(key): raise ValueError('Cached metadata differs from frozen manifest: '+case_id+'/'+key)
        if label.dtype!=np.uint8 or list(label.shape)!=cases[case_id]['shape_zyx']: raise ValueError('Cached GT dtype/grid differs: '+case_id)
        tree_extents[case_id]=full_tree_extent(label,cases[case_id]['spacing_zyx_mm'])
        tree_extents[case_id]['original_GT_source_sha256']=cases[case_id]['source_sha256']['label']; del label
        for name in ('image.npy','label.npy','gaussian.npy','branch_id.npy','depth.npy','skeleton.npy','radius.npy','sampling.npz','case.json'):
            path=folder/name; stat=path.stat(); cache_stamps[path]=(stat.st_size,stat.st_mtime_ns)
        inputs[folder/'case.json']=sha(folder/'case.json')
        if n%7==6: print(json.dumps({'phase':'training_GT_extent','completed_cases':n+1,'total_cases':28}),flush=True)
    for number,item in enumerate(bank):
        sample=dataset[item['index']]; identity=sample_identity(item,sample)
        if reference is not None:
            protocol={k:v for k,v in fixed_bank.items() if k!='samples'}
            compare_bank(protocol,{k:v for k,v in reference.items() if k!='samples'})
            compare_bank(identity,reference['samples'][number])
        fixed_bank['samples'].append(identity)
        folder=case_directory(manifest,item['case_id']); start=sample['start_zyx']; maps={}
        for name in ('branch_id','depth','skeleton','radius'):
            array=np.load(folder/(name+'.npy'),mmap_mode='r')
            if list(array.shape)!=cases[item['case_id']]['shape_zyx']: raise ValueError('Topology map grid differs: '+name)
            maps[name]=extract_patch(array,start,(128,128,128)); del array
        if any(not np.isfinite(v).all() for v in maps.values()): raise ValueError('Nonfinite topology cache patch')
        if maps['radius'].min()<0 or not np.isin(maps['skeleton'],(0,1)).all(): raise ValueError('Invalid radius/skeleton target')
        label=sample['label']; parent_priors=sample['priors']; parent_branches=sorted(int(x) for x in np.unique(maps['branch_id'][label>0]) if x>0)
        extra_identity={k:array_identity(v) for k,v in maps.items()}
        for size in ROI_SIZES:
            low=(128-size)//2; region=(slice(low,low+size),)*3; roi_start=start+low
            measured=roi_metrics(label[region],maps['branch_id'][region],maps['depth'][region],maps['skeleton'][region],maps['radius'][region],
                roi_start,tuple(cases[item['case_id']]['shape_zyx']),patch_priors,generation_bin,int((label>0).sum()),parent_branches,int(parent_priors[1]))
            if size==128 and measured['priors']!=[int(x) for x in parent_priors]: raise ValueError('Sampler prior and audited fullpatch proxy differ')
            measured.update(item,roi_size_vox=size,roi_side_mm=size*.5,start_zyx=roi_start.tolist(),
                input128_sample_identity=identity,topology128_sample_identity=extra_identity,
                whole128_priors=[int(x) for x in parent_priors],
                actual_whole128_seg_sample_weight=sample_weight(int(parent_priors[0]),int(parent_priors[1]),config['disable_anatomy_reweighting']))
            rows.append(measured)
        if number%14==13: print(json.dumps({'phase':'fixed_input_local_ROIs','completed_patches':number+1,'total_patches':len(bank)}),flush=True)
    if reference is not None: compare_bank(fixed_bank,reference)
    write(output/'fixed_sampling_bank.json',fixed_bank); write(output/'training_tree_extents.json',tree_extents)
    (output/'roi_patch_metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    scalar_keys=('case_id','index','roi_size_vox','roi_side_mm','foreground_voxels','branch_count','dominant_branch_volume_fraction',
        'multiple_generation_bins_present','higher_bin_than_dominant_volume_fraction','higher_bin_than_dominant_centerline_fraction',
        'mixed_local_structure_proxy','natural_vs_fixed128_volume_structure_disagreement','same_branch_set_structure_change_volume_threshold_only',
        'actual_whole128_seg_sample_weight','counterfactual_roi_sample_weight_NOT_APPLIED')
    with (output/'roi_patch_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=scalar_keys+('generation','structure','location')); writer.writeheader()
        for r in rows: writer.writerow({k:r[k] for k in scalar_keys}|dict(zip(('generation','structure','location'),r['priors'])))
    for path,expected in inputs.items():
        if sha(path)!=expected: raise RuntimeError('Source/input metadata changed during audit: '+str(path))
    if source_identity(code)!=(source,files): raise RuntimeError('Frozen source changed during audit')
    for path,stamp in cache_stamps.items():
        stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=stamp: raise RuntimeError('Cache changed during audit: '+str(path))
    summary={'provenance':plan,'roi_scales':summarize(rows,train_ids),'training_tree_extent_summary':{
        str(axis):stats(x['extent_mm_in_cache_axis_order_zyx'][axis] for x in tree_extents.values() if x['extent_mm_in_cache_axis_order_zyx']) for axis in range(3)},
        'limitations':['Only training28; no validation/performance claim. Two fixed samples per case by default are sparse.',
            'Cached branch-depth is a thickest-rooted geometric proxy, not validated coronary anatomical generation.',
            'ROI cropping preserves global branch IDs/depth and cached skeleton; centreline voxel weights are not exact arc length.',
            'Structure subtiles use scale-dependent thresholds; proxy disagreement is not proof of anatomical heterogeneity.',
            'Fixed128-volume counterfactual diagnoses denominator sensitivity, not a proposed new target.',
            'Wholepatch generation/structure weights multiply the entire segmentation loss; ROI weights are diagnostic and never applied.',
            'Classification frequencies/accuracy are diagnostic, not a segmentation startup gate.',
            'Any local pooling/label change needs a separate versioned candidate; active classifier80/final and training remain unchanged.']}
    write(output/'summary.json',summary)
    artifacts=('audit_plan.json','fixed_sampling_bank.json','training_tree_extents.json','roi_patch_metrics.jsonl','roi_patch_metrics.csv','summary.json')
    write(output/'LABEL_SCALE_AUDIT_COMPLETE.json',{'status':'completed','training_cases':28,'input_patches':len(bank),'roi_observations':len(rows),
        'GPU_used':False,'network_forward':False,'parameter_updates':False,'validation_GT_read':False,
        'artifacts_sha256':{name:sha(output/name) for name in artifacts}})
    print(json.dumps({'output':str(output),'training_cases':28,'input_patches':len(bank),'roi_observations':len(rows),'GPU_used':False}))


if __name__=='__main__': main()
