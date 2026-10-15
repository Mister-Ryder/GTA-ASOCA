"""Analytic array checks only; no ASOCA GT read or model/performance experiment."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True
from diagnose_label_scales import roi_metrics, full_tree_extent, sha


def main():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('label_fixture_frozen_data',root/'code/data.py')
    data=importlib.util.module_from_spec(spec); spec.loader.exec_module(data)
    shape=(128,128,128); label=np.zeros(shape,np.uint8); branch=np.zeros(shape,np.uint16)
    depth=np.zeros(shape,np.uint8); skeleton=np.zeros(shape,np.uint8); radius=np.zeros(shape,np.float16)
    # Large-volume shallow branch plus a long one-voxel high-depth branch.
    label[50:78,50:78,50:78]=1; branch[label>0]=1
    label[:,75,75]=1; branch[:,75,75]=2; depth[:,75,75]=9
    skeleton[60:68,64,64]=1; skeleton[:,75,75]=1; radius[label>0]=.5
    start=np.array([100,100,100]); full=(512,512,512)
    parent=data.patch_priors(label,branch,depth,start,full)
    measured=roi_metrics(label,branch,depth,skeleton,radius,start,full,data.patch_priors,data.generation_bin,
                         int(label.sum()),[1,2],int(parent[1]))
    assert measured['priors'][0]==1 and measured['dominant_branch_volume_fraction']>.99
    assert measured['multiple_generation_bins_present']
    assert measured['aggregation_modal_generation']['centerline_voxel']==6
    assert measured['higher_bin_than_dominant_centerline_fraction']> .9
    tests={'dominant_volume_can_hide_long_high_depth_centerline':'passed'}
    # Same global branch set, changed old log(volume) denominator only.
    label.fill(0); branch.fill(0); depth.fill(0); skeleton.fill(0)
    label[5:30,5:30,5:30]=1; branch[label>0]=1
    label[50,50,50]=1; branch[50,50,50]=1; skeleton[50,50,50]=1
    label[80,80,80]=1; branch[80,80,80]=2; depth[80,80,80]=9; skeleton[80,80,80]=1
    parent=data.patch_priors(label,branch,depth,start,full); region=(slice(32,96),)*3
    measured=roi_metrics(label[region],branch[region],depth[region],skeleton[region],radius[region],start+32,
                         full,data.patch_priors,data.generation_bin,int(label.sum()),[1,2],int(parent[1]))
    assert measured['same_visible_branches_as128'] and measured['same_branch_set_structure_change_volume_threshold_only']
    assert measured['priors'][1]==6 and int(parent[1])==3
    assert measured['counterfactual_structure_fixed128_volume']==3
    assert measured['priors'][2]==int(parent[2])
    tests['unchanged_branch_set_volume_denominator_only_crosses_structure_threshold']='passed'
    tests['concentric_roi_keeps_absolute_location_bin']='passed'
    # Distinct tile shape proxies, and all-empty ROIs retained as zeros.
    cube=np.zeros((64,64,64),np.uint8); ids=np.zeros_like(cube,np.uint16); depths=np.zeros_like(cube)
    cube[2:20,2:4,2:4]=1; ids[cube>0]=1; cube[40:45,40:45,40:45]=1; ids[40:45,40:45,40:45]=2
    mixed=roi_metrics(cube,ids,depths,np.zeros_like(cube),np.zeros_like(cube),start,full,
        data.patch_priors,data.generation_bin,int(cube.sum()),[1,2],3)
    assert mixed['mixed_local_structure_proxy'] and mixed['nonempty_subtile_structure_classes']==[1,2]
    tests['fixed_subtile_structure_proxy_mixture']='passed'
    empty=np.zeros_like(cube)
    measured=roi_metrics(empty,np.zeros_like(ids),empty,empty,empty,start,full,
        data.patch_priors,data.generation_bin,0,[],0)
    assert measured['empty_roi'] and measured['priors']==[0,0,0] and measured['nonempty_subtiles']==0
    assert all(v is None for v in measured['aggregation_modal_generation'].values())
    tests['empty_roi_and_no_centerline_are_preserved_not_filled']='passed'
    example=np.zeros((16,16,16),np.uint8); example[2:6,3:7,4:8]=1
    bbox=full_tree_extent(example,[.5,.5,.5])
    assert bbox['foreground_voxels']==64 and bbox['extent_mm_in_cache_axis_order_zyx']==[2.,2.,2.]
    assert bbox['bbox_zyx_half_open']==[[2,3,4],[6,7,8]] and bbox['used_for_sampling_or_input_crop'] is False
    tests['known_gt_bbox_support_extent_and_no_sampling_use']='passed'
    print(json.dumps({'status':'passed','fixture_only':True,'actual_ASOCA_GT_read':False,'network_forward':False,
        'tool_sha256':sha(root/'ops/diagnose_label_scales.py'),'tests':tests},indent=2))


if __name__=='__main__': main()
