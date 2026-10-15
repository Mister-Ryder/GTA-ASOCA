"""File-level CPU fixture; never a GTA/U-Net model-performance result.

Uses real40-case manifest/original4 GT files and real saved NRRDs. Identity and
empty-mask predictions have known primary metrics. No model/checkpoint is made.
Auxiliary metric metadata and training histories are synthetic fixture inputs.
"""
import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

import numpy as np
import SimpleITK as sitk

from compare_fixed_finals import (compare, digest, read, read_image, IDS, SUMMARY_FIELDS, primary_from_pixels, load_module)
from verify_postfit import DISTANCES, summary_of


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def refresh(folder, config, manifest, sources, rows):
    trajectory, evaluation = folder / 'internal4_trajectory', folder / 'internal4_trajectory/final'
    for row in rows: write(evaluation / (row['case_id'] + '.json'), row)
    (evaluation / 'case_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    summary = {'case_ids': IDS, 'smoke': False, 'fold': 0, 'method': config['method'], 'checkpoint_epoch': 199,
               'evaluation_split': 'internal_diagnostic',
               'evaluation_role': 'internal diagnostic/model-selection cases; not final independent test',
               'training_source_verified_against_live_code': True, 'geometry_only': False,
               'checkpoint_selection': 'fixed final', 'formal_final_test': False, 'internal_selection_valid': True,
               'primary_metrics': ['dice','assd_mm'], 'metric_protocol': manifest['metric_protocol'],
               'overall': summary_of(rows)}
    for group in ('Normal','Diseased'): summary[group] = summary_of([r for r in rows if r['group'] == group])
    write(evaluation / 'summary.json', summary)
    final_hash = rows[0]['checkpoint_sha256']
    write(evaluation / 'EVALUATION_COMPLETE.json', {'checkpoint_sha256': final_hash,
          'manifest_sha256': config['manifest_sha256'], 'evaluation_sources': sources,
          'case_ids': IDS, 'summary_sha256': digest(evaluation / 'summary.json'), 'wall_seconds': 1.})
    remote_training = PurePosixPath(config['output'])
    candidate = {'path': str(remote_training/'final.pt'), 'epoch': 199, 'name': 'final', 'checkpoint_sha256': final_hash}
    protocol = {'training_config_sha256': digest(folder/'training/config.json'), 'manifest_sha256': config['manifest_sha256'],
                'source_sha256': config['source_sha256'], 'evaluation_sources': sources, 'case_ids': IDS,
                'split': 'internal_diagnostic', 'smoke': False, 'formal_final_test': False,
                'outer8_used_for_selection': False, 'checkpoints': [candidate]}
    write(trajectory/'trajectory_protocol.json', protocol)
    remote_evaluation = str(remote_training.parent/'internal4_trajectory/final')
    record = candidate | {'evaluation_output': remote_evaluation, 'summary_path': remote_evaluation+'/summary.json',
                         'split': 'internal_diagnostic', 'role': summary['evaluation_role'], 'n_cases': 4,
                         'overall': summary['overall'], 'seconds': 1., 'smoke': False,
                         'checkpoint_selection': 'fixed final', 'formal_final_test': False}
    (trajectory/'validation_history.jsonl').write_text(json.dumps(record)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; out = args.output.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Fixture needs a fresh output folder')
    out.mkdir(parents=True, exist_ok=True)
    code = root/'code'; metrics = load_module('fixture_metric_definitions', code/'metrics.py')
    manifest = read(args.manifest); cases = {c['case_id']:c for c in manifest['cases']}
    context = {'metric_protocol':metrics.METRIC_PROTOCOL}
    sources = {n:digest(code/n) for n in ('evaluate.py','metrics.py','evaluate_trajectory.py')}
    snapshot = root/'server_records/snapshot_training_20261014_rootrecheck'
    records = {}
    for method, name in [('gta','gta_38773/results/gta_enhanced200x125'), ('unet','unet_17552/results/unet_recipe200x125')]:
        folder = out/method; training = folder/'training'; evaluation = folder/'internal4_trajectory/final'
        training.mkdir(parents=True); evaluation.mkdir(parents=True)
        config = read(snapshot/name/'training/config.json')
        config['output'] = f'/root/fixture_only/{method}/training'
        write(training/'config.json', config)
        # No fake .pt payload: the tool explicitly records weights_bytes_checked=false.
        final_hash = digest(training/'config.json')
        write(training/'TRAINING_COMPLETE.json', {'epoch':199,'smoke':False,'manifest_sha256':config['manifest_sha256'],
              'source_sha256':config['source_sha256'],'trained_case_ids':config['trained_case_ids'],
              'final_sha256':final_hash,'segmentation_updates':25000,'internal_selection_valid':True,
              'checkpoint_selection':'fixed final'})
        (training/'history.jsonl').write_text(''.join(json.dumps({'epoch':e,'iterations':125,'loss':.2})+'\n'
                                                       for e in range(200)), encoding='utf-8')
        rows = []
        for case_id in IDS:
            label = args.raw_root/cases[case_id]['label_relative_path']
            prediction = evaluation/(case_id+'_prediction.nrrd'); shutil.copy2(label,prediction)
            row = {k:0. if k in DISTANCES else 1. for k in SUMMARY_FIELDS}
            row.update(case_id=case_id, group=cases[case_id]['group'], empty_prediction=False,
                       method=method, fold=0, smoke=False, checkpoint_epoch=199,
                       checkpoint_sha256=final_hash, manifest_sha256=config['manifest_sha256'],
                       training_source_sha256=config['source_sha256'], label_sha256=cases[case_id]['source_sha256']['label'],
                       prediction_sha256=digest(prediction), evaluation_source_sha256=sources['evaluate.py'],
                       metrics_source_sha256=sources['metrics.py'], metric_grid='original CT',
                       training_source_verified_against_live_code=True, geometry_only=False,
                       evaluation_split='internal_diagnostic', formal_final_test=False, checkpoint_selection='fixed final',
                       internal_selection_valid=True, metric_protocol_version=metrics.METRIC_PROTOCOL_VERSION,
                       assd_definition=metrics.ASSD_POOLED_DEFINITION,
                       assd_directional_mean_definition=metrics.ASSD_DIRECTIONAL_DEFINITION,
                       surface_connectivity_neighbors=6, surface_area_weighted=False)
            rows.append(row)
        refresh(folder, config, context, sources, rows)
        records[method] = (folder,config,rows)
    def run():
        return compare(out/'gta/training',out/'gta/internal4_trajectory/final',out/'unet/training',
                       out/'unet/internal4_trajectory/final',args.manifest,args.raw_root,code)
    tests = {}; result = run()
    assert result['n_patients']==4 and all(v['dice']['mean']==1. and v['assd_mm']['mean']==0.
                                                 for v in result['patient_statistics'].values())
    tests['real_original_label_bytes_NRRD_headers_and_relocated_fixed_final_records'] = 'passed'
    missing = out/'gta/internal4_trajectory/final'/ (IDS[0]+'.json')
    hidden = missing.with_suffix('.hidden'); missing.rename(hidden)
    try:
        try: run()
        except ValueError as error: tests['missing_case_rejected'] = str(error)
        else: raise AssertionError('Missing case accepted')
    finally: hidden.rename(missing)
    folder,config,rows = records['unet']
    changed = copy.deepcopy(rows); changed[0]['label_sha256']='0'*64
    refresh(folder,config,context,sources,changed)
    try:
        try: run()
        except ValueError as error: tests['wrong_GT_SHA_rejected'] = str(error)
        else: raise AssertionError('Wrong GT accepted')
    finally: refresh(folder,config,context,sources,rows)
    reference = read_image(args.raw_root/cases[IDS[0]]['label_relative_path'])
    empty = sitk.Image(reference.GetSize(),sitk.sitkUInt8); empty.CopyInformation(reference)
    prediction = folder/'internal4_trajectory/final'/ (IDS[0]+'_prediction.nrrd')
    with tempfile.TemporaryDirectory(prefix='asoca_pair_empty_') as directory:
        temporary_prediction = Path(directory)/'empty.nrrd'
        sitk.WriteImage(empty,str(temporary_prediction),True)
        shutil.copy2(temporary_prediction,prediction)
    changed = copy.deepcopy(rows); changed[0].update({k:None for k in DISTANCES})
    changed[0].update(dice=0.,empty_prediction=True,prediction_sha256=digest(prediction))
    refresh(folder,config,context,sources,changed)
    result = run()
    assert len(result['per_case'])==4 and result['patient_statistics']['unet']['assd_mm']['mean'] is None
    assert result['patient_statistics']['unet']['assd_mm']['n_defined']==3
    assert result['paired_statistics']['assd_gta_minus_unet_mm']['mean'] is None
    tests['real_empty_prediction_kept_in_N4_with_no_complete_ASSD_mean'] = 'passed'
    one = np.zeros((9,9,9),np.uint8); shifted=one.copy(); one[3,3,3]=1; shifted[5,3,3]=1
    moved = primary_from_pixels(one,shifted,(2.,1.5,.5)); assert moved['dice']==0. and moved['assd_mm']==4.
    two=one.copy(); two[3,3,5]=1
    unequal=primary_from_pixels(one,two,(1.,1.,1.)); assert abs(unequal['assd_mm']-2/3)<1e-12
    assert abs(unequal['assd_mm']-metrics.surface_distance_metrics(one,two,(1.,1.,1.))['assd_mm'])<1e-12
    tests['analytic_translation4mm_and_unequal_surface_counts_pooled2over3'] = 'passed'
    report = {'status':'passed','fixture_only':True,'actual_model_inference':False,'server_operations':False,
              'fixture_scope':'Real original GT/SHA/grid and saved identity/empty NRRDs; synthetic training/completion/auxiliary metadata. These are not GTA/UNet experimental performance results.',
              'tool_sha256':digest(root/'ops/compare_fixed_finals.py'),'fixture_script_sha256':digest(__file__),
              'tests':tests,'fixture_files':str(out),'manifest_sha256':digest(args.manifest),
              'original_GT_sha256':{i:cases[i]['source_sha256']['label'] for i in IDS}}
    write(out/'fixture_report.json',report)
    write(out/'fixture_comparison_NOT_MODEL_RESULTS.json',result|{'fixture_only':True})
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
