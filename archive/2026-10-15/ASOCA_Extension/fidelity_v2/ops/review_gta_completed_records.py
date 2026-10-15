"""Independent completed-GTA artifact/telemetry review; SHA only for final.pt.

Never imports Torch, executes a server operation, deserializes final weights,
evaluates a model, changes source/config, or infers Dice/ASSD from training loss.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCE = 'bc8dd9dd5f56ea1b0cdf2cecd9adc7b305a54386f3ce2aa993d4fdc7f3d1f652'
CONFIG = 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
MANIFEST = '141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
FINAL = '2b24a0415710e906a01ea827982006c6e7a4d5865f22b88849db78dbcbaba820'
IDS = ['Diseased_10', 'Diseased_20', 'Normal_12', 'Normal_3']
FIELDS = ('segmentation', 'classification_weighted', 'gaussian_weighted', 'guidance_weighted',
          'segmentation_gradient_norm', 'classifier_gradient_norm')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def record(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return math.isfinite(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else True


def aggregate(rows):
    series = {'total_loss': [r['loss'] for r in rows]}
    series.update({k: [r['component_means'][k] for r in rows] for k in FIELDS})
    return {'n_epochs': len(rows), 'epochs_1based': [rows[0]['epoch']+1, rows[-1]['epoch']+1],
            'values': {k: {'mean': statistics.mean(v), 'median': statistics.median(v),
                           'minimum': min(v), 'maximum': max(v), 'sample_sd': statistics.stdev(v)} for k,v in series.items()},
            'first_epoch_end_lr': rows[0]['lr_by_group'], 'last_epoch_end_lr': rows[-1]['lr_by_group']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime-audit', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = args.capture.resolve(); output = args.output.resolve()
    assert not output.exists(), 'Use a fresh independent-review output directory'
    collection = read(capture/'collection_manifest.json')
    expected_capture = collection['files']
    assert len(expected_capture) == len({r['relative_path'] for r in expected_capture}) == 12
    actual = {}
    for entry in expected_capture:
        path = (capture/entry['relative_path']).resolve()
        assert path.is_relative_to(capture)
        actual[entry['relative_path']] = record(path)
        assert actual[entry['relative_path']] == {k:entry[k] for k in ('bytes','sha256')}, entry['relative_path']
    ready_path = capture/'shared_mirror/READY.json'; ready = read(ready_path)
    training = capture/'results/gta_enhanced200x125/training'
    config = read(training/'config.json'); complete = read(training/'TRAINING_COMPLETE.json')
    init = read(training/'initialization_audit.json')
    manifest = read(args.manifest)
    runtime = read(args.runtime_audit)
    six = {name:training/name for name in ('final.pt','config.json','history.jsonl','TRAINING_COMPLETE.json')}
    six.update({'manifest.json':args.manifest, 'input_runtime_audit.json':args.runtime_audit})
    assert set(six) == set(ready['files'])
    mirrored = {name:record(path) for name,path in six.items()}
    assert mirrored == ready['files'], 'Actual six-file evidence differs from READY'
    assert mirrored['config.json']['sha256'] == CONFIG and mirrored['manifest.json']['sha256'] == MANIFEST
    assert mirrored['final.pt']['sha256'] == ready['checkpoint_sha256'] == complete['final_sha256'] == FINAL
    assert ready['status'] == 'fixed_final_payload_ready' and ready['case_ids'] == IDS and ready['checkpoint_epoch'] == 199
    assert ready['source_sha256'] == config['source_sha256'] == complete['source_sha256'] == SOURCE
    assert ready['packages'] == runtime['packages'] and ready['cache_CT_inputs'] == runtime['cache_CT_inputs']
    assert ready['evaluation_sources'] == runtime['evaluation_sources']
    cases = {c['case_id']:c for c in manifest['cases']}
    assert ready['raw_GT_source_sha256'] == {c:cases[c]['source_sha256']['label'] for c in IDS}
    source = hashlib.sha256()
    for name, expected in sorted(config['source_files_sha256'].items()):
        path = args.code/name
        assert digest(path) == expected
        source.update(name.encode()); source.update(path.read_bytes())
    assert source.hexdigest() == SOURCE
    for name, expected in ready['evaluation_sources'].items():
        assert digest(args.code/name) == expected
    freeze_path = capture/'configs/main_source_freeze.json'
    assert digest(freeze_path) == ready['source_freeze_sha256'] == '9b294024e33b0d0aaa12b473be53dd3a81dfcd9dd020446948fe667405c7ff46'
    assert read(freeze_path)['training_dependency_files'] == config['source_files_sha256']
    assert config['epochs'] == 200 and config['steps'] == 125 and config['batch_size'] == 2 and config['patch_size'] == 128
    assert complete['epoch'] == 199 and complete['smoke'] is config['smoke'] is False
    assert complete['segmentation_updates'] == config['segmentation_updates'] == 25000
    assert complete['classifier_pretraining_updates'] == config['classifier_pretraining_updates'] == 5120
    assert complete['checkpoint_selection'] == config['checkpoint_selection'] == 'fixed final'
    train, val = config['internal_split']['train'], config['internal_split']['val']
    assert len(train) == len(set(train)) == 28 and val == IDS and not set(train)&set(val)
    assert set(train)|set(val) == set(manifest['folds'][0]['train'])
    assert not set(train+val)&set(manifest['folds'][0]['val'])
    assert complete['trained_case_ids'] == config['trained_case_ids'] == train
    assert complete['internal_selection_valid'] is config['internal_selection_valid'] is True
    assert config['allow_selection_overlap'] is config['outer8_used_for_selection'] is False
    assert config['no_ds'] is False and config['segmentation_supervision_weights'] == [1., .5, .25]
    assert config['backbone_warmup_epochs'] == 0 and config['gaussian_weight'] == 10 and config['gaussian_late_weight'] is None
    assert init['init_audits'] == config['init_audits']
    donors = {}
    for audit in init['init_audits']:
        assert audit['missing_keys'] == audit['unexpected_keys'] == audit['randomly_retained_keys'] == []
        assert audit['shape_mismatches'] == {}
        assert audit['loaded_key_count'] == audit['component_tensor_count'] == len(audit['loaded_keys'])
        donors[audit['component']] = {k:v for k,v in audit.items() if k not in ('loaded_keys',)}
    assert donors['backbone']['sha256'] == '29b6977b74e547735717fda0785a9d532397df364e1f641bed0e6682db0384d3'
    assert donors['backbone']['source_domain'] == 'airway' and donors['backbone']['loaded_key_count'] == 118
    assert donors['classifier']['sha256'] == '308ebc33d55766458ebd3241c8422dbadb9259b929570bc5f03afd51fbec31a4'
    assert donors['classifier']['source_domain'] == 'asoca' and donors['classifier']['source_epoch'] == 79
    assert donors['classifier']['loaded_key_count'] == 90 and donors['classifier']['path'].endswith('/classifier80x64/training/final.pt')
    history = [json.loads(line) for line in (training/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert [r['epoch'] for r in history] == list(range(200)) and sum(r['iterations'] for r in history) == 25000
    assert [r['stage'] for r in history] == ['frozen']*50+['partial']*20+['joint']*130
    assert all(r['iterations'] == 125 and finite_tree(r) for r in history)
    max_lr_error = 0.; max_ds_error = 0.; max_loss_sum_error = 0.
    warm = config['warmup_epochs']*config['steps']; total = 25000
    for row in history:
        epoch = row['epoch']; component = row['component_means']
        assert row['gaussian_lambda'] == row['gaussian_lambda_configured'] == component['gaussian_lambda'] == 10.
        assert row['classifier_lambda'] == component['classifier_lambda'] == (0. if epoch < 50 else .1)
        assert component['classifier_gradient_norm'] == 0 if epoch < 50 else component['classifier_gradient_norm'] > 0
        assert component['segmentation_gradient_norm'] > 0
        assert math.isclose(component['gaussian_weighted'], 10*component['gaussian'], rel_tol=2e-7, abs_tol=1e-9)
        reconstructed_ds = sum(w*component[f'segmentation_head_{i}'] for i,w in enumerate([1.,.5,.25]))/1.75
        max_ds_error = max(max_ds_error, abs(reconstructed_ds-component['segmentation']))
        assert math.isclose(reconstructed_ds, component['segmentation'], rel_tol=2e-7, abs_tol=1e-7)
        reconstructed_loss = sum(component.get(k, 0) for k in ('segmentation','classification_weighted','gaussian_weighted','guidance_weighted'))
        max_loss_sum_error = max(max_loss_sum_error, abs(reconstructed_loss-row['loss']))
        assert math.isclose(reconstructed_loss, row['loss'], rel_tol=2e-7, abs_tol=1e-7)
        iteration = (epoch+1)*125-1
        factor = iteration/max(1,warm) if iteration < warm else max(0.,1-(iteration-warm)/(total-warm))**.9
        expected_lr = config['lr']*factor
        max_lr_error = max(max_lr_error, abs(row['lr_by_group']['segmentation']-expected_lr))
        assert math.isclose(row['lr_by_group']['segmentation'], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(row['lr_by_group']['classifier'], .1*expected_lr, rel_tol=1e-12, abs_tol=1e-15)
    stage = read(capture/'stage_records/gta200.json')
    assert stage['status'] == 'completed' and stage['return_code'] == 0 and stage['final_sha256'] == FINAL
    publisher = read(capture/'EARLY_FINAL_MIRROR_STATUS.json')
    assert publisher['status'] == 'completed' and publisher['READY_sha256'] == digest(ready_path)
    windows = {name:aggregate(rows) for name,rows in [('epochs161_180',history[-40:-20]), ('epochs181_200',history[-20:]),
                                                    ('epochs181_190',history[-20:-10]), ('epochs191_200',history[-10:])]}
    late = history[-20:]
    report = {'status':'PASS_ACTUAL_COMPLETED_TRAINING_ARTIFACTS_AND_TELEMETRY',
              'role':'actual training200 review; not Dice/ASSD, independent validation or prior-benefit evidence',
              'review_script_sha256':digest(Path(__file__)), 'collection_manifest':record(capture/'collection_manifest.json'),
              'capture_files_verified':actual, 'all12_capture_byte_hashes_match':True,
              'READY_actual_fingerprint':record(ready_path), 'all6_mirror_evidence_hashes_match':True,
              'mirror_evidence_local_mapping':{name:str(path.resolve()) for name,path in six.items()},
              'mirror_evidence_fingerprints':mirrored, 'actual_final_byte_hash_verified':True,
              'final_deserialized':False, 'model_forward_or_training_started':False, 'server_operations':0,
              'source_sha256':SOURCE, 'method':'gta', 'fixed_final_epoch_1based':200,
              'train_ids':train, 'internal_ids':val, 'outer8_used_for_selection':False,
              'segmentation_updates':25000, 'sampled_training_patches':50000,
              'additional_classifier_pretraining_updates':5120,
              'stage_counts':dict(collections.Counter(r['stage'] for r in history)),
              'decoder_DS':True, 'decoder_DS_weights_normalized':[4/7,2/7,1/7], 'encoder_backbone_warmup_epochs':0,
              'Gaussian_lambda':10, 'Gaussian_late_ramp_executed':False, 'donor_audits':donors,
              'windows':windows, 'final_epoch':history[-1],
              'telemetry_reconstruction_max_errors':{'LR':max_lr_error,'DS_weighted_loss':max_ds_error,'total_component_sum':max_loss_sum_error},
              'last20_segmentation_loss_rising_transitions':sum(b['component_means']['segmentation'] > a['component_means']['segmentation'] for a,b in zip(late,late[1:])),
              'last20_total_loss_rising_transitions':sum(b['loss']>a['loss'] for a,b in zip(late,late[1:])),
              'final_segmentation_lr_fraction_of_base':history[-1]['lr_by_group']['segmentation']/config['lr'],
              'late_Gaussian_weighted_loss_to_segmentation_scalar_ratio':windows['epochs181_200']['values']['gaussian_weighted']['mean']/windows['epochs181_200']['values']['segmentation']['mean'],
              'limitations':[
                  'Final weight bytes only were hashed: no Torch deserialization or new model forward in this review.',
                  'Donor hashes/key-load counts are source-bound recorded training audits; donor checkpoint bytes are not reloaded or rehashed in this review.',
                  'Airway donor filename contains epoch599, but its audited payload source_epoch is null; this file does not independently establish its training budget.',
                  'Additional classifier80x64 budget is reported by config/marker; its full pretraining history is outside this capture.',
                  'Parameter-block gradient norms are not comparable measures of task benefit: different parameter sets, optimizer scaling and Jacobians matter.',
                  'Epoch samples and augmentations change; a sampled training-loss decrease is not evidence of validation Dice/ASSD improvement.',
                  'Stage losses differ in active auxiliary objectives: frozen and joint total losses should not be compared as identical objectives.',
                  'Capture proves terminal training; captured live/trajectory status is a timestamped snapshot, not a claim that canonical20/80 evaluation is complete.'
              ],
              'training_extension_decision':{
                  'proven_need_for_more_training':False,
                  'reason':'Late losses remain stochastic and nonzero gradients remain, but LR is near zero; neither proves held-out improvement or more effective updates.',
                  'required_evidence':'canonical internal4 late Dice/physical-mm ASSD trajectory and per-patient behavior, then matched lambda0 control',
                  'required_new_protocol':'An explicit separate extension with preserved parent state, increased update accounting and a stated effective LR schedule; merely increasing epoch number is insufficient.',
                  'current_final_kept_immutable':True,
                  'no_automatic_epoch_extension_or_Gaussian_weight_increase':True}}
    # Rehash the capture artifacts before saving an independent report.
    assert all(record(capture/name) == expected for name,expected in actual.items())
    output.mkdir(parents=True)
    (output/'training_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'epochs161_200_telemetry.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch_1based','total_loss',*FIELDS,'segmentation_lr_last_step','classifier_lr_last_step'])
        writer.writeheader()
        for row in history[-40:]:
            writer.writerow({'epoch_1based':row['epoch']+1,'total_loss':row['loss'],**{k:row['component_means'][k] for k in FIELDS},
                             'segmentation_lr_last_step':row['lr_by_group']['segmentation'],'classifier_lr_last_step':row['lr_by_group']['classifier']})
    text = '''GTA真实200周期训练完成：独立记录复核（2026-10-14）

复核12项capture文件的实际字节数与SHA全部对应collection_manifest，并复核共享READY实际字节及其六份训练证据：四份实际训练产物（config、history、TRAINING_COMPLETE、final.pt），以及此前独立下载的manifest与实际input_runtime_audit。全部对应。final.pt仅以流式SHA验证，未用Torch读取、没有新推理、服务器操作或模型修改。完整报告training_review.json保留每项实际哈希和六份证据的本地映射。

训练事实：正式fold0，28例训练、4例内部诊断且不重叠；200×125=25,000分割更新、batch2共50,000训练块。额外分类器预训练为记录中的80×64=5,120更新。实际阶段为50 frozen、20 partial、130 joint；前50周期classifier梯度为0，后150周期平均梯度均为正且有限。解码器深监督为1/.5/.25归一化（4/7、2/7、1/7），encoder/backbone warmup为0。lambda_Gaussian在全部200周期始终为10，gaussian_late_weight=null，配置中150/200两个ramp边界没有导致后期升权。

初始化记录：airway_backbone_epoch599.pkl固定SHA29b697...，118/118 tensors完整加载；ASOCA classifier80x64固定final.pt SHA308ebc...，90/90 tensors完整加载，source_epoch79。init_audit与config内记录精确一致，无missing/unexpected/shape mismatch。不是以分类准确率决定末权重；config中classification selection report是附带诊断。backbone文件的source_epoch为null，仅文件名含599，不能凭本次小记录单独证明历史训练周期。此次没有重读donor实际字节，来源证明是被冻结source绑定的训练审计记录。

末期观察（均为实际epoch训练遥测，非验证性能）
窗口          总loss均值    分割loss均值   分类加权loss    Gaussian加权loss
161–180       0.380836      0.251488       0.128863        0.000289341
181–200       0.375269      0.246111       0.128678        0.000283786
181–190       0.378769      0.248893       0.129397        0.000284268
191–200       0.371770      0.243330       0.127959        0.000283305

末20分割loss范围0.226843–0.282887，总loss范围0.335304–0.426635，均不是单调下降；191–200总loss中位数0.374419，181–190中位数0.374252，不能只看mean或最后最低的一点就认定持续优化。末20平均segmentation梯度范数0.882729，classifier梯度范数2.618514，均保持有限非零；两者覆盖不同参数组，不能直接用大小比较任务的贡献或认定分类任务妨碍分割。

每条epoch记录的LR是该epoch最后一个step，而非整个epoch平均LR。第181周期末segLR3.690905e-5，最后一周期末3.379645e-8，classifier LR为其0.1倍。全部200行LR与warmup5＋poly0.9配方复算一致；每行归一化DS分割loss与总loss组成也复算一致。第200周期总loss0.335304、分割loss0.226843、Gaussian加权loss0.000296419。末20Gaussian加权loss/分割loss的均值比约0.1153%，这是标量loss比例，不是有效梯度、结构信息利用或Dice收益。

关于更多训练：现阶段只能证明训练完整执行且没有遥测非有限异常，不能证明训练周期不足，也不能证明已达最佳性能。末期LR已接近零，直接增加epoch数量不等于增加有效优化；现有strict-resume也不能把200随意改400继续当原协议。需要先看canonical内部4例晚期Dice/物理mm ASSD、逐病例趋势和同条件lambda0控制，再决定一份独立的延长方案，明确父checkpoint/optimizer/RNG状态、额外更新预算与实际LR策略，保留本次fixed-final200不变。小Gaussian loss与残余梯度本身都不能成为自动升lambda或延长训练的依据。分类器准确率仍是辅助分析，最终主任务是分割。

本报告未读取早期预览两例指标、未计算不完整病例均值、未宣称canonical20/80评估完成，也未判定GTA和U-Net胜负。epoch161–200的可复用telemetry CSV随报告保存，完整实际history仍在原capture中。
'''
    (output/'training_review.txt').write_text(text,encoding='utf-8')
    print(json.dumps({'status':report['status'],'verified_capture_files':12,'verified_mirror_evidence_files':6,
                      'final_byte_sha256':FINAL,'output':str(output),'model_forward_started':False},indent=2))


if __name__ == '__main__':
    main()
