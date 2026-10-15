"""Read accepted trajectories only; write new diagnostic records, no model/server."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SOURCES = {
    'A_GTA': BASE/'gta_complete_trajectory_20261014/results/gta_enhanced200x125/internal4_trajectory',
    'UNet': BASE/'unet_complete_trajectory_20261014/results/unet_recipe200x125/internal4_trajectory',
}
IDS = ['Diseased_10', 'Diseased_20', 'Normal_12', 'Normal_3']
MANIFEST = '141d97d36d089ccf74ec6ce44133c893a916ae4fe4c07029ec5509aba43a8356'
NAMES = [f'epoch{i:03d}' for i in range(10, 200, 10)]+['final']
METRICS = ('dice', 'assd_mm')
sources = {}


def need(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    data = path.read_bytes()
    sources[str(path.relative_to(BASE)).replace('\\', '/')] = {
        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return data


def js(path): return json.loads(read(path))
def lines(path): return [json.loads(x) for x in read(path).decode('utf-8-sig').splitlines() if x.strip()]
def close(a, b): return (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12))


def write_csv(name, rows):
    with (OUT/name).open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    need(not (OUT/'review.json').exists(), 'Do not overwrite completed review')
    a_acceptance = js(BASE/'gta_canonical_acceptance_20261014/canonical_acceptance_and_pair.json')
    u_acceptance = js(BASE/'unet_acceptance_20261014/SINGLE_RUN_VERIFIED.json')
    need(a_acceptance['status'] == 'verified_real_canonical20_80_and_fixed_final_pair'
         and a_acceptance['canonical_preview_all4_decoded_masks_equal'] is True, 'GTA actual acceptance missing')
    need(u_acceptance['status'] == 'verified_unet_fixed_final_and_full_trajectory', 'UNet actual acceptance missing')
    points, case_points, labels = [], [], {}
    for method, directory in SOURCES.items():
        complete = js(directory/'TRAJECTORY_COMPLETE.json')
        need(complete['case_ids'] == IDS and complete['manifest_sha256'] == MANIFEST
             and complete['smoke'] is False and complete['split'] == 'internal_diagnostic', 'Trajectory identity differs')
        history = lines(directory/'validation_history.jsonl')
        need([x['name'] for x in history] == NAMES, 'Need all20 ledger points')
        curve_rows = list(csv.DictReader(read(directory/'validation_curve.csv').decode('utf-8-sig').splitlines()))
        need([x['name'] for x in curve_rows] == NAMES, 'Curve coverage differs')
        for index, name in enumerate(NAMES):
            epoch = 10*(index+1)
            record = history[index]
            evaluation = directory/name
            marker = js(evaluation/'EVALUATION_COMPLETE.json')
            summary = js(evaluation/'summary.json')
            need(marker['case_ids'] == IDS and marker['manifest_sha256'] == MANIFEST
                 and record['epoch'] == epoch-1 and record['n_cases'] == 4
                 and record['overall'] == summary['overall']
                 and marker['checkpoint_sha256'] == record['checkpoint_sha256'], 'Ledger/marker differs')
            need(marker['summary_sha256'] == hashlib.sha256((evaluation/'summary.json').read_bytes()).hexdigest(), 'Summary bytes differ')
            rows = []
            for case_id in IDS:
                row = js(evaluation/(case_id+'.json'))
                need(row['case_id'] == case_id and row['manifest_sha256'] == MANIFEST
                     and row['metric_grid'] == 'original CT' and row['smoke'] is False
                     and row['formal_final_test'] is False and row['evaluation_split'] == 'internal_diagnostic'
                     and row['checkpoint_epoch'] == epoch-1 and row['checkpoint_sha256'] == record['checkpoint_sha256']
                     and row['assd_definition'] == 'pooled_bidirectional_surface_voxel_distances'
                     and row['surface_connectivity_neighbors'] == 6 and row['surface_area_weighted'] is False,
                     'Case/metric protocol mismatch')
                if case_id in labels: need(labels[case_id] == row['label_sha256'], 'GT source differs')
                labels[case_id] = row['label_sha256']
                need(row['checkpoint_selection'] == ('fixed final' if name == 'final' else 'periodic internal-four diagnostic; not final selection'), 'Selection role changed')
                for metric in METRICS: need(row[metric] is None or math.isfinite(row[metric]), 'Unexpected nonfinite metric')
                values = dict(method=method, completed_epoch=epoch, name=name, case_id=case_id,
                              dice=row['dice'], assd_mm=row['assd_mm'], empty_prediction=row['empty_prediction'])
                rows.append(values); case_points.append(values)
            point = dict(method=method, completed_epoch=epoch, name=name, n_cases=4)
            for metric in METRICS:
                values = [x[metric] for x in rows]
                mean = statistics.mean(values) if all(x is not None for x in values) else None
                point[metric] = mean; point[metric+'_n_defined'] = sum(x is not None for x in values)
                need(close(mean, summary['overall'][metric]['mean'])
                     and close(mean, float(curve_rows[index][metric])), 'Patient mean/summary/CSV disagree')
            points.append(point)
    late = [x for x in points if x['completed_epoch'] >= 160]
    case_late = [x for x in case_points if x['completed_epoch'] >= 160]
    windows, patient_windows = {}, []
    for method in SOURCES:
        window = [x for x in late if x['method'] == method]
        windows[method] = {}
        for metric in METRICS:
            values = [x[metric] for x in window]
            need(all(x is not None for x in values), 'Undefined ASSD: cannot form complete-four late summary')
            windows[method][metric] = dict(min=min(values), max=max(values), range=max(values)-min(values),
                descriptive_timepoint_average=statistics.mean(values),
                epoch160=values[0], epoch200=values[-1], delta200_minus160=values[-1]-values[0],
                successive_changes=[b-a for a,b in zip(values, values[1:])])
        for case_id in IDS:
            rows = [x for x in case_late if x['method'] == method and x['case_id'] == case_id]
            record = dict(method=method, case_id=case_id)
            for metric in METRICS:
                values = [x[metric] for x in rows]
                record.update({metric+'_min': min(values), metric+'_max': max(values),
                               metric+'_epoch160': values[0], metric+'_epoch200': values[-1],
                               metric+'_delta200_minus160': values[-1]-values[0]})
            patient_windows.append(record)
    a140 = next(x for x in points if x['method'] == 'A_GTA' and x['completed_epoch'] == 140)
    afinal = next(x for x in points if x['method'] == 'A_GTA' and x['completed_epoch'] == 200)
    interpretation = [
        '当前完整开发轨迹不提供直接延长A训练的充分证据；A末段为小幅波动而非持续双指标改善。',
        '这不证明完全收敛、过拟合或更多训练必然无效；也不能由U-Net的波动推断A的因果机制。',
        '仍保留固定200轮final为比较终点，不用中间最好点替换，不把五个时间点作为独立重复样本。',
        '同四名患者重复评价，非独立test/folds/seeds；无p值、显著性或未来性能外推。',
        '仅复核已验收的JSON指标、摘要及轨迹账本；原验收负责预测字节/几何和final像素重算，本次不重新运行模型或像素距离。',
        'B(regionloss only, NEW38773)与lambda0(OLD17552)已实际训练启动，结果与A延长无关；本复核不启动、停止或变更任何实验。',
    ]
    result = dict(schema_version=1, status='readonly_accepted_trajectories_reviewed',
        recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        role='four development patients; descriptive repeated-timepoint diagnostic only',
        case_ids=IDS, n_methods=2, n_checkpoints_per_method=20, n_case_records=160,
        reviewed_last_completed_epochs=[160,170,180,190,200], fixed_final_epoch=200,
        checkpoint_selection_changed=False, model_or_GPU_operations=False, server_access=False,
        complete4_means=True, undefined_ASSD_case_records=sum(x['assd_mm'] is None for x in case_points),
        windows=windows, patient_windows=patient_windows,
        A_delta200_minus140={metric: afinal[metric]-a140[metric] for metric in METRICS},
        conclusions=interpretation, GT_source_sha256=labels,
        sources=sources, analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_csv('all20_cohort_points.csv', points)
    write_csv('all20_patient_points.csv', case_points)
    write_csv('last5_cohort_points.csv', late)
    write_csv('last5_patient_points.csv', case_late)
    write_csv('last5_patient_ranges_and_changes.csv', patient_windows)
    with (OUT/'review.json').open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    report = ['A与共同配方U-Net：完整轨迹的独立只读复核', '2026-10-14；内部4病例开发诊断，不是正式测试。', '', *interpretation[:4], '',
              '末五个计划点：160/170/180/190/200；Dice为百分比，ASSD单位mm；每点均值均完整含4病例。',
              '方法 | epoch | Dice% | ASSD mm']
    for row in late: report.append(f"{row['method']} | {row['completed_epoch']} | {100*row['dice']:.9f} | {row['assd_mm']:.9f}")
    report += ['', '末五点的均值范围与160→200变化（时间范围不是置信区间）：']
    for method, stats in windows.items():
        d, a = stats['dice'], stats['assd_mm']
        report.append(f"{method}: Dice均值 {100*d['min']:.9f}–{100*d['max']:.9f}%（范围{100*d['range']:.9f}个百分点）；ASSD {a['min']:.9f}–{a['max']:.9f}mm。160→200：Dice{100*d['delta200_minus160']:+.9f}个百分点，ASSD{a['delta200_minus160']:+.9f}mm。")
    report.append(f"A 140→200：Dice{100*result['A_delta200_minus140']['dice']:+.9f}个百分点，ASSD{result['A_delta200_minus140']['assd_mm']:+.9f}mm；两者均未超过140轮观测，不因此选择140轮。")
    report += ['', '逐例末五点范围；160→200变化（Dice个百分点，ASSD mm）：', '方法 | 病例 | Dice范围% | ΔDice | ASSD范围 | ΔASSD']
    for row in patient_windows:
        report.append(f"{row['method']} | {row['case_id']} | {100*row['dice_min']:.6f}–{100*row['dice_max']:.6f} | {100*row['dice_delta200_minus160']:+.6f} | {row['assd_mm_min']:.6f}–{row['assd_mm_max']:.6f} | {row['assd_mm_delta200_minus160']:+.6f}")
    report += ['', *interpretation[4:], '', '核验范围与出处：',
        '每方法20点×4病例，160份原网格JSON完整；同caseIDs/GT SHA/manifest/pooled ASSD定义/周期与fixedfinal角色；重算患者均值与每点summary、curve、ledger一致。所有160个ASSD有定义，未剔除病例。',
        'GTA实际验收：../gta_canonical_acceptance_20261014/canonical_acceptance_and_pair.json；U-Net实际验收：../unet_acceptance_20261014/SINGLE_RUN_VERIFIED.json。',
        'GTA源：../gta_complete_trajectory_20261014/results/gta_enhanced200x125/internal4_trajectory/{validation_curve.csv,validation_history.jsonl,TRAJECTORY_COMPLETE.json,各点/{病例.json,summary.json,EVALUATION_COMPLETE.json}}。',
        'U-Net源：../unet_complete_trajectory_20261014/results/unet_recipe200x125/internal4_trajectory/（同名文件结构）。',
        '逐项源SHA、全20点/逐例计算见review.json、all20_*.csv；末五点明细见last5_*.csv。原始记录、冻结代码/图、已有文档均未修改。']
    with (OUT/'review.txt').open('x', encoding='utf-8') as stream: stream.write('\n'.join(report)+'\n')
    print(json.dumps(dict(status=result['status'], windows=windows, A_delta200_minus140=result['A_delta200_minus140'], patient_windows=patient_windows), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
