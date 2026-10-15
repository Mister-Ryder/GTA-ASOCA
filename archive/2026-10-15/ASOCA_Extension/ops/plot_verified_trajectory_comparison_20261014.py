"""Render actual fixed-budget A/U trajectories; no training or new measurements."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BASE = Path(__file__).resolve().parents[1] / 'fidelity_v2/server_records'
SOURCES = {
    'A': BASE / 'gta_complete_trajectory_20261014/results/gta_enhanced200x125/internal4_trajectory/validation_curve.csv',
    'U': BASE / 'unet_complete_trajectory_20261014/results/unet_recipe200x125/internal4_trajectory/validation_curve.csv',
}
OUT = BASE / 'trajectory_comparison_figure_20261014'
CASES = ['Diseased_10', 'Diseased_20', 'Normal_12', 'Normal_3']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    acceptance = BASE / 'gta_canonical_acceptance_20261014/canonical_acceptance_and_pair.json'
    accepted = json.loads(acceptance.read_text())
    assert accepted['status'] == 'verified_real_canonical20_80_and_fixed_final_pair'
    assert accepted['case_ids'] == CASES and accepted['checkpoint_selection'] == 'fixed final'
    source_hashes = {str(p.relative_to(BASE)): sha(p) for p in [*SOURCES.values(), acceptance]}
    data, exported = {}, []
    for label, path in SOURCES.items():
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 20 and [int(r['completed_epoch']) for r in rows] == list(range(10, 201, 10))
        assert [int(r['epoch']) for r in rows] == list(range(9, 200, 10))
        assert all(int(r['n_cases']) == 4 for r in rows) and rows[-1]['name'] == 'final'
        assert all(math.isfinite(float(r[k])) and float(r[k]) > 0 for r in rows for k in ('dice', 'assd_mm'))
        data[label] = rows
        exported += [dict(run_label=label, completed_epoch=int(r['completed_epoch']), checkpoint=r['name'],
                          dice_percent=100 * float(r['dice']), assd_mm=float(r['assd_mm']), n_cases=4) for r in rows]
    assert math.isclose(float(data['A'][-1]['dice']), .8318539004693745, abs_tol=1e-12)
    assert math.isclose(float(data['A'][-1]['assd_mm']), .5678251259083507, abs_tol=1e-12)
    assert math.isclose(float(data['U'][-1]['dice']), .818455511253775, abs_tol=1e-12)
    assert math.isclose(float(data['U'][-1]['assd_mm']), .7492326552685586, abs_tol=1e-12)
    OUT.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), layout='constrained')
    styles = {'A': dict(color='#0072B2', marker='o', label='A: GTA (warm start)'),
              'U': dict(color='#D55E00', marker='s', label='U: U-Net (cold start)')}
    for label, rows in data.items():
        for index, (ax, metric, late) in enumerate(zip(axes.flat, ['dice', 'assd_mm'] * 2, [False, False, True, True])):
            chosen = [r for r in rows if not late or int(r['completed_epoch']) >= 140]
            x = [int(r['completed_epoch']) for r in chosen]
            y = [float(r[metric]) * (100 if metric == 'dice' else 1) for r in chosen]
            ax.plot(x, y, linewidth=1.5, markersize=3.5, **styles[label])
            if late:
                text = f'{y[-1]:.2f}%' if metric == 'dice' else f'{y[-1]:.3f}'
                ax.annotate(text, (x[-1], y[-1]), xytext=(5, 0), textcoords='offset points',
                            color=styles[label]['color'], fontsize=8, va='center')
    for index, ax in enumerate(axes.flat):
        ax.set_title(['(a) Dice: full trajectory', '(b) ASSD: full trajectory',
                      '(c) Dice: epochs 140–200', '(d) ASSD: epochs 140–200'][index], loc='left')
        ax.set_xlabel('Completed training epoch (125 updates/epoch)')
        ax.set_ylabel('Dice (%)' if index % 2 == 0 else 'ASSD (mm)')
        ax.grid(axis='y', color='#d9d9d9', linewidth=.6, alpha=.7)
        if index < 2:
            ax.set_xlim(5, 205)
            ax.set_xticks([10, 50, 100, 150, 200])
        else:
            ax.set_xlim(137, 215)
            ax.set_xticks([140, 160, 180, 200])
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel('ASSD (mm; logarithmic scale)')
    axes[0, 1].set_yticks([.5, 1, 2, 5, 10])
    axes[0, 1].yaxis.set_major_formatter(ScalarFormatter())
    axes[0, 0].legend(loc='lower right', frameon=False, fontsize=8)
    fig.suptitle('Internal-four development trajectories · fixed 25,000 segmentation updates', fontsize=11)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / ('A_U_verified_trajectories.' + suffix))
    plt.close(fig)
    with (OUT / 'plotted_measurements.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0])); writer.writeheader(); writer.writerows(exported)
    notes = ('Four development patients, repeated at 20 scheduled checkpoints; no independent test or significance claim.\n'
             'A includes an airway backbone warm start and 5,120 classifier-pretraining updates; U is cold start. Both use decoder deep supervision.\n'
             'Final endpoints remain epoch200. Late panels show every scheduled point from140 onward; they do not select an earlier winner.\n'
             'Points are recorded cohort means. Lines only join adjacent points; no extra/interpolated measurements or confidence bands.\n'
             'Panel(b) uses a logarithmic ASSD axis; panel(d) is linear. Patient SD is not shown as a confidence interval.\n'
             'Physical pooled boundary-voxel ASSD follows the frozen evaluator. All four distances are defined at all displayed points.\n'
             'The B and lambda0 controls are still training and are not plotted.\n')
    (OUT / 'figure_notes.txt').write_text(notes, encoding='utf-8')
    assert source_hashes == {str(p.relative_to(BASE)): sha(p) for p in [*SOURCES.values(), acceptance]}
    manifest = dict(role='FIGURE_OF_ACTUAL_VERIFIED_DEVELOPMENT_RECORDS_NOT_NEW_EXPERIMENT',
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), case_ids=CASES,
                    n_recorded_points_per_run=20, source_sha256=source_hashes,
                    checkpoint_selection='fixed final', formal_final_test=False, source_files_unchanged=True,
                    plotter_sha256=sha(Path(__file__)), artifacts={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='rendered_actual_A_U_curves', output=str(OUT), measured_points=40)))


if __name__ == '__main__':
    main()
