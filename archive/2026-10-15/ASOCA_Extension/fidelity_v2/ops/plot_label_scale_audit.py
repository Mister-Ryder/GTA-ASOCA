"""Source-backed static figure for a completed training-only label-scale audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    marker = json.loads((a.audit / 'LABEL_SCALE_AUDIT_COMPLETE.json').read_text(encoding='utf-8'))
    assert marker['status'] == 'completed' and marker['input_patches'] == 56
    for name, expected in marker['artifacts_sha256'].items():
        assert sha(a.audit / name) == expected
    data = json.loads((a.audit / 'summary.json').read_text(encoding='utf-8'))
    if a.output.exists() and any(a.output.iterdir()):
        raise RuntimeError('Fresh plot directory required')
    a.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    rows = []
    medians = [data['training_tree_extent_summary'][str(i)]['median'] for i in (2, 1, 0)]
    ax = axes[0, 0]
    ax.bar(['X', 'Y', 'Z'], medians, color='#315c96', width=.6)
    ax.axhline(64, color='#b25428', linestyle='--', label='Input patch edge: 64 mm')
    ax.set(title='A  Full coronary GT bounding box (28 patients)', ylabel='Median extent (mm)', ylim=(0, 135))
    ax.legend(loc='upper right', fontsize=9)
    for i, value in enumerate(medians):
        ax.text(i, value + 2, f'{value:g}', ha='center')
        rows.append({'panel': 'A', 'item': ['X', 'Y', 'Z'][i], 'value': value, 'denominator': 28, 'unit': 'median_mm'})
    scales = [data['roi_scales'][str(n)] for n in (64, 96, 128)]
    labels = [f"{s['physical_side_mm']:g}" for s in scales]
    nonempty = [s['n_nonempty'] for s in scales]
    empty = [s['n_empty'] for s in scales]
    ax = axes[0, 1]
    ax.bar(labels, nonempty, color='#315c96', label='Contains coronary foreground')
    ax.bar(labels, empty, bottom=nonempty, color='#ddb784', label='Empty ROI')
    ax.set(title='B  Same-center supervision ROIs', ylabel='Fixed patches (total 56)', xlabel='ROI edge (mm)', ylim=(0, 72))
    ax.legend(loc='upper left', fontsize=9)
    for i, (n, e) in enumerate(zip(nonempty, empty)):
        ax.text(i, n / 2, str(n), ha='center', color='white', fontweight='bold')
        ax.text(i, n + e / 2, str(e), ha='center')
        rows.extend([{'panel': 'B', 'item': labels[i] + '_nonempty', 'value': n, 'denominator': 56, 'unit': 'count'},
                     {'panel': 'B', 'item': labels[i] + '_empty', 'value': e, 'denominator': 56, 'unit': 'count'}])
    ax = axes[1, 0]
    mixed = [s['multiple_generation_bins_present'] for s in scales]
    values = [100 * x['fraction'] for x in mixed]
    ax.bar(labels, values, color='#4b8b77', width=.6)
    ax.set(title='C  Multiple depth bins within nonempty ROI', ylabel='Nonempty ROIs (%)', xlabel='ROI edge (mm)', ylim=(0, 105))
    for i, (v, x) in enumerate(zip(values, mixed)):
        ax.text(i, v + 2, f"{x['count']}/{x['denominator']} ({v:.1f}%)", ha='center', fontsize=10)
        rows.append({'panel': 'C', 'item': labels[i], 'value': x['count'], 'denominator': x['denominator'], 'unit': 'count'})
    ax = axes[1, 1]
    alternatives = scales[-1]['aggregation_label_disagreement']
    names = ['volume', 'equal_branch', 'centerline_voxel']
    values = [100 * alternatives[n]['fraction'] for n in names]
    ax.bar(['All-voxel mode', 'Equal-branch mode', 'Centerline-voxel mode'], values, color='#8565a6', width=.6)
    ax.set(title='D  Label differs from dominant-branch target', ylabel='Nonempty 64 mm input patches (%)', ylim=(0, 105))
    ax.tick_params(axis='x', labelsize=9)
    for i, (name, value) in enumerate(zip(names, values)):
        x = alternatives[name]
        ax.text(i, value + 2, f"{x['different_from_current_dominant_branch_label']}/{x['denominator']} ({value:.1f}%)", ha='center', fontsize=10)
        rows.append({'panel': 'D', 'item': name, 'value': x['different_from_current_dominant_branch_label'], 'denominator': x['denominator'], 'unit': 'count'})
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Training-label scale audit: unchanged 128-cubed CT context', fontsize=16, y=.985)
    fig.text(.5, .018, 'Training-only fixed bank; no model updates or performance claim. Panel C has different nonempty denominators.\nDepth bins and centerline voting are geometric proxies; two samples per patient are not independent patient replicates.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95), h_pad=2.5, w_pad=2)
    for extension in ('png', 'svg'):
        fig.savefig(a.output / ('label_scale_audit.' + extension), dpi=180)
    plt.close(fig)
    with (a.output / 'figure_values.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['panel', 'item', 'value', 'denominator', 'unit'])
        writer.writeheader(); writer.writerows(rows)
    manifest = {'scope': 'Actual completed training28 label audit, no segmentation results',
                'inputs': {str((a.audit / n).resolve()): sha(a.audit / n) for n in ('summary.json', 'LABEL_SCALE_AUDIT_COMPLETE.json')},
                'script_sha256': sha(__file__), 'artifacts': {n: sha(a.output / n) for n in ('label_scale_audit.png', 'label_scale_audit.svg', 'figure_values.csv')}}
    (a.output / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(a.output), 'actual_values': len(rows)}))


if __name__ == '__main__':
    main()
