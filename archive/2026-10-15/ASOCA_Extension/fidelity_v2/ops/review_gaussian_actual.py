"""Independent stdlib review of downloaded Gaussian diagnostic artifacts.

No model/cache reads, network, optimizer, source mutations or inference. This
checks artifact identities and reconstructible aggregates, and records what
cannot be reconstructed without per-voxel predictions/moments.
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isclose(a, b, rel_tol=2e-9, abs_tol=2e-15), (a, b)


def grad_stats(energy, dot, margin_dot):
    sc, sm, gc, gm = energy
    return dict(
        segmentation_common_energy=sc,
        segmentation_margin_energy=sm,
        gaussian_common_energy=gc,
        gaussian_margin_energy=gm,
        gaussian_common_fraction=gc/(gc+gm) if gc+gm else None,
        gaussian_to_segmentation_norm_ratio=math.sqrt((gc+gm)/(sc+sm)) if sc+sm else None,
        gaussian_to_segmentation_margin_norm_ratio=math.sqrt(gm/sm) if sm else None,
        full_cosine=dot/math.sqrt((sc+sm)*(gc+gm)) if (sc+sm)*(gc+gm) else None,
        margin_cosine=margin_dot/math.sqrt(sm*gm) if sm*gm else None,
    )


def distribution(values):
    values = [x for x in values if x is not None]
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values)) if values else dict(n=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--diagnostic-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.artifacts
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    plan = json.loads((folder/'diagnostic_plan.json').read_text(encoding='utf-8'))
    bank = json.loads((folder/'fixed_sampling_bank.json').read_text(encoding='utf-8'))
    complete = json.loads((folder/'DIAGNOSTIC_COMPLETE.json').read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in (folder/'patch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert summary['provenance'] == plan
    assert complete['status'] == 'completed' and complete['n_patches'] == len(rows) == len(bank['samples']) == 56
    assert complete['parameter_updates'] is False and complete['outer8_used'] is False
    for name, key in [('summary.json', 'summary_sha256'), ('fixed_sampling_bank.json', 'fixed_sampling_bank_sha256'),
                      ('patch_metrics.jsonl', 'patch_metrics_sha256')]:
        assert sha(folder/name) == complete[key], name
    assert sha(args.diagnostic_source) == plan['diagnostic_script_sha256']
    assert sha(args.manifest) == plan['input_sha256']['manifest']
    source_digest = hashlib.sha256()
    source_files = plan['source_files_sha256']
    for name in sorted(source_files):
        file = args.code/name
        assert sha(file) == source_files[name], name
        source_digest.update(name.encode()); source_digest.update(file.read_bytes())
    assert source_digest.hexdigest() == plan['source_sha256']
    assert bank['source_sha256'] == plan['source_sha256']
    assert bank['manifest_sha256'] == plan['input_sha256']['manifest']
    assert plan['fixed_sampling_bank_sha256'] == complete['fixed_sampling_bank_sha256']
    assert plan['parameter_updates'] is False and plan['optimizer_created'] is False
    assert plan['device_for_explicit_diagnostic'] == 'cpu' and plan['checkpoint_epoch'] == 69
    assert plan['sampling'] == 'standard' and plan['gaussian_lambda'] == 10
    assert len({row['index'] for row in rows}) == 56
    patient_counts = collections.Counter(row['case_id'] for row in rows)
    assert set(patient_counts) == set(plan['training_patients']) == set(bank['training_patients'])
    assert len(patient_counts) == 28 and set(patient_counts.values()) == {2}
    assert not (set(patient_counts) & set(plan['excluded_internal4']))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    assert not (set(patient_counts) & set(manifest['folds'][plan['fold']]['val']))
    regions = list(summary['regions'])
    energy_keys = ['segmentation_common_energy', 'segmentation_margin_energy',
                   'gaussian_common_energy', 'gaussian_margin_energy']
    pooled = {}
    for row, expected in zip(rows, bank['samples']):
        assert row['sample_identity'] == expected
        assert row['index'] == expected['index'] and row['case_id'] == expected['case_id']
        assert row['sampling_stratum'] == expected['sampling_stratum']['value']
        assert row['regions']['all']['n_voxels'] == 128**3
        fg = row['regions']['foreground']['n_voxels']
        bg = row['regions']['background']['n_voxels']
        assert fg+bg == 128**3
        assert sum(row['regions'][k]['n_voxels'] for k in ('outside_0_1mm', 'outside_1_3mm', 'background_gt3mm')) == bg
        close(row['foreground_fraction'], fg/(128**3))
        close(row['regions']['all']['mse_constant05'], row['field']['mean_squared']/4)
        for region in regions:
            r = row['regions'][region]; n = r['n_voxels']
            assert (r['mse'] is None) == (n == 0)
            assert (r['mse_constant05'] is None) == (n == 0)
            if n and region not in ('all', 'foreground'):
                close(r['mse_constant05'], 0.)
            g = row['gradient_regions'][region]
            energy = [g[k] for k in energy_keys]
            assert all(math.isfinite(v) and v >= 0 for v in energy)
            sc, sm, gc, gm = energy
            dot = g['full_cosine']*math.sqrt((sc+sm)*(gc+gm)) if g['full_cosine'] is not None else 0.
            margin_dot = g['margin_cosine']*math.sqrt(sm*gm) if g['margin_cosine'] is not None else 0.
            rebuilt = grad_stats(energy, dot, margin_dot)
            for key in rebuilt:
                close(rebuilt[key], g[key])
    field_mean = sum(row['field']['mean'] for row in rows)/56
    field_second = sum(row['field']['mean_squared'] for row in rows)/56
    close(summary['bank_optimal_constant_target'], .5+.5*field_mean)
    for region in regions:
        count = sum(row['regions'][region]['n_voxels'] for row in rows)
        error = sum(row['regions'][region]['n_voxels']*(row['regions'][region]['mse'] or 0) for row in rows)
        constant = sum(row['regions'][region]['n_voxels']*(row['regions'][region]['mse_constant05'] or 0) for row in rows)
        expected = summary['regions'][region]
        assert expected['n_voxels'] == count
        close(error/count if count else None, expected['mse'])
        close(10*error/count if count else None, expected['weighted_mse'])
        close(constant/count if count else None, expected['mse_constant05'])
        fm = field_mean if region == 'all' else field_mean*(56*128**3)/count if region == 'foreground' else 0.
        fs = 4*constant/count
        close(expected['field_mean'], fm)
        close(expected['field_mean_squared'], fs)
        close(expected['mse_bank_optimal_constant'], .25*(fs-2*field_mean*fm+field_mean**2))
        energy = [sum(row['gradient_regions'][region][k] for row in rows) for k in energy_keys]
        dot = 0.; margin_dot = 0.
        for row in rows:
            g = row['gradient_regions'][region]
            sc, sm, gc, gm = [g[k] for k in energy_keys]
            dot += (g['full_cosine'] or 0)*math.sqrt((sc+sm)*(gc+gm))
            margin_dot += (g['margin_cosine'] or 0)*math.sqrt(sm*gm)
        rebuilt = grad_stats(energy, dot, margin_dot)
        for key, value in rebuilt.items():
            close(value, summary['gradient_regions'][region][key])
        pooled[region] = dict(n_voxels=count, squared_error=error, gradient_rebuilt=rebuilt)
    assert pooled['all']['n_voxels'] == 56*128**3
    assert pooled['foreground']['n_voxels']+pooled['background']['n_voxels'] == pooled['all']['n_voxels']
    close(pooled['foreground']['squared_error']+pooled['background']['squared_error'], pooled['all']['squared_error'])
    close(sum(pooled[k]['squared_error'] for k in ('outside_0_1mm','outside_1_3mm','background_gt3mm')),
          pooled['background']['squared_error'])
    for key in energy_keys:
        close(pooled['foreground']['gradient_rebuilt'][key]+pooled['background']['gradient_rebuilt'][key],
              pooled['all']['gradient_rebuilt'][key])
        close(sum(pooled[k]['gradient_rebuilt'][key] for k in ('outside_0_1mm','outside_1_3mm','background_gt3mm')),
              pooled['background']['gradient_rebuilt'][key])
    report = dict(
        status='PASS_RECONSTRUCTIBLE_ARTIFACT_CHECKS',
        role='independent downloaded-artifact review; not model-performance validation',
        review_script_sha256=sha(Path(__file__)), artifact_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        source_sha256=plan['source_sha256'], n_training_patients=28, n_patches=56,
        sampling_strata=dict(collections.Counter(row['sampling_stratum'] for row in rows)),
        empty_patches=sum(row['regions']['foreground']['n_voxels']==0 for row in rows),
        foreground_fraction=pooled['foreground']['n_voxels']/pooled['all']['n_voxels'],
        global_mse_reduction_vs_constant05=1-summary['regions']['all']['mse']/summary['regions']['all']['mse_constant05'],
        foreground_mse_reduction_vs_constant05=1-summary['regions']['foreground']['mse']/summary['regions']['foreground']['mse_constant05'],
        background_share_of_gaussian_squared_error=pooled['background']['squared_error']/pooled['all']['squared_error'],
        patch_distributions={
            'foreground_fraction': distribution([r['foreground_fraction'] for r in rows]),
            'gaussian_common_fraction': distribution([r['gradient_regions']['all']['gaussian_common_fraction'] for r in rows]),
            'gaussian_norm_ratio': distribution([r['gradient_regions']['all']['gaussian_to_segmentation_norm_ratio'] for r in rows]),
            'gaussian_margin_norm_ratio': distribution([r['gradient_regions']['all']['gaussian_to_segmentation_margin_norm_ratio'] for r in rows]),
            'gradient_margin_cosine': distribution([r['gradient_regions']['all']['margin_cosine'] for r in rows]),
        },
        patches_with_common_energy_majority=sum((r['gradient_regions']['all']['gaussian_common_fraction'] or 0)>.5 for r in rows),
        patches_with_negative_margin_cosine=sum((r['gradient_regions']['all']['margin_cosine'] or 0)<0 for r in rows),
        rebuilt_regions=pooled,
        unverified_scope=[
            'No local epoch070 checkpoint bytes or cache voxels were supplied to this review: recorded checkpoint SHA and sample content SHA are provenance, not independently rehashed here.',
            'No local config bytes are supplied in this artifact folder: its SHA was verified by the diagnostic on server, not independently rehashed here.',
            'Pooled Pearson correlations cannot be reconstructed from stored per-patch coefficients alone: missing per-patch moment sums; source formula reviewed only.',
            'Cosine dot sums reconstructed from stored per-patch cosine and energies, not raw gradient tensors.',
            'All diagnostic evaluations use training patches at a single partial-stage checkpoint in CPU eval mode: no final Dice, ASSD, connectivity or benefit claim.'
        ]
    )
    assert not args.output.exists(), 'Use a fresh review output file'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rebuilt_regions','artifact_sha256')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
