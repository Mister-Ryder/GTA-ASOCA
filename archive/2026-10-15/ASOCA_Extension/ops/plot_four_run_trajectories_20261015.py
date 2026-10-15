"""Plot recorded internal-four trajectories. No models, SSH, or new inference."""
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BASE = Path(__file__).resolve().parents[1] / "fidelity_v2/server_records"
OUT = BASE / "four_run_trajectory_comparison_20261015"
CASES = ["Diseased_10", "Diseased_20", "Normal_12", "Normal_3"]
RUNS = {
    "A": ("gta_complete_trajectory_20261014", "gta_enhanced200x125"),
    "B": ("gaussian_region_balance_complete_trajectory_20261014", "gta_region_balance200x125_20261014"),
    "lambda0": ("gaussian_lambda0_complete_trajectory_20261014", "gta_gaussian_lambda0_200x125_20261014"),
    "U": ("unet_complete_trajectory_20261014", "unet_recipe200x125"),
}
STYLES = {
    "A": dict(color="#0072B2", marker="o", label="A: GTA, original field loss (warm start)"),
    "B": dict(color="#009E73", marker="^", label="B: GTA, region-balanced field loss (warm start)"),
    "lambda0": dict(color="#AA4499", marker="D", label=r"$\lambda$0: GTA, field-loss weight 0 (warm start)"),
    "U": dict(color="#D55E00", marker="s", label="U: U-Net (cold start)"),
}
SOURCE_HASHES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path):
    digest = sha(path)
    SOURCE_HASHES[path.relative_to(BASE).as_posix()] = digest
    return digest


def read(path):
    bind(path)
    return json.loads(path.read_text(encoding="utf-8"))


def close(x, y):
    assert math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-12), (x, y)


def write_csv(name, rows):
    with (OUT / name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    all_rows, final_cases, data, completion = [], [], {}, {}
    for label, (capture_name, result_name) in RUNS.items():
        capture = BASE / capture_name
        collection = read(capture / "collection_manifest.json")
        inventory = {r["relative_path"]: r for r in collection["files"]}
        directory = capture / "results" / result_name / "internal4_trajectory"

        def captured(path):
            entry = inventory[path.relative_to(capture).as_posix()]
            assert path.stat().st_size == entry["bytes"] and bind(path) == entry["sha256"]

        curve = directory / "validation_curve.csv"
        ledger_path = directory / "validation_history.jsonl"
        captured(curve)
        captured(ledger_path)
        with curve.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        ledger = [json.loads(s) for s in ledger_path.read_text(encoding="utf-8").splitlines() if s]
        assert len(rows) == len(ledger) == 20
        assert [int(r["completed_epoch"]) for r in rows] == list(range(10, 201, 10))
        assert [int(r["epoch"]) for r in rows] == list(range(9, 200, 10))
        assert rows[-1]["name"] == "final"
        for row, event in zip(rows, ledger):
            assert int(row["n_cases"]) == event["n_cases"] == 4
            assert row["name"] == event["name"] and int(row["epoch"]) == event["epoch"]
            point = directory / row["name"]
            summary_path = point / "summary.json"
            case_ledger = point / "case_metrics.jsonl"
            captured(summary_path)
            captured(case_ledger)
            summary = read(summary_path)["overall"]
            patients = [json.loads(s) for s in case_ledger.read_text(encoding="utf-8").splitlines() if s]
            assert [r["case_id"] for r in patients] == CASES
            assert summary["n_cases"] == 4 and summary["empty_predictions"] == 0
            for patient in patients:
                patient_path = point / (patient["case_id"] + ".json")
                captured(patient_path)
                assert read(patient_path) == patient
                assert patient["checkpoint_sha256"] == event["checkpoint_sha256"]
                assert patient["checkpoint_epoch"] == int(row["epoch"])
                assert patient["evaluation_split"] == "internal_diagnostic"
                assert patient["formal_final_test"] is False and patient["empty_prediction"] is False
                assert patient["metric_grid"] == "original CT" and patient["metric_protocol_version"] == 2
                assert patient["assd_definition"] == "pooled_bidirectional_surface_voxel_distances"
                assert patient["surface_connectivity_neighbors"] == 6 and patient["surface_area_weighted"] is False
                for metric in ("dice", "assd_mm"):
                    assert math.isfinite(patient[metric]) and patient[metric] > 0
                if row["name"] == "final":
                    final_cases.append(dict(run_label=label, case_id=patient["case_id"],
                                            dice_percent=100 * patient["dice"], assd_mm=patient["assd_mm"]))
            for metric in ("dice", "assd_mm"):
                assert summary[metric]["n_valid"] == 4 and summary[metric]["n_undefined"] == 0
                mean = statistics.mean(p[metric] for p in patients)
                close(mean, row[metric])
                close(mean, summary[metric]["mean"])
                close(mean, event["overall"][metric]["mean"])
            all_rows.append(dict(run_label=label, completed_epoch=int(row["completed_epoch"]),
                                 checkpoint=row["name"], dice_percent=100 * float(row["dice"]),
                                 assd_mm=float(row["assd_mm"]), n_cases=4,
                                 checkpoint_selection="fixed final" if row["name"] == "final" else "scheduled diagnostic"))
        data[label] = rows
        completion[label] = dict(recorded_points=20, patient_evaluations=80, all_primary_distances_defined=True,
                                 collection_manifest_sha256=sha(capture / "collection_manifest.json"))
    return data, all_rows, final_cases, completion


def acceptances():
    paths = {
        "A": BASE / "gta_canonical_acceptance_20261014/canonical_acceptance_and_pair.json",
        "U": BASE / "unet_acceptance_20261014/SINGLE_RUN_VERIFIED.json",
        "B": BASE / "B_fixed_final4_acceptance_20261014/acceptance.json",
        "lambda0": BASE / "lambda0_fixed_final4_acceptance_20261014/acceptance.json",
    }
    status = {}
    for label, path in paths.items():
        if not path.exists():
            status[label] = dict(independent_acceptance="pending", receipt=None)
            continue
        receipt = read(path)
        expected = {"A": "verified_real_canonical20_80_and_fixed_final_pair",
                    "U": "verified_unet_fixed_final_and_full_trajectory"}.get(
                        label, "verified_actual_completed_control20_80_and_fixed_final4")
        assert receipt["status"] == expected and receipt["case_ids"] == CASES
        assert receipt["checkpoint_selection"] == "fixed final" and receipt["formal_final_test"] is False
        if label in ("B", "lambda0"):
            assert receipt["kind"] == label
            capture = BASE / RUNS[label][0]
            assert receipt["control_collection_manifest"]["sha256"] == sha(capture / "collection_manifest.json")
        status[label] = dict(independent_acceptance="accepted", receipt=path.relative_to(BASE).as_posix(),
                             receipt_sha256=sha(path))
    return status


def draw(data, acceptance):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.235, top=.80, hspace=.41, wspace=.25)
    for label, rows in data.items():
        for index, ax in enumerate(axes.flat):
            late = index >= 2
            metric = "dice" if index % 2 == 0 else "assd_mm"
            points = [r for r in rows if not late or int(r["completed_epoch"]) >= 140]
            x = [int(r["completed_epoch"]) for r in points]
            y = [float(r[metric]) * (100 if metric == "dice" else 1) for r in points]
            ax.plot(x, y, linewidth=1.3, markersize=3.5, **STYLES[label])
            ax.scatter([x[-1]], [y[-1]], s=45, facecolors="white", edgecolors=STYLES[label]["color"], zorder=4)
    for index, ax in enumerate(axes.flat):
        ax.set_title(["(a) Dice: full trajectory", "(b) ASSD: full trajectory",
                      "(c) Dice: epochs 140–200", "(d) ASSD: epochs 140–200"][index], loc="left")
        ax.set_xlabel("Completed epoch (125 updates/epoch)")
        ax.set_ylabel("Dice (%)" if index % 2 == 0 else "ASSD (mm)")
        ax.grid(axis="y", color="#d9d9d9", linewidth=.6, alpha=.75)
        ax.axvline(200, color="#555555", linewidth=.7, linestyle=":", zorder=0)
        ax.set_xlim((5, 205) if index < 2 else (137, 203))
        ax.set_xticks([10, 50, 100, 150, 200] if index < 2 else [140, 160, 180, 200])
    axes[0, 0].set_ylim(60, 85)
    axes[1, 0].set_ylim(79.8, 83.6)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("ASSD (mm; log scale)")
    axes[0, 1].set_yticks([.5, 1, 2, 5, 10])
    axes[0, 1].yaxis.set_major_formatter(ScalarFormatter())
    axes[0, 1].set_ylim(.42, 12)
    axes[1, 1].set_ylim(.42, .95)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .935), ncol=2,
               fontsize=8.5, frameon=False, columnspacing=1.7, handlelength=2)
    fig.suptitle("Four internal-development patients · 20 scheduled checkpoints · fixed 25,000 updates",
                 x=.53, y=.985, fontsize=11)
    final_text = "Fixed epoch200 — " + "; ".join(
        f"{label}: {100 * float(data[label][-1]['dice']):.3f}% / {float(data[label][-1]['assd_mm']):.3f} mm"
        for label in RUNS)
    fig.text(.075, .113, final_text, fontsize=8.7)
    fig.text(.075, .078, "Recorded means; lines join adjacent measurements. No smoothing, new points, confidence intervals, or winner claim.", fontsize=8)
    fig.text(.075, .052, "GTA adds 5,120 classifier-pretraining updates and airway initialization. U / λ0: OLD hardware; A / B: NEW hardware.", fontsize=8)
    flag = "All four fixed-final CPU acceptances recorded." if all(r["independent_acceptance"] == "accepted" for r in acceptance.values()) else "B / λ0 independent CPU acceptance pending; recorded metrics are provisional."
    fig.text(.075, .026, flag + " Same four development cases; no independent test. Dotted line / open circle: fixed epoch200.", fontsize=8)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / ("four_run_internal4_trajectories." + suffix))
    plt.close(fig)


def main():
    data, measurements, cases, completion = collect()
    acceptance = acceptances()
    OUT.mkdir(parents=True, exist_ok=False)
    write_csv("plotted_measurements.csv", measurements)
    write_csv("fixed_final_patient_measurements.csv", cases)
    draw(data, acceptance)
    by_run = {label: {r["case_id"]: r for r in cases if r["run_label"] == label} for label in RUNS}
    differences = [dict(comparison="B_minus_" + reference, case_id=case,
                        dice_difference_pp=by_run["B"][case]["dice_percent"] - by_run[reference][case]["dice_percent"],
                        assd_difference_mm=by_run["B"][case]["assd_mm"] - by_run[reference][case]["assd_mm"])
                   for reference in ("A", "lambda0", "U") for case in CASES]
    write_csv("fixed_final_paired_differences.csv", differences)
    notes = [
        "Four-run internal-four development comparison; 2026-10-15. No new experiment or model forward.",
        "Sources: actual collected cohort curves, 80 checkpoint ledgers/summaries and 320 patient JSON records; collection size/hash checks passed for every plotted source.",
        "80 recorded cohort points = four runs ×20 scheduled checkpoints, each on the same four patients. These are not 80 independent runs or 320 independent patients.",
        "Primary ASSD: pooled bidirectional 6-neighbour boundary-voxel distances in physical mm on original CT grids; not surface-area-weighted and not a connectivity measure. All320 displayed patient distances are defined; no cases omitted.",
        "Panel(b) is logarithmic; panel(d) is linear. Dice axes are truncated as labelled. Lines connect recorded points only; no smoothing/interpolated measurements, best-epoch replacement, confidence intervals or significance tests.",
        "All final values use the fixed epoch200 checkpoint. A/B/lambda0 use airway backbone initialization plus 5120 classifier-pretraining updates; U is cold start. Decoder DS is shared, but architectures/capacity/objectives differ.",
        "U and lambda0 trained on OLD hardware; A/B on NEW. Hardware/driver and original U runtime differ; historical A actual theta0 was not saved. This is not strict single-factor attribution.",
        "Independent fixed-final acceptance state: " + json.dumps(acceptance, ensure_ascii=False),
        "B vs A fixed final: Dice−0.058166 percentage points, ASSD−0.085691mm (15.09% lower). This is a tradeoff, not proven Dice equivalence or a dual-metric win.",
        "B vs A ASSD improves in D10,D20,N3 and worsens in N12; Dice improves only in D20. B vs lambda0: ASSD improves in three patients; Dice improves in two. See paired CSV for every patient.",
        "B final three scheduled Dice values:83.0483/83.2476/83.1272%; ASSD:.487954/.476484/.482134mm. Lambda0:82.9843/82.8512/83.1242%; .547107/.660572/.599177mm. This repeated-time pattern is descriptive, not independent replication or proof more epochs would help.",
        "D20 remains worse for B than U:80.7318%/.856034mm versus82.6248%/.541889mm. The hard-case deficit is unresolved.",
        "The human selected the next step of checking existing B on the previously exposed eight development cases. This artifact neither freezes a candidate nor launches inference, and reads no outer-eight outputs.",
        "Original A/U figure and helpers, frozen method source/configuration, and all inputs remain unchanged. No SSH, GPU, models, U8 outputs, raw data, or upload were used by this plotter.",
    ]
    (OUT / "data_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    assert all(sha(BASE / path) == expected for path, expected in SOURCE_HASHES.items())
    manifest = dict(schema_version=1, role="FIGURE_OF_ACTUAL_INTERNAL4_RECORDS_NOT_NEW_EXPERIMENT",
                    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), case_ids=CASES,
                    measured_cohort_points=80, patient_evaluations=320, final_patient_rows=16,
                    fixed_final_epoch=200, acceptance=acceptance, runs=completion,
                    formal_final_test=False, source_files_unchanged=True, source_sha256=SOURCE_HASHES,
                    plotter_sha256=sha(Path(__file__)),
                    artifacts={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()})
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="rendered_actual_four_run_curves", output=str(OUT),
                          measured_cohort_points=80, patient_evaluations=320, acceptance=acceptance)))


if __name__ == "__main__":
    main()
