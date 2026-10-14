"""Plot observed GTA development histories without inventing validation points.

Example: --run prior=/project/results/prior/training --run gta=/project/results/gta/training
         --validation gta=/project/results/gta/validation_history.jsonl --output /project/curves
Epoch JSON fields are zero-based unless --epoch-base 1 is explicitly supplied.
Requires matplotlib only when rendering; --tables-only writes source-backed CSV/notes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


HEADS = ("generation", "structure", "location")
RAW_KEYS = {"classification", "segmentation", "gaussian", "guidance_mse", "guidance_map", "structure_correctness",
            "generation_focal", "structure_focal", "location_focal", "encoder_decoder_warmup", "cldice"}
METADATA = {"epoch", "epoch_1based", "step", "iteration", "seconds", "gpu_peak_GiB", "iterations"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def named_path(value):
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        name = Path(path).parent.name if Path(path).is_file() else Path(path).name
    if not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=PATH, or an existing run path")
    return name, Path(path)


def read_records(path, notes, log=False):
    records = []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if log and not line.startswith("{"):
            start = line.find("{")
            if start < 0:
                continue
            line = line[start:]
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if log or index == len(lines) - 1 and not text.endswith("\n"):
                notes.append(f"Ignored non-JSON/incomplete line {index + 1} in {path.name}")
                continue
            raise ValueError(f"Malformed completed JSONL at {path}:{index + 1}")
        if isinstance(value, dict):
            records.append(value)
    return records


def scalar_fields(record):
    output = {key: float(value) for key, value in record.items() if finite(value) and key not in METADATA}
    for container in ("component_means", "components", "aux", "telemetry", "loss_components", "raw_losses", "weighted_losses"):
        values = record.get(container)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if finite(value):
                key = key + "_weighted" if container == "weighted_losses" and not key.endswith("_weighted") else key
                output[key] = float(value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if finite(subvalue):
                        output[key + "." + subkey] = float(subvalue)
    for key, value in record.get("lr_by_group", {}).items():
        if finite(value):
            output["lr." + key] = float(value)
    return output


def epoch_of(record, epoch_base, steps=None):
    if finite(record.get("checkpoint_epoch")):
        return int(record["checkpoint_epoch"])
    if finite(record.get("epoch")):
        return int(record["epoch"]) - epoch_base
    if finite(record.get("epoch_1based")):
        return int(record["epoch_1based"]) - 1
    if steps and finite(record.get("iteration")):
        return int(record["iteration"]) // steps
    return None


def add_point(run, epoch, metric, value, source, source_kind, n_observations=1,
              category=None, role=None, n_cases=None, stage=None, extra=None):
    if epoch is None or epoch < 0 or not finite(value):
        return
    run["points"].append({"run": run["name"], "method": run["config"].get("method", "classifier" if run["config"].get("pretrained_classifier") else "unrecorded"),
                          "epoch_0based": epoch, "epoch_1based": epoch + 1, "metric": metric, "value": float(value),
                          "category": category or category_of(metric), "source_file": str(source.resolve()), "source_kind": source_kind,
                          "n_observations": n_observations, "selection_role": role or run["selection_role"],
                          "n_cases": n_cases, "stage": stage, **(extra or {})})


def category_of(metric):
    if metric.endswith("_weighted"):
        return "weighted_loss"
    if metric in RAW_KEYS or metric.startswith("segmentation_head_") or metric.endswith("_raw"):
        return "raw_loss"
    if metric == "loss":
        return "total_loss"
    if metric in ("lr", "learning_rate") or metric.startswith("lr."):
        return "learning_rate"
    if metric.endswith("_lambda") or metric in ("classifier_weight", "gaussian_weight", "guidance_weight"):
        return "coefficient"
    if "gradient" in metric or metric in ("segmentation_norm", "gaussian_weighted_norm", "gaussian_to_segmentation_ratio", "cosine"):
        return "gradient"
    return "other"


def add_classification(run, row, source, epoch):
    report = row.get("selection") or row.get("selection_report")
    if not isinstance(report, dict):
        return
    ids = row.get("selection_cases") or run["config"].get("selection_case_ids") or run["config"].get("internal_split", {}).get("val", [])
    role = run["selection_role"] + "; patient-level N is separate from patch N"
    if report.get("selection_positive_coverage_valid") is False:
        run["warnings"].append(f"Epoch {epoch + 1}: classifier positive coverage invalid; no valid mean-positive selection evidence")
    extra = {"n_patches": report.get("n_patches"), "positive_coverage_valid": report.get("selection_positive_coverage_valid")}
    add_point(run, epoch, "mean_positive_balanced_accuracy", report.get("mean_head_nonempty_balanced_accuracy"),
              source, "recorded_selection", category="classification_validation", role=role, n_cases=len(ids), extra=extra)
    for head, values in report.get("heads", {}).items():
        if head not in HEADS or not isinstance(values, dict):
            continue
        for key, suffix in (("nonempty_balanced_accuracy", "positive_balanced_accuracy"),
                            ("majority_baseline_nonempty_balanced_accuracy", "majority_positive_balanced_accuracy")):
            add_point(run, epoch, head + "." + suffix, values.get(key), source, "recorded_selection",
                      category="classification_validation", role=role, n_cases=len(ids),
                      extra=extra | {"positive_patch_support": values.get("nonempty_support"),
                                     "absent_positive_classes": json.dumps(values.get("absent_positive_evaluation_classes", []))})


def read_run(name, path, steps_log, epoch_base):
    history = path / "history.jsonl" if path.is_dir() else path
    if not history.is_file():
        raise FileNotFoundError("Missing recorded training history: " + str(history))
    folder = history.parent
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    ids = config.get("internal_split", {}).get("val", [])
    role = "internal-four model development" if len(ids) == 4 else "selection role unrecorded; do not infer independent validation"
    if config.get("internal_selection_valid") is False:
        role = "internal cases exposed to training/transfer; diagnostic only, invalid for selection"
    run = {"name": name, "history": history, "folder": folder, "config": config, "selection_role": role,
           "points": [], "warnings": [], "sources": [history] + ([config_path] if config_path.is_file() else []), "stage_by_epoch": {}}
    rows = read_records(history, run["warnings"])
    authoritative, seen = {}, {}
    for row in rows:
        epoch = epoch_of(row, epoch_base)
        if epoch is None or "step" in row:
            continue
        if epoch in seen and seen[epoch] != row:
            raise ValueError(f"Conflicting completed epoch {epoch} in {history}; do not mix resumed attempts")
        if epoch in seen:
            continue
        seen[epoch] = row
        run["stage_by_epoch"][epoch] = row.get("stage")
        for metric, value in scalar_fields(row).items():
            authoritative[epoch, metric] = True
            averaged = metric == "loss" or metric in row.get("component_means", {})
            add_point(run, epoch, metric, value, history, "recorded_epoch_mean" if averaged else "recorded_epoch_field", stage=row.get("stage"))
        add_classification(run, row, history, epoch)
    if steps_log:
        run["sources"].append(steps_log)
        logged, identities = {}, {}
        for row in read_records(steps_log, run["warnings"], log=True):
            epoch = epoch_of(row, epoch_base, config.get("steps"))
            if epoch is None or "step" not in row:
                continue
            identity = (epoch, row["step"])
            if identity in identities:
                if identities[identity] != row:
                    raise ValueError(f"Conflicting logged batch {identity}; separate attempts before plotting")
                continue
            identities[identity] = row
            for metric, value in scalar_fields(row).items():
                logged.setdefault((epoch, metric), []).append(value)
        for (epoch, metric), values in sorted(logged.items()):
            if (epoch, metric) not in authoritative:
                add_point(run, epoch, metric, statistics.mean(values), steps_log, "logged_batches_mean_not_full_epoch",
                          n_observations=len(values), stage=run["stage_by_epoch"].get(epoch))
        run["warnings"].append("Batch-log means use only recorded batches; they are labelled separately from full epoch means")
    probes = folder / "gradient_probes.jsonl"
    if probes.is_file():
        run["sources"].append(probes)
        grouped = {}
        for row in read_records(probes, run["warnings"]):
            epoch = epoch_of(row, epoch_base, config.get("steps"))
            for metric, value in scalar_fields(row).items():
                if category_of(metric) == "gradient":
                    grouped.setdefault((epoch, metric), []).append(value)
        for (epoch, metric), values in grouped.items():
            if (epoch, metric) not in authoritative:
                add_point(run, epoch, metric, statistics.mean(values), probes, "logged_gradient_probes_mean", len(values))
        run["warnings"].append("Gaussian/segmentation probes use shared main logits; DS branches are excluded from this derivative")
    return run


def metric_value(container, name):
    value = container.get(name)
    return value.get("mean") if isinstance(value, dict) else value


def validation_records(path, warnings):
    if path.is_file() and path.suffix == ".jsonl":
        return [(row, path) for row in read_records(path, warnings)]
    paths = [path] if path.is_file() else sorted(path.rglob("summary.json"))
    return [(json.loads(p.read_text(encoding="utf-8-sig")), p) for p in paths]


def add_validation(run, path, epoch_base):
    if not path.exists():
        raise FileNotFoundError("Missing requested validation trajectory: " + str(path))
    if path.is_file():
        run["sources"].append(path)
    seen = {}
    for row, source in validation_records(path, run["warnings"]):
        summary = row.get("summary") or row.get("evaluation_summary") or row
        if not isinstance(summary, dict):
            continue
        target = row.get("summary_path") or row.get("evaluation_output")
        if target:
            target = Path(target)
            target = target if target.is_absolute() else source.parent / target
            target = target / "summary.json" if target.is_dir() else target
            if target.is_file():
                summary = row | json.loads(target.read_text(encoding="utf-8-sig"))
                run["sources"].append(target)
        overall = summary.get("overall", summary)
        epoch = epoch_of(summary, epoch_base)
        if epoch is None:
            epoch = epoch_of(row, epoch_base)
        split = summary.get("evaluation_split", row.get("evaluation_split", row.get("split")))
        role = summary.get("evaluation_role", row.get("evaluation_role", row.get("role", run["selection_role"])))
        n_cases = overall.get("n_cases", row.get("n_cases"))
        if split != "internal_diagnostic" or n_cases != 4:
            run["warnings"].append(f"Excluded validation entry in {source.name}: split={split}, N={n_cases}; only internal4 trajectory is plotted")
            continue
        if summary.get("smoke", row.get("smoke", False)):
            run["warnings"].append(f"Excluded smoke validation entry in {source.name}")
            continue
        expected_ids = run["config"].get("internal_split", {}).get("val", [])
        actual_ids = summary.get("case_ids", row.get("case_ids"))
        if actual_ids is not None and expected_ids and set(actual_ids) != set(expected_ids):
            raise ValueError("Validation patients differ from the training run's internal four")
        if actual_ids is None:
            run["warnings"].append(f"Epoch {None if epoch is None else epoch + 1}: validation lacks patient IDs; N/split metadata only")
        if epoch is None:
            run["warnings"].append(f"Validation epoch absent in {source.name}; no epoch point invented")
            continue
        for metric in ("dice", "assd_mm"):
            value = metric_value(overall, metric)
            entry = overall.get(metric, {})
            if isinstance(entry, dict) and (entry.get("n_undefined", 0) or entry.get("n_valid", 4) != 4):
                value = None
            if not finite(value):
                run["warnings"].append(f"Epoch {epoch + 1}: internal4 {metric} missing/undefined; no complete-case replacement")
                continue
            identity = epoch, metric
            if identity in seen:
                if seen[identity] != value:
                    raise ValueError("Conflicting validation values for the same run/epoch/metric")
                continue
            seen[identity] = value
            sd = entry.get("sd") if isinstance(entry, dict) else None
            add_point(run, epoch, "validation." + metric, value, source, "recorded_internal4_validation",
                      category="segmentation_validation", role=role, n_cases=4,
                      extra={"patient_sd": sd, "evaluation_split": split, "checkpoint_selection": summary.get("checkpoint_selection")})
        if source not in run["sources"]:
            run["sources"].append(source)


def missing(ax, message):
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#666666", fontsize=10, wrap=True)


def series(run, category=None, names=None):
    output = {}
    for point in run["points"]:
        if category is not None and point["category"] != category or names is not None and point["metric"] not in names:
            continue
        output.setdefault(point["metric"], []).append(point)
    return {key: sorted(values, key=lambda p: p["epoch_0based"]) for key, values in output.items()}


def lines(ax, groups, labels=None, percent=False):
    for metric, points in groups.items():
        factor = 100 if percent else 1
        sampled = any(p["source_kind"] == "logged_batches_mean_not_full_epoch" for p in points)
        label = (labels or {}).get(metric, metric.replace("_", " ")) + (" [sampled logs]" if sampled else "")
        ax.plot([p["epoch_1based"] for p in points], [p["value"] * factor for p in points],
                marker="o", markersize=2.3, linewidth=1.25, label=label)
    if groups:
        ax.legend(fontsize=7.2, loc="best", frameon=False)


def render_run(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "font.size": 10})
    figure, axes = plt.subplots(4, 2, figsize=(15, 16), constrained_layout=True, sharex=True)
    titles = ("Raw loss components", "Weighted contributions and total loss", "Recorded regularization coefficients", "Recorded learning rates",
              "Positive-class balanced accuracy vs training-majority baseline", "Gradient diagnostics", "Internal four: Dice", "Internal four: pooled ASSD")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch (1-based)")
        ax.grid(alpha=.18)
    raw = series(run, "raw_loss")
    weighted = series(run, "weighted_loss") | series(run, "total_loss")
    for ax, groups, text in ((axes[0, 0], raw, "Component losses missing: only recorded values can be shown"),
                             (axes[0, 1], weighted, "Weighted loss contributions missing"),
                             (axes[1, 0], series(run, "coefficient"), "No logged coefficients; no schedule reconstructed"),
                             (axes[1, 1], series(run, "learning_rate"), "No logged learning rate")):
        lines(ax, groups)
        if not groups:
            missing(ax, text)
    axes[0, 0].set_ylabel("Unweighted objective value")
    axes[0, 1].set_ylabel("Objective contribution")
    axes[1, 0].set_ylabel("Coefficient")
    axes[1, 1].set_ylabel("Learning rate")
    if series(run, "learning_rate"):
        axes[1, 1].set_yscale("symlog", linthresh=1e-7)
    classified = series(run, "classification_validation")
    colors = {"generation": "#0072B2", "structure": "#D55E00", "location": "#009E73"}
    for metric, points in classified.items():
        head = metric.split(".")[0]
        baseline = "majority" in metric
        axes[2, 0].plot([p["epoch_1based"] for p in points], [p["value"] for p in points],
                         "--" if baseline else "-", marker="o", markersize=2.6,
                         color=colors.get(head, "#222222"), linewidth=1.1 if baseline else 1.6,
                         label=metric.replace("_", " "))
    if classified:
        axes[2, 0].legend(fontsize=7.1, frameon=False)
    else:
        missing(axes[2, 0], "Positive balanced accuracy and majority baseline missing\nLoss alone does not verify the classifier")
    axes[2, 0].set_ylim(0, 1)
    axes[2, 0].set_ylabel("Macro recall over present positive classes")
    gradients = series(run, "gradient")
    ratio = {key: values for key, values in gradients.items() if key in ("gaussian_to_segmentation_ratio", "cosine")}
    if ratio:
        primary = {key: values for key, values in ratio.items() if key != "cosine"}
        lines(axes[2, 1], primary)
        axes[2, 1].set_ylabel("Gaussian / segmentation gradient norm")
        if "cosine" in ratio:
            right = axes[2, 1].twinx()
            points = ratio["cosine"]
            right.plot([p["epoch_1based"] for p in points], [p["value"] for p in points], color="#888888", linestyle="--", label="gradient cosine")
            right.set_ylim(-1, 1); right.set_ylabel("Gradient cosine (shared main logits)")
    elif gradients:
        lines(axes[2, 1], gradients)
        axes[2, 1].set_ylabel("Recorded norms; different parameter groups")
    else:
        missing(axes[2, 1], "Gradient diagnostics missing\nWeighted scalar loss is not a gradient magnitude")
    for ax, metric, label, percent in ((axes[3, 0], "validation.dice", "Dice (%)", True),
                                       (axes[3, 1], "validation.assd_mm", "ASSD (mm); lower is better", False)):
        groups = series(run, names={metric})
        if groups:
            points = groups[metric]; factor = 100 if percent else 1
            ax.plot([p["epoch_1based"] for p in points], [p["value"] * factor for p in points], "o-", color="#0072B2", linewidth=1.6, markersize=4)
            with_sd = [p for p in points if finite(p.get("patient_sd"))]
            if with_sd:
                ax.errorbar([p["epoch_1based"] for p in with_sd], [p["value"] * factor for p in with_sd],
                            yerr=[p["patient_sd"] * factor for p in with_sd], fmt="none", ecolor="#8AB6D6", capsize=3, label="patient SD, N=4")
                ax.legend(fontsize=8, frameon=False)
        else:
            missing(ax, "Internal-four validation trajectory missing\nNo validation points or extrapolation invented")
        ax.set_ylabel(label)
    axes[3, 0].set_ylim(0, 100)
    axes[3, 1].set_ylim(bottom=0)
    stages = run["stage_by_epoch"]
    changes = [epoch for epoch in sorted(stages) if stages[epoch] and epoch - 1 in stages and stages[epoch] != stages[epoch - 1]]
    for ax in axes.flat:
        for epoch in changes:
            ax.axvline(epoch + 1, color="#888888", linestyle=":", alpha=.4, linewidth=.8)
        ax.set_xlim(1, max(2, max((p["epoch_1based"] for p in run["points"]), default=2)))
    train_n = len(run["config"].get("trained_case_ids", [])) or "unrecorded"
    method = run["config"].get("method", "classifier" if run["config"].get("pretrained_classifier") else "unrecorded")
    figure.suptitle(f"{run['name']} | {method} | training N={train_n}\n{run['selection_role']}; observed points only, no final-test claim", fontsize=13)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run["name"]).strip("_.") or "run"
    base = output / (safe + "_learning_curves")
    figure.savefig(base.with_suffix(".png"), dpi=170)
    figure.savefig(base.with_suffix(".svg"))
    plt.close(figure)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".svg"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=named_path)
    parser.add_argument("--validation", action="append", default=[], type=named_path)
    parser.add_argument("--steps-log", action="append", default=[], type=named_path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch-base", choices=(0, 1), type=int, default=0)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    names = [name for name, _ in args.run]
    if len(set(names)) != len(names):
        parser.error("Every run name must be distinct")
    validations, step_logs = dict(args.validation), dict(args.steps_log)
    if len(validations) != len(args.validation) or len(step_logs) != len(args.steps_log):
        parser.error("Supply one validation source and one batch log per run name; do not silently replace inputs")
    if set(validations) - set(names) or set(step_logs) - set(names):
        parser.error("Validation and step-log names must match --run names")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, path in args.run:
        run = read_run(name, path, step_logs.get(name), args.epoch_base)
        val_path = validations.get(name)
        if val_path is None:
            possible = [run["folder"] / "validation_history.jsonl", run["folder"].parent / "validation_history.jsonl"]
            val_path = next((p for p in possible if p.is_file()), None)
        if val_path:
            add_validation(run, val_path, args.epoch_base)
        if not series(run, "segmentation_validation"):
            run["warnings"].append("No usable internal-four Dice/ASSD trajectory; training loss cannot establish validation improvement")
        if not series(run, "classification_validation"):
            run["warnings"].append("No positive-balanced-accuracy/majority-baseline trajectory; classifier usefulness remains unverified by this history")
        runs.append(run)
    points = [point for run in runs for point in run["points"]]
    fields = ["run", "method", "epoch_0based", "epoch_1based", "metric", "value", "category", "stage", "source_kind", "n_observations",
              "selection_role", "n_cases", "n_patches", "positive_patch_support", "positive_coverage_valid", "absent_positive_classes",
              "patient_sd", "evaluation_split", "checkpoint_selection", "source_file"]
    with (output / "epoch_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        writer.writerows(sorted(points, key=lambda p: (p["run"], p["epoch_0based"], p["metric"])))
    artifacts = []
    if not args.tables_only:
        for run in runs:
            artifacts.extend(render_run(run, output))
    ledger = {"development_only": True, "formal_final_test": False, "epoch_display": "1-based; source epoch convention recorded below",
              "source_epoch_base": args.epoch_base, "interpolated_validation": False, "artifacts": artifacts,
              "plotter_sha256": sha256(Path(__file__)), "runs": []}
    note_lines = ["GTA observed learning curves; development only, no final-test or comparative superiority claim.",
                  "Lines connect recorded observations for readability; no unmeasured validation values are inserted.",
                  "Loss scale alone does not measure gradient size. N patches is not N independent patients."]
    for run in runs:
        sources = sorted(set(run["sources"]), key=str)
        record = {"name": run["name"], "method": run["config"].get("method", "classifier" if run["config"].get("pretrained_classifier") else "unrecorded"),
                  "selection_role": run["selection_role"], "n_training_cases": len(run["config"].get("trained_case_ids", [])) or None,
                  "selection_case_ids": run["config"].get("internal_split", {}).get("val", []),
                  "n_recorded_epochs": len({p["epoch_0based"] for p in run["points"] if p["category"] == "total_loss"}),
                  "n_internal_dice_points": len(series(run, names={"validation.dice"}).get("validation.dice", [])),
                  "n_internal_assd_points": len(series(run, names={"validation.assd_mm"}).get("validation.assd_mm", [])),
                  "warnings": sorted(set(run["warnings"])), "sources": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in sources]}
        ledger["runs"].append(record)
        note_lines.extend(["", f"Run {record['name']}: {record['method']}; training N={record['n_training_cases']}; {record['selection_role']}",
                           f"Recorded epochs {record['n_recorded_epochs']}; internal4 Dice points {record['n_internal_dice_points']}; ASSD points {record['n_internal_assd_points']}."])
        note_lines.extend("- " + warning for warning in record["warnings"])
    (output / "curve_manifest.json").write_text(json.dumps(ledger, indent=2, allow_nan=False), encoding="utf-8")
    (output / "curve_notes.txt").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": names, "n_observed_values": len(points), "artifacts": artifacts}), flush=True)


if __name__ == "__main__":
    main()
