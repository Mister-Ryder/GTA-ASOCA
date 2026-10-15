"""Run an actual, pinned nnU-Net v2 baseline on one frozen ASOCA holdout fold.

Example (after installing nnunetv2==2.6.4 in the experiment environment)::

    python code/nnunet_runner.py --root /root/autodl-tmp/ASOCA_Extension \
        --raw-root /root/autodl-tmp/ASOCA/CompressedVersion \
        --manifest /root/autodl-tmp/ASOCA_Extension/cache/manifest.json \
        --fold 0 --epochs 100

The 32 training cases alone enter native fingerprinting/planning. The independent
8-case holdout is imagesTs, with no labels in the native dataset. Native fold
"all" uses those same 32 cases for its internal training-monitor validation.
Only checkpoint_final.pth predicts the independent holdout. Nothing is selected
using its labels. This is the native 3d_fullres pipeline at 100 epochs/25,000
training updates, not the default 1,000-epoch nnU-Net or exact compute matching.

Official interfaces are pinned to these source references:
https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.4/documentation/dataset_format.md
https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.4/documentation/how_to_use_nnunet.md
https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.4/nnunetv2/run/run_training.py
https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.4/nnunetv2/inference/predict_from_raw_data.py
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import SimpleITK as sitk

from data import load_manifest
from metrics import binary_metrics

NNUNET_VERSION = "2.6.4"
PROTOCOL_VERSION = "asoca-native-nnunet-training-only-planning-v1"
CONFIGURATION = "3d_fullres"
PLANS_NAME = "nnUNetPlans"
ITERATIONS_PER_EPOCH = 250
TRAINERS = {1: "nnUNetTrainer_1epoch", 5: "nnUNetTrainer_5epochs",
            10: "nnUNetTrainer_10epochs", 20: "nnUNetTrainer_20epochs",
            50: "nnUNetTrainer_50epochs", 100: "nnUNetTrainer_100epochs",
            250: "nnUNetTrainer_250epochs", 500: "nnUNetTrainer_500epochs",
            750: "nnUNetTrainer_750epochs", 1000: "nnUNetTrainer",
            2000: "nnUNetTrainer_2000epochs", 4000: "nnUNetTrainer_4000epochs",
            8000: "nnUNetTrainer_8000epochs"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def ensure_identity(path: Path, identity: dict) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError(f"Existing output has a different experiment identity: {path}")
    else:
        atomic_json(path, identity)


def require_native_library() -> str:
    try:
        installed = metadata.version("nnunetv2")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("The native baseline requires nnunetv2==2.6.4. "
                           "Install that package in this environment; no substitute baseline will be run.") from error
    if installed != NNUNET_VERSION:
        raise RuntimeError(f"nnunetv2=={NNUNET_VERSION} is required; installed version is {installed}. "
                           "Use the pinned experiment environment.")
    return installed


def safe_source(raw_root: Path, relative_path: str) -> Path:
    source = (raw_root / relative_path).resolve()
    if not source.is_relative_to(raw_root.resolve()) or not source.is_file():
        raise RuntimeError(f"Missing or invalid raw data path: {relative_path}")
    return source


def link_source(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing native dataset symlink points at different data: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace an existing native dataset file: {destination}")
    destination.symlink_to(source.resolve())


def frozen_cases(manifest: dict, fold: int) -> tuple[dict, list[str], list[str]]:
    if manifest.get("pilot_only") or not manifest.get("complete"):
        raise ValueError("Native publication baseline requires the complete, non-pilot 40-case manifest")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if len(cases) != 40 or not 0 <= fold < 5:
        raise ValueError("Expected 40 unique cases and a frozen fold from 0 to 4")
    train_ids, heldout_ids = manifest["folds"][fold]["train"], manifest["folds"][fold]["val"]
    if len(train_ids) != 32 or len(set(train_ids)) != 32 or len(heldout_ids) != 8 \
            or len(set(heldout_ids)) != 8 or set(train_ids) & set(heldout_ids) \
            or set(train_ids) | set(heldout_ids) != set(cases):
        raise ValueError("Fold must be a disjoint, complete 32/8 patient partition")
    for group in ("Normal", "Diseased"):
        if sum(cases[c]["group"] == group for c in train_ids) != 16 \
                or sum(cases[c]["group"] == group for c in heldout_ids) != 4:
            raise ValueError("Fold has lost the frozen Normal/Diseased stratification")
    return cases, list(train_ids), list(heldout_ids)


def materialize_native_dataset(dataset: Path, raw_root: Path, cases: dict,
                               train_ids: list[str], heldout_ids: list[str]) -> dict:
    """Symlink original NRRD files; never materialize holdout label files."""
    expected = {"imagesTr": {f"{c}_0000.nrrd" for c in train_ids},
                "labelsTr": {f"{c}.nrrd" for c in train_ids},
                "imagesTs": {f"{c}_0000.nrrd" for c in heldout_ids}}
    provenance = {}
    for kind, ids in (("imagesTr", train_ids), ("labelsTr", train_ids), ("imagesTs", heldout_ids)):
        folder = dataset / kind
        folder.mkdir(parents=True, exist_ok=True)
        unexpected = {p.name for p in folder.iterdir()} - expected[kind]
        if unexpected:
            raise RuntimeError(f"Unexpected files in native {kind}: {sorted(unexpected)}")
        is_label = kind == "labelsTr"
        source_key = "label" if is_label else "image"
        for case_id in ids:
            case = cases[case_id]
            source = safe_source(raw_root, case[f"{source_key}_relative_path"])
            actual_hash = sha256_file(source)
            if actual_hash != case["source_sha256"][source_key]:
                raise RuntimeError(f"Raw {source_key} differs from the frozen audit: {case_id}")
            name = f"{case_id}.nrrd" if is_label else f"{case_id}_0000.nrrd"
            link_source(source, folder / name)
            provenance[f"{kind}/{name}"] = {"relative_source": case[f"{source_key}_relative_path"],
                                             "sha256": actual_hash}
        if {p.name for p in folder.iterdir()} != expected[kind]:
            raise RuntimeError(f"Native dataset is incomplete: {folder}")
    if (dataset / "labelsTs").exists():
        raise RuntimeError("Independent holdout labels must not be exposed to the native pipeline")
    dataset_json = {"channel_names": {"0": "CT"},
                    "labels": {"background": 0, "coronary artery": 1},
                    "numTraining": 32, "file_ending": ".nrrd",
                    "overwrite_image_reader_writer": "SimpleITKIO"}
    dataset_json_path = dataset / "dataset.json"
    ensure_identity(dataset_json_path, dataset_json)
    return provenance


# Each subprocess receives its environment before importing nnunetv2.paths.
# These are the official console-entrypoint functions, rather than a reimplemented
# planner/trainer. Initial RNG seeds are recorded; native asynchronous data
# augmentation is not claimed to be bitwise deterministic.
ENTRY_BOOTSTRAP = """
import importlib, random, sys
module, function, seed = sys.argv[1:4]
del sys.argv[1:4]
import numpy as np
import torch
seed = int(seed)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
getattr(importlib.import_module(module), function)()
"""


def native_command(module: str, function: str, arguments: list[str],
                   environment: dict, seed: int, root: Path, log: Path) -> None:
    command = [sys.executable, "-c", ENTRY_BOOTSTRAP, module, function, str(seed)] + arguments
    atomic_json(log.with_suffix(".command.json"), {"module": module, "entrypoint": function,
                                                "arguments": arguments, "initial_seed": seed})
    print(json.dumps({"stage": function, "log": str(log), "event": "started"}), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=environment, cwd=root, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line.rstrip(), flush=True)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Native {function} failed with exit status {code}; inspect {log}")
    print(json.dumps({"stage": function, "event": "finished"}), flush=True)


def check_space(root: Path, minimum_gb: float, stage: str) -> dict:
    free = shutil.disk_usage(root).free / 2 ** 30
    snapshot = {"stage": stage, "free_GiB": round(free, 3), "required_free_GiB": minimum_gb}
    print(json.dumps({"disk": snapshot}), flush=True)
    if free < minimum_gb:
        raise RuntimeError(f"Only {free:.2f} GiB free before {stage}; need at least {minimum_gb:.2f} GiB. "
                           "Existing experiment data is retained.")
    return snapshot


def verify_preprocessing(folder: Path, train_ids: list[str]) -> dict:
    plans = json.loads((folder / f"{PLANS_NAME}.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    if dataset_json["numTraining"] != 32 or dataset_json["overwrite_image_reader_writer"] != "SimpleITKIO":
        raise RuntimeError("Native preprocessed dataset metadata differs from the training-only dataset")
    configuration = plans["configurations"].get(CONFIGURATION)
    if configuration is None:
        raise RuntimeError("The native planner did not generate 3d_fullres; no replacement configuration will run")
    data_folder = folder / configuration["data_identifier"]
    identifiers = {p.stem for p in data_folder.glob("*.pkl")}
    if identifiers != set(train_ids):
        raise RuntimeError(f"Native preprocessing identifiers differ from the 32 training cases: {sorted(identifiers)}")
    if not (folder / "dataset_fingerprint.json").is_file():
        raise RuntimeError("Missing native training-only dataset fingerprint")
    return {"patch_size": configuration["patch_size"], "batch_size": configuration["batch_size"],
            "spacing": configuration["spacing"],
            "normalization_schemes": configuration["normalization_schemes"],
            "data_identifier": configuration["data_identifier"],
            "plans_sha256": sha256_file(folder / f"{PLANS_NAME}.json"),
            "fingerprint_sha256": sha256_file(folder / "dataset_fingerprint.json"),
            "preprocessed_training_case_ids": sorted(identifiers)}


def checkpoint_history(checkpoint: Path, epochs: int, trainer: str, output: Path) -> dict:
    import torch
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_data.get("trainer_name") != trainer \
            or checkpoint_data.get("init_args", {}).get("fold") != "all":
        raise RuntimeError("Native checkpoint belongs to a different trainer or native fold")
    if int(checkpoint_data.get("current_epoch", -1)) != epochs:
        raise RuntimeError("Native final checkpoint does not contain the requested completed epoch count")
    logging = checkpoint_data.get("logging", {})
    losses = logging.get("train_losses", [])
    if len(losses) != epochs or not np.isfinite(np.asarray(losses, dtype=np.float64)).all():
        raise RuntimeError("Native final checkpoint has incomplete or nonfinite training-loss history")
    start_times = logging.get("epoch_start_timestamps", [])
    end_times = logging.get("epoch_end_timestamps", [])
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for epoch, loss in enumerate(losses):
            row = {"epoch": epoch, "loss": float(loss), "iterations": ITERATIONS_PER_EPOCH,
                   "optimizer_updates_cumulative": (epoch + 1) * ITERATIONS_PER_EPOCH,
                   "history_source": "native checkpoint training logger"}
            if len(start_times) > epoch and len(end_times) > epoch:
                row["seconds"] = float(end_times[epoch] - start_times[epoch])
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(output)
    return {"completed_epochs": len(losses), "completed_optimizer_updates": len(losses) * ITERATIONS_PER_EPOCH,
            "checkpoint_sha256": sha256_file(checkpoint), "history_sha256": sha256_file(output)}


def summarize(rows: list[dict]) -> dict:
    fields = [key for key, value in rows[0].items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)]
    result = {"n_cases": len(rows), "empty_predictions": sum(bool(r["empty_prediction"]) for r in rows)}
    for field in fields:
        values = [r[field] for r in rows if r.get(field) is not None]
        result[field] = {"mean": float(np.mean(values)),
                         "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                         "n_valid": len(values)}
    if result["empty_predictions"]:
        for key in ("surface_hd95_mm", "assd_mm", "asoca_foreground_hd95_mm"):
            result[key] = {"mean": None, "reason": "empty predictions present; inspect per-case results"}
    return result


def evaluate_holdout(predictions: Path, output: Path, raw_root: Path,
                     cases: dict, heldout_ids: list[str], fold: int,
                     manifest_hash: str, checkpoint_hash: str, protocol_hash: str) -> dict:
    if {p.stem for p in predictions.glob("*.nrrd")} != set(heldout_ids):
        raise RuntimeError("Native predictions must contain exactly the eight independent holdout cases before opening references")
    output.mkdir(parents=True, exist_ok=True)
    case_jsons = {p.stem for p in output.glob("*.json") if p.name != "summary.json"}
    if case_jsons - set(heldout_ids):
        raise RuntimeError("Evaluation output contains cases outside the frozen independent holdout")
    rows = []
    for case_id in heldout_ids:
        case = cases[case_id]
        target = output / f"{case_id}.json"
        prediction_output = output / f"{case_id}_prediction.nrrd"
        label_path = safe_source(raw_root, case["label_relative_path"])
        # Holdout references are first opened after the native predictions exist.
        label_hash = sha256_file(label_path)
        if label_hash != case["source_sha256"]["label"]:
            raise RuntimeError(f"Independent reference differs from the frozen audit: {case_id}")
        if target.is_file():
            row = json.loads(target.read_text(encoding="utf-8"))
            expected = {"checkpoint_sha256": checkpoint_hash, "manifest_sha256": manifest_hash,
                        "protocol_sha256": protocol_hash, "fold": fold, "method": "nnunet", "smoke": False,
                        "case_id": case_id, "label_sha256": label_hash}
            if any(row.get(key) != value for key, value in expected.items()) or not prediction_output.is_file():
                raise RuntimeError(f"Existing evaluation has a different identity or missing prediction: {target}")
            if row.get("prediction_sha256") != sha256_file(prediction_output):
                raise RuntimeError(f"Existing prediction differs from the evaluated mask: {prediction_output}")
            rows.append(row)
            continue
        started = time.monotonic()
        native_prediction = predictions / f"{case_id}.nrrd"
        if not native_prediction.is_file():
            raise RuntimeError(f"Native inference did not produce independent holdout case {case_id}")
        predicted = sitk.ReadImage(str(native_prediction))
        reference = sitk.ReadImage(str(label_path))
        same_geometry = (predicted.GetSize() == reference.GetSize()
                         and np.allclose(predicted.GetSpacing(), reference.GetSpacing(), atol=1e-6)
                         and np.allclose(predicted.GetOrigin(), reference.GetOrigin(), atol=1e-5)
                         and np.allclose(predicted.GetDirection(), reference.GetDirection(), atol=1e-6))
        if not same_geometry:
            predicted = sitk.Resample(predicted, reference, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        prediction_array = sitk.GetArrayFromImage(predicted)
        truth = sitk.GetArrayFromImage(reference)
        if not set(np.unique(prediction_array)).issubset({0, 1}) or set(np.unique(truth)) != {0, 1}:
            raise RuntimeError(f"Independent coronary segmentation/reference is not binary: {case_id}")
        row = binary_metrics(prediction_array, truth, reference.GetSpacing()[::-1])
        row.update(case_id=case_id, group=case["group"], fold=fold, method="nnunet",
                   checkpoint_sha256=checkpoint_hash, manifest_sha256=manifest_hash,
                   protocol_sha256=protocol_hash, label_sha256=label_hash,
                   image_sha256=case["source_sha256"]["image"],
                   seconds=round(time.monotonic() - started, 2), smoke=False,
                   metric_grid="original CT", checkpoint_selection="fixed final",
                   native_nnunet_version=NNUNET_VERSION,
                   native_prediction_geometry_matches_reference=same_geometry)
        temporary_prediction = prediction_output.with_name(case_id + "_prediction.tmp.nrrd")
        sitk.WriteImage(predicted, str(temporary_prediction), True)
        temporary_prediction.replace(prediction_output)
        row["prediction_sha256"] = sha256_file(prediction_output)
        atomic_json(target, row)
        print(json.dumps(row), flush=True)
        rows.append(row)
    if len(rows) != 8 or {r["case_id"] for r in rows} != set(heldout_ids):
        raise RuntimeError("Independent native evaluation must contain exactly the eight frozen holdout cases")
    summary = {"overall": summarize(rows), "smoke": False, "fold": fold, "method": "nnunet",
               "manifest_sha256": manifest_hash, "checkpoint_sha256": checkpoint_hash,
               "protocol_sha256": protocol_hash, "checkpoint_selection": "fixed final",
               "native_nnunet_version": NNUNET_VERSION}
    for group in ("Normal", "Diseased"):
        summary[group] = summarize([row for row in rows if row["group"] == group])
    atomic_json(output / "summary.json", summary)
    return summary


def clean_owned_preprocessing(root: Path, folder: Path, dataset_name: str) -> None:
    allowed_parent = (root / "nnUNet_preprocessed").resolve()
    resolved = folder.resolve()
    if folder.is_symlink() or resolved.parent != allowed_parent or resolved.name != dataset_name \
            or not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"Preprocessing cleanup target is not the owned fold directory: {folder}")
    # Called only after predictions and every independent case metric succeeded.
    shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Experiment project root")
    parser.add_argument("--raw-root", type=Path, required=True, help="Original ASOCA CompressedVersion directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100, choices=sorted(TRAINERS))
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--cleanup-preprocessed", action="store_true",
                        help="After successful independent evaluation remove only this native fold's owned preprocessed data")
    args = parser.parse_args()
    require_native_library()
    manifest = load_manifest(args.manifest)
    cases, train_ids, heldout_ids = frozen_cases(manifest, args.fold)
    root, raw_root = args.root.resolve(), args.raw_root.resolve()
    if not raw_root.is_dir() or args.min_free_gb < 0:
        parser.error("raw-root must exist and min-free-gb must be nonnegative")
    root.mkdir(parents=True, exist_ok=True)
    dataset_id = 881 + args.fold
    dataset_name = f"Dataset{dataset_id:03d}_ASOCA_Fold{args.fold}"
    trainer = TRAINERS[args.epochs]
    output = root / "results" / "nnunet" / f"fold{args.fold}"
    training = output / "training"
    dataset = root / "nnUNet_raw" / dataset_name
    preprocessed = root / "nnUNet_preprocessed" / dataset_name
    native_model = root / "nnUNet_results" / dataset_name / f"{trainer}__{PLANS_NAME}__{CONFIGURATION}"
    checkpoint = native_model / "fold_all" / "checkpoint_final.pth"
    manifest_path = Path(manifest["_manifest_path"])
    manifest_hash = sha256_file(manifest_path)
    identity = {"protocol_version": PROTOCOL_VERSION, "nnunetv2": NNUNET_VERSION,
                "method": "nnunet", "manifest_sha256": manifest_hash, "fold": args.fold,
                "epochs": args.epochs, "initial_seed": args.seed + args.fold,
                "dataset_name": dataset_name, "trainer": trainer,
                "configuration": CONFIGURATION, "plans_identifier": PLANS_NAME,
                "native_training_fold": "all", "train_case_ids": train_ids,
                "heldout_case_ids": heldout_ids,
                "source_sha256": {c: cases[c]["source_sha256"] for c in sorted(cases)},
                "publication_result": True, "smoke": False}
    protocol_path = training / "protocol.json"
    if not protocol_path.is_file() and any(p.exists() for p in (dataset, preprocessed, native_model)):
        raise RuntimeError("Native fold directories exist without this runner's ownership/identity record; "
                           "choose a fresh experiment root")
    training.mkdir(parents=True, exist_ok=True)
    ensure_identity(protocol_path, identity)
    protocol_hash = sha256_file(protocol_path)
    environment = os.environ.copy()
    environment.update(nnUNet_raw=str(root / "nnUNet_raw"),
                       nnUNet_preprocessed=str(root / "nnUNet_preprocessed"),
                       nnUNet_results=str(root / "nnUNet_results"),
                       nnUNet_n_proc_DA="2", nnUNet_compile="false",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    provenance = materialize_native_dataset(dataset, raw_root, cases, train_ids, heldout_ids)
    atomic_json(training / "native_raw_provenance.json", provenance)
    config = {**identity, "checkpoint_path": str(checkpoint),
              "iterations_per_epoch": ITERATIONS_PER_EPOCH,
              "planned_optimizer_updates": args.epochs * ITERATIONS_PER_EPOCH,
              "native_internal_validation": "same 32 training cases, monitoring only; never reported as holdout performance",
              "independent_evaluation_case_count": 8, "checkpoint_selection": "fixed final",
              "planning_statistics_cases": "32 frozen training cases only",
              "normalization": "native CT foreground statistics estimated only from training cases",
              "optimizer": "native SGD/Nesterov momentum0.99, initial_lr0.01, weight_decay3e-5, polynomial schedule",
              "augmentation": "unmodified native spatial and intensity augmentation",
              "deep_supervision": "native enabled", "inference_mirroring_tta": True,
              "exact_compute_matched": False,
              "comparison_scope": "same patients and default 25,000 optimizer updates at epochs100; native spacing, normalization, patch/batch, optimizer, augmentation and inference TTA differ",
              "default_1000_epoch_nnunet": args.epochs == 1000,
              "randomness": "recorded initial seed; native asynchronous augmentation is not bitwise deterministic",
              "native_environment": {k: environment[k] for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_n_proc_DA", "nnUNet_compile")},
              "runner_sha256": sha256_file(Path(__file__)), "publication_result": True, "smoke": False}
    config_path = training / "config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Existing native config differs from the frozen protocol")
    atomic_json(config_path, config)
    disk_checks = []
    preprocessing_marker = training / "PREPROCESSING_COMPLETE.json"
    if not checkpoint.is_file():
        if not preprocessing_marker.is_file() or not preprocessed.is_dir():
            disk_checks.append(check_space(root, args.min_free_gb, "native planning and preprocessing"))
            native_command("nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
                           "plan_and_preprocess_entry",
                           ["-d", str(dataset_id), "-c", CONFIGURATION, "-np", "2", "-npfp", "2", "--verify_dataset_integrity"],
                           environment, args.seed + args.fold, root, training / "planning.log")
            planned = verify_preprocessing(preprocessed, train_ids)
            atomic_json(preprocessing_marker, {"protocol_sha256": protocol_hash, **planned})
        else:
            marker = json.loads(preprocessing_marker.read_text(encoding="utf-8"))
            if marker.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("Existing native preprocessing marker differs from the frozen protocol")
            planned = verify_preprocessing(preprocessed, train_ids)
        config["native_plan"] = planned
        atomic_json(config_path, config)
        disk_checks.append(check_space(root, args.min_free_gb, "native training and possible unpacking"))
        arguments = [dataset_name, CONFIGURATION, "all", "-tr", trainer, "-p", PLANS_NAME, "-device", "cuda"]
        latest = native_model / "fold_all" / "checkpoint_latest.pth"
        best = native_model / "fold_all" / "checkpoint_best.pth"
        if latest.is_file() or best.is_file():
            arguments.append("--c")
        native_command("nnunetv2.run.run_training", "run_training_entry", arguments,
                       environment, args.seed + args.fold, root, training / "native_training.log")
    if not checkpoint.is_file():
        raise RuntimeError("Native training returned without checkpoint_final.pth")
    if preprocessed.is_dir():
        config["native_plan"] = verify_preprocessing(preprocessed, train_ids)
    else:
        # Successful cleanup retains the immutable native plans in model output.
        saved_plans = json.loads((native_model / "plans.json").read_text(encoding="utf-8"))
        selected = saved_plans["configurations"][CONFIGURATION]
        config["native_plan"] = {k: selected[k] for k in ("patch_size", "batch_size", "spacing", "normalization_schemes", "data_identifier")}
    actual = checkpoint_history(checkpoint, args.epochs, trainer, training / "history.jsonl")
    debug_path = native_model / "fold_all" / "debug.json"
    if debug_path.is_file():
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
        if int(debug.get("num_iterations_per_epoch", -1)) != ITERATIONS_PER_EPOCH:
            raise RuntimeError("Actual native iteration count differs from the planned 250 per epoch")
        config["native_debug_sha256"] = sha256_file(debug_path)
    config.update(actual)
    config["disk_checks"] = disk_checks
    atomic_json(config_path, config)
    atomic_json(training / "TRAINING_COMPLETE.json",
                {**actual, "epoch": args.epochs - 1, "epochs": args.epochs, "method": "nnunet",
                 "fold": args.fold, "smoke": False, "manifest_sha256": manifest_hash,
                 "protocol_sha256": protocol_hash, "checkpoint_path": str(checkpoint),
                 "checkpoint_selection": "fixed final"})
    predictions = output / "native_predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    ensure_identity(predictions / "prediction_identity.json",
                    {"checkpoint_sha256": actual["checkpoint_sha256"], "manifest_sha256": manifest_hash,
                     "protocol_sha256": protocol_hash, "heldout_case_ids": heldout_ids,
                     "native_mirroring_tta": True, "checkpoint_selection": "fixed final"})
    unexpected_predictions = {p.stem for p in predictions.glob("*.nrrd")} - set(heldout_ids)
    if unexpected_predictions:
        raise RuntimeError("Native prediction directory contains cases outside the independent holdout")
    if any(not (predictions / f"{case_id}.nrrd").is_file() for case_id in heldout_ids):
        disk_checks.append(check_space(root, args.min_free_gb, "independent native inference"))
        native_command("nnunetv2.inference.predict_from_raw_data", "predict_entry_point",
                       ["-i", str(dataset / "imagesTs"), "-o", str(predictions), "-d", dataset_name,
                        "-c", CONFIGURATION, "-tr", trainer, "-p", PLANS_NAME, "-f", "all",
                        "-chk", "checkpoint_final.pth", "-npp", "2", "-nps", "2", "--continue_prediction"],
                       environment, args.seed + args.fold, root, training / "native_inference.log")
    evaluate_holdout(predictions, output / "evaluation", raw_root, cases, heldout_ids,
                     args.fold, manifest_hash, actual["checkpoint_sha256"], protocol_hash)
    if args.cleanup_preprocessed and preprocessed.is_dir():
        clean_owned_preprocessing(root, preprocessed, dataset_name)
        config["preprocessed_cleaned_after_success"] = True
    config["disk_checks"] = disk_checks
    atomic_json(config_path, config)
    atomic_json(output / "EXPERIMENT_COMPLETE.json",
                {"method": "nnunet", "fold": args.fold, "smoke": False,
                 "manifest_sha256": manifest_hash, "checkpoint_sha256": actual["checkpoint_sha256"],
                 "protocol_sha256": protocol_hash, "heldout_case_ids": heldout_ids,
                 "evaluation_complete": True})
    print("NNUNET_TRAINING_AND_INDEPENDENT_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
