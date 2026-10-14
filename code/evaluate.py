"""Whole-CT inference; original-grid metrics; no labels used by the predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from data import load_manifest, case_directory
from metrics import (binary_metrics, surface_distance_metrics, METRIC_PROTOCOL,
                     METRIC_PROTOCOL_VERSION, SURFACE_DISTANCE_FIELDS, ASSD_POOLED_DEFINITION,
                     ASSD_DIRECTIONAL_DEFINITION)
from train import make_model, source_hash, SOURCE_HASH_ALGORITHM


def original_prediction(prediction, case, reference):
    image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    geometry = case["resampled_geometry"]
    image.SetSpacing(geometry["spacing_xyz_mm"])
    image.SetOrigin(geometry["origin_xyz_mm"])
    image.SetDirection(geometry["direction"])
    return sitk.Resample(image,reference,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)


def summarize(rows):
    fields = ("dice", "assd_mm", "assd_directional_mean_mm", "cldice", "precision", "recall",
              "centerline_recall", "centerline_precision", "thin_centerline_recall_radius_le_1mm",
              "surface_hd95_mm", "asoca_foreground_hd95_mm", "asd_prediction_to_truth_mm",
              "asd_truth_to_prediction_mm", "prediction_components", "truth_components", "seconds")
    output = {"n_cases":len(rows),"empty_predictions":sum(bool(r["empty_prediction"]) for r in rows)}
    for field in fields:
        values = [r[field] for r in rows if isinstance(r.get(field),(int,float)) and np.isfinite(r[field])]
        undefined = [r["case_id"] for r in rows if not isinstance(r.get(field),(int,float)) or not np.isfinite(r[field])]
        output[field] = {"mean":float(np.mean(values)) if values else None,
                         "sd":float(np.std(values,ddof=1)) if len(values)>1 else None,
                         "n_valid":len(values), "n_undefined":len(undefined), "undefined_case_ids":undefined}
        if field in SURFACE_DISTANCE_FIELDS and len(values) != len(rows):
            output[field].update(mean=None, sd=None,
                                 reason="undefined or unrecorded distances; no complete-cohort mean")
    return output


def atomic_json(path, row):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(row,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def verify_training_source(config):
    """Inference/reuse may only claim compatibility with this exact v2 recipe."""
    if config.get("source_hash_algorithm") != SOURCE_HASH_ALGORITHM:
        raise ValueError("Checkpoint source protocol is not this v2 implementation; saved legacy predictions require --geometry-only")
    if source_hash() != config.get("source_sha256"):
        raise ValueError("Live training/model implementation differs from the checkpoint's frozen training source")
    source_files = config.get("source_files_sha256")
    if source_files is not None:
        if not isinstance(source_files,dict) or not source_files:
            raise ValueError("Invalid frozen per-file training source identity")
        root = Path(__file__).resolve().parent
        for name, expected in source_files.items():
            path = (root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Frozen training source path is outside the live code folder or missing")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Live training source file differs from checkpoint: " + name)


def read_reference(raw_root, case):
    path = Path(raw_root)/case["label_relative_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != case["source_sha256"]["label"]:
        raise ValueError("Raw reference differs from audited label: " + case["case_id"])
    return sitk.ReadImage(str(path)), digest


def verify_grid(predicted, reference):
    if predicted.GetSize() != reference.GetSize():
        raise ValueError("Saved prediction shape differs from its original-grid reference")
    for getter in ("GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(predicted,getter)(),getattr(reference,getter)(),rtol=0,atol=1e-6):
            raise ValueError("Saved prediction has incompatible " + getter)


def upgrade_saved_geometry(row, prediction_path, raw_root, case):
    """Add missing convention fields from saved pixels, keeping legacy values."""
    if not prediction_path.is_file():
        raise ValueError("Existing metrics lack their saved prediction: " + case["case_id"])
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    if row.get("prediction_sha256") and row["prediction_sha256"] != digest:
        raise ValueError("Saved prediction changed after its metrics were recorded")
    reference, label_hash = read_reference(raw_root, case)
    if row.get("label_sha256") != label_hash:
        raise ValueError("Existing metrics belong to another reference")
    predicted = sitk.ReadImage(str(prediction_path))
    verify_grid(predicted,reference)
    added = surface_distance_metrics(sitk.GetArrayFromImage(predicted),sitk.GetArrayFromImage(reference),
                                     reference.GetSpacing()[::-1])
    if row.get("assd_definition",added["assd_definition"]) != added["assd_definition"]:
        raise ValueError("Existing ASSD is labelled with another convention")
    for key in ("assd_mm", "surface_hd95_mm", "asoca_foreground_hd95_mm"):
        old, new = row.get(key), added[key]
        if (old is None) != (new is None) or old is not None and not np.isclose(old,new,rtol=1e-6,atol=1e-6):
            raise ValueError("Legacy metric is not reproducible from its saved pixels: " + key)
    # Never replace historical primary/distance values during this upgrade.
    preserved = {"assd_mm", "surface_hd95_mm", "asoca_foreground_hd95_mm"}
    row.update({key:value for key,value in added.items() if key not in preserved})
    row["metric_protocol_version"] = METRIC_PROTOCOL_VERSION
    row["prediction_sha256"] = digest
    row["geometry_upgrade_source"] = "saved original-grid prediction; legacy metric values preserved"
    return row


def evaluation_cases(manifest, config, fold, split):
    outer = manifest["folds"][fold]
    if split == "outer_development":
        return list(outer["val"]), "previously inspected outer development cases; not final independent test"
    internal = config.get("internal_split")
    if not isinstance(internal,dict):
        raise ValueError("Internal diagnostic split is missing; outer8 evaluation must be requested explicitly")
    train, val = internal.get("train",[]), internal.get("val",[])
    if len(train)!=28 or len(val)!=4 or len(set(train))!=28 or len(set(val))!=4:
        raise ValueError("Expected 28 development-training and four reserved diagnostic cases")
    if set(train)&set(val) or set(train)|set(val)!=set(outer["train"]):
        raise ValueError("Internal diagnostic split must partition only the frozen outer32")
    cases = {case["case_id"]:case for case in manifest["cases"]}
    if any(sum(cases[c]["group"]==g for c in val)!=2 for g in ("Normal","Diseased")):
        raise ValueError("Internal diagnostic four must include two cases from each stratum")
    if config.get("internal_selection_valid") is False or set(config.get("trained_case_ids",[])) & set(val):
        return list(val), "internal cases exposed to training/transfer; diagnostics only, invalid for model selection"
    return list(val), "internal diagnostic/model-selection cases; not final independent test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--raw-root",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--fold",type=int,default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--max-cases",type=int)
    parser.add_argument("--split",choices=("internal_diagnostic","outer_development"),default="internal_diagnostic",
                        help="Default reserves internal4; previously seen outer8 requires an explicit choice")
    parser.add_argument("--sw-batch-size",type=int,default=2)
    parser.add_argument("--geometry-only",action="store_true",
                        help="Only upgrade existing saved predictions; never infer with a legacy/unverified model source")
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = load_manifest(args.manifest)
    checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    config = checkpoint["config"]
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if config["manifest_sha256"] != manifest_hash:
        raise ValueError("Evaluation manifest differs from training manifest")
    if config.get("smoke") and not args.smoke:
        raise ValueError("Smoke weights are not publication checkpoints")
    if config["fold"] != args.fold:
        raise ValueError("Checkpoint fold differs from evaluation split")
    training_source_verified = config.get("source_hash_algorithm") == SOURCE_HASH_ALGORITHM
    if not args.geometry_only or training_source_verified:
        verify_training_source(config)
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_selection = checkpoint.get("checkpoint_role")
    if not checkpoint_selection:
        if isinstance(checkpoint_epoch,int) and checkpoint_epoch == config.get("epochs",0)-1:
            checkpoint_selection = config.get("checkpoint_selection","fixed final")
        else:
            checkpoint_selection = "intermediate checkpoint diagnostics; not fixed final"
    if not args.smoke and args.split == "outer_development" and checkpoint_selection != "fixed final":
        raise ValueError("Original outer8 require a frozen final checkpoint; intermediate trajectories use internal4")
    model = None  # saved-prediction metric upgrades never require model inference
    source_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    cases = {c["case_id"]:c for c in manifest["cases"]}
    if args.smoke:
        if manifest.get("pilot_only"):
            ids, evaluation_role = list(cases), "smoke workflow verification only; pilot patients"
        else:
            ids, evaluation_role = evaluation_cases(manifest,config,args.fold,args.split)
            evaluation_role = "smoke workflow verification only; " + evaluation_role
    else:
        ids, evaluation_role = evaluation_cases(manifest,config,args.fold,args.split)
    if args.max_cases:
        if not args.smoke:
            raise ValueError("Partial evaluation allowed only for smoke")
        ids = ids[:args.max_cases]
    rows = []
    training_source_hash = config.get("source_sha256")
    evaluator_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metrics_source_hash = hashlib.sha256((Path(__file__).parent/"metrics.py").read_bytes()).hexdigest()
    roi_size = config["patch_size"]
    roi_size = (roi_size,)*3 if isinstance(roi_size,int) else tuple(roi_size)
    if len(roi_size)!=3 or min(roi_size)<=0 or args.sw_batch_size<=0:
        raise ValueError("Invalid inference patch or batch size")
    for case_id in ids:
        if not args.geometry_only or training_source_verified:
            verify_training_source(config)
        case = cases[case_id]
        target = out/(case_id+".json")
        if target.exists():
            row = json.loads(target.read_text())
            if row["checkpoint_sha256"] != source_hash or row.get("manifest_sha256") != manifest_hash:
                raise RuntimeError("Output contains metrics from different weights")
            if row.get("case_id") != case_id or row.get("fold") != args.fold or row.get("smoke") != args.smoke:
                raise RuntimeError("Existing metrics have incompatible case/fold/smoke metadata")
            if row.get("group") != case["group"] or row.get("method") != config["method"] or row.get("label_sha256") != case["source_sha256"]["label"]:
                raise RuntimeError("Existing metrics have incompatible method/patient/reference identity")
            if row.get("evaluation_split",args.split) != args.split:
                raise RuntimeError("Output contains another evaluation split")
            new_fields = set(SURFACE_DISTANCE_FIELDS) | {"assd_definition","assd_directional_mean_definition"}
            if (row.get("metric_protocol_version") != METRIC_PROTOCOL_VERSION or not new_fields <= row.keys()
                    or not row.get("prediction_sha256")
                    or not row.get("empty_prediction") and any(row.get(k) is None for k in SURFACE_DISTANCE_FIELDS)):
                row = upgrade_saved_geometry(row,out/(case_id+"_prediction.nrrd"),args.raw_root,case)
            else:
                if (row["assd_definition"] != ASSD_POOLED_DEFINITION
                        or row["assd_directional_mean_definition"] != ASSD_DIRECTIONAL_DEFINITION):
                    raise RuntimeError("Existing metrics use another distance convention")
                prediction_path = out/(case_id+"_prediction.nrrd")
                if not prediction_path.is_file() or hashlib.sha256(prediction_path.read_bytes()).hexdigest() != row["prediction_sha256"]:
                    raise RuntimeError("Saved prediction differs from recorded metrics")
                if hashlib.sha256((Path(args.raw_root)/case["label_relative_path"]).read_bytes()).hexdigest() != row["label_sha256"]:
                    raise RuntimeError("Raw reference differs from recorded metrics")
            row.update(evaluation_split=args.split,evaluation_role=evaluation_role,formal_final_test=False,
                       checkpoint_selection=checkpoint_selection,checkpoint_epoch=checkpoint_epoch,
                       internal_selection_valid=config.get("internal_selection_valid"),
                       training_source_verified_against_live_code=training_source_verified,
                       geometry_only=args.geometry_only,
                       training_source_sha256=training_source_hash,evaluation_source_sha256=evaluator_source_hash,
                       metrics_source_sha256=metrics_source_hash)
            atomic_json(target,row)
            rows.append(row)
            continue
        if args.geometry_only:
            raise ValueError("Geometry-only mode requires metrics and a saved prediction for every requested patient")
        started = time.time()
        array = np.load(case_directory(manifest,case)/"image.npy",mmap_mode="r")
        inputs = torch.from_numpy(np.asarray(array,dtype=np.float32).copy())[None,None]
        if model is None:
            model = make_model(config["method"],not config["no_ds"]).cuda().eval()
            model.load_state_dict(checkpoint["model"])
        def predict(x):
            with torch.amp.autocast("cuda",dtype=torch.float16):
                return model(x)["logits"].float()
        with torch.inference_mode():
            logits = sliding_window_inference(inputs,roi_size,args.sw_batch_size,predict,overlap=.5,
                                              mode="gaussian",sw_device="cuda",device="cpu",cval=-1.)
        verify_training_source(config)
        mask = logits.argmax(1)[0].numpy().astype(np.uint8)
        reference, label_hash = read_reference(args.raw_root,case)
        predicted = original_prediction(mask,case,reference)
        truth = sitk.GetArrayFromImage(reference)
        prediction = sitk.GetArrayFromImage(predicted)
        row = binary_metrics(prediction,truth,reference.GetSpacing()[::-1])
        row.update(case_id=case_id,group=case["group"],fold=args.fold,method=config["method"],
                   checkpoint_sha256=source_hash,manifest_sha256=manifest_hash,label_sha256=label_hash,seconds=round(time.time()-started,2),
                   smoke=args.smoke,metric_grid="original CT",checkpoint_selection=checkpoint_selection,checkpoint_epoch=checkpoint_epoch,
                   evaluation_split=args.split,evaluation_role=evaluation_role,formal_final_test=False,
                   internal_selection_valid=config.get("internal_selection_valid"),
                   training_source_verified_against_live_code=training_source_verified,geometry_only=False,
                   inference_patch_size=list(roi_size),inference_sw_batch_size=args.sw_batch_size,
                   training_source_sha256=training_source_hash,evaluation_source_sha256=evaluator_source_hash,
                   metrics_source_sha256=metrics_source_hash)
        destination = out/(case_id+"_prediction.nrrd")
        temporary_prediction = out/(case_id+"_prediction.tmp.nrrd")
        sitk.WriteImage(predicted,str(temporary_prediction),True)
        temporary_prediction.replace(destination)
        row["prediction_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        atomic_json(target,row)
        print(json.dumps(row),flush=True)
        rows.append(row)
    summary = {"overall":summarize(rows),"smoke":args.smoke,"fold":args.fold,"method":config["method"],
               "evaluation_split":args.split,"evaluation_role":evaluation_role,"formal_final_test":False,
               "checkpoint_selection":checkpoint_selection,"checkpoint_epoch":checkpoint_epoch,
               "internal_selection_valid":config.get("internal_selection_valid"),
               "training_source_verified_against_live_code":training_source_verified,"geometry_only":args.geometry_only,
               "case_ids":ids,"metric_protocol":METRIC_PROTOCOL,"primary_metrics":["dice","assd_mm"]}
    for group in ("Normal","Diseased"):
        selected = [r for r in rows if r["group"]==group]
        if selected:
            summary[group] = summarize(selected)
    if not args.geometry_only or training_source_verified:
        verify_training_source(config)
    atomic_json(out/"summary.json",summary)
    ledger = out/"case_metrics.jsonl"
    temporary_ledger = ledger.with_suffix(".jsonl.tmp")
    temporary_ledger.write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in rows),encoding="utf-8")
    temporary_ledger.replace(ledger)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
