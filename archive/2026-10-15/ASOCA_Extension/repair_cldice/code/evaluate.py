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
from metrics import binary_metrics
from train import make_model


def original_prediction(prediction, case, reference):
    image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    geometry = case["resampled_geometry"]
    image.SetSpacing(geometry["spacing_xyz_mm"])
    image.SetOrigin(geometry["origin_xyz_mm"])
    image.SetDirection(geometry["direction"])
    return sitk.Resample(image,reference,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)


def summarize(rows):
    fields = [k for k,v in rows[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
    output = {"n_cases":len(rows),"empty_predictions":sum(bool(r["empty_prediction"]) for r in rows)}
    for field in fields:
        values = [r[field] for r in rows if r.get(field) is not None]
        output[field] = {"mean":float(np.mean(values)),"sd":float(np.std(values,ddof=1)) if len(values)>1 else 0.,"n_valid":len(values)}
    # Undefined geometry distances must not be silently excluded from a mean.
    if output["empty_predictions"]:
        for key in ("surface_hd95_mm","assd_mm","asoca_foreground_hd95_mm"):
            output[key] = {"mean":None,"reason":"empty predictions present; inspect per-case results"}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--raw-root",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--fold",type=int,default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--max-cases",type=int)
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
    model = make_model(config["method"],not config["no_ds"]).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    source_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    cases = {c["case_id"]:c for c in manifest["cases"]}
    ids = manifest["folds"][args.fold]["val"]
    if args.smoke and manifest.get("pilot_only"):
        ids = list(cases)
    if args.max_cases:
        if not args.smoke:
            raise ValueError("Partial evaluation allowed only for smoke")
        ids = ids[:args.max_cases]
    rows = []
    for case_id in ids:
        case = cases[case_id]
        target = out/(case_id+".json")
        if target.exists():
            row = json.loads(target.read_text())
            if row["checkpoint_sha256"] != source_hash or row.get("manifest_sha256") != manifest_hash:
                raise RuntimeError("Output contains metrics from different weights")
            rows.append(row)
            continue
        started = time.time()
        array = np.load(case_directory(manifest,case)/"image.npy",mmap_mode="r")
        inputs = torch.from_numpy(np.asarray(array,dtype=np.float32).copy())[None,None]
        def predict(x):
            with torch.amp.autocast("cuda",dtype=torch.float16):
                return model(x)["logits"].float()
        with torch.inference_mode():
            logits = sliding_window_inference(inputs,(config["patch_size"],)*3,2,predict,overlap=.5,
                                              mode="gaussian",sw_device="cuda",device="cpu",cval=-1.)
        mask = logits.argmax(1)[0].numpy().astype(np.uint8)
        reference = sitk.ReadImage(str(Path(args.raw_root)/case["label_relative_path"]))
        label_path = Path(args.raw_root)/case["label_relative_path"]
        label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
        if label_hash != case["source_sha256"]["label"]:
            raise ValueError("Raw reference differs from audited label: " + case_id)
        predicted = original_prediction(mask,case,reference)
        truth = sitk.GetArrayFromImage(reference)
        prediction = sitk.GetArrayFromImage(predicted)
        row = binary_metrics(prediction,truth,reference.GetSpacing()[::-1])
        row.update(case_id=case_id,group=case["group"],fold=args.fold,method=config["method"],
                   checkpoint_sha256=source_hash,manifest_sha256=manifest_hash,label_sha256=label_hash,seconds=round(time.time()-started,2),
                   smoke=args.smoke,metric_grid="original CT",checkpoint_selection="fixed final")
        destination = out/(case_id+"_prediction.nrrd")
        temporary_prediction = out/(case_id+"_prediction.tmp.nrrd")
        sitk.WriteImage(predicted,str(temporary_prediction),True)
        temporary_prediction.replace(destination)
        row["prediction_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        temporary_json = target.with_suffix(".json.tmp")
        temporary_json.write_text(json.dumps(row,indent=2),encoding="utf-8")
        temporary_json.replace(target)
        print(json.dumps(row),flush=True)
        rows.append(row)
    summary = {"overall":summarize(rows),"smoke":args.smoke,"fold":args.fold,"method":config["method"]}
    for group in ("Normal","Diseased"):
        selected = [r for r in rows if r["group"]==group]
        if selected:
            summary[group] = summarize(selected)
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
