"""Verify one frozen eight-case development ledger against cold-start GTA.

Exit 0 means the numerical DEVELOPMENT gate passed, not formal study success.
Exit 1 means a verified candidate did not pass; exit 2 means invalid provenance.
No model loading, training, inference or clinical claim is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk

from metrics import ASSD_POOLED_DEFINITION, METRIC_PROTOCOL_VERSION
from aggregate_results import atomic_json, load_protocol


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def source_identity(root, expected, algorithm=None):
    root = Path(root).resolve()
    if isinstance(expected,dict):
        actual = {}
        for name in expected:
            path = (root/name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source metadata escapes the supplied frozen source snapshot")
            actual[name] = digest(path)
        return actual
    # Reproduce the explicitly versioned source identity of the v2 trainer.
    if algorithm != "gta_fidelity_selected_python_path_and_bytes_v1":
        raise ValueError("Scalar source hash requires the declared v2 training hash algorithm")
    used = {"train.py", "model.py", "losses.py", "classification_data.py", "checkpoint_utils.py", "data.py"}
    paths = sorted((p for p in root.rglob("*.py") if "__pycache__" not in p.parts),
                   key=lambda p:p.relative_to(root).as_posix())
    paths = [p for p in paths if p.name in used or "historical" in p.relative_to(root).parts]
    if not paths:
        raise ValueError("Frozen source snapshot contains no Python files")
    value = hashlib.sha256()
    for path in paths:
        value.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    return value.hexdigest()


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def verify_geometry(path, geometry):
    # Windows ITK NRRD fopen cannot reliably read a Unicode path. Copying a
    # compressed prediction to an ASCII temporary path is read-only validation.
    if os.name=="nt" and not str(path).isascii():
        with tempfile.TemporaryDirectory(prefix="asoca_geometry_header_") as temporary:
            copy=Path(temporary)/"prediction.nrrd"
            if not str(copy).isascii():
                raise ValueError("An ASCII temporary directory is required by this Windows NRRD reader")
            shutil.copy2(path,copy)
            return verify_geometry(copy,geometry)
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if list(reader.GetSize()) != geometry["size_xyz"]:
        raise ValueError("Saved prediction size differs from original CT")
    for actual,key in ((reader.GetSpacing(),"spacing_xyz_mm"),(reader.GetOrigin(),"origin_xyz_mm"),
                       (reader.GetDirection(),"direction")):
        if not np.allclose(actual,geometry[key],rtol=0,atol=1e-6):
            raise ValueError("Saved prediction original-grid geometry differs at "+key)


def verify_candidate(candidate, manifest_path, config_path, checkpoint, source_root,
                     raw_root, reference_path, prediction_root=None):
    candidate, source_root = Path(candidate), Path(source_root).resolve()
    if candidate.is_dir():
        candidate = candidate/"case_metrics.jsonl"
    prediction_root = Path(prediction_root) if prediction_root else candidate.parent
    rows = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest, manifest_hash = load_protocol(Path(manifest_path))
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected_ids = sorted(manifest["folds"][0]["val"])
    if len(rows)!=8 or sorted(r["case_id"] for r in rows)!=expected_ids or len({r["case_id"] for r in rows})!=8:
        raise ValueError("Candidate must contain exactly the eight frozen outer-development patients")
    if reference["manifest_sha256"]!=manifest_hash or sorted(reference["case_ids"])!=expected_ids:
        raise ValueError("Cold-start reference belongs to another manifest or patient cohort")
    if reference.get("assd_definition") != ASSD_POOLED_DEFINITION:
        raise ValueError("Cold-start ASSD convention differs")
    if config.get("fold")!=0 or config.get("smoke") is not False or config.get("manifest_sha256")!=manifest_hash:
        raise ValueError("Candidate training configuration has incompatible fold/smoke/manifest")
    if type(config.get("epochs")) is not int or config["epochs"] <= 0:
        raise ValueError("Candidate must record a positive, fixed training endpoint")
    training_source = config.get("source_sha256")
    if not training_source or source_identity(source_root,training_source,config.get("source_hash_algorithm"))!=training_source:
        raise ValueError("Candidate training source differs from the frozen source snapshot")
    split = config.get("internal_split",{})
    training_ids, diagnostic_ids = split.get("train",[]), split.get("val",[])
    if (len(set(training_ids))!=28 or len(training_ids)!=28 or len(set(diagnostic_ids))!=4 or len(diagnostic_ids)!=4
            or set(training_ids)&set(diagnostic_ids)
            or set(training_ids)|set(diagnostic_ids)!=set(manifest["folds"][0]["train"])):
        raise ValueError("Internal development split must preserve the frozen 28/4 partition of outer32")
    trained_ids = config.get("trained_case_ids")
    if not trained_ids or len(trained_ids)!=len(set(trained_ids)) or not set(trained_ids)<=set(manifest["folds"][0]["train"]):
        raise ValueError("Candidate training patients must be explicitly recorded within frozen outer32")
    if config.get("outer8_used_for_selection") is not False:
        raise ValueError("Outer-eight model-selection status must be explicitly false")
    checkpoint_hash = digest(checkpoint)
    metric_source_hash = digest(source_root/"metrics.py")
    evaluator_source_hash = digest(source_root/"evaluate.py")
    cases = {c["case_id"]:c for c in manifest["cases"]}
    if any(sum(cases[c]["group"]==g for c in diagnostic_ids)!=2 for g in ("Normal","Diseased")):
        raise ValueError("Internal four must contain two patients from each stratum")
    baseline = {r["case_id"]:r for r in reference["methods"]["gta"]["cases"]}
    if len(baseline)!=8:
        raise ValueError("Cold-start reference is not complete")
    pairs=[]
    for row in rows:
        case_id=row["case_id"]; case=cases[case_id]; old=baseline[case_id]
        checks={"group":case["group"],"fold":0,"smoke":False,"metric_grid":"original CT",
                "formal_final_test":False,"evaluation_split":"outer_development",
                "checkpoint_selection":"fixed final","checkpoint_epoch":config.get("epochs",0)-1,
                "training_source_verified_against_live_code":True,"geometry_only":False,
                "metric_protocol_version":METRIC_PROTOCOL_VERSION,"assd_definition":ASSD_POOLED_DEFINITION,
                "manifest_sha256":manifest_hash,"checkpoint_sha256":checkpoint_hash,
                "training_source_sha256":training_source,"metrics_source_sha256":metric_source_hash,
                "evaluation_source_sha256":evaluator_source_hash,"method":config["method"],
                "label_sha256":case["source_sha256"]["label"]}
        if any(row.get(k)!=v for k,v in checks.items()):
            raise ValueError("Incompatible candidate metadata for "+case_id)
        if old["label_sha256"]!=row["label_sha256"]:
            raise ValueError("Candidate and reference use different labels")
        label_path=Path(raw_root)/case["label_relative_path"]
        if digest(label_path)!=row["label_sha256"]:
            raise ValueError("Raw label differs from audited data for "+case_id)
        prediction=prediction_root/(case_id+"_prediction.nrrd")
        if not row.get("prediction_sha256") or digest(prediction)!=row["prediction_sha256"]:
            raise ValueError("Saved prediction identity differs for "+case_id)
        verify_geometry(prediction,case["original_geometry"])
        if row.get("empty_prediction") is not False or not finite(row.get("dice")) or not finite(row.get("assd_mm")):
            raise ValueError("Empty prediction or undefined primary distance prevents a complete-cohort gate")
        if not 0<=row["dice"]<=1 or row["assd_mm"]<0:
            raise ValueError("Invalid primary metric for "+case_id)
        pairs.append({"case_id":case_id,"group":case["group"],"candidate_dice":row["dice"],
                      "reference_dice":old["dice"],"delta_dice":row["dice"]-old["dice"],
                      "candidate_assd_mm":row["assd_mm"],"reference_assd_mm":old["assd_mm"],
                      "delta_assd_mm":row["assd_mm"]-old["assd_mm"]})
    if {g:sum(r["group"]==g for r in rows) for g in ("Normal","Diseased")}!={"Normal":4,"Diseased":4}:
        raise ValueError("Eight-case group distribution changed")
    metrics={}
    for name in ("dice","assd_mm"):
        values=np.array([r[name] for r in rows]); old=np.array([baseline[r["case_id"]][name] for r in rows])
        metrics[name]={"candidate_mean":float(values.mean()),"candidate_sd":float(values.std(ddof=1)),
                       "reference_mean":float(old.mean()),"reference_sd":float(old.std(ddof=1)),
                       "paired_mean_difference":float((values-old).mean()),"n_cases":8}
    dice_improved=metrics["dice"]["candidate_mean"]>metrics["dice"]["reference_mean"]+1e-12
    assd_not_worse=metrics["assd_mm"]["candidate_mean"]<=metrics["assd_mm"]["reference_mean"]+1e-9
    return {"metadata_verified":True,"development_only":True,"formal_final_test":False,
            "development_gate_passed":bool(dice_improved and assd_not_worse),
            "gate":{"dice_increased":bool(dice_improved),"pooled_assd_not_worse":bool(assd_not_worse),
                    "numeric_tolerance_dice":1e-12,"numeric_tolerance_assd_mm":1e-9},
            "candidate_method":config["method"],"n_cases":8,"case_ids":expected_ids,
            "metrics":metrics,"paired_cases":sorted(pairs,key=lambda r:r["case_id"]),
            "manifest_sha256":manifest_hash,"checkpoint_sha256":checkpoint_hash,
            "training_source_sha256":training_source,"candidate_ledger_sha256":digest(candidate),
            "reference_ledger_sha256":digest(reference_path),"assd_definition":ASSD_POOLED_DEFINITION,
            "trained_case_ids":sorted(trained_ids),"n_training_cases":len(trained_ids),
            "internal_diagnostic_case_ids":sorted(diagnostic_ids),
            "internal_selection_valid":config.get("internal_selection_valid"),
            "interpretation":"Numerical development gate only. No significance, generalization, paper-target or clinical-success claim."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("candidate","manifest","config","checkpoint","source-root","raw-root","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--prediction-root")
    parser.add_argument("--reference",default=str(Path(__file__).parent.parent/"docs"/"cold_start_fold0_reference.json"))
    args=parser.parse_args()
    try:
        result=verify_candidate(args.candidate,args.manifest,args.config,args.checkpoint,args.source_root,
                                args.raw_root,args.reference,args.prediction_root)
    except (ValueError,KeyError,OSError,TypeError) as error:
        result={"metadata_verified":False,"development_gate_passed":False,"development_only":True,
                "formal_final_test":False,"error":str(error)}
        atomic_json(Path(args.output),result)
        print(json.dumps(result),flush=True)
        return 2
    atomic_json(Path(args.output),result)
    print(json.dumps({k:result[k] for k in ("metadata_verified","development_gate_passed","development_only","metrics")}),flush=True)
    return 0 if result["development_gate_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
