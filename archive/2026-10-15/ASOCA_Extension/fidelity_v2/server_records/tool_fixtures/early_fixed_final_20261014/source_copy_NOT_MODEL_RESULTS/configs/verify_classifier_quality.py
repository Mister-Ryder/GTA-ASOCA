"""Verify fixed-final classifier readiness, never a classification-score gate.

The filename is retained for deployment compatibility. Classification provides
prior features for segmentation and is not the main optimization objective.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    project = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project / "code"))
    import torch
    from checkpoint_utils import read_checkpoint
    from classification_data import internal_split
    from data import load_manifest
    folder = Path(args.training)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    marker = read(folder / "CLASSIFIER_TRAINING_COMPLETE.json")
    config = read(folder / "config.json")
    checkpoint, state, format_key = read_checkpoint(args.checkpoint)
    split = internal_split(load_manifest(args.manifest), 0, 20261014)
    manifest_hash, weight_hash = sha(args.manifest), sha(args.checkpoint)
    stage_code = project / "classifier_stage/code"
    provenance = read(project / "classifier_stage/source_provenance.json")["source_hashes"]
    for name, expected in provenance.items():
        if sha(stage_code / name) != expected:
            raise ValueError("Immutable classifier-stage source changed: " + name)
    paths = sorted((path for path in stage_code.rglob("*.py") if "__pycache__" not in path.parts),
                   key=lambda path: path.relative_to(stage_code).as_posix())
    stage_hash = hashlib.sha256(b"".join(path.relative_to(stage_code).as_posix().encode() + path.read_bytes() for path in paths)).hexdigest()
    if (config.get("smoke") is not False or marker.get("smoke") is not False or config.get("pretrained_classifier") is not True
            or config.get("epochs") != 80 or config.get("steps") != 64 or config.get("fold") != 0
            or config.get("additional_pretraining_updates") != 5120 or marker.get("epoch") != 79
            or checkpoint.get("epoch") != 79 or marker.get("final_sha256") != weight_hash
            or config.get("manifest_sha256") != manifest_hash or marker.get("manifest_sha256") != manifest_hash
            or config.get("source_sha256") != stage_hash or marker.get("source_sha256") != stage_hash):
        raise ValueError("Expected completed nonsmoke80x64 fixed-final classifier with verified source/manifest identity")
    if (config.get("internal_split") != split or config.get("trained_case_ids") != split["train"]
            or marker.get("trained_case_ids") != split["train"] or marker.get("selection_case_ids") != split["val"]
            or config.get("outer8_used_for_selection") is not False or marker.get("outer8_used_for_selection") is not False
            or set(split["train"]) & (set(split["val"]) | set(split["outer_development"]))):
        raise ValueError("Classifier patient provenance differs or leaks into held-out cohorts")
    saved = checkpoint.get("config", {})
    if any(saved.get(key) != value for key, value in config.items() if key not in ("resume", "plan")):
        raise ValueError("Fixed-final checkpoint config differs from recorded training config")
    history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines() if line.strip()]
    if ([row.get("epoch") for row in history] != list(range(80))
            or any(not isinstance(row.get("loss"), (int, float)) or not math.isfinite(row["loss"]) for row in history)):
        raise ValueError("Classifier training history is incomplete or contains nonfinite losses")
    if not state or any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Classifier parameters are missing or nonfinite")
    print(json.dumps({"status": "classifier_fixed_final_readiness_verified", "checkpoint_sha256": weight_hash,
                      "format_key": format_key, "epoch": 79, "additional_classifier_updates": 5120,
                      "classification_scores_are_diagnostic_only": True, "classification_accuracy_threshold": None,
                      "human_quality_approval_required": False, "primary_downstream_metrics": ["dice", "assd_mm"],
                      "development_only": True}), flush=True)


if __name__ == "__main__":
    main()
