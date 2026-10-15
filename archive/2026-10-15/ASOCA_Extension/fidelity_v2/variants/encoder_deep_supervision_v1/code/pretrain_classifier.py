"""Train the unchanged three-head classifier on 28 patients; select on internal four.

The original eight development cases are never used for early checkpoint choice.
Loss alone is not evidence that a classifier is useful: per-head confusion,
balanced accuracy, absent classes and training-majority baselines are reported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from checkpoint_utils import load_audited, sha256_file
from classification_data import (CohortDataset, NAMES, internal_split,
                                 training_attribute_statistics, classification_report)
from data import load_manifest
from historical.func.AnatomicalClassifier import AirwayAnatomicalClassifier
from losses import focal_loss


def source_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join(p.relative_to(root).as_posix().encode() + p.read_bytes()
                                  for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_atomic(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def evaluate(model, dataset, device, batch_size, workers, statistics):
    model.eval()
    truth, prediction = [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers):
            output = model(batch["image"].to(device))
            truth.append(batch["priors"].numpy())
            prediction.append(torch.stack([output[name].argmax(1) for name in NAMES], 1).cpu().numpy())
    return classification_report(np.concatenate(truth), np.concatenate(prediction),
                                 [statistics["heads"][name]["majority_class"] for name in NAMES], statistics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--split-seed", type=int, default=20261014)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--stats-samples", type=int, default=2048)
    parser.add_argument("--validation-patches", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--sampling", choices=("standard", "geometry"), default="geometry")
    parser.add_argument("--sampling-fov", choices=("full", "image-body"), default="image-body")
    parser.add_argument("--guidance-weight", type=float, default=.1)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-domain", choices=("airway", "asoca"), default="airway")
    parser.add_argument("--semantic-head-policy", choices=("reinit", "load"), default="reinit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnostic-original-dev", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch_size, args.stats_samples, args.validation_patches, args.eval_every) < 1:
        parser.error("Training budgets and sample counts must be positive")
    manifest = load_manifest(args.manifest)
    split = internal_split(manifest, args.fold, args.split_seed)
    if args.plan:
        print(json.dumps({"internal_split": split, "epochs": args.epochs, "updates": args.epochs * args.steps,
                          "selection": "mean three-head balanced accuracy on fixed internal-four image-only patch bank",
                          "outer8_used_for_selection": False}, indent=2))
        return
    output = Path(args.output)
    if output.exists() and (output / "config.json").exists() and not args.resume:
        raise RuntimeError("Output already exists; use a new folder or explicit compatible resume")
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and not (output / "latest.pt").exists():
        raise RuntimeError("Resume checkpoint is missing")
    device = torch.device(args.device)
    seed = args.seed + args.fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no silent CPU fallback")
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    common = dict(fold=args.fold, patch_size=args.patch_size, seed=seed,
                  sampling_fov=args.sampling_fov)
    train = CohortDataset(manifest, split["train"], role="train", sampling=args.sampling,
                          steps=args.steps * args.batch_size, augment=True, **common)
    statistic_bank = CohortDataset(manifest, split["train"], role="train", sampling=args.sampling,
                                   steps=args.stats_samples, augment=False, **common)
    selection = CohortDataset(manifest, split["val"], role="selection", sampling="uniform",
                              steps=args.validation_patches, augment=False, **common)
    statistics = training_attribute_statistics(statistic_bank, args.stats_samples)
    model = AirwayAnatomicalClassifier()
    init_audit = None
    if args.init_checkpoint and not args.resume:
        init_audit = load_audited(model, args.init_checkpoint, "classifier", args.init_domain, args.semantic_head_policy)
        if args.init_domain == "asoca":
            # Old fold-0 checkpoints used all 32 patients, including internal four.
            # Until patient provenance is positively verified, such transfer must
            # not silently contaminate this four-patient selection protocol.
            from checkpoint_utils import read_checkpoint
            source, _, _ = read_checkpoint(args.init_checkpoint)
            source_config = source.get("config", {})
            source_ids = source_config.get("trained_case_ids", source_config.get("internal_split", {}).get("train"))
            if not source_ids or not set(source_ids).issubset(split["train"]):
                raise ValueError("ASOCA classifier initialization has missing or overlapping internal-selection patient provenance")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024, growth_interval=1000000)
    hashes = {"manifest_sha256": sha256_file(Path(manifest["_manifest_path"])), "source_sha256": source_hash()}
    config = vars(args) | hashes | {"stage": "standalone_prior_pretraining", "pretrained_classifier": True,
              "internal_split": split, "trained_case_ids": split["train"], "selection_case_ids": split["val"],
              "attribute_statistics": statistics, "init_audit": init_audit,
              "selection_metric": "mean_head_nonempty_balanced_accuracy", "outer8_used_for_selection": False,
              "sampling_fov_definition": "uniform centres in CT-only HU>-170 body bounds; full CT remains inference field",
              "external_pretraining": "BAS airway feature warm-start" if init_audit and args.init_domain == "airway" else None,
              "additional_pretraining_updates": args.epochs * args.steps,
              "budget_excludes_GTA_segmentation_updates": True}
    start, best = 0, -math.inf
    if args.resume:
        old = json.loads((output / "config.json").read_text())
        for key in tuple(key for key in vars(args) if key not in ("resume", "plan")) + ("manifest_sha256", "source_sha256", "internal_split"):
            if old.get(key) != config.get(key):
                raise ValueError("Resume protocol changed: " + key)
        with torch.serialization.safe_globals([argparse.Namespace]):
            saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if any(saved.get("config", {}).get(key) != value for key, value in old.items() if key not in ("resume", "plan")):
            raise ValueError("latest.pt configuration differs from recorded config.json")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start, best = saved["epoch"] + 1, saved["best_selection"]
        torch.set_rng_state(saved["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        config["init_audit"] = old.get("init_audit")
    if start >= args.epochs:
        marker = json.loads((output / "CLASSIFIER_TRAINING_COMPLETE.json").read_text())
        if marker.get("final_sha256") != sha256_file(output / "final.pt") or marker.get("best_selection_sha256") != sha256_file(output / "best_selection.pt"):
            raise ValueError("Completed classifier checkpoints changed")
        print("CLASSIFIER_TRAINING_COMPLETE_VERIFIED", flush=True)
        return
    alpha = {name: torch.tensor(statistics["heads"][name]["focal_weights"], device=device) for name in NAMES}
    atomic_json(output / "internal_split.json", split)
    atomic_json(output / "attribute_statistics.json", statistics)
    atomic_json(output / "config.json", config)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    print(json.dumps({"config": config}), flush=True)
    for epoch in range(start, args.epochs):
        if source_hash() != hashes["source_sha256"] or sha256_file(Path(manifest["_manifest_path"])) != hashes["manifest_sha256"]:
            raise RuntimeError("Source or manifest changed during pretraining")
        train.set_epoch(epoch); model.train(); losses = []; components = []; began = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, batch in enumerate(loader):
            iteration = epoch * args.steps + step
            total, warm = args.epochs * args.steps, min(args.warmup_epochs * args.steps, args.epochs * args.steps)
            factor = iteration / warm if warm and iteration < warm else .5 * (1 + math.cos(math.pi * (iteration - warm) / max(1, total - warm)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            image, priors, label = batch["image"].to(device), batch["priors"].to(device), batch["label"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
                result = model(image)
                head_losses = {name: focal_loss(result[name], priors[:, head], class_weights=alpha[name]) for head, name in enumerate(NAMES)}
                classification = head_losses["generation"] + 1.5 * head_losses["structure"] + head_losses["location"]
                guide_target = F.avg_pool3d(label[:, None].float(), kernel_size=4, stride=4)
                guide_prediction = F.interpolate(result["guidance_map"].float(), size=guide_target.shape[2:], mode="trilinear", align_corners=False)
                guide = F.mse_loss(guide_prediction, guide_target.expand_as(guide_prediction))
                loss = classification + args.guidance_weight * guide
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classifier loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite classifier gradients")
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
            terms = {"classification": float(classification.detach()), "guidance_mse": float(guide.detach()),
                     "guidance_weighted": float((args.guidance_weight * guide).detach()),
                     "gradient_norm_before_clip": float(norm), **{name + "_focal": float(value.detach()) for name, value in head_losses.items()}}
            components.append(terms)
            if step % 25 == 0 or args.smoke:
                print(json.dumps({"epoch": epoch, "step": step, "lr": optimizer.param_groups[0]["lr"], "loss": losses[-1], "components": terms}), flush=True)
        scheduled = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        report = evaluate(model, selection, device, args.batch_size, args.workers, statistics) if scheduled else None
        if report is not None and not report["selection_positive_coverage_valid"] and not args.smoke:
            raise ValueError("Internal selection patch bank contains no positive class for a head; no valid prior selection is possible")
        score = (report["mean_head_nonempty_balanced_accuracy"] if report["selection_positive_coverage_valid"] else report["mean_head_balanced_accuracy"]) if report is not None else -math.inf
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.monotonic() - began,
               "selection": report, "selection_cases": split["val"], "lr": optimizer.param_groups[0]["lr"],
               "component_means": {key: float(np.mean([item[key] for item in components])) for key in components[0]},
               "guidance_lambda": args.guidance_weight,
               "gpu_peak_GiB": float(torch.cuda.max_memory_allocated(device) / 2 ** 30) if device.type == "cuda" else None}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        value = {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "config": config, "best_selection": max(best, score), "selection_report": report,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if score > best:
            best = score
            save_atomic(value, output / "best_selection.pt")
        save_atomic(value, output / "latest.pt")
        print(json.dumps(row), flush=True)
    save_atomic(value, output / "final.pt")
    atomic_json(output / "final_selection_report.json", report)
    if args.diagnostic_original_dev:
        diagnostic = CohortDataset(manifest, split["outer_development"], role="outer_diagnostic", sampling="uniform",
                                   steps=args.validation_patches, augment=False, **common)
        diagnostic_report = evaluate(model, diagnostic, device, args.batch_size, args.workers, statistics)
        diagnostic_report.update(role=split["outer_role"], used_for_checkpoint_selection=False)
        atomic_json(output / "outer_development_diagnostic.json", diagnostic_report)
    atomic_json(output / "CLASSIFIER_TRAINING_COMPLETE.json",
                {"epoch": args.epochs - 1, "smoke": args.smoke, **hashes, "trained_case_ids": split["train"],
                 "selection_case_ids": split["val"], "final_sha256": sha256_file(output / "final.pt"),
                 "best_selection_sha256": sha256_file(output / "best_selection.pt"),
                 "selection_metric": "mean_head_nonempty_balanced_accuracy", "best_selection_value": best,
                 "selection_positive_coverage_valid": report["selection_positive_coverage_valid"],
                 "smoke_invalid_positive_bank_is_not_prior_evidence": args.smoke and not report["selection_positive_coverage_valid"],
                 "outer8_used_for_selection": False})
    print("CLASSIFIER_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
