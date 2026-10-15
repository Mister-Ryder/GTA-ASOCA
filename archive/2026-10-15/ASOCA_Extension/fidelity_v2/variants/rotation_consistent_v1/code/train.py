"""GTA v2 transfer/staging and shared-recipe U-Net diagnostics, internal28 only.

Enhanced inference architecture is unchanged. Historical encoder/decoder double
supervision is optional backbone warm-up. Four internal patients are reserved
for development selection; the original eight are diagnostic, never a blind test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from checkpoint_utils import load_audited, read_checkpoint, sha256_file
from classification_data import CohortDataset, NAMES, internal_split
from data import load_manifest
from model import build_model
from losses import objective, dice_ce_loss, segmentation_loss


def make_model(method, ds=True):
    return build_model(method, deep_supervision=ds)


SOURCE_HASH_ALGORITHM = "gta_fidelity_selected_python_path_and_bytes_v1"


def source_paths():
    root = Path(__file__).resolve().parent
    used = {"train.py", "model.py", "losses.py", "classification_data.py", "checkpoint_utils.py", "data.py"}
    return [p for p in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix())
            if "__pycache__" not in p.parts and (p.name in used or "historical" in p.relative_to(root).parts)]


def source_hash():
    root = Path(__file__).resolve().parent
    paths = source_paths()
    return hashlib.sha256(b"".join(p.relative_to(root).as_posix().encode() + p.read_bytes() for p in paths)).hexdigest()


def save_atomic(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); temporary.replace(path)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"); temporary.replace(path)


def gradient_norm(parameters):
    terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    p.add_argument("--method", choices=("gta", "gta_no_prior", "gta_no_gaussian", "gta_no_balance", "unet"), default="gta")
    p.add_argument("--fold", type=int, default=0); p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps", type=int, default=125); p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=2); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261013); p.add_argument("--split-seed", type=int, default=20261014)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5); p.add_argument("--freeze-epochs", type=int, default=50)
    p.add_argument("--partial-until", type=int, default=70); p.add_argument("--classifier-lr-scale", type=float, default=.1)
    p.add_argument("--classifier-weight", type=float, default=.1); p.add_argument("--gaussian-weight", type=float, default=10.)
    p.add_argument("--gaussian-late-weight", type=float)
    p.add_argument("--gaussian-ramp-start", type=int, default=150)
    p.add_argument("--gaussian-ramp-end", type=int, default=200)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--gradient-probe-every", type=int, default=250)
    p.add_argument("--guidance-weight", type=float, default=.1)
    p.add_argument("--seg-loss", choices=("dice_ce", "historical"), default="dice_ce")
    p.add_argument("--sampling", choices=("standard", "geometry"), default="standard")
    p.add_argument("--sampling-fov", choices=("full", "image-body"), default="image-body")
    p.add_argument("--spatial-augmentation", choices=("none", "xy90"), default="none")
    p.add_argument("--disable-anatomy-reweighting", action="store_true")
    p.add_argument("--cohort", choices=("internal28", "train32"), default="internal28")
    p.add_argument("--backbone-warmup-epochs", type=int, default=0)
    p.add_argument("--backbone-init"); p.add_argument("--backbone-domain", choices=("airway", "asoca"), default="airway")
    p.add_argument("--classifier-init"); p.add_argument("--classifier-domain", choices=("airway", "asoca"), default="asoca")
    p.add_argument("--classifier-head-policy", choices=("reinit", "load"), default="load")
    p.add_argument("--full-init"); p.add_argument("--full-domain", choices=("airway", "asoca"), default="airway")
    p.add_argument("--semantic-head-policy", choices=("reinit", "load"), default="reinit")
    p.add_argument("--allow-selection-overlap", action="store_true", help="Explicit diagnostic transfer; invalidates internal selection")
    p.add_argument("--allow-untrained-classifier", action="store_true", help="Explicit from-scratch negative control/smoke")
    p.add_argument("--no-ds", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--plan", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.steps, args.patch_size, args.batch_size) <= 0 or args.patch_size % 16:
        p.error("Positive budgets and patch-size divisible by sixteen are required")
    if not 0 <= args.freeze_epochs <= args.partial_until or not 0 <= args.backbone_warmup_epochs <= args.epochs or args.warmup_epochs < 0:
        p.error("Invalid stage boundaries")
    if min(args.classifier_weight, args.gaussian_weight, args.guidance_weight, args.classifier_lr_scale) < 0 or (args.gaussian_late_weight is not None and args.gaussian_late_weight < 0):
        p.error("Loss coefficients and classifier LR scale must be nonnegative")
    if args.save_every < 1 or args.gradient_probe_every < 0 or (args.gaussian_late_weight is not None and not 0 <= args.gaussian_ramp_start < args.gaussian_ramp_end <= args.epochs):
        p.error("Invalid checkpoint cadence or Gaussian ramp endpoints")
    if args.full_init and args.backbone_init:
        p.error("Choose full or backbone donor; do not silently overwrite the same component twice")
    if args.method == "unet" and args.backbone_warmup_epochs:
        p.error("Encoder/decoder backbone warm-up belongs to the GTA backbone")
    manifest = load_manifest(args.manifest); split = internal_split(manifest, args.fold, args.split_seed)
    train_ids = split["train"] if args.cohort == "internal28" else split["original_train32"]
    if args.plan:
        print(json.dumps({"args": vars(args), "internal_split": split, "trained_case_ids": train_ids,
                          "segmentation_updates": args.epochs * args.steps, "outer8_used_for_selection": False}, indent=2)); return
    output = Path(args.output)
    if (output / "config.json").exists() and not args.resume:
        raise RuntimeError("Existing run is immutable; use explicit resume or a new output folder")
    if args.resume and not (output / "latest.pt").exists():
        raise RuntimeError("Resume checkpoint missing")
    output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic fallback")
    seed = args.seed + args.fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    model = make_model(args.method, not args.no_ds); uses_prior = bool(getattr(model, "use_prior", False))
    init_audits, selection_valid = [], args.cohort == "internal28"
    pretrained_prior, attribute_weights, extra_classifier_updates = False, {}, 0

    def verify_donor_patient_scope(path, domain):
        nonlocal selection_valid
        if domain != "asoca": return
        donor, _, _ = read_checkpoint(path); cfg = donor.get("config", {})
        donor_ids = cfg.get("trained_case_ids", cfg.get("internal_split", {}).get("train"))
        if donor_ids is None and cfg.get("manifest_sha256") == sha256_file(Path(manifest["_manifest_path"])):
            donor_ids = manifest["folds"][cfg["fold"]]["train"]
        if donor_ids is not None and not set(donor_ids).issubset(split["original_train32"]):
            raise ValueError("ASOCA donor used original-eight development patients or other unapproved patients")
        if not donor_ids or set(donor_ids) & set(split["val"]):
            if args.cohort == "internal28" and not args.allow_selection_overlap:
                raise ValueError("ASOCA donor saw internal selection patients or has missing patient provenance")
            selection_valid = False

    if not args.resume:
        if args.full_init:
            if args.method == "unet": raise ValueError("Full GTA checkpoint cannot initialize another U-Net architecture")
            verify_donor_patient_scope(args.full_init, args.full_domain)
            init_audits.append(load_audited(model, args.full_init, "full", args.full_domain, args.semantic_head_policy,
                                            allow_missing_prefixes=("ds_heads.",), allow_unexpected_prefixes=("adaptive_threshold.",)))
        if args.backbone_init:
            if not hasattr(model, "base_model"): raise ValueError("Original GTA backbone cannot initialize a different U-Net implementation")
            verify_donor_patient_scope(args.backbone_init, args.backbone_domain)
            init_audits.append(load_audited(model.base_model, args.backbone_init, "backbone", args.backbone_domain))
        if args.classifier_init:
            if not uses_prior: raise ValueError("This model/ablation has no classifier")
            verify_donor_patient_scope(args.classifier_init, args.classifier_domain)
            donor, _, _ = read_checkpoint(args.classifier_init); cfg = donor.get("config", {})
            pretrained_prior = (args.classifier_domain == "asoca" and cfg.get("pretrained_classifier") is True
                                and cfg.get("manifest_sha256") == sha256_file(Path(manifest["_manifest_path"]))
                                and args.classifier_head_policy == "load" and (not cfg.get("smoke", False) or args.smoke))
            init_audits.append(load_audited(model.classifier, args.classifier_init, "classifier", args.classifier_domain,
                                            args.classifier_head_policy))
            if pretrained_prior:
                attribute_weights = {name: torch.tensor(cfg["attribute_statistics"]["heads"][name]["focal_weights"], device=device) for name in NAMES}
                extra_classifier_updates = cfg["additional_pretraining_updates"]
        if uses_prior and not pretrained_prior and not args.allow_untrained_classifier:
            raise ValueError("GTA requires the separately trained ASOCA classifier; explicit controls may allow an untrained classifier")
    model.to(device)
    classifier_parameters = list(model.classifier.parameters()) if uses_prior else []
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    segmentation_parameters = [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids]
    groups = [{"params": segmentation_parameters, "lr": args.lr, "name": "segmentation"}]
    if classifier_parameters: groups.append({"params": classifier_parameters, "lr": args.lr * args.classifier_lr_scale, "name": "classifier"})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024, growth_interval=1000000)
    manifest_hash, code_hash = sha256_file(Path(manifest["_manifest_path"])), source_hash()
    config = vars(args) | {"source_sha256": code_hash, "source_hash_algorithm": SOURCE_HASH_ALGORITHM,
             "source_files_sha256": {path.relative_to(Path(__file__).resolve().parent).as_posix(): sha256_file(path) for path in source_paths()},
             "manifest_sha256": manifest_hash,
             "parameters": sum(parameter.numel() for parameter in model.parameters()), "stage": "fidelity_v2_development",
             "internal_split": split, "trained_case_ids": train_ids, "internal_selection_valid": selection_valid,
             "outer8_used_for_selection": False, "init_audits": init_audits, "pretrained_prior": pretrained_prior,
             "attribute_weights": {name: value.tolist() for name, value in attribute_weights.items()},
             "classifier_initialization_selection_metric": cfg.get("selection_metric") if args.classifier_init and not args.resume else None,
             "classifier_initialization_selection_report": donor.get("selection_report") if args.classifier_init and not args.resume else None,
             "classifier_pretraining_updates": extra_classifier_updates, "segmentation_updates": args.epochs * args.steps,
             "classifier_weight_policy": "fixed low fine-tuning coefficient; no automatic ramp to0.5",
             "frozen_classifier_policy": "no gradients, eval BN/dropout, no classifier/guidance auxiliary loss",
             "partial_classifier_policy": "deep encoder6/7 plus attention/fusion/heads; early encoder0..5 BN eval; real gradients",
             "backbone_warmup_policy": "historical encoder/decoder segmentation supervision; enhanced inference unchanged",
             "publication_result": False, "development_run": True, "checkpoint_selection": "fixed final",
             "validation": "internal-four development; original-eight diagnostic only", "balanced_sampling": args.sampling == "geometry",
             "sampling_fov_definition": "uniform centres use image-only HU>-170 body bounds; whole-CT inference is unchanged",
             "segmentation_supervision_weights": [1., .5, .25] if not args.no_ds else [1.],
             "segmentation_supervision_normalization": "weighted sum divided by sum of supervision weights",
             "torch": str(torch.__version__)}
    config["gaussian_schedule"] = {"base": args.gaussian_weight, "late": args.gaussian_late_weight,
            "start_epoch": args.gaussian_ramp_start, "end_epoch": args.gaussian_ramp_end,
            "definition": "fixed if late=None; linear by 1-based epoch number: base through start_epoch, late at end_epoch (inclusive)"}
    config["periodic_checkpoint_role"] = "internal-four trajectory diagnostics; never original-eight periodic selection"
    start = 0
    if args.resume:
        old = json.loads((output / "config.json").read_text())
        for key, value in vars(args).items():
            if key not in ("resume", "plan") and old.get(key) != value: raise ValueError("Resume configuration changed: " + key)
        if old.get("source_sha256") != code_hash or old.get("manifest_sha256") != manifest_hash: raise ValueError("Source/manifest changed since checkpoint")
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if any(saved.get("config", {}).get(key) != value for key, value in old.items() if key not in ("resume", "plan")):
            raise ValueError("latest.pt configuration differs from recorded config.json")
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start = saved["epoch"] + 1; torch.set_rng_state(saved["rng_cpu"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(saved["rng_cuda"])
        config = old | {"resume": True}; pretrained_prior = config["pretrained_prior"]
        attribute_weights = {name: torch.tensor(value, device=device) for name, value in config["attribute_weights"].items()}
    if start >= args.epochs:
        marker = json.loads((output / "TRAINING_COMPLETE.json").read_text())
        if (marker.get("final_sha256") != sha256_file(output / "final.pt") or marker.get("source_sha256") != code_hash
                or marker.get("manifest_sha256") != manifest_hash or marker.get("smoke") != args.smoke):
            raise ValueError("Completed run identity or final checkpoint changed")
        print("TRAINING_COMPLETE_VERIFIED", flush=True); return
    dataset = CohortDataset(manifest, train_ids, fold=args.fold, role="train", sampling=args.sampling,
                            sampling_fov=args.sampling_fov, steps=args.steps * args.batch_size,
                            patch_size=args.patch_size, seed=seed, augment=True,
                            spatial_augmentation=args.spatial_augmentation)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    atomic_json(output / "config.json", config); atomic_json(output / "initialization_audit.json", {"init_audits": config["init_audits"]})
    if not (output / "source_snapshot").exists(): shutil.copytree(Path(__file__).resolve().parent, output / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    print(json.dumps(config), flush=True)
    for epoch in range(start, args.epochs):
        if source_hash() != code_hash or sha256_file(Path(manifest["_manifest_path"])) != manifest_hash: raise RuntimeError("Training dependencies/manifest changed")
        warm_backbone = epoch < args.backbone_warmup_epochs; enhanced_epoch = max(0, epoch - args.backbone_warmup_epochs)
        stage = ("frozen" if enhanced_epoch < args.freeze_epochs else "partial" if enhanced_epoch < args.partial_until else "joint") if uses_prior else "segmentation_only"
        if uses_prior: model.set_classifier_stage(stage)
        model.train(); dataset.set_epoch(epoch); values = []; epoch_details = []; probes = []; began = time.monotonic()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        ramp = min(1., max(0., (epoch + 1 - args.gaussian_ramp_start) / max(1, args.gaussian_ramp_end - args.gaussian_ramp_start)))
        gaussian_lambda = args.gaussian_weight if args.gaussian_late_weight is None else args.gaussian_weight + ramp * (args.gaussian_late_weight - args.gaussian_weight)
        for step, batch in enumerate(loader):
            iteration = epoch * args.steps + step; warm = min(args.warmup_epochs * args.steps, args.epochs * args.steps)
            factor = iteration / max(1, warm) if iteration < warm else max(0., 1 - (iteration - warm) / max(1, args.epochs * args.steps - warm)) ** .9
            for group in optimizer.param_groups: group["lr"] = args.lr * factor * (args.classifier_lr_scale if group["name"] == "classifier" else 1.)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
                if warm_backbone:
                    if not hasattr(model, "backbone_supervised"): raise ValueError("Backbone warm-up belongs only to GTA")
                    heads = model.backbone_supervised(batch["image"]); function = dice_ce_loss if args.seg_loss == "dice_ce" else segmentation_loss
                    loss = sum(function(head, batch["label"]) for head in heads) / 2; details = {"encoder_decoder_warmup": float(loss.detach())}; terms = {}
                else:
                    result = model(batch["image"])
                    loss, details, terms = objective(result, batch, args.method, epoch=enhanced_epoch, segmentation=args.seg_loss,
                         classifier_active=stage != "frozen", classifier_weight=args.classifier_weight, attribute_weights=attribute_weights,
                         gaussian_weight=gaussian_lambda, guidance_weight=args.guidance_weight,
                         anatomy_reweighting=not args.disable_anatomy_reweighting, return_terms=True)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
            if not warm_backbone and "gaussian" in terms and args.gradient_probe_every and iteration % args.gradient_probe_every == 0:
                # Probe-only scaling avoids tiny FP16 output derivatives becoming
                # zeros; it never changes the training loss or optimizer update.
                probe_scale = 1024.
                probe_gradients = [torch.autograd.grad(terms[key] * probe_scale, result["logits"], retain_graph=True, allow_unused=True)[0]
                                   for key in ("segmentation", "gaussian")]
                if all(value is not None for value in probe_gradients):
                    first, second = [value.detach().float().reshape(-1) / probe_scale for value in probe_gradients]
                    if not all(torch.isfinite(value).all() for value in (first, second)):
                        raise FloatingPointError("Nonfinite diagnostic output gradients")
                    a, b = float(first.norm()), float(second.norm())
                    probe = {"iteration": iteration, "probe_space": "main logits; segmentation DS branches excluded from this shared-output derivative",
                             "segmentation_norm": a, "gaussian_weighted_norm": b, "gaussian_lambda": gaussian_lambda,
                             "probe_derivative_scale_removed": probe_scale,
                             "gaussian_to_segmentation_ratio": b / a if a else None,
                             "cosine": float(torch.dot(first, second) / (first.norm() * second.norm())) if a and b else None}
                    probes.append(probe)
                    with (output / "gradient_probes.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(probe) + "\n")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norms = {"segmentation_gradient_norm": gradient_norm(segmentation_parameters), "classifier_gradient_norm": gradient_norm(classifier_parameters)}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
            if not torch.isfinite(norm): raise FloatingPointError("Nonfinite gradients")
            if uses_prior and stage == "frozen" and norms["classifier_gradient_norm"] != 0: raise RuntimeError("Frozen classifier received gradients")
            scaler.step(optimizer); scaler.update(); values.append(float(loss.detach()))
            epoch_details.append(details | norms)
            if step % 25 == 0 or args.smoke:
                print(json.dumps({"epoch": epoch, "step": step, "stage": "backbone_warmup" if warm_backbone else stage,
                                  "loss": values[-1], "segmentation_lr": optimizer.param_groups[0]["lr"], "components": details, **norms}), flush=True)
        row = {"epoch": epoch, "loss": float(np.mean(values)), "seconds": time.monotonic() - began,
               "stage": "backbone_warmup" if warm_backbone else stage, "iterations": args.steps,
               "gaussian_lambda": gaussian_lambda if not warm_backbone and getattr(model, "use_gaussian", False) else 0.,
               "gaussian_lambda_configured": gaussian_lambda,
               "classifier_lambda": args.classifier_weight if not warm_backbone and uses_prior and stage != "frozen" else 0.,
               "lr_by_group": {group["name"]: group["lr"] for group in optimizer.param_groups},
               "component_means": {key: float(np.mean([item[key] for item in epoch_details if item.get(key) is not None]))
                                   for key in set().union(*(item.keys() for item in epoch_details)) if any(item.get(key) is not None for item in epoch_details)},
               "gradient_probe_count": len(probes),
               "gpu_peak_GiB": float(torch.cuda.max_memory_allocated(device) / 2 ** 30) if device.type == "cuda" else None}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(row) + "\n")
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                      "config": config, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_atomic(checkpoint, output / "latest.pt"); print(json.dumps(row), flush=True)
        if (epoch + 1) % args.save_every == 0:
            periodic = output / "checkpoints" / f"epoch{epoch + 1:03d}.pt"
            periodic.parent.mkdir(exist_ok=True)
            if periodic.exists(): raise RuntimeError("Immutable periodic checkpoint already exists")
            save_atomic({"model": model.state_dict(), "config": config, "epoch": epoch,
                         "checkpoint_role": "periodic internal-four diagnostic; not final selection"}, periodic)
    save_atomic({"model": model.state_dict(), "config": config, "epoch": args.epochs - 1}, output / "final.pt")
    atomic_json(output / "TRAINING_COMPLETE.json", {"epoch": args.epochs - 1, "smoke": args.smoke, "development_run": True,
                "manifest_sha256": manifest_hash, "source_sha256": code_hash, "trained_case_ids": train_ids,
                "internal_selection_valid": config["internal_selection_valid"], "final_sha256": sha256_file(output / "final.pt"),
                "classifier_pretraining_updates": config["classifier_pretraining_updates"], "segmentation_updates": args.epochs * args.steps,
                "checkpoint_selection": "fixed final"})
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
