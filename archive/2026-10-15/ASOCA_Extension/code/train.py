"""Reproducible matched-budget coronary extension; no ground-truth inference inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import PatchDataset, load_manifest
from model import build_model
from losses import objective


def make_model(method, ds=True):
    if method == "swinunetr":
        from swin_model import SwinBaseline
        return SwinBaseline(ds)
    return build_model(method, deep_supervision=ds)


def save_atomic(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="gta")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps", type=int, default=125, help="optimizer iterations per epoch")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-ds", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    code_root = Path(__file__).resolve().parent
    source_hash = hashlib.sha256(b"".join(
        p.relative_to(code_root).as_posix().encode() + p.read_bytes()
        for p in sorted(code_root.rglob("*.py")))).hexdigest()
    if manifest.get("pilot_only") and not args.smoke:
        raise RuntimeError("Pilot cache cannot produce publication experiments")
    if (out / "config.json").exists() and not args.resume:
        raise RuntimeError("Output already exists: use --resume or a fresh directory")
    if args.resume and not (out / "latest.pt").exists():
        raise RuntimeError("Resume requested but checkpoint is missing")
    seed = args.seed + args.fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    balanced = args.method.startswith("gta") and args.method != "gta_no_balance"
    pilot_ids = None
    if args.smoke and manifest.get("pilot_only"):
        available = {c["case_id"] for c in manifest["cases"]}
        pilot_ids = [c for c in manifest["folds"][args.fold]["train"] if c in available]
    ds = PatchDataset(args.manifest, fold=args.fold, split="train", patch_size=args.patch_size,
                      steps=args.steps * args.batch_size, seed=seed, balanced=balanced, augment=True, case_ids=pilot_ids)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=False)
    model = make_model(args.method, not args.no_ds).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Keep the verified scale throughout the fixed 25k-update budget; growing
    # it each 2k steps can create avoidable FP16 overflow in the prior branch.
    scaler = torch.amp.GradScaler("cuda",init_scale=1024,growth_interval=1_000_000)
    start_epoch = 0
    if args.resume and (out / "latest.pt").exists():
        checkpoint = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        expected = {"method":args.method,"fold":args.fold,"patch_size":args.patch_size,"batch_size":args.batch_size,"no_ds":args.no_ds,"epochs":args.epochs,"steps":args.steps,"seed":args.seed,"lr":args.lr,"smoke":args.smoke}
        for key,val in expected.items():
            if checkpoint["config"][key] != val:
                raise ValueError("Resume configuration changed: " + key)
        recorded = json.loads((out/"config.json").read_text(encoding="utf-8"))
        if recorded["manifest_sha256"] != hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest():
            raise ValueError("Dataset manifest changed since checkpoint")
        if recorded.get("source_sha256") != source_hash:
            raise ValueError("Training source changed since checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["rng_cpu"])
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    config = vars(args) | {"parameters":sum(p.numel() for p in model.parameters()),
                          "source_sha256":source_hash,
                          "amp_init_scale":1024,"amp_growth_interval":1_000_000,
                          "manifest_sha256":hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                          "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
                          "publication_result":not args.smoke,"balanced_sampling":balanced,
                          "validation":"fixed final checkpoint, full volumes in original CT grid"}
    (out / "config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    if not (out/"source_snapshot").exists():
        shutil.copytree(code_root,out/"source_snapshot",ignore=shutil.ignore_patterns("__pycache__"))
    print(json.dumps(config),flush=True)
    for epoch in range(start_epoch,args.epochs):
        ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        values = []
        for step,batch in enumerate(loader):
            batch = {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            iteration = epoch * args.steps + step
            lr = args.lr * max(1-iteration/(args.epochs*args.steps),0)**.9
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(batch["image"])
                loss, details = objective(output,batch,args.method,epoch=epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}: {details}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),12)
            if not torch.isfinite(gradient):
                raise FloatingPointError("Nonfinite gradient")
            scaler.step(optimizer)
            scaler.update()
            values.append(float(loss.detach()))
            if step % 25 == 0 or args.smoke:
                print(json.dumps({"epoch":epoch,"step":step,"loss":values[-1],"lr":lr,
                                  "gradient_norm":float(gradient),"aux":details}),flush=True)
        row = {"epoch":epoch,"loss":float(np.mean(values)),"seconds":round(time.time()-t0,2),
               "gpu_peak_GiB":round(torch.cuda.max_memory_allocated()/2**30,3)}
        with (out / "history.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        checkpoint = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                      "epoch":epoch,"config":config,"rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all()}
        save_atomic(checkpoint,out/"latest.pt")
    save_atomic({"model":model.state_dict(),"config":config,"epoch":args.epochs-1},out/"final.pt")
    (out / "TRAINING_COMPLETE.json").write_text(json.dumps({"epoch":args.epochs-1,"smoke":args.smoke}),encoding="utf-8")
    print("TRAINING_COMPLETE",flush=True)


if __name__ == "__main__":
    main()
