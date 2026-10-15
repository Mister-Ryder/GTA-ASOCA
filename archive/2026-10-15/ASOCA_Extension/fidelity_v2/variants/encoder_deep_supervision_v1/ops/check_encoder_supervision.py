"""One CPU synthetic wiring check; no optimizer, real images or performance claims."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from model import build_model
from losses import objective


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    size = 32
    grid = torch.arange(size)
    z, y, x = torch.meshgrid(grid, grid, grid, indexing="ij")
    label = (((x - 16).square() + (y - 16).square()) < 9).long()[None].repeat(2, 1, 1, 1)
    images = label[:, None].float() + .05 * torch.randn(2, 1, size, size, size)
    batch = {"image": images, "label": label}
    baseline = build_model("gta_backbone", encoder_deep_supervision=True).cpu().train()
    result = baseline(images)
    assert result["encoder_aux_logits"].shape == (2, 2, size, size, size)
    assert result["classifier"] == {} and result["gaussian"] is None
    loss, details, terms = objective(result, batch, "gta_backbone", segmentation="dice_ce",
                                     encoder_ds_weight=.25, return_terms=True)
    assert torch.isfinite(loss) and details["encoder_ds_lambda"] == .25
    assert torch.allclose(loss, terms["segmentation"] + terms["encoder_supervision"])
    encoder_parameters = [baseline.base_model.final_conv.weight,
                          next(baseline.base_model.fusion_module_encoder.parameters()),
                          next(baseline.base_model.encoders[0].parameters())]
    gradients = torch.autograd.grad(terms["encoder_supervision"], encoder_parameters, retain_graph=True)
    norms = [float(value.norm()) for value in gradients]
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)
    loss.backward()
    assert baseline.base_model.final_conv.weight.grad is not None
    baseline.eval()
    with torch.inference_mode():
        enabled_eval = baseline(images)
        baseline.encoder_deep_supervision = False
        disabled_eval = baseline(images)
    assert enabled_eval["encoder_aux_logits"] is None
    assert torch.equal(enabled_eval["logits"], disabled_eval["logits"])
    baseline.train()
    assert baseline(images)["encoder_aux_logits"] is None

    full = build_model("gta", encoder_deep_supervision=True).cpu()
    full.set_classifier_stage("frozen")
    full.train()
    full_batch = batch | {"priors": torch.tensor([[3, 3, 1], [3, 3, 1]]),
                          "gaussian": label[:, None].float()}
    full_output = full(images)
    full_loss, _, full_terms = objective(full_output, full_batch, "gta", segmentation="dice_ce",
                                        classifier_active=False, encoder_ds_weight=.25, return_terms=True)
    assert torch.isfinite(full_loss) and "encoder_supervision" in full_terms
    full_loss.backward()
    assert full.base_model.final_conv.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in full.classifier.parameters())

    old_path = ROOT.parent / "gaussian_region_balance_v1/code/model.py"
    spec = importlib.util.spec_from_file_location("sealed_B_model", old_path)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    old = old_module.build_model("gta").cpu().eval()
    old.load_state_dict(full.state_dict(), strict=True)
    full.eval()
    with torch.inference_mode():
        old_output, new_output = old(images), full(images)
    assert new_output["encoder_aux_logits"] is None
    assert torch.equal(old_output["logits"], new_output["logits"])
    try:
        build_model("unet", encoder_deep_supervision=True)
    except ValueError:
        pass
    else:
        raise AssertionError("PlainUNet must reject the unmatched encoder branch")
    report = {"status": "PASS", "device": "cpu", "torch": torch.__version__,
              "batch": 2, "synthetic_size": size, "optimizer_updates": 0,
              "encoder_only_gradient_norms_head_fusion_firstencoder": norms,
              "baseline_needs_no_priors": True, "full_frozen_classifier_has_no_gradient": True,
              "eval_logits_equal_sealed_B": True, "no_new_state_keys": True,
              "seconds": time.monotonic() - started,
              "limits": "Synthetic wiring only; not CUDA/AMP, real-volume or segmentation performance validation."}
    (ROOT / "ops/encoder_cpu_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
