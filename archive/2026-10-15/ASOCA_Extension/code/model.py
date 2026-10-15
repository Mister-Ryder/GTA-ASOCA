"""Minimal GTAirway v2 adaptation for a coronary-vessel transfer experiment.

The vendored classifier, three guidance modules, backbone and multiscale fusion
are byte-for-byte historical source. This wrapper copies the active enhanced
forward path to expose raw segmentation logits and optional decoder supervision.
It consumes images only, including during evaluation. Anatomical training labels
are supplied to losses by the trainer, never to this forward function.

Prior semantics in the ASOCA experiment:
  generation: 0 empty; 1..6 increasing root-based branch-depth strata.
  structure: 0 empty; the original tube / cross-section / junction complexity bins.
  location: 0 empty; 1..6 z-tertile x left/right spatial regions.

Training differs from historical checkpoint-based staging: no historical airway
weights are assumed; the classifier and segmentation model are jointly trained
from scratch. Deep supervision is a training option applied to both GTA and the
plain U-Net baseline. Its heads have no role in the final inference prediction.

Output interface:
  logits: full-resolution, unnormalized [B,C,D,H,W].
  aux_logits: [half-resolution, quarter-resolution], highest resolution first.
  classifier: the historical output dictionary, or {} for a prior ablation.
  gaussian: sigmoid [B,1,D,H,W], or None if disabled.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .historical.func.AnatomicalClassifier import AirwayAnatomicalClassifier
    from .historical.func.AnatomicalFusionModules import (
        GenerationGuidanceModule, LocationGuidanceModule, StructureGuidanceModule,
    )
    from .historical.func.model_arch_model10 import SegAirwayModel
except ImportError:
    from historical.func.AnatomicalClassifier import AirwayAnatomicalClassifier
    from historical.func.AnatomicalFusionModules import (
        GenerationGuidanceModule, LocationGuidanceModule, StructureGuidanceModule,
    )
    from historical.func.model_arch_model10 import SegAirwayModel


class GTAirwayASOCA(nn.Module):
    """Historical v2 active architecture with raw logits and optional DS heads."""

    def __init__(self, num_classes=2, deep_supervision=True, use_prior=True,
                 use_gaussian=True):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Use background plus at least one foreground class.")
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.use_prior = use_prior
        self.use_gaussian = use_gaussian
        self.base_model = SegAirwayModel(in_channels=1, out_channels=num_classes)

        if use_prior:
            self.classifier = AirwayAnatomicalClassifier(
                in_channels=1, generation_classes=7, structure_classes=7,
                location_classes=7,
            )
            self.location_guidance = LocationGuidanceModule(64, 64)
            self.structure_guidance = StructureGuidanceModule(64, 128)
            self.generation_guidance = GenerationGuidanceModule(128, 256)
            # Keep the historical training feedback / uncertainty pathways.
            self.seg_to_cls_feedback = nn.ModuleDict({
                "generation": self._feedback(128, 256),
                "structure": self._feedback(64, 128),
                "location": self._feedback(64, 64),
            })
            self.uncertainty_estimator = nn.ModuleDict({
                name: nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(ch, 7, 1))
                for name, ch in (("generation", 128), ("structure", 64),
                                 ("location", 64))
            })

        # Historical decoder fusion emits 3 x 2 channels regardless of C.
        self.anatomical_output_fusion = nn.Sequential(
            nn.Conv3d(22 if use_prior else 6, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.Conv3d(32, num_classes, 1),
        )
        if use_gaussian:
            # Exact v2 mapping from segmentation logits to Gaussian field.
            self.gau_final = nn.Conv3d(num_classes, 1, kernel_size=3, padding=1)
            self.gau_activation = nn.Sigmoid()
        if deep_supervision:
            # The two existing decoder levels, with no extra feature extractor.
            self.ds_heads = nn.ModuleList([nn.Conv3d(64, num_classes, 1),
                                           nn.Conv3d(128, num_classes, 1)])

    @staticmethod
    def _feedback(in_channels, out_channels):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, 1),
                             nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, joint_training=True) -> dict[str, Any]:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError("Expected a single-channel 3D image [B,1,D,H,W].")
        if any(size % 8 for size in x.shape[2:]):
            raise ValueError("Patch dimensions must be divisible by 8.")
        # Joint training is the explicit from-scratch experiment protocol.
        # Eval keeps the standard torch.no_grad()/inference_mode() semantics.
        cls_output = self.classifier(x) if self.use_prior else {}
        cls_features = cls_output.get("encoder_features")
        if self.use_prior:
            # Match historical argmax / confidence modulation (not new TCFF).
            with torch.no_grad():
                probs = {key: F.softmax(cls_output[key], dim=1)
                         for key in ("generation", "structure", "location")}
                preds = {key: cls_output[key].argmax(dim=1) for key in probs}

        encoder_features, feedback_features = [], {}
        x_enc = x
        for index, encoder in enumerate(self.base_model.encoders):
            x_enc = encoder(x_enc)
            if self.use_prior and index == 1:
                feedback_features["location"] = x_enc
                x_enc = self.location_guidance(x_enc, cls_features[0],
                                               preds["location"], probs["location"])
            encoder_features.append(x_enc)

        encoder_features.reverse()
        x_dec = encoder_features[0]
        decoder_features = []
        for index, decoder in enumerate(self.base_model.decoders):
            x_dec = decoder(encoder_features[index + 1], x_dec)
            if self.use_prior and index == 0:
                feedback_features["generation"] = x_dec
                x_dec = self.generation_guidance(x_dec, cls_features[2],
                                                 preds["generation"], probs["generation"])
            elif self.use_prior and index == 1:
                feedback_features["structure"] = x_dec
                x_dec = self.structure_guidance(x_dec, cls_features[1],
                                                preds["structure"], probs["structure"])
            decoder_features.append(x_dec)

        decoder_fusion = self.base_model.fusion_module_decoder(decoder_features)
        if self.use_prior:
            guidance_map = F.interpolate(cls_output["guidance_map"],
                                         size=decoder_fusion.shape[2:],
                                         mode="trilinear", align_corners=False)
            final_fusion = torch.cat([decoder_fusion, guidance_map], dim=1)
        else:
            final_fusion = decoder_fusion
        logits = self.anatomical_output_fusion(final_fusion)
        gaussian = self.gau_activation(self.gau_final(logits)) if self.use_gaussian else None

        if self.use_prior and self.training:
            enhanced_features, uncertainty = [], {}
            feature_indices = {"generation": 2, "structure": 1, "location": 0}
            for key, feature in feedback_features.items():
                enhanced_feature = self.seg_to_cls_feedback[key](feature)
                enhanced_feature = F.interpolate(
                    enhanced_feature, size=cls_features[feature_indices[key]].shape[2:],
                    mode="trilinear", align_corners=False,
                )
                enhanced_features.append((key, enhanced_feature))
                uncertainty[key] = self.uncertainty_estimator[key](feature)
            cls_output["enhanced_features"] = enhanced_features
            cls_output["uncertainty_estimates"] = uncertainty

        auxiliary = []
        if self.deep_supervision:
            auxiliary = [self.ds_heads[0](decoder_features[1]),
                         self.ds_heads[1](decoder_features[0])]
        return {"logits": logits, "aux_logits": auxiliary,
                "classifier": cls_output, "gaussian": gaussian}


class PlainUNet3D(nn.Module):
    """Four-level 3D U-Net with the same feature widths and real decoder DS."""

    def __init__(self, num_classes=2, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        widths = (32, 64, 128, 256)
        self.encoders = nn.ModuleList([self._block(1, widths[0])] + [
            self._block(a, b) for a, b in zip(widths[:-1], widths[1:])
        ])
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList([nn.ConvTranspose3d(a, b, 2, stride=2)
                                 for a, b in ((256, 128), (128, 64), (64, 32))])
        self.decoders = nn.ModuleList([self._block(2 * ch, ch) for ch in (128, 64, 32)])
        self.output = nn.Conv3d(32, num_classes, 1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([nn.Conv3d(64, num_classes, 1),
                                           nn.Conv3d(128, num_classes, 1)])

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, **kwargs):
        if any(size % 8 for size in x.shape[2:]):
            raise ValueError("Patch dimensions must be divisible by 8.")
        features = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(self.pool(x) if index else x)
            features.append(x)
        decoder_features = []
        for index, (up, decoder) in enumerate(zip(self.up, self.decoders)):
            x = decoder(torch.cat([up(x), features[-index - 2]], dim=1))
            decoder_features.append(x)
        auxiliary = ([self.ds_heads[0](decoder_features[1]),
                      self.ds_heads[1](decoder_features[0])]
                     if self.deep_supervision else [])
        return {"logits": self.output(x), "aux_logits": auxiliary,
                "classifier": {}, "gaussian": None}


def build_model(method, num_classes=2, deep_supervision=True):
    """Build a model; sampling and clDice differences are trainer-side switches."""
    method = method.lower()
    if method in ("unet", "unet_cldice"):
        return PlainUNet3D(num_classes, deep_supervision)
    if method not in ("gta", "gta_no_prior", "gta_no_gaussian", "gta_no_balance"):
        raise ValueError(f"Unknown method {method!r}.")
    return GTAirwayASOCA(num_classes=num_classes, deep_supervision=deep_supervision,
                        use_prior=method != "gta_no_prior",
                        use_gaussian=method != "gta_no_gaussian")


def historical_equivalence_test(device="cuda", size=32, batch_size=2):
    """Compare the default wrapper to the exact historical enhanced inference.

    Only the original three module import statements are replaced by already
    imported classes, so its unchanged forward can run without a top-level
    historical ``func`` package shadowing unrelated application imports.
    No weights, thresholds or anatomical labels are used in this check.
    """
    torch.manual_seed(1419)
    source = Path(__file__).parent / "historical/func/EnhancedAirwaySegmentation2gau.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = {"func.AnatomicalFusionModules", "func.model_arch_model10",
               "func.AnatomicalClassifier"}
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module in imports)]
    namespace = {"__name__": "historical_reference",
                 "SegAirwayModel": SegAirwayModel,
                 "AirwayAnatomicalClassifier": AirwayAnatomicalClassifier,
                 "GenerationGuidanceModule": GenerationGuidanceModule,
                 "StructureGuidanceModule": StructureGuidanceModule,
                 "LocationGuidanceModule": LocationGuidanceModule}
    exec(compile(tree, str(source), "exec"), namespace)
    reference = namespace["EnhancedAirwaySegmentation"]().to(device).eval()
    wrapper = build_model("gta", deep_supervision=False).to(device).eval()
    incompatible = wrapper.load_state_dict(reference.state_dict(), strict=False)
    assert incompatible.missing_keys == [], incompatible
    assert all(key.startswith("adaptive_threshold.") for key in incompatible.unexpected_keys), incompatible
    captured = {}
    def capture_logits(module, arguments, result):
        captured["logits"] = result
    hook = reference.anatomical_output_fusion.register_forward_hook(capture_logits)
    images = torch.randn(batch_size, 1, size, size, size, device=device)
    # Compare architecture in deterministic full-precision convolution mode;
    # cuDNN may otherwise choose numerically different reduction kernels.
    with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False), torch.inference_mode():
        old_segmentation, old_classifier, old_gaussian = reference(images)
        new_output = wrapper(images)
        torch.testing.assert_close(new_output["logits"], captured["logits"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(new_output["logits"].softmax(1), old_segmentation, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(new_output["gaussian"], old_gaussian, atol=1e-6, rtol=1e-5)
        for key in ("generation", "structure", "location"):
            torch.testing.assert_close(new_output["classifier"][key], old_classifier[key], atol=1e-6, rtol=1e-5)
        maximum_difference = float((new_output["logits"] - captured["logits"]).abs().max())
    hook.remove()
    return {"historical_equivalence": True, "max_abs_logit_difference": maximum_difference,
            "unused_original_keys": incompatible.unexpected_keys}


def synthetic_backward_test(device="cuda", size=48, batch_size=2,
                            methods=("gta", "gta_no_prior", "gta_no_gaussian", "unet")):
    """Check output geometry and substantive gradients using branching tubes."""
    if size % 8 or size < 32:
        raise ValueError("Synthetic size must be at least 32 and divisible by 8.")
    torch.manual_seed(1419)
    device = torch.device(device)
    axis = torch.linspace(-1, 1, size, device=device)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    radial = torch.minimum(x.square() + y.square(),
                           (x - .4 * z.clamp_min(0)).square() + (y - .4 * z.clamp_min(0)).square())
    target = (radial < .035).long().unsqueeze(0).repeat(batch_size, 1, 1, 1)
    field = torch.exp(-radial / .04).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    images = (target.float().unsqueeze(1) + .1 * torch.randn_like(field)).detach()
    report = []
    for method in methods:
        model = build_model(method, deep_supervision=True).to(device).train()
        output = model(images)
        assert output["logits"].shape == (batch_size, 2, size, size, size)
        assert [tuple(item.shape[2:]) for item in output["aux_logits"]] == [
            (size // 2,) * 3, (size // 4,) * 3,
        ]
        loss = F.cross_entropy(output["logits"], target)
        for weight, auxiliary in zip((.5, .25), output["aux_logits"]):
            resized = F.interpolate(target[:, None].float(), size=auxiliary.shape[2:],
                                    mode="nearest")[:, 0].long()
            loss = loss + weight * F.cross_entropy(auxiliary, resized)
        if output["gaussian"] is not None:
            assert output["gaussian"].shape == field.shape
            loss = loss + F.mse_loss(output["gaussian"], field)
        classifier = output["classifier"]
        if classifier:
            for index, key in enumerate(("generation", "structure", "location")):
                labels = torch.arange(batch_size, device=device).add(index + 1).remainder(7)
                loss = loss + .1 * F.cross_entropy(classifier[key], labels)
            # Exercise the historical feedback and certainty outputs as well.
            for _, feature in classifier["enhanced_features"]:
                loss = loss + .001 * feature.square().mean()
            for logits in classifier["uncertainty_estimates"].values():
                loss = loss + .001 * logits.square().mean()
        if not torch.isfinite(loss):
            raise AssertionError(f"Nonfinite synthetic loss in {method}.")
        loss.backward()
        required = ["ds_heads.0.weight", "ds_heads.1.weight"]
        if method.startswith("gta"):
            required += ["base_model.encoders.0.conv1.conv.weight",
                         "anatomical_output_fusion.3.weight"]
            if classifier:
                required += ["classifier.encoder.0.conv.weight",
                             "classifier.generation_head.3.weight",
                             "classifier.structure_head.3.weight",
                             "classifier.location_head.3.weight",
                             "generation_guidance.transform.0.weight"]
            if output["gaussian"] is not None:
                required += ["gau_final.weight"]
        else:
            required += ["encoders.0.0.weight", "output.weight"]
        named = dict(model.named_parameters())
        for key in required:
            gradient = named[key].grad
            if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
                raise AssertionError(f"Missing / invalid / zero gradient: {method}:{key}")
        report.append({"method": method, "loss": float(loss.detach()),
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "gradient_checks": len(required)})
        del model, output, loss, named
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASOCA model synthetic backward validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--equivalence-only", action="store_true")
    args = parser.parse_args()
    if args.equivalence_only:
        print(historical_equivalence_test(args.device, args.size, args.batch_size))
    else:
        for result in synthetic_backward_test(args.device, args.size, args.batch_size):
            print(result)
