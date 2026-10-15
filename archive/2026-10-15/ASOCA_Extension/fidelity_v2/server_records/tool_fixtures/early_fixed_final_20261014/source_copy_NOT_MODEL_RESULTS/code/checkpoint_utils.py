"""Explicit component transfer with safe loading, key/shape audits and provenance."""
import argparse
import hashlib
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

SEMANTIC_HEADS = ("generation_head.", "structure_head.", "location_head.")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path):
    # Historical classifier checkpoints include argparse.Namespace metadata.
    # This allowlist still rejects arbitrary custom Python globals.
    with torch.serialization.safe_globals([argparse.Namespace, TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    if "model_state_dict" in checkpoint:
        state, format_key = checkpoint["model_state_dict"], "model_state_dict"
    elif "model" in checkpoint:
        state, format_key = checkpoint["model"], "model"
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state, format_key = checkpoint, "bare_state_dict"
    else:
        raise ValueError("No model_state_dict/model tensor dictionary in checkpoint")
    clean = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("State dictionary contains non-tensor parameters")
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in clean:
            raise ValueError("Prefix removal produced duplicate parameter keys")
        clean[key] = value
    return checkpoint, clean, format_key


def load_audited(module, path, component, domain, semantic_head_policy="reinit",
                 allow_missing_prefixes=(), allow_unexpected_prefixes=()):
    if component not in ("backbone", "classifier", "full") or domain not in ("airway", "asoca"):
        raise ValueError("Declare component backbone/classifier/full and source domain airway/asoca")
    if semantic_head_policy not in ("reinit", "load"):
        raise ValueError("Explicit semantic_head_policy must be reinit or load")
    checkpoint, state, format_key = read_checkpoint(path)
    source_count = len(state)
    prefix = "base_model." if component == "backbone" else "classifier." if component == "classifier" else None
    if prefix and any(key.startswith(prefix) for key in state):
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    elif prefix and any(key.startswith(("base_model.", "classifier.")) for key in state):
        raise ValueError(f"Full checkpoint contains no requested {component} component")
    target = module.state_dict()
    semantic_prefixes = tuple("classifier." + key for key in SEMANTIC_HEADS) if component == "full" else SEMANTIC_HEADS if component == "classifier" else ()
    excluded = [key for key in state if semantic_head_policy == "reinit" and key.startswith(semantic_prefixes)]
    candidates = {key: value for key, value in state.items() if key not in excluded}
    mismatches = {key: {"source": list(value.shape), "target": list(target[key].shape)}
                  for key, value in candidates.items() if key in target and value.shape != target[key].shape}
    unexpected = sorted(set(candidates) - set(target))
    missing = sorted(set(target) - set(candidates))
    expected_semantic_missing = [key for key in missing if semantic_head_policy == "reinit" and key.startswith(semantic_prefixes)]
    disallowed_missing = [key for key in missing if key not in expected_semantic_missing and not key.startswith(tuple(allow_missing_prefixes))]
    disallowed_unexpected = [key for key in unexpected if not key.startswith(tuple(allow_unexpected_prefixes))]
    audit = {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "component": component,
             "source_domain": domain, "format_key": format_key, "source_tensor_count": source_count,
             "component_tensor_count": len(state), "semantic_head_policy": semantic_head_policy,
             "excluded_semantic_keys": excluded, "missing_keys": missing, "unexpected_keys": unexpected,
             "shape_mismatches": mismatches, "source_epoch": checkpoint.get("epoch"),
             "loaded_key_count": sum(key in target and value.shape == target[key].shape for key, value in candidates.items()),
             "allowed_missing_prefixes": list(allow_missing_prefixes), "allowed_unexpected_prefixes": list(allow_unexpected_prefixes)}
    if mismatches or disallowed_missing or disallowed_unexpected:
        raise ValueError("Checkpoint transfer rejected: " + str(audit))
    accepted = {key: value for key, value in candidates.items() if key in target}
    if not accepted:
        raise ValueError("No matching tensors were loaded")
    module.load_state_dict({**target, **accepted}, strict=True)
    audit["loaded_keys"] = sorted(accepted)
    audit["randomly_retained_keys"] = missing
    return audit
