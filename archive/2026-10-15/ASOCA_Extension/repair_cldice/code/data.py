"""Memory-mapped full-volume cache and deterministic ASOCA training patches.

No ground-truth-derived image ROI or auxiliary map is used at inference. GT-based
sampling and the Gaussian/topology arrays serve training supervision only.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "ASOCA":
        raise ValueError(f"Unsupported manifest: {path}")
    manifest["_manifest_path"] = str(path.resolve())
    manifest["_cache_root"] = str(path.resolve().parent)
    return manifest


def case_directory(manifest: dict, case: dict | str) -> Path:
    if isinstance(case, str):
        case = next(c for c in manifest["cases"] if c["case_id"] == case)
    return Path(manifest["_cache_root"]) / case["cache_subdir"]


def load_case(manifest: dict, case: dict | str, auxiliaries=False) -> dict:
    """Inference callers should keep auxiliaries=False; only CT is model input."""
    folder = case_directory(manifest, case)
    names = ["image", "label"]
    if auxiliaries:
        names += ["gaussian", "skeleton", "radius", "branch_id", "depth"]
    return {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in names}


def generation_bin(depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return 2
    if depth <= 3:
        return 3
    if depth <= 5:
        return 4
    if depth <= 8:
        return 5
    return 6


def patch_priors(label: np.ndarray, branches: np.ndarray, depths: np.ndarray,
                 start_zyx: np.ndarray, full_shape: tuple[int, ...]) -> np.ndarray:
    """Match GTAirway's three seven-way auxiliary heads using geometric labels."""
    foreground = label > 0
    if not foreground.any():
        return np.zeros(3, dtype=np.int64)
    ids, counts = np.unique(branches[foreground], return_counts=True)
    nonzero = ids > 0
    ids, counts = ids[nonzero], counts[nonzero]
    if len(ids):
        dominant = ids[int(counts.argmax())]
        dominant_depths = depths[foreground & (branches == dominant)]
        depth = int(np.bincount(dominant_depths.astype(np.int32)).argmax())
        generation = generation_bin(depth)
    else:
        generation = 1
    branch_count = len(ids)
    if branch_count <= 1:
        coords = np.nonzero(foreground)
        lengths = sorted(int(c.max() - c.min() + 1) for c in coords)
        structure = 1 if lengths[2] / max(1, lengths[1]) > 2 else 2
    else:
        # Same complexity thresholds as AirwayMetadataGenerator.py version 2.
        log_volume = np.log(int(foreground.sum()) + 1) / np.log(10000)
        normalized_branches = np.log(branch_count + 1) / np.log(10)
        complexity = normalized_branches / max(0.1, log_volume)
        structure = 6 if complexity > 1.5 else 5 if complexity > 1 else 4 if complexity > 0.8 else 3
    center = np.asarray(start_zyx) + np.asarray(label.shape) / 2
    z_bin = min(2, max(0, int(center[0] / full_shape[0] * 3)))
    x_half = int(center[2] >= full_shape[2] / 2)
    location = 1 + z_bin * 2 + x_half
    return np.array([generation, structure, location], dtype=np.int64)


def extract_patch(array: np.ndarray, start: np.ndarray, size: tuple[int, ...],
                  fill=0, dtype=None) -> np.ndarray:
    """Explicit padding supports small scans without negative numpy slices."""
    result = np.full(size, fill, dtype=dtype or array.dtype)
    source, target = [], []
    for axis, (begin, width) in enumerate(zip(start, size)):
        low = max(0, int(begin))
        high = min(array.shape[axis], int(begin) + width)
        source.append(slice(low, high))
        target.append(slice(low - int(begin), high - int(begin)))
    if all(s.stop > s.start for s in source):
        result[tuple(target)] = array[tuple(source)]
    return result


class PatchDataset:
    """Index and epoch determine a patch independently of DataLoader worker order.

    balanced=True: 1/3 uniform, 1/3 thin centreline, 1/3 junction centres
    (a missing geometric stratum falls back to ordinary foreground). Uniform
    patches represent the full image distribution and can include foreground.
    balanced=False: conventional 1/3 foreground + 2/3 uniform sampling.

    Only intensity augmentations are used. Spatial flips would invalidate the
    absolute-location head unless its target definition was also transformed.
    """
    def __init__(self, manifest_path: str | Path | dict, fold=0, split="train",
                 patch_size=64, steps=250, seed=20261013, balanced=True,
                 augment=True, return_tensors=True, max_cached_cases=4,
                 case_ids=None):
        self.manifest = manifest_path if isinstance(manifest_path, dict) else load_manifest(manifest_path)
        if "_cache_root" not in self.manifest:
            raise ValueError("Manifest dictionary must be obtained with load_manifest")
        if split not in ("train", "val") or not 0 <= fold < len(self.manifest["folds"]):
            raise ValueError("Expected split train/val and fold between 0 and 4")
        self.split = split
        self.patch_size = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(patch_size)
        if len(self.patch_size) != 3 or min(self.patch_size) <= 0:
            raise ValueError("patch_size must contain three positive dimensions")
        self.steps = int(steps)
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        self.seed, self.epoch = int(seed), 0
        self.balanced, self.augment = bool(balanced), bool(augment) and split == "train"
        self.return_tensors = bool(return_tensors)
        self.max_cached_cases = max(1, int(max_cached_cases))
        available = {c["case_id"]: c for c in self.manifest["cases"]}
        fold_case_ids = list(self.manifest["folds"][fold][split])
        if case_ids is not None and (self.manifest.get("complete")
                                     or not self.manifest.get("pilot_only", False)):
            raise ValueError("case_ids override is reserved for an incomplete, explicitly marked pilot cache")
        self.case_ids = fold_case_ids if case_ids is None else list(case_ids)
        if not self.case_ids or len(self.case_ids) != len(set(self.case_ids)) \
                or any(case_id not in fold_case_ids for case_id in self.case_ids):
            raise ValueError("Pilot case_ids must be a nonempty, unique subset of the frozen fold split")
        missing = [case_id for case_id in self.case_ids if case_id not in available]
        if missing:
            raise ValueError(f"The cache has not prepared the {split} cases: {missing}")
        self.cases = {case_id: available[case_id] for case_id in self.case_ids}
        self._cache = OrderedDict()

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _get_case(self, case_id):
        if case_id in self._cache:
            result = self._cache.pop(case_id)
            self._cache[case_id] = result
            return result
        result = load_case(self.manifest, self.cases[case_id], auxiliaries=self.split == "train")
        if self.split == "train":
            with np.load(case_directory(self.manifest, case_id) / "sampling.npz") as sampling:
                result["sampling"] = {name: sampling[name] for name in sampling.files}
        self._cache[case_id] = result
        while len(self._cache) > self.max_cached_cases:
            self._cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        if not 0 <= index < self.steps:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        case_id = self.case_ids[int(rng.integers(len(self.case_ids)))]
        arrays = self._get_case(case_id)
        shape = np.array(arrays["image"].shape)
        size = np.array(self.patch_size)
        choice = float(rng.random())
        sampling_name = "uniform"
        if self.split == "train":
            if self.balanced:
                sampling_name = "thin" if 1 / 3 <= choice < 2 / 3 else "junction" if choice >= 2 / 3 else "uniform"
            elif choice < 1 / 3:
                sampling_name = "foreground"
        if sampling_name != "uniform":
            centers = arrays["sampling"][sampling_name]
            if not len(centers):
                centers = arrays["sampling"]["foreground"]
            center = centers[int(rng.integers(len(centers)))]
            # Jitter keeps the selected positive within the central half patch.
            jitter = np.array([rng.integers(-max(1, s // 4), max(1, s // 4) + 1) for s in size])
            start = center - size // 2 + jitter
        else:
            start = np.array([rng.integers(0, max(1, int(dim - width + 1))) for dim, width in zip(shape, size)])
        start = np.where(shape >= size, np.clip(start, 0, np.maximum(0, shape - size)),
                         (shape - size) // 2)
        image = extract_patch(arrays["image"], start, self.patch_size, fill=-1, dtype=np.float32)
        label = extract_patch(arrays["label"], start, self.patch_size, dtype=np.int64)
        if self.split == "train":
            gaussian = extract_patch(arrays["gaussian"], start, self.patch_size, dtype=np.float32)
            branches = extract_patch(arrays["branch_id"], start, self.patch_size)
            depths = extract_patch(arrays["depth"], start, self.patch_size)
            priors = patch_priors(label, branches, depths, start, tuple(shape))
        else:
            # Validation labels remain metric targets. Auxiliary GT descriptors
            # are deliberately unavailable to validation/inference callers.
            gaussian = np.zeros(self.patch_size, dtype=np.float32)
            priors = np.zeros(3, dtype=np.int64)
        if self.augment:
            if rng.random() < 0.25:
                image *= rng.uniform(0.85, 1.15)
            if rng.random() < 0.25:
                image += rng.uniform(-0.10, 0.10)
            if rng.random() < 0.15:
                image += rng.normal(0, 0.025, size=image.shape).astype(np.float32)
            np.clip(image, -1.25, 1.25, out=image)
        result = {"image": image[None], "label": label,
                  "gaussian": gaussian[None], "priors": priors,
                  "case_id": case_id, "start_zyx": start.astype(np.int64),
                  "sampling_stratum": sampling_name}
        if self.return_tensors:
            import torch
            for name in ("image", "label", "gaussian", "priors", "start_zyx"):
                result[name] = torch.from_numpy(np.ascontiguousarray(result[name]))
        return result
