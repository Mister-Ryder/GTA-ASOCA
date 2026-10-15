"""Training-internal 28/4 cohorts and image-derived patch sampling bounds.

The original immutable cache and its five-fold patient partition are unchanged.
All auxiliary maps are targets only. Selection/diagnostic images are sampled
uniformly using image-derived bounds, never held-out masks or centreline points.
"""
from collections import OrderedDict

import numpy as np

from data import PatchDataset, load_manifest, extract_patch, patch_priors

NAMES = ("generation", "structure", "location")


def internal_split(manifest, fold=0, seed=20261014):
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if manifest.get("pilot_only") or not manifest.get("complete"):
        raise ValueError("Fidelity experiments require the complete immutable 40-case cache")
    original_train = list(manifest["folds"][fold]["train"])
    rng = np.random.default_rng(seed + fold)
    selection = []
    for group in ("Normal", "Diseased"):
        ids = sorted(case for case in original_train if cases[case]["group"] == group)
        if len(ids) != 16:
            raise ValueError("Original training fold must have sixteen cases in each patient stratum")
        rng.shuffle(ids)
        selection += ids[:2]
    return {"seed": seed + fold, "train": sorted(set(original_train) - set(selection)),
            "val": sorted(selection), "original_train32": sorted(original_train),
            "outer_development": sorted(manifest["folds"][fold]["val"]),
            "selection_role": "four training-internal patients for development selection",
            "outer_role": "eight previously exposed development patients; diagnostic only, never independent test"}


class CohortDataset(PatchDataset):
    def __init__(self, manifest_path, case_ids, fold=0, role="train", sampling="standard",
                 sampling_fov="image-body", spatial_augmentation="none", **kwargs):
        if role not in ("train", "selection", "outer_diagnostic") or sampling not in ("standard", "geometry", "uniform"):
            raise ValueError("Invalid cohort role or sampling policy")
        if sampling_fov not in ("full", "image-body"):
            raise ValueError("sampling_fov must be full or image-body")
        if spatial_augmentation not in ("none", "xy90"):
            raise ValueError("spatial_augmentation must be none or xy90")
        manifest = manifest_path if isinstance(manifest_path, dict) else load_manifest(manifest_path)
        # Super creates normal memory-map readers. The restricted cohort below
        # is checked against the original partition rather than masquerading as
        # the old pilot-only case_ids override.
        super().__init__(manifest, fold=fold, split="train", balanced=sampling == "geometry", **kwargs)
        allowed = manifest["folds"][fold]["val" if role == "outer_diagnostic" else "train"]
        if not case_ids or len(case_ids) != len(set(case_ids)) or not set(case_ids).issubset(allowed):
            raise ValueError("Requested cohort leaks across the original patient partition")
        self.case_ids = list(case_ids)
        all_cases = {case["case_id"]: case for case in manifest["cases"]}
        self.cases = {case_id: all_cases[case_id] for case_id in self.case_ids}
        self.role, self.sampling, self.sampling_fov = role, sampling, sampling_fov
        self.spatial_augmentation = spatial_augmentation
        if spatial_augmentation == "xy90" and self.patch_size[-2] != self.patch_size[-1]:
            raise ValueError("xy90 requires square YX patches")
        self._bounds = OrderedDict()
        if role != "train":
            self.augment = False
            self.sampling = "uniform"

    def _image_bounds(self, case_id, image):
        if self.sampling_fov == "full":
            return np.zeros(3, dtype=int), np.array(image.shape)
        if case_id not in self._bounds:
            # Image-only HU > -170 on a stride-four grid. This restricts uniform
            # negative sampling around scanned anatomy; no CT or label is cropped
            # for whole-volume evaluation and positive centres are never clipped.
            support = np.argwhere(np.asarray(image[::4, ::4, ::4]) > -.95)
            if len(support):
                low = np.maximum(0, support.min(0) * 4 - 8)
                high = np.minimum(image.shape, (support.max(0) + 1) * 4 + 8)
            else:
                low, high = np.zeros(3, dtype=int), np.array(image.shape)
            self._bounds[case_id] = (low, high)
        return self._bounds[case_id]

    def __getitem__(self, index):
        original_tensors = self.return_tensors
        original_case_ids = self.case_ids
        if self.role != "train":
            self.case_ids = [original_case_ids[index % len(original_case_ids)]]
        self.return_tensors = False
        try:
            item = super().__getitem__(index)
        finally:
            self.return_tensors = original_tensors
            self.case_ids = original_case_ids
        # Keep the original foreground/geometry centre choices. Uniform samples
        # use only the image FOV. Held-out cohorts always enter this branch.
        if item["sampling_stratum"] == "uniform" or self.sampling == "uniform":
            arrays = self._get_case(item["case_id"])
            low, high = self._image_bounds(item["case_id"], arrays["image"])
            shape, size = np.array(arrays["image"].shape), np.array(self.patch_size)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index), 841]))
            start = np.array([rng.integers(int(lo), max(int(lo) + 1, int(hi - width + 1)))
                              for lo, hi, width in zip(low, high, size)])
            start = np.where(shape >= size, np.clip(start, 0, np.maximum(0, shape - size)), (shape - size) // 2)
            image = extract_patch(arrays["image"], start, self.patch_size, fill=-1, dtype=np.float32)
            label = extract_patch(arrays["label"], start, self.patch_size, dtype=np.int64)
            branches = extract_patch(arrays["branch_id"], start, self.patch_size)
            depth = extract_patch(arrays["depth"], start, self.patch_size)
            if self.augment:
                if rng.random() < .25:
                    image *= rng.uniform(.85, 1.15)
                if rng.random() < .25:
                    image += rng.uniform(-.1, .1)
                if rng.random() < .15:
                    image += rng.normal(0, .025, image.shape).astype(np.float32)
                np.clip(image, -1.25, 1.25, out=image)
            item.update(image=image[None], label=label,
                        gaussian=extract_patch(arrays["gaussian"], start, self.patch_size, dtype=np.float32)[None],
                        priors=patch_priors(label, branches, depth, start, tuple(shape)),
                        start_zyx=start.astype(np.int64), sampling_stratum="uniform_image_fov")
        if self.role == "train" and self.augment and self.spatial_augmentation == "xy90":
            # Separate RNG preserves the existing crop, sampling and intensity draws.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index), 29015]))
            k = int(rng.integers(0, 4))
            for key in ("image", "label", "gaussian"):
                item[key] = np.ascontiguousarray(np.rot90(item[key], k=k, axes=(-2, -1)))
        if original_tensors:
            import torch
            for key in ("image", "label", "gaussian", "priors", "start_zyx"):
                item[key] = torch.from_numpy(np.ascontiguousarray(item[key]))
        return item


def training_attribute_statistics(dataset, samples=2048):
    if dataset.role != "train":
        raise ValueError("Focal weights and majority classes must use training patients only")
    counts = np.zeros((3, 7), dtype=np.int64)
    tensors, augment, epoch = dataset.return_tensors, dataset.augment, dataset.epoch
    dataset.return_tensors, dataset.augment, dataset.epoch = False, False, 0
    try:
        for index in range(min(samples, len(dataset))):
            labels = dataset[index]["priors"]
            for head, value in enumerate(labels):
                counts[head, int(value)] += 1
    finally:
        dataset.return_tensors, dataset.augment, dataset.epoch = tensors, augment, epoch
    result = {}
    for head, name in enumerate(NAMES):
        present = counts[head] > 0
        # A finite bank can miss a class that appears later. Missing support is
        # reported, but it must not permanently disable that class's gradients.
        weights = np.full(7, 5., dtype=np.float32)
        weights[present] = 1 / np.sqrt(counts[head, present])
        weights[present] /= weights[present].mean()
        weights[present] = np.clip(weights[present], .2, 5.)
        result[name] = {"counts": counts[head].tolist(), "focal_weights": weights.tolist(),
                        "absent_training_classes": np.flatnonzero(~present).tolist(),
                        "majority_class": int(counts[head].argmax())}
    return {"patients": sorted(dataset.case_ids), "samples": int(counts[0].sum()),
            "source": "fixed deterministic training-only patch bank; inverse-square-root frequency weights",
            "heads": result}


def classification_report(truth, prediction, majority_classes, training_statistics):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    result = {}
    for head, name in enumerate(NAMES):
        confusion = np.zeros((7, 7), dtype=np.int64)
        np.add.at(confusion, (truth[:, head], prediction[:, head]), 1)
        support = confusion.sum(1)
        present = support > 0
        recall = np.divide(np.diag(confusion), support, out=np.zeros(7, dtype=float), where=present)
        majority = int(majority_classes[head])
        majority_recall = np.zeros(7, dtype=float)
        majority_recall[majority] = 1.
        positive = present.copy()
        positive[0] = False
        nonempty = truth[:, head] > 0
        empty_recall = np.zeros(7, dtype=float)
        empty_recall[0] = 1.
        result[name] = {"confusion_matrix": confusion.tolist(), "true_support": support.tolist(),
                        "per_class_recall": [float(recall[i]) if present[i] else None for i in range(7)],
                        "balanced_accuracy": float(recall[present].mean()) if present.any() else None,
                        "accuracy": float(np.mean(truth[:, head] == prediction[:, head])),
                        "absent_evaluation_classes": np.flatnonzero(~present).tolist(),
                        "absent_training_classes": training_statistics["heads"][name]["absent_training_classes"],
                        "majority_baseline_class": majority,
                        "majority_baseline_accuracy": float(np.mean(truth[:, head] == majority)),
                        "majority_baseline_balanced_accuracy": float(majority_recall[present].mean()),
                        "nonempty_support": int(nonempty.sum()),
                        "nonempty_accuracy": float(np.mean(truth[nonempty, head] == prediction[nonempty, head])) if nonempty.any() else None,
                        "nonempty_balanced_accuracy": float(recall[positive].mean()) if positive.any() else None,
                        "absent_positive_evaluation_classes": [i for i in range(1, 7) if not present[i]],
                        "majority_baseline_nonempty_balanced_accuracy": float(majority_recall[positive].mean()) if positive.any() else None,
                        "always_empty_accuracy": float(np.mean(truth[:, head] == 0)),
                        "always_empty_balanced_accuracy": float(empty_recall[present].mean()),
                        "always_empty_nonempty_balanced_accuracy": 0. if positive.any() else None}
    positive_valid = all(result[name]["nonempty_support"] > 0 for name in NAMES)
    return {"n_patches": len(truth), "balanced_accuracy_definition": "macro recall over classes present in evaluation ground truth",
            "mean_head_balanced_accuracy": float(np.mean([result[name]["balanced_accuracy"] for name in NAMES])),
            "selection_positive_coverage_valid": positive_valid,
            "mean_head_nonempty_balanced_accuracy": float(np.mean([result[name]["nonempty_balanced_accuracy"] for name in NAMES])) if positive_valid else None,
            "heads": result}
