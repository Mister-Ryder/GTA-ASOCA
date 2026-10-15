"""Small synthetic CPU-only check; no model, cache, dataset, or metric access."""
import ast
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

VARIANT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT / "code"))
from data import PatchDataset, patch_priors
from classification_data import CohortDataset

spec = importlib.util.spec_from_file_location("old_cohort", VARIANT.parents[1] / "code/classification_data.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def fixture(name):
    branch = np.zeros((16, 16, 16), dtype=np.uint16)
    if name == "single_tube":
        branch[2:14, 7:9, 7:9] = 1
    elif name == "junction":
        branch[2:14, 7:9, 7:9] = 1
        branch[7:9, 8:15, 7:9] = 2
        branch[7:9, 7:9, 8:15] = 3
    elif name == "mixed_branches":
        branch[2:12, 2:6, 2:6] = 1
        branch[4:13, 9:11, 4:6] = 2
        branch[10:14, 5:8, 10:14] = 3
    label = (branch > 0).astype(np.int64)
    depth = np.zeros_like(branch, dtype=np.uint8)
    for number, value in ((1, 2 if name == "mixed_branches" else 0), (2, 7), (3, 10)):
        depth[branch == number] = value
    image = np.arange(label.size, dtype=np.float32).reshape(label.shape) / label.size - .5
    gaussian = np.where(label, image + .5, 0).astype(np.float32)
    return dict(image=image, label=label, branch_id=branch, depth=depth, gaussian=gaussian)


def main():
    start, full = np.array([20, 30, 40]), (96, 112, 128)
    manifest = {"_cache_root": "SYNTHETIC_NEVER_READ", "folds": [{"train": ["a"], "val": ["b"]}] * 5,
                "cases": [{"case_id": "a"}, {"case_id": "b"}]}
    indices = {}
    for index in range(100):
        k = int(np.random.default_rng(np.random.SeedSequence([17, 0, index, 29015])).integers(0, 4))
        indices.setdefault(k, index)
    assert set(indices) == set(range(4))
    results, wrapper_count = [], 0
    for name in ("empty", "single_tube", "junction", "mixed_branches"):
        arrays = fixture(name)
        priors = patch_priors(arrays["label"], arrays["branch_id"], arrays["depth"], start, full)
        for k in range(4):
            rotated = {key: np.rot90(value, k, axes=(-2, -1)) for key, value in arrays.items()}
            np.testing.assert_array_equal(patch_priors(rotated["label"], rotated["branch_id"], rotated["depth"], start, full), priors)
            for key in arrays:
                np.testing.assert_array_equal(np.rot90(rotated[key], -k, axes=(-2, -1)), arrays[key])
        for stratum in ("foreground", "uniform"):
            def base_item(self, index):
                return dict(image=arrays["image"][None].copy(), label=arrays["label"].copy(),
                            gaussian=arrays["gaussian"][None].copy(), priors=priors.copy(),
                            start_zyx=start.copy(), case_id=self.case_ids[0], sampling_stratum=stratum)

            def dataset(cls=CohortDataset, mode=None, role="train"):
                options = {} if mode is None else {"spatial_augmentation": mode}
                obj = cls(manifest, ["b" if role == "outer_diagnostic" else "a"], role=role,
                          patch_size=16, steps=100, seed=17, augment=True, return_tensors=False,
                          sampling_fov="full", **options)
                obj._get_case = lambda case_id: arrays
                return obj

            with patch.object(PatchDataset, "__getitem__", base_item):
                for k, index in indices.items():
                    baseline, default = dataset(old.CohortDataset)[index], dataset()[index]
                    explicit, augmented = dataset(mode="none")[index], dataset(mode="xy90")[index]
                    for key in ("image", "label", "gaussian", "priors", "start_zyx"):
                        np.testing.assert_array_equal(default[key], baseline[key])
                        np.testing.assert_array_equal(explicit[key], baseline[key])
                    for key in ("image", "label", "gaussian"):
                        np.testing.assert_array_equal(augmented[key], np.rot90(default[key], k, axes=(-2, -1)))
                        np.testing.assert_array_equal(np.rot90(augmented[key], -k, axes=(-2, -1)), default[key])
                        assert augmented[key].flags.c_contiguous
                    for key in ("priors", "start_zyx"):
                        np.testing.assert_array_equal(augmented[key], default[key])
                    wrapper_count += 1
                for role in ("selection", "outer_diagnostic"):
                    off, on = dataset(mode="none", role=role), dataset(mode="xy90", role=role)
                    assert off.augment is False and on.augment is False
                    for key in ("image", "label", "gaussian", "priors", "start_zyx"):
                        np.testing.assert_array_equal(off[indices[1]][key], on[indices[1]][key])
        results.append(f"{name}: priors={priors.tolist()}; k=0,1,2,3 invariant")
    for filename in ("train.py", "pretrain_classifier.py"):
        tree = ast.parse((VARIANT / "code" / filename).read_text(encoding="utf-8"))
        argument = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                    and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--spatial-augmentation"]
        assert len(argument) == 1 and any(k.arg == "default" and ast.literal_eval(k.value) == "none" for k in argument[0].keywords)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "CohortDataset"]
        for call in calls:
            keywords = {k.arg: k.value for k in call.keywords if k.arg}
            role = ast.literal_eval(keywords["role"])
            if role == "train" and "spatial_augmentation" in keywords:
                assert ast.unparse(keywords["spatial_augmentation"]) == "args.spatial_augmentation"
                assert ast.literal_eval(keywords["augment"]) is True
            elif role != "train":
                assert "spatial_augmentation" not in keywords and ast.literal_eval(keywords["augment"]) is False
        assert sum("spatial_augmentation" in {k.arg for k in call.keywords} for call in calls) == 1
    source = (VARIANT / "code/classification_data.py").read_text(encoding="utf-8")
    assert "torch.from_numpy(np.ascontiguousarray(item[key]))" in source
    assert "torch" not in sys.modules
    report = "\n".join(["PASS: synthetic CPU semantic validation; no real data/model/Torch/GPU/metrics.", *results,
                        f"PASS: {wrapper_count} actual CohortDataset foreground/uniform rotations; default-none matches original wrapper; paired inverse and C-contiguity exact.",
                        "PASS: selection/outer disabled; train/pretrain flag default none and training-only wiring; tensor contiguity conversion checked statically."])
    (Path(__file__).parent / "semantic_validation.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
