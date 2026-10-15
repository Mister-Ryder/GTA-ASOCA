"""Prepare the public ASOCA training set without changing the input field of view.

Only binary segmentation labels are available. The auxiliary topology categories
below are geometric training targets, not named coronary artery classes.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk
from skimage.morphology import skeletonize

SCHEMA_VERSION = 1
PREPROCESSING_VERSION = "asoca-fixed-fov-cumulative-gaussian-v1"
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_cases(raw_root: Path) -> list[dict]:
    cases = []
    for group in ("Normal", "Diseased"):
        annotations = sorted((raw_root / group / "Annotations").glob("*.nrrd"),
                             key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        for annotation in annotations:
            image = raw_root / group / "CTCA" / annotation.name
            if not image.is_file():
                raise FileNotFoundError(image)
            cases.append({"case_id": annotation.stem, "group": group,
                          "image_relative_path": image.relative_to(raw_root).as_posix(),
                          "label_relative_path": annotation.relative_to(raw_root).as_posix()})
    if len(cases) != 40 or any(sum(c["group"] == g for c in cases) != 20
                               for g in ("Normal", "Diseased")):
        raise ValueError("Expected the 40 public training cases: 20 Normal + 20 Diseased")
    return cases


def create_folds(cases: list[dict], seed: int = 20261013) -> list[dict]:
    """Five frozen patient-level folds, each with four cases from each stratum."""
    rng = np.random.default_rng(seed)
    strata = {g: [c["case_id"] for c in cases if c["group"] == g]
              for g in ("Normal", "Diseased")}
    for ids in strata.values():
        rng.shuffle(ids)
    all_ids = sorted(c["case_id"] for c in cases)
    folds = []
    for fold in range(5):
        val = sorted(strata["Normal"][fold * 4:(fold + 1) * 4]
                     + strata["Diseased"][fold * 4:(fold + 1) * 4])
        train = [case_id for case_id in all_ids if case_id not in val]
        folds.append({"fold": fold, "train": train, "val": val})
    assert sorted(x for f in folds for x in f["val"]) == all_ids
    return folds


def geometry(image: sitk.Image) -> dict:
    return {"size_xyz": list(image.GetSize()),
            "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz_mm": list(image.GetOrigin()),
            "direction": list(image.GetDirection())}


def resample_full_volume(image: sitk.Image, spacing_mm: float, is_label=False,
                         reference: sitk.Image | None = None) -> sitk.Image:
    """Preserve physical extent, origin and direction; no segmentation-derived ROI."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0 if is_label else -200)
    resampler.SetOutputPixelType(sitk.sitkUInt8 if is_label else sitk.sitkFloat32)
    if reference is None:
        size = np.asarray(image.GetSize(), dtype=np.float64)
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        new_size = np.round((size - 1) * spacing / spacing_mm).astype(int) + 1
        resampler.SetSize([int(v) for v in new_size])
        resampler.SetOutputSpacing((spacing_mm,) * 3)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
    else:
        resampler.SetReferenceImage(reference)
    return resampler.Execute(image)


def mask_bbox(mask: np.ndarray, margin=1) -> tuple[slice, ...]:
    coords = np.nonzero(mask)
    if not len(coords[0]):
        raise ValueError("A public training mask is empty")
    return tuple(slice(max(0, int(c.min()) - margin),
                       min(mask.shape[a], int(c.max()) + margin + 1))
                 for a, c in enumerate(coords))


def topology_targets(mask: np.ndarray, spacing_mm: float) -> dict:
    """Skeleton branch graph with clustered junctions, rooted by maximum radius.

    Removing all 26-neighbour junction voxels leaves chain components (branches).
    Branches touching a junction cluster share an undirected graph edge. Minimum
    junction-crossing depth is BFS distance from the thickest branch of each
    connected graph component. This is a geometric approximation to airway-tree
    generation and is explicitly not an anatomical coronary hierarchy.
    """
    bbox = mask_bbox(mask)
    cropped = np.asarray(mask[bbox], dtype=bool)
    # Padding keeps distances and skeleton endpoints correct when a vessel is
    # close to the cropped auxiliary-computation boundary.
    padded = np.pad(cropped, 1)
    radius = ndi.distance_transform_edt(padded, sampling=spacing_mm)[1:-1, 1:-1, 1:-1]
    skeleton = skeletonize(padded, method="lee")[1:-1, 1:-1, 1:-1].astype(bool)
    if not skeleton.any():
        # Degenerate tiny components must not create an invalid auxiliary map.
        skeleton.flat[int(radius.argmax())] = True
    degree = ndi.convolve(skeleton.astype(np.uint8), CONNECTIVITY, mode="constant")
    degree = degree.astype(np.int16) - skeleton.astype(np.int16)
    junction = skeleton & (degree >= 3)
    junction_labels, junction_count = ndi.label(junction, structure=CONNECTIVITY)
    branches, branch_count = ndi.label(skeleton & ~junction, structure=CONNECTIVITY)
    # Entire skeleton components may consist of a junction cluster. Give each
    # of these a valid branch so that every skeleton point has a branch label.
    adjacency = {branch: set() for branch in range(1, branch_count + 1)}
    junction_members = {}
    cluster_boxes = ndi.find_objects(junction_labels)
    for junction_id in range(1, junction_count + 1):
        cluster_box = cluster_boxes[junction_id - 1]
        local_box = tuple(slice(max(0, s.start - 1), min(s.stop + 1, skeleton.shape[a]))
                          for a, s in enumerate(cluster_box))
        cluster = junction_labels[local_box] == junction_id
        touching = np.unique(branches[local_box][ndi.binary_dilation(cluster, structure=CONNECTIVITY)])
        touching = [int(b) for b in touching if b]
        if not touching:
            branch_count += 1
            local_branches = branches[local_box]
            local_branches[cluster] = branch_count
            adjacency[branch_count] = set()
            touching = [branch_count]
        junction_members[junction_id] = touching
        for branch in touching:
            adjacency[branch].update(other for other in touching if other != branch)
    if branch_count > np.iinfo(np.uint16).max:
        raise ValueError("Too many geometric branches for uint16")
    branch_radius = np.zeros(branch_count + 1, dtype=np.float32)
    np.maximum.at(branch_radius, branches[skeleton], radius[skeleton])
    depths = np.full(branch_count + 1, -1, dtype=np.int32)
    depths[0] = 0
    roots = []
    while np.any(depths[1:] < 0):
        unvisited = np.flatnonzero(depths < 0)
        root = int(unvisited[np.argmax(branch_radius[unvisited])])
        roots.append(root)
        depths[root] = 0
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for child in sorted(adjacency[current]):
                if depths[child] < 0:
                    depths[child] = depths[current] + 1
                    queue.append(child)
    for junction_id, members in junction_members.items():
        parent = min(members, key=lambda b: (depths[b], -branch_radius[b], b))
        cluster_box = cluster_boxes[junction_id - 1]
        local_branches = branches[cluster_box]
        local_branches[junction_labels[cluster_box] == junction_id] = parent
    if depths.max() > 255:
        raise ValueError("Topology depth exceeds uint8 range")
    # Assign each foreground voxel to its nearest centreline point. Computation
    # is cropped for memory only; stored arrays retain the full CT field of view.
    nearest = ndi.distance_transform_edt(~skeleton, return_distances=False,
                                        return_indices=True)
    branch_map = branches[tuple(nearest)].astype(np.uint16)
    branch_map[~cropped] = 0
    depth_map = depths[branch_map].astype(np.uint8)
    gaussian, gaussian_info = cumulative_gaussian(skeleton, radius, cropped, spacing_mm)
    coords = np.argwhere(skeleton)
    offset = np.array([s.start for s in bbox], dtype=np.int32)
    coords_global = (coords + offset).astype(np.int32)
    radius_at_skeleton = radius[tuple(coords.T)]
    thin_cutoff = float(np.quantile(radius_at_skeleton, 1 / 3))
    thin_coords = coords_global[radius_at_skeleton <= thin_cutoff]
    junction_coords = (np.argwhere(junction) + offset).astype(np.int32)
    foreground_coords = np.argwhere(cropped).astype(np.int32) + offset
    # Cap the sampling-coordinate list, preserving deterministic coverage.
    if len(foreground_coords) > 100000:
        selected = np.linspace(0, len(foreground_coords) - 1, 100000, dtype=int)
        foreground_coords = foreground_coords[selected]
    return {"bbox": bbox, "skeleton": skeleton.astype(np.uint8),
            "radius": radius.astype(np.float16), "branch_id": branch_map,
            "depth": depth_map, "gaussian": gaussian.astype(np.float16),
            "sampling": {"foreground": foreground_coords.astype(np.int32),
                         "skeleton": coords_global, "thin": thin_coords,
                         "junction": junction_coords},
            "metadata": {"skeleton_voxels": int(skeleton.sum()),
                         "junction_voxels": int(junction.sum()),
                         "branch_count": int(branch_count), "graph_roots": roots,
                         "maximum_graph_depth": int(depths.max()),
                         "thin_radius_cutoff_mm": thin_cutoff,
                         "gaussian": gaussian_info}}


def cumulative_gaussian(skeleton: np.ndarray, radius_mm: np.ndarray,
                        mask: np.ndarray, spacing_mm: float,
                        truncate_sigma: float = 4.0) -> tuple[np.ndarray, dict]:
    """Retain the original sum of radius-aware centreline Gaussians inside mask.

    Sparse local stamping replaces the original whole-volume mesh for every
    point. A 4-sigma cube omits an individual contribution only where it is
    <= exp(-8)=0.0003355. Floating maps avoid the original int16 quantization.
    The volume maximum normalizes the summed target to [0,1].
    """
    started = time.monotonic()
    target = np.zeros(skeleton.shape, dtype=np.float32)
    points = np.argwhere(skeleton)
    for point in points:
        sigma = max(float(radius_mm[tuple(point)]) / spacing_mm, 1.0)
        extent = int(np.ceil(truncate_sigma * sigma))
        low = np.maximum(point - extent, 0)
        high = np.minimum(point + extent + 1, skeleton.shape)
        vectors = [np.exp(-((np.arange(int(lo), int(hi)) - int(p)) ** 2)
                          / (2 * sigma * sigma)).astype(np.float32)
                   for lo, hi, p in zip(low, high, point)]
        sl = tuple(slice(int(lo), int(hi)) for lo, hi in zip(low, high))
        contribution = vectors[0][:, None, None] * vectors[1][None, :, None] \
                       * vectors[2][None, None, :]
        target[sl] += contribution * mask[sl]
    maximum = float(target.max())
    if maximum > 0:
        target /= maximum
    return target, {"definition": "sum_s exp(-||x-s||^2/(2*r_s^2))*mask(x)",
                    "normalization": "divide by per-volume positive maximum",
                    "truncate_sigma": truncate_sigma,
                    "maximum_omitted_single_point_contribution": float(np.exp(-8)),
                    "conservative_unnormalized_sum_error_bound": float(len(points) * np.exp(-8)),
                    "original_sum_maximum": maximum,
                    "stamp_seconds": round(time.monotonic() - started, 3)}


def write_full_array(path: Path, shape: tuple[int, ...], array: np.ndarray,
                     bbox: tuple[slice, ...]) -> None:
    output = np.lib.format.open_memmap(path, mode="w+", dtype=array.dtype, shape=shape)
    # A fresh w+ mapping is zero initialized by the OS. Preserve sparse file
    # holes for the large empty background instead of physically writing zeros.
    output[bbox] = array
    output.flush()
    del output


def prepare_case(task: tuple) -> dict:
    raw_root, output_root, case, spacing_mm, force = task
    raw_root, output_root = Path(raw_root), Path(output_root)
    destination = output_root / "cases" / case["case_id"]
    metadata_path = destination / "case.json"
    if metadata_path.is_file() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("preprocessing_version") != PREPROCESSING_VERSION \
                or existing.get("spacing_xyz_mm") != [spacing_mm] * 3:
            raise ValueError(f"Incompatible existing cache: {destination}; use a fresh output directory")
        required = ("image", "label", "skeleton", "radius", "branch_id", "depth", "gaussian")
        if all((destination / f"{name}.npy").is_file() for name in required) \
                and (destination / "sampling.npz").is_file():
            return existing
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    image_path = raw_root / case["image_relative_path"]
    label_path = raw_root / case["label_relative_path"]
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    original = geometry(image)
    label_geometry = geometry(label)
    if any(not np.allclose(original[key], label_geometry[key], atol=1e-6)
           for key in original):
        raise ValueError(f"Image/label physical geometry mismatch: {case['case_id']}")
    values = np.unique(sitk.GetArrayViewFromImage(label)).tolist()
    if values != [0, 1]:
        raise ValueError(f"ASOCA is expected to be binary, got {values}: {case['case_id']}")
    image = sitk.DICOMOrient(image, "LPS")
    label = sitk.DICOMOrient(label, "LPS")
    image_resampled = resample_full_volume(image, spacing_mm)
    label_resampled = resample_full_volume(label, spacing_mm, True, image_resampled)
    resampled_geometry = geometry(image_resampled)
    image_array = sitk.GetArrayFromImage(image_resampled)
    np.clip(image_array, -200, 1000, out=image_array)
    image_array -= 400
    image_array /= 600
    np.save(destination / "image.npy", image_array.astype(np.float16))
    mask = sitk.GetArrayFromImage(label_resampled).astype(np.uint8)
    np.save(destination / "label.npy", mask)
    del image_array, image, label, image_resampled, label_resampled
    auxiliary = topology_targets(mask, spacing_mm)
    for name in ("skeleton", "radius", "branch_id", "depth", "gaussian"):
        write_full_array(destination / f"{name}.npy", mask.shape,
                         auxiliary[name], auxiliary["bbox"])
    np.savez_compressed(destination / "sampling.npz", **auxiliary["sampling"])
    metadata = {**case, "cache_subdir": f"cases/{case['case_id']}",
                "preprocessing_version": PREPROCESSING_VERSION,
                "shape_zyx": list(mask.shape), "spacing_xyz_mm": [spacing_mm] * 3,
                "spacing_zyx_mm": [spacing_mm] * 3, "original_geometry": original,
                "image_path": str(image_path.resolve()), "label_path": str(label_path.resolve()),
                "resampled_geometry": resampled_geometry,
                "resampled_size_xyz": resampled_geometry["size_xyz"],
                "resampled_origin_xyz_mm": resampled_geometry["origin_xyz_mm"],
                "resampled_direction": resampled_geometry["direction"],
                "original_label_values": values,
                "foreground_voxels": int(mask.sum()),
                "foreground_fraction": float(mask.mean()),
                "source_sha256": {"image": sha256_file(image_path),
                                  "label": sha256_file(label_path)},
                "auxiliary": auxiliary["metadata"],
                "preparation_seconds": round(time.monotonic() - started, 3)}
    # This completion marker is written last; interrupted cases are recomputed.
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--case-ids", nargs="+", help="Smoke-test subset; manifest still freezes all 40 folds")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.spacing <= 0 or not 1 <= args.workers <= 4:
        parser.error("Spacing must be positive; workers must be between 1 and 4")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    cases = discover_cases(args.raw_root)
    folds = create_folds(cases, args.seed)
    selected = [c for c in cases if not args.case_ids or c["case_id"] in args.case_ids]
    if args.case_ids and len(selected) != len(set(args.case_ids)):
        parser.error("Some requested case IDs do not occur in the public training set")
    args.output.mkdir(parents=True, exist_ok=True)
    # Fixed plan is saved before expensive work and does not depend on labels.
    manifest = {"schema_version": SCHEMA_VERSION, "dataset": "ASOCA",
                "label_semantics": {"0": "background", "1": "coronary artery"},
                "raw_root_at_preparation": str(args.raw_root.resolve()),
                "split_seed": args.seed, "folds": folds, "cases": [],
                "planned_case_ids": [c["case_id"] for c in cases],
                "hidden_test_cases_used": False, "pilot_only": bool(args.case_ids),
                "preprocessing": {"version": PREPROCESSING_VERSION,
                                  "spacing_xyz_mm": [args.spacing] * 3,
                                  "spacing_selection": "fixed a priori, no validation labels or image statistics",
                                  "image_interpolation": "linear", "label_interpolation": "nearest",
                                  "orientation": "canonical LPS before isotropic resampling",
                                  "ct_window_hu": [-200, 1000],
                                  "ct_normalization": "(clip(HU,-200,1000)-400)/600",
                                  "input_fov": "entire resampled CT; no mask-derived image crop",
                                  "auxiliary_use": "training targets and training patch sampling only; never inference inputs",
                                  "topology": "26-neighbour Lee skeleton; cluster junctions; BFS from maximum-radius branch per graph component",
                                  "generation_bins": [[0], [1], [2, 3], [4, 5], [6, 7, 8], [9, 255]],
                                  "structure_bins": "original GTAirway 0..6 bounding-box/branch-complexity rule",
                                  "location_bins": "0 empty; 1..6 image-z tertile crossed with image-x half; geometric, not anatomy",
                                  "gaussian": "cumulative radius-aware sum, 4-sigma cube support, positive-volume maximum normalization",
                                  "float_storage": "image/gaussian/radius float16; labels/skeleton/depth uint8; branch_id uint16"}}
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("folds") != folds or previous.get("preprocessing") != manifest["preprocessing"]:
            raise ValueError("Existing manifest uses a different frozen protocol; choose a fresh output directory")
        manifest["cases"] = previous.get("cases", [])
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    manifest["complete"] = len(manifest["cases"]) == 40
    initial_path = manifest_path.with_suffix(".json.tmp")
    initial_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    initial_path.replace(manifest_path)

    def record(result):
        by_id[result["case_id"]] = result
        manifest["cases"] = sorted(by_id.values(), key=lambda c: c["case_id"])
        manifest["complete"] = len(manifest["cases"]) == 40
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
        print(f"Prepared {result['case_id']}: {result['shape_zyx']}, "
              f"{result['preparation_seconds']:.1f}s, "
              f"Gaussian {result['auxiliary']['gaussian']['stamp_seconds']:.1f}s", flush=True)

    tasks = [(str(args.raw_root), str(args.output), c, args.spacing, args.force) for c in selected]
    if args.workers == 1:
        for task in tasks:
            record(prepare_case(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(prepare_case, task) for task in tasks]
            for future in as_completed(futures):
                record(future.result())
    print(f"Manifest: {manifest_path}; complete={manifest['complete']}", flush=True)


if __name__ == "__main__":
    main()
