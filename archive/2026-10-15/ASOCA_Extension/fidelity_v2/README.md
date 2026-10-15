# GTAirway: ASOCA fidelity extension

This is a development implementation of the GTAirway framework for coronary
artery segmentation. The intended AEI study concerns the same engineering
problem across airway and coronary CT: segmenting small, branching tubular
structures under severe foreground and scale imbalance, with intrinsic
hierarchical, structural and spatial priors guiding multiscale features. The
auxiliary tasks provide a multitask inductive bias and help stabilize
optimization; the final task is segmentation, judged by Dice and ASSD. It is
not a separate whole-heart architecture or an attribute-classification study.

**Status, 14 October 2026:** main training source is frozen. Historical backbone
CPU transfer and a historical-classifier CUDA step at `128³`, batch size 2,
have passed mechanical checks. The new GTA frozen/partial/joint CUDA smoke has
not yet completed; formal 200-epoch segmentation training has not started.
The separate 80-epoch coronary classifier has completed. Work has resumed after
the server restart. The next steps are to verify available storage and weight
identities, run the main three-stage GPU smoke using the fixed classifier
`final.pt`, then the formal 200-epoch GTA recipe and internal-four trajectory.
There are no new fidelity_v2 segmentation results to report.
A public source release can
precede final results once the execution version and its mechanical checks are
settled; it must retain this development status.

## Dataset and development split

[ASOCA](https://asoca.grand-challenge.org/) provides 40 annotated training CCTA
volumes: 20 normal and 20 diseased cases. Here the segmentation target is binary:
background and coronary artery lumen. The three seven-class auxiliary heads
are patch attributes, not seven coronary segments or additional cardiac organs.
No organizer-held-out test labels are used.

The existing patient folds remain fixed with seed `20261013`. For fold 0, the
original 32 training patients are split with seed `20261014` into 28 training
patients and four internal development patients, two from each patient stratum:
`Diseased_10`, `Diseased_20`, `Normal_12`, `Normal_3`. Classifier fitting,
training-frequency weights and segmentation fitting use the 28 patients. The
four patients support attribute diagnostics and segmentation trajectory/method
decisions; they are not an independent final test. The classifier's fixed
80-epoch `final.pt` initializes segmentation. The pretrainer also emits an
accuracy-selected `best_selection.pt`, which is retained only as an attribute
diagnostic comparison and is not the main GTA initializer.

The original eight fold-0 cases have already been inspected during development.
Their earlier results remain development references. Their past exposure is
not undone by reserving them again. Evaluation defaults to the internal four;
the original eight require explicit `--split outer_development`. No automatic
five-fold or external-baseline queue is part of the current experiment.

The cache keeps the entire CT field of view, canonical LPS orientation and
fixed `0.5 mm` isotropic spacing. CT uses linear interpolation, labels nearest
neighbor, HU window `[-200,1000]`, and normalization `(HU-400)/600` after clipping.
GT-derived geometry may select training patches or define targets. Validation
classification crops use CT-only body bounds; the segmentation predictor takes
CT only and evaluates the full volume. GT maps are never model inputs.

## Retained framework and disclosed adaptations

The active architecture retains the historical four-level backbone with widths
`32/64/128/256`, three seven-way attribute heads, shallow encoder location
guidance, deep decoder generation guidance, intermediate decoder structure
guidance, multiscale decoder fusion, the classifier's 16-channel guidance map,
and a Gaussian branch computed from final segmentation logits. Four vendored
historical files remain byte-identical; see
[historical provenance](code/historical/source_hashes.json).

The coronary auxiliary targets currently mean:

| Head | Class 0 | Classes 1–6 |
| --- | --- | --- |
| Generation | Empty patch | Root-based branch depths `0`, `1`, `2–3`, `4–5`, `6–8`, `9+` |
| Structure | Empty patch | Existing bounding-box and branch-complexity bins |
| Location | Empty patch | Image-z tertile crossed with image-x half |

Depth is not measured vessel caliber, structure bins are geometric proxies, and
location is scan-relative context rather than guaranteed named anatomy. The
prior audit documents these interpretations before any target rewrite. These
attributes encode intrinsic priors through the current geometric proxies; they
are auxiliary training targets, not the final prediction task. Classification
accuracy is a diagnostic reference, not a performance gate for starting GTA or
the final optimization criterion. Classification reports include positive support, absent classes,
nonempty balanced accuracy, training-majority and always-empty baselines.

Historical BAS airway weights are feature warm-starts. Their filename/epoch
does not establish that they are the best checkpoint or optimal initialization
for coronary CT. The coronary classifier's three semantic heads are
reinitialized and trained separately. Parameter transfer records source hashes,
component keys, excluded semantic heads and strict shape checks. Users provide
initialization weights separately; they are not bundled in the public source.

Decoder deep supervision is enabled: the final output plus existing half- and
quarter-resolution decoder outputs receive weights `1, 0.5, 0.25`, normalized
by their sum. Auxiliary logits are upsampled to the target grid. This is not
comprehensive encoder segmentation supervision. Historical encoder/decoder
double supervision belongs to optional backbone warm-up; that warm-up is
disabled in the current recipe, which instead transfers the historical backbone.
Old airway results must not be relabeled as experiments with these new DS heads.

Classifier argmax/confidence conditions remain detached as in the active
historical path. Explicit focal objectives train the semantic heads; trainable
classifier features also receive segmentation gradients after unfreezing.
Unused historical feedback features do not implement an active feedback loss.

## Current starting recipe and budget

| Stage | Patients | Budget | Main settings |
| --- | --- | --- | --- |
| Coronary classifier pretraining | 28; diagnose on internal 4 | `80 × 64 = 5,120` optimizer steps | Fixed final checkpoint; `128³`, batch 2, AdamW `1e-4`, 5-epoch warm-up, geometric training sampling |
| Enhanced segmentation | 28 | `200 × 125 = 25,000` optimizer steps | `128³`, batch 2, AdamW `3e-4`, weight decay `1e-4`, 5-epoch warm-up, standard sampling |

Segmentation classifier stages, in one-based epochs, are frozen in epochs
1–50, partially trainable in 51–70, and jointly trainable in 71–200. Frozen
means no classifier gradients, BN updates or dropout, and no classification or
guidance auxiliary loss. Partial training keeps encoder blocks 0–5 frozen with
BN in evaluation mode; deeper blocks, attention, fusion and heads receive real
gradients. Classifier LR is `0.1` of segmentation LR.

The common segmentation objective is foreground Dice plus CE, with the same
sampler, FOV and decoder DS for the optional plain 3D U-Net diagnostic. GTA
retains its anatomy weighting and auxiliary terms. Focal frequency weights
come only from a deterministic 2,048-patch bank on the 28 patients; structure
focal receives factor `1.5`, and combined classification weight is `0.1` when
active. Guidance-map MSE weight is `0.1`. Gaussian MSE weight is `10`, with the
historical target `0.5 + 0.5 × cached_unit_field`. This explicitly changes the
segmentation objective from historical squared/power Dice; it is not an
identical historical-loss reproduction.

The optional late Gaussian ramp is disabled. If tested in a separate version,
it preserves the base weight through one-based epoch 150 and reaches the late
weight at epoch 200. Logits-gradient probes record the weighted Gaussian versus
segmentation derivatives without adding parameter gradients or optimizer
updates. Raw/weighted loss components, stage, LR, gradient norms and memory
usage are logged; immutable intermediate weights are saved every 10 epochs.
The final segmentation checkpoint is the fixed final epoch, not the best of
repeated original-eight evaluations.

Report the extra **5,120 classifier-pretraining steps separately from the
25,000 segmentation steps**. Historical airway pretraining has additional
unreconstructed cost. Preprocessing, classification validation and internal-four
trajectory inference also cost time. Equal segmentation-step counts alone do
not make total compute or initialization identical to another method.

## Environment

The new server has measured the following versions; these are not inferred
from the older experiment environment:

| Component | Observed version |
| --- | --- |
| Python | `3.11.17` |
| PyTorch | `2.7.1+cu128` |
| PyTorch CUDA runtime | `12.8` |
| MONAI | `1.5.1` |
| NumPy | `1.26.4` |
| GPU | NVIDIA vGPU, 32 GB, compute capability `8.9` |

SciPy, SimpleITK and scikit-image are also required for preparation/evaluation;
Matplotlib is needed only for rendering learning curves. Their complete measured
package lock is pending collection, so this checkout is not yet a turnkey
environment release. Record versions and the tested environment lock when
publishing the execution version. CUDA inference is required by the current
evaluator.

## Source layout and portable paths

Run commands from this project root in the activated documented environment.
Paths are supplied through environment variables and CLI flags. Do not insert
data, passwords, private machine paths or initialization binaries into source.

The public layout keeps the frozen classifier's **six exact Python files**
under `classifier_pretraining/code/`, separate from `code/` used by enhanced
training. In the current development checkout that snapshot is at
`classifier_stage/code/`; set `GTA_CLASSIFIER_CODE=classifier_stage/code` when
using that checkout. Preserve relative paths and do not add Python files inside
the frozen classifier snapshot. Its source hash covers all six files. The
updated main `losses.py` and `checkpoint_utils.py` are not byte-identical to
those classifier-stage versions.

```bash
export GTA_PYTHON="${GTA_PYTHON:-python}"
export GTA_CLASSIFIER_CODE="${GTA_CLASSIFIER_CODE:-classifier_pretraining/code}"
export ASOCA_RAW_ROOT="${ASOCA_RAW_ROOT:-/path/to/ASOCA/CompressedVersion}"
export ASOCA_CACHE_ROOT="${ASOCA_CACHE_ROOT:-/path/to/prepared-ASOCA/cache}"
export ASOCA_MANIFEST="${ASOCA_MANIFEST:-${ASOCA_CACHE_ROOT}/manifest.json}"
export GTA_OUTPUT_ROOT="${GTA_OUTPUT_ROOT:-/path/to/experiment-results}"
export GTA_AIRWAY_BACKBONE_INIT="${GTA_AIRWAY_BACKBONE_INIT:-/path/to/airway_backbone.pkl}"
export GTA_AIRWAY_CLASSIFIER_INIT="${GTA_AIRWAY_CLASSIFIER_INIT:-/path/to/airway_classifier.pkl}"
export GTA_CORONARY_CLASSIFIER_INIT="${GTA_CORONARY_CLASSIFIER_INIT:-${GTA_OUTPUT_ROOT}/classifier80x64/training/final.pt}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
mkdir -p "${GTA_OUTPUT_ROOT}"
```

Use externally obtained weights with documented provenance, or explicitly
declare a separate from-scratch control. The staged GTA command below does not
silently replace missing weights with random initialization. JSON configuration
is descriptive; the entry points do not automatically expand `${...}` inside
JSON. Do not use the copied legacy full queue to launch this development recipe.

### Preparation and prior audit

Use an existing audited complete cache when available. To prepare a fresh cache,
the raw tree must contain `Normal/CTCA`, `Normal/Annotations`,
`Diseased/CTCA`, `Diseased/Annotations` with corresponding NRRD filenames:

```bash
"${GTA_PYTHON}" code/prepare_data.py \
  --raw-root "${ASOCA_RAW_ROOT}" --output "${ASOCA_CACHE_ROOT}" \
  --spacing 0.5 --workers 2 --seed 20261013

"${GTA_PYTHON}" code/audit_priors.py \
  --manifest "${ASOCA_MANIFEST}" --output "${GTA_OUTPUT_ROOT}/prior_audit" \
  --fold 0 --samples 384 --sampling both --seed 20261013 --epoch 0 --patch-size 96
```

The read-only audit samples the original training 32 and does not train a model,
rewrite labels or fit focal weights. Its existing `96³` sampling is a label
diagnostic, not a measurement of the `128³` classifier bank's class frequencies.
The latter frequencies are measured separately on the 28 training patients.
Depth/radius, geometric-junction and padding/FOV analyses are descriptive;
image-only HU probes are not proof of neural-classifier learnability.

### Bounded classifier and three-stage GTA smoke

Run these in separate outputs from formal training. They test execution,
transfer, finite gradients and stage transitions, not segmentation performance
or prior usefulness. The classifier smoke is an optional setup check; the
80-epoch classifier is already complete in the current development run. The
pending main smoke below uses that completed fixed-final checkpoint:

```bash
"${GTA_PYTHON}" -u "${GTA_CLASSIFIER_CODE}/pretrain_classifier.py" \
  --manifest "${ASOCA_MANIFEST}" --output "${GTA_OUTPUT_ROOT}/classifier_smoke/training" \
  --fold 0 --patch-size 128 --batch-size 2 --workers 2 \
  --epochs 2 --steps 2 --lr 0.0001 --warmup-epochs 5 \
  --seed 20261013 --split-seed 20261014 \
  --stats-samples 32 --validation-patches 64 --eval-every 1 \
  --sampling geometry --sampling-fov image-body --guidance-weight 0.1 \
  --init-checkpoint "${GTA_AIRWAY_CLASSIFIER_INIT}" \
  --init-domain airway --semantic-head-policy reinit --device cuda --smoke

"${GTA_PYTHON}" -u code/train.py \
  --manifest "${ASOCA_MANIFEST}" --output "${GTA_OUTPUT_ROOT}/gta_smoke/training" \
  --method gta --fold 0 --cohort internal28 \
  --epochs 3 --steps 2 --patch-size 128 --batch-size 2 --workers 2 \
  --seed 20261013 --split-seed 20261014 \
  --lr 0.0003 --weight-decay 0.0001 --warmup-epochs 0 --backbone-warmup-epochs 0 \
  --freeze-epochs 1 --partial-until 2 --classifier-lr-scale 0.1 \
  --seg-loss dice_ce --sampling standard --sampling-fov image-body \
  --classifier-weight 0.1 --guidance-weight 0.1 --gaussian-weight 10 \
  --classifier-init "${GTA_CORONARY_CLASSIFIER_INIT}" \
  --classifier-domain asoca --classifier-head-policy load \
  --backbone-init "${GTA_AIRWAY_BACKBONE_INIT}" --backbone-domain airway \
  --save-every 1 --gradient-probe-every 1 --device cuda --smoke
```

With backbone warm-up disabled, the three enhanced epochs exercise frozen,
partial and joint stages in order. The formal initializer remains the completed
80-epoch classifier `final.pt`; tiny classifier-smoke weights are separate
mechanical-test artifacts.

### Coronary classifier pretraining

```bash
"${GTA_PYTHON}" -u "${GTA_CLASSIFIER_CODE}/pretrain_classifier.py" \
  --manifest "${ASOCA_MANIFEST}" --output "${GTA_OUTPUT_ROOT}/classifier80x64/training" \
  --fold 0 --epochs 80 --steps 64 --patch-size 128 --batch-size 2 --workers 4 \
  --seed 20261013 --split-seed 20261014 --lr 0.0001 --warmup-epochs 5 \
  --stats-samples 2048 --validation-patches 512 --eval-every 5 \
  --sampling geometry --sampling-fov image-body --guidance-weight 0.1 \
  --init-checkpoint "${GTA_AIRWAY_CLASSIFIER_INIT}" \
  --init-domain airway --semantic-head-policy reinit --device cuda
```

The fixed validation bank covers the internal four with 128 CT-only patches
per case. The frozen pretrainer uses mean three-head **nonempty** balanced
accuracy to emit a diagnostic `best_selection.pt`, with absent-class reporting.
Use `final.pt` after all 80 epochs for GTA initialization. Reports on positive
support, confusion matrices, majority/empty baselines and the prior audit help
interpret the auxiliary tasks; they do not impose an accuracy threshold or
approval gate on segmentation training. Pretraining is a warm-up intended to
reduce the convergence instability of learning the auxiliary and segmentation
objectives together from random initialization. Its value is judged by the
subsequent segmentation outcome, not by maximizing attribute accuracy.

The saved-recipe continuation checks completed fixed-final checkpoint identity
and the mechanical smoke. Any attribute-quality review is diagnostic rather
than a classifier-performance requirement. If deployment wrappers are
published, include their current supporting files; older quality-gate wrappers
must not be mistaken for the current method policy.

### Fixed-budget enhanced segmentation

After the restarted server's storage/weight checks and three-stage mechanical
smoke, continue the chosen execution version with the completed classifier:

```bash
"${GTA_PYTHON}" -u code/train.py \
  --manifest "${ASOCA_MANIFEST}" --output "${GTA_OUTPUT_ROOT}/gta200x125/training" \
  --method gta --fold 0 --cohort internal28 \
  --epochs 200 --steps 125 --patch-size 128 --batch-size 2 --workers 4 \
  --seed 20261013 --split-seed 20261014 \
  --lr 0.0003 --weight-decay 0.0001 --warmup-epochs 5 --backbone-warmup-epochs 0 \
  --freeze-epochs 50 --partial-until 70 --classifier-lr-scale 0.1 \
  --seg-loss dice_ce --sampling standard --sampling-fov image-body \
  --classifier-weight 0.1 --guidance-weight 0.1 --gaussian-weight 10 \
  --classifier-init "${GTA_CORONARY_CLASSIFIER_INIT}" \
  --classifier-domain asoca --classifier-head-policy load \
  --backbone-init "${GTA_AIRWAY_BACKBONE_INIT}" --backbone-domain airway \
  --save-every 10 --gradient-probe-every 250 --device cuda
```

DS is enabled by default; do not add `--no-ds` to this recipe. Adding `--plan`
prints the intended protocol without training. Existing output directories
cannot be silently overwritten; compatible continuation requires the same
command with `--resume`. Source, manifest and protocol changes require a new
run identity rather than resuming a scientifically different experiment.

For a separately chosen plain U-Net development diagnostic, use the same
segmentation settings with `--method unet`, a different output directory and
without the backbone/classifier initialization flags. It receives no GTA
classifier pretraining. This is a shared-recipe diagnostic, not a claim to
reproduce a named paper's 3D U-Net or native nnU-Net recipe.

### Internal-four trajectory evaluation

Run separately after segmentation completes, with one GPU process at a time:

```bash
"${GTA_PYTHON}" code/evaluate_trajectory.py \
  --training "${GTA_OUTPUT_ROOT}/gta200x125/training" \
  --manifest "${ASOCA_MANIFEST}" --raw-root "${ASOCA_RAW_ROOT}" \
  --output "${GTA_OUTPUT_ROOT}/gta200x125/internal4_trajectory" \
  --python "${GTA_PYTHON}" --sw-batch-size 1

# Optional rendering, if this plotting companion is included.
"${GTA_PYTHON}" ops/plot_learning_curves.py \
  --run "gta=${GTA_OUTPUT_ROOT}/gta200x125/training" \
  --validation "gta=${GTA_OUTPUT_ROOT}/gta200x125/internal4_trajectory/validation_history.jsonl" \
  --output "${GTA_OUTPUT_ROOT}/gta200x125/learning_curves"
```

The trajectory runner checks live training dependencies, checkpoint-internal
config/epoch, manifest and prediction identities. Intermediate weights remain
trajectory diagnostics and do not replace fixed-final weights. For one final
internal-four evaluation, the equivalent entry point is:

```bash
"${GTA_PYTHON}" code/evaluate.py \
  --checkpoint "${GTA_OUTPUT_ROOT}/gta200x125/training/final.pt" \
  --manifest "${ASOCA_MANIFEST}" --raw-root "${ASOCA_RAW_ROOT}" \
  --output "${GTA_OUTPUT_ROOT}/gta200x125/internal4_final" \
  --fold 0 --split internal_diagnostic --sw-batch-size 1
```

## Metrics and previous reference only

Primary metrics are Dice and `assd_mm`, measured after prediction is returned
to the original CT grid. ASSD pools the two lists of nearest boundary-voxel
distances in physical millimeters:
`(sum(pred→GT)+sum(GT→pred))/(n_pred_surface+n_GT_surface)`.
The separate equal-weight mean of directional means is also recorded. ASSD
measures surface distance, not connectivity. clDice and centerline measures are
structural diagnostics. Empty-prediction distances are null; a complete-cohort
ASSD mean is not reported if any distance is undefined.

These are **previously inspected eight-case development references**, not new
fidelity_v2 results or a literature ranking:

| Previous experiment | Mean Dice (%) | Mean pooled ASSD (mm) |
| --- | ---: | ---: |
| Cold-start GTA with decoder DS | 80.00695 | 1.17734 |
| Shared old-recipe plain U-Net | 78.70210 | 1.27216 |
| Shared old-recipe Swin UNETR | 78.57259 | 1.53376 |
| Native nnU-Net, 100 × 250 updates | 81.474844 | 2.48754067 |

The first three old experiments used `96³` patches and the earlier recipe;
the native nnU-Net row follows its own preprocessing, sampling and training
recipe. All are eight-case development references. They are not fair ablations
of the new 28-patient staged candidate. No result in this table demonstrates
that the new initialization, stages, `128³` context or prior labels improve
segmentation. See [the metric/development protocol](docs/methodology_and_metric_protocol.txt)
for the precise evaluation conventions.

A useful pending common-preprocessing sanity check is original GT → cached
`0.5 mm` GT → original-grid nearest-neighbor roundtrip, with per-case Dice,
ASSD and foreground-volume change. This diagnoses quantization; it is not a
mathematical upper bound, and no explanatory conclusion is claimed before
measuring it.

## Public release, provenance and citation

See [PUBLIC_RELEASE_NOTES.md](PUBLIC_RELEASE_NOTES.md) for the source allowlist,
frozen classifier snapshot, portable paths and excluded data/weights/runtime
artifacts. The direct CLI examples require the listed main source files plus
the exact six-file classifier snapshot and dependencies. The saved deployment
wrappers additionally require their `preflight.sh`, source-freeze verifier,
complete corresponding freeze manifest and current continuation helpers; copying a
wrapper alone is not a runnable minimum release. A reduced public package must
receive its own package-membership manifest rather than claiming the original
full development package hash.

Retain the unchanged [NaviAirway MIT notice](code/historical/NaviAirway-MIT.txt)
and [third-party notice](code/historical/THIRD_PARTY_NOTICES.md). The team's
remaining GTAirway source has no independently selected license; this release
is not declared wholly MIT-licensed. Cite
[NaviAirway](https://arxiv.org/abs/2203.04294) for inherited source lineage.
Follow the [ASOCA citation policy](https://asoca.grand-challenge.org/CitationPolicy/)
and cite both the
[2022 challenge paper](https://doi.org/10.1016/j.compmedimag.2022.102049) and
[2023 data descriptor](https://doi.org/10.1038/s41597-023-02016-2).

One common GTA-paper claim requires coherent architecture, disclosed task
adaptations, transparent prior diagnostics, informative ablations and final
validation after method choices are fixed. The current development work does
not yet establish superiority, cross-anatomy generalization or clinical utility.
