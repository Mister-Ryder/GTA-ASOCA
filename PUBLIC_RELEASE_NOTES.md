# Public source scope, attribution and limitations

This is a source-only development snapshot. Main training Python source and
the frozen classifier stage retain their execution identities. The three-stage
GTA GPU smoke has passed using a separate two-epoch classifier-smoke final.pt;
formal segmentation is running with fixed classifier80 final.pt. New model
segmentation results are pending. Per-file hashes
and the separate public package identity are in PUBLIC_SOURCE_PROVENANCE.json.

## Provenance, license and citations

Retain the complete, unchanged NaviAirway MIT notice, including
`Copyright (c) 2022 Anton Wang`, with any redistributed substantial derived
code. The notice already exists at
`code/historical/NaviAirway-MIT.txt`. Its upstream source is
[NaviAirway/LICENSE](https://github.com/AntonotnaWang/NaviAirway/blob/main/LICENSE).
Keep `code/historical/THIRD_PARTY_NOTICES.md` and `source_hashes.json` together
with the vendored historical source. Those four Python files preserve the
historical source bytes; adapted execution lives in `code/model.py`.

The remaining GTAirway research code has no independently selected license.
Do not add a whole-repository MIT claim or attribute the GTAirway additions to
the NaviAirway copyright holder. This can be described as publicly available
research source; an independent license for the team's own contributions is a
separate decision. Third-party dependencies keep their own licenses and are
installed as dependencies rather than copied wholesale.

Cite the source lineage with Wang et al., *NaviAirway: a Bronchiole-sensitive
Deep Learning-based Airway Segmentation Pipeline*,
[arXiv:2203.04294](https://arxiv.org/abs/2203.04294), and link the
[upstream repository](https://github.com/AntonotnaWang/NaviAirway).
Preserve the GTAirway source snapshot identity, source hashes and an accurate
description of the new wrapper. Do not invent a publication or DOI for the
team's GTAirway work if it has not been established.

The [ASOCA citation policy](https://asoca.grand-challenge.org/CitationPolicy/)
requests both Gharleghi et al. papers:

- *Automated segmentation of normal and diseased coronary arteries — The ASOCA
  challenge*, Computerized Medical Imaging and Graphics 97, 102049 (2022),
  [doi:10.1016/j.compmedimag.2022.102049](https://doi.org/10.1016/j.compmedimag.2022.102049).
- *Annotated computed tomography coronary angiogram images and associated
  data of normal and diseased arteries*, Scientific Data 10, 128 (2023),
  [doi:10.1038/s41597-023-02016-2](https://doi.org/10.1038/s41597-023-02016-2).

Direct users to the [organizer's data access page](https://asoca.grand-challenge.org/).
Do not place CT images, masks or derived patient arrays in the source release.
The article's open-access license is not a substitute for checking the data
download's terms. Historical pretrained weights are also excluded from this
initial code-only release; document their provenance and availability without
silently implying that they are supplied or randomly initialized.

## Minimal runnable source scope

Use an explicit allowlist in a separate clean staging directory. Do not copy
the complete development directory or its Git history into the public release.

Required source files:

- `code/train.py`, `model.py`, `losses.py`, `data.py`,
  `classification_data.py`, `checkpoint_utils.py`, `pretrain_classifier.py`.
- `code/prepare_data.py`, `audit_priors.py`, `evaluate.py`,
  `evaluate_trajectory.py`, `metrics.py`.
- `code/historical/__init__.py`, `code/historical/func/__init__.py`, and the
  four exact historical sources listed in `code/historical/source_hashes.json`.
- `code/historical/README.md`, `THIRD_PARTY_NOTICES.md`,
  `NaviAirway-MIT.txt`, `source_hashes.json`.

For the already frozen classifier stage, additionally copy only its six exact
Python files, preserving relative paths, into
`classifier_pretraining/code/`: `pretrain_classifier.py`,
`classification_data.py`, `data.py`, `checkpoint_utils.py`, `losses.py`, and
`historical/func/AnatomicalClassifier.py`. Record their hashes without local
machine paths. Do not add new Python files inside that snapshot or rewrite its
source: the classifier's source hash covers all Python files in its stage.
Keep the license notices with the release. The frozen stage's
`checkpoint_utils.py` and `losses.py` differ from the updated common main
sources, although the used focal-loss formula is unchanged. Current main code
alone cannot be claimed to reproduce the old checkpoint's exact source hash.
This preserves the tested pretraining identity without publishing its weights,
logs or runtime directories.

Add a root README with tested commands, required externally obtained data and
weights, the actual environment versions, split seeds and explicit development
status; the tested core requirements.txt and optional requirements-plot.txt; this release note;
`.gitignore`; and `configs/public_paths.env.example`. The included `configs/scientific_settings.json` is a sanitized
training configuration containing the chosen scientific settings without
private machine paths. JSON is descriptive configuration: the current entry
points accept CLI arguments and do not automatically expand `${...}` strings.

Optional source files: `ops/plot_learning_curves.py` and its interpretation
guide. Include `verify_candidate.py` only with its complete, sanitized reference
protocol and documentation; it is a development gate, not an independent-test
validator. The copied legacy `run_queue.py`, `nnunet_runner.py`, `swin_model.py` and
five-fold aggregation helpers are retained as inactive source-package files.
Do not execute them for this protocol or infer that external baselines or
five-fold validation were performed for the new candidate.

Exclude `initialization/`, `classifier_stage/`, `prior_audit/`, `curves/`,
`runs/`, cache/raw data, server records, credentials, SSH launch helpers,
training/evaluation logs, checkpoints, saved predictions, binary archives,
`__pycache__/`, local environment folders, machine-specific provenance JSON and
temporary transfer files. Curated numerical results may be added later as a
separate evidence release after protocol and patient-role checks; do not
publish raw development output folders by default.

## Paths overridden through environment variables

Customize `configs/public_paths.env.example` outside the release and source it
in the shell. Pass these variables to the entry points; no Python source change
is needed. Example for a Linux research server:

```bash
source /path/to/customized/public_paths.env
python classifier_pretraining/code/pretrain_classifier.py \
  --manifest "${ASOCA_MANIFEST}" \
  --output "${GTA_OUTPUT_ROOT}/classifier" \
  --init-checkpoint "${GTA_AIRWAY_CLASSIFIER_INIT}" \
  --init-domain airway --semantic-head-policy reinit \
  --patch-size 128 --batch-size 2 --epochs 80 --steps 64 \
  --eval-every 5 --stats-samples 2048 --validation-patches 512 \
  --sampling geometry --sampling-fov image-body

python code/train.py \
  --manifest "${ASOCA_MANIFEST}" \
  --output "${GTA_OUTPUT_ROOT}/gta" \
  --backbone-init "${GTA_AIRWAY_BACKBONE_INIT}" --backbone-domain airway \
  --classifier-init "${GTA_OUTPUT_ROOT}/classifier/final.pt" \
  --classifier-domain asoca --classifier-head-policy load \
  --patch-size 128 --batch-size 2 --epochs 200 --steps 125 \
  --freeze-epochs 50 --partial-until 70 --seg-loss dice_ce \
  --sampling standard --sampling-fov image-body

python code/evaluate_trajectory.py \
  --manifest "${ASOCA_MANIFEST}" --raw-root "${ASOCA_RAW_ROOT}" \
  --training "${GTA_OUTPUT_ROOT}/gta" \
  --output "${GTA_OUTPUT_ROOT}/internal_four_trajectory"
```

These examples use the current 128-cubed starting recipe and illustrate path
overrides; verify them against the final tested release's scientific settings.
Run from the staged project root with the documented environment.
The private server defaults in `ops/start_classifier_stage.py` and the local
`configs/first_stage.json` are not public entry points; omit that helper and
sanitize the configuration during release staging.

## Method identity and accurate claims

The defensible common scope is fine, branching tubular structures in 3D CT.
Keep one GTA architecture: the same four-level backbone, three seven-way
auxiliary attribute heads, generation/structure/location guidance placement,
multiscale decoder fusion, classifier guidance map and Gaussian output branch.
Auxiliary class definitions describe the task's geometric context; they do
not add segmentation classes. ASOCA in this release is binary coronary lumen
segmentation, not multiclass whole-heart segmentation.

Document the genuine training differences: historical airway feature transfer,
coronary classifier pretraining on 28 patients with four internal diagnostic
patients, fixed 80-epoch final classifier initialization, explicit
frozen/partial/joint stages, decoder deep supervision,
training-only focal class weights, the selected shared segmentation loss and
sampler, and any optional Gaussian-weight schedule. The standard Dice+CE recipe
is a disclosed training change, not an identical historical-loss reproduction.
Deep supervision is a shared training aid; do not present it as a new GTA
architectural contribution.

The current enhanced segmentation path has no encoder segmentation deep
supervision. Encoder/decoder segmentation supervision belongs to the optional
backbone warm-up, which is disabled in the current starting recipe. Conditional
argmax/confidence decisions remain detached; classifier heads learn through
explicit classification losses, and trainable classifier features can learn
through segmentation after unfreezing. Historical feedback features remain
unused by the loss. Do not claim that these dead branches implement active
bidirectional refinement or differentiable soft semantic routing.

Classifier pretraining supplies a warm-up to stabilize joint optimization. Its
three attributes encode intrinsic priors and provide a multitask inductive
bias; classification is not the final task. Attribute accuracy and the
accuracy-selected `best_selection.pt` are diagnostic comparisons, not a GTA
startup performance gate. Main segmentation initializes from the completed
80-epoch `final.pt` and is judged by Dice and ASSD.

The ASOCA seven-way generation head currently represents branch-depth bins,
not measured vessel caliber; location is scan-relative, not guaranteed named
anatomy. Report the prior audit and classifier nonempty-class support as
auxiliary diagnostics without using them as an accuracy threshold for
segmentation training. Dice and explicitly defined ASSD are the
primary metrics; clDice is a structural diagnostic. ASSD is a surface-distance
measure, not connectivity.

The four internal development patients support trajectory/method decisions.
The original eight cases have already been examined during development and
are not an independent final test. A new method may be reported as a GTA
coronary extension after the common protocol and ablations support that claim;
these development runs alone do not establish generalization or superiority
over literature values obtained under different protocols.
