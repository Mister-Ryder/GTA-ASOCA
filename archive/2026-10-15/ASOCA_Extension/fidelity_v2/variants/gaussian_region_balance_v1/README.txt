B candidate prepared only; no training authorization

This folder copies exactly the frozen A package's 28 files. Only code/losses.py changes: one Gaussian reduction helper, its explicit semantic marker, and the existing Gaussian MSE call site. The other27 files remain byte-identical. losses_only.diff is the complete reviewable code delta. No trainer, classifier label, model, sampling, DS, initialization, phase, lambda or evaluation change is included.

B first computes separate foreground/background conditional MSE for each patch, averages present regions, then averages patches. With both regions present it is .5*mean_FG+.5*mean_BG. Empty-FG and all-FG patches equal A and remain in the batch. clamp_min(1) only prevents division by zero for an absent region; it does not cap small positive foreground counts or gradients. Target .5+.5G and lambda10 remain fixed. Existing gaussian/gaussian_weighted logs now mean B/10B; they are a changed loss measure and are not directly comparable to A's logged values.

The unchanged trainer already writes source_sha256 and source_files_sha256 into actual config and checkpoints. protocol.json states the B measure and binds its source/file map. ops/check_variant.py --audit-config reads the original A config and actual B config, rejects every actual difference except output/new source identities, and emits a sidecar with exact actual config SHA + protocol SHA + reduction semantics. It never patches config/checkpoint files. Actual config has no new gaussian_reduction key; the separate protocol binding is explicit, not hidden behind a claim that the old loss stayed unchanged.

Read-only source inspection:
  python -B ops/check_variant.py --inspect

Small hand/Torch fixture, with a fresh report path:
  python -B ops/check_variant.py --fixture --report candidate_torch_fixture.json

Actual config binding after a separately authorized B run writes its config:
  python -B ops/check_variant.py --audit-config --main-config ACTUAL_A_CONFIG --actual-config ACTUAL_B_CONFIG --expected-output NEW_ROOT/results/DISTINCT_B/training --report actual_config_binding.json

The fixture covers uneven_fg, all_empty, all_fg, one_voxel and batchavg. In a Torch environment it invokes the candidate helper, checks hand values and gradients against the reviewed actual-shadow B reduction, checks empty/full loss and gradient equality to A, and verifies mean-of-patches. This machine has no Torch: only Fraction hand calculations and source/AST checks have run locally. The real OLD shadow's five Torch fixtures passed, but that does not falsely certify this newly copied candidate helper's execution. Its actual Torch fixture remains a gate.

Actual final200 shadow decision note

The completed 56-patch shadow used the immutable A final model, the same training28 bank and detached FP32 main logits. A lambda10 total-logit norm / segmentation is .007813741; B is3.068482098. Margin ratios are .003867024 and1.422731024. B/A total norm is392.7033. Common energy fraction increases from75.5074% to78.5020%; B does not remove common-mode freedom. B's margin cosine to segmentation is-.0149393. These local values show a much stronger constraint, not a demonstrated useful optimization direction or segmentation gain.

Mean lambda10 A=.0002245285, B=.0331529482; segmentation=.260680605. Pooled q-derivative norm multipliers are621.2750 in FG and .536761 in BG. Five tiny-FG patches (120,196,243,217,305 voxels) supply54.4799% of pooled B logit-gradient energy. Thus small regions can dominate this direct balancing rule. No alpha adjustment, positive-count clamp, lambda change, common projection or added candidate is silently included.

The shadow is evaluated at final A weights, not the donor-initialized B model. Its logit derivatives omit parameter Jacobians and DS sibling paths; they do not establish the actual training parameter-gradient scale, clip fraction or future validation performance. Before a full B run, inspect batch2 actual parameter gradients and stability from the same initialization/recipe, including Gaussian-head gradients and nonfinite/clip12 behavior. Such a diagnostic must create no optimizer update and preserve/restore model state and RNG if training-mode BatchNorm is exercised. It is not executed by preparation or the small fixture.

Existing train.py:166-170 places all classifier parameters in one bucket and every other parameter, including gau_final, in the bucket called segmentation. Its segmentation_gradient_norm is therefore a parameter-group norm under total loss, not segmentation-loss-only or Gaussian-head-only strength. It records the two pre-clip group norms; clip_grad_norm_(12) is applied afterward, but the returned total norm/clip flag is not persisted. At logged steps the global pre-clip norm can be reconstructed as sqrt(group1^2+group2^2); epoch means and every25-step logs cannot give the exact full-run clip fraction. Epoch1/51/71 telemetry is useful subsequent stability evidence, and cannot replace the initial no-update parameter check or prove Gaussian-specific effects. No BC8 code/active run is changed to add telemetry.

Full B remains gated on actual-shadow review, canonical20 late-curve review, candidate Torch fixture, initial batch2 parameter-gradient/stability review, independent variant review, all canonical work finishing on NEW, and an explicit ROOT launch decision. The intended budget remains200x125 with lambda10 and the same NEW GPU; this folder starts nothing.
