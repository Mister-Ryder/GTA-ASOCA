Isolated clDice baseline repair; author-smoothing objective, 2026-10-14

The active ASOCA_Extension/code directory is untouched. code/ here is a byte
copy of the frozen experiment source, except for losses.py and one new test.
repair_provenance.json records every Python source hash and the exact diff.
The source author's MIT license and pinned reference files are retained under
author_reference/ (outside the training-source tree).

Failure evidence and limits
  The archived queue log reports unet_cldice/fold0 failed. Its history contains
  153 complete saved epochs (0..152), with final mean loss 0.26166009; there is
  no evidence of an exploding forward loss before the failure. The parent agent
  reports a nonfinite gradient during epoch 153. The actual child-training log
  and failing batch tensors were not present in the local status archive.
  Thus this package identifies a credible numerical failure mode and supplies
  a reproducible stress test; it does not claim to have proven the exact
  failing-batch cause without replaying that batch.

Why the former term is fragile
  Its per-sample empty-GT clDice loss, for predicted soft-skeleton mass S and
  epsilon e=1e-5, is approximately S/(S+2e). The derivative with respect to S
  near zero is 1/(2e)=50,000. When near-uniform probabilities produce a tiny
  skeleton while their sigmoid/softmax derivatives remain appreciable, backward
  can overflow a half-precision model output after multiplying by scale1024.
  The current trainer rejects nonfinite gradients before GradScaler.step/update
  can skip the step and reduce its scale, making a recoverable AMP overflow fatal.

Precise objective repair
  Pinned author commit: 47d31a6cc4a8101b1ffe8052994821961e57af9f
  Author file: cldice_loss/pytorch/cldice.py, soft_cldice class.
  https://github.com/jocpae/clDice/blob/47d31a6cc4a8101b1ffe8052994821961e57af9f/cldice_loss/pytorch/cldice.py
  The clDice term now uses smooth=1, sums over the complete batch and foreground
  channel, and uses tprec+tsens without an extra harmonic-denominator epsilon.
  Its ten-iteration skeletonization already matched the author's implementation
  and is unchanged. With smooth1, the empty-target mass derivative near zero
  is about 0.5 rather than 50,000.
  The common background-Dice plus foreground-power-Dice, additive topology
  coefficient0.2, decoder DS, sampling, model, optimizer, LR, seed, patch/batch
  sizes and 200x125 optimizer-step budget are unchanged.
  train.py is byte-identical to the frozen main trainer. No overflow retry,
  skipped-step policy or numerical mode change is introduced without evidence
  that the author-term fix remains insufficient.

Fresh training and scientific reporting
  Changing smoothing and batch reduction changes the objective. Each of all
  five repaired baseline folds must start at epoch0 in a fresh output directory.
  Do not resume epoch153, copy its optimizer/model state, or combine its history
  with the repaired result. Its partial failed run is retained as failure
  evidence and contributes no publication baseline score.
  A resume from the failed run can be used only for a separately labeled
  numerical diagnosis, never as this fresh 25,000-step baseline.
  Existing successful GTA/U-Net/Swin/nnU-Net experiments need no change. Any
  aggregate that admits the repaired baseline must explicitly record this
  repair identity/source hash rather than pretending it belongs to the old
  unmodified loss protocol.

Checks before staging GPU training
  All repaired Python files passed syntax parsing locally. Local Python lacks
  torch, so tensor checks have not yet run here. The root agent coordinates
  server tests; this agent has made no server connection during the repair.
  CUDA_VISIBLE_DEVICES='' python code/test_official_cldice.py --device cpu
  python code/test_official_cldice.py --device cuda
  The tests compare author formula/soft skeleton on filled, empty and mixed
  batches, check perfect-match and empty-skeleton behavior, force finite-forward
  but infinite-half-backward in the old term on a near-uniform empty patch, and
  confirm the repaired term remains finite at scale1024 and coefficient0.2.
  They also verify the base segmentation objective is unchanged.

Fresh-run command pattern (root supplies the authoritative server paths)
  python code/train.py --manifest <frozen-cache/manifest.json>
    --output <fresh-repaired-results/unet_cldice/foldN/training>
    --method unet_cldice --fold N --epochs 200 --steps 125 --patch-size 96
    --batch-size 2 --workers 4 --seed 20261013 --lr 0.0002
  Repeat with N=0..4. Do not pass --resume or --smoke.
  python code/evaluate.py --manifest <same-manifest> --raw-root <same-raw-root>
    --checkpoint <fresh-fold/training/final.pt> --output <fresh-fold/evaluation>
    --fold N
  Do not start the copied full run_queue.py; this package's purpose is the
  repaired baseline only, under root-controlled single-GPU scheduling.
