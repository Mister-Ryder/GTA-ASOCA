LOCAL PREPARATION ONLY: existing-head encoder deep supervision

Copied from the sealed gaussian_region_balance_v1 code. This is independent of
rotation_consistent_v1; no rotation augmentation is included. No server upload,
training or performance validation has been performed.

Only code/model.py, code/losses.py and code/train.py change. The original A/B
code and weights remain unchanged. --encoder-deep-supervision defaults off.
--encoder-ds-weight defaults to0.25, a candidate setting rather than an optimum;
it must be finite and nonnegative. With the flag off its effective weight is0.

During model.train(), the enabled auxiliary branch reuses the already computed
reversed encoder features, existing base_model.fusion_module_encoder and
base_model.final_conv. For128-cube input its raw encoder_aux_logits are
[batch,2,128,128,128]. Historical dropout1 is deliberately not called: this adds
auxiliary supervision, not restored dropout or a changed inference fusion.
No softmax precedes the loss and no new modules or state_dict keys are added.
model.eval() skips the encoder auxiliary branch, even when the flag is enabled;
the existing evaluator's two-argument make_model call therefore needs no extra
inference computation or encoder-head output.

The original final/two decoder losses retain their normalized1,.5,.25 weights.
Encoder supervision uses the same selected DiceCE or historical segmentation
function and sample weights, then adds lambda_encoder*encoder_loss separately.
Logs record encoder_supervision, encoder_supervision_weighted and actual
encoder_ds_lambda. Existing segmentation terms remain unchanged. Decoder --no-ds
and encoder supervision are independent switches.

gta_backbone explicitly removes the classifier/prior/guidance and Gaussian
modules while retaining the GTA wrapper's shared SegAirwayModel encoder/decoder.
It receives no implicit GT-attribute loss reweighting; the actual policy is
recorded as effective_anatomy_reweighting. It is not the old PlainUNet baseline,
whose architecture differs. Requesting encoder supervision for PlainUNet raises
an error. A future full/ablation comparison should use matching backbone donors,
supervision, sampling and budget; use --disable-anatomy-reweighting on both when
isolating prior/Gaussian effects, and study weighting separately if claiming it.

Reused heads preserve model weight structure, not saved source identity. Old
checkpoints remain governed by their original frozen source and verification;
this new source must not be claimed to pass the old strict source hash checks.
Only syntax has been checked so far. One small CPU synthetic forward/backward
check of enabled/disabled, baseline weighting and inference skipping is pending.
