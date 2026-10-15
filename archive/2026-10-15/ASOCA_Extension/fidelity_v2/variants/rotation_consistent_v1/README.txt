LOCAL PREPARATION ONLY: XY rotation-consistent training augmentation

This variant starts from the sealed gaussian_region_balance_v1 code. It has not
been uploaded, trained or shown to improve segmentation performance.

Exactly three code files change:
  code/classification_data.py
  code/train.py
  code/pretrain_classifier.py

The optional --spatial-augmentation flag defaults to none. Choosing xy90 rotates
the final sampled training patch by k=0,1,2,3 quarter-turns in the YX plane, with
the same rotation applied to CT, segmentation GT and Gaussian target. A separate
SeedSequence([seed, epoch, index, 29015]) chooses k without consuming the existing
sampling or intensity-augmentation RNG. It is applied after the wrapper's final
uniform recrop and before tensor conversion, covering foreground and uniform
samples. Patches must have square YX dimensions when xy90 is requested.

Only role=train and augment=True enable rotation. The three attribute labels and
original crop start remain unchanged: generation/structure are rigid-rotation
invariant local geometry summaries; location denotes the original full-volume
patch center. Selection and diagnostic datasets remain untouched. Classifier
pretraining accepts the flag only for its training dataset; no classifier has
been retrained for this preparation.

No flips, interpolation, model/loss/field changes, new labels, changed sampling,
deep-supervision weights, donor initialization or staged schedule are introduced.
Small CPU semantic checks establish intended label/array behavior only. A future
training decision and actual performance evidence remain separate requirements.
