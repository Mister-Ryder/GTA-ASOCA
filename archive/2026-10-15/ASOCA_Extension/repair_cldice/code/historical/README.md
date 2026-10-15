# Historical GTAirway v2 source

The four files in `func/` are unmodified copies of the current GTAirway main
snapshot. Their SHA-256 hashes are recorded in `source_hashes.json`. All four
also matched the original ICME-GTAirway-main-BAS files byte for byte when copied.

The original import paths in `EnhancedAirwaySegmentation2gau.py` are retained
for source inspection. The executable ASOCA wrapper imports the classifier,
guidance classes and backbone through `historical.func`, then copies the active
enhanced forward path in `model.py` to expose raw segmentation logits.

The original active computation uses four encoder widths `[32,64,128,256]`,
three decoder widths `[128,64,32]`, three seven-class patch attribute heads,
shallow location guidance, intermediate structure guidance, deep generation
guidance, six-channel decoder fusion, a sixteen-channel classifier guidance map,
and a Gaussian head applied to final segmentation logits. These are retained.
The unused historical adaptive threshold module and inactive code branches are
excluded from the wrapper; no threshold-based inference correction is added.

Changes in the executable wrapper:

- Exposes raw segmentation logits instead of post-softmax outputs.
- Uses the same architecture for two semantic segmentation classes (ASOCA
  background and coronary lumen). Attribute bins are auxiliary labels, not
  extra coronary segmentation classes.
- Adds optional half- and quarter-resolution decoder segmentation heads, with
  the output order explicitly documented. The plain U-Net receives the same
  optional supervision levels.
- Jointly trains the classifier from scratch because historical airway model
  checkpoints are unavailable; no frozen or partially frozen checkpoint stages
  are silently reproduced.
- Provides architectural ablations that remove the entire prior path or the
  Gaussian output head. The no-balancing ablation changes the trainer sampler,
  so its model is identical to the full model.

The original backbone contains some registered modules that the historical
enhanced forward does not use. They remain in the wrapper's `base_model` to
preserve source architecture and checkpoint naming; compute/parameter reports
should distinguish registered parameters from parameters in the active graph.

NaviAirway-derived portions retain the original MIT notice in
`NaviAirway-MIT.txt`; see `THIRD_PARTY_NOTICES.md`. No new license is asserted for
the remaining GTAirway research source.
