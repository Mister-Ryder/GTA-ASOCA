# Third-party notices

This project includes code descended from NaviAirway by Anton Wang (2022). The original MIT license and copyright notice are preserved in [NaviAirway-MIT.txt](NaviAirway-MIT.txt).

The file-lineage inventory below describes the broader historical GTAirway
snapshot. This release vendors only the four sources listed in
`source_hashes.json`; the inventory is not a claim that all listed files are
included here or that every GTAirway addition is covered by the upstream MIT
license. The remaining GTAirway research source has no independently selected
license.

The following files are byte-identical to files in the earlier NaviAirway snapshot at the time this repository was prepared:

- func/__init__.py
- func/points_to_mesh.py
- func/post_process.py
- func/semi_supervise_learning.py
- func/ulti.py
- func/unet_3d_basic.py

Other files, including func/detect_tree.py, func/load_dataset.py, func/loss_func.py, func/model_arch.py and func/model_run.py, have a shared lineage with NaviAirway and contain later changes. Keep the original license notice when redistributing substantial portions of that code.
