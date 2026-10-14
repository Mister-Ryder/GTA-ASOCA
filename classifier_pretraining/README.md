# Exact classifier-pretraining source snapshot

The six Python files in code/ are preserved byte-for-byte from the executed
standalone classifier stage. Their source hash is
61d752700a57dc02c6c1d7cdbcc59d1171957bf072f4fd46394af3980eff0bb1.
Do not add Python files under code/ when verifying or resuming that identity.
The final 80-epoch final.pt supplies GTA initialization; best_selection.pt is
an attribute diagnostic comparison, not the segmentation initializer.
Weights, logs and patient arrays are not included. License scope and upstream
attribution are documented at ../LICENSE_SCOPE.md and
../code/historical/THIRD_PARTY_NOTICES.md.
