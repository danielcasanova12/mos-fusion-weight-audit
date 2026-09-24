# Code map

`analysis/validate_public_release.py` is the only script that runs directly
from this public release without private data. It checks the cached CSVs and
the final manuscript-aligned numbers.

`analysis/run_system_disjoint_selection_equal_system.py` is the producing
driver for the final nested benchmark. It expects cached expert/OOF arrays and
the dataset manifests used on the compute server; those inputs are not part of
the release. The remaining analysis scripts are the producing/audit drivers
for the attribution, null, player-partition, interaction, and solver checks.

`extraction/extract_explainability_embeddings.py` is the resumable extractor.
It requires the licensed MOS audio, local model checkpoints, and a configured
output directory. No credentials or checkpoint binaries are embedded here.
