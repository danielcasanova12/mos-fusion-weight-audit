# Reproducibility artifact — final corrected version

Use `supplementary_reproducibility_artifact_final_20260924.tgz` as the public
artifact for the submitted PDF. It replaces the earlier
`supplementary_reproducibility_artifact.tgz`, whose nested-selection CSVs were
from a preliminary utterance-weighted run and therefore do not match the
submitted manuscript.

The corrected release is rooted at `mos-fusion-weight-audit-v1.1/` and contains:

- the submitted PDF and an editable TeX companion;
- `configs/experiments.yaml`, with the final equal-system objective, folds,
  seeds, SLSQP tolerances, and uncertainty settings;
- all 20 outer folds and 10,000 system-bootstrap results under
  `results/selection/equal_system/`;
- BRSpeechMOS-excluded and leave-one-corpus-out sensitivity under
  `results/selection/sensitivity/`;
- the complete 2,000-row, 50-refit Shapley table and regenerated 40-cell
  summaries under `results/attribution/`;
- analysis and extraction scripts under `code/`;
- `MANIFEST.json` with SHA-256 hashes.

The authoritative macro selector results are Gate $-1.51\%$, Singleton
$-0.72\%$, and Shapley $-2.15\%$ relative to exhaustive inner search, all
under equal-system MSE. Excluding BRSpeechMOS changes the Shapley point
estimate to $+0.63\%$; the Holm-adjusted comparison remains non-significant.

Run `python code/analysis/validate_public_release.py` after extraction to
verify the cached results. Raw audio, expert caches, and third-party weights
are not redistributed. Because hub revisions and weight-file hashes were not
recorded at extraction time, the release does not claim bit-identical feature
re-extraction; the extraction driver is included for users with their own
licensed data and checkpoints.
