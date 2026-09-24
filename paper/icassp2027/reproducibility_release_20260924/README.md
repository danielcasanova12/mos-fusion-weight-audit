# ICASSP 2027 MOS fusion audit — corrected reproducibility release

This release is aligned with the submitted manuscript
`manuscript/submitted_manuscript.pdf`. It supersedes the earlier artifact
that contained an utterance-weighted preliminary selector run. The earlier
archive is intentionally not nested here, so that two incompatible result
versions cannot be mistaken for one another.

## What was corrected

The authoritative nested selector benchmark uses equal-system MSE both for
inner ranking and for the outer held-system evaluation. The final macro
contrasts relative to exhaustive inner search are:

| selector | relative MSE change | 95% system-bootstrap CI | Holm-adjusted $p$ |
|---|---:|---:|---:|
| Gate | $-1.51\%$ | $[-2.82,+0.74]\%$ | .694 |
| Singleton | $-0.72\%$ | $[-2.91,+1.44]\%$ | 1.000 |
| Shapley | $-2.15\%$ | $[-3.77,+0.71]\%$ | .694 |

The BRSpeechMOS-excluded Shapley--exhaustive point estimate is $+0.63\%$
(95% CI $[-0.02,+1.30]\%$; Holm-adjusted $p=.172$). These values are in
`results/selection/equal_system/macro_method_bootstrap.csv` and
`results/selection/sensitivity/macro_sensitivity.csv`.

The 40-cell attribution table now uses all 50 complete training-system
refits per corpus. The individual values (2,000 rows) are in
`results/attribution/train_shapley_50reps.csv`; the 40-cell summaries and
LaTeX table are regenerated from that file. Sixteen cells have a near-zero
full-gate weight and positive point-estimate Shapley utility; eight of those
sixteen are positive in all 50 refits. These are descriptive stability
summaries, not joint confidence intervals.

## Contents

* `manuscript/` — the submitted PDF, an editable companion source, the
  bibliography/template files, and the two figures used by that source.
* `configs/` — representation definitions, seeds, environment, and the final
  equal-system protocol (including SLSQP settings and the explicit limitation
  on checkpoint revisions/hashes).
* `results/selection/equal_system/` — all 20 outer folds, inner selections,
  equal-system errors, solver audits, protocol, log, and 10,000-replicate
  paired/macro bootstrap results.
* `results/selection/sensitivity/` — BRSpeechMOS-excluded and
  leave-one-corpus-out reaggregations.
* `results/attribution/` — the 40-cell lattice, 50-refit Shapley table,
  discovery/confirmation outputs, uncertainty intervals, and case lists.
* `results/controls/` and `results/interactions/` — alternative-fusion,
  null, player-partition, Rashomon, and interaction diagnostics.
* `code/analysis/` — the final selector runner, audit scripts, and the
  self-contained `validate_public_release.py` check.
* `code/extraction/` — the resumable feature-extraction driver and its usage
  notes. It requires the licensed corpora and locally obtained checkpoints.
* `MANIFEST.json` — SHA-256 and byte counts for every release file except the
  manifest itself.

Run the release-level check with:

```text
python code/analysis/validate_public_release.py
```

It validates the final selector values, the BRSpeechMOS-excluded result, the
40-cell/50-refit correspondence, the 16 discordant cells, and SHA-256 values.
It does not re-extract audio or refit neural encoders.

## Reproducibility scope and limitations

Raw audio, listener ratings, cached expert predictions, and third-party model
weights are not redistributed. The public configuration records the model
families, layers, pooling, preprocessing, seeds, folds, and optimizer
tolerances. Checkpoint hub revisions and weight-file hashes were not recorded
at extraction time; consequently this release does **not** claim bit-identical
independent feature re-extraction. The analysis scripts can be rerun when the
same licensed data and cached expert/OOF arrays are supplied.

`results/attribution/train_shapley_50reps.csv` is the complete individual
50-refit table; no hidden refit values are needed to reproduce its summary.
The submitted PDF is the reference for reported manuscript numbers. The
editable TeX file is supplied as a companion source and may differ in layout
or wording from the exact private source used to generate the uploaded PDF.

The old utterance-weighted archive is preserved outside this release for
provenance only. Do not combine its `nested_selection_bootstrap.csv` with the
equal-system files above.
