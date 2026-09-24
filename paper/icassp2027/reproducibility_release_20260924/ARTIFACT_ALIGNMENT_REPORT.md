# Artifact alignment report — 2026-09-24

## Reference used

The submitted reference is the PDF stored as
`manuscript/submitted_manuscript.pdf` in this release.
Its selector table reports Gate $-1.51\%$, Singleton $-0.72\%$, and Shapley
$-2.15\%$ relative to exhaustive inner search, with the equal-system
confidence intervals shown in the final release.

## Changes made

1. Created `supplementary_reproducibility_artifact_final_20260924.tgz`.
2. Removed the stale utterance-weighted selector directory from the public
   release; it remains on disk only as historical provenance.
3. Added the final equal-system selection outputs for all 20 outer folds,
   including protocol, logs, solver checks, per-fold selections, and 10,000
   paired system-bootstrap replicates.
4. Added BRSpeechMOS-excluded and leave-one-corpus-out sensitivity results.
5. Added the submitted PDF and an editable TeX companion.
6. Added the analysis drivers and the resumable extraction driver.
7. Recomputed the attribution summaries from all 50 refits (2,000 individual
   rows), replacing the stale 30-refit summaries.
8. Added `MANIFEST.json` and `validate_public_release.py`; the archive was
   extracted and the validator passed.

## Remaining limitation stated explicitly

The original extraction did not record model-hub revisions or checkpoint
weight-file hashes. The new release records the exact identifiers, layers,
pooling, preprocessing, seeds, folds, and solver settings, but it does not
claim bit-identical independent feature re-extraction. Raw audio, expert
caches, and third-party weights remain excluded for licensing and size.
