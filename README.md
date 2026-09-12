# MOS Fusion Weight Audit

Reproducibility artifact for the ICASSP 2027 submission **“Do MOS Fusion
Weights Select Useful Embeddings? A System-Disjoint Attribution Audit.”**

The paper studies a simple but consequential distinction in multi-embedding MOS
prediction: a gate weight is an allocation in one fitted ensemble, Shapley is
contextual utility across coalitions, and subset selection is a separate
decision problem. The repository contains the public aggregate results and the
code snapshot used for the submission. It deliberately contains no speech,
listener-level ratings, cached embeddings, credentials, or private paths.

## Headline results

- Ten frozen representation families, four MOS corpora, and all 1,023 non-empty
  coalitions were audited.
- Sixteen representation/corpus cells have zero full-gate mass and positive
  Shapley utility; the existence of the mismatch survives player regrouping,
  although its count is player-definition dependent.
- In nested system-disjoint selection, singleton ranking was within 0.14% of
  exhaustive inner search (95% CI `[-0.88%, 1.02%]`) using 10 instead of 1,023
  coalition evaluations.
- Gate ranking was 0.62% worse than exhaustive inner search before multiplicity
  correction, but the contrast was not confirmatory after Holm correction
  (`p_Holm = 0.208`).

These results do **not** establish equivalence, a universal encoder ranking, or
a causal interpretation of speech representations.

## Repository map

| Location | Purpose |
|---|---|
| `configs/datasets.yaml` | Corpus roles, grouped-split contract, and redistribution status |
| `configs/representations.yaml` | Ten player definitions and feature-cache contract |
| `configs/experiments.yaml` | Frozen primary, uncertainty, Rashomon, null, and nested protocols |
| `src/experts.py` | Fold-local winsorization/normalization and grouped ridge OOF predictions |
| `src/fusion.py`, `src/coalitions.py` | Deterministic simplex gates and exhaustive coalition evaluation |
| `src/shapley.py`, `src/interactions.py` | Exact attribution, grouped players, and pairwise interactions |
| `src/rashomon.py`, `src/bootstrap.py` | Near-optimal weight ranges and system bootstrap helpers |
| `src/nested_selection.py` | Budgeted ranking and inner-validation subset-choice primitives |
| `src/run_*.py` | Archived full-run drivers for licensed feature caches |
| `scripts/verify_release.py` | Integrity, schema, and headline-number checks |
| `scripts/build_public_tables.py` | Rebuilds the public Markdown/LaTeX tables |
| `scripts/build_supplement.py` | Rebuilds the complete claim-organized supplement PDF |
| `scripts/smoke_pipeline.py` | Synthetic expert→OOF→gate→coalition→attribution smoke test |
| `results/` | Aggregate coalition, attribution, bootstrap, and selection outputs |
| `supplement/` | Human-readable supplement and generated tables |
| `metadata/` | Seeds, checkpoints, dataset provenance, and file hashes |
| `paper/` | Submission source snapshot; the compiled submission is not redistributed here |

## Paper-to-artifact mapping

| Paper statement | Public evidence | Reproduction command |
|---|---|---|
| Complete 40-cell attribution table | `results/all_40_cells.csv`, `results/attribution_40cells.csv` | `python scripts/verify_release.py` |
| 16 zero-gate/positive-Shapley cells with intervals/refit frequency | `results/discordant_cases.csv`, `results/shapley_values.csv` | `python scripts/verify_release.py` |
| Player granularity changes 16 to 7–13 | `results/player_partition_summary.csv` | `python scripts/build_public_tables.py` |
| All 120 Rashomon weight intervals and KKT audit | `results/rashomon_intervals.csv`, `results/rashomon_kkt_audit.csv` | `python scripts/verify_release.py` |
| 50 system refits for every corpus/expert cell | `results/training_refits.csv`, `results/training_gate_refit_summary.csv` | `python scripts/verify_release.py` |
| Fixed-coalition system contrasts | `results/fixed_coalition_contrasts.csv` | `python scripts/build_public_tables.py` |
| Complete common-alpha coalition lattice | `results/coalition_values.csv` | `python scripts/verify_release.py` |
| Pairwise interactions and correlation uncertainty | `results/interaction_values.csv`, `results/interaction_bootstrap.csv` | `python scripts/verify_release.py` |
| Nested selection table | `results/nested_selection_summary.csv` | `python scripts/build_public_tables.py` |
| Nested bootstrap intervals and Holm tests | `results/nested_selection_bootstrap.csv` | `python scripts/verify_release.py` |
| All outer-fold choices and corpus×budget metrics | `results/nested_selections.csv`, `results/nested_selection_metrics.csv` | `python scripts/verify_release.py` |
| Four null/control families | `results/null_experiments.csv` | `python scripts/verify_release.py` |

## Quick start: reproduce public tables and supplement

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/build_supplement.py
```

The two commands above require only committed files. Verification checks file
hashes, schemas, cross-file keys, 40 cells, every 1,023-mask lattice, all 50
refits per cell, Rashomon convergence, nested folds, null families, regenerated
public tables, and a synthetic end-to-end run of the modular code. The PDF build
regenerates the complete supplement. PDF bytes can differ across platforms due
to font/backend metadata; the canonical numeric CSVs and generated text tables
are the reproducibility targets.

For a release-maintainer rebuild after intentional edits:

```bash
python scripts/build_public_tables.py
python scripts/build_supplement.py
git add -A
python scripts/build_manifest.py
python scripts/verify_release.py
```

`build_manifest.py` hashes the canonical staged Git blobs, so intentional edits
must be staged before the manifest is rebuilt. The manifest files themselves
are excluded to avoid a recursive hash.

## Full recomputation from embeddings

The public modules cover the pipeline from already extracted representations to
the published quantities: fold-local preprocessing, ridge experts, grouped OOF
predictions, simplex gates, 1,023 coalitions, exact Shapley/interactions,
Rashomon ranges, grouped bootstrap, nested system-disjoint selection, and table
generation. The full experiment cannot run from this repository alone because
upstream speech and cached embeddings are not redistributable. After obtaining
each dataset under its original terms and generating the ten feature caches in
the contract documented by `configs/representations.yaml`, run:

```bash
python src/run_system_disjoint_selection.py \
  --root /path/to/private/experiment-root \
  --cache results/explainability_ridge_full/cache \
  --primary results/paper_submission_final_cpu/primary_oof \
  --output results/paper_system_disjoint_selection \
  --bootstrap 10000 --seed 20260912
```

The archived driver is fold-resumable and writes progress, diagnostics, errors,
fold choices, item predictions, and aggregate bootstrap results. The companion
drivers preserve the loaders used in the run; the small modules named in the
repository map expose the scientifically central operations for inspection and
testing without private data.

## Environment, seeds, and checkpoints

The reference analysis environment used Python 3.10.12, NumPy 2.2.6,
pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, and PyTorch 2.5.1+cu118.
Exact seeds and checkpoint/layer identifiers are in
`metadata/seeds.json` and `metadata/checkpoints.md`. Dataset split membership
and system identifiers were frozen before fitting; no system crosses grouped
folds. The nested benchmark holds complete systems out of every preprocessing,
expert-fitting, scoring, subset-selection, and gate-calibration step.

## Data provenance and redistribution

See `metadata/datasets.md` and `metadata/dataset_splits.md`. Users must obtain BRSpeechMOS, BVCC, SingMOS, and
TMHINT-QI from their original distributors and comply with the terms of every
upstream component. In particular, the BVCC release notes that Blizzard
Challenge samples may not be redistributed. This repository therefore contains
no waveform, rating-level record, listener metadata, or cached embedding.

## Compute cost

- Representation extraction: GPU recommended; the submission used sequential
  extraction on a 16 GB NVIDIA GPU to control memory. Runtime depends strongly
  on storage and encoder versions and was not preserved as a portable benchmark.
- Coalition and nested selection from cached embeddings: CPU-only. The archived
  nested run completed 20 outer folds plus 10,000 bootstrap replicates in about
  3.6 minutes on the submission server; hardware details and raw extraction time
  were not recorded precisely enough for a cross-machine claim.
- Public verification/table/PDF rebuild: normally under one minute on a laptop.

## Versioning and citation

The immutable submission snapshot is intended to be tagged
`v1.0-icassp2027-submission`. Verify the exact checkout with:

```bash
git describe --tags --always --dirty
git rev-parse HEAD
python scripts/verify_release.py
```

The GitHub Release and Zenodo DOI are deliberately not final until a clean-clone
installation and verification pass. Once Zenodo has archived the final Release,
add the real DOI and release date to `CITATION.cff` and the manuscript. Until
then, do not cite a placeholder DOI or describe the branch as immutable.

## Exclusions and security boundary

Committed content must not include audio, listener-level ratings, access
tokens, passwords, machine-specific absolute paths, third-party checkpoints,
large embedding caches, temporary outputs, or runtime caches. The manifest
hashes tracked release files only. Dataset and model licenses remain upstream.

## License

Repository code is released under the MIT License. Dataset licenses and terms
are independent and are not changed or granted by this repository.
