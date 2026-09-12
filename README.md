# MOS Fusion Audit

Compact public artifact for the ICASSP 2027 paper **“Do MOS Fusion Weights
Select Useful Embeddings? A System-Disjoint Attribution Audit.”** It contains
only the frozen configurations, checkpoint identifiers, numerical results, and
the claim-organized supplementary PDF cited by the paper. Audio, listener-level
ratings, cached embeddings, credentials, and third-party checkpoint binaries
are not redistributed.

Authors: Daniel Casanova, Alef Iury Ferreira, Lucas Gris, Pedro Lustosa Rege
Botelho, Fernanda Silva, Frederico Oliveira, Arlindo Galvão Filho, and Anderson
da Silva Soares. Affiliations are AKCIT, Brazil; Federal University of Goiás
(UFG), Brazil; and Federal University of Technology – Paraná (UTFPR), Brazil.

## Contents

```text
mos-fusion-audit/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml
├── supplementary_material.pdf
├── configs/
│   ├── experiments.yaml
│   ├── representations.yaml
│   └── seeds.json
├── results/
│   ├── all_40_cells.csv
│   ├── coalition_values.csv
│   ├── shapley_values.csv
│   ├── interaction_values.csv
│   ├── rashomon_intervals.csv
│   ├── bootstrap_results.csv
│   ├── nested_selection_metrics.csv
│   └── null_experiments.csv
├── checkpoints/
│   ├── README.md
│   └── checksums.sha256
└── tables/
    ├── table_1.csv
    ├── table_2.csv
    └── supplementary_tables/
        └── file_manifest.csv
```

The two numbered tables reproduce the paper's main tabular evidence. The
supplementary-table directory contains the additional CSV tables explicitly
described in the paper (fixed-coalition contrasts, player partitions, complete
Rashomon audit, nested-selection bootstrap, and QAP inference).

## Results and protocol

- `all_40_cells.csv` contains the 4-corpus × 10-representation attribution
  lattice, including gate, singleton, leave-one-out, exact Shapley, test
  intervals, and the final 50 training-system refit frequencies.
- `coalition_values.csv` contains the 1,023 non-empty coalition losses for
  every corpus under the common-alpha protocol.
- `shapley_values.csv`, `interaction_values.csv`, and
  `rashomon_intervals.csv` contain the paper's attribution, pairwise
  interaction, and near-optimal-simplex analyses.
- `bootstrap_results.csv` is a labelled union of the attribution, interaction,
  and nested-selection bootstrap summaries; the `analysis` column identifies
  the source experiment.
- `nested_selection_metrics.csv` reports every corpus × method × budget metric
  used by the system-disjoint selection benchmark.
- `null_experiments.csv` records the four control/null families. The geometric
  control uses 1,000 draws; the QAP inference in `tables/supplementary_tables`
  uses 100,000 node-label permutations. They are different experiments.

All CSVs are derived from the final fold-wise winsorized eGeMAPS protocol,
common ridge alpha `10.0`, grouped system splits, nonnegative simplex gates,
and 10,000 system-bootstrap replicates where indicated. The supplement gives
column definitions, confidence-interval scopes, and the distinction between
100 full training-system gate refits and 50 complete-game Shapley refits.

## Data and checkpoints

Datasets and pretrained models must be obtained from their original sources
under their own licenses:

- **BRSpeechMOS:** use the original authors' acquisition/preparation repository,
  <https://github.com/freds0/BSpeech-MOS-Prediction>, and follow its current terms.
- **BVCC:** use the VoiceMOS Challenge 2022 record,
  <https://doi.org/10.5281/zenodo.6572572>. Blizzard Challenge audio is not
  redistributable; follow the upstream acquisition instructions.
- **SingMOS:** use the VoiceMOS Challenge resource page,
  <https://sites.google.com/view/voicemos-challenge/resources>, and the dataset
  description at <https://arxiv.org/abs/2406.10911>.
- **TMHINT-QI:** use the original corpus description at
  <https://arxiv.org/abs/2111.02585> and the VoiceMOS resource page above.

The current upstream terms control in every case; this artifact grants no data
license. `configs/representations.yaml` and
`checkpoints/README.md` specify the model/layer identifiers and feature
definitions used for extraction. No checkpoint weight file is included; the
checksum file authenticates the canonical identifier lines rather than model
weights. Bit-identical re-extraction from mutable upstream registries is not
claimed.

`tables/supplementary_tables/file_manifest.csv` lists path, byte count, and
SHA-256 for every release file except the manifest itself. It is regenerated
only after the release contents are frozen.

## Environment

The `environment.yml` file records the lightweight Python environment used to
inspect the CSVs and PDF. This compact artifact is a frozen results package;
the private extraction/fitting drivers and licensed feature caches are not
part of this public snapshot.

## Citation and release

Please cite the paper and archived release at
<https://doi.org/10.5281/zenodo.22729723> (published 2026-09-12). The archived
tag is `v1.0-icassp2027-submission`.
