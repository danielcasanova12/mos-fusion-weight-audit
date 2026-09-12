# Dataset split contract

The public artifact does not contain utterance identifiers because those can
reveal restricted corpus membership.  The executable split contract is:

1. Load the corpus-provided utterance identifier, MOS, and synthesis/enhancement
   system identifier.
2. Sort systems and utterance identifiers lexicographically before passing them
   to the deterministic grouped splitter.
3. Fit winsorization, normalization, ridge experts, ranking statistics, subset
   size, and fusion gates using training/inner systems only.
4. Keep every utterance from a system in the same fold.  A system may never
   cross train/validation/test roles within a split.
5. The primary attribution lattice uses grouped OOF predictions for fitting the
   gate and frozen held-out predictions for value evaluation.  The nested
   benchmark repeats the whole procedure inside each outer system fold.

The exact public seeds are in `metadata/seeds.json`; split policies and fold
counts are in `configs/datasets.yaml` and `configs/experiments.yaml`.  A user
with licensed data can regenerate the private split manifests using
`src/run_system_disjoint_selection.py`.  We intentionally do not publish a
table mapping restricted utterance IDs to folds.
