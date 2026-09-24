# Selection sensitivity reanalysis

CPU-only reanalysis of the completed equal-system nested run. No encoder, ridge expert, or gate was refit.

- Bootstrap replicates per scenario: 10000
- Protocol checks passed: True
- Primary parity max point difference: 1.318e-16

The primary parity row must match the published macro run; the other scenarios are new sensitivity analyses.

## Scenarios

- `exclude_brspeechmos`: all primary budgets on BVCC, SingMOS, and TMHINT-QI.
- `exclude_k1`: budgets 2, 3, and 5 on all four corpora.
- `exclude_brspeechmos_k_ge_2`: both restrictions together.
- `leave_out_*`: one leave-one-corpus-out analysis per corpus.
