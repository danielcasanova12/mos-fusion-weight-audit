# Equal-system nested selection — corrected completed run

- CPU execution on the remote `venv_embs` environment; no GPU or encoder
  extraction was used.
- Four corpora × five outer GroupKFold splits × four inner grouped folds;
  20/20 outer folds completed.
- Every inner coalition gate was fitted with row weights `1/|I_g|`, exactly
  optimizing equal-system MSE. The same objective drove singleton, Shapley,
  exhaustive selection, and final gate calibration; outer-held systems were
  evaluated once with equal-system MSE.
- 10,000 paired system-bootstrap replicates; Holm correction over five
  prespecified contrasts.
- Audit: complete item coverage `true`, one outer fold per system `true`, max
  Shapley efficiency error `1.3e-15`, and no error file.

Macro normalized MSE differences relative to exhaustive inner search:

| selector | point | 95% bootstrap CI | Holm-adjusted p |
|---|---:|---:|---:|
| Gate | -1.51% | [-2.82%, +0.74%] | .694 |
| Singleton | -0.72% | [-2.91%, +1.44%] | 1.000 |
| Shapley | -2.15% | [-3.77%, +0.71%] | .694 |

The Shapley--singleton contrast is -1.43% (95% CI [-3.23%, -0.07%]) before
Holm adjustment and .148 after adjustment. These values replace the
preliminary run; the earlier non-weighted gate run is retained only as an
invalid protocol record.
