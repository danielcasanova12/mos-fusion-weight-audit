# Full-coalition fusion games + redundancy null (CPU-only, frozen experts)

Computed 2026-09-11/12 from frozen `primary_oof/*.npz` + train/test metadata. No high-dim expert refits.

## Fusion games (1,023 coalitions x 4 corpora x 4 rules)
- simplex: lattice `weights_json` (common alpha=10); uniform: mean of member experts;
  NNLS: `LinearRegression(positive=True, fit_intercept=True)` on grouped OOF per coalition;
  ridge: per-coalition GroupKFold CV over 17 alphas `logspace(-4,4)` + refit (same folds as paper).
- Empty coalition = train mean. Test MSE utterance-weighted; exact Shapley per rule.
- Full test MSE: BVCC .210/.283/.209/.210, SingMOS .336/.356/.333/.334,
  TMHINT-QI .334/.417/.334/.339, BRSpeechMOS .704/.722/.904/.730 (simplex/uniform/NNLS/ridge).
- Shapley rank vs simplex (uniform/NNLS/ridge): BVCC .99/1.00/1.00, TMHINT-QI .99/.99/.98,
  SingMOS .98/.78/.78, BRSpeechMOS .95/.67/.73 (exploratory; NNLS overfits 6-system BRS).
- Mismatch beyond simplex: NNLS zero-gate/positive 4/4 BVCC, 2/2 SingMOS, 2 of 3 TMHINT-QI
  (BRS NNLS 0/4, overfit); ridge bottom-3 weights positive 3/3 BVCC, 3/3 SingMOS, 2/3 TMHINT-QI;
  uniform hides spreads up to .19 MSE.

Script: `compute_fusion_null.py` (6.2 min CPU). Outputs: `fusion_shapley_all_rules.csv`,
`fusion_mismatch.csv`, `fusion_full_weights.json`.

## Redundancy null (1,000 target-shuffled reps per corpus)
- Preserves OOF expert correlations + frozen simplex coalition predictions; permutes `y_test` globally.
- Recomputes all coalition MSEs, exact pairwise `-I_ij`, Spearman(corr, -I).
- Observed .37/.94/.44/.96 vs null mean -.38/-.85/-.34/-.85, max -.20/-.80/-.22/-.81;
  observed beats all 1,000 nulls (`p=.001`). Geometry alone predicts opposite sign.
- Script: `fix_null_fast.py` (45x1024 weight-matrix matmul, CPU seconds). Output: `redundancy_null.csv`.

## Honesty notes
- BRSpeechMOS (6 systems) is exploratory throughout; NNLS failure there is reported, not hidden.
- Partial singleton controls: mean .49/.66/.08/.85, min .43/.10/.35/.80 — BVCC/SingMOS attenuated.
- Train/test system overlap 6/6, 175/187, 31/35, 97/98: test is largely seen systems.
