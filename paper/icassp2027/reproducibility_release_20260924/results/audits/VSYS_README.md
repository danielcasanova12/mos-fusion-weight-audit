# Equal-system (v_sys) Shapley game + test-system bootstrap

Computed 2026-09-11 from frozen final-protocol artifacts without refitting experts or gates.

## Inputs (SHA-verified by reuse, not rehashed here)
- `primary_oof/*.npz` from `results/paper_submission_final_cpu/` (oof/y_train/y_val/y_test/val/test, 10 experts in FEATURES order)
- `converged_oof_lattice.csv` (1,023 weights_json per corpus, common alpha=10, fold-wise eGeMAPS winsor 0.1%)
- `embeddings/<ds>/test/metadata_explainability.csv` (system_id order matches npz rows; y_test vs meta mos max diff <2.2e-07)

## Method
For each mask S, `pred_S = test_experts[:,idx(S)] @ w_S`, `w_S` from lattice; empty = train mean.
Per-utterance squared errors `E[i,S]`. Utterance game `v=-mean_i E`; system game `v_sys=-G^-1 sum_g mean_{i in g} E`.
Exact Shapley (10 players, 1,024 values incl. empty) for both losses from same predictions.
Test bootstrap (10,000 system resamples, seed 20260911):
- utt: `(counts @ per-utterance-contribution-sums)/(counts @ sizes)` (utterance-weighted, matches `cmd_explain`)
- sys: `(counts @ per-system-contributions)/G` (equal-system)
Gates fixed, so discordant-count distributions isolate test variation.

Producer-side script path is intentionally omitted from the public release;
the saved CSVs above are the auditable outputs.

## Key results
- Rank `phi_utt` vs `phi_sys`: .927 BRSpeechMOS, .988 BVCC, .976 SingMOS, .988 TMHINT-QI.
- Sign agreement: 8/10 BRS, 10/10 BVCC, 9/10 SingMOS, 10/10 TMHINT-QI.
- Zero-gate (`w<1e-6`) cases: 6/4/3/4 per corpus (17 total). Positive under both games: 16 (only BRSpeechMOS RMVPE-cont/quant swap: utt RMVPE-cont>0/RMVPE-quant<0; sys reversed).
- Test 95% above zero among 17 zero gates: 11 utterance-weighted, 12 equal-system.
- Discordant-count mean [95%]: BRS 5.3 [4,6] sys vs 4.9 [4,6] utt; BVCC 4.0 [4,4] vs 3.9 [3,4]; SingMOS 2.9 [2,3] vs 2.7 [2,3]; TMHINT-QI 3.7 [3,4] vs 3.8 [3,4].
- Recomputed `phi_utt` matches `oof_shapley_test_cluster_bootstrap.csv` within 1.7e-09; CIs match within Monte Carlo noise.

## Train/test system overlap (from fetched metadata)
- BRSpeechMOS train 6 / test 6 / overlap 6; BVCC 175/187/175; SingMOS 31/35/31; TMHINT-QI 98/97/97.
- Hence test is largely seen systems; only 12 BVCC + 4 SingMOS systems are unseen. Added to Methods.

## Files
- `vsys_shapley_point.csv`: gate, mse_utt_full, mse_sys_full, phi_utt, phi_sys, loo_utt, loo_sys per 40 cells.
- `vsys_bootstrap_intervals.csv`: phi + 95% for both weightings, gate, zero_gate.
- `vsys_discordant_counts.csv`: observed + bootstrap mean/2.5/97.5% + histograms per corpus.
