# Attribution evidence audit

Run `python paper/icassp2027/audit_attribution_evidence.py` with numpy and pandas.

The 40-cell table uses only common ridge alpha=10, grouped OOF predictions, and fold-wise 0.1% winsorization of both eGeMAPS tails. All 4,092 coalition losses, 40 gates, and 40 Shapley point estimates agree with the explicitly labelled alpha10 game. Shapley and LOO were independently recomputed from the final lattice. The same-protocol test interval point estimates agree within 1e-12. Selected-alpha and older preprocessing results are excluded.

Training gate probabilities use 100 system-bootstrap refits per corpus. Training Shapley summaries use all 50 full-system refits and complete coalition games per corpus, conditional on the fixed observed test set. The percentile ranges are descriptive stability ranges, not precise 95% confidence intervals. Positive frequency means the observed fraction of those 50 replicates with Shapley > 0; its resolution is 1/50. It is not a posterior probability or a joint training/test confidence statement.

The separate test intervals use 10,000 test-system resamples with the experts and coalition gates fixed; system clusters are sampled, then utterance-weighted means are formed. The separate frozen-expert gate probabilities use 200 training-system resamples. None of these uncertainty sources is combined.

Among the 16 point zero-gate/positive-Shapley cells, 10 have training percentile lower bounds above zero and 11 have conditional test interval lower bounds above zero. These are descriptive, unadjusted per-cell summaries. Positive point utility is not uniformly stable under retraining: all frequencies are exposed in the table.

## The 18-case wording correction

The producer calls LOO <= 1e-6 and Shapley > 0 effectively zero LOO. This is a signed threshold, so it also accepts substantially negative utility. Use 'nonpositive or numerically negligible LOO utility' for the 18-case set. Absolute LOO <= 1e-6 yields 16 cases with positive Shapley, exactly the 16 zero-gate/positive-Shapley cases. There are 17 zero gates in total; BRSpeechMOS RMVPE-quant has negative Shapley and is excluded from the positive-utility set.

The two additional signed-LOO cases are BVCC Whisper (LOO -0.002392055566558887, gate 0.24948356301409777) and TMHINT-QI Speaker (LOO -1.1852257730815552e-6, gate 0.01175879950483767). Their gates are active; removal improves test MSE.

## Threshold convention

Gate zero means w < t for positive t, and exact w == 0 at t=0. Both signed LOO <= t and absolute LOO <= t are exported. Every positive-Shapley count uses phi > 0. MSE thresholds are not gate-weight units. Counts are provided overall and by corpus at 0, 1e-8, 1e-6, 1e-4, and 1e-3.

## loo_signed_18_cases (18)

- brspeech/beats
- brspeech/auditory_erb
- brspeech/speaker
- brspeech/rmvpe_cont
- brspeech/ced
- bvcc/whisper
- bvcc/auditory_erb
- bvcc/rmvpe_cont
- bvcc/rmvpe_quant
- bvcc/egemaps
- singmos/auditory_erb
- singmos/rmvpe_cont
- singmos/rmvpe_quant
- tmhintqi/auditory_erb
- tmhintqi/speaker
- tmhintqi/rmvpe_cont
- tmhintqi/rmvpe_quant
- tmhintqi/egemaps

## zero_gate_positive_shapley_cases (16)

- brspeech/beats
- brspeech/auditory_erb
- brspeech/speaker
- brspeech/rmvpe_cont
- brspeech/ced
- bvcc/auditory_erb
- bvcc/rmvpe_cont
- bvcc/rmvpe_quant
- bvcc/egemaps
- singmos/auditory_erb
- singmos/rmvpe_cont
- singmos/rmvpe_quant
- tmhintqi/auditory_erb
- tmhintqi/rmvpe_cont
- tmhintqi/rmvpe_quant
- tmhintqi/egemaps

## loo_absolute_zero_positive_shapley_cases (16)

- brspeech/beats
- brspeech/auditory_erb
- brspeech/speaker
- brspeech/rmvpe_cont
- brspeech/ced
- bvcc/auditory_erb
- bvcc/rmvpe_cont
- bvcc/rmvpe_quant
- bvcc/egemaps
- singmos/auditory_erb
- singmos/rmvpe_cont
- singmos/rmvpe_quant
- tmhintqi/auditory_erb
- tmhintqi/rmvpe_cont
- tmhintqi/rmvpe_quant
- tmhintqi/egemaps

## Files

`attribution_40cells_alpha10.csv` contains all numeric summaries, replicate counts, and scope labels. `attribution_supplement_alpha10.tex` contains two longtables (requires booktabs and longtable). `attribution_supplement_standalone.tex` supplies an A4 landscape wrapper; run a LaTeX compiler on it from this evidence directory. Do not insert a longtable directly in the two-column paper body. `attribution_threshold_sweep.csv` and the case-membership/list CSVs make each threshold claim inspectable. `attribution_provenance_audit.json` records source line references, input SHA-256 hashes, and numerical identity checks. NA is reserved for missing or unverified uncertainty evidence.
