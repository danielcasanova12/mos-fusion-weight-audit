# Software, seeds, checkpoints, and manifests for the ICASSP 2027 MOS audit.
# Remote computes ran in the recorded `venv_embs` environment (CPU-only:
# CUDA_VISIBLE_DEVICES="" for every result used in the paper).
# Local recomputes (v_sys, validation, nulls, selection curves) used the
# versions below unless noted.
remote_venv:
  python: 3.10.12
  sklearn: 1.7.2
  numpy: 2.2.6
  pandas: 2.3.3
  scipy: 1.15.3
  torch: 2.5.1+cu118
analysis_seeds:
  primary_protocol: 20260910
  test_bootstrap_and_contrasts: 20260909
  vsys_and_bootstrap: 20260911
  matched_nulls: 20260912
  system_disjoint_selection_equal_system: 20260916
  selection_sensitivity_reaggregation: 20260917 + scenario_index
  solver_multistart: 20260915
  train_bootstrap_draws: stable_seed(dataset,9)+104729*replicate
  validation_discovery_test_confirmation: 20260916 + 1000*dataset_index + feature_index (zero-based); 10000 system resamples
  qap_permutations: 20270910 + sum(ord(c) for c in lowercase_dataset_id); 100000 node-label permutations
checkpoints_families:
  whisper: openai/whisper-large-v3, last hidden layer, 1280-D
  contentvec: lengyue233/content-vec-best, layer 12, 768-D
  wavlm: microsoft/wavlm-base-plus, layer 12, 768-D
  beats: Bencr/beats-checkpoints/BEATs_iter3_plus_AS2M.pt, final hidden layer, 768-D
  erb: 64 triangular ERB-spaced bands, power STFT n_fft=512 Hann win=400 hop=160, log1p, 50-8000 Hz, time mean
  speaker: speechbrain/spkrec-ecapa-voxceleb, 192-D, L2-normalized
  rmvpe_cont: RVC-Boss/RMVPE weights/rmvpe.pt, hop=160 at 16 kHz, unvoiced threshold=0.03; 5-D voiced-F0 mean/population-std/min/max/voiced-fraction
  rmvpe_quant: 256-D normalized histogram: bin 0 unvoiced, 255 uniform-mel bins 50-1100 Hz
  ced: mispeech/ced-small, final encoder state, 384-D
  egemaps: eGeMAPSv02 openSMILE functionals, 88-D; 0.1/99.9-percentile fold-wise winsorization; package version not recorded
audio: torchaudio.load; downmix channels by mean; torchaudio.functional.resample to mono 16 kHz; temporal encoder outputs mean-pooled; Whisper uses crop_padding=True
ridge_probe: alpha=10; unpenalized intercept; scikit-learn LSQR solver; tol=1e-10; max_iter=10000; fold-wise standardization; std<1e-6 uses scale 1
nested_system_disjoint_selection:
  outer: GroupKFold(5) by training system_id
  inner: GroupKFold(4) within outer-development systems
  folds: deterministic, unshuffled, one assignment; no repeated group splits
  objective: equal-system MSE for inner selection, gate calibration, and outer evaluation
  bootstrap: 10000 paired system resamples, seed 20260916
simplex_gate:
  optimizer: SLSQP, analytic gradient, uniform initialization, bounds [0,1], sum(weights)=1
  tolerance_and_limit: ftol=1e-12, maxiter=1000
  failure_fallback: deterministic projected gradient, 4000 iterations, step=0.2/max(||P||_2^2,1), simplex projection and renormalization
  convergence_metadata: per-coalition SLSQP success/fallback flags were not retained
  full_gate_multistart: 101 starts per corpus (uniform + 100 Dirichlet(1,...,1)); seed=20260915; all succeeded
notes:
  - Exact weight-file hashes were not recorded at extraction; family/layer
    identifiers above plus the extraction-code snapshot define the pipeline.
    Hub revisions and torchaudio/openSMILE package versions/configuration hash
    were not archived, so bit-identical independent feature re-extraction is
    not claimed.
    eGeMAPS vectors were independently re-extracted for four outlier items
    and matched cached vectors (egemaps_reextraction_audit.csv).
  - Train/test/val metadata (audio ids, system groups, split membership) are
    fixed manifests; test system overlap: 6/6 BRSpeechMOS, 175/187 BVCC,
    31/35 SingMOS, 97/98 TMHINT-QI.
