# Checkpoint identifiers

Checkpoint binaries are not redistributed. The lines below are the canonical
identifiers used by the extraction protocol; `checksums.sha256` contains their
SHA-256 digests as an auditable manifest. The digest is **not** a weight-file
hash and does not claim bit-identical re-extraction from a mutable model hub.
Hub revisions and checkpoint weight hashes were not recorded when the features
were extracted. Consequently, this release makes no bitwise re-extraction
claim; `configs/representations.yaml` records that limitation explicitly.

```text
whisper|openai/whisper-large-v3|last_hidden|mean_pool|1280D
contentvec12|lengyue233/content-vec-best|layer_12|mean_pool|768D
wavlm|microsoft/wavlm-base-plus|layer_12|mean_pool|768D
beats|Bencr/beats-checkpoints|BEATs_iter3_plus_AS2M.pt|final_hidden_mean_pool|768D
speaker|speechbrain/spkrec-ecapa-voxceleb|192D|L2_normalized
ced|mispeech/ced-small|final_encoder_state|mean_pool|384D
auditory_erb|64_bands|FFT_512|hop_160|50-8000Hz
rmvpe_cont|RMVPE|voiced_F0_summary|5_statistics
rmvpe_quant|RMVPE|256_bin_normalized_histogram|50-1100Hz
egemaps|eGeMAPSv02|functionals|88D|winsor_0.1_99.9_percentile_foldwise
```
