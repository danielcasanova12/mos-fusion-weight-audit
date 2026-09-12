# Checkpoints and feature definitions

| Public name | Frozen model / definition |
|---|---|
| Whisper | `openai/whisper-large-v3`, last hidden layer, mean pooled |
| ContentVec | `lengyue233/content-vec-best`, layer 12, mean pooled |
| WavLM | `microsoft/wavlm-base-plus`, layer 12, mean pooled |
| BEATs | `BEATs_iter3_plus_AS2M.pt` from `Bencr/beats-checkpoints`, final hidden state, mean pooled |
| Auditory ERB | 64 bands, 512-point FFT, 160-sample hop, 50–8,000 Hz |
| Speaker | `speechbrain/spkrec-ecapa-voxceleb`, 192-D L2-normalized embedding |
| RMVPE continuous | Voiced-F0 mean, standard deviation, minimum, maximum, and voiced fraction |
| RMVPE quantized | Normalized histogram: bin 0 unvoiced; 255 mel-uniform bins over 50–1,100 Hz |
| CED | `mispeech/ced-small`, final encoder state, mean pooled |
| eGeMAPS | eGeMAPSv02 functionals with fold-wise 0.1/99.9 percentile winsorization |

Exact weight-file hashes were not recorded at extraction time. Consequently,
the checkpoint family/layer identifiers and extraction-code snapshot define the
publicly reportable pipeline; bit-identical re-extraction is not claimed. The
four eGeMAPS outlier items were independently re-extracted and matched the
cached vectors.

The identifiers above are download instructions, not redistributed weights.
Model owners retain their licenses and terms. The archived extraction manifest
records the same identifiers. To
reduce silent version drift, archive the resolved Hub revisions and local
weight hashes before any future exact replication; the present artifact claims
protocol reproducibility from the saved representations, not bit-identical
re-extraction from mutable third-party model registries.
