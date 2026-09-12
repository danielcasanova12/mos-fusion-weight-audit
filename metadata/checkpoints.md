# Checkpoints and feature definitions

| Public name | Frozen model / definition |
|---|---|
| Whisper | Whisper-large-v3, last hidden layer |
| ContentVec | ContentVec, layer 12 |
| WavLM | WavLM-base-plus, layer 12 |
| BEATs | BEATs iter3+ |
| Auditory ERB | 64 bands, 512-point FFT, 160-sample hop, 50–8,000 Hz |
| Speaker | ECAPA-TDNN speaker encoder |
| RMVPE continuous | Voiced-F0 mean, standard deviation, minimum, maximum, and voiced fraction |
| RMVPE quantized | Normalized histogram: bin 0 unvoiced; 255 mel-uniform bins over 50–1,100 Hz |
| CED | CED-Small |
| eGeMAPS | eGeMAPSv02 functionals with fold-wise 0.1/99.9 percentile winsorization |

Exact weight-file hashes were not recorded at extraction time. Consequently,
the checkpoint family/layer identifiers and extraction-code snapshot define the
publicly reportable pipeline; bit-identical re-extraction is not claimed. The
four eGeMAPS outlier items were independently re-extracted and matched the
cached vectors.

