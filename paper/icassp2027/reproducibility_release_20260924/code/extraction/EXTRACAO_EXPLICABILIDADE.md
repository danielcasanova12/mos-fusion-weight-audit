# Extração de embeddings — estudo de explicabilidade MOS

## Escopo

Datasets: `brspeech`, `bvcc`, `singmos`, `tmhintqi`.

Famílias do power set: `whisper`, `contentvec12`, `wavlm`, `beats`,
`auditory_erb`, `speaker`, `rmvpe_cont`, `rmvpe_quant`, `ced`, `egemaps`.

Ablações de troca de encoder: `hubert`, `wav2vec2`.

ContentVec está fixado na camada 12. WavLM, HuBERT e wav2vec2 usam somente
a representação final; os antigos arquivos BVCC com 13 camadas são preservados.

## Comandos

Auditoria rasa do BVCC:

```bash
./venv_embs/bin/python extract_explainability_embeddings.py \
  --audit-only --datasets bvcc
```

Auditoria profunda, carregando e validando cada tensor:

```bash
CUDA_VISIBLE_DEVICES='' ./venv_embs/bin/python extract_explainability_embeddings.py \
  --audit-only --deep-audit --datasets bvcc
```

Fila CPU, para eGeMAPS e frontend ERB:

```bash
nohup ./run_explainability_extraction.sh cpu > logs/explainability_extraction/cpu_launcher.log 2>&1 &
```

Fila GPU retomável:

```bash
nohup ./run_explainability_extraction.sh gpu > logs/explainability_extraction/gpu_launcher.log 2>&1 &
```

A fila GPU aguarda memória/utilização segura. Ela não encerra processos de
outros usuários. Cada feature tem até três tentativas e arquivos válidos são
pulados.

Extração seletiva:

```bash
./venv_embs/bin/python extract_explainability_embeddings.py \
  --datasets singmos tmhintqi --splits train val test \
  --features contentvec12 rmvpe_cont rmvpe_quant --device cuda
```

## Saídas

Embeddings:

```text
embeddings/<dataset>/<split>/<feature>/<caminho-relativo>.pt
```

Metadata por split:

```text
embeddings/<dataset>/<split>/metadata_explainability.csv
```

Relatórios:

```text
reports/explainability_extraction/
```

Logs:

```text
logs/explainability_extraction/
```
