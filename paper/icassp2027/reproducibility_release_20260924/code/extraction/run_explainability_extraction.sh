#!/usr/bin/env bash
set -uo pipefail

PROJECT="${MOS_PROJECT_ROOT:-/path/to/Embeddings}"
PYTHON="$PROJECT/venv_embs/bin/python"
EXTRACTOR="$PROJECT/extract_explainability_embeddings.py"
LOG_DIR="$PROJECT/logs/explainability_extraction"
MODE="${1:-gpu}"

# Work around CUDNN_STATUS_NOT_INITIALIZED on the current alphaai runtime.
export MOS_DISABLE_CUDNN="${MOS_DISABLE_CUDNN:-1}"

mkdir -p "$LOG_DIR"
cd "$PROJECT" || exit 1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

if [[ "$MODE" == "cpu" ]]; then
  LOG="$LOG_DIR/cpu_$(date '+%Y%m%d_%H%M%S').log"
  echo "$(timestamp) starting CPU features" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="" "$PYTHON" -u "$EXTRACTOR" \
    --device cpu \
    --datasets brspeech bvcc singmos tmhintqi \
    --splits train val test \
    --features egemaps auditory_erb \
    --deep-audit-after >>"$LOG" 2>&1
  status=$?
  echo "$(timestamp) CPU queue status=$status" | tee -a "$LOG"
  exit "$status"
fi

if [[ "$MODE" != "gpu" ]]; then
  echo "Usage: $0 [cpu|gpu]" >&2
  exit 2
fi

LOG="$LOG_DIR/gpu_$(date '+%Y%m%d_%H%M%S').log"
FEATURES=(rmvpe_cont rmvpe_quant speaker contentvec12 wavlm hubert wav2vec2 ced beats whisper)

gpu_snapshot() {
  nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1
}

wait_for_gpu() {
  local feature="$1"
  local required_free=8000
  if [[ "$feature" == "whisper" ]]; then
    required_free=12000
  fi
  while true; do
    local snapshot free util
    snapshot="$(gpu_snapshot 2>/dev/null || echo '0, 100')"
    free="$(echo "$snapshot" | cut -d, -f1 | tr -d ' ')"
    util="$(echo "$snapshot" | cut -d, -f2 | tr -d ' ')"
    if [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] && (( free >= required_free && util <= 70 )); then
      echo "$(timestamp) GPU ready feature=$feature free_mib=$free util=$util" | tee -a "$LOG"
      return 0
    fi
    echo "$(timestamp) waiting GPU feature=$feature free_mib=$free util=$util" | tee -a "$LOG"
    sleep 180
  done
}

echo "$(timestamp) starting GPU queue" | tee -a "$LOG"
for feature in "${FEATURES[@]}"; do
  success=0
  for attempt in 1 2 3; do
    wait_for_gpu "$feature"
    echo "$(timestamp) feature=$feature attempt=$attempt" | tee -a "$LOG"
    "$PYTHON" -u "$EXTRACTOR" \
      --device cuda \
      --datasets brspeech bvcc singmos tmhintqi \
      --splits train val test \
      --features "$feature" \
      --deep-audit-after >>"$LOG" 2>&1
    status=$?
    if (( status == 0 )); then
      success=1
      echo "$(timestamp) feature=$feature complete" | tee -a "$LOG"
      break
    fi
    echo "$(timestamp) feature=$feature failed status=$status" | tee -a "$LOG"
    sleep 180
  done
  if (( success == 0 )); then
    echo "$(timestamp) feature=$feature exhausted retries; continuing" | tee -a "$LOG"
  fi
done

echo "$(timestamp) final shallow audit" | tee -a "$LOG"
"$PYTHON" -u "$EXTRACTOR" --audit-only \
  --datasets brspeech bvcc singmos tmhintqi \
  --splits train val test \
  --features whisper contentvec12 wavlm beats auditory_erb speaker rmvpe_cont rmvpe_quant ced egemaps hubert wav2vec2 \
  >>"$LOG" 2>&1
status=$?
echo "$(timestamp) GPU queue finished audit_status=$status" | tee -a "$LOG"
exit "$status"
