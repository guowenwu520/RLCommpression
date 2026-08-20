#!/usr/bin/env bash
set -euo pipefail

# 与旧版本相同：默认复用原输出目录中的 manifest / Fresh Blueprint / Full Reference / Fixed25 cache。
# 只会失效旧 RL policy / Router state。
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NPROC:-2}}"

MODEL_PATH="${MODEL_PATH:-/data4/guowenwu/MMDITModelCompression/models/Qwen-Image-Edit-2511}"
DATASET_ROOT="${DATASET_ROOT:-/data4/guowenwu/MMDITModelCompression/dataset/images1024x1024}"
PROMPT_FILE="${PROMPT_FILE:-/data4/guowenwu/MMDITModelCompression/portrait_prompts.md}"
OUTPUT_DIR="${OUTPUT_DIR:-/data4/guowenwu/RLCompression/outputs/qwen_rainbow_router_sixway_v1}"
BLUEPRINT_CALIBRATION_COUNT="${BLUEPRINT_CALIBRATION_COUNT:-100}"
TRAIN_COUNT="${TRAIN_COUNT:-100}"
EVAL_COUNT="${EVAL_COUNT:-10}"
COMPUTE_RATIO="${COMPUTE_RATIO:-0.25}"

# Router 参数。日志默认只打印 epoch loss / deterministic validation convergence。
export ROUTER_VAL_COUNT="${ROUTER_VAL_COUNT:-8}"
export ROUTER_REPLAY_CAPACITY="${ROUTER_REPLAY_CAPACITY:-50000}"
export ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-256}"
export ROUTER_GRADIENT_STEPS="${ROUTER_GRADIENT_STEPS:-160}"
export ROUTER_WARMUP="${ROUTER_WARMUP:-1000}"
export ROUTER_N_STEP="${ROUTER_N_STEP:-3}"
export ROUTER_EPS_START="${ROUTER_EPS_START:-0.30}"
export ROUTER_EPS_MIN="${ROUTER_EPS_MIN:-0.05}"
export ROUTER_EPS_DECAY="${ROUTER_EPS_DECAY:-0.85}"
export ROUTER_MIN_EPOCHS="${ROUTER_MIN_EPOCHS:-3}"
export ROUTER_PATIENCE="${ROUTER_PATIENCE:-3}"
export ROUTER_MAX_EPOCHS="${ROUTER_MAX_EPOCHS:-20}"
export ROUTER_QUIET_INNER="${ROUTER_QUIET_INNER:-1}"

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/qwen_rainbow_router_sixway_v1.py"
  --mode "$MODE"
  --model-path "$MODEL_PATH"
  --dataset-root "$DATASET_ROOT"
  --prompt-file "$PROMPT_FILE"
  --output-dir "$OUTPUT_DIR"
  --blueprint-calibration-count "$BLUEPRINT_CALIBRATION_COUNT"
  --train-count "$TRAIN_COUNT"
  --eval-count "$EVAL_COUNT"
  --compute-ratio "$COMPUTE_RATIO"
  --profile-quantile "${PROFILE_QUANTILE:-0.90}"
  --target-cache-ratio "${TARGET_CACHE_RATIO:-0.70}"
  --blueprint-max-cache-age "${BLUEPRINT_MAX_CACHE_AGE:-5}"
  --max-token-cache-age "${MAX_TOKEN_CACHE_AGE:-5}"
  --token-execution-mode "${TOKEN_EXECUTION_MODE:-sparse}"
  --dtype "${DTYPE:-bf16}"
  --device "${DEVICE:-cuda}"
  --policy-device "${POLICY_DEVICE:-cuda}"
  --num-inference-steps "${NUM_INFERENCE_STEPS:-50}"
  --seed "${SEED:-20260814}"
)

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${CMD[@]:1}"
else
  exec "${CMD[@]}"
fi
