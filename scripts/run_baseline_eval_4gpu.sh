#!/bin/bash
# =============================================================================
# Baseline eval: official declare-lab/delta-mem_qwen3_4b-instruct adapter
# Tasks: locomo + hotpotqa
# GPUs: 4 (gpu-server-office: 4x H100)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

# Paths
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/public/liuyr/AGM/huggingface/hub/Qwen3-4B-Instruct-2507}"
OFFICIAL_ADAPTER_DIR="${OFFICIAL_ADAPTER_DIR:-/public/liuyr/AGM/huggingface/hub/delta-mem_qwen3_4b-instruct}"
LOCOMO_DATA_FILE="${LOCOMO_DATA_FILE:-${ROOT_DIR}/data/locomo10.json}"
HF_HOME="${HF_HOME:-/public/liuyr/AGM/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-/public/liuyr/AGM/huggingface/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/.cache/huggingface/datasets}"
SUITE_ROOT="${SUITE_ROOT:-/public/liuyr/AGM/outputs/baseline_official_adapter}"
LOG_ROOT="${SUITE_ROOT}/logs"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NPROC="${NPROC:-4}"
BASE_INFERENCE_BACKEND="${BASE_INFERENCE_BACKEND:-transformers}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

mkdir -p "${SUITE_ROOT}" "${LOG_ROOT}"

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname -- "${log_file}")"
  print_cmd "$@"
  "$@" > >(tee "${log_file}") 2> >(tee -a "${log_file}" >&2)
}

is_complete() {
  local output_json="$1"
  [[ -f "${output_json}" ]]
}

# ─── LoCoMo: base_model (no adapter) ──────────────────────────────────────────
run_locomo_base() {
  local output_json="${SUITE_ROOT}/base_model/locomo.json"
  local log_file="${LOG_ROOT}/base_model_locomo.log"
  is_complete "${output_json}" && { echo "Skip base_model locomo"; return 0; }
  mkdir -p "$(dirname "${output_json}")"
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC}" \
    --master_addr 127.0.0.1 \
    --master_port 29671 \
    -m deltamem.eval.locomo_delta \
    --model-path "${BASE_MODEL_PATH}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-tokens 50 \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --answer-reserve-tokens 50 \
    --full-history-mode official_prompt \
    --categories 1 2 3 4 \
    --output-json "${output_json}" \
    --data-file "${LOCOMO_DATA_FILE}"
}

# ─── LoCoMo: official delta-mem adapter ───────────────────────────────────────
run_locomo_official() {
  local output_json="${SUITE_ROOT}/official_adapter/locomo.json"
  local log_file="${LOG_ROOT}/official_adapter_locomo.log"
  is_complete "${output_json}" && { echo "Skip official_adapter locomo"; return 0; }
  mkdir -p "$(dirname "${output_json}")"
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC}" \
    --master_addr 127.0.0.1 \
    --master_port 29672 \
    -m deltamem.eval.locomo_delta \
    --model-path "${BASE_MODEL_PATH}" \
    --adapter-dir "${OFFICIAL_ADAPTER_DIR}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-tokens 50 \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --answer-reserve-tokens 50 \
    --skip-base \
    --delta-conditions full_history_replay \
    --full-history-mode official_prompt \
    --categories 1 2 3 4 \
    --output-json "${output_json}" \
    --data-file "${LOCOMO_DATA_FILE}"
}

# ─── HotpotQA: base_model (no adapter) ────────────────────────────────────────
run_hotpotqa_base() {
  local output_json="${SUITE_ROOT}/base_model/hotpotqa.json"
  local log_file="${LOG_ROOT}/base_model_hotpotqa.log"
  is_complete "${output_json}" && { echo "Skip base_model hotpotqa"; return 0; }
  mkdir -p "$(dirname "${output_json}")"
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC}" \
    --master_addr 127.0.0.1 \
    --master_port 29771 \
    -m deltamem.eval.benchmark_compare \
    --model-path "${BASE_MODEL_PATH}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --datasets-cache-dir "${HF_DATASETS_CACHE}" \
    --hub-cache-dir "${HF_HUB_CACHE}" \
    --tasks hotpotqa \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --base-inference-backend "${BASE_INFERENCE_BACKEND}" \
    --hotpotqa-max-new-tokens 32 \
    --hotpotqa-official-decoding \
    --eval-do-sample \
    --eval-temperature 0.4 \
    --eval-top-p 0.9 \
    --eval-top-k 10 \
    --local-files-only \
    --skip-delta \
    --skip-lora \
    --output-json "${output_json}"
}

# ─── HotpotQA: official delta-mem adapter ─────────────────────────────────────
run_hotpotqa_official() {
  local output_json="${SUITE_ROOT}/official_adapter/hotpotqa.json"
  local log_file="${LOG_ROOT}/official_adapter_hotpotqa.log"
  is_complete "${output_json}" && { echo "Skip official_adapter hotpotqa"; return 0; }
  mkdir -p "$(dirname "${output_json}")"
  run_with_log "${log_file}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC}" \
    --master_addr 127.0.0.1 \
    --master_port 29772 \
    -m deltamem.eval.benchmark_compare \
    --model-path "${BASE_MODEL_PATH}" \
    --delta-adapter-dir "${OFFICIAL_ADAPTER_DIR}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --datasets-cache-dir "${HF_DATASETS_CACHE}" \
    --hub-cache-dir "${HF_HUB_CACHE}" \
    --tasks hotpotqa \
    --seed 42 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --base-inference-backend "${BASE_INFERENCE_BACKEND}" \
    --hotpotqa-max-new-tokens 32 \
    --hotpotqa-official-decoding \
    --eval-do-sample \
    --eval-temperature 0.4 \
    --eval-top-p 0.9 \
    --eval-top-k 10 \
    --local-files-only \
    --skip-base \
    --skip-lora \
    --output-json "${output_json}"
}

echo "=== Baseline eval: official adapter on LoCoMo + HotpotQA ==="
echo "Base model : ${BASE_MODEL_PATH}"
echo "Adapter    : ${OFFICIAL_ADAPTER_DIR}"
echo "Output     : ${SUITE_ROOT}"
echo ""

run_locomo_base
run_locomo_official
run_hotpotqa_base
run_hotpotqa_official

echo ""
echo "=== Done: ${SUITE_ROOT} ==="
