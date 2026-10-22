#!/usr/bin/env bash
set -euo pipefail

# Activate the model environment first, or set STARVLA_PYTHON.
# Required: CKPT=/path/to/checkpoints/steps_XXXX_pytorch_model.pt
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"

# Point CKPT at the checkpoint you want to serve, or pass it inline:
#   CKPT=/path/to/checkpoints/steps_XXXXXX_pytorch_model.pt bash run_policy_server.sh
: "${CKPT:?Set CKPT to the model checkpoint file}"
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}" >&2; exit 2; }
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"

STARVLA_NUM_INFERENCE_TIMESTEPS="${STARVLA_NUM_INFERENCE_TIMESTEPS:-}"
STARVLA_EVAL_MEM_LENGTH="${STARVLA_EVAL_MEM_LENGTH:-}"
STARVLA_EVAL_SEED="${STARVLA_EVAL_SEED:-}"
DEBUG="${DEBUG:-}"
LOG_DIR="${LOG_DIR-${STARVLA_DIR}/playground/serving_logs}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# 记忆库推理模式 streaming = 跨帧记忆生效。
# 设为 stateless = 每步清空记忆。server 加载模型时读取此变量。
export STARVLA_EVAL_MEMORY_MODE="${STARVLA_EVAL_MEMORY_MODE:-streaming}"
# 留空则不覆盖，完全交给 policy_wrapper.py 用 checkpoint 自带的默认值
[[ -n "${STARVLA_NUM_INFERENCE_TIMESTEPS}" ]] && export STARVLA_NUM_INFERENCE_TIMESTEPS
[[ -n "${STARVLA_EVAL_MEM_LENGTH}" ]] && export STARVLA_EVAL_MEM_LENGTH
[[ -n "${STARVLA_EVAL_SEED}" ]] && export STARVLA_EVAL_SEED
[[ -n "${DEBUG}" ]] && export DEBUG

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
  --idle_timeout "${IDLE_TIMEOUT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

# ---- 落盘本次启动的完整参数快照，避免以后忘记当时用了什么配置 ----
if [[ -n "${LOG_DIR}" ]]; then
  mkdir -p "${LOG_DIR}"
  GIT_COMMIT="$(git -C "${STARVLA_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  GIT_DIRTY="$(git -C "${STARVLA_DIR}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/policy_server_${RUN_STAMP}.log"
  cat > "${LOG_DIR}/policy_server_${RUN_STAMP}.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "git_commit": "${GIT_COMMIT}",
  "git_dirty_files": ${GIT_DIRTY:-0},
  "ckpt": "${CKPT}",
  "gpu_id": "${GPU_ID}",
  "port": ${PORT},
  "use_bf16": "${USE_BF16}",
  "idle_timeout": ${IDLE_TIMEOUT},
  "starvla_eval_memory_mode": "${STARVLA_EVAL_MEMORY_MODE}",
  "starvla_eval_mem_length": "${STARVLA_EVAL_MEM_LENGTH}",
  "starvla_eval_seed": "${STARVLA_EVAL_SEED}",
  "starvla_num_inference_timesteps": "${STARVLA_NUM_INFERENCE_TIMESTEPS}"
}
EOF
  echo "[run_policy_server] 参数快照已写入: ${LOG_DIR}/policy_server_${RUN_STAMP}.json"
  echo "[run_policy_server] 完整日志将同时写入: ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
fi
