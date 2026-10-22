#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

# Default precision = fp32 (stable). Set ROBOTWIN_USE_BF16=1 to opt back into bf16.
use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-0}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"

# start_eval.sh terminates and waits for this wrapper's PID after each task.
# Replace the shell so that PID owns the actual listener; a separate Python child
# would survive shell cleanup and block the next run from reusing its port.
export STARVLA_EVAL_MEMORY_MODE="${STARVLA_EVAL_MEMORY_MODE:-streaming}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
exec "${star_vla_python}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${use_bf16_flag[@]}"
