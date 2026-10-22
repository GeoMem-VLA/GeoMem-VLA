#!/usr/bin/env bash
set -euo pipefail

# Basic evaluation: one task, one trial. This is a smoke run, not a benchmark score.
# Activate the LIBERO environment first, or set LIBERO_PYTHON.
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
: "${LIBERO_HOME:?Set LIBERO_HOME to the external LIBERO repository}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"

# Point CKPT at the checkpoint you want to evaluate, or pass it inline:
#   CKPT=/path/to/checkpoints/steps_XXXXXX_pytorch_model.pt bash eval_libero.sh
: "${CKPT:?Set CKPT to the same checkpoint used by the policy server}"
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}" >&2; exit 2; }
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_TASKS="${MAX_TASKS:-1}"
LIBERO_MAX_STEPS="${LIBERO_MAX_STEPS:-}"
LIBERO_SAVE_ACTIONS="${LIBERO_SAVE_ACTIONS:-0}"
SEED="${SEED:-7}"
UNNORM_KEY="${UNNORM_KEY:-}"
JOB_NAME="${JOB_NAME:-test}"

# 模型输入 / 兼容性开关 —— 必须和训练该 checkpoint 时的配置对齐
LIBERO_SEND_STATE="${LIBERO_SEND_STATE:-1}"
EVAL_LEGACY_DELTA_RECON="${EVAL_LEGACY_DELTA_RECON:-0}"
LIBERO_EXEC_HORIZON="${LIBERO_EXEC_HORIZON:-0}"

# AAC 自适应 chunk（实验性，默认全部关闭）
LIBERO_AAC="${LIBERO_AAC:-0}"
LIBERO_AAC_N_SAMPLES="${LIBERO_AAC_N_SAMPLES:-20}"
LIBERO_AAC_METHOD="${LIBERO_AAC_METHOD:-gaussian_bernoulli}"
LIBERO_AAC_MOVE_TH="${LIBERO_AAC_MOVE_TH:-0.0}"
LIBERO_AAC_MIN_CHUNK="${LIBERO_AAC_MIN_CHUNK:-2}"
LIBERO_AAC_MAX_CHUNK="${LIBERO_AAC_MAX_CHUNK:-}"
LIBERO_AAC_CHUNK_ID="${LIBERO_AAC_CHUNK_ID:-0}"
LIBERO_AAC_DEBUG="${LIBERO_AAC_DEBUG:-0}"

MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"
DEBUG="${DEBUG:-}"

LIBERO_TAG_RESULTS_DIR="${LIBERO_TAG_RESULTS_DIR:-0}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0"
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"
# 下面这些由 model2libero_interface.py 内部通过 os.environ.get 读取，这里 export 出去只是为了
# 保证子进程一定能看到（即便脚本被 `source` 之类的方式调用）。
export LIBERO_SEND_STATE EVAL_LEGACY_DELTA_RECON LIBERO_EXEC_HORIZON LIBERO_SAVE_ACTIONS
export LIBERO_AAC LIBERO_AAC_N_SAMPLES LIBERO_AAC_METHOD LIBERO_AAC_MOVE_TH
export LIBERO_AAC_MIN_CHUNK LIBERO_AAC_CHUNK_ID LIBERO_AAC_DEBUG
# 留空 = 让 python 端用 action_chunk_size 自动兜底，所以只有非空时才 export（避免 int("") 报错）
[[ -n "${LIBERO_AAC_MAX_CHUNK}" ]] && export LIBERO_AAC_MAX_CHUNK
# 同理：留空 = 用 MAX_STEPS_BY_SUITE 里的标准上限
[[ -n "${LIBERO_MAX_STEPS}" ]] && export LIBERO_MAX_STEPS
[[ -n "${DEBUG}" ]] && export DEBUG

RESULT_TAG=""
if [[ "${LIBERO_TAG_RESULTS_DIR}" == "1" ]]; then
  RESULT_TAG="_trials${NUM_TRIALS_PER_TASK}"
  [[ "${LIBERO_SEND_STATE}" == "1" ]] && RESULT_TAG="${RESULT_TAG}_state1"
  [[ "${EVAL_LEGACY_DELTA_RECON}" == "1" ]] && RESULT_TAG="${RESULT_TAG}_legacydelta"
  [[ "${LIBERO_AAC}" == "1" ]] && RESULT_TAG="${RESULT_TAG}_aac"
fi

FOLDER_NAME="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')${RESULT_TAG}"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-${MODEL_ROOT}/results/${TASK_SUITE_NAME}/${FOLDER_NAME}}"

mkdir -p "${VIDEO_OUT_PATH}"

# ---- 落盘本次运行的完整参数快照，避免以后忘记当时用了什么配置 ----
GIT_COMMIT="$(git -C "${STARVLA_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "${STARVLA_DIR}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
cat > "${VIDEO_OUT_PATH}/run_config.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "git_commit": "${GIT_COMMIT}",
  "git_dirty_files": ${GIT_DIRTY:-0},
  "ckpt": "${CKPT}",
  "host": "${HOST}",
  "port": ${PORT},
  "task_suite_name": "${TASK_SUITE_NAME}",
  "num_trials_per_task": ${NUM_TRIALS_PER_TASK},
  "num_steps_wait": ${NUM_STEPS_WAIT},
  "max_tasks": ${MAX_TASKS},
  "libero_max_steps": "${LIBERO_MAX_STEPS}",
  "libero_save_actions": "${LIBERO_SAVE_ACTIONS}",
  "seed": ${SEED},
  "unnorm_key": "${UNNORM_KEY}",
  "job_name": "${JOB_NAME}",
  "libero_send_state": "${LIBERO_SEND_STATE}",
  "eval_legacy_delta_recon": "${EVAL_LEGACY_DELTA_RECON}",
  "libero_exec_horizon": "${LIBERO_EXEC_HORIZON}",
  "libero_aac": "${LIBERO_AAC}",
  "libero_aac_n_samples": "${LIBERO_AAC_N_SAMPLES}",
  "libero_aac_method": "${LIBERO_AAC_METHOD}",
  "libero_aac_move_th": "${LIBERO_AAC_MOVE_TH}",
  "libero_aac_min_chunk": "${LIBERO_AAC_MIN_CHUNK}",
  "libero_aac_max_chunk": "${LIBERO_AAC_MAX_CHUNK}",
  "libero_aac_chunk_id": "${LIBERO_AAC_CHUNK_ID}",
  "libero_aac_debug": "${LIBERO_AAC_DEBUG}",
  "mujoco_gl": "${MUJOCO_GL_VALUE}",
  "video_out_path": "${VIDEO_OUT_PATH}"
}
EOF
echo "[eval_libero] 参数快照已写入: ${VIDEO_OUT_PATH}/run_config.json"
echo "[eval_libero] 完整日志将同时写入: ${VIDEO_OUT_PATH}/eval_run.log"

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.num-steps-wait "${NUM_STEPS_WAIT}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.seed "${SEED}" \
  --args.job-name "${JOB_NAME}" \
  ${UNNORM_KEY:+--args.unnorm-key "${UNNORM_KEY}"} \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  2>&1 | tee -a "${VIDEO_OUT_PATH}/eval_run.log"
