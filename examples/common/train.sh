#!/usr/bin/env bash
# Shared public launcher. Activate your environment before running this script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

benchmark="${1:?Expected LIBERO or Robotwin}"
case "${benchmark}" in
  LIBERO)
    config="examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
    data_root="${LIBERO_DATA_ROOT:?Set LIBERO_DATA_ROOT to the LeRobot dataset root}"
    data_mix="${DATA_MIX:-libero_spatial_state}"
    accumulation="${GRADIENT_ACCUMULATION_STEPS:-4}"
    ;;
  Robotwin)
    config="examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml"
    data_root="${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT to the LeRobot dataset root}"
    data_mix="${DATA_MIX:-robotwin_all_50_q99}"
    accumulation="${GRADIENT_ACCUMULATION_STEPS:-8}"
    # The basic recipe uses official joint-space data, without an EEF column.
    export ROBOTWIN_USE_EEF=0
    ;;
  *) echo "Unknown benchmark: ${benchmark}" >&2; exit 2 ;;
esac

: "${BASE_VLM:?Set BASE_VLM to a local Qwen3-VL-4B-Instruct directory}"
: "${VGGT_CKPT:?Set VGGT_CKPT to the VGGT weights file}"
: "${VGGT_REPO:?Set VGGT_REPO to the VGGT source directory}"
for directory in "${BASE_VLM}" "${data_root}" "${VGGT_REPO}"; do
  [[ -d "${directory}" ]] || { echo "Directory not found: ${directory}" >&2; exit 2; }
done
[[ -f "${VGGT_CKPT}" ]] || { echo "VGGT weights not found: ${VGGT_CKPT}" >&2; exit 2; }
[[ -f "${VGGT_REPO}/vggt/models/vggt.py" ]] || { echo "VGGT_REPO must contain vggt/models/vggt.py" >&2; exit 2; }

batch="${PER_DEVICE_BATCH_SIZE:-16}"
group="${GROUP_SIZE:-16}"
memory="${MEM_LENGTH:-${group}}"
processes="${NUM_PROCESSES:-1}"
for value in "${batch}" "${group}" "${memory}" "${processes}" "${accumulation}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch/group/memory/process/accumulation values must be positive integers" >&2; exit 2; }
done
(( batch % group == 0 && memory == group )) || {
  echo "Require batch size % group size == 0 and memory length == group size" >&2; exit 2;
}
# Trainer and DeepSpeed must agree on accumulation and optimizer-step boundaries.
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${accumulation}"
echo "${benchmark}: batch=${batch}, group=${group}, memory=${memory}, effective batch=$((batch * processes * accumulation))"

exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${processes}" \
  --gradient_accumulation_steps "${accumulation}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config}" \
  --framework.geoplan.group_size "${group}" \
  --framework.geoplan.mem_length "${memory}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.group_size "${group}" \
  --datasets.vla_data.per_device_batch_size "${batch}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS:-4}" \
  --trainer.gradient_accumulation_steps "${accumulation}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-20000}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS:-100}" \
  --trainer.save_interval "${SAVE_INTERVAL:-1000}" \
  --trainer.is_resume "${IS_RESUME:-False}" \
  --run_root_dir "${RUN_ROOT_DIR:-results/Checkpoints}" \
  --run_id "${RUN_ID:-${benchmark}_geomem_basic}"
