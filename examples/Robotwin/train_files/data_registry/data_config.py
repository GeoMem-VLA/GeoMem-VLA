"""RobotWin benchmark — data config, embodiment tags, and mixtures."""

import os

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    RelativeToObsState,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Agilex (RobotWin, action_indices=16)
# ---------------------------------------------------------------------------
class AgilexDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    # Per-key dims for PolicyNormProcessor (Agilex 6-DOF arms + binary gripper = 14-D total)
    action_key_dims = {"action.left_joints": 6, "action.right_joints": 6, "action.left_gripper": 1, "action.right_gripper": 1}
    state_key_dims  = {"state.left_joints": 6, "state.right_joints": 6, "state.left_gripper": 1, "state.right_gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    # Dual-arm end-effector Cartesian position column (added incrementally to the LeRobot
    # dataset from the raw RoboTwin `endpose`): observation.eef = [left_xyz(3), right_xyz(3)].
    # Used ONLY by the GeoPlan EEF-waypoint auxiliary (future-horizon end-effector trajectory);
    # read at the action horizon (future W steps), NOT fed to the action head.
    eef_keys = ["eef.full"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        cfg = {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }
        # EEF endpose sequence over the action horizon (future W steps), for the GeoPlan aux target.
        if getattr(self, "eef_keys", None):
            cfg["eef"] = ModalityConfig(delta_indices=self.action_indices, modality_keys=self.eef_keys)
        return cfg

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 (action_indices=50)
# ---------------------------------------------------------------------------
class AgilexData50Config(AgilexDataConfig):
    action_indices = list(range(50))
    # eef modality is OPT-IN via the ROBOTWIN_USE_EEF env var (default OFF).
    #   OFF (default): eef_keys empty -> modality_config() skips the "eef" modality, so the
    #     dataloader never reads observation.eef. Use this for the official RoboTwin LeRobot
    #     download, which has NO observation.eef. The yaml keeps the EEF aux disabled
    #     (use_eef_endpose=false + lambda_eef=0).
    #   ON (ROBOTWIN_USE_EEF=1): read observation.eef. Enable this ONLY on a dataset that has
    #     the column — either robotwin2_lerobot (real sim endpose) or the official data after
    #     running add_eef_via_fk.py (FK-computed endpose). The launch script must also flip the
    #     yaml switches back via CLI (use_eef_endpose True, lambda_eef 3.0, eef_target_mode
    #     endpose_rel) via the training launch script's CLI overrides.
    # NOTE: read at import time; the launch script exports the var before `accelerate launch`.
    eef_keys = ["eef.full"] if os.environ.get("ROBOTWIN_USE_EEF", "0") == "1" else []

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 V2 (QwenGeoGR00T_RoboTwin pipeline)
#   * q99 joint normalization (quantile stats; min/max outliers no longer squeeze the
#     normalized range, which amplified denoising residuals -> arm jitter on unapply).
#   * eef.campose modality: per-frame head-camera intrinsic/extrinsic + original HW
#     (observation.campose, added by add_campose_to_lerobot.py) for the
#     EEF-in-pointcloud geo target (Strategy A).
# ---------------------------------------------------------------------------
class AgilexData50V2Config(AgilexData50Config):
    eef_keys = ["eef.full", "eef.campose"]

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "q99", "state.right_joints": "q99",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "q99", "action.right_joints": "q99",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 Q99 (robust-quantile normalization, no hard clipping)
#
# Fixes the min_max pathology diagnosed on robotwin_all_50 (see
# inspect_action_norm.py): the mixture-global min/max is inflated by rare
# multi-turn rotations (|q| up to ~8.5 rad from a handful of tasks), squeezing
# the normalized std of the base/wrist joints down to ~0.045 — the FM head gets
# almost no gradient signal on exactly the joints dual-arm tasks (lift_pot)
# depend on.
#
# Design (OpenPI / LingBot-VLA style):
#   * joints -> q99 normalization with FROZEN task-balanced robust quantiles
#     (robotwin_50_robust_stats.json, derived from LingBot-VLA's public
#     assets/norm_stats/robotwin_50.json; verified to match our own
#     task-balanced recomputation, see inspect_qpos_wrap.py).
#     freeze_statistics=True so the mixture's min/max-mixed q01/q99 can NOT
#     overwrite them (compute_overall_statistics defaults to min/max mixing,
#     which would re-inflate the band).
#   * q99_clip=None -> NO hard clipping: data inside [q01,q99] maps to [-1,1];
#     real out-of-band motion (multi-turn rotations, verified to be genuine
#     continuous motion with zero 2*pi wraparound jumps) simply exceeds +-1 and
#     unapply() inverts it exactly.
#   * grippers stay binary (threshold 0.49), same as robotwin50.
#
# NOTE: checkpoints trained with robotwin50 (min_max) are NOT compatible with
# this config — retrain from scratch and eval with a matching checkpoint.
# ---------------------------------------------------------------------------
_ROBUST_STATS_PATH = os.path.join(os.path.dirname(__file__), "robotwin_50_robust_stats.json")
_REL_ROBUST_STATS_PATH = os.path.join(os.path.dirname(__file__), "robotwin_50_rel_robust_stats.json")


def _load_robust_stats(keys, path=_ROBUST_STATS_PATH):
    import json
    with open(path) as f:
        stats = json.load(f)
    return {k: stats[k] for k in keys}


class AgilexData50Q99Config(AgilexData50Config):

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                q99_clip=None,
                freeze_statistics=True,
                normalization_statistics=_load_robust_stats(
                    ["state.left_joints", "state.right_joints"]
                ),
                normalization_modes={
                    "state.left_joints": "q99", "state.right_joints": "q99",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                q99_clip=None,
                freeze_statistics=True,
                normalization_statistics=_load_robust_stats(
                    ["action.left_joints", "action.right_joints"]
                ),
                normalization_modes={
                    "action.left_joints": "q99", "action.right_joints": "q99",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 REL+Q99 (relative-to-current-obs joints, robust quantiles)
#
# Third jitter experiment (after min_max -> q99): change the ACTION REPRESENTATION
# itself. Each 50-step chunk's joint targets are re-expressed relative to the
# chunk's CURRENT observation state (one shared reference frame, RelativeToObsState):
#
#     rel_action[t] = abs_action[t] - state[obs]        (grippers stay binary)
#
# Why this should help beyond plain q99:
#   * The action distribution now only depends on "how far the arm moves within
#     a chunk" — the rare multi-turn ABSOLUTE rotations that inflated the min_max
#     band (and still stretch the q99 tails) drop out entirely. Measured
#     task-balanced std_norm: 0.25~0.32 on ALL 12 joints (see
#     compute_rel_robust_stats.py output).
#   * Chunk-boundary continuity: the first action of a fresh chunk = state + small
#     delta, so re-planning can no longer produce an absolute jump from the
#     current qpos (a known jitter source at exec-horizon boundaries).
#   * NO per-step integration: one shared reference frame per chunk, so denoising
#     noise stays position noise (does not accumulate like per-step deltas would).
#
# Normalization:
#   * actions -> q99 over the RELATIVE distribution, FROZEN task-balanced robust
#     quantiles (robotwin_50_rel_robust_stats.json), q99_clip=None (OpenPI-style).
#   * state   -> unchanged ABSOLUTE q99 (robotwin_50_robust_stats.json): the
#     state_encoder still needs to know where the arm actually is.
#   * grippers stay binary (threshold 0.49) on both sides.
#
# Eval: PolicyNormProcessor detects RelativeToObsState (requires_state_on_unapply)
# and adds the current raw qpos back on unapply — the client MUST send the state
# (ROBOTWIN_SEND_STATE=1, which state-conditioned ckpts need anyway).
#
# NOTE: the dataloader also has a NATIVE datasets.vla_data.action_mode="rel" path
# (LeRobotSingleDataset._apply_action_mode, same a_t - s_0 semantics). We deliberately
# do NOT use it here: (1) its stats are recomputed per dataset and then mixture-mixed
# (min/max mixing) — the exact pathology the frozen task-balanced quantiles fixed;
# (2) it has no server-side inverse (the eval client's action_mode="rel" branch adds
# the EPISODE-INITIAL state in env order — different semantics + known layout bug).
# Keep the yaml at action_mode=abs so that native path stays a no-op.
#
# NOT checkpoint-compatible with robotwin50 / robotwin50_q99 — train from scratch.
# ---------------------------------------------------------------------------
class AgilexData50RelQ99Config(AgilexData50Config):

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            # Runs on RAW tensors: both operands must be un-normalized here.
            RelativeToObsState(
                apply_to=["action.left_joints", "action.right_joints"],
                state_key_map={
                    "action.left_joints": "state.left_joints",
                    "action.right_joints": "state.right_joints",
                },
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                q99_clip=None,
                freeze_statistics=True,
                normalization_statistics=_load_robust_stats(
                    ["state.left_joints", "state.right_joints"]
                ),
                normalization_modes={
                    "state.left_joints": "q99", "state.right_joints": "q99",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                q99_clip=None,
                freeze_statistics=True,
                normalization_statistics=_load_robust_stats(
                    ["action.left_joints", "action.right_joints"],
                    path=_REL_ROBUST_STATS_PATH,
                ),
                normalization_modes={
                    "action.left_joints": "q99", "action.right_joints": "q99",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — ARX X5
# ---------------------------------------------------------------------------
class ArxX5DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "robotwin": AgilexDataConfig(),
    "robotwin50": AgilexData50Config(),
    "robotwin50_v2": AgilexData50V2Config(),
    "robotwin50_q99": AgilexData50Q99Config(),
    "robotwin50_rel_q99": AgilexData50RelQ99Config(),
    "arx_x5": ArxX5DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}

# ---------------------------------------------------------------------------
# Data-ratio / OOD experiment scaffolding
# ---------------------------------------------------------------------------
# Canonical list of all 50 RoboTwin 2.0 task names (order = README task list).
ROBOTWIN_ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate",
    "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
    "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet",
    "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
    "stamp_seal", "turn_switch",
]

# ===========================================================================
# >>>>> EDIT HERE: define your data-ratio / OOD experiments <<<<<
# ---------------------------------------------------------------------------
# Each entry maps an experiment id -> the list of tasks the model is TRAINED on
# ("seen" / in-distribution). Every task NOT in this list is AUTOMATICALLY held
# out for OOD evaluation (unseen). Add as many experiments as you want; each one
# auto-registers training mixtures below.
#
# The examples below are placeholders — replace the task lists with your own
# chosen subsets. Task names must come from ROBOTWIN_ALL_TASKS.
# ===========================================================================
ROBOTWIN_TASK_SPLITS = {
    # A small 5-task example split (just to show the format — replace freely).
    "exp5": [
        "adjust_bottle",
        "beat_block_hammer",
        "lift_pot",
        "place_empty_cup",
        "stack_blocks_two",
    ],
    # A 10-task example split.
    "exp10": [
        "adjust_bottle", "beat_block_hammer", "lift_pot", "place_empty_cup",
        "stack_blocks_two", "grab_roller", "open_laptop", "shake_bottle",
        "place_container_plate", "click_alarmclock",
    ],
    "exp13": [
        "grab_roller", "handover_block", "lift_pot", "move_can_pot",
        "open_laptop", "pick_dual_bottles", "place_object_basket", "place_dual_shoes",
        "place_phone_stand", "put_bottles_dustbin", "put_object_cabinet", "stack_blocks_two", "stack_bowls_two",
    ],
}

# ---------------------------------------------------------------------------
# Inline "custom" split for the training launch script.
# A training launch script may write the chosen
# seen-task list to `seen_tasks_custom.txt` (next to this file). We read it here
# and register it as experiment "custom" -> mixture `robotwin_custom_seen`.
#
# IMPORTANT: a FILE is used (not an env var) because DeepSpeed/accelerate do NOT
# reliably forward custom env vars to worker processes, so an env-var-only design
# fails under distributed launch. The env var ROBOTWIN_SEEN_TASKS is kept only as
# a fallback for single-process runs.
# ---------------------------------------------------------------------------
_SEEN_TASKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seen_tasks_custom.txt")
_seen_raw = ""
if os.path.isfile(_SEEN_TASKS_FILE):
    with open(_SEEN_TASKS_FILE) as _f:
        _seen_raw = _f.read()
if not _seen_raw.strip():
    _seen_raw = os.environ.get("ROBOTWIN_SEEN_TASKS", "")
_custom_seen = [t.strip() for t in _seen_raw.replace(",", " ").split() if t.strip() and not t.strip().startswith("#")]
if _custom_seen:
    ROBOTWIN_TASK_SPLITS["custom"] = _custom_seen


def get_unseen_tasks(seen_tasks):
    """Tasks NOT in `seen_tasks` (kept in canonical order) -> OOD eval set."""
    seen = set(seen_tasks)
    return [t for t in ROBOTWIN_ALL_TASKS if t not in seen]


def build_robotwin_mixture(tasks, splits=("Clean", "Randomized"), robot_type="robotwin50", weight=1.0):
    """Build a mixture spec (list of (data_name, weight, robot_type)) for `tasks`.

    `splits` selects the demonstration source(s): "Clean" (50 traj/task) and/or
    "Randomized" (500 traj/task). Matches the layout used by robotwin_all_50.
    """
    spec = []
    for split in splits:
        for t in tasks:
            spec.append((f"{split}/{t}", weight, robot_type))
    return spec


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "robotwin_all": [
        ("Clean/adjust_bottle", 1.0, "robotwin"), ("Randomized/adjust_bottle", 1.0, "robotwin"),
        ("Clean/beat_block_hammer", 1.0, "robotwin"), ("Randomized/beat_block_hammer", 1.0, "robotwin"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin"), ("Randomized/blocks_ranking_size", 1.0, "robotwin"),
        ("Clean/click_alarmclock", 1.0, "robotwin"), ("Randomized/click_alarmclock", 1.0, "robotwin"),
        ("Clean/click_bell", 1.0, "robotwin"), ("Randomized/click_bell", 1.0, "robotwin"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin"),
        ("Clean/grab_roller", 1.0, "robotwin"), ("Randomized/grab_roller", 1.0, "robotwin"),
        ("Clean/handover_block", 1.0, "robotwin"), ("Randomized/handover_block", 1.0, "robotwin"),
        ("Clean/handover_mic", 1.0, "robotwin"), ("Randomized/handover_mic", 1.0, "robotwin"),
        ("Clean/hanging_mug", 1.0, "robotwin"), ("Randomized/hanging_mug", 1.0, "robotwin"),
        ("Clean/lift_pot", 1.0, "robotwin"), ("Randomized/lift_pot", 1.0, "robotwin"),
        ("Clean/move_can_pot", 1.0, "robotwin"), ("Randomized/move_can_pot", 1.0, "robotwin"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin"),
        ("Clean/move_playingcard_away", 1.0, "robotwin"), ("Randomized/move_playingcard_away", 1.0, "robotwin"),
        ("Clean/move_stapler_pad", 1.0, "robotwin"), ("Randomized/move_stapler_pad", 1.0, "robotwin"),
        ("Clean/open_laptop", 1.0, "robotwin"), ("Randomized/open_laptop", 1.0, "robotwin"),
        ("Clean/open_microwave", 1.0, "robotwin"), ("Randomized/open_microwave", 1.0, "robotwin"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin"), ("Randomized/pick_dual_bottles", 1.0, "robotwin"),
        ("Clean/place_a2b_left", 1.0, "robotwin"), ("Randomized/place_a2b_left", 1.0, "robotwin"),
        ("Clean/place_a2b_right", 1.0, "robotwin"), ("Randomized/place_a2b_right", 1.0, "robotwin"),
        ("Clean/place_bread_basket", 1.0, "robotwin"), ("Randomized/place_bread_basket", 1.0, "robotwin"),
        ("Clean/place_bread_skillet", 1.0, "robotwin"), ("Randomized/place_bread_skillet", 1.0, "robotwin"),
        ("Clean/place_burger_fries", 1.0, "robotwin"), ("Randomized/place_burger_fries", 1.0, "robotwin"),
        ("Clean/place_can_basket", 1.0, "robotwin"), ("Randomized/place_can_basket", 1.0, "robotwin"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin"),
        ("Clean/place_container_plate", 1.0, "robotwin"), ("Randomized/place_container_plate", 1.0, "robotwin"),
        ("Clean/place_dual_shoes", 1.0, "robotwin"), ("Randomized/place_dual_shoes", 1.0, "robotwin"),
        ("Clean/place_empty_cup", 1.0, "robotwin"), ("Randomized/place_empty_cup", 1.0, "robotwin"),
        ("Clean/place_fan", 1.0, "robotwin"), ("Randomized/place_fan", 1.0, "robotwin"),
        ("Clean/place_mouse_pad", 1.0, "robotwin"), ("Randomized/place_mouse_pad", 1.0, "robotwin"),
        ("Clean/place_object_basket", 1.0, "robotwin"), ("Randomized/place_object_basket", 1.0, "robotwin"),
        ("Clean/place_object_scale", 1.0, "robotwin"), ("Randomized/place_object_scale", 1.0, "robotwin"),
        ("Clean/place_object_stand", 1.0, "robotwin"), ("Randomized/place_object_stand", 1.0, "robotwin"),
        ("Clean/place_phone_stand", 1.0, "robotwin"), ("Randomized/place_phone_stand", 1.0, "robotwin"),
        ("Clean/place_shoe", 1.0, "robotwin"), ("Randomized/place_shoe", 1.0, "robotwin"),
        ("Clean/press_stapler", 1.0, "robotwin"), ("Randomized/press_stapler", 1.0, "robotwin"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin"),
        ("Clean/put_object_cabinet", 1.0, "robotwin"), ("Randomized/put_object_cabinet", 1.0, "robotwin"),
        ("Clean/rotate_qrcode", 1.0, "robotwin"), ("Randomized/rotate_qrcode", 1.0, "robotwin"),
        ("Clean/scan_object", 1.0, "robotwin"), ("Randomized/scan_object", 1.0, "robotwin"),
        ("Clean/shake_bottle", 1.0, "robotwin"), ("Randomized/shake_bottle", 1.0, "robotwin"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Clean/stack_blocks_three", 1.0, "robotwin"), ("Randomized/stack_blocks_three", 1.0, "robotwin"),
        ("Clean/stack_blocks_two", 1.0, "robotwin"), ("Randomized/stack_blocks_two", 1.0, "robotwin"),
        ("Clean/stack_bowls_three", 1.0, "robotwin"), ("Randomized/stack_bowls_three", 1.0, "robotwin"),
        ("Clean/stack_bowls_two", 1.0, "robotwin"), ("Randomized/stack_bowls_two", 1.0, "robotwin"),
        ("Clean/stamp_seal", 1.0, "robotwin"), ("Randomized/stamp_seal", 1.0, "robotwin"),
        ("Clean/turn_switch", 1.0, "robotwin"), ("Randomized/turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_all_50": [
        ("Clean/adjust_bottle", 1.0, "robotwin50"), ("Randomized/adjust_bottle", 1.0, "robotwin50"),
        ("Clean/beat_block_hammer", 1.0, "robotwin50"), ("Randomized/beat_block_hammer", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin50"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin50"), ("Randomized/blocks_ranking_size", 1.0, "robotwin50"),
        ("Clean/click_alarmclock", 1.0, "robotwin50"), ("Randomized/click_alarmclock", 1.0, "robotwin50"),
        ("Clean/click_bell", 1.0, "robotwin50"), ("Randomized/click_bell", 1.0, "robotwin50"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin50"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin50"),
        ("Clean/grab_roller", 1.0, "robotwin50"), ("Randomized/grab_roller", 1.0, "robotwin50"),
        ("Clean/handover_block", 1.0, "robotwin50"), ("Randomized/handover_block", 1.0, "robotwin50"),
        ("Clean/handover_mic", 1.0, "robotwin50"), ("Randomized/handover_mic", 1.0, "robotwin50"),
        ("Clean/hanging_mug", 1.0, "robotwin50"), ("Randomized/hanging_mug", 1.0, "robotwin50"),
        ("Clean/lift_pot", 1.0, "robotwin50"), ("Randomized/lift_pot", 1.0, "robotwin50"),
        ("Clean/move_can_pot", 1.0, "robotwin50"), ("Randomized/move_can_pot", 1.0, "robotwin50"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin50"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin50"),
        ("Clean/move_playingcard_away", 1.0, "robotwin50"), ("Randomized/move_playingcard_away", 1.0, "robotwin50"),
        ("Clean/move_stapler_pad", 1.0, "robotwin50"), ("Randomized/move_stapler_pad", 1.0, "robotwin50"),
        ("Clean/open_laptop", 1.0, "robotwin50"), ("Randomized/open_laptop", 1.0, "robotwin50"),
        ("Clean/open_microwave", 1.0, "robotwin50"), ("Randomized/open_microwave", 1.0, "robotwin50"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin50"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin50"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin50"), ("Randomized/pick_dual_bottles", 1.0, "robotwin50"),
        ("Clean/place_a2b_left", 1.0, "robotwin50"), ("Randomized/place_a2b_left", 1.0, "robotwin50"),
        ("Clean/place_a2b_right", 1.0, "robotwin50"), ("Randomized/place_a2b_right", 1.0, "robotwin50"),
        ("Clean/place_bread_basket", 1.0, "robotwin50"), ("Randomized/place_bread_basket", 1.0, "robotwin50"),
        ("Clean/place_bread_skillet", 1.0, "robotwin50"), ("Randomized/place_bread_skillet", 1.0, "robotwin50"),
        ("Clean/place_burger_fries", 1.0, "robotwin50"), ("Randomized/place_burger_fries", 1.0, "robotwin50"),
        ("Clean/place_can_basket", 1.0, "robotwin50"), ("Randomized/place_can_basket", 1.0, "robotwin50"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin50"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin50"),
        ("Clean/place_container_plate", 1.0, "robotwin50"), ("Randomized/place_container_plate", 1.0, "robotwin50"),
        ("Clean/place_dual_shoes", 1.0, "robotwin50"), ("Randomized/place_dual_shoes", 1.0, "robotwin50"),
        ("Clean/place_empty_cup", 1.0, "robotwin50"), ("Randomized/place_empty_cup", 1.0, "robotwin50"),
        ("Clean/place_fan", 1.0, "robotwin50"), ("Randomized/place_fan", 1.0, "robotwin50"),
        ("Clean/place_mouse_pad", 1.0, "robotwin50"), ("Randomized/place_mouse_pad", 1.0, "robotwin50"),
        ("Clean/place_object_basket", 1.0, "robotwin50"), ("Randomized/place_object_basket", 1.0, "robotwin50"),
        ("Clean/place_object_scale", 1.0, "robotwin50"), ("Randomized/place_object_scale", 1.0, "robotwin50"),
        ("Clean/place_object_stand", 1.0, "robotwin50"), ("Randomized/place_object_stand", 1.0, "robotwin50"),
        ("Clean/place_phone_stand", 1.0, "robotwin50"), ("Randomized/place_phone_stand", 1.0, "robotwin50"),
        ("Clean/place_shoe", 1.0, "robotwin50"), ("Randomized/place_shoe", 1.0, "robotwin50"),
        ("Clean/press_stapler", 1.0, "robotwin50"), ("Randomized/press_stapler", 1.0, "robotwin50"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin50"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin50"),
        ("Clean/put_object_cabinet", 1.0, "robotwin50"), ("Randomized/put_object_cabinet", 1.0, "robotwin50"),
        ("Clean/rotate_qrcode", 1.0, "robotwin50"), ("Randomized/rotate_qrcode", 1.0, "robotwin50"),
        ("Clean/scan_object", 1.0, "robotwin50"), ("Randomized/scan_object", 1.0, "robotwin50"),
        ("Clean/shake_bottle", 1.0, "robotwin50"), ("Randomized/shake_bottle", 1.0, "robotwin50"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin50"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin50"),
        ("Clean/stack_blocks_three", 1.0, "robotwin50"), ("Randomized/stack_blocks_three", 1.0, "robotwin50"),
        ("Clean/stack_blocks_two", 1.0, "robotwin50"), ("Randomized/stack_blocks_two", 1.0, "robotwin50"),
        ("Clean/stack_bowls_three", 1.0, "robotwin50"), ("Randomized/stack_bowls_three", 1.0, "robotwin50"),
        ("Clean/stack_bowls_two", 1.0, "robotwin50"), ("Randomized/stack_bowls_two", 1.0, "robotwin50"),
        ("Clean/stamp_seal", 1.0, "robotwin50"), ("Randomized/stamp_seal", 1.0, "robotwin50"),
        ("Clean/turn_switch", 1.0, "robotwin50"), ("Randomized/turn_switch", 1.0, "robotwin50"),
    ],
    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_task1": [("adjust_bottle", 1.0, "robotwin")],
    "robotwin_task2": [("place_a2b_left", 1.0, "robotwin"), ("place_a2b_right", 1.0, "robotwin")],
    "arx_x5": [("arx_x5", 1.0, "arx_x5")],
}

# V2 mixture: identical dataset list to robotwin_all_50, but routed through the
# robotwin50_v2 DataConfig (q99 normalization + campose modality).
DATASET_NAMED_MIXTURES["robotwin_all_50_v2"] = [
    (name, weight, "robotwin50_v2")
    for (name, weight, _rt) in DATASET_NAMED_MIXTURES["robotwin_all_50"]
]

# Q99 mixture: identical dataset list to robotwin_all_50, but routed through the
# robotwin50_q99 DataConfig (frozen task-balanced robust quantiles, no hard clipping,
# no campose modality — compatible with the generic QwenGeoGR00T pipeline).
DATASET_NAMED_MIXTURES["robotwin_all_50_q99"] = [
    (name, weight, "robotwin50_q99")
    for (name, weight, _rt) in DATASET_NAMED_MIXTURES["robotwin_all_50"]
]

# REL+Q99 mixture: identical dataset list, routed through robotwin50_rel_q99
# (relative-to-current-obs joint actions + frozen task-balanced robust quantiles
# of the RELATIVE distribution; state stays absolute q99).
DATASET_NAMED_MIXTURES["robotwin_all_50_rel_q99"] = [
    (name, weight, "robotwin50_rel_q99")
    for (name, weight, _rt) in DATASET_NAMED_MIXTURES["robotwin_all_50"]
]

# ---------------------------------------------------------------------------
# Auto-register data-ratio experiment mixtures from ROBOTWIN_TASK_SPLITS.
# For each experiment id "<exp>" (incl. the inline "custom" one), registers:
#   robotwin_<exp>_seen         -> Clean + Randomized of the seen tasks (main)
#   robotwin_<exp>_seen_clean   -> Clean only of the seen tasks
# e.g. the launch script sets data_mix=robotwin_custom_seen.
# ---------------------------------------------------------------------------
for _exp, _seen in ROBOTWIN_TASK_SPLITS.items():
    _unknown = [t for t in _seen if t not in ROBOTWIN_ALL_TASKS]
    if _unknown:
        raise ValueError(f"[data_config] experiment {_exp!r} has unknown tasks: {_unknown}")
    DATASET_NAMED_MIXTURES[f"robotwin_{_exp}_seen"] = build_robotwin_mixture(_seen)
    DATASET_NAMED_MIXTURES[f"robotwin_{_exp}_seen_clean"] = build_robotwin_mixture(_seen, splits=("Clean",))
