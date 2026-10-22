"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    # ------------------------------------------------------------------
    # Weak-task focused post-training (hard-example reweighting).
    #
    # Build the per-task subset folders first with (stats are reused from the
    # source suite BY DEFAULT, which is required for co-training consistency):
    #   examples/LIBERO/train_files/build_weak_subsets.sh
    #
    # The 4 currently-weak tasks (each lives in a different suite):
    #   libero_spatial : pick up the black bowl on the ramekin and place it on the plate
    #   libero_object  : pick up the bbq sauce and place it in the basket
    #   libero_goal    : put the wine bottle on top of the cabinet
    #   libero_10      : put both moka pots on the stove
    # ------------------------------------------------------------------
    "libero_spatial_ramekin_only": [
        ("libero_spatial_ramekin_hard_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    # Legacy single-task hard-mix (ramekin only). Kept for reproducibility.
    "libero_all_ramekin_hardmix_v1": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_ramekin_hard_no_noops_1.0.0_lerobot", 2.0, "libero_franka"),
    ],
    # RECOMMENDED low-forgetting hard-mix covering ALL 4 weak tasks.
    #
    # Sampling-share math (weights are normalized across all datasets in the
    # mix; see LeRobotMixtureDataset._dataset_sampling_weights):
    #   total weight = 4*1.0 (full suites) + 4*0.5 (weak subsets) = 6.0
    #   each full suite  -> 1.0/6 ~= 16.7%
    #   each weak subset -> 0.5/6 ~=  8.3%   (4 subsets -> ~33% of all samples)
    # i.e. the 4 weak tasks are oversampled to ~1/3 of the data WITHOUT dropping
    # any original task (anti-forgetting). To push harder on the weak tasks, raise
    # the 0.5 weights (e.g. 1.0 -> ~50% weak share); to be more conservative,
    # lower them (e.g. 0.25 -> ~20% weak share). Keep at least one dataset at
    # weight==1.0 so the mixture has a valid "primary" set.
    "libero_all_weak4_hardmix": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_ramekin_hard_no_noops_1.0.0_lerobot", 0.5, "libero_franka"),
        ("libero_object_bbq_hard_no_noops_1.0.0_lerobot", 0.5, "libero_franka"),
        ("libero_goal_wine_hard_no_noops_1.0.0_lerobot", 0.5, "libero_franka"),
        ("libero_10_moka_hard_no_noops_1.0.0_lerobot", 0.5, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}
