"""
Convert RoboTwin 2.0 zip/HDF5 data to LeRobot v2.1 format for starVLA training.

RoboTwin 2.0 HDF5 structure:
  joint_action/left_arm   (T, 6)
  joint_action/left_gripper (T,)
  joint_action/right_arm  (T, 6)
  joint_action/right_gripper (T,)
  joint_action/vector     (T, 14)
  observation/head_camera/rgb   (T,) compressed JPEG bytes
  observation/left_camera/rgb   (T,) compressed JPEG bytes
  observation/right_camera/rgb  (T,) compressed JPEG bytes
  observation/front_camera/rgb  (T,) compressed JPEG bytes

Output LeRobot keys:
  observation.state  (T, 14)  <- joint_action/vector (state ≈ current joints)
  action             (T, 14)  <- joint_action/vector
  observation.images.cam_high       <- head_camera
  observation.images.cam_left_wrist <- left_camera
  observation.images.cam_right_wrist <- right_camera

Usage (in lingbot conda env):
  python convert_robotwin_to_lerobot.py \
    --data-root /path/to/RoboTwin/data/dataset \
    --output-root /path/to/playground/Datasets/RoboTwin \
    --task-instruction-dir /path/to/RoboTwin/description/task_instruction \
    [--tasks adjust_bottle place_shoe ...]   # default: all 50 tasks
    [--types clean randomized]               # default: both
    [--dry-run]
"""

import argparse
import dataclasses
import io
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
import torch
import tqdm

# ── LeRobot import ──────────────────────────────────────────────────────────
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ── All 50 RoboTwin 2.0 tasks ────────────────────────────────────────────────
ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox",
    "place_container_plate", "place_dual_shoes", "place_empty_cup", "place_fan",
    "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle", "shake_bottle_horizontally", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
]

# zip names for each data type
ZIP_NAMES = {
    "clean": "aloha-agilex_clean_50.zip",
    "randomized": "aloha-agilex_randomized_500.zip",
}

# folder inside the zip
ZIP_INNER_DIRS = {
    "clean": "aloha-agilex_clean_50",
    "randomized": "aloha-agilex_randomized_500",
}

# output subdirectory names
OUTPUT_SUBDIRS = {
    "clean": "Clean",
    "randomized": "Randomized",
}

# camera mapping: HDF5 key → LeRobot observation.images key suffix
CAMERA_MAP = {
    "head_camera":  "cam_high",
    "left_camera":  "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

FPS = 15
ROBOT_TYPE = "aloha"


# ── Dataset config ────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 4
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_CFG = DatasetConfig()

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]


def build_features(img_h: int = 480, img_w: int = 640) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [MOTORS],
        },
    }
    for cam_suffix in CAMERA_MAP.values():
        features[f"observation.images.{cam_suffix}"] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def detect_image_shape(zip_path: Path, inner_dir: str) -> tuple[int, int]:
    """Read one frame from the first episode to detect actual image dimensions."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        hdf5_names = sorted(
            n for n in zf.namelist()
            if n.startswith(f"{inner_dir}/data/") and n.endswith(".hdf5")
        )
        if not hdf5_names:
            return 480, 640
        with zf.open(hdf5_names[0]) as f:
            ep_bytes = f.read()
    with h5py.File(io.BytesIO(ep_bytes), "r") as ep:
        raw = ep["observation/head_camera/rgb"][0]
    img = decode_image(raw)
    return img.shape[0], img.shape[1]  # H, W


def count_zip_episodes(zip_path: Path, inner_dir: str) -> int:
    """Count how many episode HDF5 files are inside the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return sum(
            1 for n in zf.namelist()
            if n.startswith(f"{inner_dir}/data/") and n.endswith(".hdf5")
        )


def is_conversion_complete(dataset_dir: Path, expected_episodes: int) -> bool:
    """Check whether a previously-written LeRobot dataset already has all episodes.

    Used to resume an interrupted run: if a task/type combo was fully converted
    before, skip it instead of re-doing the work.
    """
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        with open(info_path) as f:
            info = json.load(f)
        return info.get("total_episodes", -1) == expected_episodes
    except Exception:
        return False


def create_empty_dataset(
    repo_id: str,
    output_root: Path,
    img_h: int = 480,
    img_w: int = 640,
    cfg: DatasetConfig = DEFAULT_CFG,
) -> LeRobotDataset:
    # root = the exact dataset directory (not parent)
    dataset_dir = output_root / repo_id
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    # LeRobotDatasetMetadata.create does mkdir with exist_ok=False, so don't pre-create it
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=dataset_dir,   # exact dataset path, not parent
        robot_type=ROBOT_TYPE,
        features=build_features(img_h, img_w),
        use_videos=cfg.use_videos,
        tolerance_s=cfg.tolerance_s,
        image_writer_processes=cfg.image_writer_processes,
        image_writer_threads=cfg.image_writer_threads,
        video_backend=cfg.video_backend,
    )


# ── Image decoding ────────────────────────────────────────────────────────────
def decode_image(raw: bytes | np.ndarray) -> np.ndarray:
    """Decode compressed JPEG bytes stored in HDF5 to HWC uint8 RGB ndarray."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None – invalid JPEG bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Task instruction loading ──────────────────────────────────────────────────
def load_instructions(task_instruction_dir: Path, task: str) -> list[str]:
    """Load 'seen' instruction templates; fall back to full_description."""
    json_path = task_instruction_dir / f"{task}.json"
    if not json_path.exists():
        return [task.replace("_", " ")]
    with open(json_path) as f:
        d = json.load(f)
    # Use seen list; strip {A}/{a} placeholders → use full_description as fallback
    seen = d.get("seen", [])
    clean = []
    for s in seen:
        # Only keep templates without substitution variables
        if "{" not in s:
            clean.append(s)
    if clean:
        return clean
    # Fall back to full description
    full = d.get("full_description", task.replace("_", " "))
    return [full]


# ── Episode reading ───────────────────────────────────────────────────────────
def read_episode_from_hdf5(ep_bytes: bytes) -> dict:
    """Read one HDF5 episode from raw bytes; returns dict with arrays."""
    with h5py.File(io.BytesIO(ep_bytes), "r") as ep:
        action = torch.from_numpy(ep["joint_action/vector"][:].astype(np.float32))
        state = action.clone()  # state ≈ current joint positions in sim

        imgs: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_MAP}
        for hdf5_cam in CAMERA_MAP:
            path = f"observation/{hdf5_cam}/rgb"
            if path not in ep:
                raise KeyError(f"Camera path '{path}' not found in HDF5. "
                               f"Available: {list(ep['observation'].keys())}")
            raw_arr = ep[path]
            for raw in raw_arr:
                imgs[hdf5_cam].append(decode_image(raw))

    return {"action": action, "state": state, "imgs": imgs}


# ── Main conversion for one task ──────────────────────────────────────────────
def convert_task(
    task: str,
    data_type: str,  # "clean" or "randomized"
    data_root: Path,
    output_root: Path,
    task_instruction_dir: Path,
    dry_run: bool = False,
    resume: bool = True,
) -> None:
    zip_path = data_root / task / ZIP_NAMES[data_type]
    if not zip_path.exists():
        print(f"  [SKIP] zip not found: {zip_path}")
        return

    inner_dir = ZIP_INNER_DIRS[data_type]
    output_subdir = OUTPUT_SUBDIRS[data_type]
    repo_id_str = f"{output_subdir}/{task}"
    leftover = output_root / repo_id_str

    if resume and leftover.exists():
        expected_episodes = count_zip_episodes(zip_path, inner_dir)
        if is_conversion_complete(leftover, expected_episodes):
            print(f"  [SKIP] already complete ({expected_episodes} episodes) → {leftover}")
            return

    instructions = load_instructions(task_instruction_dir, task)
    print(f"  Instructions sample: {instructions[:2]}")

    if dry_run:
        print(f"  [DRY RUN] would convert {zip_path} → {output_root / repo_id_str}")
        return

    # Clean up any leftover from a previous incomplete/failed run
    if leftover.exists():
        print(f"  [RESUME] incomplete leftover found, redoing task from scratch: {leftover}")
        shutil.rmtree(leftover)

    try:
        img_h, img_w = detect_image_shape(zip_path, inner_dir)
        print(f"  Detected image shape: {img_h}x{img_w}")
        dataset = create_empty_dataset(repo_id_str, output_root, img_h, img_w)
    except Exception as e:
        print(f"  [ERROR] create_empty_dataset: {e}")
        return

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Collect all episode HDF5 paths inside the zip
        hdf5_names = sorted(
            n for n in zf.namelist()
            if n.startswith(f"{inner_dir}/data/") and n.endswith(".hdf5")
        )
        print(f"  Found {len(hdf5_names)} episodes")

        for ep_name in tqdm.tqdm(hdf5_names, desc=f"{task}/{data_type}"):
            with zf.open(ep_name) as f:
                ep_bytes = f.read()

            try:
                ep_data = read_episode_from_hdf5(ep_bytes)
            except Exception as e:
                print(f"  [WARN] skip {ep_name}: {e}")
                continue

            action = ep_data["action"]
            state = ep_data["state"]
            imgs = ep_data["imgs"]
            num_frames = action.shape[0]
            instruction = random.choice(instructions)

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": instruction,
                }
                for hdf5_cam, lerobot_cam in CAMERA_MAP.items():
                    frame[f"observation.images.{lerobot_cam}"] = imgs[hdf5_cam][i]

                dataset.add_frame(frame)

            dataset.save_episode()

    print(f"  ✓ Saved to {output_root / repo_id_str} ({dataset.num_episodes} episodes)")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Convert RoboTwin 2.0 data to LeRobot format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("${ROBOTWIN_DATA_ROOT:-}"),
        help="Directory containing per-task subdirs with zip files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("${ROBOTWIN_DATA_ROOT:-}"),
        help="Root output directory (will contain Clean/ and Randomized/ subdirs)",
    )
    parser.add_argument(
        "--task-instruction-dir",
        type=Path,
        default=Path("${ROBOTWIN_TASK_ROOT:-}"),
        help="Directory containing <task>.json instruction files",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Task names to convert, or 'all' for all 50 tasks",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["clean", "randomized"],
        default=["clean", "randomized"],
        help="Data types to convert",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without doing it")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume: always re-convert every task/type even if already complete",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tasks = ALL_TASKS if args.tasks == ["all"] else args.tasks
    invalid = [t for t in tasks if t not in ALL_TASKS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}. Valid tasks: {ALL_TASKS}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Data root:    {args.data_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Tasks ({len(tasks)}): {tasks}")
    print(f"Types: {args.types}")
    print()

    for task in tasks:
        for data_type in args.types:
            print(f"[{task}] type={data_type}")
            convert_task(
                task=task,
                data_type=data_type,
                data_root=args.data_root,
                output_root=args.output_root,
                task_instruction_dir=args.task_instruction_dir,
                dry_run=args.dry_run,
                resume=not args.no_resume,
            )
            print()


if __name__ == "__main__":
    main()
