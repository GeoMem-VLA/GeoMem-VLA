"""
Add a dual-arm end-effector (EEF) position column to a RoboTwin LeRobot v2.1
dataset by FORWARD KINEMATICS from the per-frame joint state — WITHOUT touching
the videos and WITHOUT needing the raw HDF5 zips.

Why this exists (vs add_eef_to_lerobot.py):
  add_eef_to_lerobot.py copies the sim `endpose/left|right_endpose` out of the raw
  RoboTwin HDF5 zips, matching episode-for-episode by frame count + action equality.
  That ONLY works when the LeRobot dataset was converted from exactly those zips.
  The officially-released RoboTwin LeRobot download (robot_type=aloha) is a DIFFERENT
  capture: its episodes do not line up with our local zips (different frame counts),
  and it ships joint-space `observation.state`/`action` only — no endpose column.

  This script instead computes the EEF xyz from each frame's OWN `observation.state`
  (14-D: [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]) via
  forward kinematics of the aloha-agilex URDF (links fl_link6 / fr_link6). Because it
  reads the dataset's own joints, the result is ALWAYS frame-aligned.

Frame note:
  FK returns each arm's flange (link6) position in the URDF root frame. The GeoPlan
  EEF aux uses `eef_target_mode=endpose_rel` (each arm made relative to its own first
  future frame), so a constant base/world offset is irrelevant — only the per-arm
  relative trajectory matters. Validated against robotwin2_lerobot's real sim endpose:
  the moving arm matches to a few mm (up to a rigid transform); the flange-vs-TCP
  offset is a constant in the EE frame and does not affect the relative target.

Output column (identical schema to add_eef_to_lerobot.py so the dataloader is unchanged):
  observation.eef : float32, shape [6] = [left_xyz(3), right_xyz(3)]

Requires: numpy, pyarrow, yourdfpy  (all present in the RoboTwin conda env).
Idempotent (skips datasets that already have observation.eef) and atomic (tmp + replace).

Usage:
  <RoboTwin_env>/bin/python add_eef_via_fk.py \
    --lerobot-root /path/to/RoboTwin_lerobot \
    [--urdf /path/to/aloha-agilex/urdf/arx5_description_isaac.urdf] \
    [--tasks turn_switch ...] [--types clean randomized] [--dry-run]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yourdfpy

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

OUTPUT_SUBDIRS = {"clean": "Clean", "randomized": "Randomized"}

EEF_KEY = "observation.eef"
EEF_DIM = 6
EEF_NAMES = ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]

# aloha-agilex arm end-effector links and their driving joints (from config.yml:
# move_group=[fl_link6, fr_link6], arm_joints_name=[[fl_joint1..6],[fr_joint1..6]]).
LEFT_EE, RIGHT_EE = "fl_link6", "fr_link6"
# observation.state / action layout (from meta/modality.json, identical for all tasks):
#   [0:6] left_joints, [6] left_gripper, [7:13] right_joints, [13] right_gripper
LEFT_JOINT_COLS = list(range(0, 6))
RIGHT_JOINT_COLS = list(range(7, 13))

DEFAULT_URDF = os.environ.get(
    "ROBOTWIN_URDF",
    "RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
)


# ---------------------------------------------------------------------------
# Forward kinematics (vectorized over time; verified == yourdfpy to 4e-16)
# ---------------------------------------------------------------------------
class ArmFK:
    """Serial-chain FK from the URDF root to one end-effector link, vectorized over
    a batch of joint configurations. Chain transforms/axes are extracted from
    yourdfpy so we never hand-parse the URDF."""

    def __init__(self, urdf: "yourdfpy.URDF", ee_link: str):
        child2joint = {j.child: j for j in urdf.joint_map.values()}
        seq = []
        link = ee_link
        while link in child2joint:
            seq.append(child2joint[link])
            link = child2joint[link].parent
        seq = seq[::-1]  # root -> ee order
        self.elements = []  # (origin[4,4], axis[3] | None for fixed)
        self.n_revolute = 0
        for j in seq:
            origin = np.array(j.origin, dtype=np.float64).reshape(4, 4)
            if j.type == "fixed":
                axis = None
            else:
                a = np.array(j.axis, dtype=np.float64)
                axis = a / np.linalg.norm(a)
                self.n_revolute += 1
            self.elements.append((origin, axis))

    @staticmethod
    def _rodrigues(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotation matrices [T,3,3] for angles q about a fixed unit `axis`."""
        x, y, z = axis
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        I = np.eye(3)[None]
        s = np.sin(q)[:, None, None]
        c = np.cos(q)[:, None, None]
        return I + s * K[None] + (1.0 - c) * (K @ K)[None]

    def __call__(self, joints: np.ndarray) -> np.ndarray:
        """joints: [T, n_revolute] (radians, chain order) -> ee xyz [T, 3]."""
        T = joints.shape[0]
        acc = np.tile(np.eye(4), (T, 1, 1))
        qi = 0
        for origin, axis in self.elements:
            local = np.tile(origin, (T, 1, 1))
            if axis is not None:
                M = np.tile(np.eye(4), (T, 1, 1))
                M[:, :3, :3] = self._rodrigues(axis, joints[:, qi])
                local = local @ M
                qi += 1
            acc = acc @ local
        return acc[:, :3, 3].astype(np.float32)


def make_fixed_size_list(arr2d: np.ndarray, dim: int) -> pa.Array:
    """np [T, dim] float32 -> pyarrow fixed_size_list<float>[dim] (matches observation.state)."""
    flat = pa.array(arr2d.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def eef_stats(eef: np.ndarray) -> dict:
    """Per-episode stats matching the episodes_stats.jsonl format."""
    return {
        "min": eef.min(axis=0).tolist(),
        "max": eef.max(axis=0).tolist(),
        "mean": eef.mean(axis=0).tolist(),
        "std": eef.std(axis=0).tolist(),
        "count": [int(eef.shape[0])],
    }


def compute_eef(state: np.ndarray, fk_left: ArmFK, fk_right: ArmFK) -> np.ndarray:
    """state [T,14] -> eef [T,6] = [left_xyz, right_xyz] via FK on the joint columns."""
    left = fk_left(state[:, LEFT_JOINT_COLS].astype(np.float64))
    right = fk_right(state[:, RIGHT_JOINT_COLS].astype(np.float64))
    return np.concatenate([left, right], axis=1).astype(np.float32)


def process_task(task: str, data_type: str, lerobot_root: Path,
                 fk_left: ArmFK, fk_right: ArmFK, dry_run: bool) -> None:
    ds_dir = lerobot_root / OUTPUT_SUBDIRS[data_type] / task
    if not ds_dir.exists():
        print(f"  [SKIP] lerobot dataset not found: {ds_dir}")
        return

    info_path = ds_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    if EEF_KEY in info.get("features", {}):
        print(f"  [SKIP] already has {EEF_KEY}: {ds_dir}")
        return

    data_path_tmpl = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))
    total_eps = int(info["total_episodes"])

    # Pass 1: read state + compute eef in memory (also verifies every parquet exists).
    pending = []  # (episode_index, parquet_path, table, eef[T,6])
    for i in range(total_eps):
        chunk = i // chunks_size
        pq_path = ds_dir / data_path_tmpl.format(episode_chunk=chunk, episode_index=i)
        if not pq_path.exists():
            print(f"  [ERROR] parquet missing: {pq_path}. Aborting task.")
            return
        tbl = pq.read_table(pq_path)
        state = np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 14:
            print(f"  [ERROR] ep{i}: unexpected observation.state shape {state.shape}. Aborting task.")
            return
        eef = compute_eef(state, fk_left, fk_right)
        pending.append((i, pq_path, tbl, eef))

    if dry_run:
        sample = pending[0][3]
        print(f"  [DRY RUN] {task}/{data_type}: {len(pending)} episodes; would append '{EEF_KEY}' [6]. "
              f"ep0 eef[0]={np.round(sample[0], 4).tolist()}")
        return

    # Pass 2: append column + atomic write; accumulate stats.
    stats_by_ep = {}
    for i, pq_path, tbl, eef in pending:
        if EEF_KEY in tbl.column_names:  # resume-safety: drop a partial prior append
            tbl = tbl.remove_column(tbl.column_names.index(EEF_KEY))
        new_tbl = tbl.append_column(EEF_KEY, make_fixed_size_list(eef, EEF_DIM))
        tmp_path = pq_path.with_suffix(".parquet.tmp")
        pq.write_table(new_tbl, tmp_path)
        os.replace(tmp_path, pq_path)
        stats_by_ep[i] = eef_stats(eef)

    info["features"][EEF_KEY] = {"dtype": "float32", "shape": [EEF_DIM], "names": [EEF_NAMES]}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    stats_path = ds_dir / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        lines = []
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep_idx = int(rec["episode_index"])
                if ep_idx in stats_by_ep:
                    rec.setdefault("stats", {})[EEF_KEY] = stats_by_ep[ep_idx]
                lines.append(rec)
        tmp = stats_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, stats_path)

    print(f"  [OK] {task}/{data_type}: appended {EEF_KEY} to {len(stats_by_ep)} episodes + updated meta.")


def parse_args():
    p = argparse.ArgumentParser(description="Add observation.eef to a RoboTwin LeRobot dataset via forward kinematics")
    p.add_argument("--lerobot-root", type=Path,
                   default=Path(os.environ.get("ROBOTWIN_LEROBOT_ROOT", "RoboTwin_lerobot")))
    p.add_argument("--urdf", type=Path, default=Path(DEFAULT_URDF))
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--types", nargs="+", choices=["clean", "randomized"], default=["clean", "randomized"])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = ALL_TASKS if args.tasks == ["all"] else args.tasks
    invalid = [t for t in tasks if t not in ALL_TASKS]
    if invalid:
        raise ValueError(f"Unknown tasks: {invalid}")

    print(f"URDF:         {args.urdf}")
    print(f"LeRobot root: {args.lerobot_root}")
    print(f"Tasks ({len(tasks)}), types={args.types}, dry_run={args.dry_run}\n")

    urdf = yourdfpy.URDF.load(str(args.urdf), load_meshes=False, build_scene_graph=True)
    fk_left = ArmFK(urdf, LEFT_EE)
    fk_right = ArmFK(urdf, RIGHT_EE)
    if fk_left.n_revolute != 6 or fk_right.n_revolute != 6:
        raise RuntimeError(
            f"Expected 6 revolute joints per arm, got L={fk_left.n_revolute} R={fk_right.n_revolute}. "
            f"Check the URDF / end-effector link names."
        )

    for task in tasks:
        for data_type in args.types:
            print(f"[{task}] type={data_type}")
            process_task(task, data_type, args.lerobot_root, fk_left, fk_right, args.dry_run)
            print()


if __name__ == "__main__":
    main()
