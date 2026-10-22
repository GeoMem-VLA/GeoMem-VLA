"""
Incrementally add a dual-arm end-effector (EEF) position column to an EXISTING
RoboTwin LeRobot v2.1 dataset — WITHOUT re-encoding the videos.

Why: the original `convert_robotwin_to_lerobot.py` only extracted joint-space
actions (`joint_action/vector`, 14-D). The GeoPlan EEF-waypoint auxiliary needs
Cartesian end-effector positions, which ARE present in the raw RoboTwin HDF5 as
`endpose/left_endpose` / `endpose/right_endpose` (each (T,7) = xyz + quaternion),
but were never written to the LeRobot dataset.

This script reads those endposes from the original zips and appends a new column
`observation.eef` (shape [6] = [left_xyz(3), right_xyz(3)], float32) to every
episode parquet, then updates `meta/info.json` and `meta/episodes_stats.jsonl`.
It does NOT touch the videos.

SAFETY:
  * Alignment check: for every episode, the parquet `action` column is compared
    value-by-value against the HDF5 `joint_action/vector` (what convert stored).
    A mismatch aborts that task (catches any episode misordering).
  * Idempotent: skips a dataset that already has `observation.eef`.
  * Atomic parquet writes (tmp file + os.replace).

Usage (any env with pyarrow + h5py; e.g. base):
  python add_eef_to_lerobot.py \
    --lerobot-root /path/to/robotwin2_lerobot \
    --data-root    /path/to/RoboTwin/data/dataset \
    [--tasks adjust_bottle ...] [--types clean randomized] [--dry-run]
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

ZIP_NAMES = {"clean": "aloha-agilex_clean_50.zip", "randomized": "aloha-agilex_randomized_500.zip"}
ZIP_INNER_DIRS = {"clean": "aloha-agilex_clean_50", "randomized": "aloha-agilex_randomized_500"}
OUTPUT_SUBDIRS = {"clean": "Clean", "randomized": "Randomized"}

EEF_KEY = "observation.eef"
EEF_DIM = 6
EEF_NAMES = ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]
# tolerance for the action<->joint_action alignment check (convert stored float32)
ALIGN_ATOL = 1e-3


def read_endpose_and_joint(ep_bytes: bytes):
    """Return (eef[T,6] float32, joint_vector[T,14] float32) from one HDF5 episode."""
    with h5py.File(io.BytesIO(ep_bytes), "r") as ep:
        left = ep["endpose/left_endpose"][:, :3]     # (T,3) xyz
        right = ep["endpose/right_endpose"][:, :3]    # (T,3) xyz
        joint = ep["joint_action/vector"][:]          # (T,14)
    eef = np.concatenate([left, right], axis=1).astype(np.float32)   # (T,6)
    return eef, joint.astype(np.float32)


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


def process_task(task: str, data_type: str, lerobot_root: Path, data_root: Path, dry_run: bool) -> None:
    subdir = OUTPUT_SUBDIRS[data_type]
    ds_dir = lerobot_root / subdir / task
    zip_path = data_root / task / ZIP_NAMES[data_type]
    inner = ZIP_INNER_DIRS[data_type]

    if not ds_dir.exists():
        print(f"  [SKIP] lerobot dataset not found: {ds_dir}")
        return
    if not zip_path.exists():
        print(f"  [SKIP] zip not found: {zip_path}")
        return

    info_path = ds_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    if EEF_KEY in info.get("features", {}):
        print(f"  [SKIP] already has {EEF_KEY}: {ds_dir}")
        return

    data_path_tmpl = info["data_path"]      # data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    chunks_size = int(info.get("chunks_size", 1000))
    total_eps = int(info["total_episodes"])

    with zipfile.ZipFile(zip_path, "r") as zf:
        hdf5_names = sorted(
            n for n in zf.namelist()
            if n.startswith(f"{inner}/data/") and n.endswith(".hdf5")
        )
        if len(hdf5_names) != total_eps:
            print(f"  [ERROR] episode count mismatch: zip has {len(hdf5_names)}, "
                  f"lerobot info says {total_eps}. Aborting this task.")
            return

        # Pass 1: read all endposes + verify alignment against parquet action, in-memory stage.
        pending = []  # (episode_index, parquet_path, eef[T,6])
        for i, hname in enumerate(hdf5_names):
            eef, joint = read_endpose_and_joint(zf.open(hname).read())
            chunk = i // chunks_size
            pq_path = ds_dir / data_path_tmpl.format(episode_chunk=chunk, episode_index=i)
            if not pq_path.exists():
                print(f"  [ERROR] parquet missing: {pq_path}. Aborting task.")
                return
            tbl = pq.read_table(pq_path)
            if tbl.num_rows != eef.shape[0]:
                print(f"  [ERROR] ep{i}: parquet rows={tbl.num_rows} != endpose frames={eef.shape[0]}. "
                      f"Aborting task (possible misalignment).")
                return
            # STRONG alignment check: parquet 'action' must equal hdf5 joint_action/vector.
            act = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)  # [T,14]
            if act.shape != joint.shape or not np.allclose(act, joint, atol=ALIGN_ATOL):
                maxdiff = float(np.abs(act - joint).max()) if act.shape == joint.shape else float("nan")
                print(f"  [ERROR] ep{i}: parquet action != hdf5 joint_action (max|diff|={maxdiff}). "
                      f"Episode ordering mismatch — aborting task to avoid corrupting data.")
                return
            pending.append((i, pq_path, eef, tbl))

        if dry_run:
            print(f"  [DRY RUN] {task}/{data_type}: {len(pending)} episodes verified & aligned; "
                  f"would append '{EEF_KEY}' [6] + update meta.")
            return

        # Pass 2: append column + write atomically; accumulate stats.
        stats_by_ep = {}
        for i, pq_path, eef, tbl in pending:
            # Resume-safety: if a previous interrupted run already appended the column to this
            # parquet, drop it first so we never end up with a duplicate observation.eef column.
            if EEF_KEY in tbl.column_names:
                tbl = tbl.remove_column(tbl.column_names.index(EEF_KEY))
            eef_arr = make_fixed_size_list(eef, EEF_DIM)
            new_tbl = tbl.append_column(EEF_KEY, eef_arr)
            tmp_path = pq_path.with_suffix(".parquet.tmp")
            pq.write_table(new_tbl, tmp_path)
            os.replace(tmp_path, pq_path)
            stats_by_ep[i] = eef_stats(eef)

    # Update meta/info.json features
    info["features"][EEF_KEY] = {
        "dtype": "float32",
        "shape": [EEF_DIM],
        "names": [EEF_NAMES],
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # Update meta/episodes_stats.jsonl (add observation.eef stats to each episode line)
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
    p = argparse.ArgumentParser(description="Incrementally add observation.eef to RoboTwin LeRobot dataset")
    p.add_argument("--lerobot-root", type=Path,
                   default=Path("${ROBOTWIN_DATA_ROOT:-}"))
    p.add_argument("--data-root", type=Path,
                   default=Path("${ROBOTWIN_DATA_ROOT:-}"))
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

    print(f"LeRobot root: {args.lerobot_root}")
    print(f"Data root:    {args.data_root}")
    print(f"Tasks ({len(tasks)}), types={args.types}, dry_run={args.dry_run}\n")

    for task in tasks:
        for data_type in args.types:
            print(f"[{task}] type={data_type}")
            process_task(task, data_type, args.lerobot_root, args.data_root, args.dry_run)
            print()


if __name__ == "__main__":
    main()
