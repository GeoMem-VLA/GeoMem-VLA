# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
import os
import hashlib
import io
import json, torch
import copy
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import os, random
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform
from starVLA.dataloader.task_sampling import (
    config_mapping,
    parse_task_sampling_weights,
    task_weight_multipliers,
)

from functools import partial
from typing import Tuple, List
import pickle
import gc
import time

# LeRobot v2.0 dataset file names 
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_STATS_FORMAT_VERSION = 3
EPSILON = 5e-4

#  LeRobot v3.0 dataset file names 
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)

    if not all_low_dim_data_list:
        raise FileNotFoundError(
            f"No parquet files found under the provided paths: {[str(p) for p in parquet_paths[:3]]}..."
            f" — make sure the dataset has been downloaded/converted before training."
        )

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        try:
            np_data = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
            )
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue  

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def _normalize_action_mode(mode: str) -> str:
    """Normalize action mode names to {abs, delta, rel}.""" 
    # TODO: move this out of the dataloader so it stays independent of the action-mode logic
    # (e.g. into a transform or a separate lerobot utility).
    mode = str(mode).lower()
    if mode in {"absolute", "raw"}:
        mode = "abs"
    if mode not in {"abs", "delta", "rel"}:
        mode = "abs"
    return mode


def _normalize_action_mode_apply_keys(
    action_mode_apply_keys: Sequence[str] | None,
    fallback_keys: Sequence[str] | None = None,
) -> list[str]:
    source_keys = action_mode_apply_keys if action_mode_apply_keys else (fallback_keys or [])
    normalized = []
    for key in source_keys:
        key = str(key)
        if not key.startswith("action."):
            key = f"action.{key}"
        normalized.append(key)
    return normalized


def _normalize_action_mode_state_map(action_mode_state_map: dict[str, str] | None) -> dict[str, str]:
    normalized = {}
    for action_key, state_key in (action_mode_state_map or {}).items():
        action_key = str(action_key)
        state_key = str(state_key)
        if not action_key.startswith("action."):
            action_key = f"action.{action_key}"
        if not state_key.startswith("state."):
            state_key = f"state.{state_key}"
        normalized[action_key] = state_key
    return normalized


def _build_stats_cache_config(
    action_mode: str,
    parquet_paths: Sequence[Path],
    *,
    action_keys_full: Sequence[str],
    state_keys_full: Sequence[str],
    action_indices: Sequence[int] | None,
    state_indices: Sequence[int] | None,
    action_mode_apply_keys: Sequence[str],
    action_mode_state_map: dict[str, str],
) -> dict:
    """Describe both the transform mode and the exact parquet set behind a stats cache."""
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(Path(p) for p in parquet_paths):
        stat = path.stat()
        total_bytes += stat.st_size
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return {
        "mode": action_mode,
        "num_parquet_files": len(parquet_paths),
        "total_parquet_bytes": total_bytes,
        "dataset_fingerprint": digest.hexdigest(),
        "action_keys": list(action_keys_full),
        "state_keys": list(state_keys_full),
        "action_indices": list(action_indices) if action_indices is not None else None,
        "state_indices": list(state_indices) if state_indices is not None else None,
        "action_mode_apply_keys": list(action_mode_apply_keys),
        "action_mode_state_map": dict(sorted(action_mode_state_map.items())),
    }


def _invalidate_legacy_stats_cache(stats_path: Path, reason: str) -> None:
    if not stats_path.exists():
        return
    print(f"Removing stale dataset statistics cache at {stats_path}: {reason}")
    stats_path.unlink()


def _load_stats_cache(
    stats_path: Path,
    expected_config: dict,
    *,
    invalidate_legacy: bool,
) -> dict | None:
    if not stats_path.exists():
        return None

    try:
        with open(stats_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, f"failed to load JSON ({exc})")
        return None

    if not isinstance(payload, dict):
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "unexpected top-level format")
        return None

    format_version = payload.get("__format_version")
    cache_config = payload.get("__cache_config")
    statistics = payload.get("statistics")
    if format_version != LE_ROBOT_STATS_FORMAT_VERSION or cache_config is None or statistics is None:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "legacy statistics format detected")
        return None

    if cache_config != expected_config:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "statistics config mismatch, rebuilding cache")
        return None

    return statistics


def _save_stats_cache(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    payload = {
        "__format_version": LE_ROBOT_STATS_FORMAT_VERSION,
        "__cache_config": cache_config,
        "statistics": statistics,
    }
    tmp_path = stats_path.with_suffix(".tmp")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, stats_path)


def _compute_statistics_for_mode(
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[RANK 0] Calculating dataset statistics for {dataset_name} (mode={action_mode})")

    base_stats = calculate_dataset_statistics(parquet_paths)
    
    if action_mode == "abs":
        return base_stats

    if action_indices is None or state_indices is None:
        raise ValueError(
            "Both action and state modalities are required to compute "
            f"{action_mode} action mode statistics."
        )

    if action_mode == "delta":
        return calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    if action_mode == "rel":
        return calculate_rel_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    raise ValueError(f"Unsupported action mode for statistics: {action_mode}")


def _load_or_compute_statistics(
    stats_path: Path,
    stats_cache_config: dict,
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    le_statistics = _load_stats_cache(
        stats_path,
        stats_cache_config,
        invalidate_legacy=True,
    )
    if le_statistics is not None:
        return le_statistics

    le_statistics = _compute_statistics_for_mode(
        parquet_paths=parquet_paths,
        dataset_name=dataset_name,
        action_mode=action_mode,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_mode_apply_keys,
        action_mode_state_map=action_mode_state_map,
    )
    _save_stats_cache(stats_path, stats_cache_config, le_statistics)
    return le_statistics


def _get_action_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[tuple[int, int], str, tuple[int, int], str, str]]]:
    apply_keys = _normalize_action_mode_apply_keys(action_mode_apply_keys, action_keys_full)
    action_mode_state_map = _normalize_action_mode_state_map(action_mode_state_map)

    action_meta = lerobot_modality_meta.action
    state_meta = lerobot_modality_meta.state

    # Build per-column mapping: action column -> list of (action_slice, state_column, state_slice)
    action_col_slices: dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")
        state_key = action_mode_state_map.get(action_key, action_key.replace("action.", "state.", 1))
        if state_key not in state_keys_full:
            raise ValueError(
                f"State key {state_key} not found for action key {action_key}. "
                f"Add it to action_mode_state_map or remove {action_key} from action_mode_apply_keys."
            )

        action_subkey = action_key.replace("action.", "", 1)
        state_subkey = state_key.replace("state.", "", 1)
        if action_subkey not in action_meta or state_subkey not in state_meta:
            raise ValueError(f"Action/state key missing in metadata: {action_key} -> {state_key}")

        action_cfg = action_meta[action_subkey]
        state_cfg = state_meta[state_subkey]
        action_col = action_cfg.original_key or action_subkey
        state_col = state_cfg.original_key or state_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_slice = (state_cfg.start, state_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        state_padding = "first_last" if state_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            (action_slice, state_col, state_slice, action_padding, state_padding)
        )

    return action_col_slices


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using delta mode.

    Rule:
      - For t>0: a_t - a_{t-1}
      - For t=0: a_0 - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting delta action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk.copy()
                    if len(out) > 1:
                        out[1:] = action_part_chunk[1:] - action_part_chunk[:-1]
                    out[0] = action_part_chunk[0] - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        delta_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using rel mode.

    Rule:
      - For all t: a_t - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting rel action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        rel_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return rel_stats

class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg = None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict letobot version
        self._lerobot_version =  self.data_cfg.get("lerobot_version", "v2.0") #self._indict_lerobot_version(**kwargs)

        self._action_mode = None
        self._action_mode_state_map = {}
        self._action_mode_apply_keys = None

        # ---- optional rule-based keyframe labels (event_prior for the event-guided memory bank) ----
        # Off unless data_cfg enables it, so every existing dataset/checkpoint is untouched.
        self._keyframe_cfg = None
        self._keyframe_cache_id = None
        self._keyframe_cache = None
        self._keyframe_len = 0
        self._keyframe_emit_aux = False
        kf_cfg = (self.data_cfg or {}).get("keyframe_event_prior", None)
        if kf_cfg not in (None, False, "False"):
            from starVLA.dataloader.keyframe_labeler import KeyframeConfig

            kf_cfg = {} if kf_cfg is True else dict(kf_cfg)
            self._keyframe_state_col = kf_cfg.pop("state_column", "observation.state")
            self._keyframe_action_col = kf_cfg.pop("action_column", "action")
            # Separate Cartesian end-effector column, when the position does not live in `state`.
            # RoboTwin's state is 14-D JOINT ANGLES and keeps eef xyz in `observation.eef`;
            # differencing joint angles would measure joint motion, not end-effector pauses.
            self._keyframe_pos_col = kf_cfg.pop("pos_column", None)
            # emit_aux also emits the 9-D backward-difference event signal consumed by the scorer's
            # aux head; it must be paired with framework.geoplan.event_aux_dim = 9.
            self._keyframe_emit_aux = bool(kf_cfg.pop("emit_aux", False))
            # YAML gives lists; KeyframeConfig indexes with them, so normalise to tuples.
            for _k in ("eef_cols", "rot_cols", "gripper_cols"):
                if kf_cfg.get(_k) is not None:
                    kf_cfg[_k] = tuple(kf_cfg[_k])
            self._keyframe_cfg = KeyframeConfig(**kf_cfg)

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self._init_action_mode()
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_tasks: list[tuple[str, ...]] = []
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def trajectory_tasks(self) -> list[tuple[str, ...]]:
        """Natural-language task annotations aligned with ``trajectory_ids``."""

        return self._trajectory_tasks

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                try:
                    channels = le_video_meta["info"]["video.channels"]
                    fps = le_video_meta["info"]["video.fps"]
                except (ValueError, KeyError):
                    # Fallback for image-only datasets (e.g. VLA-Arena) that lack video_info
                    channels = 3
                    fps = le_info.get("fps", 30)
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }


        # 2. Dataset statistics
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0
        
        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")

        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        action_cfg = self.modality_configs.get("action")
        state_cfg = self.modality_configs.get("state")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        state_keys_full = list(state_cfg.modality_keys) if state_cfg else []
        action_indices = list(action_cfg.delta_indices) if action_cfg else None
        state_indices = list(state_cfg.delta_indices) if state_cfg else None

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        parquet_files_filtered = [
            pf for pf in parquet_files if "episode_033675.parquet" not in pf.name
        ]
        stats_cache_config = _build_stats_cache_config(
            action_mode=action_mode,
            parquet_paths=parquet_files_filtered,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=apply_keys,
            action_mode_state_map=normalized_state_map,
        )

        if is_main():
            le_statistics = _load_or_compute_statistics(
                stats_path,
                stats_cache_config=stats_cache_config,
                parquet_paths=parquet_files_filtered,
                dataset_name=self.dataset_name,
                action_mode=action_mode,
                lerobot_modality_meta=le_modality_meta,
                action_keys_full=action_keys_full,
                state_keys_full=state_keys_full,
                action_indices=action_indices,
                state_indices=state_indices,
                action_mode_apply_keys=apply_keys,
                action_mode_state_map=normalized_state_map,
            )
        else:
            le_statistics = None

        if dist.is_initialized():
            dist.barrier()

        if le_statistics is None:
            le_statistics = _load_stats_cache(
                stats_path,
                stats_cache_config,
                invalidate_legacy=False,
            )
            if le_statistics is None:
                raise RuntimeError(
                    f"Dataset statistics cache is missing or invalid after sync: {stats_path}"
                )

        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)


        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            trajectory_tasks: list[tuple[str, ...]] = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
                raw_tasks = episode.get("tasks", [])
                if isinstance(raw_tasks, str):
                    raw_tasks = [raw_tasks]
                trajectory_tasks.append(tuple(str(task) for task in raw_tasks))
            self._trajectory_tasks = trajectory_tasks
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            trajectory_tasks: list[tuple[str, ...]] = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])
                    raw_tasks = episode.get("tasks", [])
                    if isinstance(raw_tasks, str):
                        raw_tasks = [raw_tasks]
                    elif raw_tasks is None or (
                        isinstance(raw_tasks, float) and np.isnan(raw_tasks)
                    ):
                        raw_tasks = []
                    trajectory_tasks.append(tuple(str(task) for task in raw_tasks))

                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        # videos/{video_key}/from_timestamp -> {video_key}
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)

                    # TODO auto map key 
                    # Collect video file indices for each video key
                    #已修改的lerobotv3.0的视频索引（提取视频和文件的索引）
                    video_file_indices = {}
                    for col in timestamp_cols:
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        if chunk_col in episode and file_col in episode:
                            video_file_indices[video_key] = {
                                "chunk_index": int(episode[chunk_col]),
                                "file_index": int(episode[file_col]),
                            }
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                        "videos/file_indices": video_file_indices,
                    }
                    # episode_meta = {
                    #     "data/chunk_index": episode["data/chunk_index"],
                    #     "data/file_index": episode["data/file_index"],
                    #     "data/file_from_index": index,
                    #     "videos/from_timestamps": from_timestamps,
                    # }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            self._trajectory_tasks = trajectory_tasks
            # Should be able to directly read the saved index info here
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0
    
        config_key = self._get_steps_config_key()
        steps_filename = "steps_data_index.pkl"
        steps_path = self.dataset_path / "meta" / steps_filename
    
        # ---------- try to read from cache  ----------
        if steps_path.exists():
            try:
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                if cached_data.get("config_key") != config_key:
                    raise ValueError("dataset/config fingerprint mismatch")
                if cached_data.get("num_trajectories") != len(self.trajectory_ids):
                    raise ValueError("trajectory count mismatch")
                steps = cached_data["steps"]
                if cached_data.get("total_steps") != len(steps):
                    raise ValueError("cached step count mismatch")
                return steps
            except Exception as e:
                # include EOFError / PickleError / KeyError
                print(
                    f"[RANK {os.environ.get('RANK', 'NA')}] "
                    f"Failed to load cached steps ({e}), will rebuild."
                )
    
        # ---------- only build by rank0  ----------
        if is_main():
            all_steps = self._get_all_steps_single_process()
    
            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
            }
    
            steps_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = steps_path.with_suffix(".tmp")
    
            with open(tmp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, steps_path)
    
            print(f"[RANK 0] Cached steps saved to {steps_path}")
    
        # ---------- sync after rank0  ----------
        if dist.is_initialized():
            dist.barrier()
    
        # ---------- read by all rank ----------
        with open(steps_path, "rb") as f:
            cached_data = pickle.load(f)
        if cached_data.get("config_key") != config_key:
            raise RuntimeError(f"Rebuilt steps cache has a stale fingerprint: {steps_path}")
        return cached_data["steps"]

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        digest = hashlib.sha256()
        digest.update(str(bool(self.delete_pause_frame)).encode("ascii"))
        digest.update(str(self.dataset_name).encode("utf-8"))
        language_keys = sorted(self.modality_keys.get("language", []))
        digest.update(json.dumps(language_keys, sort_keys=True).encode("utf-8"))
        for metadata_name in (LE_ROBOT_TASKS_FILENAME, LE_ROBOT_EPISODE_FILENAME):
            metadata_path = self.dataset_path / metadata_name
            if metadata_path.is_file():
                stat = metadata_path.stat()
                digest.update(metadata_name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
        trajectory_ids = np.asarray(self.trajectory_ids, dtype=np.int64)
        trajectory_lengths = np.asarray(self.trajectory_lengths, dtype=np.int64)
        digest.update(trajectory_ids.tobytes())
        digest.update(trajectory_lengths.tobytes())
        return digest.hexdigest()[:24]


    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        # LeRobot v3 episode metadata already carries the task text and length.
        # Reading the same multi-episode Parquet file once per episode to check
        # language would rescan RoboDojo's 1.86M frames thousands of times.
        if (
            self._lerobot_version == "v3.0"
            and len(self._trajectory_tasks) == len(self.trajectory_ids)
            and all(tasks and all(str(task).strip() for task in tasks)
                    for tasks in self._trajectory_tasks)
        ):
            return [
                (int(trajectory_id), base_index)
                for trajectory_id, length in zip(self.trajectory_ids, self.trajectory_lengths)
                for base_index in range(int(length))
            ]
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0
        
        # Check if language modality is configured
        has_language_modality = 'language' in self.modality_keys and len(self.modality_keys['language']) > 0
        # TODO why trajectory_length here, why not use data length?
        for trajectory_id, trajectory_length in tqdm(zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"):
            try:
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                elif self._lerobot_version == "v3.0":
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)
                
                trajectory_skipped = False
            
                # Check if trajectory has valid language instruction (if language modality is configured)
                if has_language_modality:
                    self.curr_traj_data = data  # Set current trajectory data for get_language to work

                    language_instruction = self.get_language(trajectory_id, self.modality_keys['language'][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        skipped_trajectories += 1
                        trajectory_skipped = True
                        continue

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                trajectory_skipped = True
                continue
        
            if not trajectory_skipped:
                processed_trajectories += 1
        
            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))
                
        # Print summary statistics
        print(f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories")
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")
                   
        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])
        
        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']
        
        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue
        
        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue
            
            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()
        
        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy() 
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")
        
        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']
        
        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue
        
        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")
        
        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode

        apply_keys = _normalize_action_mode_apply_keys(self.data_cfg.get("action_mode_apply_keys", None))
        if apply_keys:
            self._action_mode_apply_keys = apply_keys

        self._action_mode_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) or {}
        )

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]

        if not action_key.startswith("action."):
            return None
        base = action_key.replace("action.", "", 1)
        if f"state.{base}" in self.modality_keys.get("state", []):
            return f"state.{base}"
        return None

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            if action_key not in data:
                print(f"[WARNING] Action key {action_key} not found in data")
                continue
            state_key = self._infer_state_key_for_action(action_key)

            # for safety, check if the state key is valid
            if state_key is None or state_key not in data:
                continue

            action_values = np.asarray(data[action_key])
            state_values = np.asarray(data[state_key])
            if action_values.ndim != 2 or state_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for action/state, got {action_key}: {action_values.shape}, {state_key}: {state_values.shape}"
                )
            if action_values.shape[1] != state_values.shape[1]:
                raise ValueError(
                    f"Action/state dim mismatch for {action_key} vs {state_key}: {action_values.shape} vs {state_values.shape}"
                )

            state0 = state_values[0]
            if self._action_mode == "delta":
                out = action_values.copy()
                if len(out) > 1:
                    out[1:] = action_values[1:] - action_values[:-1]
                out[0] = action_values[0] - state0
            elif self._action_mode == "rel":
                out = action_values - state0
            else:
                out = action_values

            data[action_key] = out

        return data

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return df.set_index("task_index")
        
        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # convert index to a column, typically named 'index'
            df = df.rename(columns={'index': 'task'})  # rename 'index' column to 'task'
            df = df[['task_index', 'task']]  # reorder columns
            return df
    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions" or key.startswith("eef."):
                    continue  # lapa/dream come normalized; eef is a self-contained modality read
                    # directly from the observation.eef column (no modality.json metadata needed).
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        sample = self._pack_sample(data)
        return self._add_episode_fields(sample, trajectory_id, base_index)

    def _add_episode_fields(
        self,
        sample: dict,
        trajectory_id: int,
        base_index: int,
        *,
        episode_id: int | None = None,
    ) -> dict:
        """Attach memory metadata and optional event labels to a packed sample.

        Training normally reads :class:`LeRobotSingleDataset` through
        :class:`LeRobotMixtureDataset`, whose ``__getitem__`` packs samples itself instead of
        calling this class' ``__getitem__``. Keeping the injection in one helper is therefore
        important: both direct-dataset and mixture/group sampling must call the same path or the
        configured BCE labels silently disappear from real training batches.
        """
        t = int(base_index)
        sample["episode_id"] = int(trajectory_id) if episode_id is None else int(episode_id)
        sample["frame_index"] = t
        if self._keyframe_cfg is not None:
            prior, aux = self._episode_keyframes(trajectory_id)
            if not 0 <= t < len(prior):
                raise IndexError(
                    f"frame_index={t} is outside keyframe labels [0, {len(prior)}) "
                    f"for trajectory_id={trajectory_id}"
                )
            sample["event_prior"] = float(prior[t])
            if self._keyframe_emit_aux:
                sample["event_aux"] = aux[t]
        return sample

    def _episode_keyframes(self, trajectory_id: int):
        """Rule-based keyframe mask for one episode, cached per trajectory.

        Returns ``(event_prior[T], event_aux[T, 9])`` for the event-guided memory bank. The prior is
        an offline teacher label; the auxiliary vector is causal and uses only current proprio plus
        the previously executed gripper command, so it can also be supplied online.
        """
        if self._keyframe_cache_id == trajectory_id and self._keyframe_cache is not None:
            return self._keyframe_cache

        from starVLA.dataloader.keyframe_labeler import compute_event_aux, compute_keyframes

        episode = self.get_trajectory_data(trajectory_id)
        state = np.stack(episode[self._keyframe_state_col].values)
        action = np.stack(episode[self._keyframe_action_col].values)
        pos = (
            np.stack(episode[self._keyframe_pos_col].values)
            if self._keyframe_pos_col is not None
            else None
        )
        prior = compute_keyframes(state, action, self._keyframe_cfg, pos=pos).prior.astype(np.float32)
        aux = (
            compute_event_aux(state, action, self._keyframe_cfg, pos=pos)
            if self._keyframe_emit_aux
            else np.zeros((len(prior), 0), dtype=np.float32)
        )

        self._keyframe_cache_id = trajectory_id
        self._keyframe_cache = (prior, aux)
        self._keyframe_len = len(prior)
        return self._keyframe_cache

    def _pack_sample(self, data: dict) -> dict:
        """Pack transformed modality data into training sample format."""
        step_images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]
            image = Image.fromarray(image).resize((224, 224))
            step_images.append(image)

        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "image": step_images,
            "lang": language,
            "robot_tag": self.tag
        }

        # Optional end-effector endpose sequence (future horizon) for the GeoPlan EEF aux target.
        # Only present when the DataConfig declares an "eef" modality AND the dataset has the column.
        # "eef.campose" (camera intrinsic/extrinsic + orig HW; see add_campose_to_lerobot.py) is
        # packed SEPARATELY as sample["campose"] so sample["eef"] stays pure endpose xyz.
        eef_keys = self.modality_keys.get("eef", [])
        if eef_keys:
            eef = [data[k] for k in eef_keys if k in data and k != "eef.campose"]
            if eef:
                sample["eef"] = np.concatenate(eef, axis=1).astype(np.float32)
            if "eef.campose" in eef_keys and "eef.campose" in data:
                sample["campose"] = np.asarray(data["eef.campose"]).astype(np.float32)

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                state.append(data[state_key])
            if not state:
                import warnings
                warnings.warn(
                    "include_state=True but no state modality keys found in modality_configs "
                    "(state modality may be disabled in the dataset's DataConfig). "
                    "Skipping state packing.",
                    stacklevel=2,
                )
            else:
                state = np.concatenate(state, axis=1).astype(np.float16)
                sample["state"] = state

        return sample

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities # just for action base data
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        self.curr_traj_id = trajectory_id
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data = self._apply_action_mode(data)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":
        
            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                return pd.read_parquet(parquet_path)
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)
    
    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Read only this episode's row groups when their statistics prove isolation.

        Some v3 exports pack hundreds of episodes into each Parquet. Cache a file's row-group
        index, but keep the original full-file filter when any group mixes episodes or lacks
        usable statistics. Null-containing files also retain that path: reading only non-null
        groups can change pandas' inferred integer/bool dtype. Preserve index and physical order.
        """
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else: #TODO check detail later
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            chunk_index = episode_meta["data/chunk_index"]
            file_index = self.get_episode_file_index(trajectory_id)
            # file_from_index = self.get_episode_file_from_index(trajectory_id)
            
            
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            import pyarrow.parquet as pq

            if not hasattr(self, "_v3_episode_row_groups"):
                self._v3_episode_row_groups = {}
            parquet = pq.ParquetFile(parquet_path)
            if parquet_path not in self._v3_episode_row_groups:
                groups_by_episode = defaultdict(list)
                offset = 0
                episode_column = parquet.schema.names.index("episode_index")
                for row_group in range(parquet.num_row_groups):
                    group = parquet.metadata.row_group(row_group)
                    statistics = group.column(episode_column).statistics
                    if (
                        statistics is None
                        or not statistics.has_min_max
                        or statistics.min != statistics.max
                        or not isinstance(statistics.min, (int, np.integer))
                        or any(
                            group.column(column).statistics is None
                            or group.column(column).statistics.null_count != 0
                            for column in range(group.num_columns)
                        )
                    ):
                        groups_by_episode = None
                        break
                    groups_by_episode[int(statistics.min)].append(
                        (row_group, offset, group.num_rows)
                    )
                    offset += group.num_rows
                self._v3_episode_row_groups[parquet_path] = groups_by_episode

            groups_by_episode = self._v3_episode_row_groups[parquet_path]
            groups = groups_by_episode.get(int(trajectory_id), []) if groups_by_episode is not None else []
            if groups:
                file_data = parquet.read_row_groups([group[0] for group in groups]).to_pandas()
                # A stored index column is already preserved by Arrow. A metadata-only RangeIndex
                # (or the default index when pandas metadata is absent) must be rebuilt from the
                # selected groups' original offsets, including non-contiguous groups of one episode.
                pandas_metadata = (parquet.schema_arrow.metadata or {}).get(b"pandas")
                index_columns = json.loads(pandas_metadata).get("index_columns", []) if pandas_metadata else []
                if not index_columns or (
                    len(index_columns) == 1
                    and isinstance(index_columns[0], dict)
                    and index_columns[0].get("kind") == "range"
                ):
                    range_metadata = index_columns[0] if index_columns else {}
                    index_start = int(range_metadata.get("start", 0))
                    index_step = int(range_metadata.get("step", 1))
                    positions = np.concatenate([
                        np.arange(offset, offset + count) for _, offset, count in groups
                    ])
                    file_data.index = pd.Index(
                        index_start + positions * index_step, name=range_metadata.get("name")
                    )
            else:
                file_data = pd.read_parquet(parquet_path)

            # filter by trajectory_id
            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()
            return episode_data


    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size
    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]
    
    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]


    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]

            video_file_indices = episode_meta.get("videos/file_indices", {})
            # print(f"{video_file_indices=}")
            #已修改的lerobotv3.0的视频索引
            if original_key in video_file_indices:
                video_chunk_index = video_file_indices[original_key]["chunk_index"]
                video_file_index = video_file_indices[original_key]["file_index"]
            else:
                video_chunk_index = episode_meta["data/chunk_index"]
                video_file_index = episode_meta["data/file_index"]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")

        # Image-only LeRobot datasets (e.g. VLA-Arena) may store frames directly
        # in parquet columns and have total_videos == 0 (no mp4 files). In this
        # case, load frames from the original image column and pad by step indices.
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self.curr_traj_data is not None and original_key in self.curr_traj_data.columns:
            image_entries = self.curr_traj_data[original_key].tolist()

            def _decode_image_entry(entry):
                if isinstance(entry, np.ndarray):
                    return entry
                if isinstance(entry, Image.Image):
                    return np.array(entry)
                if isinstance(entry, dict):
                    img_bytes = entry.get("bytes", None)
                    img_path = entry.get("path", None)

                    if img_bytes is not None:
                        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

                    if img_path is not None:
                        path_obj = Path(img_path)
                        if not path_obj.is_absolute():
                            path_obj = self.dataset_path / path_obj
                        return np.array(Image.open(path_obj).convert("RGB"))

                raise TypeError(f"Unsupported image entry type: {type(entry)}")

            frames = []
            for idx in step_indices:
                safe_idx = int(min(max(idx, 0), len(image_entries) - 1))
                frames.append(_decode_image_entry(image_entries[safe_idx]))

            return np.stack(frames)

        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self.lerobot_modality_meta.video[key].original_key
            if original_video_key is None:
                original_video_key = key
            from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend, # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
            # padding_strategy="zero",           # HACK for realdata
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)): # 
            # task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
            value = self.curr_traj_data[original_key].iloc[step_indices[i]] # TODO check v2.0 
            task_indices.append(value if isinstance(value, (int, float)) else value.item())

        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "eef":
            return self.get_eef(trajectory_id, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def get_eef(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        """Get an auxiliary raw-column sequence for a trajectory (eef / campose ...).

        Self-contained reader for optional raw parquet columns that bypass
        ``modality.json`` / ``metadata.modalities`` (which are limited to state/action/video),
        so no schema plumbing is required. Padding uses ``first_last`` (values are absolute).

        Key -> column mapping:
            ``eef.full``    -> ``observation.eef``      (legacy; dual-arm EEF xyz, add_eef_to_lerobot.py)
            ``eef.<sub>``   -> ``observation.<sub>``    (e.g. ``eef.campose`` -> ``observation.campose``,
                                                         add_campose_to_lerobot.py)

        Args:
            trajectory_id (int): The trajectory ID.
            key (str): The eef modality key. Only the delta indices matter.
            base_index (int): The base step index.

        Returns:
            np.ndarray: [len(delta_indices), dim] window (future horizon).
        """
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        sub = key.split(".", 1)[1] if "." in key else key
        le_key = "observation.eef" if sub == "full" else f"observation.{sub}"
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, (
            f"'{le_key}' column not found for {trajectory_id=}; run the corresponding "
            f"add_*_to_lerobot.py script first (eef -> add_eef_to_lerobot.py, "
            f"campose -> add_campose_to_lerobot.py)."
        )
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # [T, eef_dim]
        assert data_array.ndim == 2, f"Expected 2D eef array, got {data_array.shape}"
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last",
        )

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)
        
        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}
        
        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action
            
            # Flat checkpoint statistics must follow the packed action order.
            # RoboDojo places each gripper immediately after its arm joints.
            filtered_action_stats = {
                key: action_stats[key] for key in used_action_keys if key in action_stats
            }
            
            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)
                
                # mask=False for dimensions whose normalization mode is "binary"
                _action_norm_modes = _extract_action_normalization_modes(self.transforms)
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys(),
                    normalization_modes=_action_norm_modes,
                )
                combined_action_stats["mask"] = mask
                
                tag_stats["action"] = combined_action_stats
        
        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state
            
            filtered_state_stats = {
                key: state_stats[key] for key in used_state_keys if key in state_stats
            }
            
            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats
        
        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)
        
        statistics_data[tag] = tag_stats
        
        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (packed order): {list(used_action_keys)}")
        print(f"Used state keys (packed order): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                
                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed
                
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }
    
    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object
        
        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))
    
    return combined_stats

def _extract_action_normalization_modes(transforms) -> dict:
    """Extract normalization modes for action keys from a ComposedModalityTransform.

    Returns:
        dict: {subkey_without_action_prefix -> normalization_mode_str}
    """
    modes = {}
    for t in transforms.transforms:
        if isinstance(t, StateActionTransform):
            for key, mode in t.normalization_modes.items():
                if key.startswith("action."):
                    subkey = key[len("action."):]
                    modes[subkey] = mode
    return modes


def generate_action_mask_for_used_keys(
    action_modalities: dict,
    used_action_keys_ordered,
    normalization_modes: dict | None = None,
) -> list[bool]:
    """Generate per-dimension mask for action statistics.

    A dimension gets ``mask=False`` only when its normalization mode is ``"binary"``.
    This tells the inference code to skip continuous de-normalization for that dimension.
    All other modes (q99, mean_std, min_max ...) produce ``mask=True``.

    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys (no "action." prefix).
        normalization_modes (dict | None): Mapping {subkey -> mode} (no "action." prefix).
            If ``None``, all dimensions default to ``mask=True``.

    Returns:
        list[bool]: Per-dimension mask values.
    """
    mask = []

    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # mask=False only when the normalization mode is explicitly "binary"
            is_binary = (
                normalization_modes.get(subkey) == "binary"
                if normalization_modes is not None
                else False
            )

            for _ in range(dim_count):
                mask.append(not is_binary)

    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []
    
    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)
    
    # Extract state keys (remove "state." prefix)  
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)
    
    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        
        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")
        
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None
        configured_task_weights = parse_task_sampling_weights(
            self.data_cfg.get("task_sampling_weights", None)
            if self.data_cfg is not None
            else None
        )
        matched_task_weights: set[str] = set()

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")
        self._getitem_count = 0
        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        
        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)
        
        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum

        # Sampling ratios and normalization-statistics ratios are separate controls.  During
        # weight-only post-training, changing the action/state normalization underneath a loaded
        # checkpoint creates an avoidable target discontinuity.  An optional dataset-name mapping
        # therefore lets callers preserve the base run's metadata mixture while changing only the
        # examples that are drawn.
        raw_metadata_weights = (
            self.data_cfg.get("metadata_dataset_weights", None)
            if self.data_cfg is not None
            else None
        )
        if raw_metadata_weights is None:
            self._metadata_dataset_weights = self._dataset_sampling_weights.copy()
        else:
            raw_metadata_weights = config_mapping(
                raw_metadata_weights, "metadata_dataset_weights"
            )
            dataset_names = [dataset.dataset_name for dataset in self.datasets]
            missing = set(dataset_names) - set(raw_metadata_weights)
            extra = set(raw_metadata_weights) - set(dataset_names)
            if missing or extra:
                raise ValueError(
                    f"metadata_dataset_weights keys must exactly match loaded datasets; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            metadata_weights = np.asarray(
                [float(raw_metadata_weights[name]) for name in dataset_names], dtype=np.float64
            )
            if np.any(~np.isfinite(metadata_weights)) or np.any(metadata_weights <= 0):
                raise ValueError("metadata_dataset_weights values must be finite and > 0")
            self._metadata_dataset_weights = metadata_weights / metadata_weights.sum()
            print(
                "[task-sampling] dataset draw weights differ from metadata/statistics weights: "
                f"draw={self._dataset_sampling_weights.tolist()}, "
                f"metadata={self._metadata_dataset_weights.tolist()}"
            )

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths

            if configured_task_weights:
                if len(dataset.trajectory_tasks) != len(dataset.trajectory_lengths):
                    raise ValueError(
                        f"Dataset {dataset.dataset_name} has {len(dataset.trajectory_lengths)} "
                        f"trajectories but {len(dataset.trajectory_tasks)} task annotations"
                    )
                multipliers, matched = task_weight_multipliers(
                    dataset.trajectory_tasks, configured_task_weights
                )
                trajectory_sampling_weights *= multipliers
                matched_task_weights.update(matched)
                weighted_episodes = int(np.count_nonzero(multipliers != 1.0))
                if weighted_episodes:
                    print(
                        f"[task-sampling] dataset={dataset.dataset_name}: "
                        f"reweighted {weighted_episodes}/{len(multipliers)} trajectories; "
                        f"multiplier range=[{multipliers.min():g}, {multipliers.max():g}]"
                    )
            
            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)
            
            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum
            
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        missing_task_weights = set(configured_task_weights) - matched_task_weights
        if missing_task_weights:
            raise ValueError(
                "task_sampling_weights contains task text not found in any loaded episode: "
                + ", ".join(repr(task) for task in sorted(missing_task_weights))
            )

        # 4. All positively weighted datasets define the synthetic epoch. The former special case
        # "raw weight == 1 means primary" breaks legitimate fractional mixtures (such as multiple
        # Astribot variants sharing one semantic task's mass) and makes __len__ depend on an arbitrary
        # representation detail rather than the normalized sampling distribution.
        self._primary_dataset_indices = self._dataset_sampling_weights > 0

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def metadata_dataset_weights(self) -> np.ndarray:
        """Dataset weights used only to merge normalization statistics."""

        return self._metadata_dataset_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset.

        Two regimes (selected by ``data_cfg.sequential_step_sampling`` + ``data_cfg.group_size``):

          * Default (random): each index independently samples a (dataset, trajectory, step). This is
            the original behaviour -- consecutive indices are unrelated.
          * Group / sequential: indices are bucketed into groups of ``group_size``. All indices in a
            group share one sampled (dataset, trajectory) and return CONSECUTIVE, time-increasing
            steps. This guarantees group integrity at the DATASET-INDEX level only.

        IMPORTANT: whether each GPU's *local batch* actually receives complete, contiguous groups also
        depends on how the DataLoader sampler + the distributed launcher (Accelerate/DeepSpeed) shard
        indices. It holds when indices are read in order (``shuffle=False``),
        ``per_device_batch_size % group_size == 0``, and each rank receives whole contiguous batches
        (Accelerate default: dispatch_batches=False, split_batches=False). If the launcher instead
        stride-shards indices across ranks, a local batch may mix multiple episodes -- in which case
        the memory bank's per-batch clear + ``i % group_size`` boundary handling can leave one
        episode's bank residing while processing another. ALWAYS verify the per-rank batch layout
        empirically (set QWENGEO_DEBUG_BATCH=1) and/or enable ``validate_group_layout`` before
        trusting cross-frame memory in multi-GPU training; if it fails, use a group-aware sampler.
        """
        group_size = int(self.data_cfg.get("group_size", 1)) if self.data_cfg is not None else 1
        use_group = (
            bool(self.data_cfg.get("sequential_step_sampling", False)) if self.data_cfg is not None else False
        )

        if use_group and group_size > 1:
            group_id = index // group_size
            pos = index % group_size
            seed = group_id if self.mode != "train" else safe_hash((self.epoch, group_id, self.seed))
            rng = np.random.default_rng(seed)
            dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
            dataset = self.datasets[dataset_index]
            # Only sample trajectories long enough to hold a full group (consecutive window or
            # group_size DISTINCT frames -- both need traj_len >= group_size to keep strict
            # time-ordering without repeating frames).
            lengths = np.asarray(dataset.trajectory_lengths)
            eligible = np.flatnonzero(lengths >= group_size)
            if len(eligible) == 0:
                raise ValueError(
                    f"Dataset {dataset.dataset_name} has no trajectory with length >= "
                    f"group_size={group_size}; reduce group_size."
                )
            weights = self.trajectory_sampling_weights[dataset_index][eligible]
            weights = weights / weights.sum()
            trajectory_index = int(rng.choice(eligible, p=weights))
            trajectory_id = dataset.trajectory_ids[trajectory_index]
            traj_len = int(dataset.trajectory_lengths[trajectory_index])

            # group_sampling_mode selects HOW the group's frames are drawn from the trajectory:
            #   consecutive   (default) -> a contiguous stride-1 window [start, start+group_size).
            #                  Legacy behaviour; the memory bank only ever sees the IMMEDIATE past.
            #   random_sorted -> group_size DISTINCT frames sampled uniformly over the WHOLE
            #                  trajectory, sorted by time.
            #                  The memory bank then trains on sparse long-range history with
            #                  variable timestep gaps -- matching what a STREAMING rollout's bank
            #                  looks like, which is what makes streaming inference work.
            # Both modes are strictly time-increasing within the group, so the banks' causal
            # retrieval and validate_group_layout checks hold unchanged.
            sampling_mode = str(self.data_cfg.get("group_sampling_mode", "consecutive"))
            if sampling_mode == "random_sorted":
                frames = np.sort(rng.choice(traj_len, size=group_size, replace=False))
                step = int(frames[pos])
            elif sampling_mode == "consecutive":
                # The window [start, start+group_size) is fully inside the trajectory.
                start = int(rng.integers(0, traj_len - group_size + 1))
                step = start + pos
            else:
                raise ValueError(
                    f"Unknown group_sampling_mode={sampling_mode!r} "
                    "(expected 'consecutive' or 'random_sorted')."
                )
            return dataset, trajectory_id, step

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = rng.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        self._getitem_count += 1
        if self._getitem_count % 1000 == 0:
            gc.collect()

        # In group/sequential mode we MUST NOT replace a single failed frame with a random index
        # from another group -- that would corrupt the group's episode contiguity. So group mode uses
        # a dedicated tolerant loader that retries the SAME frame (transient network/FS read glitches)
        # and, only if that keeps failing, substitutes a readable same-episode frame from this
        # slot's disjoint time interval (preserving real, strictly ordered frame indices even when
        # adjacent slots both fail). See `_getitem_group`.
        use_group_sampling = (
            bool(self.data_cfg.get("sequential_step_sampling", False))
            and int(self.data_cfg.get("group_size", 1)) > 1
            if self.data_cfg is not None
            else False
        )

        if use_group_sampling:
            return self._getitem_group(index)

        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                sample_tries = 0
                max_sample_tries = 200
                while True:
                    sample_tries += 1
                    if sample_tries > max_sample_tries:
                        raise RuntimeError(
                            f"Unable to sample a valid item after {max_sample_tries} attempts. "
                            f"dataset={self.datasets[0].dataset_name if len(self.datasets)>0 else 'unknown'}"
                        )

                    dataset, trajectory_id, step = self.sample_step(index)
                    # If dataset has no physical videos (e.g., image frames in parquet
                    # for VLA-Arena), do not gate sampling on mp4 existence.
                    total_videos = int(dataset.lerobot_info_meta.get("total_videos", 0))
                    if total_videos == 0:
                        break

                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)
                    
                raw_data = dataset.get_step_data(trajectory_id, step)    
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)
                dataset_index = self.datasets.index(dataset)
                episode_id = int(dataset_index) * 10_000_000 + int(trajectory_id)
                return dataset._add_episode_fields(
                    sample, trajectory_id, step, episode_id=episode_id
                )
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print(f"Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def _getitem_group(self, index: int) -> dict:
        """Tolerant __getitem__ for group / sequential sampling.

        Retry the intended frame before considering same-trajectory alternates. On persistent
        failure, reconstruct the deterministic group's intended frames and partition the trajectory
        at adjacent midpoints. Each slot may use only its own disjoint interval, so simultaneous
        fallbacks cannot duplicate or reorder frames. Metadata always records the frame actually
        read. If no candidate in that interval is readable, raise; a readable frame elsewhere in
        the episode cannot safely replace this slot. Normal sampling is unchanged.
        """
        dataset, trajectory_id, step = self.sample_step(index)
        step = int(step)
        dataset_index = self.datasets.index(dataset)
        episode_id = int(dataset_index) * 10_000_000 + int(trajectory_id)

        def _load(cur_step: int) -> dict:
            raw_data = dataset.get_step_data(trajectory_id, cur_step)
            data = dataset.transforms(raw_data)
            sample = dataset._pack_sample(data)
            return dataset._add_episode_fields(
                sample, trajectory_id, cur_step, episode_id=episode_id
            )

        per_frame_tries = 3  # transient-read retries on the SAME frame before moving on
        last_exception = None

        def _try_load(cur_step: int) -> dict | None:
            nonlocal last_exception
            for attempt in range(per_frame_tries):
                try:
                    return _load(cur_step)
                except Exception as e:
                    last_exception = e
                    if attempt < per_frame_tries - 1:
                        # brief backoff to ride out a transient network-storage read glitch
                        time.sleep(0.5)
            return None

        sample = _try_load(step)
        if sample is not None:
            return sample

        # Reconstruct only on failure: successful reads retain the original sampling path.
        group_size = int(self.data_cfg.get("group_size", 1))
        group_start = index - index % group_size
        group_frames = []
        for slot_index in range(group_start, group_start + group_size):
            slot_dataset, slot_trajectory_id, slot_step = self.sample_step(slot_index)
            if slot_dataset is not dataset or slot_trajectory_id != trajectory_id:
                raise RuntimeError("Group fallback requires one deterministic dataset/trajectory.")
            group_frames.append(int(slot_step))
        if any(right <= left for left, right in zip(group_frames, group_frames[1:])):
            raise RuntimeError("Group fallback requires strictly increasing intended frame indices.")

        pos = index % group_size
        if group_frames[pos] != step:
            raise RuntimeError("Group fallback sampling changed the intended frame during retry.")
        traj_len = int(dataset.trajectory_lengths[dataset.get_trajectory_index(trajectory_id)])
        lower = 0 if pos == 0 else (group_frames[pos - 1] + step) // 2 + 1
        upper = traj_len - 1 if pos == group_size - 1 else (step + group_frames[pos + 1]) // 2
        if not (0 <= lower <= step <= upper < traj_len):
            raise RuntimeError(f"Invalid group fallback interval [{lower}, {upper}] for step={step}.")

        # Adjacent slots partition the trajectory at the same midpoint, with the right slot
        # beginning one frame later. Restrict both neighbour and random candidates to this range.
        candidates = []
        for delta in (1, -1, 2, -2, 3, -3):
            candidate = step + delta
            if lower <= candidate <= upper and candidate not in candidates:
                candidates.append(candidate)
        if lower < upper:
            alt_rng = np.random.default_rng(safe_hash((self.epoch, index, self.seed, "group_fallback")))
            for candidate in alt_rng.integers(lower, upper + 1, size=8).tolist():
                if candidate != step and candidate not in candidates:
                    candidates.append(int(candidate))

        print(
            f"[dataloader] group-mode: read failed for (traj={trajectory_id}, step={step}) "
            f"after {per_frame_tries} tries: {type(last_exception).__name__}: {last_exception}. "
            f"Trying same-episode frames only within slot interval [{lower}, {upper}]..."
        )
        for cur_step in candidates:
            sample = _try_load(cur_step)
            if sample is not None:
                print(
                    f"[dataloader] group-mode: frame (traj={trajectory_id}, step={step}) "
                    f"unreadable; substituted same-episode frame step={cur_step} "
                    f"within [{lower}, {upper}] (episode and strict time order preserved)."
                )
                return sample

        # A consecutive group's interior slot may contain only its intended frame. Fail clearly
        # after its transient retries instead of borrowing a neighbour and breaking causal order.
        raise RuntimeError(
            f"Group-mode read failed for trajectory {trajectory_id} (dataset={dataset.dataset_name}), "
            f"intended step={step}: none of {len(candidates) + 1} tried frames in legal slot "
            f"interval [{lower}, {upper}] was readable. Cannot substitute outside this interval; "
            "repair or pre-filter the unreadable frame/group."
        ) from last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0
            
        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)
        
        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0
        
        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0
        
        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]
        
        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()
        
        result = int(max_ratio)
        group_size = int(self.data_cfg.get("group_size", 1)) if self.data_cfg is not None else 1
        use_group = (
            bool(self.data_cfg.get("sequential_step_sampling", False))
            if self.data_cfg is not None else False
        )
        if use_group and group_size > 1:
            # Every dataset index group must be complete. Truncating the synthetic epoch only
            # discards at most group_size-1 random samples and prevents a tail group from being
            # padded/repeated across distributed ranks with a different episode.
            result -= result % group_size
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.metadata_dataset_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []
        
        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)
        
        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}
            
            # Process action statistics
            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action
                
                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []
                
                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]
                
                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)
                    
                    # Collect action normalization modes from datasets of this tag.
                    # "binary" takes precedence: if any dataset marks a key as binary, use binary.
                    _action_norm_modes: dict = {}
                    for _ds in self.datasets:
                        if _ds.tag == tag:
                            for _k, _m in _extract_action_normalization_modes(_ds.transforms).items():
                                if _k not in _action_norm_modes or _m == "binary":
                                    _action_norm_modes[_k] = _m
                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys(),
                        normalization_modes=_action_norm_modes,
                    )
                    combined_action_stats["mask"] = mask
                    
                    tag_stats["action"] = combined_action_stats
            
            # Process state statistics
            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state
                
                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []
                
                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]
                
                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats
            
            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))
            
            statistics_data[tag] = tag_stats
        
        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")


    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.
        
        Args:
            tag (str): embodiment tag
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0
        
        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)
        
        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")
        
        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)
            
            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return
        
        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue
                
            # Convert back to DatasetMetadata format
            metadata_dict = {
                "embodiment_tag": tag,
                "statistics": {
                    "action": {},
                    "state": {}
                },
                "modalities": {}
            }
            
            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data
            
            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data
            
            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)
        
        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])
        
        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
