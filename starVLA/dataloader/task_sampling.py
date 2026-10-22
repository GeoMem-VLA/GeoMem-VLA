"""Utilities for task-aware trajectory sampling.

Task weights are keyed by the natural-language task stored in LeRobot's
``meta/episodes.jsonl``.  Keeping this logic independent of LIBERO task IDs is
intentional: dataset task indices and benchmark evaluation IDs do not share the
same ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np


def normalize_task_name(value: str) -> str:
    """Normalize task text without erasing semantically meaningful words."""

    return " ".join(str(value).strip().lower().split())


def config_mapping(raw: object | None, field_name: str) -> dict:
    """Convert Python and OmegaConf mapping-like objects to a plain dict.

    Some OmegaConf releases do not register ``DictConfig`` as a
    ``collections.abc.Mapping``, despite exposing the normal ``items`` API.
    Accepting that API keeps configuration parsing independent of the exact
    OmegaConf version used by a training cluster.
    """

    if raw is None:
        return {}
    items = getattr(raw, "items", None)
    if not isinstance(raw, Mapping) and not callable(items):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        return dict(raw.items())  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a mapping") from exc


def parse_task_sampling_weights(raw: object | None) -> dict[str, float]:
    """Validate and normalize a ``task text -> positive multiplier`` mapping."""

    raw_mapping = config_mapping(raw, "task_sampling_weights")
    if not raw_mapping:
        return {}

    parsed: dict[str, float] = {}
    for task, value in raw_mapping.items():
        normalized = normalize_task_name(task)
        if not normalized:
            raise ValueError("task_sampling_weights contains an empty task name")
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"task_sampling_weights[{task!r}] must be finite and > 0, got {value!r}"
            )
        if normalized in parsed and parsed[normalized] != weight:
            raise ValueError(
                f"Duplicate normalized task {normalized!r} has conflicting weights: "
                f"{parsed[normalized]} and {weight}"
            )
        parsed[normalized] = weight
    return parsed


def task_weight_multipliers(
    trajectory_tasks: Sequence[Sequence[str]],
    configured_weights: Mapping[str, float],
) -> tuple[np.ndarray, set[str]]:
    """Return one multiplier per trajectory and the configured tasks that matched.

    LeRobot episodes normally contain one task.  For the uncommon multi-task
    episode, the largest configured multiplier is used; this avoids multiplying
    weights merely because an episode has several equivalent annotations.
    """

    multipliers = np.ones(len(trajectory_tasks), dtype=np.float64)
    matched: set[str] = set()
    for index, tasks in enumerate(trajectory_tasks):
        episode_weights = []
        for task in tasks:
            normalized = normalize_task_name(task)
            if normalized in configured_weights:
                matched.add(normalized)
                episode_weights.append(float(configured_weights[normalized]))
        if episode_weights:
            multipliers[index] = max(episode_weights)
    return multipliers, matched
