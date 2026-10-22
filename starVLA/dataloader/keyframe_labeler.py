"""Rule-based keyframe labelling for the event-guided memory bank.

Produces a per-frame ``event_prior`` teacher target for the event-guided memory bank: 1.0 where the
frame is a task-relevant event, 0.0 elsewhere (optionally smoothed around the event).

This is deliberately a *rule* rather than a learned detector. Near-zero-velocity bottleneck
extraction is the standard keyframe heuristic in manipulation imitation learning (PerAct and
successors), so a hand-written detector here is a mature baseline, not a stopgap. Everything below
runs on the low-dimensional proprio/action stream only -- no images, no network, no GPU -- so labels
can be precomputed offline for a whole dataset in seconds. The keyframe target is an *offline
teacher*: pause confirmation, priority NMS, proximity smoothing, and task completion may use episode
context that is unavailable online. It is therefore suitable for BCE supervision, but should not
directly override training-time writes unless the same external detector is also available at
deployment. ``compute_event_aux`` is kept causal separately.

Signals, in decreasing order of reliability:

1. ``gripper``    -- open/close transitions. Every grasp and release is unambiguously an event, and
                     this is the single strongest signal available.
2. ``bottleneck`` -- the end-effector comes to rest after moving. These are the sub-goal boundaries
                     PerAct keys on.
3. ``turn``       -- a sharp reversal in end-effector direction: an action changepoint.

Post-processing matters as much as the signals. Pauses span many consecutive frames, so the raw
bottleneck mask fires in long runs; without suppression a single pause would be written to memory
five or six times as near-identical neighbours -- exactly the redundancy the memory fix was meant to
avoid. ``min_interval`` performs a priority-aware non-maximum suppression so each event contributes
at most one frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# Priority decides which candidate survives non-maximum suppression when two events collide.
_PRIORITY = {"gripper": 3, "last": 2, "bottleneck": 1, "turn": 1}


@dataclass
class KeyframeConfig:
    """Thresholds for :func:`compute_keyframes`.

    Defaults are tuned on LIBERO (delta-xyz actions, binary gripper) and yield roughly 4-6 keyframes
    per episode. Re-check ``summarize`` on any new embodiment before training on the labels.
    """

    # Columns of the gripper command inside `action`. A tuple so bimanual embodiments are handled:
    # RoboTwin has TWO grippers (indices 6 and 13 of its 14-D joint vector) and reading only the
    # last one would silently drop every left-arm grasp.
    gripper_cols: tuple = (-1,)
    # End-effector position columns. These index whatever array the caller passes as `pos` -- for
    # LIBERO that is the 8-D state (xyz at 0:3), for RoboTwin the separate 6-D `observation.eef`
    # (left xyz + right xyz), because its `state` holds JOINT ANGLES and differencing those would
    # measure joint motion rather than end-effector motion.
    eef_cols: tuple = (0, 1, 2)
    # Rotation columns inside `pos`; None auto-selects the three columns following `eef_cols`
    # (correct for LIBERO xyz+rpy), and an empty tuple disables the channel (RoboTwin eef has no
    # orientation).
    rot_cols: tuple | None = None
    vel_quantile: float = 0.15     # "at rest" = speed below this fraction of the median speed
    bottleneck_min_run: int = 2    # consecutive at-rest frames required to call it a pause
    turn_angle_deg: float = 60.0   # direction change above this counts as a changepoint
    turn_min_speed: float = 0.5    # ... but only when moving faster than this fraction of median
    # NMS radius in frames. The arm pauses to close the gripper, so the bottleneck detector fires
    # just before AND just after every grasp: at radius 4 a single grasp was labelled three times
    # (measured: 87 pairs less than 10 frames apart across 60 episodes). Radius 12 removes all of
    # them while still keeping 215/218 gripper transitions.
    min_interval: int = 12
    include_last: bool = True      # the final frame is the task-completion event
    warmup: int = 2                # ignore the first frames (settling, no reliable velocity yet)
    soft_sigma: float = 0.0        # >0 turns the point mask into a proximity score (see below)


@dataclass
class KeyframeResult:
    keyframe: np.ndarray                       # bool [T]
    reason: list = field(default_factory=list)  # per-kept-frame ("frame_index", "reason")
    raw_counts: dict = field(default_factory=dict)  # candidates per signal, before NMS
    prior: np.ndarray | None = None            # float [T] in [0, 1]; == keyframe when soft_sigma == 0


def proximity_prior(keyframe: np.ndarray, sigma: float) -> np.ndarray:
    """Turn point keyframes into a smooth "how close is the nearest event" score.

    A hard point label is difficult for group-mode training, because the memory bank never sees the
    full frame rate: a group may sample only ~16 frames across a ~250-frame episode.

    Scoring each frame by proximity to the nearest offline event gives the BCE objective useful
    positive mass even when the exact keyframe is not sampled. ``sigma`` is in frames; frames within
    ~1.2 sigma of an event exceed 0.5. This smoothing is symmetric and therefore intentionally an
    offline supervision target, not an online detector output.
    """
    T = len(keyframe)
    idx = np.nonzero(keyframe)[0]
    if T == 0 or idx.size == 0 or sigma <= 0:
        return keyframe.astype(np.float32)
    dist = np.abs(np.arange(T)[:, None] - idx[None, :]).min(axis=1)
    return np.exp(-(dist.astype(np.float64) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


def _resolve_rot_cols(cfg: KeyframeConfig, width: int) -> list[int]:
    """Rotation columns: explicit if configured, else the three right after the position ones."""
    if cfg.rot_cols is not None:
        return list(cfg.rot_cols)
    start = max(cfg.eef_cols) + 1
    return list(range(start, start + 3)) if width >= start + 3 else []


def compute_keyframes(
    state: np.ndarray,
    action: np.ndarray,
    cfg: KeyframeConfig | None = None,
    pos: np.ndarray | None = None,
) -> KeyframeResult:
    """Label task-relevant events in one episode.

    Args:
        state: ``[T, S]`` proprio stream (used as the position source when ``pos`` is omitted).
        action: ``[T, A]`` action stream; ``cfg.gripper_cols`` selects the gripper command(s).
        pos: optional ``[T, P]`` end-effector position stream. Pass it whenever the position does
            not live inside ``state`` -- RoboTwin keeps joint angles in ``state`` and Cartesian
            end-effector positions in a separate ``observation.eef`` column, and differencing joint
            angles would detect joint motion instead of the end-effector pauses we are after.
    """
    cfg = cfg or KeyframeConfig()
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    pos_src = state if pos is None else np.asarray(pos, dtype=np.float64)
    if state.ndim != 2 or action.ndim != 2 or pos_src.ndim != 2:
        raise ValueError(f"state/action/pos must be 2-D, got {state.shape}, {action.shape}, {pos_src.shape}")
    T = len(state)
    if len(action) != T or len(pos_src) != T:
        raise ValueError(f"state has {T} frames but action has {len(action)} and pos has {len(pos_src)}")
    if T == 0:
        return KeyframeResult(keyframe=np.zeros(0, dtype=bool))

    candidates: list[tuple[int, str]] = []

    # --- 1) gripper transitions -------------------------------------------------------------
    # In a behaviour-cloning row, action[t] is the command predicted *from* observation[t], so it
    # cannot be an input to the current frame's gate. A change action[t-1] -> action[t] first becomes
    # available after action[t] has executed, at observation[t+1]. Shift the teacher event by one
    # frame; transitions at the final command have no subsequent observation and are omitted.
    # Any gripper changing is an event, so bimanual arms are OR-ed rather than requiring both.
    for col in cfg.gripper_cols:
        grip = action[:, col]
        for t in np.nonzero(np.diff(grip) != 0)[0] + 2:
            if t < T:
                candidates.append((int(t), "gripper"))

    # --- 2) bottleneck poses ----------------------------------------------------------------
    # For a bimanual embodiment eef_cols spans both arms, so "at rest" means the whole system is
    # still -- which is what delimits a shared sub-goal.
    eef = pos_src[:, list(cfg.eef_cols)]
    step = np.diff(eef, axis=0)
    speed = np.concatenate([[0.0], np.linalg.norm(step, axis=1)])
    moving = speed[speed > 0]
    med = float(np.median(moving)) if moving.size else 0.0
    if med > 0:
        at_rest = speed < cfg.vel_quantile * med
        at_rest[: cfg.warmup] = False
        # keep the FIRST frame of each at-rest run that is long enough and follows actual motion
        t = cfg.warmup
        while t < T:
            if not at_rest[t]:
                t += 1
                continue
            run_end = t
            while run_end < T and at_rest[run_end]:
                run_end += 1
            if (run_end - t) >= cfg.bottleneck_min_run and t > 0 and not at_rest[t - 1]:
                candidates.append((int(t), "bottleneck"))
            t = run_end

        # --- 3) direction changepoints ------------------------------------------------------
        norms = np.linalg.norm(step, axis=1)
        cos_lim = np.cos(np.deg2rad(cfg.turn_angle_deg))
        for i in range(1, len(step)):
            a, b = norms[i - 1], norms[i]
            if a < cfg.turn_min_speed * med or b < cfg.turn_min_speed * med:
                continue
            cos = float(np.dot(step[i - 1], step[i]) / (a * b))
            if cos < cos_lim:
                candidates.append((int(i + 1), "turn"))

    # --- 4) task completion -----------------------------------------------------------------
    if cfg.include_last and T > 0:
        candidates.append((T - 1, "last"))

    raw_counts: dict = {}
    for _, r in candidates:
        raw_counts[r] = raw_counts.get(r, 0) + 1

    # --- 5) priority-aware NMS --------------------------------------------------------------
    # Strong events win collisions; ties break towards the earlier frame so a pause is anchored at
    # its onset rather than drifting to the end of the run.
    candidates.sort(key=lambda c: (-_PRIORITY.get(c[1], 0), c[0]))
    kept: list[tuple[int, str]] = []
    for t, reason in candidates:
        if all(abs(t - k) >= cfg.min_interval for k, _ in kept):
            kept.append((t, reason))
    kept.sort()

    mask = np.zeros(T, dtype=bool)
    for t, _ in kept:
        mask[t] = True
    return KeyframeResult(
        keyframe=mask,
        reason=kept,
        raw_counts=raw_counts,
        prior=proximity_prior(mask, cfg.soft_sigma),
    )


# Width of :func:`compute_event_aux` for the DEFAULT (LIBERO single-arm) config. Other embodiments
# have a different width -- call :func:`event_aux_dim` with your config instead of assuming 9.
EVENT_AUX_DIM = 9


def event_aux_dim(cfg: KeyframeConfig | None = None, pos_width: int = 8) -> int:
    """Width of :func:`compute_event_aux` for a given config; wire into ``geoplan.event_aux_dim``.

    Layout: ``[Δpos, Δrot, |Δpos|, grippers, Δgrippers]``, so the width depends on the embodiment:
    LIBERO (3 pos + 3 rot + 1 speed + 1 gripper * 2) = 9, RoboTwin (6 pos + 0 rot + 1 speed +
    2 grippers * 2) = 11. The bank raises if this disagrees with the configured ``event_aux_dim``,
    so a mismatch fails loudly at startup rather than training on a misaligned vector.
    """
    cfg = cfg or KeyframeConfig()
    return len(cfg.eef_cols) + len(_resolve_rot_cols(cfg, pos_width)) + 1 + 2 * len(cfg.gripper_cols)


def compute_event_aux(
    state: np.ndarray,
    action: np.ndarray,
    cfg: KeyframeConfig | None = None,
    pos: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame low-dimensional event signal, the ``ΔP_eef / Δh_obj`` input of the event scorer.

    Returns ``[T, event_aux_dim(cfg)]``:
    ``[Δpos, Δrot, |Δpos|, gripper_cmds, Δgripper_cmds]``. Widths follow the embodiment -- 9 for
    LIBERO's single arm, 11 for RoboTwin's two arms -- so use :func:`event_aux_dim`, not a constant.
    ``pos`` overrides the position source (RoboTwin keeps Cartesian eef in a separate column).

    Every entry is **strictly backward-looking**. State channels compare observation ``t`` against
    ``t-1``. The gripper command at frame ``t`` is ``action[t-1]`` (the command that was just
    executed), never the behaviour-cloning target ``action[t]``. Frame 0 has no predecessor and gets
    zeros; at frame 1 the previous command is known but its change is not yet defined.

    The gripper channels ultimately come from ``action``, not ``state``, but are shifted by one frame
    to represent the previously executed command. In LIBERO the last state column is a
    finger joint position -- a small continuous value whose largest changes do NOT coincide with
    grasps (measured: |Δ| is *lower* at keyframes than elsewhere), so using it would feed the scorer
    an actively misleading signal. The action's last column is the binary open/close command, which
    is exactly the grasp/release event.
    """
    cfg = cfg or KeyframeConfig()
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    pos_src = state if pos is None else np.asarray(pos, dtype=np.float64)
    if state.ndim != 2 or action.ndim != 2 or pos_src.ndim != 2:
        raise ValueError(f"state/action/pos must be 2-D, got {state.shape}, {action.shape}, {pos_src.shape}")
    T = len(state)
    if len(action) != T or len(pos_src) != T:
        raise ValueError(f"state has {T} frames but action has {len(action)} and pos has {len(pos_src)}")
    if T == 0:
        return np.zeros((0, event_aux_dim(cfg, pos_src.shape[1] if pos_src.size else 8)), dtype=np.float32)

    pos_xyz = pos_src[:, list(cfg.eef_cols)]
    rc = _resolve_rot_cols(cfg, pos_src.shape[1])
    rot = pos_src[:, rc] if rc else np.zeros((T, 0))
    raw_grip = action[:, list(cfg.gripper_cols)].reshape(T, len(cfg.gripper_cols))
    # Causal command available at observation t. Do not expose action[0] at frame 0: that row is the
    # current action target. The unknown pre-episode command is represented by zero.
    grip = np.zeros_like(raw_grip)
    if T > 1:
        grip[1:] = raw_grip[:-1]

    def _delta(x: np.ndarray) -> np.ndarray:
        d = np.zeros_like(x)
        d[1:] = np.diff(x, axis=0)
        return d

    dpos, drot = _delta(pos_xyz), _delta(rot)
    dgrip = np.zeros_like(grip)
    if T > 2:
        dgrip[2:] = np.diff(raw_grip[:-1], axis=0)
    speed = np.linalg.norm(dpos, axis=1, keepdims=True)
    aux = np.concatenate([dpos, drot, speed, grip, dgrip], axis=1)
    return aux.astype(np.float32)


def calibrate(
    episodes: list[tuple],
    cfg: KeyframeConfig,
    *,
    group_size: int,
    exec_horizon: int,
    sigmas: tuple = (0.0, 1.0, 2.0, 4.0, 6.0),
    seed: int = 0,
) -> dict:
    """Sweep ``soft_sigma`` and report what each value implies for training and deployment.

    ``episodes`` is a list of ``(state, action, pos_or_None)`` arrays.

    The two knobs must be calibrated IN ORDER, because the second is derived from the first:

    1. ``soft_sigma`` -- chosen so the bank actually receives events. It depends on sampling
       density, not on the label rate: training sees ``group_size`` frames spread over an episode
       and a rollout queries every ``exec_horizon`` steps, so a point label is almost never sampled
       (measured 0 hits per 16-frame LIBERO group). Pick the smallest sigma whose per-group hit
       count is a useful fraction of the events in the episode.
    2. ``event_supervision_pos_weight`` = ``(1 - prior_mean) / prior_mean`` for THAT sigma. It is a
       function of the label distribution, so changing sigma (or min_interval, or the data mixture)
       silently invalidates it.

    ``event_aux_dim`` is structural rather than statistical: it follows from the embodiment's column
    layout, and only matters when the deployment client actually sends aux.
    """
    rng = np.random.RandomState(seed)
    out = []
    for sig in sigmas:
        c = replace(cfg, soft_sigma=sig)
        priors, tr_hits, inf_hits, n_events = [], [], [], []
        for state, action, pos in episodes:
            r = compute_keyframes(state, action, c, pos=pos)
            pr = r.prior if r.prior is not None else r.keyframe.astype(np.float32)
            T = len(pr)
            priors.append(float(pr.mean()))
            n_events.append(int(r.keyframe.sum()))
            idx = np.sort(rng.choice(T, min(group_size, T), replace=False))
            tr_hits.append(int((pr[idx] > 0.5).sum()))
            inf_hits.append(int((pr[np.arange(0, T, max(exec_horizon, 1))] > 0.5).sum()))
        m = float(np.mean(priors))
        out.append(
            {
                "soft_sigma": sig,
                "prior_mean": m,
                "pos_weight": (1 - m) / m if m > 0 else float("inf"),
                "train_hits_per_group": float(np.mean(tr_hits)),
                "infer_hits_per_episode": float(np.mean(inf_hits)),
                "events_per_episode": float(np.mean(n_events)),
            }
        )
    return {
        "sweep": out,
        "event_aux_dim": event_aux_dim(cfg, len(episodes[0][2][0]) if episodes and episodes[0][2] is not None else 8),
        "episodes": len(episodes),
        "median_length": float(np.median([len(e[0]) for e in episodes])) if episodes else 0.0,
        "queries_per_episode": (
            float(np.median([len(e[0]) for e in episodes])) / max(exec_horizon, 1) if episodes else 0.0
        ),
    }


def summarize(results: list[KeyframeResult], *, episode_balanced: bool = False) -> dict:
    """Aggregate label statistics over episodes.

    ``pos_weight`` balances the actual BCE targets. With ``soft_sigma > 0`` this must use the soft
    target mass, not the number of hard keyframe points. Set ``episode_balanced=True`` when the
    sampler chooses episodes uniformly and then draws a fixed-size group from each one; otherwise
    the default frame-weighted statistic matches ordinary uniform-over-frame sampling.
    """
    total = sum(len(r.keyframe) for r in results)
    hard_pos = sum(int(r.keyframe.sum()) for r in results)
    prior_mass = sum(
        float((r.prior if r.prior is not None else r.keyframe.astype(np.float32)).sum())
        for r in results
    )
    threshold_pos = sum(
        int(((r.prior if r.prior is not None else r.keyframe.astype(np.float32)) > 0.5).sum())
        for r in results
    )
    per_reason: dict = {}
    for r in results:
        for _, reason in r.reason:
            per_reason[reason] = per_reason.get(reason, 0) + 1
    hard_rate = hard_pos / total if total else 0.0
    prior_mean = prior_mass / total if total else 0.0
    episode_prior_means = [
        float((r.prior if r.prior is not None else r.keyframe.astype(np.float32)).mean())
        for r in results
        if len(r.keyframe) > 0
    ]
    episode_balanced_prior_mean = (
        float(np.mean(episode_prior_means)) if episode_prior_means else 0.0
    )
    selected_prior_mean = episode_balanced_prior_mean if episode_balanced else prior_mean
    return {
        "episodes": len(results),
        "frames": total,
        "keyframes": hard_pos,
        "rate": hard_rate,  # backward-compatible name for the hard point-label rate
        "hard_rate": hard_rate,
        "prior_mean": prior_mean,
        "episode_balanced_prior_mean": episode_balanced_prior_mean,
        "prior_threshold_rate": threshold_pos / total if total else 0.0,
        "per_episode": hard_pos / len(results) if results else 0.0,
        "pos_weight": (
            (1 - selected_prior_mean) / selected_prior_mean
            if selected_prior_mean > 0
            else float("inf")
        ),
        "frame_balanced_pos_weight": (
            (1 - prior_mean) / prior_mean if prior_mean > 0 else float("inf")
        ),
        "episode_balanced_pos_weight": (
            (1 - episode_balanced_prior_mean) / episode_balanced_prior_mean
            if episode_balanced_prior_mean > 0
            else float("inf")
        ),
        "hard_pos_weight": (1 - hard_rate) / hard_rate if hard_rate > 0 else float("inf"),
        "kept_by_reason": per_reason,
    }


def _cli() -> None:
    """Calibrate the event-memory knobs on a dataset and print a ready-to-paste YAML block.

    Examples
    --------
    LIBERO (single arm, eef pose inside state, chunk 8)::

        python -m starVLA.dataloader.keyframe_labeler \\
            --data-root playground/Datasets/LEROBOT_LIBERO_DATA \\
            --group-size 16 --exec-horizon 8

    RoboTwin (bimanual, eef in its own column, chunk 50)::

        python -m starVLA.dataloader.keyframe_labeler \\
            --data-root playground/Datasets/RoboTwin/Clean \\
            --gripper-cols 6 13 --pos-column observation.eef \\
            --eef-cols 0 1 2 3 4 5 --rot-cols \\
            --group-size 8 --exec-horizon 50
    """
    import argparse
    import glob
    import os

    import pandas as pd

    p = argparse.ArgumentParser(description=_cli.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="directory containing LeRobot datasets")
    p.add_argument("--episodes-per-dataset", type=int, default=15)
    p.add_argument("--max-datasets", type=int, default=12)
    p.add_argument("--state-column", default="observation.state")
    p.add_argument("--action-column", default="action")
    p.add_argument("--pos-column", default=None, help="separate Cartesian eef column, if any")
    p.add_argument("--gripper-cols", type=int, nargs="+", default=[-1])
    p.add_argument("--eef-cols", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rot-cols", type=int, nargs="*", default=None,
                   help="pass with no values to disable rotation channels")
    p.add_argument("--min-interval", type=int, default=12)
    p.add_argument("--group-size", type=int, required=True, help="datasets.vla_data.group_size")
    p.add_argument("--exec-horizon", type=int, required=True,
                   help="steps executed per server query at eval (LIBERO 8, RoboTwin 50)")
    a = p.parse_args()

    cfg = KeyframeConfig(
        gripper_cols=tuple(a.gripper_cols),
        eef_cols=tuple(a.eef_cols),
        rot_cols=None if a.rot_cols is None else tuple(a.rot_cols),
        min_interval=a.min_interval,
    )

    files = []
    for ds in sorted(glob.glob(os.path.join(a.data_root, "*")))[: a.max_datasets]:
        files += sorted(glob.glob(os.path.join(ds, "data", "chunk-*", "*.parquet")))[: a.episodes_per_dataset]
    if not files:
        raise SystemExit(f"No parquet episodes found under {a.data_root}")

    episodes = []
    for f in files:
        d = pd.read_parquet(f)
        episodes.append(
            (
                np.stack(d[a.state_column].values),
                np.stack(d[a.action_column].values),
                np.stack(d[a.pos_column].values) if a.pos_column else None,
            )
        )

    rep = calibrate(episodes, cfg, group_size=a.group_size, exec_horizon=a.exec_horizon)
    print(f"{len(episodes)} episodes, median length {rep['median_length']:.0f} frames")
    print(f"exec_horizon={a.exec_horizon} -> ~{rep['queries_per_episode']:.1f} server queries per episode")
    if rep["queries_per_episode"] < 8:
        print("  WARNING: with so few queries the bank sees only a handful of frames per episode,")
        print("           so event-gated writing has very little to select from. Consider a smaller")
        print("           exec_horizon at eval before concluding the mechanism does not help.")
    print()
    print(f"{'soft_sigma':>11}{'prior_mean':>12}{'pos_weight':>12}{'hits/group':>12}{'hits/rollout':>14}{'events/ep':>11}")
    for r in rep["sweep"]:
        print(
            f"{r['soft_sigma']:>11.1f}{r['prior_mean']:>12.4f}{r['pos_weight']:>12.2f}"
            f"{r['train_hits_per_group']:>12.2f}{r['infer_hits_per_episode']:>14.2f}{r['events_per_episode']:>11.1f}"
        )
    # Pick the smallest sigma that puts a useful number of events in front of the bank on BOTH sides.
    ok = [r for r in rep["sweep"] if r["train_hits_per_group"] >= 1.5 and r["infer_hits_per_episode"] >= 1.0]
    best = ok[0] if ok else rep["sweep"][-1]
    print()
    print("Suggested (verify against the table above before trusting it):")
    print("  framework.geoplan:")
    print(f"    event_supervision_pos_weight: {best['pos_weight']:.1f}")
    print(f"    event_aux_dim: {rep['event_aux_dim']}   # only if your eval client sends event_aux; else 0")
    print("    event_prior_controls_write: false   # keep the offline prior out of the write path")
    print("  datasets.vla_data.keyframe_event_prior:")
    print(f"    soft_sigma: {best['soft_sigma']}")
    print(f"    min_interval: {a.min_interval}")
    print(f"    gripper_cols: {list(a.gripper_cols)}")
    if a.pos_column:
        print(f"    pos_column: {a.pos_column}")
        print(f"    eef_cols: {list(a.eef_cols)}")
        if a.rot_cols is not None:
            print(f"    rot_cols: {list(a.rot_cols)}")


if __name__ == "__main__":
    _cli()
