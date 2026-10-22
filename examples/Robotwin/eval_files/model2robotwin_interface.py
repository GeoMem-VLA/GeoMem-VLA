import atexit
import os
from collections import deque
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None

try:
    from .aac_chunk_robotwin import select_chunk_id as select_aac_chunk_id
    from .aac_chunk_robotwin import select_chunk_size as select_aac_chunk_size
except ImportError:
    # RoboTwin imports policy modules by filename after adding eval_files/ to sys.path,
    # while tests may import this file through the examples.Robotwin namespace.
    from aac_chunk_robotwin import select_chunk_id as select_aac_chunk_id
    from aac_chunk_robotwin import select_chunk_size as select_aac_chunk_size

try:
    from .pace_chunk_robotwin import select_pace_horizon
except ImportError:
    from pace_chunk_robotwin import select_pace_horizon


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = None
        self.state_norm_stats = None
        self.raw_actions = None

        # Streaming-memory bookkeeping: a monotonically-increasing episode id so the server's
        # per-episode memory bank never mixes two rollouts. Forwarded to the server inside each
        # example (together with the env step as `timestep`); stateless models simply ignore them.
        self.episode_id = -1

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta["action_chunk_size"]

        # Execution horizon: how many actions of each predicted chunk to execute open-loop before
        # re-observing + re-predicting. ROBOTWIN_EXEC_HORIZON<=0 (default) keeps the legacy behaviour
        # (execute the full chunk). A smaller value (e.g. 8/16) re-plans more often -> tighter
        # closed-loop control, less open-loop drift, and denser memory entries (closer to the
        # training-time group density). It never exceeds action_chunk_size (can't use more actions
        # than the model predicts).
        # ROBOTWIN_EXEC_HORIZON=auto picks the horizon from one chunk's motion profile (see
        # _decide_horizon). ROBOTWIN_EXEC_HORIZON=pace selects prominent low-speed phase
        # boundaries independently for each arm. ROBOTWIN_EXEC_HORIZON=aac requests N parallel
        # candidates and applies Adaptive Action Chunking's cross-sample uncertainty rule (see
        # _select_aac_action_and_horizon).
        _eh = os.environ.get("ROBOTWIN_EXEC_HORIZON", "0").strip().lower()
        self.adaptive_horizon = _eh in ("auto", "adaptive")
        self.pace_horizon = _eh == "pace"
        self.aac_horizon = _eh == "aac"
        if self.adaptive_horizon or self.pace_horizon or self.aac_horizon:
            self.exec_horizon = self.action_chunk_size
        else:
            try:
                _eh = int(_eh)
            except ValueError as e:
                raise ValueError(
                    "ROBOTWIN_EXEC_HORIZON must be an integer, 'auto', 'pace', or 'aac'; "
                    f"got {_eh!r}"
                ) from e
            self.exec_horizon = self.action_chunk_size if _eh <= 0 else min(_eh, self.action_chunk_size)

        # Adaptive-horizon knobs. The horizon is chosen so the chunk is cut where its motion has
        # essentially finished, never in the middle of it: ROBOTWIN_ADAPT_QUANTILE is the fraction
        # of the chunk's total joint travel that must be executed before re-planning. Bounded by
        # ROBOTWIN_ADAPT_MIN/MAX. A chunk whose total travel falls below ROBOTWIN_ADAPT_STILL is
        # treated as "the policy is holding still", where the cheapest thing to do is re-observe
        # soon. ROBOTWIN_ADAPT_GRIP_MARGIN keeps a gripper open/close transition from being cut off.
        # q=0.7 replayed over the click_bell dumps covers 100% of the predicted presses while
        # letting the horizon vary over 26-40; q=0.9 also covers them but collapses to a constant
        # 40, which is just the fixed setting again. The cap defaults to 0.8*chunk because
        # re-observing any later than that lands inside the aliased stretch on the click tasks
        # (click_bell: 0.90 at 40 vs 0.10 at 50).
        self.adapt_quantile = float(os.environ.get("ROBOTWIN_ADAPT_QUANTILE", "0.7"))
        self.adapt_min = int(os.environ.get("ROBOTWIN_ADAPT_MIN", "8"))
        self.adapt_max = int(os.environ.get(
            "ROBOTWIN_ADAPT_MAX", str(max(1, int(0.8 * self.action_chunk_size)))))
        self.adapt_still = float(os.environ.get("ROBOTWIN_ADAPT_STILL", "0.05"))
        self.adapt_grip_margin = int(os.environ.get("ROBOTWIN_ADAPT_GRIP_MARGIN", "4"))

        # Phase-Aware Chunk Execution (PACE-style) settings.  The selector operates directly on
        # this checkpoint's 14-D absolute-joint chunk: each arm gets its own smoothed first-
        # difference speed curve, significant valleys are pooled, and the earliest boundary wins.
        # The original PACE paper calibrates its prominence threshold from training demonstrations;
        # this plug-in port uses a scale-free local prominence ratio so it can run without a
        # calibration dataset.  Keep this distinction explicit when reporting results.
        self.pace_min = int(os.environ.get("ROBOTWIN_PACE_MIN", "5"))
        self.pace_max = int(os.environ.get(
            "ROBOTWIN_PACE_MAX", str(self.action_chunk_size)))
        self.pace_smooth_window = int(os.environ.get("ROBOTWIN_PACE_SMOOTH_WINDOW", "7"))
        self.pace_min_separation = int(os.environ.get("ROBOTWIN_PACE_MIN_SEPARATION", "10"))
        self.pace_prominence = float(os.environ.get("ROBOTWIN_PACE_PROMINENCE", "0.30"))
        self.pace_prominence_window = int(os.environ.get("ROBOTWIN_PACE_PROM_WINDOW", "10"))
        # Gripper protection is a RoboTwin-specific extension, not part of PACE: guard only the
        # first predicted transition so a click/grasp is not perpetually discarded, while a later
        # open/close event cannot drag an otherwise useful early phase boundary across the chunk.
        self.pace_grip_protect = os.environ.get("ROBOTWIN_PACE_GRIP_PROTECT", "1") not in (
            "0", "", "false", "False"
        )
        self.pace_grip_margin = int(os.environ.get("ROBOTWIN_PACE_GRIP_MARGIN", "4"))
        self.pace_grip_threshold = float(os.environ.get("ROBOTWIN_PACE_GRIP_THRESHOLD", "0.5"))
        self.pace_verbose = os.environ.get("ROBOTWIN_PACE_VERBOSE", "0") not in (
            "0", "", "false", "False"
        )
        self.pace_max = min(self.pace_max, self.action_chunk_size)
        if self.pace_horizon:
            if self.action_mode != "abs":
                raise ValueError(
                    "ROBOTWIN_EXEC_HORIZON=pace requires action_mode='abs': its speed profiles "
                    "are first differences of absolute joint targets."
                )
            invalid_pace = {
                "ROBOTWIN_PACE_MIN": self.pace_min < 1,
                "ROBOTWIN_PACE_MAX": self.pace_max < self.pace_min,
                "ROBOTWIN_PACE_SMOOTH_WINDOW": self.pace_smooth_window < 1,
                "ROBOTWIN_PACE_MIN_SEPARATION": self.pace_min_separation < 1,
                "ROBOTWIN_PACE_PROMINENCE": self.pace_prominence < 0,
                "ROBOTWIN_PACE_PROM_WINDOW": self.pace_prominence_window < 1,
                "ROBOTWIN_PACE_GRIP_MARGIN": self.pace_grip_margin < 0,
                "ROBOTWIN_PACE_GRIP_THRESHOLD": self.pace_grip_threshold < 0,
            }
            bad = [name for name, is_invalid in invalid_pace.items() if is_invalid]
            if bad:
                raise ValueError(f"Invalid PACE setting(s): {', '.join(bad)}.")

        # Full AAC settings. Unlike `auto`, AAC estimates uncertainty from N independently sampled
        # chunks at the same observation. RoboTwin predicts 14-D ABSOLUTE joint targets rather than
        # the paper's 7-D delta-EEF commands, so entropy defaults to temporal differences of the
        # normalized targets and the motion threshold is expressed in joint-space radians.
        # `move_th` must ultimately be calibrated from run_probe_aac_q99.sh; 0.05 is deliberately a
        # conservative starting point, not a task-specific optimum.
        self.aac_n_samples = int(os.environ.get("ROBOTWIN_AAC_N_SAMPLES", "20"))
        self.aac_method = os.environ.get(
            "ROBOTWIN_AAC_METHOD", "gaussian_bernoulli"
        ).strip().lower()
        self.aac_entropy_space = os.environ.get(
            "ROBOTWIN_AAC_ENTROPY_SPACE", "delta"
        ).strip().lower()
        self.aac_selector = os.environ.get("ROBOTWIN_AAC_SELECTOR", "elbow").strip().lower()
        self.aac_move_th = float(os.environ.get("ROBOTWIN_AAC_MOVE_TH", "0.05"))
        self.aac_magnitude_mode = os.environ.get(
            "ROBOTWIN_AAC_MAGNITUDE_MODE", "net"
        ).strip().lower()
        self.aac_min = int(os.environ.get("ROBOTWIN_AAC_MIN", "8"))
        self.aac_max = int(os.environ.get(
            "ROBOTWIN_AAC_MAX", str(max(1, int(0.8 * self.action_chunk_size)))))
        self.aac_min = min(self.aac_min, self.action_chunk_size)
        self.aac_max = min(self.aac_max, self.action_chunk_size)
        self.aac_chunk_selector = os.environ.get(
            "ROBOTWIN_AAC_CHUNK_SELECTOR", "first"
        ).strip().lower()
        self.aac_verbose = os.environ.get("ROBOTWIN_AAC_VERBOSE", "0") not in (
            "0", "", "false", "False"
        )
        if self.aac_horizon:
            if self.action_mode != "abs":
                raise ValueError(
                    "ROBOTWIN_EXEC_HORIZON=aac currently requires action_mode='abs': the "
                    "RoboTwin AAC motion floor is defined on absolute joint targets."
                )
            if self.aac_n_samples < 2:
                raise ValueError(
                    f"ROBOTWIN_AAC_N_SAMPLES must be >= 2, got {self.aac_n_samples}"
                )
            if self.aac_n_samples < 10 and self.aac_method == "gaussian_bernoulli":
                print(
                    f"*** [AAC] WARNING: n_samples={self.aac_n_samples} is small for a 6-D "
                    "per-arm covariance; 20 is recommended. ***"
                )
            if self.aac_chunk_selector not in ("first", "mean"):
                raise ValueError(
                    "ROBOTWIN_AAC_CHUNK_SELECTOR must be 'first' or 'mean'; "
                    f"got {self.aac_chunk_selector!r}"
                )
            valid_aac_options = {
                "method": (self.aac_method, ("gaussian_bernoulli", "variance")),
                "entropy space": (self.aac_entropy_space, ("abs", "delta")),
                "selector": (self.aac_selector, ("elbow", "argmin")),
                "magnitude mode": (self.aac_magnitude_mode, ("net", "path")),
            }
            for label, (value, choices) in valid_aac_options.items():
                if value not in choices:
                    raise ValueError(
                        f"Invalid AAC {label} {value!r}; expected one of {choices}."
                    )
            if self.aac_min < 1 or self.aac_max < self.aac_min:
                raise ValueError(
                    "AAC bounds must satisfy 1 <= ROBOTWIN_AAC_MIN <= ROBOTWIN_AAC_MAX; "
                    f"got {self.aac_min} / {self.aac_max}."
                )

        # Chunk bookkeeping. Under a fixed horizon these track exactly what `step % exec_horizon`
        # used to; under an adaptive one the chunk length varies, so the offset must be explicit.
        self._chunk_start = 0
        self._cur_horizon = self.exec_horizon

        # Optional temporal smoothing of each predicted chunk (jitter mitigation). The action-dump
        # analysis showed the model's absolute-qpos chunks carry per-step high-frequency sampling
        # noise (~53% direction-flip rate vs 0.6% in the training data), which TOPP tracking turns
        # into visible arm jitter. ROBOTWIN_SMOOTH_CHUNK=<odd window, e.g. 5/7/9> low-pass filters
        # the 12 ARM-JOINT dims of each chunk along time (Savitzky-Golay if scipy is available,
        # else a centered moving average). Gripper dims are NEVER smoothed (binary open/close
        # commands must keep sharp transitions). Unset or 0 (default) = OFF (legacy behaviour).
        _sw = int(os.environ.get("ROBOTWIN_SMOOTH_CHUNK", "0"))
        if _sw > 0 and _sw % 2 == 0:
            _sw += 1  # enforce odd window
        self.smooth_window = _sw

        # ROBOTWIN_SEND_STATE=1: forward the env proprio state (raw qpos, reordered from env order
        # to model order = [L_joints, R_joints, L_grip, R_grip]) to the policy server. ONLY for
        # checkpoints trained with datasets.vla_data.include_state=true (the server normalizes it
        # with the training stats; state-less checkpoints must keep this OFF since their
        # state_encoder was never trained). Default 0 = legacy behaviour (state stripped).
        self.send_state = os.environ.get("ROBOTWIN_SEND_STATE", "0") not in ("0", "", "false", "False")
        # env order [L1-6, Lgrip, R1-6, Rgrip] -> model order [L1-6, R1-6, Lgrip, Rgrip]
        self._state_env2model_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]

        # Diagnostic action dump: set ROBOTWIN_DUMP_ACTIONS=<dir> to save, per episode, the raw
        # predicted chunks (model order, un-normalized) + the actually-executed action trajectory
        # (env order) as .npy, plus a per-joint plot. Lets us see whether the jitter is INSIDE a
        # chunk (model prediction problem) or only at chunk boundaries (open-loop re-plan problem).
        self.dump_dir = os.environ.get("ROBOTWIN_DUMP_ACTIONS", "").strip()
        self.horizon_verbose = os.environ.get("ROBOTWIN_HORIZON_VERBOSE", "0") not in (
            "0", "", "false", "False"
        )
        self._dump_exec = []       # list[(step, action14 env-order)]
        self._dump_chunks = []     # list[(start_step, chunk[N,D] model-order)]
        self._dump_horizons = []   # list[horizon actually granted to each chunk]
        if self.dump_dir:
            # An episode is only flushed when the NEXT one resets, so the final episode of a run
            # would otherwise be lost -- and it is just as likely to be a success worth comparing
            # against as any other.
            atexit.register(self._flush_dump)
        horizon_mode = (
            "aac" if self.aac_horizon
            else (
                "pace" if self.pace_horizon
                else ("auto" if self.adaptive_horizon else str(self.exec_horizon))
            )
        )
        self.horizon_mode = horizon_mode
        horizon_details = ""
        if self.adaptive_horizon:
            horizon_details = (
                f" (q={self.adapt_quantile}, bounds=[{self.adapt_min},{self.adapt_max}])"
            )
        elif self.pace_horizon:
            horizon_details = (
                f" (smooth={self.pace_smooth_window}, separation={self.pace_min_separation}, "
                f"prominence={self.pace_prominence}, prom_window={self.pace_prominence_window}, "
                f"grip_protect={self.pace_grip_protect}, grip_margin={self.pace_grip_margin}, "
                f"bounds=[{self.pace_min},{self.pace_max}])"
            )
        elif self.aac_horizon:
            horizon_details = (
                f" (N={self.aac_n_samples}, {self.aac_entropy_space}/{self.aac_selector}, "
                f"move_th={self.aac_move_th}, bounds=[{self.aac_min},{self.aac_max}], "
                f"candidate={self.aac_chunk_selector})"
            )
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"exec_horizon: {horizon_mode}/{self.action_chunk_size}{horizon_details}, "
            f"smooth_chunk: {self.smooth_window if self.smooth_window > 0 else 'off'}, "
            f"send_state: {self.send_state}, "
            f"server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        # Flush the just-finished episode's action dump (if any) before starting a new one.
        if self.dump_dir:
            self._flush_dump()

        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self._chunk_start = 0
        self._cur_horizon = self.exec_horizon
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None

        # New episode: bump the episode id and tell the server to clear its cross-frame memory so
        # streaming inference starts from a clean bank. Fail-fast: a silently-swallowed reset would
        # let a streaming rollout read the PREVIOUS episode's memory and make results impossible to
        # trust, so surface the error instead of only warning.
        self.episode_id += 1
        try:
            self.client.reset(episode_id=self.episode_id)
        except Exception as e:
            raise RuntimeError(
                f"Policy server memory reset failed for episode_id={self.episode_id}; "
                f"streaming memory cannot be trusted. Check the server supports the 'reset' message."
            ) from e

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        # if state is not None:
        #     state = self.normalize_state(state, self.state_norm_stats)
        #     state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        #     example["state"] = state.reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        # Episode boundary: force a clean reset at the FIRST step of every episode (robust even if
        # the eval harness's reset_model() were not called), plus on any task change. This is what
        # guarantees streaming memory does not leak across rollouts.
        if step == 0 or task_description != self.task_description:
            self.reset(task_description)
            # Re-store initial state after reset if in delta/rel mode
            if self.action_mode in ["delta", "rel"] and state is not None:
                self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        example_copy = example.copy()
        _state_raw = example_copy.pop("state")
        _st = None
        if _state_raw is not None and (self.send_state or self.aac_horizon):
            _st = np.asarray(_state_raw, dtype=np.float32).reshape(-1)
            _st = _st[self._state_env2model_idx]
        if self.send_state and _st is not None:
            # (1, 14): matches the training-time state shape (observation_indices=[0]).
            # Sent RAW; the server normalizes it with the training-time transform.
            example_copy["state"] = _st.reshape(1, -1)
        # Forward streaming-memory metadata to the server. `timestep` uses the env step index so
        # it is strictly increasing within an episode (required by causal memory retrieval). Only
        # the frames actually sent to the server (once per action chunk) become memory entries.
        example_copy["episode_id"] = int(self.episode_id)
        example_copy["timestep"] = int(step)
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key
        if self.aac_horizon:
            vla_input["n_samples"] = self.aac_n_samples

        # Re-query the server once the current chunk's granted horizon is used up. The model always
        # predicts a full chunk; only its first `_cur_horizon` actions are executed before we
        # re-observe. Under a fixed horizon this reproduces the old `step % exec_horizon` schedule.
        if self.raw_actions is None or (step - self._chunk_start) >= self._cur_horizon:
            response = self.client.predict_action(vla_input)
            data = response["data"]
            if self.aac_horizon:
                raw_actions, aac_horizon, aac_diag = self._select_aac_action_and_horizon(
                    data=data,
                    state_model=_st,
                )
            else:
                # Server already un-normalized via the training-time transform.
                raw_actions = np.array(data["actions"][0])  # (chunk, D)

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

            # Optional jitter mitigation: low-pass the arm-joint dims of the fresh chunk.
            if self.smooth_window > 0:
                self.raw_actions = self._smooth_chunk(self.raw_actions)

            self._chunk_start = step
            if self.aac_horizon:
                self._cur_horizon = aac_horizon
                if self.aac_verbose:
                    print(
                        f"[AAC] ep={int(self.episode_id)} t={int(step)} "
                        f"N={aac_diag['n_samples']} k={self._cur_horizon} "
                        f"entropy={aac_diag['k_entropy']} motion={aac_diag['k_motion']} "
                        f"candidate={aac_diag['chunk_id']}"
                    )
            elif self.pace_horizon:
                self._cur_horizon, pace_diag = select_pace_horizon(
                    self.raw_actions,
                    min_horizon=self.pace_min,
                    max_horizon=self.pace_max,
                    smooth_window=self.pace_smooth_window,
                    min_separation=self.pace_min_separation,
                    prominence_threshold=self.pace_prominence,
                    prominence_window=self.pace_prominence_window,
                    protect_gripper=self.pace_grip_protect,
                    gripper_margin=self.pace_grip_margin,
                    gripper_threshold=self.pace_grip_threshold,
                )
                if self.pace_verbose:
                    selected = (
                        "fallback"
                        if pace_diag["fallback"]
                        else f"{pace_diag['selected_arm']}@{pace_diag['selected_boundary']}"
                    )
                    left = [int(item["horizon"]) for item in pace_diag["left_candidates"]]
                    right = [int(item["horizon"]) for item in pace_diag["right_candidates"]]
                    print(
                        f"[PACE] ep={int(self.episode_id)} t={int(step)} "
                        f"phase={selected} final_k={self._cur_horizon} "
                        f"candidates=L{left}/R{right} grip_floor={pace_diag['gripper_floor']} "
                        f"bounds={pace_diag['bounds']}"
                    )
            else:
                self._cur_horizon = (
                    self._decide_horizon(self.raw_actions)
                    if self.adaptive_horizon else self.exec_horizon
                )

            if self.horizon_verbose:
                print(
                    f"[HORIZON] ep={int(self.episode_id)} t={int(step)} "
                    f"mode={self.horizon_mode} k={int(self._cur_horizon)}"
                )

            # Diagnostic: record the freshly predicted chunk (model order, un-normalized).
            if self.dump_dir:
                self._dump_chunks.append((int(step), np.array(self.raw_actions).copy()))
                self._dump_horizons.append(int(self._cur_horizon))

        action_idx = step - self._chunk_start
        # Safety clamp (the horizon never exceeds the chunk length, so this normally never fires).
        if action_idx >= len(self.raw_actions):
            action_idx = len(self.raw_actions) - 1

        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]

        # Diagnostic: record the action actually executed this step (env order).
        if self.dump_dir:
            self._dump_exec.append((int(step), np.array(current_action).copy()))

        return current_action

    def _select_aac_action_and_horizon(
        self,
        data: dict,
        state_model: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int, dict]:
        """Validate a multi-sample response and let AAC choose this replan horizon.

        Candidate selection is intentionally independent of horizon selection. ``first`` executes
        candidate 0, which has the same sampling semantics as the legacy single-chunk path; ``mean``
        executes the candidate nearest the sample-set mean and is available only as an explicit
        ablation because it changes the action policy as well as the horizon.
        """
        try:
            unnorm = np.asarray(data["actions"])
            normalized = np.asarray(data["normalized_actions"])
        except KeyError as e:
            raise KeyError(
                "ROBOTWIN_EXEC_HORIZON=aac requires both 'actions' and "
                f"'normalized_actions' from the policy server; got keys={list(data.keys())}."
            ) from e

        if unnorm.ndim != 3 or normalized.ndim != 3:
            raise ValueError(
                "AAC expected actions/normalized_actions with shape (N,T,D), got "
                f"{unnorm.shape} / {normalized.shape}."
            )
        if unnorm.shape != normalized.shape:
            raise ValueError(
                f"AAC candidate shape mismatch: {unnorm.shape} vs {normalized.shape}."
            )
        if unnorm.shape[0] < 2:
            raise ValueError(
                f"AAC requested n_samples={self.aac_n_samples}, but the server returned "
                f"actions shape {unnorm.shape}; this framework likely ignored n_samples."
            )
        if unnorm.shape[2] != 14:
            raise ValueError(f"RoboTwin AAC requires 14-D actions, got shape {unnorm.shape}.")

        chunk_id = (
            select_aac_chunk_id(normalized)
            if self.aac_chunk_selector == "mean" else 0
        )
        ref_state = None
        if state_model is not None:
            ref_state = np.asarray(state_model, dtype=np.float32).reshape(-1)[:12]

        horizon, diag = select_aac_chunk_size(
            normalized=normalized,
            unnorm=unnorm,
            method=self.aac_method,
            entropy_space=self.aac_entropy_space,
            selector=self.aac_selector,
            move_th=self.aac_move_th,
            magnitude_mode=self.aac_magnitude_mode,
            ref_state=ref_state,
            min_chunk=self.aac_min,
            max_chunk=self.aac_max,
            chunk_id=chunk_id,
        )
        diag["chunk_id"] = int(chunk_id)
        return np.asarray(unnorm[chunk_id]), int(horizon), diag

    def _decide_horizon(self, chunk: np.ndarray) -> int:
        """How many steps of this chunk to execute before re-observing.

        Motion is not spread evenly across a chunk. On the click tasks the press sits around steps
        18-40 while the first twenty are approach and hover, so a short fixed horizon truncates the
        chunk before its decisive motion ever runs, re-plans, and truncates again -- the policy
        predicts a correct press over and over and never performs one (exec_horizon=8 scored 0.00
        on click_bell while predicting presses at 92-118% of the demonstrated amplitude).

        So the cut is placed where the motion is essentially spent: execute until
        `adapt_quantile` of the chunk's total joint travel is behind us. Chunks that front-load
        their motion yield a short horizon and tight closed-loop control; chunks that carry a late
        motion get enough room to finish it. This is the `move_th` half of Adaptive Action Chunking
        (arXiv 2604.04161) -- their other half, the cross-sample entropy elbow, needs N parallel
        samples per observation AND would cut right at the press, which is where uncertainty peaks.

        Two guards on top: a chunk that barely moves means the policy is holding still, where
        re-observing sooner is the only way out; and a gripper transition must never be cut mid-way.
        """
        lo = max(1, min(self.adapt_min, len(chunk)))
        hi = max(lo, min(self.adapt_max, len(chunk)))

        # A gripper transition has to be seen through even when the arm itself is parked, which is
        # exactly what a click looks like: hold the pose, close the fingers.
        grip = np.asarray(chunk[:, 12:14], dtype=np.float64)
        transitions = np.nonzero(np.abs(np.diff(grip, axis=0)) > 0.5)[0]
        grip_floor = int(transitions.max()) + 1 + self.adapt_grip_margin if transitions.size else 0

        arm = np.asarray(chunk[:, :12], dtype=np.float64)
        speed = np.linalg.norm(np.diff(arm, axis=0), axis=1)
        total = float(speed.sum())
        if total < self.adapt_still:
            return int(np.clip(max(lo, grip_floor), lo, hi))

        cumulative = np.cumsum(speed) / total
        horizon = int(np.argmax(cumulative >= self.adapt_quantile)) + 1
        return int(np.clip(max(horizon, grip_floor), lo, hi))

    def _smooth_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Temporally smooth the ARM-JOINT dims of a predicted chunk (model order).

        Model-order layout: [L_joints 0..5, R_joints 6..11, L_gripper 12, R_gripper 13] — only
        dims 0..11 are filtered; gripper dims pass through untouched. Uses Savitzky-Golay
        (order 3) when scipy is available — it preserves the macro trajectory shape while
        killing per-step sampling noise — else falls back to a centered moving average with
        edge-replication padding. Window is self.smooth_window (odd, >= 3).
        """
        T = chunk.shape[0]
        win = min(self.smooth_window, T if T % 2 == 1 else T - 1)
        if win < 3:
            return chunk
        smoothed = np.array(chunk, dtype=np.float64, copy=True)
        arm = smoothed[:, :12]
        try:
            from scipy.signal import savgol_filter
            arm_s = savgol_filter(arm, window_length=win, polyorder=min(3, win - 1), axis=0)
        except ImportError:
            pad = win // 2
            padded = np.concatenate([np.repeat(arm[:1], pad, 0), arm, np.repeat(arm[-1:], pad, 0)], axis=0)
            kernel = np.ones(win) / win
            arm_s = np.stack(
                [np.convolve(padded[:, d], kernel, mode="valid") for d in range(arm.shape[1])],
                axis=1,
            )
        smoothed[:, :12] = arm_s
        return smoothed.astype(chunk.dtype, copy=False)

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute actions."""
        return rel_actions + self.initial_state

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _flush_dump(self) -> None:
        """Save the just-finished episode's predicted chunks + executed trajectory (npy + plots).

        Produces, under ``ROBOTWIN_DUMP_ACTIONS/episode_XXXX/``:
          - ``executed_actions.npy`` (T, 14) env-order commands actually sent to take_action;
          - ``executed_steps.npy`` (T,) their env step indices;
          - ``chunk_starts.npy`` + ``chunk_III_tSS.npy`` each predicted chunk (model order);
          - ``executed_trajectory.png`` per-joint executed curve with chunk boundaries (red);
          - ``first_chunk.png`` the first predicted chunk (to inspect INTRA-chunk smoothness).
        """
        if not self.dump_dir or (not self._dump_exec and not self._dump_chunks):
            return
        ep_dir = os.path.join(self.dump_dir, f"episode_{int(self.episode_id):04d}")
        try:
            os.makedirs(ep_dir, exist_ok=True)
        except Exception as e:
            print(f"[dump] mkdir failed: {e}")
            self._dump_exec, self._dump_chunks = [], []
            return

        exec_steps = np.array([s for s, _ in self._dump_exec], dtype=np.int64)
        exec_arr = (
            np.stack([a for _, a in self._dump_exec], axis=0)
            if self._dump_exec else np.zeros((0, 14), dtype=np.float32)
        )
        chunk_starts = np.array([s for s, _ in self._dump_chunks], dtype=np.int64)
        np.save(os.path.join(ep_dir, "executed_actions.npy"), exec_arr)
        np.save(os.path.join(ep_dir, "executed_steps.npy"), exec_steps)
        np.save(os.path.join(ep_dir, "chunk_starts.npy"), chunk_starts)
        np.save(os.path.join(ep_dir, "chunk_horizons.npy"),
                np.array(self._dump_horizons, dtype=np.int64))
        for i, (s, c) in enumerate(self._dump_chunks):
            np.save(os.path.join(ep_dir, f"chunk_{i:03d}_t{int(s)}.npy"), np.asarray(c))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = ["L1", "L2", "L3", "L4", "L5", "L6", "Lgrip",
                      "R1", "R2", "R3", "R4", "R5", "R6", "Rgrip"]
            if exec_arr.shape[0] > 0:
                fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
                axes = axes.ravel()
                for d in range(min(14, exec_arr.shape[1])):
                    ax = axes[d]
                    ax.plot(exec_steps, exec_arr[:, d], lw=1.0)
                    for cs in chunk_starts:
                        ax.axvline(cs, color="r", ls="--", lw=0.5, alpha=0.5)
                    ax.set_title(labels[d] if d < len(labels) else f"dim{d}", fontsize=9)
                    ax.grid(True, alpha=0.3)
                fig.suptitle(f"episode {int(self.episode_id)} executed actions (red dashed = chunk boundary)")
                fig.tight_layout()
                fig.savefig(os.path.join(ep_dir, "executed_trajectory.png"), dpi=100)
                plt.close(fig)

            if self._dump_chunks:
                c0 = np.asarray(self._dump_chunks[0][1])
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for d in range(c0.shape[1]):
                    ax2.plot(range(c0.shape[0]), c0[:, d], lw=1.0)
                ax2.set_title(
                    f"episode {int(self.episode_id)} first predicted chunk "
                    f"(model order, {c0.shape[0]} steps) — inspect intra-chunk smoothness"
                )
                ax2.set_xlabel("within-chunk step")
                ax2.grid(True, alpha=0.3)
                fig2.tight_layout()
                fig2.savefig(os.path.join(ep_dir, "first_chunk.png"), dpi=100)
                plt.close(fig2)
        except Exception as e:
            print(f"[dump] plot skipped ({e}); npy still saved to {ep_dir}")

        horizons = (f", horizons {min(self._dump_horizons)}-{max(self._dump_horizons)}"
                    f" (mean {np.mean(self._dump_horizons):.1f})" if self._dump_horizons else "")
        print(
            f"[dump] episode {int(self.episode_id)}: "
            f"{exec_arr.shape[0]} steps, {len(self._dump_chunks)} chunks{horizons} -> {ep_dir}"
        )
        self._dump_exec = []
        self._dump_chunks = []
        self._dump_horizons = []


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
    )


def reset_model(model):
    model.reset(task_description="")


# Optional scene-pose recording, so a failure can be attributed to position / orientation /
# release after the fact instead of being guessed from the video. See scene_probe.py.
_PROBE = None
if os.environ.get("ROBOTWIN_PROBE_SCENE", "").strip():
    try:
        from scene_probe import SceneProbe

        _PROBE = SceneProbe(os.environ["ROBOTWIN_PROBE_SCENE"].strip())
        atexit.register(_PROBE.flush)
        print(f"*** scene probe -> {_PROBE.out_dir} ***")
    except Exception as e:  # never let diagnostics break an evaluation run
        print(f"[probe] disabled ({e})")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)

    if _PROBE is not None:
        # step==0 means the harness has started a new episode, so close out the previous one.
        if TASK_ENV.take_action_cnt <= 1:
            _PROBE.flush()
        _PROBE.record(TASK_ENV)
