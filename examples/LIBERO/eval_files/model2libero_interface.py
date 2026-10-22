# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.
"""

import os
import zlib
from collections import deque
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from examples.LIBERO.eval_files.aac_chunk import select_chunk_id, select_chunk_size


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size: Sequence[int] = (224, 224),
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self._server_metadata = meta
        self.event_aux_dim = int(meta.get("event_aux_dim", 0))
        if self.event_aux_dim not in (0, 9):
            raise ValueError(
                f"LIBERO client supports event_aux_dim 0 or 9, server requests {self.event_aux_dim}"
            )
        self._event_prev_state = None
        self._event_prev_gripper_action = None
        if self.event_aux_dim:
            print(
                "*** [ModelClient] forwarding 9-D causal event_aux requested by the server ***"
            )

        # --- Adaptive Action Chunking (AAC) ---
        # When enabled, the server samples LIBERO_AAC_N_SAMPLES candidate action
        # chunks per query; this client then decides how many LEADING steps to
        # execute from the per-step disagreement (entropy) among the candidates,
        # and replans when that plan runs out (instead of using a fixed
        # exec_horizon). See aac_chunk.select_chunk_size for the algorithm.
        self.aac = os.environ.get("LIBERO_AAC", "0") not in ("0", "", "false", "False")
        self.aac_n_samples = int(os.environ.get("LIBERO_AAC_N_SAMPLES", "20"))
        self.aac_method = os.environ.get("LIBERO_AAC_METHOD", "gaussian_bernoulli")
        self.aac_move_th = float(os.environ.get("LIBERO_AAC_MOVE_TH", "0.0"))
        self.aac_min_chunk = int(os.environ.get("LIBERO_AAC_MIN_CHUNK", "2"))
        self.aac_max_chunk = int(os.environ.get("LIBERO_AAC_MAX_CHUNK", str(self.action_chunk_size)))
        self.aac_chunk_id = os.environ.get("LIBERO_AAC_CHUNK_ID", "0")  # "0" | "mean"
        self.aac_debug = os.environ.get("LIBERO_AAC_DEBUG", "0") not in ("0", "", "false", "False")
        # Current per-round plan: unnormalized actions (k, D) to execute before replanning.
        self._aac_plan = None
        self._aac_ptr = 0
        if self.aac:
            # Temporal ensembling and AAC's dynamic chunking are mutually exclusive.
            action_ensemble = False
            print(
                f"*** [ModelClient] AAC ENABLED: n_samples={self.aac_n_samples}, "
                f"method={self.aac_method}, move_th={self.aac_move_th}, "
                f"chunk in [{self.aac_min_chunk}, {self.aac_max_chunk}], "
                f"chunk_id={self.aac_chunk_id} — action_ensemble forced OFF ***"
            )

        self.image_size: tuple = tuple(image_size)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        _eh_env = int(os.environ.get("LIBERO_EXEC_HORIZON", "0"))
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"exec_horizon: {self.action_chunk_size if _eh_env <= 0 else min(_eh_env, self.action_chunk_size)}, "
            f"server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        # OPTIONAL legacy-compat: reconstruct raw env actions from a policy that was trained with the
        # buggy datasets.vla_data.action_mode='delta' double-differencing. That transform produced
        # targets out[0]=a[0]-state0, out[t]=a[t]-a[t-1], so the true action chunk is recovered by
        # a = state0 + cumsum(out) using the chunk-start proprio state. This is a salvage path for
        # already-trained checkpoints; correctly-trained (action_mode='abs') checkpoints MUST leave
        # this OFF (the default). state index map: action [x,y,z,roll,pitch,yaw,gripper] <- state
        # [x,y,z,roll,pitch,yaw,pad,gripper] => indices [0,1,2,3,4,5,7] (the 'pad' at 6 is skipped).
        self.legacy_delta_recon = os.environ.get("EVAL_LEGACY_DELTA_RECON", "0") not in ("0", "", "false", "False")
        self._delta_recon_state_idx = [0, 1, 2, 3, 4, 5, 7]
        if self.legacy_delta_recon:
            print("*** [ModelClient] EVAL_LEGACY_DELTA_RECON=1: reconstructing raw actions via "
                  "a = state0 + cumsum(out) (compat for action_mode='delta' checkpoints) ***")

        # LIBERO_SEND_STATE=1: forward the 8-D proprio state ([x,y,z,roll,pitch,yaw,pad,gripper],
        # already in model order — same layout the eval harness concatenates) to the policy server.
        # ONLY for checkpoints trained with datasets.vla_data.include_state=true
        # (data_mix=libero_all_state); the server normalizes it with the training stats.
        # State-less checkpoints must keep this OFF (their state_encoder was never trained).
        # Default 0 = legacy behaviour (state stripped before sending).
        self.send_state = os.environ.get("LIBERO_SEND_STATE", "0") not in ("0", "", "false", "False")
        if self.send_state:
            print("*** [ModelClient] LIBERO_SEND_STATE=1: forwarding raw proprio state to the server "
                  "(requires an include_state=true checkpoint) ***")

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `exec_horizon` steps.
        self.raw_actions: Optional[np.ndarray] = None

        # Execution horizon: how many actions of each predicted chunk to execute before re-querying
        # the server. LIBERO_EXEC_HORIZON<=0 (default) keeps the legacy behaviour (execute the full
        # chunk = action_chunk_size). Set it to 1 to call the server EVERY step, which (in streaming
        # mode) updates the cross-frame memory every step — matching the training group's stride-1
        # consecutive frames — so streaming vs stateless becomes a clean, density-matched A/B.
        _eh = int(os.environ.get("LIBERO_EXEC_HORIZON", "0"))
        self.exec_horizon = self.action_chunk_size if _eh <= 0 else min(_eh, self.action_chunk_size)

        # Streaming-memory bookkeeping (mirror of the RoboTwin client): an episode id so the
        # server's per-episode memory bank never mixes two rollouts. Forwarded to the server inside
        # each example (with the env step as `timestep`); stateless models simply ignore them.
        #
        # When the harness supplies the per-task episode index (see ``reset``), the id is derived
        # from (task_description, episode_index) instead of a running counter. The server seeds its
        # sampling RNG from this id, so deriving it from the scene identity means the SAME scene
        # gets the SAME noise no matter how many episodes ran before it -- a 20-trial run and the
        # first 20 episodes of a 50-trial run then become comparable rollout-by-rollout. A running
        # counter would give them different ids, hence different noise, for identical scenes.
        self.episode_id = -1
        self._episode_index: Optional[int] = None

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str, episode_index: Optional[int] = None) -> None:
        self.task_description = task_description
        if episode_index is not None:
            self._episode_index = int(episode_index)
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self._event_prev_gripper_action = None
        self._event_prev_state = None
        self.raw_actions = None
        # AAC: drop any leftover plan so a new episode always replans from scratch.
        self._aac_plan = None
        self._aac_ptr = 0

        # New episode: refresh the episode id and clear the server's cross-frame memory so
        # streaming inference starts from an empty bank. Fail-fast so a broken reset can't silently
        # let a rollout read the previous episode's memory.
        if self._episode_index is not None:
            # Stable across runs and idempotent: reset() is called twice per episode (once by the
            # harness, once by step() at step==0), and both calls must yield the same id.
            self.episode_id = zlib.crc32(
                f"{task_description}#{self._episode_index}".encode()
            ) & 0x7FFFFFFF
        else:
            self.episode_id += 1
        try:
            self.client.reset(episode_id=self.episode_id)
        except Exception as e:
            raise RuntimeError(
                f"Policy server memory reset failed for episode_id={self.episode_id}; "
                f"streaming memory cannot be trusted. Check the server supports the 'reset' message."
            ) from e

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        # Force a clean reset at the FIRST step of every episode (robust even if the eval harness
        # did not call reset()), plus on any task change — this is what stops streaming memory
        # from leaking across rollouts that share the same instruction.
        if step == 0 or task_description != self.task_description:
            self.reset(task_description)

        # Full proprio remains local unless LIBERO_SEND_STATE=1. Event-guided checkpoints can still
        # receive the derived causal event_aux advertised by the server handshake.
        proprio_state = example.get("state", None)

        # Resize images to self.image_size if needed.
        if self.image_size and example.get("image"):
            resized = []
            target_hw = self.image_size  # (H, W)
            for img in example["image"]:
                arr = np.asarray(img)
                if arr.shape[:2] != target_hw:
                    arr = np.asarray(
                        Image.fromarray(arr).resize(
                            (target_hw[1], target_hw[0]), Image.BILINEAR
                        )
                    )
                resized.append(arr)
            example = {**example, "image": resized}

        # AAC path: dynamic chunk size from candidate disagreement (own scheduler).
        if self.aac:
            return self._step_aac(example, step, proprio_state)

        # Refresh chunk if needed (every `exec_horizon` steps; each server call also updates the
        # streaming memory, so a smaller exec_horizon => denser memory closer to training).
        if step % self.exec_horizon == 0 or self.raw_actions is None:
            server_example = self._make_server_example(example, step, proprio_state)
            vla_input = {
                "examples": [server_example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            chunk = np.asarray(actions_batch)[0]  # (T, D)

            # Optional legacy-delta reconstruction (see __init__). Recovers the true action chunk
            # from a model trained on the double-differenced target: a = state0 + cumsum(out).
            if self.legacy_delta_recon:
                if proprio_state is None:
                    raise ValueError(
                        "EVAL_LEGACY_DELTA_RECON=1 but no `state` provided in the example; the eval "
                        "harness must pass observation.state for reconstruction."
                    )
                st = np.asarray(proprio_state, dtype=np.float32).reshape(-1)
                D = chunk.shape[-1]
                if st.shape[0] >= max(self._delta_recon_state_idx) + 1 and D == len(self._delta_recon_state_idx):
                    anchor = st[self._delta_recon_state_idx]
                else:
                    # Fallback: anchor on the leading D state dims (covers state layouts without a pad).
                    anchor = st[:D]
                chunk = anchor[None, :] + np.cumsum(chunk, axis=0)

            self.raw_actions = chunk

        raw_actions = self.raw_actions[step % self.exec_horizon][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        self._record_event_step(proprio_state, raw_action["open_gripper"])
        return {"raw_action": raw_action}

    def _build_event_aux(self, proprio_state) -> Optional[np.ndarray]:
        """Build the causal signal expected by ``keyframe_labeler.compute_event_aux``."""
        if self.event_aux_dim == 0:
            return None
        if proprio_state is None:
            raise ValueError("event_aux_dim=9 requires LIBERO proprio state in every client step")
        state = np.asarray(proprio_state, dtype=np.float32).reshape(-1)
        if state.size < 6:
            raise ValueError(f"event_aux requires at least xyz+rpy state, got shape {state.shape}")

        dpose = np.zeros(6, dtype=np.float32)
        if self._event_prev_state is not None:
            dpose = state[:6] - self._event_prev_state[:6]
        speed = np.array([np.linalg.norm(dpose[:3])], dtype=np.float32)
        prev_grip = (
            0.0 if self.previous_gripper_action is None else float(self.previous_gripper_action)
        )
        dgrip = (
            0.0
            if self.previous_gripper_action is None or self._event_prev_gripper_action is None
            else prev_grip - float(self._event_prev_gripper_action)
        )
        return np.concatenate(
            [dpose, speed, np.array([prev_grip, dgrip], dtype=np.float32)]
        ).astype(np.float32)

    def _record_event_step(self, proprio_state, gripper_action) -> None:
        """Advance event state after choosing the command that will now be executed."""
        if proprio_state is not None:
            self._event_prev_state = np.asarray(
                proprio_state, dtype=np.float32
            ).reshape(-1).copy()
        self._event_prev_gripper_action = self.previous_gripper_action
        grip = float(np.asarray(gripper_action).reshape(-1)[0])
        self.previous_gripper_action = float(grip > 0.5)

    def _make_server_example(self, example: dict, step: int, proprio_state) -> dict:
        """Assemble the request payload's example (shared by AAC / legacy paths)."""
        server_example = {k: v for k, v in example.items() if k != "state"}
        if self.send_state and proprio_state is not None:
            _st = np.asarray(proprio_state, dtype=np.float32).reshape(-1)
            server_example["state"] = _st.reshape(1, -1)
        event_aux = self._build_event_aux(proprio_state)
        if event_aux is not None:
            server_example["event_aux"] = event_aux
        server_example["episode_id"] = int(self.episode_id)
        server_example["timestep"] = int(step)
        return server_example

    def _step_aac(self, example: dict, step: int, proprio_state) -> dict:
        """One env step under Adaptive Action Chunking.

        Replans (queries the server for ``aac_n_samples`` candidate chunks and
        selects a dynamic chunk size) whenever the current plan is exhausted,
        then hands back one action at a time.
        """
        if self.legacy_delta_recon:
            raise NotImplementedError(
                "EVAL_LEGACY_DELTA_RECON is not supported together with LIBERO_AAC "
                "(delta reconstruction would need to be applied per-candidate and in "
                "the normalized space used for entropy). Use an action_mode='abs' checkpoint."
            )

        if self._aac_plan is None or self._aac_ptr >= len(self._aac_plan):
            server_example = self._make_server_example(example, step, proprio_state)
            vla_input = {
                "examples": [server_example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
                "n_samples": self.aac_n_samples,
            }
            response = self.client.predict_action(vla_input)
            data = response["data"]
            try:
                unnorm = np.asarray(data["actions"])  # (N, T, D), un-normalized
                normalized = np.asarray(data["normalized_actions"])  # (N, T, D)
            except KeyError:
                raise KeyError(
                    "AAC needs both 'actions' and 'normalized_actions' in the server response; "
                    f"got keys={list(data.keys())}. Is the server running the AAC-enabled "
                    "policy_wrapper (which returns normalized_actions)?"
                )
            if unnorm.ndim != 3 or unnorm.shape[0] < 2:
                raise ValueError(
                    f"AAC expected >=2 candidate chunks (N>=2); got actions shape {unnorm.shape}. "
                    "Check LIBERO_AAC_N_SAMPLES and the server's n_samples handling."
                )

            # Build the pred_action_dict in the layout aac_chunk expects.
            # LIBERO action layout: [:3]=eef position, [3:6]=eef rotation (axis-angle
            # delta), [6]=gripper (open in [0,1] -> binarize at 0.5 for the Bernoulli split).
            pred = {
                "normalized_action": normalized,
                "action.end_effector_position": unnorm[:, :, :3],
                "action.end_effector_rotation": unnorm[:, :, 3:6],
                "action.gripper_close": (unnorm[:, :, 6] > 0.5).astype(int),
            }
            chunk_size, bd = select_chunk_size(
                pred,
                method=self.aac_method,
                move_th=self.aac_move_th,
                min_chunk=self.aac_min_chunk,
                max_chunk=self.aac_max_chunk,
            )
            chunk_id = select_chunk_id(pred) if self.aac_chunk_id == "mean" else 0
            self._aac_plan = unnorm[chunk_id, :chunk_size]  # (k, D)
            self._aac_ptr = 0
            if self.aac_debug:
                chunk_m = np.round(np.asarray(bd.get("chunk_m", [])), 3).tolist()
                print(
                    f"[AAC] step={step} replan -> k={chunk_size} "
                    f"(entropy_k={bd.get('best_k_entropy')}, motion_k={bd.get('min_id')}, "
                    f"N={unnorm.shape[0]}, T={unnorm.shape[1]}, chunk_id={chunk_id}) "
                    f"chunk_m[k=2..T]={chunk_m}"
                )

        raw = self._aac_plan[self._aac_ptr]
        self._aac_ptr += 1
        raw_action = {
            "world_vector": np.array(raw[:3]),
            "rotation_delta": np.array(raw[3:6]),
            "open_gripper": np.array(raw[6:7]),  # 1 = open; 0 = close
        }
        self._record_event_step(proprio_state, raw_action["open_gripper"])
        return {"raw_action": raw_action}

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
