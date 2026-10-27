# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor

# Optional determinism for streaming eval. predict_action() draws a fresh flow-matching noise
# tensor per action chunk from the GLOBAL torch RNG, which is never reset -- so the noise an
# episode sees depends on how many chunks every preceding episode happened to consume. One
# flipped rollout therefore shifts the noise for the whole remainder of the run, and two runs of
# the same config are not reproducible. Setting STARVLA_EVAL_SEED=<int> re-seeds torch at every
# episode boundary from (seed, episode_id), which makes each episode's noise a pure function of
# its own id: runs become bit-reproducible and A/B comparisons share one noise stream (common
# random numbers). UNSET (default) preserves the legacy non-deterministic behaviour.
_EVAL_SEED_ENV = os.environ.get("STARVLA_EVAL_SEED", "").strip()
_EVAL_SEED = int(_EVAL_SEED_ENV) if _EVAL_SEED_ENV else None


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Optional inference-time override of the flow-matching denoising step count.
        # Diagnosis (see examples/Robotwin action-dump analysis): the checkpoint default
        # (num_inference_timesteps=4 Euler steps) leaves ~0.015 rad high-frequency noise in
        # absolute-joint action chunks -> visible arm jitter on RoboTwin. More steps = smoother
        # chunks at the cost of slower inference. UNSET (default) keeps the checkpoint's
        # training-time value, so legacy behaviour is fully preserved.
        _nit_env = os.environ.get("STARVLA_NUM_INFERENCE_TIMESTEPS", "").strip()
        if _nit_env:
            _nit = int(_nit_env)
            _am = getattr(framework, "action_model", None)
            if _am is not None and hasattr(_am, "num_inference_timesteps"):
                _old = _am.num_inference_timesteps
                _am.num_inference_timesteps = _nit
                logging.info(
                    "PolicyServerWrapper: num_inference_timesteps overridden by env "
                    "STARVLA_NUM_INFERENCE_TIMESTEPS: %s -> %s", _old, _nit,
                )
            else:
                logging.warning(
                    "PolicyServerWrapper: STARVLA_NUM_INFERENCE_TIMESTEPS=%s set but the loaded "
                    "framework has no action_model.num_inference_timesteps; ignored.", _nit_env,
                )

        if _EVAL_SEED is not None:
            logging.info(
                "PolicyServerWrapper: STARVLA_EVAL_SEED=%s — torch RNG is re-seeded per episode "
                "from (seed, episode_id) at every episode reset; rollouts are reproducible.",
                _EVAL_SEED,
            )
        else:
            logging.info(
                "PolicyServerWrapper: STARVLA_EVAL_SEED unset — action sampling noise is NOT "
                "reproducible across runs (set it to make rollouts deterministic)."
            )

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]
        
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def reset_memory(self, episode_id: Optional[int] = None) -> Dict[str, Any]:
        """Reset the framework's cross-frame memory (for streaming inference).

        Called by the websocket server on a ``reset`` control message at each episode
        boundary. Does a FULL reset (clears every bank): correct for the sequential
        single-env eval harnesses in this repo, and avoids leaking stale per-episode
        banks. A no-op for frameworks without a memory bank.

        (A future parallel-env streaming setup that keys many concurrent episodes into
        one server would instead want per-episode ``clear_episode_memory(episode_id)``.)

        Also re-seeds the sampling RNG when STARVLA_EVAL_SEED is set, so the episode's action
        noise depends only on its own id (see the module-level note).
        """
        seeded = None
        if _EVAL_SEED is not None and episode_id is not None:
            # Mix rather than add: a plain (seed + episode_id) makes base seeds 7 and 8 share every
            # stream but the first, so two "independent" replicates would silently reuse the same
            # noise and under-report variance.
            seeded = (_EVAL_SEED * 1_000_003 + int(episode_id)) % (2**31 - 1)
            torch.manual_seed(seeded)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeded)

        fw = self._framework
        did = False
        if hasattr(fw, "reset_memory"):
            fw.reset_memory()
            did = True
        return {"reset": bool(did), "episode_id": episode_id, "torch_seed": seeded}

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        geoplan_cfg = self._model_cfg.get("framework", {}).get("geoplan", {})
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "event_aux_dim": int(geoplan_cfg.get("event_aux_dim", 0)),
            "eval_memory_mode": getattr(self._framework, "eval_memory_mode", None),
            "eval_mem_length": getattr(getattr(self._framework, "obs2_mem_bank", None), "mem_length", None),
            "num_inference_timesteps": getattr(getattr(self._framework, "action_model", None), "num_inference_timesteps", None),
            "eval_seed": _EVAL_SEED,
            "geo_history_intervention": getattr(self._framework, "geo_history_intervention", "none"),
            "geo_history_seed": getattr(self._framework, "geo_history_seed", None),
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        # State-conditioned checkpoints (datasets.vla_data.include_state=true): clients send the
        # RAW env-space proprio state in MODEL order; normalize it here with the training-time
        # transform so it matches what the state_encoder saw at train time. Requests without a
        # `state` key (all legacy clients) are untouched.
        # The RAW copy is kept: relative-action checkpoints need it again at un-normalization
        # time (unapply adds the current state back to recover absolute env actions).
        raw_states: List[Optional[np.ndarray]] = []
        for ex in examples:
            if isinstance(ex, dict) and ex.get("state") is not None:
                raw = np.asarray(ex["state"])
                raw_states.append(raw)
                ex["state"] = proc.apply_state(raw)
            else:
                raw_states.append(None)

        # `n_samples` (AAC) is forwarded transparently via **kwargs to the framework.
        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B or N, T, D)

        # AAC (n_samples>1) expands a single example into N sampled chunks; every
        # sample shares that example's state.
        if len(raw_states) == 1 and normalized.shape[0] > 1:
            raw_states = raw_states * normalized.shape[0]

        unnorm = np.stack(
            [
                proc.unapply_actions(normalized[b], raw_state=raw_states[b])
                for b in range(normalized.shape[0])
            ],
            axis=0,
        )
        # Also return the normalized chunk: AAC's entropy/chunk-size selection is
        # computed in the normalized action space (see aac_chunk.select_chunk_size).
        # Legacy clients simply ignore this extra key.
        return {"actions": unnorm, "normalized_actions": normalized}
