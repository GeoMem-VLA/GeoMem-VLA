# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""QwenGeoGR00T Framework.

Qwen3-VL-4B backbone + GR00T flow-matching action head + GeoPlan 3D auxiliary task + a
perception/cognition/geo memory bank (ported from MemoryVLA).

Pipeline (training):
  1. Append K GeoPlan MARKER tokens to the instruction (OFT-style: reuse an existing single-token
     emoji as the marker -- these are NOT newly-registered learnable special tokens; they get the
     pretrained emoji input embedding and are differentiated mainly by position). They are processed
     by every Qwen layer; their final hidden states form the geo "3D abstraction".
  2. Run Qwen3-VL once. From the last hidden states extract:
       - geo tokens  : the K GeoPlan marker positions -> projected to the geo work dim (geo_cond).
       - cog token   : the last NON-PADDING position (robust to padding side) -> [B, 1, H].
       - per tokens  : the vision tokens (input_ids == image_token_id), length-normalized by a
                       Perceiver resampler to a fixed [B, M, H] (Qwen3-VL vision count is dynamic).
  3. Run the cog / per / geo memory banks across frames (no-op fallback when no same-episode history
     is available in the batch).
  4. Gated, non-destructive fusion of the (memory-fused) geo guidance into the perceptual tokens.
  5. Build the action-head cross-attention context (``action_context_type``):
       - "compressed"       : [cog | per]  (MemoryVLA-style information bottleneck; DEFAULT).
       - "full_plus_memory" : [full last_hidden | cog | per]  (keeps the full language/vision seq).
  6. GeoPlan auxiliary loss (only when a VGGT teacher checkpoint is configured): the frozen VGGT
     teacher produces current-frame 3D point-map targets; together with a relative-EEF-displacement
     target (from the action deltas) the self-contained flow-matching GeoFlowHead reconstructs them
     from geo_cond (``geo_supervise`` selects raw / memory / both). Scene-flow (future point maps)
     is NOT implemented yet (requires paired future frames) and is hard-disabled.

NOTE (coordinate frames): the VGGT point map lives in the VGGT/camera frame, while the EEF target is
a relative displacement in NORMALIZED ACTION space. They do NOT currently share a 3D coordinate
frame/scale (that would need camera extrinsics); the EEF segment is a separate relative-trajectory
target, not co-registered geometry.

Inference: the VGGT teacher and the GeoFlowHead are dropped. Memory at inference is controlled by
``eval_memory_mode``: "stateless" (reset every call -> NO cross-step memory; safe default) or
"streaming" (memory persists across steps; the eval harness MUST call reset_episode_memory() at each
episode boundary and pass episode_id/timestep per step). With "stateless" the memory banks are
effectively a no-op at inference -- a known train/inference asymmetry until the eval harness wires up
streaming.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.geoplan import GatedGeoFusion, GeoFlowHead, GeoInseqBridge, GeoPlanQueryResampler, VGGTGeoTeacher
from starVLA.model.modules.memory import (
    CogMemBank,
    EventGuidedMemBank,
    EventGuidedPerMemBank,
    PerMemBank,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

DEFAULT_IMAGE_TOKEN_INDEX = 151655  # Qwen3-VL image placeholder token id (fallback; read at runtime)


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGeoGR00T
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGeoGR00TDefaultConfig:
    """QwenGeoGR00T defaults. YAML ``framework:`` values override these; extra keys are preserved."""

    name: str = "QwenGeoGR00T"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2560,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2560,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # === GeoPlan 3D auxiliary + memory ===
    geoplan: dict = field(
        default_factory=lambda: {
            # --- master switches ---
            "use_geoplan": True,          # run the GeoPlan bridge + (if teacher available) aux loss
            "use_geo_aux": True,          # build/train the GeoFlowHead (only if a VGGT ckpt is set)
            "use_memory": True,           # run cog/per/geo memory banks
            "geo_bridge_type": "inseq_token",  # "inseq_token" (OFT-style) | "perceiver"
            # action-head cross-attention context:
            #   "compressed"       -> [cog | per]                  (MemoryVLA-style; DEFAULT)
            #   "full_plus_memory" -> [full last_hidden | cog | per]
            #   "full"             -> [full last_hidden]            (== base QwenGR00T, no memory)
            "action_context_type": "compressed",
            # which geo_cond the GeoFlow aux loss supervises: "raw" | "memory" | "both"
            "geo_supervise": "raw",
            # --- perceptual token-count normalization (handles Qwen3-VL's DYNAMIC vision token
            #     count: a fixed set of M learnable queries cross-attend the variable #vision tokens
            #     -> fixed [B, M, H], so the per memory bank / fusion / vl_embs are length-stable) ---
            "per_num_tokens": 64,         # M perceptual tokens after resampling
            "per_resampler_layers": 2,
            # --- bridge ---
            "geo_num_queries": 16,        # K GeoPlan tokens
            "geo_work_dim": 512,          # geo conditioning work dim
            "geo_emoji": "\U0001F9ED",   # 🧭 compass; one token, appended K times (OFT mechanism)
            # --- flow-matching head ---
            "geo_grid_size": 16,          # current point map = grid_size**2 patches
            "geo_head_hidden": 512,
            "geo_head_depth": 4,
            "geo_head_heads": 8,
            "lambda_cur": 0.5,
            "lambda_fut": 1.0,
            "lambda_eef": 3.0,
            "geo_loss_coeff": 0.1,        # weight of the GeoPlan aux loss vs the action loss
            # EEF target = cumulative future EEF xyz deltas (relative displacement in normalized
            # action space), multiplied by this scale. NOT divided by the horizon (that would shrink
            # the trajectory); NOT co-registered with the VGGT point-map frame.
            "eef_waypoint_scale": 1.0,
            # --- scene-flow (future point map); NOT implemented (needs paired future frames) ---
            "geo_use_future": False,
            "geo_flow_stride": 5,
            # --- VGGT teacher (target generator) ---
            "geo_vggt_ckpt": None,        # required to actually compute the aux loss
            "geo_vggt_repo_path": None,
            "geo_vggt_img_size": 518,
            # 0 runs the whole local batch at once. A small positive value bounds the frozen
            # teacher's forward-only activation peak without changing the train micro-batch.
            "geo_teacher_batch_size": 0,
            # --- memory bank ---
            "mem_length": 16,
            "retrieval_layers": 2,
            "use_timestep_pe": True,
            "fusion_type": "gate",        # "gate" | "add"
            "consolidate_type": "tome",   # "tome" | "fifo"
            "update_fused": False,
            "causal_retrieval": True,     # drop non-strictly-earlier history (enforces causality)
            # Hard-check that each batch is fixed-size, single-episode, strictly time-ordered groups.
            # Keep False until the dataloader provides real episode_id/timestep + group sampling
            # (otherwise the synthetic per-sample episode ids would fail the check).
            "validate_group_layout": False,
            "dataloader_type": "group",   # "group" clears the bank each batch (safe w/ random sampling)
            "group_size": 16,
            # eval memory: "stateless" resets the banks every predict_action call (safe default, no
            # cross-step memory); "streaming" keeps memory across steps and expects the eval harness
            # to call reset_memory() at each episode boundary.
            "eval_memory_mode": "stateless",
            # --- memory bank flavour ---
            #   "cog"           -> the validated MemoryVLA bank: every frame is written, one gate
            #                      controls how much history is blended in (DEFAULT, unchanged).
            #   "event_guided"  -> additionally learns WHICH frames are worth storing (scene-change
            #                      + event-novelty write gate) with importance-aware retrieval and
            #                      eviction. At init it behaves like "cog" and only becomes
            #                      selective as the gate learns.
            "mem_bank_type": "cog",
            # The settings below only apply when mem_bank_type == "event_guided".
            "event_write_threshold": 0.5,   # gate value above which a frame is committed
            # Anti-starvation fallback measured in ENVIRONMENT STEPS, not in frames seen by the bank.
            # Frames arrive ~8 steps apart at inference (exec_horizon) and ~17 apart in group-mode
            # training, so any small value is true on every frame and silently disables the write
            # gate. 0 = off; use 32-64 if you want a net.
            "event_min_write_stride": 0,
            "event_aux_dim": 0,             # width of an external eef/object signal (0 = unused)
            # Mix external aux novelty with learned retrieval-residual novelty. 1 preserves the
            # original aux-only behaviour; <1 also trains the fallback used when aux is absent.
            "event_aux_mix": 1.0,
            "event_importance_bias": 1.0,   # scale of the log-importance retrieval bias
            "event_read_modulation": True,  # surprise down-weights recalled memory when fusing
            "event_evict_policy": "importance",  # "importance" | "tome"
            "event_init_write_gate": 0.75,  # start by storing everything, like the original bank
            # Optional scale-stable EventScorer input. Both default off for checkpoint/config
            # compatibility; experiment launchers opt in explicitly.
            "event_scorer_normalize_inputs": False,
            "event_scorer_stop_gradient": False,
            # Optional per-example fields. ``event_aux`` must already be a current-frame delta
            # (e.g. ΔEEF / Δobject state), never a future action-label trajectory.
            "event_aux_key": "event_aux",
            "event_prior_key": "event_prior",
            # Set True only when event_prior is produced by the same causal detector at deployment.
            # Offline teacher labels should supervise BCE without changing the actual memory path.
            "event_prior_controls_write": True,
            # If >0 and event_prior is present, train the learned write heads to imitate the external
            # detector. Whether it also controls hard writes is configured independently above.
            "event_supervision_coeff": 0.0,
            # For smoothed targets, compute this from their mean mass (not the hard point count);
            # keyframe_labeler.summarize() reports the matching value.
            "event_supervision_pos_weight": 1.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGeoGR00T")
class Qwen_GeoGR00T(baseframework):
    """Qwen3-VL + GR00T action head + GeoPlan 3D auxiliary + perception/cognition/geo memory."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGeoGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Backbone hidden size (Qwen3-VL-4B == 2560). Read at runtime; never hardcode.
        self.hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.hidden_size
        # Image placeholder token id read from the backbone config (robust across Qwen versions).
        self.image_token_id = int(
            getattr(self.qwen_vl_interface.model.config, "image_token_id", DEFAULT_IMAGE_TOKEN_INDEX)
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # ---- GeoPlan / memory config (defaults merged with YAML overrides) ----
        default_geo = QwenGeoGR00TDefaultConfig().geoplan
        user_geo = dict(self.config.framework.get("geoplan", {})) if self.config.framework.get("geoplan", None) else {}
        gcfg = {**default_geo, **user_geo}
        self.geo_cfg = gcfg

        self.use_geoplan = bool(gcfg["use_geoplan"])
        self.use_memory = bool(gcfg["use_memory"])
        self.geo_bridge_type = str(gcfg["geo_bridge_type"])
        self.geo_num_queries = int(gcfg["geo_num_queries"])
        self.geo_work_dim = int(gcfg["geo_work_dim"])
        self.geo_grid_size = int(gcfg["geo_grid_size"])
        self.geo_use_future = bool(gcfg["geo_use_future"])
        self.geo_flow_stride = int(gcfg["geo_flow_stride"])
        self.geo_loss_coeff = float(gcfg["geo_loss_coeff"])
        self.per_num_tokens = int(gcfg["per_num_tokens"])
        self.eval_memory_mode = str(gcfg.get("eval_memory_mode", "stateless"))
        # Deployment override: allow the eval entrypoint to switch the memory mode WITHOUT
        # re-baking the checkpoint config (e.g. STARVLA_EVAL_MEMORY_MODE=streaming to actually
        # engage the cross-frame memory bank at inference; the eval harness MUST then reset the
        # memory at each episode boundary and pass episode_id/timestep per step).
        _env_mm = os.environ.get("STARVLA_EVAL_MEMORY_MODE")
        if _env_mm:
            self.eval_memory_mode = str(_env_mm).strip().lower()
            print(f"[QwenGeoGR00T] eval_memory_mode overridden by env -> {self.eval_memory_mode!r}")
        # Used only when a streaming inference client does not supply its own frame_index/timestep.
        # reset_memory() resets this counter at every episode boundary.
        self._infer_timestep = 0
        self.mem_dataloader_type = str(gcfg["dataloader_type"])
        self.mem_bank_type = str(gcfg.get("mem_bank_type", "cog")).strip().lower()
        self.geo_history_intervention = os.environ.get("STARVLA_GEO_HISTORY_INTERVENTION", "none").strip().lower()
        self.geo_history_seed = int(os.environ.get("STARVLA_GEO_HISTORY_SEED", "0"))
        self.geo_history_trace = os.environ.get("STARVLA_GEO_HISTORY_TRACE", "").strip()
        if self.geo_history_intervention not in {"none", "empty", "shuffle_time"}:
            raise ValueError(f"Unknown geo history intervention: {self.geo_history_intervention!r}")
        if self.geo_history_intervention != "none" or self.geo_history_trace:
            if not (self.use_geoplan and self.use_memory and self.eval_memory_mode == "streaming"
                    and self.mem_bank_type == "cog" and not bool(gcfg["update_fused"])):
                raise ValueError("Geo-history diagnostics require GeoPlan + streaming CogMemBank with raw writes")
            print(f"[QwenGeoGR00T] geo_history_intervention={self.geo_history_intervention!r}")
        self.event_aux_dim = int(gcfg.get("event_aux_dim", 0))
        self.event_aux_key = str(gcfg.get("event_aux_key", "event_aux"))
        self.event_prior_key = str(gcfg.get("event_prior_key", "event_prior"))
        self.event_prior_controls_write = bool(gcfg.get("event_prior_controls_write", True))
        self.event_supervision_coeff = float(gcfg.get("event_supervision_coeff", 0.0))
        if self.event_supervision_coeff < 0:
            raise ValueError("event_supervision_coeff must be >= 0")
        self.event_supervision_pos_weight = float(gcfg.get("event_supervision_pos_weight", 1.0))
        if self.event_supervision_pos_weight <= 0:
            raise ValueError("event_supervision_pos_weight must be > 0")
        self.action_context_type = str(gcfg.get("action_context_type", "compressed"))
        self.geo_supervise = str(gcfg.get("geo_supervise", "raw"))
        self.eef_waypoint_scale = float(gcfg.get("eef_waypoint_scale", 1.0))
        if self.action_context_type not in {"compressed", "full_plus_memory", "full"}:
            raise ValueError(f"Unknown action_context_type={self.action_context_type}")
        if self.geo_supervise not in {"raw", "memory", "both"}:
            raise ValueError(f"Unknown geo_supervise={self.geo_supervise}")
        if self.eval_memory_mode not in {"stateless", "streaming"}:
            raise ValueError(f"Unknown eval_memory_mode={self.eval_memory_mode!r}")
        # The DATASET packs actions according to datasets.vla_data.action_mode (abs|delta|rel); read
        # it only for backward-compatible defaulting of the EEF aux-target mode below.
        try:
            self.action_mode = str(self.config.datasets.vla_data.get("action_mode", "abs")).lower()
        except Exception:
            self.action_mode = "abs"
        if self.action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Unknown action_mode={self.action_mode!r} (expected abs|delta|rel)")

        # How the EEF waypoint aux target is reconstructed from the action chunk. This is DECOUPLED
        # from the dataset's action_mode: e.g. LIBERO actions are already per-step EEF deltas, so the
        # action head trains on them with action_mode='abs' (no re-differencing) while the EEF target
        # still treats them as deltas via 'delta_cumsum'. If not provided, derive from action_mode so
        # older configs keep their previous behaviour.
        #   delta_cumsum    -> cumsum(xyz) over the horizon (no state needed).
        #   rel             -> xyz as-is.
        #   abs_minus_state -> xyz - current_eef_state (requires include_state=True).
        eef_target_mode = gcfg.get("eef_target_mode", None)
        if eef_target_mode is None:
            eef_target_mode = {
                "delta": "delta_cumsum",
                "rel": "rel",
                "abs": "abs_minus_state",
            }[self.action_mode]
        self.eef_target_mode = str(eef_target_mode).lower()
        if self.eef_target_mode not in {"delta_cumsum", "rel", "abs_minus_state", "endpose_rel"}:
            raise ValueError(
                f"Unknown eef_target_mode={self.eef_target_mode!r} "
                "(expected delta_cumsum|rel|abs_minus_state|endpose_rel)"
            )

        # --- Adaptive multi-arm EEF aux (RoboTwin is bimanual joint-space) ---
        # When ``use_eef_endpose`` is True, the EEF waypoint target is built from a dedicated
        # ``example["eef"]`` sequence (real Cartesian endpose, e.g. RoboTwin observation.eef =
        # [left_xyz(3), right_xyz(3)]) instead of the joint-space action's first 3 dims. The head
        # then reconstructs ``eef_arms`` end-effector trajectories (each 3-D xyz, W steps), so the
        # EEF segment length auto-scales with the number of arms. eef_target_mode='endpose_rel'
        # makes each arm's target relative to its own first future frame (comparable to the
        # normalized point-map space). Single-arm datasets simply set eef_arms=1.
        self.use_eef_endpose = bool(gcfg.get("use_eef_endpose", False))
        self.eef_arms = int(gcfg.get("eef_arms", 1))
        if self.eef_arms < 1:
            raise ValueError(f"eef_arms must be >= 1, got {self.eef_arms}")

        # --- GeoPlan aux diagnostics / target conditioning ---
        # Per-sample normalization of the VGGT current-frame point-map target: center by the
        # per-cloud centroid and scale to unit RMS radius. The raw VGGT point map lives in an
        # un-normalized (≈metric) camera frame whose absolute scale varies per scene, which makes the
        # flow-matching geo_cur loss scale-dependent and prone to drift. Normalizing makes geo_cur
        # scale-invariant and stable/comparable across runs. Train-only target; no inverse needed.
        self.geo_normalize_pts = bool(gcfg.get("geo_normalize_pts", True))
        # How often (in forward() calls) to run the (cheap, no-grad) condition-sensitivity probe and
        # log geo/cond_delta_* = L(shuffled cond) - L(correct cond). >0 means the GeoFlow head
        # actually USES geo_cond for that modality; ≈0 is a red flag (the 3D aux is being ignored).
        # 0 disables the probe.
        self.geo_diag_interval = int(gcfg.get("geo_diag_interval", 0))
        self._geo_diag_step = 0
        # "memory"/"both" supervise the geo MEMORY output, which only exists when memory is on.
        if self.geo_supervise in {"memory", "both"} and not self.use_memory:
            raise ValueError(f"geo_supervise={self.geo_supervise!r} requires use_memory=True")
        # Scene-flow is not implemented yet: future inputs are not paired, so future loss is masked
        # out and nothing would be trained. Hard-fail rather than silently "enable" it.
        if self.geo_use_future:
            raise NotImplementedError(
                "geo_use_future requires paired future frames from the dataloader (not available); "
                "leave it False until scene-flow is wired up."
            )

        H = self.hidden_size
        # Number of EEF waypoints: one 3-D xyz trajectory per arm over the action horizon when the
        # real-endpose target is enabled (bimanual -> 2*W), otherwise the legacy single-arm W.
        W = self.action_horizon * (self.eef_arms if self.use_eef_endpose else 1)
        P = self.geo_grid_size * self.geo_grid_size          # current point-map tokens
        n_fut = 0                                            # scene-flow disabled (see above)
        self.geo_num_points = P
        self.geo_num_waypoints = W

        # ---- perceptual token-count normalization (Qwen3-VL vision token count is DYNAMIC) ----
        # M learnable queries cross-attend the (variable-length) vision tokens -> fixed [B, M, H].
        # work_dim == H so there is no channel down-projection (in_proj is Identity); this only fixes
        # the token LENGTH so the per memory bank's cross-frame stacking and vl_embs stay length-stable.
        self.per_resampler = GeoPlanQueryResampler(
            dim=H,
            num_queries=self.per_num_tokens,
            num_layers=int(gcfg["per_resampler_layers"]),
            work_dim=H,
        )

        # ---- GeoPlan bridge (backbone -> geo_cond) ----
        if self.use_geoplan:
            if self.geo_bridge_type == "inseq_token":
                self.geo_inseq = GeoInseqBridge(num_queries=self.geo_num_queries, dim=H, work_dim=self.geo_work_dim)
                self.geo_emoji = str(gcfg["geo_emoji"])
                tok = self.qwen_vl_interface.processor.tokenizer
                geo_ids = tok(self.geo_emoji, add_special_tokens=False)["input_ids"]
                if len(geo_ids) != 1:
                    # Must be exactly one token, otherwise the appended marker count != K and the
                    # gather would pick wrong positions. Fail loudly rather than silently mis-index.
                    raise ValueError(
                        f"[QwenGeoGR00T] geo_emoji {self.geo_emoji!r} must tokenize to exactly one "
                        f"token, but got ids={geo_ids}. Pick a single-token emoji."
                    )
                self.geo_token_id = geo_ids[0]
            elif self.geo_bridge_type == "perceiver":
                self.geo_resampler = GeoPlanQueryResampler(
                    dim=H, num_queries=self.geo_num_queries, num_layers=int(gcfg["retrieval_layers"]),
                    work_dim=self.geo_work_dim,
                )
            else:
                raise ValueError(f"Unknown geo_bridge_type={self.geo_bridge_type}")

            # ---- non-destructive gated fusion of geo guidance into perceptual tokens ----
            self.geo_fusion = GatedGeoFusion(per_dim=H, geo_dim=self.geo_work_dim)
            # ---- frozen VGGT teacher (held in a list -> NOT a registered submodule) ----
            # Treat None / empty string as "no teacher" (the aux 3D loss is then skipped).
            ckpt = gcfg["geo_vggt_ckpt"]
            self.geo_vggt_ckpt = ckpt if ckpt else None
            repo = gcfg["geo_vggt_repo_path"]
            self.geo_vggt_repo_path = repo if repo else None
            self.geo_vggt_img_size = int(gcfg["geo_vggt_img_size"])
            self.geo_teacher_batch_size = int(gcfg.get("geo_teacher_batch_size", 0))
            if self.geo_teacher_batch_size < 0:
                raise ValueError("geo_teacher_batch_size must be >= 0")
            self._geo_teacher_holder: List[Optional[VGGTGeoTeacher]] = [None]

            # ---- self-contained flow-matching geo head ----
            # Only build it when the aux loss can actually run (teacher ckpt present), otherwise it
            # would be dead weight in the optimizer / ZeRO / checkpoint (P2-1).
            self.use_geo_aux = bool(gcfg.get("use_geo_aux", True)) and (self.geo_vggt_ckpt is not None)
            if self.use_geo_aux:
                self.geo_head = GeoFlowHead(
                    seg_lens=(P, n_fut, W),
                    cond_dim=self.geo_work_dim,
                    in_channels=3,
                    hidden_size=int(gcfg["geo_head_hidden"]),
                    depth=int(gcfg["geo_head_depth"]),
                    num_heads=int(gcfg["geo_head_heads"]),
                    lambda_cur=float(gcfg["lambda_cur"]),
                    lambda_fut=float(gcfg["lambda_fut"]),
                    lambda_eef=float(gcfg["lambda_eef"]),
                )
            else:
                logger.warning(
                    "[QwenGeoGR00T] No VGGT teacher checkpoint -> GeoFlowHead NOT built and the 3D "
                    "aux loss is disabled. The geo bridge + gated fusion still train via the action "
                    "loss. Set framework.geoplan.geo_vggt_ckpt to enable the 3D supervision."
                )
        else:
            self.use_geo_aux = False

        # ---- perception / cognition / geo memory banks ----
        if self.use_memory:
            # Deployment override for the memory-bank capacity. TRAINING only ever accumulates up to
            # (group_size - 1) history frames per group, but at STREAMING inference the bank grows to
            # `mem_length` across the whole episode. If mem_length > group_size (e.g. 16 vs 8) the
            # retrieval attention is fed ~2x more history than it ever saw in training, and the ToMe
            # consolidation (which never fires when history < mem_length in training) starts firing
            # at inference -- both are train/inference distribution shifts that can make streaming
            # WORSE than stateless. Set STARVLA_EVAL_MEM_LENGTH=<group_size> to cap the inference
            # memory to the training window (e.g. 8) without retraining. Does not touch any weights.
            _mem_len = int(gcfg["mem_length"])
            _env_ml = os.environ.get("STARVLA_EVAL_MEM_LENGTH")
            if _env_ml:
                _mem_len = int(_env_ml)
                print(f"[QwenGeoGR00T] mem_length overridden by env -> {_mem_len} (train group_size={int(gcfg['group_size'])})")
            mem_kwargs = dict(
                dataloader_type=str(gcfg["dataloader_type"]),
                group_size=int(gcfg["group_size"]),
                mem_length=_mem_len,
                retrieval_layers=int(gcfg["retrieval_layers"]),
                use_timestep_pe=bool(gcfg["use_timestep_pe"]),
                fusion_type=str(gcfg["fusion_type"]),
                consolidate_type=str(gcfg["consolidate_type"]),
                update_fused=bool(gcfg["update_fused"]),
                causal_retrieval=bool(gcfg.get("causal_retrieval", True)),
                validate_group_layout=bool(gcfg.get("validate_group_layout", False)),
            )
            bank_type = self.mem_bank_type
            if bank_type not in {"cog", "event_guided"}:
                raise ValueError(
                    f"Unknown mem_bank_type={bank_type!r}; expected 'cog' or 'event_guided'"
                )
            if bank_type == "event_guided":
                mem_kwargs.update(
                    write_threshold=float(gcfg.get("event_write_threshold", 0.5)),
                    min_write_stride=int(gcfg.get("event_min_write_stride", 0)),
                    aux_dim=int(gcfg.get("event_aux_dim", 0)),
                    aux_mix=float(gcfg.get("event_aux_mix", 1.0)),
                    importance_bias=float(gcfg.get("event_importance_bias", 1.0)),
                    read_modulation=bool(gcfg.get("event_read_modulation", True)),
                    evict_policy=str(gcfg.get("event_evict_policy", "importance")),
                    init_write_gate=float(gcfg.get("event_init_write_gate", 0.75)),
                    scorer_normalize_inputs=bool(
                        gcfg.get("event_scorer_normalize_inputs", False)
                    ),
                    scorer_stop_gradient=bool(
                        gcfg.get("event_scorer_stop_gradient", False)
                    ),
                )
                cog_cls, per_cls = EventGuidedMemBank, EventGuidedPerMemBank
                print(
                    "[QwenGeoGR00T] mem_bank_type='event_guided' "
                    f"(mem_length={_mem_len}, "
                    f"scorer_normalize={mem_kwargs['scorer_normalize_inputs']}, "
                    f"scorer_stop_gradient={mem_kwargs['scorer_stop_gradient']})"
                )
            else:
                cog_cls, per_cls = CogMemBank, PerMemBank

            self.cog_mem_bank = cog_cls(token_size=H, **mem_kwargs)
            self.per_mem_bank = per_cls(token_size=H, **mem_kwargs)
            if self.use_geoplan:
                self.geo_mem_bank = cog_cls(token_size=self.geo_work_dim, **mem_kwargs)

        self._last_geo_logs: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers
    def _get_geo_teacher(self, device: torch.device) -> Optional[VGGTGeoTeacher]:
        """Lazily build / move the frozen VGGT teacher. Returns None if no checkpoint is configured."""
        if not self.use_geoplan or self.geo_vggt_ckpt is None:
            return None
        teacher = self._geo_teacher_holder[0]
        if teacher is None:
            teacher = VGGTGeoTeacher(
                ckpt_path=self.geo_vggt_ckpt,
                vggt_repo_path=self.geo_vggt_repo_path,
                img_size=self.geo_vggt_img_size,
                grid_size=self.geo_grid_size,
                device=device,
                dtype=torch.bfloat16,
            )
            self._geo_teacher_holder[0] = teacher
        else:
            teacher.to(device)
        return teacher

    @staticmethod
    def _episode_meta(examples: List[dict], device: torch.device):
        """Return (episode_ids, timesteps) as numpy arrays. Falls back to per-sample unique episode
        ids + zero timesteps when the dataloader does not provide them (memory becomes a no-op)."""
        B = len(examples)
        if "episode_id" in examples[0]:
            episode_ids = np.array([int(e["episode_id"]) for e in examples])
        else:
            episode_ids = np.arange(B)
        if "timestep" in examples[0]:
            timesteps = np.array([int(e["timestep"]) for e in examples])
        elif "frame_index" in examples[0]:
            timesteps = np.array([int(e["frame_index"]) for e in examples])
        else:
            timesteps = np.zeros(B, dtype=np.int64)
        return episode_ids, timesteps

    def _gather_special_tokens(self, last_hidden: torch.Tensor, input_ids: torch.Tensor, token_id: int, k: int) -> torch.Tensor:
        """Gather the ``k`` occurrences of ``token_id`` per sample -> [B, k, H] (temporal order).

        Requires EXACTLY ``k`` occurrences per sample (the appended marker block), so that the geo
        condition is well-defined and not contaminated by a stray marker elsewhere in the prompt.
        """
        B, L, H = last_hidden.shape
        device = input_ids.device
        mask = input_ids == token_id  # [B, L]
        counts = mask.sum(dim=1)
        if not bool((counts == k).all()):
            bad = (counts != k).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"[QwenGeoGR00T] samples {bad} do not have exactly {k} geo marker tokens "
                f"(counts={counts.tolist()}). The marker must be a single token that appears only in "
                f"the appended GeoPlan block."
            )
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=k, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values  # [B, k]
        expanded = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded)  # [B, k, H]

    @staticmethod
    def _extract_cog(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Cognition token = hidden at the last non-padding position -> [B, 1, H] (padding-side robust)."""
        if attention_mask is None:
            return last_hidden[:, -1:, :]
        H = last_hidden.shape[-1]
        last_idx = attention_mask.long().cumsum(dim=1).argmax(dim=1)  # [B] index of last valid token
        return last_hidden.gather(1, last_idx[:, None, None].expand(-1, 1, H))

    def _encode_and_extract(self, batch_images, instructions):
        """Run Qwen3-VL once and return (cog, per, geo_cond, full_hidden, attention_mask)."""
        if self.use_geoplan and self.geo_bridge_type == "inseq_token":
            suffix = " <geoplan>" + (self.geo_emoji * self.geo_num_queries) + "</geoplan>"
            instructions = [ins + suffix for ins in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = qwen_inputs.get("input_ids", None)
        attention_mask = qwen_inputs.get("attention_mask", None)

        # The resamplers read the (large) backbone hidden states; run them under the same bf16
        # autocast as the VLM so their fp32 master weights are cast for the matmul (no fp32 blow-up).
        vision_mask = (input_ids == self.image_token_id).to(dtype=torch.long)  # [B, L], 1 at vision tokens
        if int(vision_mask.sum().item()) == 0:
            raise RuntimeError("[QwenGeoGR00T] no image tokens found; cannot build perceptual tokens.")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, H]
            # per: token-count normalization. M queries attend ONLY the vision positions (mask=1),
            # so a DYNAMIC #vision tokens (per frame / per sample) becomes a fixed [B, M, H].
            per = self.per_resampler(last_hidden, attention_mask=vision_mask)  # [B, M, H]
            geo_cond = None
            if self.use_geoplan and self.geo_bridge_type == "perceiver":
                geo_cond = self.geo_resampler(last_hidden, attention_mask=attention_mask)  # [B, K, work_dim]

        # Cast the (small) extracted tensors to fp32 for the downstream side-modules. This keeps the
        # cog/per/geo conditioning in full precision (bf16's 8-bit mantissa is coarse and can inject
        # noise into low-magnitude channels), and avoids bf16-input vs fp32-weight mismatches. The
        # side-modules self-align dtype where needed, and the action head runs in its own fp32
        # autocast downstream, so keeping these fp32 is the safe/stable choice now that OOM is
        # resolved via smaller batch / geo_vggt_img_size.
        # cog: the cognition summary = hidden state at the LAST NON-PADDING position (robust to the
        # padding side via argmax-of-cumsum). Under Qwen's causal attention it has attended to the
        # full multimodal context (the Qwen3-VL analogue of MemoryVLA's last-LLM-token cog).
        cog = self._extract_cog(last_hidden, attention_mask).float()   # [B, 1, H]
        per = per.float()                                               # [B, M, H]
        if self.use_geoplan:
            if self.geo_bridge_type == "inseq_token":
                # .float() BEFORE geo_inseq.project(): geo_hidden is bf16 (sliced from last_hidden,
                # which is produced inside the bf16 autocast block), but project() runs OUTSIDE that
                # block. In fp32 deployment (--use_bf16 off) geo_inseq weights are fp32, so a bf16
                # input would raise a bf16-input vs fp32-weight mismatch. Cast the input first.
                geo_hidden = self._gather_special_tokens(
                    last_hidden, input_ids, self.geo_token_id, self.geo_num_queries
                ).float()
                geo_cond = self.geo_inseq.project(geo_hidden).float()  # [B, K, work_dim]
            elif geo_cond is not None:  # perceiver
                geo_cond = geo_cond.float()
        return cog, per, geo_cond, last_hidden, attention_mask

    @staticmethod
    def _param_dtype(module: torch.nn.Module, default: torch.dtype = torch.float32) -> torch.dtype:
        """Dtype of a module's parameters (falls back to ``default`` for parameter-less modules)."""
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return default

    def _event_memory_inputs(self, examples: List[dict], device: torch.device):
        """Collect optional external event deltas / keyframe scores from per-sample examples.

        These fields are deliberately explicit. Reusing ``action`` or the future ``eef`` trajectory
        here would leak labels from the future into a gate that must also run online at inference.
        """
        if not (self.use_memory and self.mem_bank_type == "event_guided"):
            return None, None

        def _field_presence(key: str):
            present = [key in example and example[key] is not None for example in examples]
            if any(present) and not all(present):
                raise ValueError(f"Event field {key!r} must be present for every sample or none")
            return all(present)

        event_aux = None
        if self.event_aux_key and _field_presence(self.event_aux_key):
            if self.event_aux_dim <= 0:
                raise ValueError(
                    f"Examples contain {self.event_aux_key!r}, but event_aux_dim={self.event_aux_dim}"
                )
            aux_values = []
            for example in examples:
                value = torch.as_tensor(example[self.event_aux_key], device=device, dtype=torch.float32).reshape(-1)
                if value.numel() != self.event_aux_dim:
                    raise ValueError(
                        f"{self.event_aux_key!r} must have {self.event_aux_dim} values per sample; "
                        f"got {value.numel()}"
                    )
                aux_values.append(value)
            event_aux = torch.stack(aux_values)

        event_prior = None
        if self.event_prior_key and _field_presence(self.event_prior_key):
            prior_values = []
            for example in examples:
                value = torch.as_tensor(example[self.event_prior_key], device=device, dtype=torch.float32).reshape(-1)
                if value.numel() != 1:
                    raise ValueError(f"{self.event_prior_key!r} must be a scalar per sample")
                prior_values.append(value[0])
            event_prior = torch.stack(prior_values)
        return event_aux, event_prior

    def _event_supervision_loss(self, event_prior: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """BCE imitation loss for learned write heads using optional external teacher targets.

        Uses logits + ``pos_weight`` rather than BCE on probabilities. For hard targets, balance from
        the positive rate; for smoothed targets, balance from their mean target mass. The labeller's
        ``summarize()`` reports the matching value directly.
        """
        if event_prior is None or self.event_supervision_coeff <= 0:
            return None
        banks = [self.cog_mem_bank, self.per_mem_bank]
        if self.use_geoplan and hasattr(self, "geo_mem_bank"):
            banks.append(self.geo_mem_bank)
        losses = []
        for bank in banks:
            scores = getattr(bank, "last_scores", None)
            if not scores or "g_write_logit" not in scores:
                continue
            logit = scores["g_write_logit"]
            target = event_prior.to(device=logit.device, dtype=logit.dtype)
            pos_weight = torch.as_tensor(
                self.event_supervision_pos_weight, device=logit.device, dtype=logit.dtype
            )
            losses.append(
                F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
            )
        return torch.stack(losses).mean() if losses else None

    def _event_memory_logs(self) -> Dict[str, torch.Tensor]:
        """Detached gate/write diagnostics used to catch saturation or memory starvation."""
        if not (self.use_memory and self.mem_bank_type == "event_guided"):
            return {}
        named_banks = [("cog", self.cog_mem_bank), ("per", self.per_mem_bank)]
        if self.use_geoplan and hasattr(self, "geo_mem_bank"):
            named_banks.append(("geo", self.geo_mem_bank))
        logs = {}
        for name, bank in named_banks:
            scores = getattr(bank, "last_scores", None)
            if not scores:
                continue
            for key in ("s_scene", "s_event", "g_write", "g_write_effective", "g_read", "wrote"):
                if key in scores:
                    logs[f"memory/{name}_{key}_mean"] = scores[key].detach().float().mean()
        return logs

    def _apply_memory_and_fusion(
        self,
        cog,
        per,
        geo_cond,
        episode_ids,
        timesteps,
        reset_before_batch,
        update_state,
        event_aux=None,
        event_prior=None,
    ):
        """Run the memory banks and the gated geo fusion; return the fused (cog, per).

        State lifecycle is controlled EXPLICITLY (not via train()/eval()): ``reset_before_batch``
        clears the banks first (group-mode training), ``update_state`` writes the current frames.

        Every sub-module input is aligned to that module's parameter dtype first: the fp32
        conditioning (cog/per/geo_cond from `_encode_and_extract`) must match whatever dtype the
        side-modules ended up in (bf16 under DeepSpeed / `--use_bf16`, fp32 otherwise), so no
        bf16-input vs fp32-weight (or vice-versa) mismatch can ever raise.
        """
        mem_kw = dict(reset_before_batch=reset_before_batch, update_state=update_state)
        if self.mem_bank_type == "event_guided":
            write_prior = event_prior if self.event_prior_controls_write else None
            mem_kw.update(aux=event_aux, event_prior=write_prior)
        if self.use_memory:
            cog = cog.to(self._param_dtype(self.cog_mem_bank))
            per = per.to(self._param_dtype(self.per_mem_bank))
            cog = self.cog_mem_bank.process_batch(cog, episode_ids, timesteps, **mem_kw)
            per = self.per_mem_bank.process_batch(per, episode_ids, timesteps, **mem_kw)
        # geo_guidance is the MEMORY-fused geo output; it is None when memory is off (so callers can
        # tell raw from memory-fused). Fusion still runs on raw geo_cond when there is no memory.
        geo_guidance = None
        if self.use_geoplan and geo_cond is not None:
            if self.use_memory:
                geo_cond = geo_cond.to(self._param_dtype(self.geo_mem_bank))
                geo_kw = dict(mem_kw)
                intervention = getattr(self, "geo_history_intervention", "none")
                if intervention != "none":
                    geo_kw.update(history_intervention=intervention,
                                  intervention_seed=self.geo_history_seed)
                geo_guidance = self.geo_mem_bank.process_batch(geo_cond, episode_ids, timesteps, **geo_kw)
                trace_path = getattr(self, "geo_history_trace", "")
                if trace_path and not self.training:
                    with open(trace_path, "a", encoding="utf-8") as trace:
                        for i, (eid, ts) in enumerate(zip(episode_ids, timesteps)):
                            record = {"episode_id": int(eid), "timestep": int(ts),
                                      "condition": intervention, "update_state": bool(update_state)}
                            for name in ("cog", "per", "geo"):
                                bank = getattr(self, f"{name}_mem_bank")
                                record[name] = {**bank.last_history_diagnostics[i],
                                                "stored_after": len(bank.bank.get(eid, []))}
                            trace.write(json.dumps(record) + "\n")
            fusion_geo = geo_guidance if geo_guidance is not None else geo_cond
            # GatedGeoFusion runs raw Linear layers and does NOT self-align, so align both inputs.
            _fdt = self._param_dtype(self.geo_fusion)
            per = per.to(_fdt)
            fusion_geo = fusion_geo.to(_fdt)
            per = self.geo_fusion(per, fusion_geo)
        return cog, per, geo_guidance

    def _build_eef_displacement(
        self,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        eef: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Future-EEF displacement target -> [B, W, 3], built according to ``eef_target_mode``.

        Two sources are supported:

        (A) Real endpose (``use_eef_endpose=True``, RECOMMENDED for joint-space / bimanual robots):
            ``eef`` is a dedicated Cartesian endpose sequence [B, T, arms*3] (e.g. RoboTwin
            observation.eef = [left_xyz, right_xyz], T = action_horizon). Each arm's trajectory is
            made relative to its own first future frame (``endpose_rel`` / ``rel``) and the arms are
            concatenated along the waypoint dim -> [B, arms*T, 3]. This is the geometrically-correct
            target for robots whose actions are joint angles (the joint-space action[:3] is NOT xyz).

        (B) Legacy action-xyz (single-arm EEF-space datasets like LIBERO): the first 3 action
            channels are treated as the end-effector xyz and reconstructed per ``eef_target_mode``
            (delta_cumsum | rel | abs_minus_state).

        Result is a RELATIVE displacement (NOT co-registered with the VGGT point-map camera frame),
        scaled by ``eef_waypoint_scale``.
        """
        # ---- (A) real multi-arm endpose ----
        if self.use_eef_endpose:
            if eef is None:
                raise ValueError(
                    "use_eef_endpose=True but no 'eef' sequence was provided. Ensure the DataConfig "
                    "declares an 'eef' modality and the dataset has an 'observation.eef' column."
                )
            T = self.action_horizon
            B = eef.shape[0]
            expected = self.eef_arms * 3
            if eef.shape[-1] != expected:
                raise ValueError(
                    f"eef last dim {eef.shape[-1]} != eef_arms*3 ({expected}); "
                    f"set framework.geoplan.eef_arms to match observation.eef ({eef.shape[-1] // 3} arms)."
                )
            e = eef[:, -T:, :].reshape(B, T, self.eef_arms, 3)          # [B, T, arms, 3]
            if self.eef_target_mode in ("endpose_rel", "rel"):
                e = e - e[:, :1]                                        # per-arm relative to first future frame
            elif self.eef_target_mode == "delta_cumsum":
                e = torch.cumsum(e, dim=1)
            # arm-major flatten: [B, T, arms, 3] -> [B, arms, T, 3] -> [B, arms*T, 3]
            e = e.permute(0, 2, 1, 3).reshape(B, self.eef_arms * T, 3)
            return e * self.eef_waypoint_scale

        # ---- (B) legacy single-arm action-xyz ----
        W = self.geo_num_waypoints
        xyz = actions[:, -W:, :3]
        if self.eef_target_mode == "delta_cumsum":
            eef_out = torch.cumsum(xyz, dim=1)
        elif self.eef_target_mode == "rel":
            eef_out = xyz
        elif self.eef_target_mode == "abs_minus_state":
            if state is None:
                raise ValueError(
                    "eef_target_mode='abs_minus_state' requires the current EEF state; set "
                    "datasets.vla_data.include_state=True (or use eef_target_mode='delta_cumsum')."
                )
            eef_out = xyz - state[:, :1, :3]
        else:
            raise ValueError(f"Unsupported eef_target_mode={self.eef_target_mode}")
        return eef_out * self.eef_waypoint_scale

    @torch.no_grad()
    def _prepare_geo_targets(self, batch_images, actions, device, state=None, eef=None):
        """Run the (expensive) VGGT teacher ONCE per batch and assemble the geo target sequence.

        Targets = [current-frame VGGT point map (camera frame) | relative EEF displacement (action
        space)]. The two segments do NOT share a coordinate frame (see module docstring); the geo
        head reconstructs each as its own modality (distinguished by its type embedding). Returns
        None when no teacher is available.
        """
        teacher = self._get_geo_teacher(device)
        if teacher is None or not self.use_geo_aux:
            return None
        if hasattr(teacher, "eval"):
            teacher.eval()

        # Current-frame RGB (primary view), resized to a common square for batching -> [B,3,S,S] in
        # [0,1]. mean/std=0/1 since the teacher expects [0,1] (it interpolates to its own img_size).
        S = self.geo_vggt_img_size
        cur = torch.stack(
            [TF.resize(self._pil_to_chw01(imgs[0]), [S, S], antialias=True) for imgs in batch_images], dim=0
        ).to(device)
        # The teacher is frozen, so splitting only this forward is numerically equivalent to a
        # full-batch call while avoiding a large DPT activation spike. In particular, do not
        # reduce the actual train batch or MEMFIX group to solve teacher OOMs.
        teacher_bs = self.geo_teacher_batch_size
        if teacher_bs > 0 and teacher_bs < cur.shape[0]:
            parts: Dict[str, List[torch.Tensor]] = {}
            for start in range(0, cur.shape[0], teacher_bs):
                result = teacher.compute_targets(
                    cur[start : start + teacher_bs],
                    cur[start : start + teacher_bs],
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0),
                )
                for key, value in result.items():
                    parts.setdefault(key, []).append(value)
            targets = {key: torch.cat(values, dim=0) for key, values in parts.items()}
        else:
            targets = teacher.compute_targets(cur, cur, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        pts_cur = targets["pts_cur"].to(device).float()   # [B, P, 3] (VGGT/camera frame)
        w_cur = targets["w_cur"].to(device).float()       # [B, P]

        # Per-sample isotropic normalization of the point-map target (see __init__): zero-center by
        # the per-cloud centroid, then divide by the per-cloud RMS radius so every scene has unit
        # scale. This removes the scene-dependent absolute scale that otherwise makes geo_cur drift
        # and incomparable across runs. Isotropic (single scalar per sample) preserves 3D shape.
        if self.geo_normalize_pts:
            mu = pts_cur.mean(dim=1, keepdim=True)                                  # [B, 1, 3]
            centered = pts_cur - mu
            rms = centered.pow(2).sum(dim=-1).mean(dim=1).clamp_min(1e-12).sqrt()   # [B]
            pts_cur = centered / rms[:, None, None]

        eef_disp = self._build_eef_displacement(actions, state, eef=eef).to(device).float()  # [B, W, 3]
        return {"target": torch.cat([pts_cur, eef_disp], dim=1), "w_cur": w_cur}

    def _geo_loss_from_targets(self, geo_cond, geo_targets):
        """GeoFlow loss for one condition against PRE-COMPUTED targets (reuses VGGT output)."""
        return self.geo_head.loss(geo_targets["target"], geo_cond.float(), w_cur=geo_targets["w_cur"])

    @staticmethod
    def _pil_to_chw01(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _build_action_context(self, cog, per, full_hidden, attention_mask):
        """Assemble the action-head cross-attention context + its key-padding/attention mask."""
        B = cog.shape[0]
        if self.action_context_type == "compressed":
            vl = torch.cat([cog, per], dim=1)
            mask = torch.ones(vl.shape[:2], dtype=torch.bool, device=vl.device)
            return vl, mask
        # Keep full_hidden in its native dtype (bf16 under DeepSpeed) so that
        # torch.cat([fh, cog, per]) does not hit a Float vs BFloat16 mismatch.
        # The action_model is wrapped by its own fp32 autocast context at the call site.
        fh = full_hidden
        if attention_mask is not None:
            am = attention_mask.to(dtype=torch.bool, device=fh.device)
        else:
            am = torch.ones(fh.shape[:2], dtype=torch.bool, device=fh.device)
        if self.action_context_type == "full":
            return fh, am
        # full_plus_memory: [full last_hidden | cog | per]
        total_len = fh.shape[1] + cog.shape[1] + per.shape[1]
        max_seq = int(self.config.framework.action_model.get("max_seq_len", 1024))
        if total_len > max_seq:
            raise ValueError(
                f"action_context_type='full_plus_memory' produces {total_len} context tokens "
                f"(> action_model.max_seq_len={max_seq}). Increase max_seq_len or use 'compressed'."
            )
        extra = torch.ones(B, cog.shape[1] + per.shape[1], dtype=torch.bool, device=fh.device)
        vl = torch.cat([fh, cog.to(fh.dtype), per.to(fh.dtype)], dim=1)
        mask = torch.cat([am, extra], dim=1)
        return vl, mask

    def reset_memory(self) -> None:
        """Clear ALL memory banks (every episode). Call at episode start in streaming evaluation."""
        self._infer_timestep = 0
        if self.use_memory:
            self.cog_mem_bank.reset()
            self.per_mem_bank.reset()
            if self.use_geoplan and hasattr(self, "geo_mem_bank"):
                self.geo_mem_bank.reset()

    # Eval-harness facing aliases / per-episode cleanup (for parallel-env streaming inference).
    reset_episode_memory = reset_memory
    _reset_memory = reset_memory

    def clear_episode_memory(self, episode_id) -> None:
        """Drop the memory of a single finished episode (use with parallel-env streaming eval)."""
        if self.use_memory:
            self.cog_mem_bank.clear_episode(episode_id)
            self.per_mem_bank.clear_episode(episode_id)
            if self.use_geoplan and hasattr(self, "geo_mem_bank"):
                self.geo_mem_bank.clear_episode(episode_id)

    def _maybe_debug_batch_layout(self, episode_ids, timesteps) -> None:
        """Set env QWENGEO_DEBUG_BATCH=1 to print, for the first few steps, this rank's per-group
        (episode_id, frame_index) layout so multi-GPU group integrity can be VERIFIED empirically."""
        if os.environ.get("QWENGEO_DEBUG_BATCH", "0") != "1":
            return
        n = getattr(self, "_dbg_batch_count", 0)
        if n >= 5:
            return
        self._dbg_batch_count = n + 1
        rank = int(os.environ.get("RANK", "0"))
        G = int(self.geo_cfg.get("group_size", 1))
        eids = np.asarray(episode_ids).tolist()
        frames = np.asarray(timesteps).tolist()
        lines = []
        ok = True
        for s in range(0, len(eids), max(1, G)):
            ge, gf = eids[s : s + G], frames[s : s + G]
            single = len(set(ge)) == 1
            ordered = len(gf) <= 1 or all(b > a for a, b in zip(gf, gf[1:]))
            ok = ok and single and ordered
            lines.append(
                f"    grp[{s}:{s+len(ge)}] eids={ge} frames={gf} "
                f"single={single} strictly_ordered={ordered}"
            )
        logger.warning(f"[QWENGEO_DEBUG_BATCH] rank={rank} step={n} group_size={G} layout_ok={ok}\n" + "\n".join(lines))

    # ------------------------------------------------------------------ forward
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        eef = [example["eef"] for example in examples] if "eef" in examples[0] else None

        # Generate frozen VGGT targets before building the Qwen/action autograd graph. Previously
        # this ran near the end of the train forward while all trainable activations were live,
        # which pushed an 80-GiB GPU to ~78.2 GiB and intermittently OOMed in DPT's final conv.
        # Only the small detached targets survive this block; EEF supervision remains fully on.
        geo_targets = None
        if self.use_geo_aux and self.training:
            model_device = next(self.parameters()).device
            actions_geo = torch.as_tensor(np.asarray(actions), device=model_device, dtype=torch.float32)
            state_geo = (
                torch.as_tensor(np.asarray(state), device=model_device, dtype=torch.float32)
                if state is not None else None
            )
            eef_geo = (
                torch.as_tensor(np.asarray(eef), device=model_device, dtype=torch.float32)
                if eef is not None else None
            )
            geo_targets = self._prepare_geo_targets(
                batch_images,
                actions_geo,
                model_device,
                state=state_geo,
                eef=eef_geo,
            )
            del actions_geo, state_geo, eef_geo

        cog, per, geo_cond, full_hidden, attn_mask = self._encode_and_extract(batch_images, instructions)
        episode_ids, timesteps = self._episode_meta(examples, cog.device)
        event_aux, event_prior = self._event_memory_inputs(examples, cog.device)
        self._maybe_debug_batch_layout(episode_ids, timesteps)
        # Group-mode training: clear the bank at the start of every batch (each batch is fresh groups).
        cog, per, geo_guidance = self._apply_memory_and_fusion(
            cog, per, geo_cond, episode_ids, timesteps,
            reset_before_batch=(self.mem_dataloader_type == "group"),
            update_state=True,
            event_aux=event_aux,
            event_prior=event_prior,
        )

        vl_embs, enc_mask0 = self._build_action_context(cog, per, full_hidden, attn_mask)

        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(np.array(actions), device=vl_embs.device, dtype=vl_embs.dtype)
            actions_target = actions_t[:, -self.action_horizon :, :]

            repeated = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_rep = actions_target.repeat(repeated, 1, 1)
            vl_embs_rep = vl_embs.repeat(repeated, 1, 1)
            enc_mask = enc_mask0.repeat(repeated, 1)

            state_rep = None
            state_t = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=vl_embs.device, dtype=vl_embs.dtype)
                state_rep = state_t.repeat(repeated, 1, 1)

            eef_t = None
            if eef is not None:
                eef_t = torch.tensor(np.array(eef), device=vl_embs.device, dtype=vl_embs.dtype)

            action_loss = self.action_model(
                vl_embs_rep, actions_rep, state_rep, encoder_attention_mask=enc_mask
            )

        total = action_loss
        out: Dict[str, torch.Tensor] = {}
        out.update(self._event_memory_logs())
        event_loss = self._event_supervision_loss(event_prior)
        if event_loss is not None:
            total = total + self.event_supervision_coeff * event_loss
            out["memory/event_loss"] = event_loss.detach()
        if self.use_geo_aux and self.training and geo_cond is not None:
            # Reuse the targets generated before the trainable forward graph was built.
            if geo_targets is not None:
                # geo_supervise: which geo_cond the 3D aux loss supervises.
                #   raw    -> per-frame bridge output (current-frame 3D abstraction).
                #   memory -> cross-frame memory-fused guidance.
                #   both   -> MEAN of the two (so geo_loss_coeff has a comparable scale across modes).
                losses = []
                if self.geo_supervise in ("raw", "both"):
                    loss_raw, logs_raw = self._geo_loss_from_targets(geo_cond, geo_targets)
                    losses.append(loss_raw)
                    for k, v in logs_raw.items():
                        out[f"geo/{k}"] = v.detach() if torch.is_tensor(v) else torch.as_tensor(v)
                if self.geo_supervise in ("memory", "both") and geo_guidance is not None:
                    loss_mem, logs_mem = self._geo_loss_from_targets(geo_guidance, geo_targets)
                    losses.append(loss_mem)
                    for k, v in logs_mem.items():
                        out[f"geo/mem_{k}"] = v.detach() if torch.is_tensor(v) else torch.as_tensor(v)
                if losses:
                    geo_loss = torch.stack(losses).mean()
                    total = total + self.geo_loss_coeff * geo_loss
                    out["geo/geo_loss"] = geo_loss.detach()
                    self._last_geo_logs = {k: v for k, v in out.items() if k.startswith("geo/")}

                # Periodic condition-sensitivity probe (cheap, no-grad). Logs, per modality,
                # geo/cond_delta_* = L(shuffled cond) - L(correct cond): >0 means the GeoFlow head
                # USES geo_cond for that modality, ~0 means the 3D aux is effectively ignored.
                if self.geo_diag_interval > 0:
                    self._geo_diag_step += 1
                    if self._geo_diag_step % self.geo_diag_interval == 0:
                        diag = self.geo_head.condition_sensitivity(
                            geo_targets["target"], geo_cond.float()
                        )
                        for k, v in diag.items():
                            out[f"geo/{k}"] = torch.as_tensor(float(v))

        out["action_loss"] = total
        out["action_only_loss"] = action_loss.detach()
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> Dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        # Memory at inference:
        #   "stateless" -> reset every call (no cross-step memory; safe default, always runnable).
        #   "streaming" -> keep memory across steps; the eval harness must call reset_memory() at
        #                  each episode boundary (and pass episode_id/timestep per step).
        if self.eval_memory_mode == "stateless":
            self.reset_memory()

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        cog, per, geo_cond, full_hidden, attn_mask = self._encode_and_extract(batch_images, instructions)
        episode_ids, timesteps = self._episode_meta(examples, cog.device)
        event_aux, event_prior = self._event_memory_inputs(examples, cog.device)
        # Most real-robot websocket clients send only image/state/lang. Give those calls a causal,
        # monotonically increasing time index so streaming memory can actually retrieve earlier
        # frames. Explicit client metadata always wins (e.g. offline datasets and parallel eval).
        has_explicit_time = "timestep" in examples[0] or "frame_index" in examples[0]
        if self.eval_memory_mode == "streaming" and not has_explicit_time:
            timesteps = np.full(len(examples), self._infer_timestep, dtype=np.int64)
            self._infer_timestep += 1
        # Inference: stateless mode already reset above; streaming mode accumulates across steps and
        # expects reset_episode_memory() at episode boundaries. Never reset_before_batch mid-episode.
        cog, per, _geo_guidance = self._apply_memory_and_fusion(
            cog, per, geo_cond, episode_ids, timesteps,
            reset_before_batch=False,
            update_state=(self.eval_memory_mode == "streaming"),
            event_aux=event_aux,
            event_prior=event_prior,
        )
        vl_embs, enc_mask = self._build_action_context(cog, per, full_hidden, attn_mask)

        # AAC multi-sample: draw `n_samples` candidate action chunks in parallel.
        # The backbone + memory above ran ONCE (batch == len(examples)); only the
        # small action-head DiT sees the N-fold replication. Default 1 keeps the
        # legacy single-chunk behaviour for every other client (RoboTwin etc.).
        n_samples = int(kwargs.get("n_samples", 1))
        if n_samples > 1 and vl_embs.shape[0] != 1:
            raise ValueError(
                f"n_samples={n_samples} (AAC) only supports a single input example per request; "
                f"got {vl_embs.shape[0]} examples."
            )

        with torch.autocast("cuda", dtype=torch.float32):
            state_t = (
                torch.from_numpy(np.array(state)).to(vl_embs.device, dtype=vl_embs.dtype)
                if state is not None
                else None
            )
            pred_actions = self.action_model.predict_action(
                vl_embs, state_t, encoder_attention_mask=enc_mask, n_samples=n_samples
            )

        # (n_samples, T, D) when AAC is on, else (batch, T, D) as before.
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    @torch.inference_mode()
    def predict_geo(self, examples: List[dict], num_steps: int = 10, **kwargs) -> Dict[str, np.ndarray]:
        """Inference-time 3D-scene reconstruction hook (for ablation / visualization).

        Runs the SAME perception path as ``predict_action`` (encode -> geo_cond -> memory/fusion),
        then samples the GeoFlow head to reconstruct the 3D target sequence conditioned on geo_cond.
        Returns numpy arrays:
            {"geo_points": [B, P, 3], "geo_eef": [B, W, 3] (if present)}
        where ``geo_points`` is the reconstructed current-frame point map (P = geo_grid_size**2) and
        ``geo_eef`` the reconstructed future EEF waypoints. See ``GeoFlowHead.sample`` for the
        coordinate-frame caveat (normalized vs raw VGGT frame).

        Does NOT need the VGGT teacher (that is train-only). Requires ``use_geo_aux=True`` (so the
        ``geo_head`` exists in the checkpoint).
        """
        if not (self.use_geoplan and getattr(self, "use_geo_aux", False) and hasattr(self, "geo_head")):
            raise RuntimeError(
                "predict_geo requires a trained GeoFlow head (framework.geoplan.use_geo_aux=True with "
                "a geo_vggt_ckpt set at train time). This checkpoint has no geo_head."
            )
        if not isinstance(examples, list):
            examples = [examples]
        if self.eval_memory_mode == "stateless":
            self.reset_memory()

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        cog, per, geo_cond, full_hidden, attn_mask = self._encode_and_extract(batch_images, instructions)
        episode_ids, timesteps = self._episode_meta(examples, cog.device)
        event_aux, event_prior = self._event_memory_inputs(examples, cog.device)
        # Reproduce predict_action's memory/fusion so geo_guidance matches deployment behaviour.
        # IMPORTANT: by default do NOT write this frame into the memory bank — predict_geo is a
        # visualization/ablation hook, and if it were called alongside predict_action during a
        # streaming rollout it would double-write the current frame (inflating memory length and
        # corrupting retrieval). Pass update_memory=True only for a standalone geo-only stream.
        _update_mem = bool(kwargs.get("update_memory", False)) and (self.eval_memory_mode == "streaming")
        _cog, _per, geo_guidance = self._apply_memory_and_fusion(
            cog, per, geo_cond, episode_ids, timesteps,
            reset_before_batch=False,
            update_state=_update_mem,
            event_aux=event_aux,
            event_prior=event_prior,
        )
        # Condition the geo head on the same signal the training loss supervised (geo_supervise).
        if self.geo_supervise in ("memory", "both") and geo_guidance is not None:
            cond = geo_guidance
        else:
            cond = geo_cond

        with torch.autocast("cuda", dtype=torch.bfloat16):
            seq = self.geo_head.sample(cond, num_steps=num_steps)
        segs = self.geo_head._slice_named(seq)

        out: Dict[str, np.ndarray] = {"geo_points": segs["pts_cur"].float().cpu().numpy()}
        if segs["eef"] is not None:
            out["geo_eef"] = segs["eef"].float().cpu().numpy()
        if segs["pts_fut"] is not None:
            out["geo_points_future"] = segs["pts_fut"].float().cpu().numpy()
        return out


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="examples/LIBERO/train_files/starvla_cotrain_libero_geoplan.yaml")
    args, _ = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    model: Qwen_GeoGR00T = Qwen_GeoGR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(8, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = dict(sample)
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = model(batch)
    print({k: float(v) for k, v in out.items()})
    pred = model.predict_action(examples=[sample])
    print("pred", pred["normalized_actions"].shape)
    print("Finished")
