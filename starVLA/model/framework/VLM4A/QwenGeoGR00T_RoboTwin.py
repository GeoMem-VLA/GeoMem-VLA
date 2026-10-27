# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""QwenGeoGR00T_RoboTwin — RoboTwin-dedicated pipeline (one pipeline == one code).

A STANDALONE fork of ``QwenGeoGR00T.py`` frozen for RoboTwin (dual-arm, absolute joint qpos).
The base pipeline is identical to the parent (Qwen3-VL backbone -> GeoPlan inseq-token bridge ->
obs2/obs1/geo memory banks -> gated geo fusion -> GR00T flow-matching action head; VGGT-supervised
GeoFlow aux head, train-only) — see QwenGeoGR00T.py for the full pipeline documentation.
The parent file is NOT imported and NOT modified; experiments on either framework cannot break
the other. Only the deltas below differ, and they are ALL config-defaults (no env vars).

Deltas vs QwenGeoGR00T:

1. EEF-IN-POINTCLOUD geo target ("Strategy A", fixes the RoboTwin geo-aux degeneration where
   cond_delta sat at the noise floor because the tiny ``endpose_rel`` target was fit
   unconditionally): the future EEF trajectory GT now lives IN THE VGGT POINT-CLOUD FRAME.
     - Per sample, the metric->VGGT similarity scale ``s`` is estimated by projecting the
       CURRENT left/right EEF (world -> camera via the per-frame ``extrinsic_cv``, camera ->
       pixel via ``intrinsic_cv``) and reading the VGGT full-resolution point map at those
       pixels: ``s = median(z_vggt / z_cam)`` over valid gripper anchors. (VGGT single-frame
       point maps are in the OpenCV camera frame up to scale, so R,t are identity; the saved
       dataset has no GT depth, hence anchor-based scale instead of depth-based umeyama.)
     - The FULL free-space future trajectory (world -> cam, x ``s``) and the current point map
       are then normalized TOGETHER (single centroid/RMS), so the trajectory is literally
       embedded in the same normalized 3D scene the point-map segment reconstructs.
     - The target is now a joint function of the action labels AND the scene geometry/camera
       pose -> no longer a deterministic function of the actions -> non-redundant with
       action_loss; both segments demand the same 3D understanding (competition -> synergy).
     - Samples whose anchors are invalid (out of view / bad depth) get ``w_eef = 0``.
   Requires the ``observation.campose`` column (add_campose_to_lerobot.py) exposed through the
   ``eef.campose`` modality key (robot_type ``robotwin50_v2``).

2. ANTI-JITTER (absolute-joint chunks tolerate no high-frequency residual):
     - ``SmoothFlowmatchingActionHead``: adds a second-difference (acceleration-matching) loss.
       From the predicted velocity the implied clean chunk is reconstructed as
       ``x_hat = x_t + (1 - t) * v_hat`` and ``|| dd(x_hat) - dd(x_gt) ||^2`` is penalised
       (``dd`` = discrete second difference along the horizon). This targets exactly the
       high-frequency noise that shows up as arm jitter while matching the DEMONSTRATED
       acceleration (not zero), so it does not over-smooth. Weight:
       ``framework.action_model.action_smooth_coeff``.
     - Default ``num_inference_timesteps: 16`` (vs 4): finer Euler integration, baked into the
       checkpoint config so eval needs no STARVLA_NUM_INFERENCE_TIMESTEPS.
     - Pair with the ``robotwin50_v2`` DataConfig (q99 joint normalization): quantile stats stop
       min/max outliers squeezing the normalized range (which amplifies residuals on unapply).

3. NO ENV-VAR OVERRIDES: the STARVLA_EVAL_MEMORY_MODE / STARVLA_EVAL_MEM_LENGTH escape hatches
   are removed; RoboTwin-correct defaults (action/state 14-D, horizon 50, eef_arms=2,
   mem_length == group_size == 8, memfix group sampling) are the dataclass defaults.

Train with examples/Robotwin/train_files/run_robotwin_train_v2.sh
(single YAML: examples/Robotwin/train_files/starvla_robotwin_v2.yaml).
"""

from __future__ import annotations

import os
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
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.geoplan import GatedGeoFusion, GeoFlowHead, GeoInseqBridge, GeoPlanQueryResampler, VGGTGeoTeacher
from starVLA.model.modules.memory import EpisodeMemoryBank, EventMemoryBank
from starVLA.model.modules.memory.legacy_compat import legacy_state_dict_pre_hook, remap_legacy_geoplan_cfg
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

DEFAULT_IMAGE_TOKEN_INDEX = 151655  # Qwen3-VL image placeholder token id (fallback; read at runtime)


# ──────────────────────────────────────────────────────────────────────
#  Anti-jitter action head (RoboTwin absolute joint qpos)
# ──────────────────────────────────────────────────────────────────────
class SmoothFlowmatchingActionHead(FlowmatchingActionHead):
    """GR00T flow-matching head + a second-difference (acceleration-matching) loss.

    Absolute-joint chunks are tracked directly by the controller, so any high-frequency
    residual left by the flow-matching sampler becomes visible arm jitter. The standard
    FM loss is pointwise and does not penalise this. Here, from the model's velocity
    prediction we reconstruct the implied clean chunk

        ``x_hat = x_t + (1 - t) * v_hat``   (exact when v_hat is the true velocity)

    and match its DISCRETE SECOND DIFFERENCE along the horizon to the ground truth's:
    high-frequency noise explodes under ``dd`` while the demonstrated dynamics are
    preserved (we match the GT acceleration, not zero => no over-smoothing bias).

    Extra config key: ``framework.action_model.action_smooth_coeff`` (0 disables).
    The last loss breakdown is stashed in ``self.last_loss_logs`` for wandb.
    """

    def __init__(self, full_config):
        super().__init__(full_config)
        self.action_smooth_coeff = float(
            getattr(full_config.framework.action_model, "action_smooth_coeff", 0.0)
        )
        self.last_loss_logs: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _second_diff(x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T-2, D] discrete second difference along the horizon."""
        return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]

    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None):
        # Mirrors FlowmatchingActionHead.forward, additionally keeping (noisy, t, pred)
        # to build the smoothness term. Kept self-contained so the parent stays untouched.
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_velocity = pred[:, -actions.shape[1]:]

        fm_loss = ((pred_velocity - velocity) ** 2).mean()

        total = fm_loss
        logs = {"action_fm_loss": fm_loss.detach()}
        if self.action_smooth_coeff > 0 and actions.shape[1] >= 3:
            # Implied clean chunk under the model's velocity: x_hat = x_t + (1 - t) * v_hat.
            x_hat = noisy_trajectory + (1 - t) * pred_velocity
            smooth = ((self._second_diff(x_hat) - self._second_diff(actions)) ** 2).mean()
            total = total + self.action_smooth_coeff * smooth
            logs["action_smooth_loss"] = smooth.detach()
        self.last_loss_logs = logs
        return total


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGeoGR00T_RoboTwin
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGeoGR00TRoboTwinDefaultConfig:
    """QwenGeoGR00T_RoboTwin defaults (RoboTwin-correct out of the box).

    YAML ``framework:`` values override these; extra keys are preserved."""

    name: str = "QwenGeoGR00T_RoboTwin"

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
            # RoboTwin dual-arm joint space: [left_joints(6), left_gripper, right_joints(6), right_gripper]
            "action_dim": 14,
            "state_dim": 14,
            "action_horizon": 50,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Anti-jitter: finer Euler integration of the flow ODE (was 4). Baked into the
            # checkpoint config -> eval does not need STARVLA_NUM_INFERENCE_TIMESTEPS.
            "num_inference_timesteps": 16,
            # Anti-jitter: weight of the second-difference (acceleration-matching) loss in
            # SmoothFlowmatchingActionHead. 0 disables (falls back to the plain FM loss).
            "action_smooth_coeff": 0.05,
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

    # === GeoPlan 3D auxiliary + memory (RoboTwin defaults) ===
    geoplan: dict = field(
        default_factory=lambda: {
            # --- master switches ---
            "use_geoplan": True,          # run the GeoPlan bridge + (if teacher available) aux loss
            "use_geo_aux": True,          # build/train the GeoFlowHead (only if a VGGT ckpt is set)
            "use_memory": True,           # run obs2/obs1/geo memory banks
            "geo_bridge_type": "inseq_token",  # "inseq_token" (OFT-style) | "perceiver"
            # action-head cross-attention context:
            #   "compressed"       -> [obs2 | obs1]                  (DEFAULT)
            #   "full_plus_memory" -> [full last_hidden | obs2 | obs1]
            #   "full"             -> [full last_hidden]            (== base QwenGR00T, no memory)
            "action_context_type": "compressed",
            # which geo_cond the GeoFlow aux loss supervises: "raw" | "memory" | "both"
            "geo_supervise": "raw",
            # --- EEF-in-pointcloud target (Strategy A; replaces the legacy endpose_rel target) ---
            # The dual-arm future EEF trajectory (world frame, from observation.eef) is mapped
            # into the VGGT point-cloud frame using the per-frame camera intrinsic/extrinsic
            # (observation.campose -> 'eef.campose' modality) plus a per-sample scale estimated
            # from gripper-pixel anchors, then normalized JOINTLY with the current point map.
            "use_eef_endpose": True,      # requires the 'eef' modality (observation.eef column)
            "eef_arms": 2,                # RoboTwin dual-arm
            # Scale anchors: how many leading future steps to use for the metric->VGGT scale
            # (more anchors = robuster median but only while the gripper stays in view).
            "geo_eef_anchor_steps": 4,
            # Reject anchors whose camera-frame depth is below this (metres) or whose pixel
            # projection falls outside the image.
            "geo_eef_min_depth": 0.05,
            # Reject per-sample scale estimates outside [1/geo_eef_scale_clip, geo_eef_scale_clip]
            # around the batch median scale (guards against degenerate anchors).
            "geo_eef_scale_clip": 4.0,
            # --- visual-token count normalization (obs1) ---
            "obs1_num_tokens": 64,         # M visual tokens after resampling
            "obs1_resampler_layers": 2,
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
            # --- scene-flow (future point map); NOT implemented (needs paired future frames) ---
            "geo_use_future": False,
            "geo_flow_stride": 5,
            # --- VGGT teacher (target generator) ---
            "geo_vggt_ckpt": None,        # required to actually compute the aux loss
            "geo_vggt_repo_path": None,
            "geo_vggt_img_size": 336,
            # --- memory bank (memfix: capacity == training group window) ---
            "mem_length": 8,
            "retrieval_layers": 2,
            "use_timestep_pe": True,
            "fusion_type": "gate",        # "gate" | "add"
            "consolidate_type": "tome",   # "tome" | "fifo"
            "update_fused": False,
            "causal_retrieval": True,     # drop non-strictly-earlier history (enforces causality)
            "validate_group_layout": False,
            "dataloader_type": "group",   # "group" clears the bank each batch (safe w/ random sampling)
            "group_size": 8,
            # eval memory: "stateless" resets the banks every predict_action call; "streaming"
            # keeps memory across steps (eval harness must reset at episode boundaries).
            # Config-only: this framework has NO env-var override.
            "eval_memory_mode": "stateless",
            # Opt-in event-guided memory. The default remains the validated bank.
            "mem_bank_type": "similarity",
            "event_write_threshold": 0.5,
            # Measured in ENVIRONMENT STEPS, not frames seen by the bank. Frames arrive far apart
            # (exec_horizon at inference, trajectory/group_size in training), so a small value fires
            # on every frame and silently disables the write gate. 0 = off; use 32-64 for a net.
            "event_min_write_stride": 0,
            "event_aux_dim": 0,
            "event_aux_mix": 1.0,
            "event_importance_bias": 1.0,
            "event_read_modulation": True,
            "event_evict_policy": "importance",
            "event_init_write_gate": 0.75,
            "event_scorer_normalize_inputs": False,
            "event_scorer_stop_gradient": False,
            "event_aux_key": "event_aux",
            "event_prior_key": "event_prior",
            "event_prior_controls_write": True,
            "event_supervision_coeff": 0.0,
            # For smoothed targets, compute pos_weight from their mean mass; the labeler's
            # summarize() helper reports the matching value.
            "event_supervision_pos_weight": 1.0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGeoGR00T_RoboTwin")
class Qwen_GeoGR00T_RoboTwin(baseframework):
    """Qwen3-VL + GR00T action head + GeoPlan 3D aux (EEF-in-pointcloud) — RoboTwin pipeline."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGeoGR00TRoboTwinDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Backbone hidden size (Qwen3-VL-4B == 2560). Read at runtime; never hardcode.
        self.hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.hidden_size
        # Image placeholder token id read from the backbone config (robust across Qwen versions).
        self.image_token_id = int(
            getattr(self.qwen_vl_interface.model.config, "image_token_id", DEFAULT_IMAGE_TOKEN_INDEX)
        )

        # Anti-jitter action head (flow-matching + second-difference smoothness loss).
        self.action_model: SmoothFlowmatchingActionHead = SmoothFlowmatchingActionHead(
            full_config=self.config
        )
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # ---- GeoPlan / memory config (defaults merged with YAML overrides) ----
        default_geo = QwenGeoGR00TRoboTwinDefaultConfig().geoplan
        user_geo = dict(self.config.framework.get("geoplan", {})) if self.config.framework.get("geoplan", None) else {}
        remap_legacy_geoplan_cfg(user_geo)  # accept configs saved under the old stream names
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
        self.obs1_num_tokens = int(gcfg["obs1_num_tokens"])
        # Config-only (NO env-var override in this framework — one pipeline, one config).
        self.eval_memory_mode = str(gcfg.get("eval_memory_mode", "stateless"))
        self.mem_dataloader_type = str(gcfg["dataloader_type"])
        self.mem_bank_type = str(gcfg.get("mem_bank_type", "similarity")).strip().lower()
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
        if self.action_context_type not in {"compressed", "full_plus_memory", "full"}:
            raise ValueError(f"Unknown action_context_type={self.action_context_type}")
        if self.geo_supervise not in {"raw", "memory", "both"}:
            raise ValueError(f"Unknown geo_supervise={self.geo_supervise}")
        if self.eval_memory_mode not in {"stateless", "streaming"}:
            raise ValueError(f"Unknown eval_memory_mode={self.eval_memory_mode!r}")

        # --- EEF-in-pointcloud target (Strategy A) ---
        # The dual-arm future EEF trajectory (world frame) is mapped into the VGGT point-cloud
        # frame via the per-frame camera intrinsic/extrinsic + an anchor-based per-sample scale,
        # then normalized JOINTLY with the current point map (see _prepare_geo_targets).
        self.use_eef_endpose = bool(gcfg.get("use_eef_endpose", True))
        self.eef_arms = int(gcfg.get("eef_arms", 2))
        if self.eef_arms < 1:
            raise ValueError(f"eef_arms must be >= 1, got {self.eef_arms}")
        self.geo_eef_anchor_steps = int(gcfg.get("geo_eef_anchor_steps", 4))
        self.geo_eef_min_depth = float(gcfg.get("geo_eef_min_depth", 0.05))
        self.geo_eef_scale_clip = float(gcfg.get("geo_eef_scale_clip", 4.0))

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

        # ---- visual-token count normalization (obs1) (Qwen3-VL vision token count is DYNAMIC) ----
        # M learnable queries cross-attend the (variable-length) vision tokens -> fixed [B, M, H].
        # work_dim == H so there is no channel down-projection (in_proj is Identity); this only fixes
        # the token LENGTH so the obs1 memory bank's cross-frame stacking and vl_embs stay length-stable.
        self.obs1_resampler = GeoPlanQueryResampler(
            dim=H,
            num_queries=self.obs1_num_tokens,
            num_layers=int(gcfg["obs1_resampler_layers"]),
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
                        f"[QwenGeoGR00T_RoboTwin] geo_emoji {self.geo_emoji!r} must tokenize to exactly one "
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

            # ---- non-destructive gated fusion of geo guidance into the visual observation tokens (obs1) ----
            self.geo_fusion = GatedGeoFusion(obs_dim=H, geo_dim=self.geo_work_dim)
            # ---- frozen VGGT teacher (held in a list -> NOT a registered submodule) ----
            # Treat None / empty string as "no teacher" (the aux 3D loss is then skipped).
            ckpt = gcfg["geo_vggt_ckpt"]
            self.geo_vggt_ckpt = ckpt if ckpt else None
            repo = gcfg["geo_vggt_repo_path"]
            self.geo_vggt_repo_path = repo if repo else None
            self.geo_vggt_img_size = int(gcfg["geo_vggt_img_size"])
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
                    "[QwenGeoGR00T_RoboTwin] No VGGT teacher checkpoint -> GeoFlowHead NOT built and the 3D "
                    "aux loss is disabled. The geo bridge + gated fusion still train via the action "
                    "loss. Set framework.geoplan.geo_vggt_ckpt to enable the 3D supervision."
                )
        else:
            self.use_geo_aux = False

        # ---- observation / geometric episode memory banks ----
        if self.use_memory:
            # memfix default: mem_length == group_size (8/8), so the streaming-inference bank
            # capacity matches the training window and ToMe consolidation behaves identically at
            # train and inference. Config-only; NO env-var override in this framework.
            _mem_len = int(gcfg["mem_length"])
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
            if self.mem_bank_type not in {"similarity", "event_guided"}:
                raise ValueError(
                    f"Unknown mem_bank_type={self.mem_bank_type!r}; expected 'similarity' or 'event_guided'"
                )
            if self.mem_bank_type == "event_guided":
                mem_kwargs.update(
                    write_threshold=float(gcfg.get("event_write_threshold", 0.5)),
                    min_write_stride=int(gcfg.get("event_min_write_stride", 0)),
                    aux_dim=self.event_aux_dim,
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
                bank_cls = EventMemoryBank
                print(
                    "[QwenGeoGR00T_RoboTwin] mem_bank_type='event_guided' "
                    f"(mem_length={_mem_len}, "
                    f"scorer_normalize={mem_kwargs['scorer_normalize_inputs']}, "
                    f"scorer_stop_gradient={mem_kwargs['scorer_stop_gradient']})"
                )
            else:
                bank_cls = EpisodeMemoryBank
            self.obs2_mem_bank = bank_cls(token_size=H, **mem_kwargs)
            self.obs1_mem_bank = bank_cls(token_size=H, **mem_kwargs)
            if self.use_geoplan:
                self.geo_mem_bank = bank_cls(token_size=self.geo_work_dim, **mem_kwargs)

        self._last_geo_logs: Dict[str, torch.Tensor] = {}
        # Checkpoints saved under the previous stream naming still load (keys are renamed on the fly).
        self._register_load_state_dict_pre_hook(legacy_state_dict_pre_hook)

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
                f"[QwenGeoGR00T_RoboTwin] samples {bad} do not have exactly {k} geo marker tokens "
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
    def _extract_state_token(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """State token = final VLM hidden state at the last non-padding position -> [B, 1, H] (padding-side robust)."""
        if attention_mask is None:
            return last_hidden[:, -1:, :]
        H = last_hidden.shape[-1]
        last_idx = attention_mask.long().cumsum(dim=1).argmax(dim=1)  # [B] index of last valid token
        return last_hidden.gather(1, last_idx[:, None, None].expand(-1, 1, H))

    def _encode_and_extract(self, batch_images, instructions):
        """Run Qwen3-VL once and return (obs2, obs1, geo_cond, full_hidden, attention_mask)."""
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
            raise RuntimeError("[QwenGeoGR00T_RoboTwin] no image tokens found; cannot build visual observation tokens.")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, H]
            # obs1: token-count normalization. M queries attend ONLY the vision positions (mask=1),
            # so a DYNAMIC #vision tokens (per frame / per sample) becomes a fixed [B, M, H].
            obs1 = self.obs1_resampler(last_hidden, attention_mask=vision_mask)  # [B, M, H]
            geo_cond = None
            if self.use_geoplan and self.geo_bridge_type == "perceiver":
                geo_cond = self.geo_resampler(last_hidden, attention_mask=attention_mask)  # [B, K, work_dim]

        # Cast the (small) extracted tensors to fp32 for the downstream side-modules. This keeps the
        # obs2/obs1/geo conditioning in full precision (bf16's 8-bit mantissa is coarse and can inject
        # noise into low-magnitude channels), and avoids bf16-input vs fp32-weight mismatches. The
        # side-modules self-align dtype where needed, and the action head runs in its own fp32
        # autocast downstream, so keeping these fp32 is the safe/stable choice now that OOM is
        # resolved via smaller batch / geo_vggt_img_size.
        # obs2: the state token = hidden state at the LAST NON-PADDING position (robust to the
        # padding side via argmax-of-cumsum). Under Qwen's causal attention it has attended to the
        # full multimodal context.
        obs2 = self._extract_state_token(last_hidden, attention_mask).float()   # [B, 1, H]
        obs1 = obs1.float()                                               # [B, M, H]
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
        return obs2, obs1, geo_cond, last_hidden, attention_mask

    @staticmethod
    def _param_dtype(module: torch.nn.Module, default: torch.dtype = torch.float32) -> torch.dtype:
        """Dtype of a module's parameters (falls back to ``default`` for parameter-less modules)."""
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return default

    def _event_memory_inputs(self, examples: List[dict], device: torch.device):
        """Collect explicit online event deltas / keyframe scores without future-label leakage."""
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
        if event_prior is None or self.event_supervision_coeff <= 0:
            return None
        banks = [self.obs2_mem_bank, self.obs1_mem_bank]
        if self.use_geoplan and hasattr(self, "geo_mem_bank"):
            banks.append(self.geo_mem_bank)
        losses = []
        for bank in banks:
            scores = getattr(bank, "last_scores", None)
            if not scores or "g_write_logit" not in scores:
                continue
            # Logits + pos_weight, not BCE on probabilities: keyframes are a few percent of frames,
            # so an unweighted objective is minimised by never predicting one, which would keep the
            # write head shut and the bank empty.
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
        if not (self.use_memory and self.mem_bank_type == "event_guided"):
            return {}
        named_banks = [("obs2", self.obs2_mem_bank), ("obs1", self.obs1_mem_bank)]
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
        obs2,
        obs1,
        geo_cond,
        episode_ids,
        timesteps,
        reset_before_batch,
        update_state,
        event_aux=None,
        event_prior=None,
    ):
        """Run the memory banks and the gated geo fusion; return the fused (obs2, obs1).

        State lifecycle is controlled EXPLICITLY (not via train()/eval()): ``reset_before_batch``
        clears the banks first (group-mode training), ``update_state`` writes the current frames.

        Every sub-module input is aligned to that module's parameter dtype first: the fp32
        conditioning (obs2/obs1/geo_cond from `_encode_and_extract`) must match whatever dtype the
        side-modules ended up in (bf16 under DeepSpeed / `--use_bf16`, fp32 otherwise), so no
        bf16-input vs fp32-weight (or vice-versa) mismatch can ever raise.
        """
        mem_kw = dict(reset_before_batch=reset_before_batch, update_state=update_state)
        if self.mem_bank_type == "event_guided":
            write_prior = event_prior if self.event_prior_controls_write else None
            mem_kw.update(aux=event_aux, event_prior=write_prior)
        if self.use_memory:
            obs2 = obs2.to(self._param_dtype(self.obs2_mem_bank))
            obs1 = obs1.to(self._param_dtype(self.obs1_mem_bank))
            obs2 = self.obs2_mem_bank.retrieve_and_write(obs2, episode_ids, timesteps, **mem_kw)
            obs1 = self.obs1_mem_bank.retrieve_and_write(obs1, episode_ids, timesteps, **mem_kw)
        # geo_guidance is the MEMORY-fused geo output; it is None when memory is off (so callers can
        # tell raw from memory-fused). Fusion still runs on raw geo_cond when there is no memory.
        geo_guidance = None
        if self.use_geoplan and geo_cond is not None:
            if self.use_memory:
                geo_cond = geo_cond.to(self._param_dtype(self.geo_mem_bank))
                geo_guidance = self.geo_mem_bank.retrieve_and_write(geo_cond, episode_ids, timesteps, **mem_kw)
            fusion_geo = geo_guidance if geo_guidance is not None else geo_cond
            # GatedGeoFusion runs raw Linear layers and does NOT self-align, so align both inputs.
            _fdt = self._param_dtype(self.geo_fusion)
            obs1 = obs1.to(_fdt)
            fusion_geo = fusion_geo.to(_fdt)
            obs1 = self.geo_fusion(obs1, fusion_geo)
        return obs2, obs1, geo_guidance

    def _eef_world_to_vggt(
        self,
        eef: torch.Tensor,       # [B, T, arms*3] future EEF xyz in the WORLD/robot frame (metric)
        campose: torch.Tensor,   # [B, 23] current-frame camera params (K[9] | E[12] | H_orig, W_orig)
        world_full: torch.Tensor,  # [B, S, S, 3] FULL-RES VGGT point map (VGGT frame)
    ):
        """Map the metric world-frame EEF trajectory into the VGGT point-cloud frame (Strategy A).

        Per sample:
          1. world -> camera: ``X_c = R X_w + t`` with the OpenCV extrinsic (metric camera frame).
          2. camera -> pixel: perspective projection with the intrinsic, rescaled from the
             original image resolution to the VGGT input size ``S``.
          3. metric->VGGT scale ``s``: VGGT single-frame point maps live in the camera frame UP TO
             AN UNKNOWN SCALE, so ``s = median(z_vggt(u, v) / z_cam)`` over "anchor" projections —
             the first ``geo_eef_anchor_steps`` steps of both arms, where the gripper is (nearly)
             at its current, VISIBLE position, so the point map at that pixel IS the gripper
             surface. Anchors that project out of view or behind the camera are dropped.
          4. ``eef_vggt = X_c * s``: the FULL free-space trajectory (heights preserved — no
             surface snapping), now co-registered with the point map.

        Returns:
          eef_vggt [B, arms*T, 3] (arm-major flatten), valid [B] bool (False -> supervise with
          w_eef = 0; the scale could not be estimated for that sample).
        """
        B, T_all, D = eef.shape
        T = self.action_horizon
        arms = self.eef_arms
        if D != arms * 3:
            raise ValueError(f"eef last dim {D} != eef_arms*3 ({arms * 3})")
        S = float(self.geo_vggt_img_size)
        dev = eef.device

        e_w = eef[:, -T:, :].reshape(B, T, arms, 3).float()              # [B, T, arms, 3]
        K = campose[:, :9].reshape(B, 3, 3).float()                       # intrinsic (orig resolution)
        E = campose[:, 9:21].reshape(B, 3, 4).float()                     # extrinsic_cv: world -> cam
        H_orig = campose[:, 21].clamp_min(1.0)                            # [B]
        W_orig = campose[:, 22].clamp_min(1.0)                            # [B]

        # 1) world -> camera (metric).
        R, tvec = E[:, :, :3], E[:, :, 3]                                 # [B,3,3], [B,3]
        x_c = torch.einsum("bij,btaj->btai", R, e_w) + tvec[:, None, None, :]  # [B, T, arms, 3]
        z = x_c[..., 2]

        # 2) camera -> pixel at the VGGT input resolution (non-uniform rescale of the square resize).
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        z_safe = z.clamp_min(1e-6)
        u = fx[:, None, None] * x_c[..., 0] / z_safe + cx[:, None, None]  # orig-pixel coords
        v = fy[:, None, None] * x_c[..., 1] / z_safe + cy[:, None, None]
        u = u * (S / W_orig[:, None, None])
        v = v * (S / H_orig[:, None, None])

        # 3) anchor-based per-sample scale.
        n_anchor = max(1, min(self.geo_eef_anchor_steps, T))
        au = u[:, :n_anchor].reshape(B, -1)                               # [B, A] A = n_anchor*arms
        av = v[:, :n_anchor].reshape(B, -1)
        az = z[:, :n_anchor].reshape(B, -1)
        in_view = (au >= 0) & (au <= S - 1) & (av >= 0) & (av <= S - 1) & (az > self.geo_eef_min_depth)
        ui = au.round().long().clamp(0, int(S) - 1)
        vi = av.round().long().clamp(0, int(S) - 1)
        b_idx = torch.arange(B, device=dev)[:, None].expand_as(ui)
        z_vggt = world_full[b_idx, vi, ui, 2]                             # [B, A]
        ratio = z_vggt / az.clamp_min(1e-6)
        anchor_ok = in_view & torch.isfinite(ratio) & (ratio > 0)

        # Per-sample median over valid anchors (masked: invalid -> nan, then nanmedian).
        ratio_masked = torch.where(anchor_ok, ratio, torch.full_like(ratio, float("nan")))
        s_per = torch.nanmedian(ratio_masked, dim=1).values               # [B]
        valid = anchor_ok.any(dim=1) & torch.isfinite(s_per) & (s_per > 0)

        # Cross-batch sanity: reject scales wildly off the batch median (degenerate anchors).
        if valid.any():
            batch_med = torch.median(s_per[valid])
            clip = self.geo_eef_scale_clip
            valid = valid & (s_per > batch_med / clip) & (s_per < batch_med * clip)
            s_fill = batch_med
        else:
            s_fill = torch.tensor(1.0, device=dev)
        s_per = torch.where(valid, s_per, s_fill.expand_as(s_per))

        # 4) metric camera frame -> VGGT frame; arm-major flatten to [B, arms*T, 3].
        eef_vggt = x_c * s_per[:, None, None, None]
        eef_vggt = eef_vggt.permute(0, 2, 1, 3).reshape(B, arms * T, 3)
        return eef_vggt, valid

    @torch.no_grad()
    def _prepare_geo_targets(self, batch_images, device, eef=None, campose=None):
        """Run the (frozen) VGGT teacher ONCE per batch and assemble the geo target sequence.

        Targets = [current-frame point map | future EEF trajectory], BOTH in the SAME normalized
        VGGT frame (Strategy A): the trajectory is mapped world -> camera -> VGGT scale via
        :meth:`_eef_world_to_vggt`, then point map + trajectory are normalized JOINTLY with the
        point map's centroid/RMS. The EEF target is therefore a function of the actions AND the
        scene geometry / camera pose — non-redundant with the action loss, and reconstructing it
        requires the same 3D understanding as the point map (synergy instead of competition).

        Returns None when no teacher is available.
        """
        teacher = self._get_geo_teacher(device)
        if teacher is None or not self.use_geo_aux:
            return None
        if hasattr(teacher, "eval"):
            teacher.eval()
        if eef is None or campose is None:
            raise ValueError(
                "[QwenGeoGR00T_RoboTwin] the EEF-in-pointcloud geo target requires both the 'eef' "
                "and 'campose' sample keys. Use robot_type 'robotwin50_v2' (eef.full + eef.campose "
                "modalities) and run add_campose_to_lerobot.py on the dataset first."
            )

        # Current-frame RGB (primary view = cam_high), resized to S x S in [0, 1].
        S = self.geo_vggt_img_size
        cur = torch.stack(
            [TF.resize(self._pil_to_chw01(imgs[0]), [S, S], antialias=True) for imgs in batch_images], dim=0
        ).to(device)

        # Run VGGT directly (mirrors teacher.compute_targets with two identical frames, but keeps
        # the FULL-RESOLUTION point map, which the anchor-scale lookup needs).
        seq = torch.stack([cur, cur], dim=1)  # [B, 2, 3, S, S]
        with torch.autocast("cuda", dtype=teacher.autocast_dtype):
            out = teacher.model(seq)
        world_full = out["world_points"].float()[:, 0]       # [B, S, S, 3] (VGGT frame)
        conf_full = out["world_points_conf"].float()[:, 0]   # [B, S, S]

        pts_cur = teacher._downsample(world_full)                              # [B, P, 3]
        w_cur = teacher._downsample(conf_full.unsqueeze(-1)).squeeze(-1)       # [B, P]
        lo = w_cur.amin(dim=1, keepdim=True)
        hi = w_cur.amax(dim=1, keepdim=True)
        w_cur = (w_cur - lo) / (hi - lo).clamp_min(1e-6)

        # EEF trajectory -> VGGT frame (Strategy A).
        eef_vggt, eef_valid = self._eef_world_to_vggt(eef.to(device).float(), campose.to(device).float(), world_full)

        # JOINT per-sample normalization: centroid/RMS from the point map, applied to BOTH
        # segments so the trajectory stays embedded in the normalized scene.
        if self.geo_normalize_pts:
            mu = pts_cur.mean(dim=1, keepdim=True)                                  # [B, 1, 3]
            centered = pts_cur - mu
            rms = centered.pow(2).sum(dim=-1).mean(dim=1).clamp_min(1e-12).sqrt()   # [B]
            pts_cur = centered / rms[:, None, None]
            eef_vggt = (eef_vggt - mu) / rms[:, None, None]

        # Zero-weight the EEF segment of samples whose metric->VGGT scale failed.
        w_eef = eef_valid.float()[:, None].expand(-1, eef_vggt.shape[1]).contiguous()  # [B, W]

        return {
            "target": torch.cat([pts_cur, eef_vggt], dim=1),
            "w_cur": w_cur,
            "w_eef": w_eef,
            "eef_valid_frac": eef_valid.float().mean().detach(),
        }

    def _geo_loss_from_targets(self, geo_cond, geo_targets):
        """GeoFlow loss for one condition against PRE-COMPUTED targets (reuses VGGT output)."""
        return self.geo_head.loss(
            geo_targets["target"],
            geo_cond.float(),
            w_cur=geo_targets["w_cur"],
            w_eef=geo_targets["w_eef"],
        )

    @staticmethod
    def _pil_to_chw01(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _build_action_context(self, obs2, obs1, full_hidden, attention_mask):
        """Assemble the action-head cross-attention context + its key-padding/attention mask."""
        B = obs2.shape[0]
        if self.action_context_type == "compressed":
            vl = torch.cat([obs2, obs1], dim=1)
            mask = torch.ones(vl.shape[:2], dtype=torch.bool, device=vl.device)
            return vl, mask
        # Keep full_hidden in its native dtype (bf16 under DeepSpeed) so that
        # torch.cat([fh, obs2, obs1]) does not hit a Float vs BFloat16 mismatch.
        # The action_model is wrapped by its own fp32 autocast context at the call site.
        fh = full_hidden
        if attention_mask is not None:
            am = attention_mask.to(dtype=torch.bool, device=fh.device)
        else:
            am = torch.ones(fh.shape[:2], dtype=torch.bool, device=fh.device)
        if self.action_context_type == "full":
            return fh, am
        # full_plus_memory: [full last_hidden | obs2 | obs1]
        total_len = fh.shape[1] + obs2.shape[1] + obs1.shape[1]
        max_seq = int(self.config.framework.action_model.get("max_seq_len", 1024))
        if total_len > max_seq:
            raise ValueError(
                f"action_context_type='full_plus_memory' produces {total_len} context tokens "
                f"(> action_model.max_seq_len={max_seq}). Increase max_seq_len or use 'compressed'."
            )
        extra = torch.ones(B, obs2.shape[1] + obs1.shape[1], dtype=torch.bool, device=fh.device)
        vl = torch.cat([fh, obs2.to(fh.dtype), obs1.to(fh.dtype)], dim=1)
        mask = torch.cat([am, extra], dim=1)
        return vl, mask

    def reset_memory(self) -> None:
        """Clear ALL memory banks (every episode). Call at episode start in streaming evaluation."""
        # self._infer_timestep = 0
        if self.use_memory:
            self.obs2_mem_bank.reset()
            self.obs1_mem_bank.reset()
            if self.use_geoplan and hasattr(self, "geo_mem_bank"):
                self.geo_mem_bank.reset()

    # Eval-harness facing aliases / per-episode cleanup (for parallel-env streaming inference).
    reset_episode_memory = reset_memory
    _reset_memory = reset_memory

    def clear_episode_memory(self, episode_id) -> None:
        """Drop the memory of a single finished episode (use with parallel-env streaming eval)."""
        if self.use_memory:
            self.obs2_mem_bank.clear_episode(episode_id)
            self.obs1_mem_bank.clear_episode(episode_id)
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
        # Current-frame camera params (K | extrinsic_cv | orig H, W) for the EEF-in-pointcloud
        # target. The dataloader packs it per-sample as [T, 23]; only frame 0 (current) is used.
        campose = [example["campose"] for example in examples] if "campose" in examples[0] else None

        obs2, obs1, geo_cond, full_hidden, attn_mask = self._encode_and_extract(batch_images, instructions)
        episode_ids, timesteps = self._episode_meta(examples, obs2.device)
        event_aux, event_prior = self._event_memory_inputs(examples, obs2.device)
        self._maybe_debug_batch_layout(episode_ids, timesteps)
        # Group-mode training: clear the bank at the start of every batch (each batch is fresh groups).
        obs2, obs1, geo_guidance = self._apply_memory_and_fusion(
            obs2, obs1, geo_cond, episode_ids, timesteps,
            reset_before_batch=(self.mem_dataloader_type == "group"),
            update_state=True,
            event_aux=event_aux,
            event_prior=event_prior,
        )

        vl_embs, enc_mask0 = self._build_action_context(obs2, obs1, full_hidden, attn_mask)

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
            campose_t = None
            if campose is not None:
                campose_t = torch.tensor(np.array(campose), device=vl_embs.device, dtype=torch.float32)
                if campose_t.ndim == 3:  # [B, T, 23] -> current frame only
                    campose_t = campose_t[:, 0]

            action_loss = self.action_model(
                vl_embs_rep, actions_rep, state_rep, encoder_attention_mask=enc_mask
            )

        total = action_loss
        out: Dict[str, torch.Tensor] = {}
        out.update(self._event_memory_logs())
        # Surface the action-head loss breakdown (FM term + smoothness term).
        for k, v in getattr(self.action_model, "last_loss_logs", {}).items():
            out[k] = v
        event_loss = self._event_supervision_loss(event_prior)
        if event_loss is not None:
            total = total + self.event_supervision_coeff * event_loss
            out["memory/event_loss"] = event_loss.detach()
        if self.use_geo_aux and self.training and geo_cond is not None:
            # Run the expensive VGGT teacher ONCE; reuse the same targets for raw/memory conditions.
            geo_targets = self._prepare_geo_targets(batch_images, geo_cond.device, eef=eef_t, campose=campose_t)
            if geo_targets is not None:
                out["geo/eef_valid_frac"] = geo_targets["eef_valid_frac"]
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

        obs2, obs1, geo_cond, full_hidden, attn_mask = self._encode_and_extract(batch_images, instructions)
        episode_ids, timesteps = self._episode_meta(examples, obs2.device)
        event_aux, event_prior = self._event_memory_inputs(examples, obs2.device)
        # if self.eval_memory_mode == "streaming":
        #     timesteps = np.array([self._infer_timestep] * len(examples), dtype=np.int64)
        #     self._infer_timestep += 1
        # Inference: stateless mode already reset above; streaming mode accumulates across steps and
        # expects reset_episode_memory() at episode boundaries. Never reset_before_batch mid-episode.
        obs2, obs1, _geo_guidance = self._apply_memory_and_fusion(
            obs2, obs1, geo_cond, episode_ids, timesteps,
            reset_before_batch=False,
            update_state=(self.eval_memory_mode == "streaming"),
            event_aux=event_aux,
            event_prior=event_prior,
        )
        vl_embs, enc_mask = self._build_action_context(obs2, obs1, full_hidden, attn_mask)

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

        obs2, obs1, geo_cond, full_hidden, attn_mask = self._encode_and_extract(batch_images, instructions)
        episode_ids, timesteps = self._episode_meta(examples, obs2.device)
        event_aux, event_prior = self._event_memory_inputs(examples, obs2.device)
        # Reproduce predict_action's memory/fusion so geo_guidance matches deployment behaviour.
        # IMPORTANT: by default do NOT write this frame into the memory bank — predict_geo is a
        # visualization/ablation hook, and if it were called alongside predict_action during a
        # streaming rollout it would double-write the current frame (inflating memory length and
        # corrupting retrieval). Pass update_memory=True only for a standalone geo-only stream.
        _update_mem = bool(kwargs.get("update_memory", False)) and (self.eval_memory_mode == "streaming")
        _obs2, _obs1, geo_guidance = self._apply_memory_and_fusion(
            obs2, obs1, geo_cond, episode_ids, timesteps,
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
    parser.add_argument("--config_yaml", type=str, default="examples/Robotwin/train_files/starvla_robotwin_v2.yaml")
    args, _ = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    model: Qwen_GeoGR00T_RoboTwin = Qwen_GeoGR00T_RoboTwin(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Fake campose: identity-ish intrinsic/extrinsic + orig HW (only exercises the code path).
    _K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]], dtype=np.float32).reshape(-1)
    _E = np.hstack([np.eye(3, dtype=np.float32), np.array([[0], [0], [1.0]], dtype=np.float32)]).reshape(-1)
    _campose = np.concatenate([_K, _E, np.array([480.0, 640.0], dtype=np.float32)])
    sample = {
        "action": np.random.uniform(-1, 1, size=(50, 14)).astype(np.float16),
        "state": np.random.uniform(-1, 1, size=(1, 14)).astype(np.float16),
        "eef": np.random.uniform(-0.3, 0.3, size=(50, 6)).astype(np.float32),
        "campose": np.tile(_campose[None, :], (50, 1)).astype(np.float32),
        "image": [image, image, image],
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
