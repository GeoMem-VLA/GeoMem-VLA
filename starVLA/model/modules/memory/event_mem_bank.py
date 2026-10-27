"""Event-guided (surprise-gated) episodic memory bank.

This is an ADDITIVE variant of :class:`~starVLA.model.modules.memory.mem_bank.EpisodeMemoryBank`. The
similarity-consolidated bank is left untouched so already-validated checkpoints keep reproducing
exactly; pick this one explicitly via ``framework.geoplan.mem_bank_type = "event_guided"``.

What the similarity-consolidated bank does
------------------------------------------
Every frame is written into the episode bank, and a single gate decides *how much of the retrieved
history* is blended into the current frame. The gate therefore only controls memory READS.

What this bank adds
-------------------
1. ``s_scene`` -- scene-change score from the frame-to-frame feature delta. Answers "did the
   environment change?".
2. ``s_event`` -- event-novelty score. Answers "does the current observation disagree with the
   stored history?". By default it is derived from the absolute feature mismatch between the current
   observation and the bank summary, which needs no extra inputs; when the caller has proprio/eef/
   object *delta* signals it can pass them as ``aux`` and blend them with that fallback, matching
   the ``ΔP_eef / Δh_obj`` path while still training deployment without aux.
3. A **write gate** ``g_t = σ(W_g · [s_scene; s_event; mem_summary; obs_summary])`` that decides
   whether the frame is worth *storing*. The bank then holds key events rather than every frame.
4. Importance-aware retrieval and eviction: each slot keeps its gate value, retrieval biases
   attention towards important slots, and eviction drops the least important slot instead of blindly
   merging the most similar adjacent pair.

Where the gradients come from (important, and a real constraint)
-----------------------------------------------------------------
The two gates use two different differentiable paths. ``g_write`` is stored as a live
importance tensor and biases retrieval by later frames in the *same* forward call; ``g_read``
modulates fusion on the current frame. At the end of the call, stored importance values are detached
so stream-mode training cannot accidentally backpropagate through a graph from an earlier optimizer
step. High read surprise means the history poorly matches the present, so the current observation is
trusted more.

If the dataloader can provide an online keyframe signal, ``last_scores['g_write']`` retains the
learned prediction while ``last_scores['g_write_effective']`` records the external value that
actually controlled writes. Offline labels can instead be kept out of ``retrieve_and_write`` and used by
the framework for supervision only.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mem_bank import EpisodeMemoryBank

logger = logging.getLogger(__name__)


class BiasedHistoryAttentionBlock(nn.Module):
    """Cross-attention + FFN block accepting an additive bias over the memory slots.

    Mirrors :class:`~starVLA.model.modules.memory.mem_bank.HistoryAttentionBlock` exactly except for
    ``attn_bias``, which lets the caller steer attention towards high-importance slots.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        if attn_bias is not None:
            attn_bias = attn_bias.to(q.dtype)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        x = self.attn_norm(query + attn_out)
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class EventScorer(nn.Module):
    """Scene-change score, event-novelty score, and the write / read-surprise gates.

    All heads are deliberately tiny: they consume pooled (mean-over-token) summaries, so the
    parameter cost is negligible next to the retrieval blocks.

    Why two gates and not one
    -------------------------
    A single score cannot serve both jobs, because the two ends want opposite initial values: the
    write gate must start HIGH (store everything, so retrieval has history to learn from) while the
    read-surprise gate must start LOW (do not suppress memory before the scorer knows anything).
    Sharing one score starves the bank at initialisation -- measured: 1 of 8 frames written, leaving
    retrieval with almost no history.

    They share the trunk (scene/event scores and the pooled summaries) and only split at the last
    layer, whose biases are initialised so that at step 0 this bank stays close to the original
    :class:`EpisodeMemoryBank`: ``g_write ~ 0.75`` (write every frame) and ``g_read ~ 0.0025`` (only a tiny
    fusion perturbation). It is not mathematically identical because a finite sigmoid cannot produce
    exactly zero and the learned importance values are only approximately equal.
    """

    def __init__(
        self,
        token_size: int,
        aux_dim: int = 0,
        aux_mix: float = 1.0,
        hidden: Optional[int] = None,
        init_write_gate: float = 0.75,
        normalize_inputs: bool = False,
        stop_gradient: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= aux_mix <= 1.0:
            raise ValueError(f"aux_mix must be in [0, 1], got {aux_mix}")
        h = hidden if hidden is not None else max(32, min(256, token_size // 4))
        self.token_size = token_size
        self.aux_dim = aux_dim
        self.aux_mix = float(aux_mix)
        # Both switches are opt-in so existing event-guided checkpoints and configs retain their
        # exact scoring path. Normalisation removes feature-norm drift while preserving gradients;
        # stop_gradient additionally turns the scorer into an auxiliary observer of the backbone.
        self.normalize_inputs = bool(normalize_inputs)
        self.stop_gradient = bool(stop_gradient)

        self.scene_head = nn.Sequential(nn.Linear(token_size, h), nn.SiLU(), nn.Linear(h, 1))
        # Keep separate adapters for the learned-feature fallback and optional external deltas.
        # A single head cannot accept both token_size and aux_dim; the previous implementation
        # crashed whenever aux_dim > 0 but a batch omitted aux instead of falling back as promised.
        self.event_head = nn.Sequential(nn.Linear(token_size, h), nn.SiLU(), nn.Linear(h, 1))
        # LayerNorm first: the aux channels are raw physical deltas whose magnitudes sit around
        # 1e-3..1e-2 (measured on LIBERO end-effector displacements). Fed straight into a Linear
        # they are indistinguishable from zero, the head output collapses onto its bias, and
        # ``s_event`` becomes a constant -- the channel would be wired up but carry no information.
        self.aux_event_head = (
            nn.Sequential(nn.LayerNorm(aux_dim), nn.Linear(aux_dim, h), nn.SiLU(), nn.Linear(h, 1))
            if aux_dim > 0
            else None
        )

        self.obs_proj = nn.Linear(token_size, h)
        self.mem_proj = nn.Linear(token_size, h)
        self.gate_trunk = nn.Sequential(nn.Linear(2 * h + 2, h), nn.SiLU())
        self.write_head = nn.Linear(h, 1)
        self.read_head = nn.Linear(h, 1)

        # Start as the original bank: write everything, never suppress memory on read.
        p = min(max(init_write_gate, 1e-4), 1 - 1e-4)
        nn.init.normal_(self.write_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.write_head.bias, float(np.log(p / (1 - p))))
        nn.init.normal_(self.read_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.read_head.bias, -6.0)  # sigmoid(-6) ~ 0.0025

    def _prepare_feature(self, value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        """Prepare a pooled backbone feature for event scoring only.

        The actual memory slot/retrieval/fusion tensors are never normalised or detached here. When
        enabled, L2 normalisation is computed in fp32 for numerical stability and cast back to the
        scorer dtype; autograd remains intact unless ``stop_gradient`` was explicitly requested.
        """
        value = value.to(device=device, dtype=self.obs_proj.weight.dtype).reshape(-1)
        if value.numel() != self.token_size:
            raise ValueError(
                f"Expected pooled feature with {self.token_size} values, got {value.numel()}"
            )
        if self.stop_gradient:
            value = value.detach()
        if self.normalize_inputs:
            value = F.normalize(value.float(), p=2.0, dim=-1, eps=1e-6).to(
                self.obs_proj.weight.dtype
            )
        return value

    def forward(
        self,
        cur_pool: torch.Tensor,          # (D,)
        prev_pool: Optional[torch.Tensor],
        mem_pool: Optional[torch.Tensor],
        residual_pool: Optional[torch.Tensor] = None,
        aux_delta: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = self.obs_proj.weight.dtype
        cur_pool = self._prepare_feature(cur_pool, device=cur_pool.device)
        prev_score = (
            self._prepare_feature(prev_pool, device=cur_pool.device)
            if prev_pool is not None
            else None
        )
        mem_score = (
            self._prepare_feature(mem_pool, device=cur_pool.device)
            if mem_pool is not None
            else None
        )

        # 1) scene change: absolute feature delta against the previous frame. The first frame of an
        #    episode has no predecessor, so use its magnitude relative to a zero reference.
        delta = (
            (cur_pool - prev_score).abs()
            if prev_score is not None
            else cur_pool.abs()
        )
        s_scene = torch.sigmoid(self.scene_head(delta)).reshape(())

        # 2) event novelty: always evaluate the learned feature-residual path. When an external
        #    proprio/object delta is present, blend its score with the residual score. Keeping a
        #    non-zero feature share during training is important because deployment may omit aux;
        #    the old exclusive branch left event_head completely untrained in that case.
        if self.normalize_inputs and mem_score is not None:
            # Preserve the *magnitude of disagreement between unit-normalised observations*.
            # Normalising the already-absolute residual itself would force every non-zero residual
            # to unit length and discard exactly the novelty magnitude this branch should measure.
            residual_in = (cur_pool - mem_score).abs()
        elif residual_pool is not None:
            residual_in = self._prepare_feature(residual_pool, device=cur_pool.device)
        else:
            residual_in = torch.zeros_like(cur_pool)
        s_event_feature = torch.sigmoid(self.event_head(residual_in)).reshape(())

        if aux_delta is not None and self.aux_dim > 0:
            ev_in = aux_delta.to(device=cur_pool.device, dtype=dtype).reshape(-1)
            if ev_in.numel() != self.aux_dim:
                raise ValueError(
                    f"Expected aux_delta with {self.aux_dim} values, got {ev_in.numel()}"
                )
            s_event_aux = torch.sigmoid(self.aux_event_head(ev_in)).reshape(())
            s_event = (1.0 - self.aux_mix) * s_event_feature + self.aux_mix * s_event_aux
        else:
            s_event = s_event_feature

        # 3) shared trunk -> write gate + read-surprise gate
        obs = self.obs_proj(cur_pool)
        mem = (
            self.mem_proj(mem_score)
            if mem_score is not None
            else torch.zeros_like(obs)
        )
        z = self.gate_trunk(torch.cat([obs, mem, s_scene.reshape(1), s_event.reshape(1)], dim=-1))
        # The write LOGIT is returned alongside the probability so a supervised keyframe loss can use
        # `binary_cross_entropy_with_logits`, which supports pos_weight and is numerically stable.
        # The logit is exposed so the framework can apply a numerically stable, appropriately
        # balanced BCE objective for either hard or smoothed keyframe targets.
        w_logit = self.write_head(z).reshape(())
        g_write = torch.sigmoid(w_logit)
        g_read = torch.sigmoid(self.read_head(z)).reshape(())
        return s_scene, s_event, g_write, g_read, w_logit


class EventMemoryBank(EpisodeMemoryBank):
    """Episodic memory that stores *events* rather than every frame.

    Extra arguments on top of :class:`EpisodeMemoryBank`:
      write_threshold   gate value above which a frame is committed to the bank.
      min_write_stride  anti-starvation fallback: force a commit once this many ENVIRONMENT STEPS
                        passed without one. 0 disables it (the default).
                        Read the unit carefully -- it is environment steps, not "frames seen by this
                        bank", and consecutive frames reaching the bank are far apart: streaming
                        inference calls the server every ``exec_horizon`` (=8) steps, and group-mode
                        training samples ~16 frames spread over a ~280-step episode (~17 steps
                        apart). A small value such as 4 is therefore true at EVERY frame, forcing a
                        write every time and silently disabling the write gate and any external
                        keyframe prior entirely. If you want a safety net, size it well above the
                        sampling interval (32-64); the first-frame seed already guarantees the bank
                        is never completely empty.
      aux_dim           width of the optional ``aux`` event-delta signal (eef displacement / object
                        state change). 0 disables it.
      aux_mix           contribution of the auxiliary event score when aux is present. Values below
                        1 keep the feature-residual fallback trained for deployments without aux.
      importance_bias   scale of the log-importance added to the retrieval attention logits.
      read_modulation   if True, surprise down-weights the retrieved memory during fusion. This is
                        the direct gradient path for the read head; the write head uses the
                        importance-bias path or an external keyframe loss.
      evict_policy      "importance" drops the least important slot, "tome" keeps the original
                        merge-most-similar-adjacent-pair behaviour.
      init_write_gate   initial write probability. Defaults to 0.75 so the bank starts out storing
                        every frame, i.e. staying close to the original bank, and only becomes
                        selective as the gate learns.
      scorer_normalize_inputs
                        L2-normalise pooled backbone features inside EventScorer only. The actual
                        memory storage/retrieval/fusion tensors remain untouched.
      scorer_stop_gradient
                        stop EventScorer losses/gates from updating their pooled backbone inputs.
                        Scorer parameters and the normal action/memory paths still train.
    """

    def __init__(
        self,
        *args,
        write_threshold: float = 0.5,
        min_write_stride: int = 0,
        aux_dim: int = 0,
        aux_mix: float = 1.0,
        importance_bias: float = 1.0,
        read_modulation: bool = True,
        evict_policy: str = "importance",
        init_write_gate: float = 0.75,
        scorer_normalize_inputs: bool = False,
        scorer_stop_gradient: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if evict_policy not in {"importance", "tome"}:
            raise ValueError(f"Invalid evict_policy={evict_policy!r}")
        if not 0.0 <= write_threshold <= 1.0:
            raise ValueError(f"write_threshold must be in [0, 1], got {write_threshold}")
        if min_write_stride < 0:
            raise ValueError("min_write_stride must be >= 0")
        if aux_dim < 0:
            raise ValueError("aux_dim must be >= 0")
        if not 0.0 <= aux_mix <= 1.0:
            raise ValueError("aux_mix must be in [0, 1]")
        if importance_bias < 0:
            raise ValueError("importance_bias must be >= 0")

        self.write_threshold = float(write_threshold)
        self.min_write_stride = int(min_write_stride)
        self.importance_bias = float(importance_bias)
        self.read_modulation = bool(read_modulation)
        self.evict_policy = evict_policy
        self.aux_dim = int(aux_dim)
        self.aux_mix = float(aux_mix)
        self.scorer_normalize_inputs = bool(scorer_normalize_inputs)
        self.scorer_stop_gradient = bool(scorer_stop_gradient)

        # Retrieval blocks that understand the importance bias. Copy the already-initialized parent
        # weights before replacing the modules, so merely opting into the event variant does not
        # silently redraw the whole retrieval stack.
        parent_history_attn = self.history_attn
        biased_history_attn = nn.ModuleList(
            [BiasedHistoryAttentionBlock(self.token_size) for _ in range(self.retrieval_layers)]
        )
        for biased, parent in zip(biased_history_attn, parent_history_attn):
            biased.load_state_dict(parent.state_dict())
        self.history_attn = biased_history_attn
        self.event_scorer = EventScorer(
            self.token_size,
            aux_dim=self.aux_dim,
            aux_mix=self.aux_mix,
            init_write_gate=init_write_gate,
            normalize_inputs=self.scorer_normalize_inputs,
            stop_gradient=self.scorer_stop_gradient,
        )

        # Per-frame diagnostics for logging or an external keyframe-supervision loss.
        self.last_scores: dict = {}

        if self.min_write_stride > 0:
            logger.warning(
                "EventMemoryBank: min_write_stride=%d is measured in ENVIRONMENT STEPS. Frames "
                "reaching this bank are typically %s steps apart (exec_horizon at inference, "
                "trajectory/group_size in training), so any value below that forces a write on every "
                "frame and disables both the write gate and any external keyframe prior. Use 0, or a "
                "value well above the sampling interval.",
                self.min_write_stride, "8-17",
            )
        if self.dataloader_type == "group" and self.mem_length < self.group_size:
            # Deliberately the opposite advice from a plain EpisodeMemoryBank. There, capacity had to stay
            # below group_size so the merge path was exercised in training instead of firing only at
            # inference. Here the write gate already keeps the bank sparse (a handful of events per
            # episode), so a capacity that a whole group can overflow
            # re-creates the same asymmetry in reverse: training overflows while a gated inference
            # rollout never does. Size capacity ABOVE the expected number of events per episode and
            # the consolidation path simply never runs on either side.
            logger.warning(
                "EventMemoryBank: mem_length=%d < group_size=%d. With event-gated writing the "
                "bank normally holds only a few slots per episode, so a capacity this small will "
                "overflow during training but not during a gated rollout. Prefer mem_length >= the "
                "expected events per episode (the current LIBERO 4-suite labeler measures ~3-6).",
                self.mem_length, self.group_size,
            )

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        super().reset()
        # bank[eid] entries are (timestep, feat, importance); these track the extra per-episode state
        self._prev_pool: dict = {}
        self._last_write_t: dict = {}

    def clear_episode(self, episode_id) -> None:
        super().clear_episode(episode_id)
        self._prev_pool.pop(episode_id, None)
        self._last_write_t.pop(episode_id, None)

    # ----------------------------------------------------------- consolidation
    def _evict_least_important(self, episode_id) -> None:
        bank = self.bank.get(episode_id, [])
        if len(bank) < 2:
            return
        # Never evict the most recent slot: the immediate past is what the current frame needs most.
        importance = torch.stack([entry[2].detach().float().cpu() for entry in bank[:-1]])
        idx = int(importance.argmin().item())
        bank.pop(idx)

    def _merge_most_similar(self, episode_id) -> None:
        """ToMe merge, importance-aware: the summary inherits the max importance and later timestep."""
        bank = self.bank.get(episode_id, [])
        if len(bank) < 2:
            return
        with torch.no_grad():
            feats = torch.stack([f for (_, f, _) in bank], dim=0)
            sims = F.cosine_similarity(feats.flatten(1)[:-1], feats.flatten(1)[1:], dim=-1)
            i = int(sims.argmax().item())
        t_i, f_i, w_i = bank[i]
        t_j, f_j, w_j = bank[i + 1]
        # Causal safety (same rule as the original): a merged summary contains BOTH frames, so its
        # effective time is the LATER one, otherwise a later query could read it as "past".
        if t_i is None:
            merged_t = t_j
        elif t_j is None:
            merged_t = t_i
        else:
            merged_t = max(int(t_i), int(t_j))
        with torch.no_grad():
            merged_f = (0.5 * (f_i + f_j)).detach().clone()
        # torch.maximum keeps the autograd path so the write gate still gets a gradient through
        # slots that were merged later in the same forward pass.
        bank[i] = (merged_t, merged_f, torch.maximum(w_i, w_j))
        bank.pop(i + 1)

    def _commit(self, episode_id, feat: torch.Tensor, timestep, importance: torch.Tensor) -> None:
        """Append a slot. The FEATURE is detached (no cross-frame graph, as in the original bank),
        but the IMPORTANCE is kept as a live tensor.

        That asymmetry is deliberate and is what makes the write gate trainable. Within a training
        group, frame ``i`` retrieves slots written by frames ``0..i-1`` of the SAME forward pass, so
        their ``g_write`` tensors are still in the graph; feeding them into the retrieval attention
        bias gives ``write_head`` a gradient from the action loss. Without this the write head would
        receive no gradient at all, stay at its initial value forever, and the bank would degenerate
        to "store every frame". The importance state is detached at the end of this process call, so
        no live graph crosses an optimizer step.
        """
        if episode_id not in self.bank:
            self.bank[episode_id] = []
        self.bank[episode_id].append((timestep, feat.detach().clone(), importance))
        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length :]
            elif self.evict_policy == "importance":
                self._evict_least_important(episode_id)
            else:
                self._merge_most_similar(episode_id)

    def _detach_stored_importance(self) -> None:
        """Cut cross-forward graphs while preserving all within-call retrieval gradients.

        Attention biases already created earlier in this call keep their own references to the live
        tensors, so replacing the bank entries here does not break the current loss graph. It only
        prevents the next call (notably stream-mode training) from reusing a graph after backward.
        """
        for eid, entries in self.bank.items():
            self.bank[eid] = [(t, f, importance.detach()) for (t, f, importance) in entries]

    # -------------------------------------------------------------- main entry
    def retrieve_and_write(
        self,
        tokens: torch.Tensor,            # [B, N, D]
        episode_ids: np.ndarray,
        timesteps: np.ndarray,
        reset_before_batch: bool = False,
        update_state: bool = True,
        aux: Optional[torch.Tensor] = None,        # [B, aux_dim] eef / object-state delta, optional
        event_prior: Optional[torch.Tensor] = None,  # [B] external keyframe score in [0, 1], optional
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B, N, D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.token_size:
            raise ValueError(f"Expected token dim {self.token_size}, got {tokens.shape[-1]}")
        if episode_ids is None or len(episode_ids) != tokens.shape[0]:
            raise ValueError("episode_ids must be provided and match the batch size")
        if (self.use_timestep_pe or self.causal_retrieval) and timesteps is None:
            raise ValueError("timesteps must be provided when use_timestep_pe=True or causal_retrieval=True")
        if timesteps is not None and len(timesteps) != tokens.shape[0]:
            raise ValueError("timesteps length does not match the batch size")
        if aux is not None and aux.shape[0] != tokens.shape[0]:
            raise ValueError("aux batch size does not match tokens")
        if aux is not None:
            if self.aux_dim <= 0:
                raise ValueError("aux was provided but this bank was constructed with aux_dim=0")
            if aux.numel() != tokens.shape[0] * self.aux_dim:
                raise ValueError(
                    f"aux must contain exactly {self.aux_dim} values per sample; "
                    f"got shape {tuple(aux.shape)}"
                )
            aux = aux.reshape(tokens.shape[0], self.aux_dim).to(tokens.device)
        if event_prior is not None:
            if event_prior.numel() != tokens.shape[0]:
                raise ValueError(
                    f"event_prior must contain one score per sample; got shape {tuple(event_prior.shape)}"
                )
            event_prior = event_prior.reshape(tokens.shape[0]).to(tokens.device)
            if not bool(torch.isfinite(event_prior).all().item()):
                raise ValueError("event_prior contains NaN or Inf")
            if bool(((event_prior < 0) | (event_prior > 1)).any().item()):
                raise ValueError("event_prior values must be in [0, 1]")
        if self.validate_group_layout and self.dataloader_type == "group":
            self._validate_group_layout(episode_ids, timesteps)

        B, N, D = tokens.shape
        outputs = []

        try:
            param_dtype = next(self.parameters()).dtype
            if tokens.dtype != param_dtype:
                tokens = tokens.to(param_dtype)
        except StopIteration:
            pass

        if reset_before_batch:
            self.bank.clear()
            self._prev_pool.clear()
            self._last_write_t.clear()
            self.eid_stream = None
        if self.dataloader_type == "stream":
            first_eid = episode_ids[0]
            if self.eid_stream is not None and self.eid_stream != first_eid:
                self.clear_episode(self.eid_stream)
            self.eid_stream = first_eid

        s_scene_all, s_event_all, write_logit_all = [], [], []
        write_pred_all, write_effective_all, read_all, wrote_all = [], [], [], []

        # Pull the external prior to host memory once. The write decision needs a python bool, so
        # reading it per frame would cost one GPU->CPU sync per frame per bank (~48 per step for
        # obs1+obs2+geo at B=16); hoisting it removes them all on the supervised path.
        prior_host = event_prior.detach().float().cpu().tolist() if event_prior is not None else None

        for i in range(B):
            eid = episode_ids[i]
            if self.dataloader_type == "group":
                if i > 0 and i % self.group_size == 0:
                    self.clear_episode(episode_ids[i - self.group_size])
            elif self.dataloader_type == "stream":
                if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                    self.clear_episode(episode_ids[i - 1])
                    self.eid_stream = episode_ids[i]

            cur_tokens = tokens[i].unsqueeze(0)  # (1, N, D)
            hist = self.bank.get(eid, [])
            if self.causal_retrieval and timesteps is not None and len(hist) > 0:
                ti = int(timesteps[i])
                hist = [h for h in hist if h[0] is not None and int(h[0]) < ti]

            # ---- 1) retrieval, biased towards important slots
            if len(hist) > 0:
                hist_feats = [f for (_, f, _) in hist]
                hist_tokens = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)
                if self.use_timestep_pe:
                    hist_ts = torch.tensor([t for (t, _, _) in hist]).to(cur_tokens.device)
                    pe = self.time_encoder(hist_ts).unsqueeze(0).repeat_interleave(N, dim=1)
                else:
                    pe = torch.zeros_like(hist_tokens)

                if self.importance_bias > 0.0:
                    # Differentiable for slots written earlier in this same forward pass -- this is
                    # the only gradient path into the write head (see `_commit`).
                    w = torch.stack(
                        [x[2].to(cur_tokens.device).reshape(()) for x in hist]
                    ).clamp_min(1e-4)
                    # additive logit bias, broadcast over queries and repeated per token slot
                    bias = (self.importance_bias * torch.log(w)).repeat_interleave(N).view(1, 1, -1)
                else:
                    bias = None

                query = cur_tokens
                for block in self.history_attn:
                    query = block(query, hist_tokens + pe, hist_tokens, attn_bias=bias)
                retrieved = query
                mem_pool = hist_tokens.mean(dim=(0, 1))
            else:
                retrieved = cur_tokens
                mem_pool = None

            # ---- 2) event scoring
            cur_pool = cur_tokens.mean(dim=(0, 1))
            # Cross-attention returns query + attended history, not a standalone prediction of the
            # current frame. Compare pooled current/history features directly and use the magnitude
            # so opposite token changes cannot cancel each other.
            residual_pool = (cur_pool - mem_pool).abs() if mem_pool is not None else None
            aux_delta = aux[i] if (aux is not None and self.aux_dim > 0) else None
            s_scene, s_event, g_write_pred, g_read, w_logit = self.event_scorer(
                cur_pool, self._prev_pool.get(eid), mem_pool,
                residual_pool=residual_pool, aux_delta=aux_delta,
            )
            g_write_effective = g_write_pred
            if event_prior is not None:
                # An external keyframe detector replaces the learned gate value; retain g_write_pred
                # so it can still be trained to imitate the detector. First-frame seeding and the
                # anti-starvation stride remain safety overrides around this effective value.
                g_write_effective = event_prior[i].to(
                    device=g_write_pred.device, dtype=g_write_pred.dtype
                ).reshape(())

            # ---- 3) fusion; read-surprise down-weights memory and carries the scorer's gradient
            if self.fusion_type == "add":
                fused = (cur_tokens + retrieved) * 0.5
            else:
                fused = self.history_gate(cur_tokens, retrieved)
            if self.read_modulation and len(hist) > 0:
                # g_read -> 1 means history poorly matches this frame: trust the observation.
                fused = g_read * cur_tokens.to(fused.dtype) + (1.0 - g_read) * fused
            outputs.append(fused)

            # ---- 4) write only the frames worth remembering
            wrote = 0.0
            if update_state:
                ts_i = timesteps[i] if timesteps is not None else None
                last_t = self._last_write_t.get(eid)
                stale = (
                    self.min_write_stride > 0
                    and ts_i is not None
                    and last_t is not None
                    and int(ts_i) - int(last_t) >= self.min_write_stride
                )
                # Always seed an episode with its first frame, otherwise nothing can ever be recalled.
                if prior_host is not None:
                    should_write = prior_host[i] > self.write_threshold
                else:
                    should_write = g_write_effective.detach().item() > self.write_threshold
                if len(self.bank.get(eid, [])) == 0 or should_write or stale:
                    src = fused.squeeze(0) if self.update_fused else tokens[i]
                    self._commit(eid, src, ts_i, g_write_effective)
                    self._last_write_t[eid] = ts_i
                    wrote = 1.0
                self._prev_pool[eid] = cur_pool.detach()

            s_scene_all.append(s_scene)
            s_event_all.append(s_event)
            write_logit_all.append(w_logit)
            write_pred_all.append(g_write_pred)
            write_effective_all.append(g_write_effective)
            read_all.append(g_read)
            wrote_all.append(wrote)

        # Exposed so the framework can log them or attach a supervised keyframe loss on `g_write`.
        self.last_scores = {
            "s_scene": torch.stack(s_scene_all),
            "s_event": torch.stack(s_event_all),
            "g_write": torch.stack(write_pred_all),
            "g_write_logit": torch.stack(write_logit_all),
            "g_write_effective": torch.stack(write_effective_all),
            "g_read": torch.stack(read_all),
            "wrote": torch.tensor(wrote_all, device=tokens.device),
        }
        result = torch.cat(outputs, dim=0)
        self._detach_stored_importance()
        return result
