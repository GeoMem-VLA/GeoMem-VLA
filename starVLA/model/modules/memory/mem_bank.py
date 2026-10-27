"""Episode memory banks (observation and geometric streams) for GeoMem-VLA.

``EpisodeMemoryBank`` keeps a bounded, per-episode history for ONE token stream. It retrieves
earlier entries of the same episode via cross-attention (queries = current tokens, keys/values =
stored tokens plus learned timestamp encodings), fuses the retrieved context with the current tokens
through a learnable gate, and consolidates the bank by merging the most similar temporally adjacent
entries (ToMe) or by FIFO eviction. The framework instantiates one bank per stream: two observation
streams (visual tokens, final VLM state token) and one geometric stream (GeoPlan tokens).

Design notes:
  - Everything is ``token_size``-agnostic; the caller passes the working dimension at runtime
    (Qwen3-VL-4B hidden size = 2560, geo work dim = 512, etc.). Nothing is hardcoded.
  - No separate vision compressor is used: in starVLA the visual observation tokens are the Qwen3-VL
    vision hidden states (already at the backbone hidden dim).

Memory state lives in ``self.bank`` (a python dict keyed by episode id). For ``dataloader_type
== "group"`` the bank is cleared at the start of every batch, which makes this safe to use even when
the dataloader samples frames randomly (no cross-batch state leak): when no same-episode history is
present, retrieval falls back to the current tokens and -- because both gate inputs are then equal --
the fusion output equals the current tokens exactly (a genuine identity in that degenerate case).

Distributed / DeepSpeed notes:
  - ``self.bank`` is plain per-rank python state (detached GPU tensors), NOT registered parameters or
    buffers, so it is never sharded by ZeRO, never enters ``state_dict``, and is not synced across
    ranks. This is correct for data-parallel training: each rank keeps the memory for its own local
    micro-batch. With ``dataloader_type="group"`` the bank is cleared every batch, so its footprint
    is bounded by ``mem_length`` and nothing leaks across steps.
  - Stored history is detached, so retrieval never keeps the cross-frame autograd graph. HOWEVER, ``retrieve_and_write`` also MUTATES ``self.bank``; if the
    region that calls it were re-run by gradient checkpointing, a frame could be written into memory
    twice. Keep the memory banks OUTSIDE any gradient-checkpoint recompute region (the Qwen backbone
    checkpointing does not re-run these external modules), or pass ``update_state=False`` on recompute.

Causality:
  - ``retrieve_and_write`` consolidates each sample into the bank AFTER producing its output, so within a
    batch sample ``i`` can only retrieve samples ``0..i-1``. For this to be temporally causal the
    dataloader MUST present same-episode frames in increasing-time order. As a hard guard,
    ``causal_retrieval=True`` additionally drops any history entry whose stored timestep is not
    strictly earlier than the current frame's timestep (a no-op when ordering is already correct).
    ToMe-merged summaries store the LATER source timestep so merging cannot leak future information.

State lifecycle (explicit, NOT tied to train()/eval()):
  - The caller controls clearing/writing via ``reset_before_batch`` and ``update_state`` on
    ``retrieve_and_write`` (plus ``reset()`` / ``clear_episode(eid)``). Group-mode training passes
    ``reset_before_batch=True``; streaming inference resets at episode boundaries instead.

Stream-mode limitation:
  - The current ``stream`` logic clears the previous episode whenever the episode id changes between
    adjacent samples, so it only supports ONE continuous episode at a time (A,A,A -> B,B,B ...). It
    does NOT support interleaved parallel episodes (A,B,C / A,B,C); that would need a done-mask driven
    cleanup instead. Use ``group`` mode (the default) unless your dataloader guarantees single-stream.
"""

from __future__ import annotations

import math
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeStampEncoder(nn.Module):
    """Embeds scalar timesteps into vector representations (sinusoidal + MLP)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        return self.mlp(t_freq)


class HistoryAttentionBlock(nn.Module):
    """Single cross-attention + FFN block (pre-projected q/k/v, post-LN residual)."""

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

    def forward(self, query: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = self.attn_norm(query + attn_out)
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class HistoryGate(nn.Module):
    """Learnable gate that blends the current tokens with the retrieved episode history.

    NOTE on initialization: with proj weight/bias ~ 0, ``sigmoid(proj(...)) ~ 0.5`` at the start of
    training, so the output is ~ ``0.5 * current + 0.5 * retrieved`` -- a BALANCED AVERAGE, not an
    identity. (It only becomes an exact identity in the degenerate case where the two inputs are
    equal, e.g. when there is no retrieved history.)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Align inputs to the proj weight dtype. Retrieved/consolidated memory can
        # drift to fp32 across frames (stale stored feats, PE, token-merge), which
        # would otherwise hit "mat1 and mat2 must have the same dtype" here.
        w_dtype = self.proj.weight.dtype
        x1 = x1.to(w_dtype)
        x2 = x2.to(w_dtype)
        scale = torch.sigmoid(self.proj(torch.cat([x1, x2], dim=-1)))
        return scale * x1 + (1 - scale) * x2


class EpisodeMemoryBank(nn.Module):
    """Per-episode memory bank for one token stream: causal retrieval, gated fusion, consolidation.

    The bank is dimension-agnostic: ``token_size`` is the working dim of the tokens it operates on
    (e.g. the backbone hidden size for the observation banks, or the geo work dim for the geo bank).

    IMPORTANT: ``retrieve_and_write`` stacks per-frame features across an episode, so every frame fed for
    a given episode must have the SAME number of tokens ``N``. The state token (N=1) and geo (N=K) tokens
    are already length-stable; visual tokens from a backbone with a DYNAMIC vision-token count
    (e.g. Qwen3-VL) MUST be length-normalized to a fixed N (e.g. via a Perceiver resampler) BEFORE
    being passed here, otherwise the cross-frame stacking will raise a shape error.
    """

    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
        causal_retrieval: bool = True,
        validate_group_layout: bool = False,
    ) -> None:
        super().__init__()
        # Explicit validation (NOT `assert`, which is stripped under `python -O`).
        if dataloader_type not in {"stream", "group"}:
            raise ValueError(f"Invalid dataloader_type={dataloader_type!r}")
        if fusion_type not in {"gate", "add"}:
            raise ValueError(f"Invalid fusion_type={fusion_type!r}")
        if consolidate_type not in {"fifo", "tome"}:
            raise ValueError(f"Invalid consolidate_type={consolidate_type!r}")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if mem_length <= 0:  # mem_length=0 would make the FIFO slice bank[-0:] loop forever
            raise ValueError("mem_length must be positive")
        if retrieval_layers <= 0:
            raise ValueError("retrieval_layers must be positive")

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused
        self.causal_retrieval = causal_retrieval
        self.validate_group_layout = validate_group_layout

        self.history_attn = nn.ModuleList(
            [HistoryAttentionBlock(self.token_size) for _ in range(self.retrieval_layers)]
        )

        if self.fusion_type == "gate":
            self.history_gate = HistoryGate(self.token_size)

        if self.use_timestep_pe:
            self.time_encoder = TimeStampEncoder(
                self.token_size, frequency_embedding_size=max(8, self.token_size // 4)
            )
        else:
            self.time_encoder = None

        self.reset()

    def reset(self) -> None:
        # bank[episode_id] = [(timestep, feat[N, D]), ...]
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id) -> None:
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _merge_closest_pair(self, episode_id) -> None:
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return
        # Vectorized adjacent cosine similarity (single GPU->CPU sync via one argmax .item()).
        feats = torch.stack([feat for (_, feat) in bank], dim=0)  # [T, N, D] (or [T, D])
        f = feats.flatten(1)                                       # [T, N*D]
        sims = F.cosine_similarity(f[:-1], f[1:], dim=-1)          # [T-1]
        idx_max = int(sims.argmax().item())
        timestep_i, feat_i = bank[idx_max]
        timestep_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)
        # Causal safety: a merged summary contains BOTH source frames, so its effective time is the
        # LATER of the two. Storing the earlier timestamp would let the causal filter leak future
        # information (e.g. merge t=3 & t=7 -> if stored as t=3, a later t=5 query would read it).
        if timestep_i is None:
            merged_ts = timestep_j
        elif timestep_j is None:
            merged_ts = timestep_i
        else:
            merged_ts = max(int(timestep_i), int(timestep_j))
        bank[idx_max] = (merged_ts, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _append_and_compact(self, episode_id, feat: torch.Tensor, timestep) -> None:
        if episode_id not in self.bank:
            self.bank[episode_id] = []
        self.bank[episode_id].append((timestep, feat.detach().clone()))
        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length :]
            elif self.consolidate_type == "tome":
                self._merge_closest_pair(episode_id)
            else:
                raise NotImplementedError

    def _validate_group_layout(self, episode_ids: np.ndarray, timesteps: np.ndarray) -> None:
        """Assert the batch is laid out as fixed-size, single-episode, strictly time-ordered groups."""
        B = len(episode_ids)
        for start in range(0, B, self.group_size):
            end = min(start + self.group_size, B)
            group_eids = episode_ids[start:end]
            if len(np.unique(group_eids)) != 1:
                raise ValueError(
                    f"Group [{start}:{end}] contains multiple episodes: {np.asarray(group_eids).tolist()}"
                )
            if timesteps is not None:
                group_ts = np.asarray(timesteps[start:end])
                if len(group_ts) > 1 and np.any(np.diff(group_ts) <= 0):
                    raise ValueError(
                        f"Group [{start}:{end}] is not strictly time-ordered: {group_ts.tolist()}"
                    )

    def retrieve_and_write(
        self,
        tokens: torch.Tensor,        # [B, N, D]
        episode_ids: np.ndarray,
        timesteps: np.ndarray,
        reset_before_batch: bool = False,
        update_state: bool = True,
        history_intervention: str = "none",
        intervention_seed: int = 0,
    ) -> torch.Tensor:               # [B, N, D]
        """Retrieve + fuse cross-frame memory for a batch.

        State lifecycle is controlled EXPLICITLY by the caller (not by ``self.training``):
          - ``reset_before_batch``: clear the whole bank first (use True for group-mode training,
            where each batch is a fresh set of groups).
          - ``update_state``: write the current frames into the bank afterwards (use False for
            diagnostics / repeated passes over the same data to avoid double-writing).
        For streaming inference, call ``reset()`` at each episode boundary and pass
        ``reset_before_batch=False, update_state=True``.
        """
        if history_intervention not in {"none", "empty", "shuffle_time"}:
            raise ValueError(f"Unknown history_intervention={history_intervention!r}")
        if history_intervention != "none":
            if self.training:
                raise ValueError("History interventions are evaluation-only")
            if self.update_fused:
                raise ValueError("History interventions require update_fused=False to preserve raw writes")
            if history_intervention == "shuffle_time" and not self.use_timestep_pe:
                raise ValueError("shuffle_time requires use_timestep_pe=True")
        self.last_history_diagnostics = []
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
        if self.validate_group_layout and self.dataloader_type == "group":
            self._validate_group_layout(episode_ids, timesteps)

        B, N, D = tokens.shape
        outputs = []

        # Single dtype chokepoint: align incoming tokens to the bank's parameter
        # dtype (bf16 under DeepSpeed) so current tokens, stored history, retrieval
        # and fusion all stay on one dtype.
        try:
            param_dtype = next(self.parameters()).dtype
            if tokens.dtype != param_dtype:
                tokens = tokens.to(param_dtype)
        except StopIteration:
            pass

        if reset_before_batch:
            self.bank.clear()
            self.eid_stream = None
        if self.dataloader_type == "stream":
            first_eid = episode_ids[0]
            if self.eid_stream is not None and self.eid_stream != first_eid:
                self.clear_episode(self.eid_stream)
            self.eid_stream = first_eid

        for i in range(B):
            eid = episode_ids[i]
            # Within-batch episode boundary management (explicit; independent of train()/eval()).
            if self.dataloader_type == "group":
                if i > 0 and i % self.group_size == 0:
                    self.clear_episode(episode_ids[i - self.group_size])
            elif self.dataloader_type == "stream":
                if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                    self.clear_episode(episode_ids[i - 1])
                    self.eid_stream = episode_ids[i]

            # 1) memory retrieval
            cur_tokens = tokens[i].unsqueeze(0)  # (1, N, D)
            hist = self.bank.get(eid, [])
            # Causal guard (independent of use_timestep_pe): only retrieve strictly-earlier frames.
            # No-op when the dataloader already orders same-episode frames by increasing time.
            if self.causal_retrieval and timesteps is not None and len(hist) > 0:
                ti = int(timesteps[i])
                hist = [(t, f) for (t, f) in hist if t is not None and int(t) < ti]
            # Alter this read only, after causal filtering. Stored history and the
            # normal raw-feature write/ToMe path below remain untouched.
            available = len(hist)
            permutation = list(range(available))
            if history_intervention == "empty":
                hist = []
            elif history_intervention == "shuffle_time" and available > 1:
                # Sattolo permutation: every entry changes temporal assignment.
                # A private RNG avoids perturbing the action head's noise stream.
                key = f"{int(intervention_seed)}:{eid}:{int(timesteps[i])}".encode()
                rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
                for j in range(available - 1, 0, -1):
                    k = rng.randrange(j)
                    permutation[j], permutation[k] = permutation[k], permutation[j]
                hist = [(t, hist[permutation[j]][1]) for j, (t, _) in enumerate(hist)]
            self.last_history_diagnostics.append({
                "available": available,
                "read": len(hist),
                "intervention": history_intervention,
                "permutation": permutation if history_intervention == "shuffle_time" else None,
            })
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                hist_tokens = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)  # (1, T*N, D)
                if self.use_timestep_pe:
                    hist_timesteps = torch.tensor([t for t, _ in hist]).to(cur_tokens.device)
                    pe = self.time_encoder(hist_timesteps).unsqueeze(0)  # (1, T, D)
                    pe = pe.repeat_interleave(N, dim=1)  # (1, T*N, D)
                else:
                    pe = torch.zeros_like(hist_tokens)
                query = cur_tokens
                for block in self.history_attn:
                    query = block(query, hist_tokens + pe, hist_tokens)
                retrieved = query
            else:
                retrieved = cur_tokens  # (1, N, D)

            # 2) adaptive fusion
            if self.fusion_type == "add":
                fused_feats = (cur_tokens + retrieved) * 0.5
            else:  # gate
                fused_feats = self.history_gate(cur_tokens, retrieved)

            outputs.append(fused_feats)

            # 3) consolidate (timestep stored independently of the PE switch)
            if update_state:
                timestep_i = timesteps[i] if timesteps is not None else None
                if self.update_fused:
                    self._append_and_compact(eid, fused_feats.squeeze(0), timestep_i)
                else:
                    self._append_and_compact(eid, tokens[i], timestep_i)

        return torch.cat(outputs, dim=0)  # [B, N, D]
