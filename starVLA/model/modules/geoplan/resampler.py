"""Perceiver-style learnable-query resampler that reads the VLA backbone hidden states.

A small set of learnable ``geometry / planning`` queries cross-attend to the LLM last-layer
hidden states (visual patch tokens + text tokens). The resulting query embeddings condition the
lightweight GeoPlan diffusion head. Because the queries only *read* the backbone hidden states,
gradients from the auxiliary geometry/planning loss still flow into the VLA backbone, shaping its
internal representations -- while at inference the whole resampler is simply not invoked, so the
main VLA computation is byte-for-byte identical to the original model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Cross-attention: queries read the backbone hidden states.
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + attn_out
        # Self-attention among queries.
        qn = self.norm_sa(q)
        sa_out, _ = self.self_attn(qn, qn, qn, need_weights=False)
        q = q + sa_out
        # FFN.
        q = q + self.mlp(self.norm_mlp(q))
        return q


class GeoInseqBridge(nn.Module):
    """In-sequence special-token bridge.

    Holds K learnable special tokens (appended into the LLM sequence) and a projection from the LLM
    hidden dim to the geo work dim. Wrapping these in a module (rather than bare parameters on the
    root) ensures they are captured by the framework's checkpoint save/load (which iterates child
    modules).
    """

    def __init__(self, num_queries: int, dim: int, work_dim: int) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.special_tokens = nn.Parameter(torch.randn(num_queries, dim) * (dim ** -0.5))
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, work_dim))

    def expand(self, batch_size: int) -> torch.Tensor:
        return self.special_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # Align input to the proj weight dtype (bf16 under DeepSpeed) so a caller
        # passing fp32 hidden states does not trigger a dtype mismatch in LayerNorm/Linear.
        return self.proj(x.to(self.proj[0].weight.dtype))


class GeoPlanQueryResampler(nn.Module):
    """Maps backbone hidden states -> ``num_queries`` conditioning tokens.

    To keep this train-only module lightweight (and to force the *backbone* -- rather than this
    side head -- to encode the geometry), the backbone hidden states (``dim``, e.g. 4096) are first
    projected down to a small working dim (``work_dim``, e.g. 512) where all attention/FFN happens.
    """

    def __init__(
        self,
        dim: int,
        num_queries: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        work_dim: Optional[int] = 512,
    ) -> None:
        super().__init__()
        if work_dim is None:
            work_dim = dim
        self.dim = dim
        self.work_dim = work_dim
        self.num_queries = num_queries
        self.in_proj = nn.Linear(dim, work_dim) if work_dim != dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(num_queries, work_dim) * (work_dim ** -0.5))
        self.blocks = nn.ModuleList(
            [_CrossAttnBlock(work_dim, num_heads) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(work_dim)

    @property
    def out_dim(self) -> int:
        return self.work_dim

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden: [B, L, dim] backbone hidden states (visual + text tokens).
        attention_mask: [B, L] with 1 = valid, 0 = padding (or None).
        returns: [B, num_queries, work_dim]
        """
        b = hidden.shape[0]
        # Align input to the module's parameter dtype (bf16 under DeepSpeed). in_proj
        # may be nn.Identity (no params), so key off self.queries which always exists.
        hidden = hidden.to(self.queries.dtype)
        kv = self.in_proj(hidden)  # [B, L, work_dim]
        q = self.queries.unsqueeze(0).expand(b, -1, -1).to(kv.dtype)
        # nn.MultiheadAttention key_padding_mask uses True = ignore.
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        for block in self.blocks:
            q = block(q, kv, key_padding_mask)
        return self.out_norm(q)
