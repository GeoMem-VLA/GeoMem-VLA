"""Gated geometry fusion: injects memory-augmented GeoPlan guidance into the visual observation tokens.

The visual observation tokens query the (geo) guidance via cross-attention to get ``h``, then a learned
gate adaptively mixes them:

    h   = FFN(CrossAttn(obs_tokens, geo, geo))
    g   = sigmoid(MLP([h, obs_tokens]))
    out = g * obs_tokens + (1 - g) * h

Crucially, the ``h`` path is zero-initialized and the gate is biased so that ``g ≈ 1`` at the start
of training. This means ``out ≈ obs_tokens`` initially -> the base policy is *not* disturbed, and
the model only learns to pull in the geo guidance if it actually helps (non-destructive add-on).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedGeoFusion(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        geo_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        gate_bias_init: float = 4.0,
    ) -> None:
        super().__init__()
        self.geo_proj = nn.Linear(geo_dim, obs_dim)
        self.norm_q = nn.LayerNorm(obs_dim)
        self.norm_kv = nn.LayerNorm(obs_dim)
        self.cross_attn = nn.MultiheadAttention(obs_dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(obs_dim)
        hidden = int(obs_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, obs_dim),
        )
        self.gate = nn.Linear(2 * obs_dim, obs_dim)

        # Zero-init the `h` path so h ≈ 0 at start; bias the gate so g ≈ 1 -> out ≈ obs_tokens.
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(self, obs_tokens: torch.Tensor, geo_guidance: torch.Tensor) -> torch.Tensor:
        """
        obs_tokens:   [B, N_o, obs_dim]
        geo_guidance: [B, K, geo_dim]
        returns:      [B, N_o, obs_dim]  (gated fusion; ≈ obs_tokens at init)
        """
        kv = self.norm_kv(self.geo_proj(geo_guidance))
        q = self.norm_q(obs_tokens)
        attn, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h = attn + self.ffn(self.norm_ffn(attn))
        g = torch.sigmoid(self.gate(torch.cat([h, obs_tokens], dim=-1)))
        return g * obs_tokens + (1.0 - g) * h
