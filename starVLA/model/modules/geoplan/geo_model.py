"""Self-contained flow-matching GeoPlan head for starVLA.

Uses flow-matching (same paradigm as starVLA's GR00T action head), no external diffusion dependency.
It reconstructs a target token sequence whose channels are 3D coords (xyz):

    [ current point map (Pc) | (optional) future point map (Pf) | EEF waypoints (W) ]

conditioned on the GeoPlan tokens (`geo_cond`) via cross-attention. Train-only.

Design notes (addressing common failure modes):
  - Cross-attention is active from the first optimization step, allowing gradients to reach
    cond_tokens/backbone immediately. This reduces the optimization bias toward an unconditional
    solution, but condition usage should still be verified with shuffled-condition diagnostics.
  - The output layer uses a small init (std=1e-3) instead of strict zero, so gradients reach the
    condition / backbone from the very first step (no gradient delay), while staying stable.
  - A learnable TYPE embedding distinguishes current / future / waypoint tokens. It is initialized
    to the same magnitude as the positional embedding so it only signals modality, not content.
  - The loss is computed PER MODALITY (each normalized by its own token/channel count) and combined
    with per-modality weights, so the many point-map tokens do not drown the few EEF waypoint tokens.
  - Per-token confidence weights (w_cur / w_fut / w_eef) and a per-sample future-validity mask are
    supported. Invalid future tokens are masked out of self-attention (not just the loss).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _sinusoidal_time_embed(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _GeoDiTBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        h = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, h), nn.GELU(), nn.Linear(h, hidden))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor] = None,
        cond_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xn = self.norm1(x)
        x = x + self.self_attn(
            xn, xn, xn,
            key_padding_mask=target_padding_mask,
            need_weights=False,
        )[0]
        xn = self.norm_cross(x)
        x = x + self.cross_attn(
            xn, cond, cond,
            key_padding_mask=cond_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.mlp(self.norm2(x))
        return x


class GeoFlowDiT(nn.Module):
    def __init__(
        self,
        seg_lens: Tuple[int, int, int],   # (num_current, num_future, num_waypoints); future may be 0
        cond_dim: int,
        in_channels: int = 3,
        hidden_size: int = 512,
        depth: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        if hidden_size < 2:
            raise ValueError("hidden_size must be at least 2")
        if any(n < 0 for n in seg_lens):
            raise ValueError(f"seg_lens must be non-negative, got {seg_lens}")
        if sum(seg_lens) <= 0:
            raise ValueError("Total sequence length must be positive")

        self.n_cur, self.n_fut, self.n_eef = seg_lens
        seq_len = self.n_cur + self.n_fut + self.n_eef
        self.seq_len = seq_len
        self.in_channels = in_channels

        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_dim = hidden_size
        self.t_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.cond_proj = nn.Linear(cond_dim, hidden_size)

        scale = hidden_size ** -0.5
        self.pos_embed = nn.Parameter(scale * torch.randn(seq_len, hidden_size))
        # modality type embedding: 0=current, 1=future, 2=waypoint.
        # Initialized to the same magnitude as pos_embed so it only signals modality, not content.
        self.type_embed = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.type_embed.weight, mean=0.0, std=scale)
        type_ids = [0] * self.n_cur + [1] * self.n_fut + [2] * self.n_eef
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long), persistent=False)

        self.blocks = nn.ModuleList([_GeoDiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, in_channels)
        # Small (not strictly zero) output init: stable start + gradients reach the condition step-1.
        nn.init.normal_(self.proj_out.weight, std=1e-3)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor] = None,
        cond_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, L, C], got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.shape[1]}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[-1]}")
        if cond_tokens.ndim != 3:
            raise ValueError(f"cond_tokens must be [B, K, D], got {tuple(cond_tokens.shape)}")
        if cond_tokens.shape[0] != x.shape[0]:
            raise ValueError("x and cond_tokens batch sizes do not match")

        cond = self.cond_proj(cond_tokens)                                   # [N, K, H]
        h = self.x_embedder(x)                                               # [N, L, H]
        pos = self.pos_embed.to(dtype=h.dtype)
        # ZeRO-3 only gathers *parameters* on forward, not buffers, so self.type_ids
        # may stay on CPU while type_embed.weight is on GPU -> embedding lookup crashes.
        # Move type_ids to the input device explicitly.
        types = self.type_embed(self.type_ids.to(x.device)).to(dtype=h.dtype)
        h = h + pos.unsqueeze(0) + types.unsqueeze(0)
        t_emb = self.t_mlp(_sinusoidal_time_embed(t, self.t_dim).to(h.dtype))
        h = h + t_emb.unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, cond, target_padding_mask=target_padding_mask, cond_padding_mask=cond_padding_mask)
        return self.proj_out(self.norm_final(h))


def _weighted_mean(
    se: torch.Tensor,
    w: Optional[torch.Tensor],
    sample_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """se: [B, n, C]; w: [B, n] per-token weight or None; sample_mask: [B] in {0,1} or None."""
    if se.numel() == 0:
        return se.new_zeros(())
    if w is None:
        per_sample = se.mean(dim=(1, 2))                       # [B]
    else:
        wn = w.to(dtype=se.dtype).unsqueeze(-1)                # [B, n, 1]
        numerator = (se * wn).sum(dim=(1, 2))
        # Divide by (token weight sum) * (channel count) so a uniform weight matches the unweighted mean.
        denominator = (wn.sum(dim=(1, 2)) * se.shape[-1]).clamp_min(1e-6)
        per_sample = numerator / denominator                  # [B]
    if sample_mask is not None:
        m = sample_mask.to(dtype=per_sample.dtype)
        return (per_sample * m).sum() / m.sum().clamp_min(1e-6)
    return per_sample.mean()


class GeoFlowHead(nn.Module):
    """Flow-matching wrapper with per-modality balanced loss + condition-sensitivity diagnostic."""

    def __init__(
        self,
        seg_lens: Tuple[int, int, int],
        cond_dim: int,
        in_channels: int = 3,
        hidden_size: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        lambda_cur: float = 0.5,
        lambda_fut: float = 1.0,
        lambda_eef: float = 3.0,
    ) -> None:
        super().__init__()
        self.net = GeoFlowDiT(seg_lens, cond_dim, in_channels, hidden_size, depth, num_heads)
        self.n_cur, self.n_fut, self.n_eef = seg_lens
        self.lambda_cur, self.lambda_fut, self.lambda_eef = lambda_cur, lambda_fut, lambda_eef

    def _slices(self):
        c0, f0 = 0, self.n_cur
        e0 = self.n_cur + self.n_fut
        return slice(c0, f0), slice(f0, e0), slice(e0, e0 + self.n_eef)

    def _build_target_padding_mask(
        self,
        batch_size: int,
        future_valid: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Mask invalid future tokens out of self-attention (True == masked for MultiheadAttention)."""
        if self.n_fut == 0 or future_valid is None:
            return None
        mask = torch.zeros(
            batch_size, self.n_cur + self.n_fut + self.n_eef,
            dtype=torch.bool, device=device,
        )
        fut_start = self.n_cur
        fut_end = self.n_cur + self.n_fut
        mask[:, fut_start:fut_end] = ~future_valid.bool().unsqueeze(1)
        return mask

    def loss(
        self,
        target: torch.Tensor,           # [B, L, 3]
        cond_tokens: torch.Tensor,      # [B, K, cond_dim]
        w_cur: Optional[torch.Tensor] = None,
        w_fut: Optional[torch.Tensor] = None,
        w_eef: Optional[torch.Tensor] = None,
        future_valid: Optional[torch.Tensor] = None,
        cond_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = target.shape[0]
        noise = torch.randn_like(target)
        # Generate t in float32 so bf16 targets don't lose time-step precision.
        t = torch.rand(B, device=target.device, dtype=torch.float32)
        tb = t.to(dtype=target.dtype)[:, None, None]
        noisy = (1 - tb) * noise + tb * target
        velocity = target - noise

        target_padding_mask = self._build_target_padding_mask(B, future_valid, target.device)
        pred = self.net(
            noisy, t, cond_tokens,
            target_padding_mask=target_padding_mask,
            cond_padding_mask=cond_padding_mask,
        )
        # Compute the loss in float32 for stable 3D-coord regression at small errors.
        se = (pred.float() - velocity.float()).square()            # [B, L, 3]

        cur_s, fut_s, eef_s = self._slices()
        loss_cur = _weighted_mean(se[:, cur_s], w_cur, None)
        loss_eef = _weighted_mean(se[:, eef_s], w_eef, None)
        total = self.lambda_cur * loss_cur + self.lambda_eef * loss_eef
        # Return detached tensors (not floats) to avoid a CPU-GPU sync every step.
        logs: Dict[str, torch.Tensor] = {"geo_cur": loss_cur.detach(), "geo_eef": loss_eef.detach()}
        if self.n_fut > 0:
            loss_fut = _weighted_mean(se[:, fut_s], w_fut, future_valid)
            total = total + self.lambda_fut * loss_fut
            logs["geo_fut"] = loss_fut.detach()
        return total, logs

    def _slice_named(self, seq: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Split a [B, L, 3] sequence into its modality segments."""
        cur_s, fut_s, eef_s = self._slices()
        return {
            "pts_cur": seq[:, cur_s],
            "pts_fut": seq[:, fut_s] if self.n_fut > 0 else None,
            "eef": seq[:, eef_s] if self.n_eef > 0 else None,
        }

    @torch.no_grad()
    def sample(
        self,
        cond_tokens: torch.Tensor,          # [B, K, cond_dim]
        num_steps: int = 10,
        future_valid: Optional[torch.Tensor] = None,
        cond_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reverse flow-matching sampling: reconstruct the 3D target sequence from `cond_tokens`.

        Mirrors the action head's Euler integration: start from noise at t=0 and integrate the
        predicted velocity to t=1. Returns the reconstructed sequence ``[B, L, 3]`` =
        ``[current point map (P) | future (n_fut) | EEF waypoints (W)]`` (use :meth:`_slice_named`
        to split). Inference-only; no VGGT teacher / target needed.

        NOTE: the reconstruction lives in whatever frame the TRAINING target used. With
        ``geoplan.geo_normalize_pts=True`` that is the per-sample normalized frame (centroid 0, unit
        RMS radius), so the point map is shape-faithful but NOT metric. Without normalization it is
        the raw VGGT/camera frame.
        """
        B = cond_tokens.shape[0]
        device = cond_tokens.device
        dtype = cond_tokens.dtype
        x = torch.randn(B, self.net.seq_len, self.net.in_channels, device=device, dtype=dtype)
        target_padding_mask = self._build_target_padding_mask(B, future_valid, device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i / float(num_steps), device=device, dtype=torch.float32)
            v = self.net(
                x, t, cond_tokens,
                target_padding_mask=target_padding_mask,
                cond_padding_mask=cond_padding_mask,
            )
            x = x + dt * v.to(x.dtype)
        return x

    @torch.no_grad()
    def condition_sensitivity(
        self,
        target: torch.Tensor,
        cond_tokens: torch.Tensor,
        future_valid: Optional[torch.Tensor] = None,
        cond_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Per-modality ΔL = L(shuffled cond) - L(correct cond) with SHARED noise/t.

        >0 means the head uses the condition for that modality; ≈0 means it ignores geo_cond
        (a red flag). Uses a deterministic cyclic shift and mid-noise t∈[0.1, 0.5] to reduce
        target leakage. Cheap diagnostic for training logs.
        """
        B = target.shape[0]
        if B < 2:
            out = {"cond_delta_cur": 0.0, "cond_delta_eef": 0.0}
            if self.n_fut > 0:
                out["cond_delta_fut"] = 0.0
            return out

        noise = torch.randn_like(target)
        # Mid-noise path positions reduce target leakage from high-t (near-clean) samples.
        t = torch.empty(B, device=target.device, dtype=torch.float32).uniform_(0.1, 0.5)
        tb = t.to(dtype=target.dtype)[:, None, None]
        noisy = (1 - tb) * noise + tb * target
        velocity = target - noise

        # Deterministic shift: with B>=2 every sample gets a different sample's condition.
        perm = torch.roll(torch.arange(B, device=target.device), shifts=1)

        target_padding_mask = self._build_target_padding_mask(B, future_valid, target.device)
        cpm_perm = cond_padding_mask[perm] if cond_padding_mask is not None else None

        pred_correct = self.net(
            noisy, t, cond_tokens,
            target_padding_mask=target_padding_mask, cond_padding_mask=cond_padding_mask,
        )
        pred_shuffled = self.net(
            noisy, t, cond_tokens[perm],
            target_padding_mask=target_padding_mask, cond_padding_mask=cpm_perm,
        )

        se_correct = (pred_correct.float() - velocity.float()).square()
        se_shuffled = (pred_shuffled.float() - velocity.float()).square()

        cur_s, fut_s, eef_s = self._slices()
        delta_cur = se_shuffled[:, cur_s].mean() - se_correct[:, cur_s].mean()
        delta_eef = se_shuffled[:, eef_s].mean() - se_correct[:, eef_s].mean()
        result = {"cond_delta_cur": float(delta_cur), "cond_delta_eef": float(delta_eef)}

        if self.n_fut > 0:
            correct_fut = se_correct[:, fut_s].mean(dim=(1, 2))
            shuffled_fut = se_shuffled[:, fut_s].mean(dim=(1, 2))
            if future_valid is not None:
                m = future_valid.to(correct_fut.dtype)
                denom = m.sum().clamp_min(1.0)
                delta_fut = ((shuffled_fut - correct_fut) * m).sum() / denom
            else:
                delta_fut = shuffled_fut.mean() - correct_fut.mean()
            result["cond_delta_fut"] = float(delta_fut)

        return result
