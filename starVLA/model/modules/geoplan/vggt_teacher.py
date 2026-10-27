"""Frozen VGGT teacher that produces 3D geometry targets online during training.

The teacher is intentionally kept *outside* the registered ``nn.Module`` tree of the policy model
(it is held inside a plain python list attribute) so that:
  * FSDP / ``.to()`` / ``state_dict()`` never touch it (it is frozen and never saved);
  * ``train.py``'s "all params must be fp32" assertion is unaffected.

It only enables the VGGT point head (camera / depth / track heads are disabled) to save memory.
Given the current frame and a paired future frame it returns per-patch 3D point maps in a shared
coordinate frame, already normalized to ~unit scale, together with confidence weights.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _save_ply(path: str, xyz, rgb, max_points: int = 60000) -> None:
    """Write a colored point cloud to an ASCII .ply file.

    xyz: [N, 3] float array; rgb: [N, 3] array in [0, 1].
    """
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    # Drop non-finite points.
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
    if xyz.shape[0] > max_points:
        sel = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb_u8):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def _maybe_add_vggt_to_path(vggt_repo_path: Optional[str]) -> None:
    """Make the ``vggt`` package importable.

    Priority: explicit ``vggt_repo_path`` arg > existing ``sys.path`` (the training script already
    exports ``Spatial-Forcing/openvla-SF`` onto ``PYTHONPATH``).
    """
    if vggt_repo_path is not None and vggt_repo_path not in sys.path:
        if os.path.isdir(vggt_repo_path):
            sys.path.insert(0, vggt_repo_path)


class VGGTGeoTeacher:
    """Thin wrapper around a frozen VGGT model used as a geometry target generator."""

    def __init__(
        self,
        ckpt_path: str,
        vggt_repo_path: Optional[str] = None,
        img_size: int = 518,
        grid_size: int = 16,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        conf_weight: bool = True,
    ) -> None:
        _maybe_add_vggt_to_path(vggt_repo_path)
        try:
            from vggt.models.vggt import VGGT
        except Exception as e:  # pragma: no cover - environment specific
            raise ImportError(
                "Could not import `vggt`. Make sure the VGGT repo (e.g. "
                "`Spatial-Forcing/openvla-SF`) is on PYTHONPATH or pass `geo_vggt_repo_path`."
            ) from e

        self.img_size = img_size
        # VGGT requires the input resolution to be divisible by the patch size (14).
        assert img_size % 14 == 0, f"geo_vggt_img_size must be divisible by 14, got {img_size}"
        self.grid_size = grid_size
        # NOTE: this is the *autocast* dtype used only for the forward pass. The VGGT weights are
        # kept in fp32 because its prediction heads run under `autocast(enabled=False)` and require
        # fp32 params (matching Spatial-Forcing's usage). Casting the model to bf16 breaks the heads.
        self.autocast_dtype = dtype
        self.conf_weight = conf_weight

        # Only the point head is needed -> disable the rest to save compute / memory.
        # Note: this version of VGGT does not have a `feature_only` argument.
        model = VGGT(
            enable_camera=False,
            enable_point=True,
            enable_depth=False,
            enable_track=False,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state and "aggregator.patch_embed.cls_token" not in state:
            state = state["model"]
        missing, _unexpected = model.load_state_dict(state, strict=False)
        # The point head should be present; warn loudly if it is missing.
        if any(k.startswith("point_head") for k in missing):
            print(f"[VGGTGeoTeacher] WARNING: point_head weights missing from checkpoint: "
                  f"{[k for k in missing if k.startswith('point_head')][:4]} ...")
        model.eval()
        model.requires_grad_(False)
        if device is not None:
            model = model.to(device=device)  # keep fp32 weights
        self.model = model
        self._device = device

    # ------------------------------------------------------------------ utils
    def to(self, device: torch.device) -> "VGGTGeoTeacher":
        if self._device != device:
            self.model = self.model.to(device=device)  # keep fp32 weights
            self._device = device
        return self

    @staticmethod
    def _unnormalize(x: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
        """Invert a (mean, std) normalization to recover images in [0, 1]."""
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * s + m).clamp(0.0, 1.0)

    def _prep(self, siglip_pix: torch.Tensor, mean, std) -> torch.Tensor:
        """[B,3,h,w] normalized siglip tensor -> [B,3,img_size,img_size] in [0,1]."""
        img = self._unnormalize(siglip_pix.float(), mean, std)
        img = F.interpolate(img, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False)
        return img.clamp(0.0, 1.0)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """[BS, H, W, C] -> [BS, grid*grid, C] via adaptive average pooling."""
        bs, _h, _w, c = x.shape
        x = x.permute(0, 3, 1, 2)  # [BS, C, H, W]
        x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))  # [BS, C, P, P]
        x = x.permute(0, 2, 3, 1).reshape(bs, self.grid_size * self.grid_size, c)
        return x

    # --------------------------------------------------------------- targets
    @torch.no_grad()
    def compute_targets(
        self,
        siglip_cur: torch.Tensor,
        siglip_fut: torch.Tensor,
        mean,
        std,
        dump_prefix: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run VGGT jointly on (current, future) frames and return normalized point-map targets.

        Returns a dict with:
          ``pts_cur``  [B, P, 3]  normalized current-frame world points (per-patch)
          ``pts_fut``  [B, P, 3]  normalized future-frame world points (shared frame)
          ``w_cur``    [B, P]     confidence weights in [0, 1]
          ``w_fut``    [B, P]     confidence weights in [0, 1]
        """
        device = next(self.model.parameters()).device
        cur = self._prep(siglip_cur.to(device), mean, std)
        fut = self._prep(siglip_fut.to(device), mean, std)
        seq = torch.stack([cur, fut], dim=1)  # [B, 2, 3, H, W]

        # Run aggregator in bf16 to save memory; VGGT's forward now casts the
        # aggregated tokens back to fp32 before entering the autocast(enabled=False)
        # prediction-head zone, so the fp32-weight LayerNorm inside the heads is safe.
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            out = self.model(seq)
        world = out["world_points"].float()        # [B, 2, H, W, 3]
        conf = out["world_points_conf"].float()    # [B, 2, H, W]

        # Optionally dump the full-resolution colored point clouds for inspection (sample 0 only).
        if dump_prefix is not None:
            try:
                import os as _os
                _os.makedirs(_os.path.dirname(dump_prefix), exist_ok=True)
                cur_rgb = cur[0].permute(1, 2, 0).detach().cpu().numpy()   # [H, W, 3]
                fut_rgb = fut[0].permute(1, 2, 0).detach().cpu().numpy()
                _save_ply(f"{dump_prefix}_current.ply", world[0, 0].detach().cpu().numpy(), cur_rgb)
                _save_ply(f"{dump_prefix}_future.ply", world[0, 1].detach().cpu().numpy(), fut_rgb)
                import numpy as _np
                _np.savez_compressed(
                    f"{dump_prefix}_points.npz",
                    world_current=world[0, 0].detach().cpu().numpy(),
                    world_future=world[0, 1].detach().cpu().numpy(),
                    conf_current=conf[0, 0].detach().cpu().numpy(),
                    conf_future=conf[0, 1].detach().cpu().numpy(),
                    rgb_current=cur_rgb,
                    rgb_future=fut_rgb,
                )
                print(f"[VGGTGeoTeacher] dumped point clouds -> {dump_prefix}_current.ply / _future.ply")
            except Exception as _e:  # pragma: no cover - dumping must never break training
                print(f"[VGGTGeoTeacher] point cloud dump failed: {_e}")

        b = world.shape[0]
        world = world.reshape(b * 2, *world.shape[2:])      # [B*2, H, W, 3]
        conf = conf.reshape(b * 2, *conf.shape[2:]).unsqueeze(-1)  # [B*2, H, W, 1]

        pts = self._downsample(world).reshape(b, 2, -1, 3)  # [B, 2, P, 3]
        w = self._downsample(conf).reshape(b, 2, -1)        # [B, 2, P]

        pts_cur, pts_fut = pts[:, 0], pts[:, 1]
        w_cur, w_fut = w[:, 0], w[:, 1]

        # Normalize geometry to ~unit scale using the *current* frame statistics so that the
        # current geometry and the future geometry (scene flow) live in the same metric space.
        center = pts_cur.mean(dim=1, keepdim=True)                   # [B, 1, 3]
        scale = pts_cur.reshape(b, -1).std(dim=1).clamp_min(1e-4)    # [B]
        scale = scale.view(b, 1, 1)
        pts_cur = (pts_cur - center) / scale
        pts_fut = (pts_fut - center) / scale

        if self.conf_weight:
            # Per-sample min-max normalize confidence into [0, 1].
            def _norm_w(x: torch.Tensor) -> torch.Tensor:
                lo = x.amin(dim=1, keepdim=True)
                hi = x.amax(dim=1, keepdim=True)
                return ((x - lo) / (hi - lo).clamp_min(1e-6))
            w_cur = _norm_w(w_cur)
            w_fut = _norm_w(w_fut)
        else:
            w_cur = torch.ones_like(w_cur)
            w_fut = torch.ones_like(w_fut)

        return {
            "pts_cur": pts_cur.contiguous(),
            "pts_fut": pts_fut.contiguous(),
            "w_cur": w_cur.contiguous(),
            "w_fut": w_fut.contiguous(),
        }
