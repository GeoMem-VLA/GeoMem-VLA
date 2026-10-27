"""GeoPlan-Forcing modules for starVLA (3D geometry + planning auxiliary, train-only).

Designed to attach to a Qwen3-VL backbone via the QwenGeoGR00T framework:
  - VGGTGeoTeacher       : frozen VGGT teacher producing 3D point-map targets (online).
  - GeoFlowHead          : self-contained flow-matching head that reconstructs the targets from
                           the geo condition tokens (no external diffusion dependency).
  - GatedGeoFusion       : gated fusion of geo guidance into the visual observation tokens.
  - GeoPlanQueryResampler: optional perceiver bridge (read-only) if not using in-sequence tokens.
"""

from .fusion import GatedGeoFusion
from .resampler import GeoPlanQueryResampler, GeoInseqBridge
from .geo_model import GeoFlowHead, GeoFlowDiT
from .vggt_teacher import VGGTGeoTeacher

__all__ = [
    "GatedGeoFusion",
    "GeoPlanQueryResampler",
    "GeoInseqBridge",
    "GeoFlowHead",
    "GeoFlowDiT",
    "VGGTGeoTeacher",
]
