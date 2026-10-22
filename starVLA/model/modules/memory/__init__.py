"""Perception-Cognition-Geo memory banks (ported from MemoryVLA, dimension-agnostic).

  - CogMemBank : per-episode working memory with cross-frame retrieval + ToMe/FIFO consolidation.
  - PerMemBank : same mechanism, used for perceptual tokens.
  - GateFusion : near-identity learnable gate used inside the banks.

Event-guided variants (opt-in; the banks above are unchanged so existing checkpoints reproduce):

  - EventGuidedMemBank    : stores key events instead of every frame, via a scene-change /
                            event-novelty write gate, with importance-aware retrieval and eviction.
  - EventGuidedPerMemBank : perceptual counterpart.
"""

from .event_mem_bank import (
    BiasedCrossTransformerBlock,
    EventGuidedMemBank,
    EventGuidedPerMemBank,
    EventScorer,
)
from .mem_bank import (
    CogMemBank,
    CrossTransformerBlock,
    GateFusion,
    PerMemBank,
    TimestepEmbedder,
)

__all__ = [
    "CogMemBank",
    "PerMemBank",
    "GateFusion",
    "CrossTransformerBlock",
    "TimestepEmbedder",
    "EventGuidedMemBank",
    "EventGuidedPerMemBank",
    "EventScorer",
    "BiasedCrossTransformerBlock",
]
