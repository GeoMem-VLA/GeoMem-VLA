"""Episode memory banks for GeoMem-VLA (dimension-agnostic).

One bank instance is created per token stream (two observation streams + one geometric stream).

  - EpisodeMemoryBank : bounded per-episode history with causal cross-attention retrieval, gated
                        fusion, and similarity-based (ToMe) / FIFO consolidation.
  - HistoryGate       : near-identity learnable gate used inside the banks.

Event-aware variant (opt-in; the banks above are unchanged so existing checkpoints reproduce):

  - EventMemoryBank : stores key events instead of every frame, via a scene-change /
                      event-novelty write gate, with importance-aware retrieval and eviction.
"""

from .event_mem_bank import (
    BiasedHistoryAttentionBlock,
    EventMemoryBank,
    EventScorer,
)
from .mem_bank import (
    EpisodeMemoryBank,
    HistoryAttentionBlock,
    HistoryGate,
    TimeStampEncoder,
)

__all__ = [
    "EpisodeMemoryBank",
    "HistoryGate",
    "HistoryAttentionBlock",
    "TimeStampEncoder",
    "EventMemoryBank",
    "EventScorer",
    "BiasedHistoryAttentionBlock",
]
