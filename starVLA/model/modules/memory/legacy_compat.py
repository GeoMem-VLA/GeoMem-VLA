"""Backward compatibility for checkpoints / configs produced before the stream-naming refactor.

Old names -> new names:
  config  : ``per_num_tokens`` -> ``obs1_num_tokens``; ``per_resampler_layers`` -> ``obs1_resampler_layers``;
            ``mem_bank_type: cog`` -> ``similarity``
  weights : ``cog_mem_bank`` -> ``obs2_mem_bank``; ``per_mem_bank`` -> ``obs1_mem_bank``;
            ``per_resampler`` -> ``obs1_resampler``; inside every memory bank
            ``retrieval_blocks`` -> ``history_attn``, ``gate_fusion_blocks`` -> ``history_gate``,
            ``timestep_encoder`` -> ``time_encoder``.
"""

from __future__ import annotations

from typing import Any, MutableMapping

_CONFIG_KEY_MAP = {
    "per_num_tokens": "obs1_num_tokens",
    "per_resampler_layers": "obs1_resampler_layers",
}

_TOP_LEVEL_KEY_MAP = (
    ("cog_mem_bank.", "obs2_mem_bank."),
    ("per_mem_bank.", "obs1_mem_bank."),
    ("per_resampler.", "obs1_resampler."),
)

_BANK_INNER_KEY_MAP = (
    (".retrieval_blocks.", ".history_attn."),
    (".gate_fusion_blocks.", ".history_gate."),
    (".timestep_encoder.", ".time_encoder."),
)


def remap_legacy_geoplan_cfg(gcfg: MutableMapping[str, Any]) -> None:
    """Translate legacy geoplan config keys / values in place (new keys win if both are present)."""
    for old, new in _CONFIG_KEY_MAP.items():
        if old in gcfg:
            gcfg.setdefault(new, gcfg[old])
    if str(gcfg.get("mem_bank_type", "")).strip().lower() == "cog":
        gcfg["mem_bank_type"] = "similarity"


def _remap_key(key: str) -> str:
    for old, new in _TOP_LEVEL_KEY_MAP:
        idx = key.find(old)
        if idx >= 0 and (idx == 0 or key[idx - 1] == "."):
            key = key[:idx] + new + key[idx + len(old):]
            break
    if "_mem_bank." in key:
        for old, new in _BANK_INNER_KEY_MAP:
            key = key.replace(old, new)
    return key


def remap_legacy_state_dict_keys(state_dict: MutableMapping[str, Any]) -> None:
    """Rename legacy keys in place. Intended as a ``load_state_dict`` pre-hook."""
    for key in list(state_dict.keys()):
        new_key = _remap_key(key)
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)


def legacy_state_dict_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    remap_legacy_state_dict_keys(state_dict)
