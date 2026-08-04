from __future__ import annotations

import os
from dataclasses import replace


_INSTALLED = False
_DEFAULT_REVIEWER_MODEL = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"


def _pause_ms() -> int:
    raw = os.getenv("V28_INTER_SEGMENT_PAUSE_MS", "140")
    value = int(raw)
    if not 80 <= value <= 300:
        raise ValueError("V28_INTER_SEGMENT_PAUSE_MS must be between 80 and 300")
    return value


def install_production_voice_runtime_v28() -> None:
    """Select the stable reviewer and a natural pause budget before Settings is resolved.

    The Modal image previously forced the 3B reviewer even though the reviewer implementation
    and memory plan were designed for the quantized 7B checkpoint. The shorter sentence-aligned
    v28 segments retain their own edge silence, so 140 ms of inserted silence produces a natural
    join without lowering the assembled track below the accepted WPM range.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from .config import Settings

    reviewer_model = os.getenv("V28_REVIEWER_MODEL", _DEFAULT_REVIEWER_MODEL).strip()
    if not reviewer_model:
        raise ValueError("V28_REVIEWER_MODEL must not be empty")
    os.environ["QWEN_OMNI_REVIEW_MODEL"] = reviewer_model

    current_from_env = Settings.from_env.__func__

    def v28_voice_runtime_from_env(cls: type[Settings]) -> Settings:
        settings = current_from_env(cls)
        return replace(
            settings,
            qwen_omni_model=reviewer_model,
            audio_segment_pause_ms=_pause_ms(),
        )

    Settings.from_env = classmethod(v28_voice_runtime_from_env)
    _INSTALLED = True
