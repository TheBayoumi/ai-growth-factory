from __future__ import annotations

from typing import Any


_INSTALLED = False
_MAX_PRODUCTION_TEMPO = 1.35


def install_production_pacing() -> None:
    """Allow pitch-preserving tempo correction to reach short-form pace."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import audio_qc, voice_pipeline

    original_factor = audio_qc.tempo_correction_factor
    audio_qc.MAX_TEMPO_FACTOR = _MAX_PRODUCTION_TEMPO

    def production_factor(
        *,
        estimated_wpm: float,
        target_wpm: int,
        tolerance: int,
        minimum_factor: float = audio_qc.MIN_TEMPO_FACTOR,
        maximum_factor: float = _MAX_PRODUCTION_TEMPO,
        **_ignored: Any,
    ) -> float | None:
        return original_factor(
            estimated_wpm=estimated_wpm,
            target_wpm=target_wpm,
            tolerance=tolerance,
            minimum_factor=minimum_factor,
            maximum_factor=maximum_factor,
        )

    voice_pipeline.tempo_correction_factor = production_factor
    _INSTALLED = True
