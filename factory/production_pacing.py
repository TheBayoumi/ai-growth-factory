from __future__ import annotations

from typing import Any


_INSTALLED = False
_MAX_PRODUCTION_TEMPO = 1.45


def install_production_pacing() -> None:
    """Allow bounded pitch-preserving correction to reach short-form pace.

    Qwen3-TTS can produce otherwise clean narration well below the requested pace.
    Production permits a larger, still bounded correction only when the projected pace
    lands inside the existing WPM gate. The voice pipeline applies this to segment assets
    before review so Omni hears the same paced audio that will be composed.
    """
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
