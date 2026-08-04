from __future__ import annotations

from pathlib import Path
from typing import Any

from .video_profile import VideoProfile


_INSTALLED = False
_EPSILON = 1e-9


def install_production_voice_bounds_v28() -> None:
    """Make the v28 tempo ceiling authoritative at every voice-pipeline call site.

    The legacy production pacing adapter captured its former 1.45 maximum in a function
    default before v28 lowered the module-level value to 1.15. Track-level correction could
    therefore still return a factor above 1.15, which the stricter audio renderer correctly
    rejected. This adapter replaces both references with one profile-bound implementation.
    Corrections that cannot reach the accepted WPM range within the ceiling return ``None``;
    the existing voice retry loop must regenerate the take instead of over-speeding it.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import audio_qc, voice_pipeline

    profile = VideoProfile.from_env()
    minimum = float(audio_qc.MIN_TEMPO_FACTOR)
    maximum = float(profile.maximum_tempo_factor)
    original_correct = audio_qc.correct_audio_tempo

    def strict_factor(
        *,
        estimated_wpm: float,
        target_wpm: int,
        tolerance: int,
        minimum_factor: float = minimum,
        maximum_factor: float = maximum,
        **_ignored: Any,
    ) -> float | None:
        bounded_minimum = max(minimum, float(minimum_factor))
        bounded_maximum = min(maximum, float(maximum_factor))
        factor = audio_qc.tempo_correction_factor(
            estimated_wpm=estimated_wpm,
            target_wpm=target_wpm,
            tolerance=tolerance,
            minimum_factor=bounded_minimum,
            maximum_factor=bounded_maximum,
        )
        if factor is None:
            return None
        if factor < minimum - _EPSILON or factor > maximum + _EPSILON:
            raise audio_qc.AudioQCError(
                f"v28 tempo factor {factor:.12f} escaped {minimum:.2f}-{maximum:.2f}"
            )
        return min(max(float(factor), minimum), maximum)

    def strict_correct_audio_tempo(
        input_path: Path,
        output_path: Path,
        *,
        factor: float,
    ) -> Path:
        numeric = float(factor)
        if numeric < minimum - _EPSILON or numeric > maximum + _EPSILON:
            raise audio_qc.AudioQCError(
                f"Tempo factor must be between {minimum:.2f} and {maximum:.2f}"
            )
        exact = min(max(numeric, minimum), maximum)
        return original_correct(input_path, output_path, factor=exact)

    audio_qc.MAX_TEMPO_FACTOR = maximum
    voice_pipeline.tempo_correction_factor = strict_factor
    voice_pipeline.correct_audio_tempo = strict_correct_audio_tempo
    _INSTALLED = True
