from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import NarrationSegment, VoiceContract


_INSTALLED = False
_MAX_PRODUCTION_TEMPO = 1.45
_MAX_CLOSING_TEMPO = 1.15


def production_segment_tempo_factor(
    *,
    estimated_wpm: float,
    target_wpm: int,
    tolerance: int,
    is_closing: bool,
) -> float | None:
    """Return a natural segment correction while preserving whole-track pacing.

    Explanatory segments can use the existing production maximum because Qwen3-TTS often
    produces them consistently but slowly. The final segment is different: forcing a short
    closing clause from roughly 112 WPM to 155 WPM required about 1.38x acceleration and was
    repeatedly rejected by Qwen Omni as strained and unnatural. The close is therefore capped
    at 1.15x. The assembled track remains responsible for the unchanged 145-165 WPM gate.
    """
    from . import audio_qc

    if is_closing:
        lower_wpm = target_wpm - tolerance
        upper_wpm = target_wpm + tolerance
        if lower_wpm <= estimated_wpm <= upper_wpm:
            return None
        requested = target_wpm / max(estimated_wpm, 0.001)
        factor = min(max(requested, audio_qc.MIN_TEMPO_FACTOR), _MAX_CLOSING_TEMPO)
        if abs(factor - 1.0) < 1e-6:
            return None
        return factor

    return audio_qc.tempo_correction_factor(
        estimated_wpm=estimated_wpm,
        target_wpm=target_wpm,
        tolerance=tolerance,
        minimum_factor=audio_qc.MIN_TEMPO_FACTOR,
        maximum_factor=_MAX_PRODUCTION_TEMPO,
    )


def install_production_pacing() -> None:
    """Pace segment assets conservatively and enforce speed on the assembled track.

    Qwen Omni reviews the exact segment assets retained for composition. Explanatory segments
    may receive a larger pitch-preserving correction, but the final segment is capped at 1.15x
    so a naturally spoken close is not compressed into synthetic speech. The final assembled
    narration still passes the existing deterministic WPM gate, and track-level correction
    remains available when pace is the only remaining failure.
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

    def production_pace_correct_segment_assets(
        segments: list[NarrationSegment],
        *,
        workdir: Path,
        pipeline_attempt: int,
        contract: VoiceContract,
        settings: Any,
    ) -> tuple[list[NarrationSegment], list[dict[str, object]]]:
        corrected: list[NarrationSegment] = []
        events: list[dict[str, object]] = []
        closing_id = max((segment.segment_id for segment in segments), default=-1)

        for segment in segments:
            duration = voice_pipeline.wav_duration(segment.audio_path)
            before_wpm = len(segment.text.split()) / max(duration, 0.001) * 60.0
            is_closing = segment.segment_id == closing_id
            factor = production_segment_tempo_factor(
                estimated_wpm=before_wpm,
                target_wpm=contract.target_wpm,
                tolerance=settings.audio_wpm_tolerance,
                is_closing=is_closing,
            )
            if factor is None:
                corrected.append(segment)
                continue

            paced = workdir / "segments" / (
                f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-paced-"
                f"{pipeline_attempt}.wav"
            )
            voice_pipeline.correct_audio_tempo(segment.audio_path, paced, factor=factor)
            normalized = workdir / "segments" / (
                f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-"
                f"paced-normalized-{pipeline_attempt}.wav"
            )
            voice_pipeline.normalize_audio(
                paced,
                normalized,
                target_lufs=settings.audio_target_lufs,
                peak_dbfs=settings.audio_peak_limit_dbfs,
            )
            after_duration = voice_pipeline.wav_duration(normalized)
            after_wpm = len(segment.text.split()) / max(after_duration, 0.001) * 60.0
            corrected.append(replace(segment, audio_path=normalized))
            events.append(
                {
                    "attempt": pipeline_attempt,
                    "type": "deterministic_segment_tempo_correction",
                    "segment_id": segment.segment_id,
                    "factor": round(factor, 6),
                    "before_wpm": round(before_wpm, 3),
                    "after_wpm": round(after_wpm, 3),
                    "closing_segment": is_closing,
                    "maximum_factor": (
                        _MAX_CLOSING_TEMPO if is_closing else _MAX_PRODUCTION_TEMPO
                    ),
                    "audio_path": str(normalized),
                }
            )
        return corrected, events

    voice_pipeline.tempo_correction_factor = production_factor
    voice_pipeline._pace_correct_segment_assets = production_pace_correct_segment_assets
    _INSTALLED = True
