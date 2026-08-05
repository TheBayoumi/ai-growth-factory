from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import NarrationSegment, VoiceContract
from .production_voice_technical_identifier_v42 import (
    speech_equivalent_word_count_v42,
    speech_equivalent_wpm_v42,
    written_word_count_v42,
)
from .video_profile import VideoProfile


_INSTALLED = False


def technical_editorial_factor_v43(
    text: str,
    duration_seconds: float,
    *,
    profile: VideoProfile,
    minimum_factor: float = 0.85,
) -> tuple[float, float] | None:
    """Return the bounded editorial factor using spoken-equivalent technical units."""
    from .production_voice_editorial_pacing_v28 import editorial_segment_factor_v28

    return editorial_segment_factor_v28(
        speech_equivalent_wpm_v42(text, duration_seconds),
        profile=profile,
        minimum_factor=minimum_factor,
    )


def install_production_voice_technical_pacing_v43() -> None:
    """Make final sentence pacing use the same technical-token metric as calibration."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import audio_qc, production_voice_calibration_v28 as calibration, voice_pipeline
    from . import production_voice_editorial_pacing_v28 as editorial

    profile = VideoProfile.from_env()

    def pace_technical_editorial_segments(
        segments: list[NarrationSegment],
        *,
        workdir: Path,
        pipeline_attempt: int,
        contract: VoiceContract,
        settings: Any,
    ) -> tuple[list[NarrationSegment], list[dict[str, object]]]:
        corrected: list[NarrationSegment] = []
        events: list[dict[str, object]] = []
        for segment in segments:
            compacted = workdir / "segments" / (
                f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-"
                f"technical-compacted-{pipeline_attempt}.wav"
            )
            compaction = calibration.compact_excess_silence_v28(
                segment.audio_path,
                compacted,
            )
            before_duration = audio_qc.wav_duration(compacted)
            written_count = written_word_count_v42(segment.text)
            equivalent_count = speech_equivalent_word_count_v42(segment.text)
            written_before_wpm = written_count / max(before_duration, 0.001) * 60.0
            before_wpm = equivalent_count / max(before_duration, 0.001) * 60.0
            result = editorial.editorial_segment_factor_v28(
                before_wpm,
                profile=profile,
                minimum_factor=float(audio_qc.MIN_TEMPO_FACTOR),
            )
            if result is None:
                raise voice_pipeline.VoiceGenerationError(
                    f"Segment {segment.segment_id} cannot enter the natural editorial pace "
                    f"range within {profile.maximum_tempo_factor:.2f}x; observed "
                    f"{before_wpm:.2f} spoken-equivalent WPM "
                    f"({written_before_wpm:.2f} written-word WPM)"
                )
            factor, projected = result
            final_path = compacted
            if abs(factor - 1.0) > 0.002:
                paced = workdir / "segments" / (
                    f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-"
                    f"technical-paced-{pipeline_attempt}.wav"
                )
                voice_pipeline.correct_audio_tempo(compacted, paced, factor=factor)
                normalized = workdir / "segments" / (
                    f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-"
                    f"technical-normalized-{pipeline_attempt}.wav"
                )
                voice_pipeline.normalize_audio(
                    paced,
                    normalized,
                    target_lufs=settings.audio_target_lufs,
                    peak_dbfs=settings.audio_peak_limit_dbfs,
                )
                final_path = normalized
            after_duration = audio_qc.wav_duration(final_path)
            after_wpm = equivalent_count / max(after_duration, 0.001) * 60.0
            written_after_wpm = written_count / max(after_duration, 0.001) * 60.0
            if not (
                editorial._SEGMENT_MINIMUM_WPM - 0.5
                <= after_wpm
                <= editorial._SEGMENT_MAXIMUM_WPM + 0.5
            ):
                raise voice_pipeline.VoiceGenerationError(
                    f"Segment {segment.segment_id} escaped the natural editorial pace range: "
                    f"{after_wpm:.2f} spoken-equivalent WPM"
                )
            corrected.append(replace(segment, audio_path=final_path))
            events.append(
                {
                    "attempt": pipeline_attempt,
                    "type": "deterministic_technical_editorial_pacing_v43",
                    "segment_id": segment.segment_id,
                    "factor": round(factor, 6),
                    "written_word_count": written_count,
                    "speech_equivalent_word_count": round(equivalent_count, 3),
                    "written_before_wpm": round(written_before_wpm, 3),
                    "before_wpm": round(before_wpm, 3),
                    "projected_wpm": round(projected, 3),
                    "written_after_wpm": round(written_after_wpm, 3),
                    "after_wpm": round(after_wpm, 3),
                    "removed_silence_seconds": compaction["removed_seconds"],
                    "audio_path": str(final_path),
                }
            )
        return corrected, events

    voice_pipeline._pace_correct_segment_assets = pace_technical_editorial_segments
    _INSTALLED = True
