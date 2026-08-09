from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import FailedSegment, NarrationSegment, VoiceContract
from .video_profile import VideoProfile


_INSTALLED = False
_SEGMENT_MINIMUM_WPM = 132.0
_SEGMENT_MAXIMUM_WPM = 152.0
_SEGMENT_TARGET_OFFSET_WPM = 2.0
_DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def editorial_segment_target_v28(profile: VideoProfile) -> float:
    return min(float(profile.maximum_wpm) + 2.0, float(profile.target_wpm) + _SEGMENT_TARGET_OFFSET_WPM)


def editorial_segment_factor_v28(
    observed_wpm: float,
    *,
    profile: VideoProfile,
    minimum_factor: float = 0.85,
) -> tuple[float, float] | None:
    """Return a bounded factor when a sentence can enter the natural editorial range.

    The complete narration remains constrained to 138-146 WPM. Individual sentences need a
    wider range so technical clauses may breathe while hooks and transitions remain energetic.
    The target is two WPM above the track target to offset only the short inter-segment joins.
    """
    observed = float(observed_wpm)
    if observed <= 0:
        raise ValueError("observed_wpm must be positive")
    target = editorial_segment_target_v28(profile)
    factor = min(
        max(target / observed, float(minimum_factor)),
        float(profile.maximum_tempo_factor),
    )
    projected = observed * factor
    if not _SEGMENT_MINIMUM_WPM <= projected <= _SEGMENT_MAXIMUM_WPM:
        return None
    return factor, projected


def segment_candidate_publishable_v28(
    observed_wpm: float,
    *,
    profile: VideoProfile,
    minimum_tempo_factor: float = 0.85,
) -> bool:
    return (
        editorial_segment_factor_v28(
            observed_wpm,
            profile=profile,
            minimum_factor=minimum_tempo_factor,
        )
        is not None
    )


def ground_editorial_pace_failure_v28(
    failure: FailedSegment,
    *,
    measured_wpm: float,
    profile: VideoProfile,
) -> FailedSegment | None:
    """Let deterministic timing own pace while Qwen still owns audible quality."""
    from . import production_voice_convergence_v28 as convergence

    combined = f"{failure.reason} {failure.tts_instruction}"
    if not convergence._PACE_RE.search(combined):
        return failure
    if not _SEGMENT_MINIMUM_WPM <= measured_wpm <= _SEGMENT_MAXIMUM_WPM:
        return failure
    if convergence._CRITICAL_RE.search(combined):
        return failure
    if not convergence._PAUSE_RE.search(combined):
        return None
    return replace(
        failure,
        reason=(
            f"{failure.reason} Deterministic sentence pace is {measured_wpm:.1f} WPM, inside "
            f"the permitted {_SEGMENT_MINIMUM_WPM:.0f}-{_SEGMENT_MAXIMUM_WPM:.0f} editorial "
            "range; only cadence and pause placement may change."
        ),
        tts_instruction=(
            f"Preserve this sentence near {measured_wpm:.0f} words per minute. Improve only "
            "clause-boundary pauses, cadence, and natural phrasing. Do not request a faster or "
            "slower overall delivery, do not alter the transcript, and keep every word clear."
        ),
    )


def install_production_voice_editorial_pacing_v28() -> None:
    """Apply natural sentence variation while retaining a strict whole-track contract."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import audio_qc
    from . import production_voice_calibration_v28 as calibration
    from . import production_voice_convergence_v28 as convergence
    from . import qwen_omni_reviewer, voice_pipeline

    os.environ["QWEN_TTS_MODEL"] = os.getenv("V28_TTS_MODEL", _DEFAULT_TTS_MODEL).strip()
    profile = VideoProfile.from_env()

    def editorial_reachable(
        observed_wpm: float,
        *,
        profile: VideoProfile,
        minimum_tempo_factor: float = 0.85,
    ) -> bool:
        return segment_candidate_publishable_v28(
            observed_wpm,
            profile=profile,
            minimum_tempo_factor=minimum_tempo_factor,
        )

    calibration.segment_candidate_reachable_v28 = editorial_reachable
    convergence.ground_pace_failure_v28 = ground_editorial_pace_failure_v28

    previous_prompt = qwen_omni_reviewer._segment_prompt

    def editorial_segment_prompt(
        *,
        segment: NarrationSegment,
        contract: VoiceContract,
        metrics: Any,
        attempt: int,
    ) -> str:
        base = previous_prompt(
            segment=segment,
            contract=contract,
            metrics=metrics,
            attempt=attempt,
        )
        rule = (
            "\n\nEditorial timing authority: deterministic QC permits an individual sentence "
            f"between {_SEGMENT_MINIMUM_WPM:.0f} and {_SEGMENT_MAXIMUM_WPM:.0f} WPM while the "
            f"complete narration must remain {profile.minimum_wpm}-{profile.maximum_wpm} WPM. "
            "Natural sentence-to-sentence pace variation is intentional. Do not request a pace "
            "change solely because a sentence differs from the full-track target; report only "
            "audible rushing, dragging, clipped articulation, awkward pauses, or cadence defects."
        )
        marker = "\n\nReturn one JSON object only"
        return base.replace(marker, rule + marker)

    qwen_omni_reviewer._segment_prompt = editorial_segment_prompt

    def pace_editorial_segments(
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
                f"editorial-compacted-{pipeline_attempt}.wav"
            )
            compaction = calibration.compact_excess_silence_v28(segment.audio_path, compacted)
            before_duration = audio_qc.wav_duration(compacted)
            before_wpm = len(segment.text.split()) / max(before_duration, 0.001) * 60.0
            result = editorial_segment_factor_v28(
                before_wpm,
                profile=profile,
                minimum_factor=float(audio_qc.MIN_TEMPO_FACTOR),
            )
            if result is None:
                raise voice_pipeline.VoiceGenerationError(
                    f"Segment {segment.segment_id} cannot enter the natural editorial pace "
                    f"range within {profile.maximum_tempo_factor:.2f}x; observed "
                    f"{before_wpm:.2f} WPM"
                )
            factor, projected = result
            final_path = compacted
            if abs(factor - 1.0) > 0.002:
                paced = workdir / "segments" / (
                    f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-"
                    f"editorial-paced-{pipeline_attempt}.wav"
                )
                voice_pipeline.correct_audio_tempo(compacted, paced, factor=factor)
                normalized = workdir / "segments" / (
                    f"segment-{segment.segment_id:02d}-attempt-{segment.attempt}-"
                    f"editorial-normalized-{pipeline_attempt}.wav"
                )
                voice_pipeline.normalize_audio(
                    paced,
                    normalized,
                    target_lufs=settings.audio_target_lufs,
                    peak_dbfs=settings.audio_peak_limit_dbfs,
                )
                final_path = normalized
            after_duration = audio_qc.wav_duration(final_path)
            after_wpm = len(segment.text.split()) / max(after_duration, 0.001) * 60.0
            if not _SEGMENT_MINIMUM_WPM - 0.5 <= after_wpm <= _SEGMENT_MAXIMUM_WPM + 0.5:
                raise voice_pipeline.VoiceGenerationError(
                    f"Segment {segment.segment_id} escaped the natural editorial pace range: "
                    f"{after_wpm:.2f} WPM"
                )
            corrected.append(replace(segment, audio_path=final_path))
            events.append(
                {
                    "attempt": pipeline_attempt,
                    "type": "deterministic_segment_editorial_pacing_v28",
                    "segment_id": segment.segment_id,
                    "factor": round(factor, 6),
                    "before_wpm": round(before_wpm, 3),
                    "projected_wpm": round(projected, 3),
                    "after_wpm": round(after_wpm, 3),
                    "removed_silence_seconds": compaction["removed_seconds"],
                    "audio_path": str(final_path),
                }
            )
        return corrected, events

    voice_pipeline._pace_correct_segment_assets = pace_editorial_segments
    _INSTALLED = True
