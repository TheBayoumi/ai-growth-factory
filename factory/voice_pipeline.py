from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audio_qc import (
    analyze_audio,
    concatenate_segments,
    convert_for_reviewer,
    normalize_audio,
    split_narration,
    write_manifest,
)
from .config import Settings
from .models import AudioMetrics, AudioReview, FailedSegment, NarrationSegment, VoiceContract
from .qwen_tts import Qwen3TTS
from .qwen_omni_reviewer import QwenOmniReviewer
from .reviewer import OpenAIRealtimeReviewer, review_passes


class VoiceGenerationError(RuntimeError):
    pass


class TTSProvider(Protocol):
    def generate(self, *, text: str, instruction: str, output_path: Path, seed: int) -> Path: ...


class ReviewerProvider(Protocol):
    def review(
        self,
        *,
        audio_path: Path,
        narration: str,
        contract: object,
        segments: list[NarrationSegment],
        metrics: AudioMetrics,
        attempt: int,
    ) -> AudioReview: ...


@dataclass(frozen=True)
class VoicePipelineResult:
    audio_path: Path
    manifest_path: Path
    metrics: AudioMetrics
    review: AudioReview | None
    attempts: int
    segments: tuple[NarrationSegment, ...]
    voice_contract: VoiceContract


def _seed(text: str, segment_id: int, attempt: int) -> int:
    material = f"{text}|{segment_id}|{attempt}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def _global_repair(metrics: AudioMetrics, target_wpm: int) -> str:
    corrections: list[str] = []
    if metrics.estimated_wpm < target_wpm:
        corrections.append("Increase the speaking pace slightly while preserving clarity")
    elif metrics.estimated_wpm > target_wpm:
        corrections.append("Reduce the speaking pace slightly and articulate technical terms")
    if metrics.max_silence_seconds > 1.25 or metrics.silence_ratio > 0.28:
        corrections.append("Use shorter pauses and avoid dead air")
    if metrics.rms_dbfs < -32:
        corrections.append("Use a confident, projected delivery rather than a quiet delivery")
    if not corrections:
        corrections.append("Improve naturalness and consistency while speaking the words exactly")
    return ". ".join(corrections) + "."


def _threshold_repair(review: AudioReview) -> str:
    scores = review.scores
    corrections: list[str] = []
    if scores.script_fidelity < 0.98:
        corrections.append("Speak every supplied word exactly, without additions, omissions, or repetitions")
    if scores.naturalness < 0.85:
        corrections.append("Use smoother, more human phrasing and avoid a synthetic cadence")
    if scores.pronunciation < 0.92:
        corrections.append("Improve pronunciation and articulate names, numbers, and technical terms")
    if scores.audio_artifacts < 0.90:
        corrections.append("Produce clean speech without glitches, distortion, or abrupt boundaries")
    if not corrections:
        corrections.append("Increase overall publication quality while preserving the exact transcript")
    return ". ".join(corrections) + "."


def _unload(provider: object | None) -> None:
    if provider is None:
        return
    unload = getattr(provider, "unload", None)
    if callable(unload):
        unload()


def build_reviewed_narration(
    settings: Settings,
    narration: str,
    workdir: Path,
    *,
    tts: TTSProvider | None = None,
    reviewer: ReviewerProvider | None = None,
    voice_contract: VoiceContract | None = None,
) -> VoicePipelineResult:
    workdir.mkdir(parents=True, exist_ok=True)
    contract = voice_contract or settings.voice_contract
    contract.validate()
    segment_texts = split_narration(narration, settings.narration_segments)
    tts_provider = tts or Qwen3TTS(settings)
    reviewer_provider = reviewer
    generator_metadata: dict[str, object] = {
        "backend": settings.tts_backend,
        "model": settings.qwen_tts_model,
    }
    reviewer_metadata: dict[str, object] = {
        "required": settings.reviewer_required,
        "backend": settings.reviewer_backend,
        "model": settings.reviewer_model,
    }
    if settings.reviewer_required and reviewer_provider is None:
        if settings.reviewer_backend == "qwen_omni":
            reviewer_provider = QwenOmniReviewer(settings)
        elif settings.reviewer_backend == "openai":
            reviewer_provider = OpenAIRealtimeReviewer(settings)
        else:
            raise VoiceGenerationError(
                f"Unsupported reviewer backend: {settings.reviewer_backend}"
            )

    segments: list[NarrationSegment] = []
    for segment_id, text in enumerate(segment_texts):
        instruction = contract.to_instruction(
            segment_index=segment_id,
            segment_count=len(segment_texts),
        )
        output = workdir / "segments" / f"segment-{segment_id:02d}-attempt-1.wav"
        tts_provider.generate(
            text=text,
            instruction=instruction,
            output_path=output,
            seed=_seed(text, segment_id, 1),
        )
        segments.append(
            NarrationSegment(
                segment_id=segment_id,
                text=text,
                instruction=instruction,
                audio_path=output,
                attempt=1,
            )
        )

    review_history: list[dict[str, object]] = []
    final_review: AudioReview | None = None
    final_metrics: AudioMetrics | None = None
    final_segments: list[NarrationSegment] = []
    final_audio = workdir / "voice.wav"

    for attempt in range(1, settings.reviewer_max_attempts + 1):
        raw_audio = workdir / f"voice-raw-attempt-{attempt}.wav"
        raw_audio, timed_segments = concatenate_segments(
            segments,
            raw_audio,
            pause_ms=settings.audio_segment_pause_ms,
        )
        normalized = workdir / f"voice-normalized-attempt-{attempt}.wav"
        normalize_audio(
            raw_audio,
            normalized,
            target_lufs=settings.audio_target_lufs,
            peak_dbfs=settings.audio_peak_limit_dbfs,
        )
        metrics = analyze_audio(
            normalized, narration=narration, settings=settings, target_wpm=contract.target_wpm
        )
        final_metrics = metrics
        final_segments = timed_segments

        if not metrics.passed:
            review_history.append(
                {
                    "attempt": attempt,
                    "type": "deterministic_qc",
                    "decision": "retry_segments" if attempt < settings.reviewer_max_attempts else "reject",
                    "failures": list(metrics.failures),
                }
            )
            if attempt >= settings.reviewer_max_attempts:
                break
            repair = _global_repair(metrics, contract.target_wpm)
            regenerated: list[NarrationSegment] = []
            for segment in segments:
                next_attempt = segment.attempt + 1
                instruction = contract.to_instruction(
                    segment_index=segment.segment_id,
                    segment_count=len(segments),
                    repair=repair,
                )
                output = workdir / "segments" / (
                    f"segment-{segment.segment_id:02d}-attempt-{next_attempt}.wav"
                )
                tts_provider.generate(
                    text=segment.text,
                    instruction=instruction,
                    output_path=output,
                    seed=_seed(segment.text, segment.segment_id, next_attempt),
                )
                regenerated.append(
                    NarrationSegment(
                        segment_id=segment.segment_id,
                        text=segment.text,
                        instruction=instruction,
                        audio_path=output,
                        attempt=next_attempt,
                    )
                )
            segments = regenerated
            continue

        if not settings.reviewer_required:
            _unload(tts_provider)
            shutil.copy2(normalized, final_audio)
            manifest = write_manifest(
                workdir / "voice-review-manifest.json",
                segments=timed_segments,
                metrics=metrics,
                attempts=attempt,
                reviews=review_history,
                voice_contract=contract.as_dict(),
                generator=generator_metadata,
                reviewer=reviewer_metadata,
            )
            return VoicePipelineResult(
                audio_path=final_audio,
                manifest_path=manifest,
                metrics=metrics,
                review=None,
                attempts=attempt,
                segments=tuple(timed_segments),
                voice_contract=contract,
            )

        assert reviewer_provider is not None
        _unload(tts_provider)
        review_audio = workdir / f"voice-review-attempt-{attempt}.wav"
        convert_for_reviewer(normalized, review_audio)
        review = reviewer_provider.review(
            audio_path=review_audio,
            narration=narration,
            contract=contract,
            segments=timed_segments,
            metrics=metrics,
            attempt=attempt,
        )
        final_review = review
        review_history.append({"attempt": attempt, "type": "model_review", **review.as_dict()})
        if review_passes(review, settings):
            _unload(reviewer_provider)
            shutil.copy2(normalized, final_audio)
            manifest = write_manifest(
                workdir / "voice-review-manifest.json",
                segments=timed_segments,
                metrics=metrics,
                attempts=attempt,
                reviews=review_history,
                voice_contract=contract.as_dict(),
                generator=generator_metadata,
                reviewer=reviewer_metadata,
            )
            return VoicePipelineResult(
                audio_path=final_audio,
                manifest_path=manifest,
                metrics=metrics,
                review=review,
                attempts=attempt,
                segments=tuple(timed_segments),
                voice_contract=contract,
            )
        if review.decision == "reject" or attempt >= settings.reviewer_max_attempts:
            break
        failures_by_id = {failure.segment_id: failure for failure in review.failed_segments}
        if not failures_by_id:
            # Reviewer output can say "approve" while still missing locally enforced score
            # thresholds. Treat that contradiction as a retry of all segments rather than
            # publishing weak audio or crashing without a repair path.
            repair = _threshold_repair(review)
            failures_by_id = {
                segment.segment_id: FailedSegment(
                    segment_id=segment.segment_id,
                    reason="Reviewer approval did not satisfy local score thresholds.",
                    tts_instruction=repair,
                )
                for segment in segments
            }
        valid_ids = {segment.segment_id for segment in segments}
        unknown = sorted(set(failures_by_id) - valid_ids)
        if unknown:
            raise VoiceGenerationError(f"Reviewer referenced unknown segment IDs: {unknown}")
        _unload(reviewer_provider)
        repaired: list[NarrationSegment] = []
        for segment in segments:
            failure = failures_by_id.get(segment.segment_id)
            if failure is None:
                repaired.append(segment)
                continue
            next_attempt = segment.attempt + 1
            instruction = contract.to_instruction(
                segment_index=segment.segment_id,
                segment_count=len(segments),
                repair=failure.tts_instruction,
            )
            output = workdir / "segments" / (
                f"segment-{segment.segment_id:02d}-attempt-{next_attempt}.wav"
            )
            tts_provider.generate(
                text=segment.text,
                instruction=instruction,
                output_path=output,
                seed=_seed(segment.text, segment.segment_id, next_attempt),
            )
            repaired.append(
                NarrationSegment(
                    segment_id=segment.segment_id,
                    text=segment.text,
                    instruction=instruction,
                    audio_path=output,
                    attempt=next_attempt,
                )
            )
        segments = repaired

    _unload(tts_provider)
    _unload(reviewer_provider)
    assert final_metrics is not None
    manifest = write_manifest(
        workdir / "voice-review-manifest.json",
        segments=final_segments,
        metrics=final_metrics,
        attempts=settings.reviewer_max_attempts,
        reviews=review_history,
        voice_contract=contract.as_dict(),
        generator=generator_metadata,
        reviewer=reviewer_metadata,
    )
    reason = (
        final_review.summary
        if final_review is not None and final_review.summary
        else "; ".join(final_metrics.failures) or "review thresholds were not met"
    )
    raise VoiceGenerationError(
        f"Narration failed closed after {settings.reviewer_max_attempts} attempts: {reason}. "
        f"Manifest: {manifest}"
    )
