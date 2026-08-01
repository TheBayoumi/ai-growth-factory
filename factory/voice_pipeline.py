from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from .audio_qc import analyze_audio, concatenate_segments, normalize_audio, split_narration, write_manifest
from .config import Settings
from .models import AudioMetrics, AudioReview, NarrationSegment, VoiceContract
from .qwen_omni_reviewer import QwenOmniReviewer
from .qwen_tts import Qwen3TTS
from .reviewer import OpenAIReviewer, review_passes


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
    return int(hashlib.sha256(f"{text}|{segment_id}|{attempt}".encode()).hexdigest()[:8], 16)


def _unload(provider: object) -> None:
    unload = getattr(provider, "unload", None)
    if callable(unload):
        unload()


def build_reviewed_narration(settings: Settings, narration: str, workdir: Path, voice_contract: VoiceContract | None = None) -> VoicePipelineResult:
    contract = voice_contract or settings.voice_contract
    texts = split_narration(narration, settings.narration_segments)
    tts = Qwen3TTS(settings)
    reviewer = QwenOmniReviewer(settings) if settings.reviewer_backend == "qwen_omni" else OpenAIReviewer(settings)
    segments = []
    for segment_id, text in enumerate(texts):
        instruction = contract.to_instruction(segment_id, len(texts))
        path = workdir / "segments" / f"segment-{segment_id:02d}-attempt-1.wav"
        tts.generate(text, instruction, path, _seed(text, segment_id, 1))
        segments.append(NarrationSegment(segment_id, text, instruction, path))
    history = []
    for attempt in range(1, settings.reviewer_max_attempts + 1):
        raw, timed = concatenate_segments(segments, workdir / f"voice-raw-{attempt}.wav", settings.audio_segment_pause_ms)
        normalized = normalize_audio(raw, workdir / f"voice-normalized-{attempt}.wav")
        metrics = analyze_audio(normalized, narration, settings, contract.target_wpm)
        if not metrics.passed:
            history.append({"type": "deterministic_qc", "attempt": attempt, "decision": "retry_segments", "failures": list(metrics.failures)})
            if attempt == settings.reviewer_max_attempts:
                break
            repair = ". ".join(metrics.failures)
            segments = [NarrationSegment(item.segment_id, item.text, contract.to_instruction(item.segment_id, len(segments), repair), workdir / "segments" / f"segment-{item.segment_id:02d}-attempt-{attempt + 1}.wav", attempt=attempt + 1) for item in segments]
            for item in segments:
                tts.generate(item.text, item.instruction, item.audio_path, _seed(item.text, item.segment_id, item.attempt))
            continue
        _unload(tts)
        review = reviewer.review(normalized, narration, contract, timed, metrics, attempt)
        history.append({"type": "model_review", "attempt": attempt, **review.as_dict()})
        if review_passes(review, settings):
            _unload(reviewer)
            final = workdir / "voice.wav"
            shutil.copy2(normalized, final)
            manifest = write_manifest(workdir / "voice-review-manifest.json", timed, metrics, history, {"backend": "qwen3", "model": settings.qwen_tts_model}, {"required": True, "backend": settings.reviewer_backend, "model": settings.reviewer_model})
            return VoicePipelineResult(final, manifest, metrics, review, attempt, tuple(timed), contract)
        if review.decision == "reject" or attempt == settings.reviewer_max_attempts:
            break
        failed = {item.segment_id: item for item in review.failed_segments}
        _unload(reviewer)
        next_segments = []
        for item in segments:
            failure = failed.get(item.segment_id)
            if not failure:
                next_segments.append(item)
                continue
            next_attempt = item.attempt + 1
            instruction = contract.to_instruction(item.segment_id, len(segments), failure.tts_instruction)
            path = workdir / "segments" / f"segment-{item.segment_id:02d}-attempt-{next_attempt}.wav"
            tts.generate(item.text, instruction, path, _seed(item.text, item.segment_id, next_attempt))
            next_segments.append(NarrationSegment(item.segment_id, item.text, instruction, path, attempt=next_attempt))
        segments = next_segments
    _unload(tts)
    _unload(reviewer)
    raise RuntimeError("Narration failed closed after bounded repair attempts")
