from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .models import AudioReview, FailedSegment, NarrationSegment
from .video_profile import VideoProfile


_INSTALLED = False
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")
_PACE_RE = re.compile(r"\b(?:pace|faster|slower|speed|words per minute|wpm)\b", re.IGNORECASE)
_PAUSE_RE = re.compile(r"\b(?:pause|pauses|cadence|phrasing|naturalness|rhythm)\b", re.IGNORECASE)
_CRITICAL_RE = re.compile(
    r"\b(?:omit|omission|missing|insert|addition|repeat|substitut|paraphras|"
    r"mispronoun|pronunciation|noise|click|distort|artifact|metallic|corrupt|garbl)\w*\b",
    re.IGNORECASE,
)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _split_long_unit(unit: str, maximum_words: int) -> list[str]:
    clauses = [part.strip() for part in _CLAUSE_RE.split(unit) if part.strip()]
    if len(clauses) == 1:
        words = unit.split()
        return [
            " ".join(words[index : index + maximum_words])
            for index in range(0, len(words), maximum_words)
        ]

    result: list[str] = []
    current: list[str] = []
    for clause in clauses:
        clause_words = clause.split()
        if len(clause_words) > maximum_words:
            if current:
                result.append(" ".join(current))
                current = []
            result.extend(
                " ".join(clause_words[index : index + maximum_words])
                for index in range(0, len(clause_words), maximum_words)
            )
            continue
        if current and len(current) + len(clause_words) > maximum_words:
            result.append(" ".join(current))
            current = clause_words
        else:
            current.extend(clause_words)
    if current:
        result.append(" ".join(current))
    return result


def split_narration_for_voice_v28(text: str, target_segments: int) -> list[str]:
    """Create sentence-aligned TTS segments with bounded word counts.

    The legacy fixed-count splitter merged multiple sentences into 35-39 word segments. Qwen
    then generated those long segments far below the requested pace. v28 keeps semantic
    boundaries, splits long units, and merges only genuinely short fragments.
    """
    clean = " ".join(text.split()).strip()
    if not clean:
        raise ValueError("Narration is empty")
    minimum_words = _env_int("V28_MIN_VOICE_SEGMENT_WORDS", 8, 3, 20)
    maximum_words = _env_int("V28_MAX_VOICE_SEGMENT_WORDS", 24, minimum_words, 40)
    maximum_segments = _env_int("V28_MAX_VOICE_SEGMENTS", 12, 4, 20)

    sentences = [part.strip() for part in _SENTENCE_RE.split(clean) if part.strip()] or [clean]
    segments: list[str] = []
    for sentence in sentences:
        if len(sentence.split()) <= maximum_words:
            segments.append(sentence)
        else:
            segments.extend(_split_long_unit(sentence, maximum_words))

    index = 0
    while index < len(segments):
        if len(segments[index].split()) >= minimum_words or len(segments) == 1:
            index += 1
            continue
        merged = False
        if index + 1 < len(segments):
            candidate = f"{segments[index]} {segments[index + 1]}"
            if len(candidate.split()) <= maximum_words:
                segments[index] = candidate
                del segments[index + 1]
                merged = True
        if not merged and index > 0:
            candidate = f"{segments[index - 1]} {segments[index]}"
            if len(candidate.split()) <= maximum_words:
                segments[index - 1] = candidate
                del segments[index]
                merged = True
                index -= 1
        if not merged:
            index += 1

    requested = max(1, int(target_segments))
    while len(segments) < requested:
        candidates = [
            (len(segment.split()), index)
            for index, segment in enumerate(segments)
            if len(segment.split()) >= minimum_words * 2
        ]
        if not candidates:
            break
        _size, index = max(candidates)
        words = segments[index].split()
        split_at = len(words) // 2
        segments[index : index + 1] = [
            " ".join(words[:split_at]),
            " ".join(words[split_at:]),
        ]

    if len(segments) > maximum_segments:
        raise ValueError(
            f"Narration requires {len(segments)} bounded TTS segments, above configured maximum "
            f"{maximum_segments}"
        )
    if any(not segment.strip() or len(segment.split()) > maximum_words for segment in segments):
        raise ValueError("Narration segmentation violated the v28 word-count contract")
    if " ".join(" ".join(segments).split()) != clean:
        raise ValueError("Narration segmentation changed the supplied transcript")
    return segments


def segment_wpm_v28(segment: NarrationSegment) -> float:
    from .audio_qc import wav_duration

    duration = wav_duration(segment.audio_path)
    return len(segment.text.split()) / max(duration, 0.001) * 60.0


def best_reachable_calibration_event_v28(
    events: Sequence[dict[str, object]],
    *,
    target_wpm: float = 142.0,
) -> dict[str, object] | None:
    reachable = [event for event in events if bool(event.get("reachable"))]
    if not reachable:
        return None
    return min(
        reachable,
        key=lambda event: (
            abs(float(event.get("projected_wpm") or 0.0) - target_wpm),
            abs(float(event.get("required_tempo_factor") or 1.0) - 1.0),
            -int(event.get("internal_attempt") or 0),
        ),
    )


def ground_pace_failure_v28(
    failure: FailedSegment,
    *,
    measured_wpm: float,
    profile: VideoProfile,
) -> FailedSegment | None:
    """Prevent subjective pace feedback from contradicting deterministic segment timing."""
    combined = f"{failure.reason} {failure.tts_instruction}"
    if not _PACE_RE.search(combined):
        return failure
    if not profile.minimum_wpm <= measured_wpm <= profile.maximum_wpm:
        return failure
    if _CRITICAL_RE.search(combined):
        return failure
    if not _PAUSE_RE.search(combined):
        return None
    return replace(
        failure,
        reason=(
            f"{failure.reason} Deterministic measurement is already {measured_wpm:.1f} WPM, "
            "inside the accepted range; only pause timing and natural phrasing may change."
        ),
        tts_instruction=(
            f"Keep the speaking pace unchanged near {measured_wpm:.0f} words per minute. "
            "Improve only clause-boundary pauses, cadence, and natural phrasing. Do not make "
            "the segment faster, do not alter the transcript, and keep every word clear."
        ),
    )


def install_production_voice_convergence_v28() -> None:
    """Make TTS regeneration convergent without weakening the voice-quality gate."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import production_voice_calibration_v28 as calibration
    from . import qwen_omni_reviewer, qwen_tts, voice_pipeline

    profile = VideoProfile.from_env()
    base_tts = voice_pipeline.Qwen3TTS
    base_reviewer = qwen_omni_reviewer.QwenOmniReviewer
    original_segment_prompt = qwen_omni_reviewer._segment_prompt
    calibration_rounds = _env_int("V28_TTS_CALIBRATION_ROUNDS", 3, 1, 5)

    class ConvergentQwen3TTS(base_tts):
        def __init__(self, settings: Any) -> None:
            super().__init__(settings)
            self._best_reachable: dict[str, tuple[float, Path, dict[str, object]]] = {}

        def generate(
            self,
            *,
            text: str,
            instruction: str,
            output_path: Path,
            seed: int,
        ) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
            current_instruction = instruction
            last_observed: float | None = None
            generation_errors: list[str] = []

            for convergence_round in range(1, calibration_rounds + 1):
                candidate_output = output_path.with_name(
                    f"{output_path.stem}-convergence-{convergence_round}{output_path.suffix}"
                )
                event_start = len(calibration._CALIBRATION_EVENTS)
                round_seed = (int(seed) + convergence_round * 0x85EBCA6B) & 0xFFFFFFFF
                try:
                    super().generate(
                        text=text,
                        instruction=current_instruction,
                        output_path=candidate_output,
                        seed=round_seed,
                    )
                except Exception as exc:
                    generation_errors.append(str(exc))
                    candidate_output.unlink(missing_ok=True)
                    continue

                round_events = [
                    event
                    for event in calibration._CALIBRATION_EVENTS[event_start:]
                    if str(event.get("requested_output") or "") == str(candidate_output)
                ]
                selected = next(
                    (event for event in round_events if bool(event.get("selected"))),
                    None,
                )
                if selected is None:
                    candidate_output.unlink(missing_ok=True)
                    raise qwen_tts.QwenTTSError(
                        "v28 calibration produced no auditable selected candidate"
                    )

                last_observed = float(selected.get("observed_compacted_wpm") or 0.0)
                if bool(selected.get("reachable")):
                    score = (
                        abs(float(selected.get("projected_wpm") or 0.0) - profile.target_wpm)
                        + abs(float(selected.get("required_tempo_factor") or 1.0) - 1.0) * 4.0
                    )
                    stable = output_path.parent / f".v28-best-reachable-{key}.wav"
                    previous = self._best_reachable.get(key)
                    if previous is None or score <= previous[0]:
                        shutil.copy2(candidate_output, stable)
                        self._best_reachable[key] = (score, stable, dict(selected))
                    shutil.copy2(candidate_output, output_path)
                    calibration._CALIBRATION_EVENTS.append(
                        {
                            "type": "tts_candidate_arbitration_v28",
                            "requested_output": str(output_path),
                            "convergence_round": convergence_round,
                            "decision": "accept_current_reachable",
                            "observed_compacted_wpm": round(last_observed, 3),
                            "required_tempo_factor": selected.get("required_tempo_factor"),
                        }
                    )
                    candidate_output.unlink(missing_ok=True)
                    return output_path

                candidate_output.unlink(missing_ok=True)
                requested_wpm = calibration.synthesis_target_for_observation_v28(
                    last_observed,
                    profile=profile,
                )
                current_instruction = calibration._calibration_instruction(
                    instruction,
                    observed_wpm=last_observed,
                    requested_wpm=requested_wpm,
                )

            cached = self._best_reachable.get(key)
            if cached is not None and cached[1].is_file():
                shutil.copy2(cached[1], output_path)
                calibration._CALIBRATION_EVENTS.append(
                    {
                        "type": "tts_candidate_arbitration_v28",
                        "requested_output": str(output_path),
                        "decision": "reuse_best_reachable",
                        "cached_candidate": str(cached[1]),
                        "cached_score": round(cached[0], 6),
                        "last_unreachable_wpm": round(last_observed or 0.0, 3),
                    }
                )
                return output_path

            detail = "; ".join(generation_errors[-2:])
            raise qwen_tts.QwenTTSError(
                "Qwen TTS produced no candidate reachable within the v28 1.15x tempo ceiling"
                + (f": {detail}" if detail else "")
            )

        def unload(self) -> None:
            for _score, path, _event in self._best_reachable.values():
                path.unlink(missing_ok=True)
            self._best_reachable.clear()
            super().unload()

    def grounded_segment_prompt(
        *,
        segment: NarrationSegment,
        contract: Any,
        metrics: Any,
        attempt: int,
    ) -> str:
        base = original_segment_prompt(
            segment=segment,
            contract=contract,
            metrics=metrics,
            attempt=attempt,
        )
        measured = segment_wpm_v28(segment)
        rule = (
            f"\n\nDeterministic segment measurement: {measured:.2f} words per minute. "
            f"The accepted publication range is {profile.minimum_wpm}-{profile.maximum_wpm} WPM. "
            "Do not request faster or slower speech when this measurement is inside the range. "
            "Judge pauses and naturalness separately. A pause-only defect must request pause and "
            "cadence repair while explicitly preserving the measured speaking pace."
        )
        marker = "\n\nReturn one JSON object only"
        return base.replace(marker, rule + marker)

    class GroundedQwenOmniReviewer(base_reviewer):
        def review(self, **kwargs: Any) -> AudioReview:
            review = super().review(**kwargs)
            if review.decision == "reject" or not review.failed_segments:
                return review
            segments = {
                segment.segment_id: segment
                for segment in kwargs.get("segments") or []
            }
            grounded: list[FailedSegment] = []
            for failure in review.failed_segments:
                segment = segments.get(failure.segment_id)
                if segment is None:
                    grounded.append(failure)
                    continue
                adjusted = ground_pace_failure_v28(
                    failure,
                    measured_wpm=segment_wpm_v28(segment),
                    profile=profile,
                )
                if adjusted is not None:
                    grounded.append(adjusted)
            return replace(
                review,
                decision="retry_segments" if grounded else "approve",
                failed_segments=tuple(grounded),
                summary=(
                    review.summary
                    if grounded
                    else "Subjective pace-only retries were cleared by deterministic segment timing."
                ),
            )

    voice_pipeline.Qwen3TTS = ConvergentQwen3TTS
    voice_pipeline.QwenOmniReviewer = GroundedQwenOmniReviewer
    voice_pipeline.split_narration = split_narration_for_voice_v28
    voice_pipeline._pace_is_only_failure = lambda _metrics: False
    qwen_omni_reviewer._segment_prompt = grounded_segment_prompt
    _INSTALLED = True
