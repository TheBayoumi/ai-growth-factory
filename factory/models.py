from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scene:
    heading: str
    body: str
    visual: str
    source_index: int = 0


@dataclass(frozen=True)
class VideoPackage:
    topic: str
    narration: str
    title: str
    description: str
    tags: list[str]
    thumbnail_text: str
    top_comment: str
    scenes: list[Scene]
    source_urls: list[str]
    source_publishers: list[str]


@dataclass(frozen=True)
class VoiceContract:
    baseline_style: str = "authoritative"
    target_wpm: int = 155
    energy: float = 0.68
    warmth: float = 0.56
    pitch_variation: float = 0.42
    hook_intensity: float = 0.82
    articulation: float = 0.86
    pause_style: str = "short rhetorical pauses; no dramatic dead air"
    pronunciation_lexicon: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceContract":
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        clean = {key: value for key, value in data.items() if key in allowed}
        contract = cls(**clean)
        contract.validate()
        return contract

    def validate(self) -> None:
        if not 100 <= self.target_wpm <= 210:
            raise ValueError("voice contract target_wpm must be between 100 and 210")
        for name in ("energy", "warmth", "pitch_variation", "hook_intensity", "articulation"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"voice contract {name} must be between 0 and 1")

    def to_instruction(self, *, segment_index: int, segment_count: int, repair: str | None = None) -> str:
        section = "opening hook" if segment_index == 0 else "closing CTA" if segment_index == segment_count - 1 else "explanation"
        instruction = (
            f"Professional technology narrator. Baseline style: {self.baseline_style}. "
            f"This is the {section}. Target about {self.target_wpm} words per minute. "
            f"Energy {self.energy:.2f}/1, warmth {self.warmth:.2f}/1, pitch variation "
            f"{self.pitch_variation:.2f}/1, articulation {self.articulation:.2f}/1. "
            f"Hook intensity {self.hook_intensity:.2f}/1 for the first segment. "
            f"Pauses: {self.pause_style}. Sound credible and human, never sensational, "
            "never imitate a named person, and speak the supplied words exactly."
        )
        if repair:
            instruction += f" Reviewer correction: {repair.strip()}"
        return instruction

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NarrationSegment:
    segment_id: int
    text: str
    instruction: str
    audio_path: Path
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    attempt: int = 1

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(frozen=True)
class AudioMetrics:
    duration_seconds: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    max_silence_seconds: float
    estimated_wpm: float
    dc_offset: float
    passed: bool
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailedSegment:
    segment_id: int
    reason: str
    tts_instruction: str


@dataclass(frozen=True)
class ReviewScores:
    script_fidelity: float
    naturalness: float
    authority: float
    engagement: float
    pronunciation: float
    pace: float
    pause_quality: float
    emotional_match: float
    audio_artifacts: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewScores":
        fields = cls.__dataclass_fields__.keys()
        scores = cls(**{name: float(data.get(name, 0.0)) for name in fields})
        for name in fields:
            value = getattr(scores, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"review score {name} must be between 0 and 1")
        return scores

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AudioReview:
    decision: str
    overall_score: float
    scores: ReviewScores
    failed_segments: tuple[FailedSegment, ...]
    summary: str
    reviewer_model: str
    raw_response: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, model: str, raw_response: str = "") -> "AudioReview":
        decision = str(data.get("decision", "reject"))
        if decision not in {"approve", "retry_segments", "reject"}:
            raise ValueError("review decision must be approve, retry_segments, or reject")
        overall = float(data.get("overall_score", 0.0))
        if not 0.0 <= overall <= 1.0:
            raise ValueError("overall_score must be between 0 and 1")
        failed = tuple(
            FailedSegment(
                segment_id=int(item["segment_id"]),
                reason=str(item.get("reason", "")).strip(),
                tts_instruction=str(item.get("tts_instruction", "")).strip(),
            )
            for item in data.get("failed_segments", [])
        )
        segment_ids = [item.segment_id for item in failed]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("failed_segments must not contain duplicate segment IDs")
        if decision == "approve" and failed:
            raise ValueError("approve reviews must not contain failed_segments")
        if decision == "retry_segments":
            if not failed:
                raise ValueError("retry_segments reviews must identify at least one segment")
            if any(not item.reason or not item.tts_instruction for item in failed):
                raise ValueError("retry segment feedback requires a reason and TTS instruction")
        return cls(
            decision=decision,
            overall_score=overall,
            scores=ReviewScores.from_dict(dict(data.get("scores") or {})),
            failed_segments=failed,
            summary=str(data.get("summary", "")),
            reviewer_model=model,
            raw_response=raw_response,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "overall_score": self.overall_score,
            "scores": self.scores.as_dict(),
            "failed_segments": [asdict(item) for item in self.failed_segments],
            "summary": self.summary,
            "reviewer_model": self.reviewer_model,
        }
