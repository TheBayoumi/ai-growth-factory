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
        allowed = set(cls.__dataclass_fields__)
        contract = cls(**{key: value for key, value in data.items() if key in allowed})
        contract.validate()
        return contract

    def validate(self) -> None:
        if not 100 <= self.target_wpm <= 210:
            raise ValueError("target_wpm must be between 100 and 210")
        for name in ("energy", "warmth", "pitch_variation", "hook_intensity", "articulation"):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def to_instruction(self, segment_index: int, segment_count: int, repair: str | None = None) -> str:
        section = "opening hook" if segment_index == 0 else "closing CTA" if segment_index == segment_count - 1 else "explanation"
        text = (
            f"Professional technology narrator. Style: {self.baseline_style}. Section: {section}. "
            f"Target {self.target_wpm} WPM. Energy {self.energy:.2f}, warmth {self.warmth:.2f}, "
            f"pitch variation {self.pitch_variation:.2f}, articulation {self.articulation:.2f}. "
            f"Pauses: {self.pause_style}. Speak the supplied words exactly."
        )
        return text + (f" Reviewer correction: {repair}" if repair else "")

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
        return cls(**{name: float(data.get(name, 0)) for name in cls.__dataclass_fields__})

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
    def from_dict(cls, data: dict[str, Any], model: str, raw_response: str = "") -> "AudioReview":
        decision = str(data.get("decision", "reject"))
        if decision not in {"approve", "retry_segments", "reject"}:
            raise ValueError("invalid review decision")
        failed = tuple(FailedSegment(int(item["segment_id"]), str(item.get("reason", "")), str(item.get("tts_instruction", ""))) for item in data.get("failed_segments", []))
        return cls(decision, float(data.get("overall_score", 0)), ReviewScores.from_dict(dict(data.get("scores") or {})), failed, str(data.get("summary", "")), model, raw_response)

    def as_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "overall_score": self.overall_score, "scores": self.scores.as_dict(), "failed_segments": [asdict(item) for item in self.failed_segments], "summary": self.summary, "reviewer_model": self.reviewer_model}
