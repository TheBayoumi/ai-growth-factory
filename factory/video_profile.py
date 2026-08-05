from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class VideoProfile:
    """Runtime-editable editorial contract for one short-form channel profile.

    ``target_shots`` controls the preferred editorial density. ``maximum_shots`` is a hard
    fail-closed capacity for narrations whose natural beat boundaries need additional short shots.
    The planner never pads a video to the hard capacity and never exceeds the shot-duration gate.
    """

    name: str = "tech_news_explainer_v35"
    target_wpm: int = 142
    minimum_wpm: int = 138
    maximum_wpm: int = 146
    maximum_tempo_factor: float = 1.15
    segment_pause_ms: int = 280
    pre_cta_pause_ms: int = 550
    minimum_shots: int = 16
    target_shots: int = 20
    maximum_shots: int = 24
    minimum_shot_seconds: float = 1.65
    maximum_shot_seconds: float = 4.25
    maximum_wan_shot_seconds: float = 3.30
    first_ten_seconds_minimum_shots: int = 4
    wan_shots: int = 6
    caption_minimum_words: int = 2
    caption_maximum_words: int = 5
    caption_maximum_characters: int = 34
    caption_minimum_seconds: float = 0.65
    caption_maximum_seconds: float = 1.90
    maximum_single_word_cues: int = 2
    caption_baseline_ratio: float = 0.80
    allow_asset_looping: bool = False
    allow_destructive_caption_matte: bool = False

    def validate(self) -> None:
        if not 110 <= self.target_wpm <= 170:
            raise ValueError("target_wpm must be between 110 and 170")
        if not self.minimum_wpm <= self.target_wpm <= self.maximum_wpm:
            raise ValueError("target_wpm must be inside the configured WPM range")
        if not 1.0 <= self.maximum_tempo_factor <= 1.20:
            raise ValueError("maximum_tempo_factor must be between 1.0 and 1.20")
        if not 8 <= self.minimum_shots <= self.target_shots <= self.maximum_shots <= 30:
            raise ValueError("shot-count limits are inconsistent")
        if self.maximum_shots - self.target_shots > 8:
            raise ValueError("maximum_shots may exceed target_shots by at most eight")
        if not 1.0 <= self.minimum_shot_seconds < self.maximum_shot_seconds <= 6.0:
            raise ValueError("shot-duration limits are inconsistent")
        if not self.minimum_shot_seconds <= self.maximum_wan_shot_seconds <= 4.0:
            raise ValueError("maximum_wan_shot_seconds is outside the shot-duration range")
        if not 1 <= self.wan_shots <= self.maximum_shots:
            raise ValueError("wan_shots is invalid")
        if not 1 <= self.caption_minimum_words <= self.caption_maximum_words <= 8:
            raise ValueError("caption word limits are inconsistent")
        if self.caption_maximum_characters < 16:
            raise ValueError("caption_maximum_characters is too small")
        if not 0.55 <= self.caption_baseline_ratio <= 0.88:
            raise ValueError("caption_baseline_ratio must stay in the platform-safe range")
        if self.allow_asset_looping:
            raise ValueError("v35 never permits source-asset looping")
        if self.allow_destructive_caption_matte:
            raise ValueError("v35 never permits destructive caption mattes")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "VideoProfile":
        profile = cls()
        raw = os.getenv("VIDEO_PROFILE_JSON", "").strip()
        if raw:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("VIDEO_PROFILE_JSON must contain an object")
            allowed = set(profile.__dataclass_fields__)
            unknown = sorted(set(payload) - allowed)
            if unknown:
                raise ValueError("Unknown video profile fields: " + ", ".join(unknown))
            profile = replace(profile, **payload)

        scalar_overrides: tuple[tuple[str, str, type], ...] = (
            ("VIDEO_TARGET_WPM", "target_wpm", int),
            ("VIDEO_MIN_WPM", "minimum_wpm", int),
            ("VIDEO_MAX_WPM", "maximum_wpm", int),
            ("VIDEO_MAX_TEMPO_FACTOR", "maximum_tempo_factor", float),
            ("VIDEO_SEGMENT_PAUSE_MS", "segment_pause_ms", int),
            ("V28_MIN_SHOTS", "minimum_shots", int),
            ("V28_TARGET_SHOTS", "target_shots", int),
            ("V28_MAX_SHOTS", "maximum_shots", int),
            ("V28_MAX_SHOT_SECONDS", "maximum_shot_seconds", float),
            ("V28_MAX_WAN_SHOT_SECONDS", "maximum_wan_shot_seconds", float),
            ("V28_WAN_SHOTS", "wan_shots", int),
        )
        changes: dict[str, Any] = {}
        for env_name, field_name, converter in scalar_overrides:
            value = os.getenv(env_name)
            if value is not None and value.strip():
                changes[field_name] = converter(value)
        if changes:
            profile = replace(profile, **changes)
        profile.validate()
        return profile
