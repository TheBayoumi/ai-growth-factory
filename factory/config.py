from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import VoiceContract


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _json(name: str) -> dict[str, Any] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _voice_contract() -> VoiceContract:
    path = os.getenv("VOICE_CONTRACT_FILE", "").strip()
    if path and Path(path).is_file():
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return VoiceContract.from_dict(value)
    return VoiceContract()


@dataclass(frozen=True)
class Settings:
    youtube_oauth: dict[str, Any] | None
    publish_enabled: bool
    privacy_status: str
    reviewer_required: bool
    reviewer_backend: str
    openai_api_key: str | None
    openai_reviewer_model: str
    qwen_omni_model: str
    qwen_omni_device: str
    qwen_omni_dtype: str
    qwen_omni_attention: str
    qwen_omni_max_new_tokens: int
    reviewer_max_attempts: int
    reviewer_overall_threshold: float
    reviewer_fidelity_threshold: float
    reviewer_naturalness_threshold: float
    reviewer_pronunciation_threshold: float
    llm_base_url: str
    llm_model: str
    llm_hf_model: str
    llm_executable: str
    llm_managed: bool
    llm_gpu_layers: int
    llm_context_tokens: int
    llm_timeout_seconds: int
    llm_startup_timeout_seconds: int
    llm_temperature: float
    qwen_tts_model: str
    qwen_tts_mode: str
    qwen_tts_speaker: str
    qwen_tts_language: str
    qwen_tts_device: str
    qwen_tts_dtype: str
    qwen_tts_attention: str
    qwen_ref_audio: Path | None
    qwen_ref_text: str | None
    voice_contract: VoiceContract
    narration_segments: int
    audio_segment_pause_ms: int
    audio_peak_limit_dbfs: float
    audio_min_rms_dbfs: float
    audio_max_clipping_ratio: float
    audio_max_silence_ratio: float
    audio_max_silence_seconds: float
    audio_wpm_tolerance: int
    min_primary_sources: int
    max_source_age_hours: int
    max_recent_videos: int
    width: int
    height: int
    fps: int
    language: str
    region: str
    timezone: str
    monetization_url: str | None
    monetization_label: str
    work_root: Path
    state_root: Path
    cron_secret: str | None
    factory_secret: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            youtube_oauth=_json("YOUTUBE_OAUTH_JSON"),
            publish_enabled=_bool("PUBLISH_ENABLED"),
            privacy_status=os.getenv("YOUTUBE_PRIVACY_STATUS", "private"),
            reviewer_required=_bool("REVIEWER_REQUIRED", True),
            reviewer_backend=os.getenv("REVIEWER_BACKEND", "qwen_omni"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_reviewer_model=os.getenv("OPENAI_REVIEW_MODEL", "gpt-realtime-2.1"),
            qwen_omni_model=os.getenv("QWEN_OMNI_REVIEW_MODEL", "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"),
            qwen_omni_device=os.getenv("QWEN_OMNI_DEVICE", "cuda:0"),
            qwen_omni_dtype=os.getenv("QWEN_OMNI_DTYPE", "float16"),
            qwen_omni_attention=os.getenv("QWEN_OMNI_ATTENTION", "sdpa"),
            qwen_omni_max_new_tokens=_int("QWEN_OMNI_MAX_NEW_TOKENS", 900, 128, 2400),
            reviewer_max_attempts=_int("VOICE_REVIEW_MAX_ATTEMPTS", 3, 1, 5),
            reviewer_overall_threshold=_float("REVIEW_OVERALL_THRESHOLD", 0.87, 0.5, 1),
            reviewer_fidelity_threshold=_float("REVIEW_FIDELITY_THRESHOLD", 0.98, 0.5, 1),
            reviewer_naturalness_threshold=_float("REVIEW_NATURALNESS_THRESHOLD", 0.85, 0.5, 1),
            reviewer_pronunciation_threshold=_float("REVIEW_PRONUNCIATION_THRESHOLD", 0.92, 0.5, 1),
            llm_base_url=os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/"),
            llm_model=os.getenv("LLAMA_CPP_MODEL", "Qwen3-4B-Q4_K_M.gguf"),
            llm_hf_model=os.getenv("LLAMA_CPP_HF_MODEL", "Qwen/Qwen3-4B-GGUF:Q4_K_M"),
            llm_executable=os.getenv("LLAMA_CPP_EXECUTABLE", "llama-server"),
            llm_managed=_bool("LLAMA_CPP_MANAGED", True),
            llm_gpu_layers=_int("LLAMA_CPP_GPU_LAYERS", 99, 0, 999),
            llm_context_tokens=_int("LLAMA_CPP_CONTEXT_TOKENS", 16384, 4096, 65536),
            llm_timeout_seconds=_int("LLAMA_CPP_TIMEOUT_SECONDS", 240, 30, 900),
            llm_startup_timeout_seconds=_int("LLAMA_CPP_STARTUP_TIMEOUT_SECONDS", 240, 30, 900),
            llm_temperature=_float("LLAMA_CPP_TEMPERATURE", 0.45, 0, 1.5),
            qwen_tts_model=os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            qwen_tts_mode=os.getenv("QWEN_TTS_MODE", "custom_voice"),
            qwen_tts_speaker=os.getenv("QWEN_TTS_SPEAKER", "Ryan"),
            qwen_tts_language=os.getenv("QWEN_TTS_LANGUAGE", "English"),
            qwen_tts_device=os.getenv("QWEN_TTS_DEVICE", "cuda:0"),
            qwen_tts_dtype=os.getenv("QWEN_TTS_DTYPE", "float16"),
            qwen_tts_attention=os.getenv("QWEN_TTS_ATTENTION", "sdpa"),
            qwen_ref_audio=Path(os.environ["QWEN_REF_AUDIO"]) if os.getenv("QWEN_REF_AUDIO") else None,
            qwen_ref_text=os.getenv("QWEN_REF_TEXT") or None,
            voice_contract=_voice_contract(),
            narration_segments=_int("NARRATION_SEGMENTS", 6, 1, 12),
            audio_segment_pause_ms=_int("AUDIO_SEGMENT_PAUSE_MS", 140, 0, 1000),
            audio_peak_limit_dbfs=_float("AUDIO_PEAK_LIMIT_DBFS", -1, -6, -0.1),
            audio_min_rms_dbfs=_float("AUDIO_MIN_RMS_DBFS", -32, -50, -10),
            audio_max_clipping_ratio=_float("AUDIO_MAX_CLIPPING_RATIO", 0.0005, 0, 0.05),
            audio_max_silence_ratio=_float("AUDIO_MAX_SILENCE_RATIO", 0.28, 0, 0.8),
            audio_max_silence_seconds=_float("AUDIO_MAX_SILENCE_SECONDS", 1.25, 0.1, 10),
            audio_wpm_tolerance=_int("AUDIO_WPM_TOLERANCE", 32, 5, 80),
            min_primary_sources=_int("MIN_PRIMARY_SOURCES", 2, 2, 6),
            max_source_age_hours=_int("MAX_SOURCE_AGE_HOURS", 48, 12, 168),
            max_recent_videos=_int("MAX_RECENT_VIDEOS", 30, 5, 50),
            width=_int("VIDEO_WIDTH", 720, 360, 1080),
            height=_int("VIDEO_HEIGHT", 1280, 640, 1920),
            fps=_int("VIDEO_FPS", 24, 15, 30),
            language=os.getenv("CHANNEL_LANGUAGE", "en"),
            region=os.getenv("CHANNEL_REGION", "EG"),
            timezone=os.getenv("TIMEZONE_NAME", "Africa/Cairo"),
            monetization_url=os.getenv("MONETIZATION_URL") or None,
            monetization_label=os.getenv("MONETIZATION_LABEL", "Get the full implementation"),
            work_root=Path(os.getenv("WORK_ROOT", "/tmp/ai-growth-factory")),
            state_root=Path(os.getenv("STATE_ROOT", "./state")),
            cron_secret=os.getenv("CRON_SECRET") or None,
            factory_secret=os.getenv("FACTORY_SECRET") or None,
        )
        settings.validate()
        return settings

    @property
    def reviewer_model(self) -> str:
        return self.openai_reviewer_model if self.reviewer_backend == "openai" else self.qwen_omni_model

    @property
    def setup_status(self) -> dict[str, bool | str]:
        return {"local_llm": bool(self.llm_base_url), "qwen3_tts": bool(self.qwen_tts_model), "reviewer": self.reviewer_backend, "youtube": bool(self.youtube_oauth), "publishing": self.publish_enabled}

    def validate(self) -> None:
        if self.privacy_status not in {"private", "unlisted", "public"}:
            raise ValueError("invalid YouTube privacy status")
        if self.qwen_tts_mode == "voice_clone" and (not self.qwen_ref_audio or not self.qwen_ref_text):
            raise ValueError("voice_clone requires QWEN_REF_AUDIO and QWEN_REF_TEXT")
        if self.publish_enabled and not self.youtube_oauth:
            raise ValueError("Publishing is enabled but YOUTUBE_OAUTH_JSON is missing")
