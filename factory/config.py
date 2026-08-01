from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import VoiceContract


def _load_env_file() -> None:
    path = Path(os.getenv("ENV_FILE", ".env")).expanduser()
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {"\"", "'"} and value[-1:] == value[0]:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _json_object(name: str) -> dict[str, Any] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _voice_contract() -> VoiceContract:
    path_raw = os.getenv("VOICE_CONTRACT_FILE", "").strip()
    json_raw = os.getenv("VOICE_CONTRACT_JSON", "").strip()
    if path_raw and json_raw:
        raise ValueError("Set only one of VOICE_CONTRACT_FILE or VOICE_CONTRACT_JSON")
    if path_raw:
        data = json.loads(Path(path_raw).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("VOICE_CONTRACT_FILE must contain a JSON object")
        return VoiceContract.from_dict(data)
    if json_raw:
        data = json.loads(json_raw)
        if not isinstance(data, dict):
            raise ValueError("VOICE_CONTRACT_JSON must contain a JSON object")
        return VoiceContract.from_dict(data)
    return VoiceContract()


@dataclass(frozen=True)
class Settings:
    youtube_oauth: dict[str, Any] | None
    openai_api_key: str | None
    cron_secret: str | None
    factory_secret: str | None
    publish_enabled: bool
    reviewer_required: bool
    reviewer_backend: str
    privacy_status: str
    language: str
    region: str
    timezone: str
    monetization_url: str | None
    monetization_label: str

    llm_base_url: str
    llm_model: str
    llm_api_key: str | None
    llm_timeout_seconds: int
    llm_temperature: float
    llm_managed: bool
    llm_executable: str
    llm_hf_model: str
    llm_context_tokens: int
    llm_gpu_layers: int
    llm_startup_timeout_seconds: int

    tts_backend: str
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

    openai_reviewer_model: str
    reviewer_reasoning_effort: str
    qwen_omni_model: str
    qwen_omni_device: str
    qwen_omni_dtype: str
    qwen_omni_attention: str
    qwen_omni_max_new_tokens: int
    reviewer_max_attempts: int
    reviewer_http_timeout_seconds: int
    reviewer_overall_threshold: float
    reviewer_fidelity_threshold: float
    reviewer_naturalness_threshold: float
    reviewer_pronunciation_threshold: float

    audio_target_lufs: float
    audio_peak_limit_dbfs: float
    audio_min_rms_dbfs: float
    audio_max_clipping_ratio: float
    audio_max_silence_ratio: float
    audio_max_silence_seconds: float
    audio_wpm_tolerance: int
    audio_segment_pause_ms: int
    narration_segments: int

    target_seconds: int
    max_recent_videos: int
    min_primary_sources: int
    max_source_age_hours: int
    width: int
    height: int
    fps: int
    work_root: Path
    state_root: Path

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file()
        oauth = _json_object("YOUTUBE_OAUTH_JSON")
        privacy = os.getenv("YOUTUBE_PRIVACY_STATUS", "private").strip().lower()
        if privacy not in {"private", "unlisted", "public"}:
            raise ValueError("YOUTUBE_PRIVACY_STATUS must be private, unlisted, or public")
        tts_mode = os.getenv("QWEN_TTS_MODE", "custom_voice").strip().lower()
        if tts_mode not in {"custom_voice", "voice_clone"}:
            raise ValueError("QWEN_TTS_MODE must be custom_voice or voice_clone")
        reasoning = os.getenv("OPENAI_REVIEW_REASONING", "medium").strip().lower()
        if reasoning not in {"minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("OPENAI_REVIEW_REASONING must be minimal, low, medium, high, or xhigh")
        reviewer_backend = os.getenv("REVIEWER_BACKEND", "qwen_omni").strip().lower()
        if reviewer_backend not in {"qwen_omni", "openai", "disabled"}:
            raise ValueError("REVIEWER_BACKEND must be qwen_omni, openai, or disabled")
        reviewer_required = _bool("REVIEWER_REQUIRED", reviewer_backend != "disabled")
        if reviewer_backend == "disabled":
            reviewer_required = False

        settings = cls(
            youtube_oauth=oauth,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            cron_secret=os.getenv("CRON_SECRET") or None,
            factory_secret=os.getenv("FACTORY_SECRET") or None,
            publish_enabled=_bool("PUBLISH_ENABLED", False),
            reviewer_required=reviewer_required,
            reviewer_backend=reviewer_backend,
            privacy_status=privacy,
            language=os.getenv("CHANNEL_LANGUAGE", "en").strip().lower(),
            region=os.getenv("CHANNEL_REGION", "EG").strip().upper(),
            timezone=os.getenv("TIMEZONE_NAME", "Africa/Cairo").strip(),
            monetization_url=os.getenv("MONETIZATION_URL") or None,
            monetization_label=os.getenv("MONETIZATION_LABEL", "Get the full implementation"),
            llm_base_url=os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/"),
            llm_model=os.getenv("LLAMA_CPP_MODEL", "Qwen3-4B-Q4_K_M.gguf"),
            llm_api_key=os.getenv("LLAMA_CPP_API_KEY") or None,
            llm_timeout_seconds=_int("LLAMA_CPP_TIMEOUT_SECONDS", 240, 30, 900),
            llm_temperature=_float("LLAMA_CPP_TEMPERATURE", 0.45, 0.0, 1.5),
            llm_managed=_bool("LLAMA_CPP_MANAGED", True),
            llm_executable=os.getenv("LLAMA_CPP_EXECUTABLE", "llama-server").strip(),
            llm_hf_model=os.getenv("LLAMA_CPP_HF_MODEL", "Qwen/Qwen3-4B-GGUF:Q4_K_M").strip(),
            llm_context_tokens=_int("LLAMA_CPP_CONTEXT_TOKENS", 16384, 4096, 65536),
            llm_gpu_layers=_int("LLAMA_CPP_GPU_LAYERS", 0, 0, 999),
            llm_startup_timeout_seconds=_int("LLAMA_CPP_STARTUP_TIMEOUT_SECONDS", 240, 30, 900),
            tts_backend=os.getenv("TTS_BACKEND", "qwen3").strip().lower(),
            qwen_tts_model=os.getenv(
                "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            ).strip(),
            qwen_tts_mode=tts_mode,
            qwen_tts_speaker=os.getenv("QWEN_TTS_SPEAKER", "Ryan").strip(),
            qwen_tts_language=os.getenv("QWEN_TTS_LANGUAGE", "English").strip(),
            qwen_tts_device=os.getenv("QWEN_TTS_DEVICE", "cuda:0").strip(),
            qwen_tts_dtype=os.getenv("QWEN_TTS_DTYPE", "float16").strip().lower(),
            qwen_tts_attention=os.getenv("QWEN_TTS_ATTENTION", "sdpa").strip(),
            qwen_ref_audio=Path(os.environ["QWEN_REF_AUDIO"]).expanduser()
            if os.getenv("QWEN_REF_AUDIO")
            else None,
            qwen_ref_text=os.getenv("QWEN_REF_TEXT") or None,
            voice_contract=_voice_contract(),
            openai_reviewer_model=os.getenv("OPENAI_REVIEW_MODEL", "gpt-realtime-2.1").strip(),
            reviewer_reasoning_effort=reasoning,
            qwen_omni_model=os.getenv(
                "QWEN_OMNI_REVIEW_MODEL", "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
            ).strip(),
            qwen_omni_device=os.getenv("QWEN_OMNI_DEVICE", "cuda:0").strip(),
            qwen_omni_dtype=os.getenv("QWEN_OMNI_DTYPE", "float16").strip().lower(),
            qwen_omni_attention=os.getenv("QWEN_OMNI_ATTENTION", "sdpa").strip(),
            qwen_omni_max_new_tokens=_int("QWEN_OMNI_MAX_NEW_TOKENS", 900, 256, 2400),
            reviewer_max_attempts=_int("VOICE_REVIEW_MAX_ATTEMPTS", 3, 1, 5),
            reviewer_http_timeout_seconds=_int("VOICE_REVIEW_TIMEOUT_SECONDS", 180, 30, 600),
            reviewer_overall_threshold=_float("REVIEW_OVERALL_THRESHOLD", 0.87, 0.5, 1.0),
            reviewer_fidelity_threshold=_float("REVIEW_FIDELITY_THRESHOLD", 0.98, 0.5, 1.0),
            reviewer_naturalness_threshold=_float("REVIEW_NATURALNESS_THRESHOLD", 0.85, 0.5, 1.0),
            reviewer_pronunciation_threshold=_float("REVIEW_PRONUNCIATION_THRESHOLD", 0.92, 0.5, 1.0),
            audio_target_lufs=_float("AUDIO_TARGET_LUFS", -16.0, -24.0, -9.0),
            audio_peak_limit_dbfs=_float("AUDIO_PEAK_LIMIT_DBFS", -1.0, -6.0, -0.1),
            audio_min_rms_dbfs=_float("AUDIO_MIN_RMS_DBFS", -32.0, -50.0, -10.0),
            audio_max_clipping_ratio=_float("AUDIO_MAX_CLIPPING_RATIO", 0.0005, 0.0, 0.05),
            audio_max_silence_ratio=_float("AUDIO_MAX_SILENCE_RATIO", 0.28, 0.0, 0.8),
            audio_max_silence_seconds=_float("AUDIO_MAX_SILENCE_SECONDS", 1.25, 0.1, 10.0),
            audio_wpm_tolerance=_int("AUDIO_WPM_TOLERANCE", 32, 5, 80),
            audio_segment_pause_ms=_int("AUDIO_SEGMENT_PAUSE_MS", 140, 0, 1000),
            narration_segments=_int("NARRATION_SEGMENTS", 6, 1, 12),
            target_seconds=_int("TARGET_SECONDS", 62, 45, 95),
            max_recent_videos=_int("MAX_RECENT_VIDEOS", 30, 5, 50),
            min_primary_sources=_int("MIN_PRIMARY_SOURCES", 2, 2, 6),
            max_source_age_hours=_int("MAX_SOURCE_AGE_HOURS", 48, 12, 168),
            width=_int("VIDEO_WIDTH", 720, 360, 1080),
            height=_int("VIDEO_HEIGHT", 1280, 640, 1920),
            fps=_int("VIDEO_FPS", 24, 15, 30),
            work_root=Path(os.getenv("WORK_ROOT", "/tmp/ai-growth-factory")).expanduser(),
            state_root=Path(os.getenv("STATE_ROOT", "./state")).expanduser(),
        )
        settings.validate()
        return settings

    @property
    def reviewer_model(self) -> str:
        if self.reviewer_backend == "openai":
            return self.openai_reviewer_model
        if self.reviewer_backend == "qwen_omni":
            return self.qwen_omni_model
        return "disabled"

    def validate(self) -> None:
        if self.tts_backend != "qwen3":
            raise ValueError("TTS_BACKEND currently supports only qwen3")
        if self.qwen_tts_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("QWEN_TTS_DTYPE must be float16, bfloat16, or float32")
        if self.qwen_omni_dtype not in {"float16", "bfloat16", "float32", "auto"}:
            raise ValueError("QWEN_OMNI_DTYPE must be float16, bfloat16, float32, or auto")
        if self.qwen_tts_mode == "voice_clone":
            if not self.qwen_ref_audio:
                raise ValueError("QWEN_REF_AUDIO is required for voice_clone mode")
            if not self.qwen_ref_audio.exists():
                raise ValueError(f"QWEN_REF_AUDIO does not exist: {self.qwen_ref_audio}")
            if not self.qwen_ref_text:
                raise ValueError("QWEN_REF_TEXT is required for high-quality voice_clone mode")
        if self.youtube_oauth:
            required = {"client_id", "client_secret", "refresh_token"}
            missing_oauth = sorted(required - set(self.youtube_oauth))
            if missing_oauth:
                raise ValueError("YOUTUBE_OAUTH_JSON missing: " + ", ".join(missing_oauth))
        if self.publish_enabled:
            missing: list[str] = []
            if not self.youtube_oauth:
                missing.append("YOUTUBE_OAUTH_JSON")
            if self.reviewer_required and self.reviewer_backend == "openai" and not self.openai_api_key:
                missing.append("OPENAI_API_KEY")
            if missing:
                raise ValueError("Publishing is enabled but missing: " + ", ".join(missing))

    @property
    def setup_status(self) -> dict[str, bool | str]:
        if not self.reviewer_required:
            reviewer: bool | str = "disabled"
        elif self.reviewer_backend == "openai":
            reviewer = bool(self.openai_api_key)
        else:
            reviewer = bool(self.qwen_omni_model)
        return {
            "local_llm": bool(self.llm_base_url),
            "qwen3_tts": bool(self.qwen_tts_model),
            "reviewer_backend": self.reviewer_backend,
            "reviewer": reviewer,
            "youtube": bool(self.youtube_oauth),
            "publishing": self.publish_enabled,
        }
