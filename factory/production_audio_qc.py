from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


_INSTALLED = False
_PEAK_HEADROOM_DB = 0.5


def production_peak_target(peak_limit_dbfs: float) -> float:
    """Return a stricter normalization target while preserving the acceptance gate."""
    return peak_limit_dbfs - _PEAK_HEADROOM_DB


def normalize_audio_with_headroom(
    input_path: Path,
    output_path: Path,
    *,
    target_lufs: float,
    peak_dbfs: float,
    normalizer: Callable[..., Path],
) -> Path:
    """Normalize against a conservative target to absorb single-pass loudnorm drift."""
    return normalizer(
        input_path,
        output_path,
        target_lufs=target_lufs,
        peak_dbfs=production_peak_target(peak_dbfs),
    )


def deterministic_qc_failure(exc: Exception) -> Exception:
    """Prefer final deterministic QC evidence over stale reviewer feedback."""
    manifest_path = getattr(exc, "manifest_path", None)
    if not isinstance(manifest_path, Path) or not manifest_path.is_file():
        return exc
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return exc
    metrics = payload.get("metrics") or {}
    failures = metrics.get("failures") or []
    if not isinstance(failures, list) or not failures:
        return exc

    from .voice_pipeline import VoiceGenerationError

    attempts = payload.get("attempts") or "bounded"
    reason = "; ".join(str(item).strip() for item in failures if str(item).strip())
    if not reason:
        return exc
    return VoiceGenerationError(
        f"Narration failed deterministic QC after {attempts} attempts: {reason}. "
        f"Manifest: {manifest_path}",
        manifest_path=manifest_path,
        failed_segments=(),
    )


def install_production_audio_qc() -> None:
    """Install peak-safe normalization and truthful final voice diagnostics."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import audio_qc

    original_normalize = audio_qc.normalize_audio

    def normalize_with_headroom(
        input_path: Path,
        output_path: Path,
        *,
        target_lufs: float,
        peak_dbfs: float,
    ) -> Path:
        return normalize_audio_with_headroom(
            input_path,
            output_path,
            target_lufs=target_lufs,
            peak_dbfs=peak_dbfs,
            normalizer=original_normalize,
        )

    audio_qc.normalize_audio = normalize_with_headroom

    from . import voice_pipeline

    voice_pipeline.normalize_audio = normalize_with_headroom
    original_build = voice_pipeline.build_reviewed_narration

    def build_with_truthful_failure(*args: Any, **kwargs: Any):
        try:
            return original_build(*args, **kwargs)
        except voice_pipeline.VoiceGenerationError as exc:
            rewritten = deterministic_qc_failure(exc)
            if rewritten is exc:
                raise
            raise rewritten from exc

    voice_pipeline.build_reviewed_narration = build_with_truthful_failure
    _INSTALLED = True
