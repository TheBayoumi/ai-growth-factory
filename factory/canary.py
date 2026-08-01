from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .feeds import fetch_diverse_recent, fetch_recent
from .llm_runtime import managed_llama_server
from .policy import Strategy
from .render import render_video
from .source_attributed_llm import generate_package
from .video_qc import verify_video_output
from .voice_pipeline import build_reviewed_narration
from .voice_policy import contract_for_strategy


CANARY_STRATEGY = Strategy(
    hook="practical",
    pacing="balanced",
    visual="dashboard",
    duration="55-62",
    cta="subscribe",
)


def _scene_durations(segments: tuple[Any, ...], total_duration: float) -> list[float]:
    ordered = sorted(segments, key=lambda item: item.segment_id)
    return [
        (
            ordered[index + 1].start_seconds
            if index + 1 < len(ordered)
            else total_duration
        )
        - segment.start_seconds
        for index, segment in enumerate(ordered)
    ]


def _copy(path: Path, destination: Path, name: str | None = None) -> Path:
    output = destination / (name or path.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output)
    return output


def run_production_canary(settings: Settings, output_root: Path) -> dict[str, Any]:
    """Execute the real generation stack without publishing to YouTube.

    The canary uses current primary-source research, the managed Qwen script model,
    Qwen3-TTS, the configured perceptual reviewer, the production renderer, and the
    same fail-closed video verifier used by publication.
    """
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    canary_id = f"{stamp}-{hashlib.sha256(stamp.encode()).hexdigest()[:8]}"
    destination = output_root / canary_id
    destination.mkdir(parents=True, exist_ok=False)
    settings.work_root.mkdir(parents=True, exist_ok=True)
    research_dir = Path(tempfile.mkdtemp(prefix="canary-research-", dir=settings.work_root))
    workdir = Path(tempfile.mkdtemp(prefix="canary-run-", dir=settings.work_root))

    try:
        selection = fetch_diverse_recent(
            max_age_hours=settings.max_source_age_hours,
            min_publishers=settings.min_primary_sources,
            fetcher=fetch_recent,
        )
        sources = list(selection.items)
        source_publishers = set(selection.publishers)
        if selection.publisher_count < settings.min_primary_sources:
            raise RuntimeError(
                f"Canary found only {selection.publisher_count} primary publishers "
                f"within {selection.max_age_hours} hours; "
                f"required {settings.min_primary_sources}"
            )

        with managed_llama_server(settings, research_dir):
            package = generate_package(settings, sources, CANARY_STRATEGY)

        if len(set(package.source_publishers)) < settings.min_primary_sources:
            raise RuntimeError("Canary package did not retain enough primary publishers")

        voice_contract = contract_for_strategy(settings.voice_contract, CANARY_STRATEGY)
        voice = build_reviewed_narration(
            settings,
            package.narration,
            workdir,
            voice_contract=voice_contract,
        )
        video_path, thumbnail_path = render_video(
            settings,
            package,
            CANARY_STRATEGY,
            workdir,
            segments=voice.segments,
        )
        scene_durations = _scene_durations(voice.segments, voice.metrics.duration_seconds)
        qc_path = workdir / "video-qc-report.json"
        video_qc = verify_video_output(
            settings,
            video_path,
            thumbnail_path,
            expected_duration=voice.metrics.duration_seconds,
            scene_durations=scene_durations,
            voice_manifest_path=voice.manifest_path,
            require_production_voice=True,
            report_path=qc_path,
        )

        _copy(video_path, destination, "video.mp4")
        _copy(thumbnail_path, destination, "thumbnail.png")
        _copy(voice.audio_path, destination, "narration.wav")
        _copy(voice.manifest_path, destination, "voice-review-manifest.json")
        _copy(qc_path, destination, "video-qc-report.json")
        llama_log = research_dir / "llama-server.log"
        if llama_log.exists():
            _copy(llama_log, destination)

        package_payload = asdict(package)
        package_payload["strategy"] = asdict(CANARY_STRATEGY)
        package_payload["source_feed_publishers"] = sorted(source_publishers)
        package_payload["source_max_age_hours"] = selection.max_age_hours
        (destination / "package.json").write_text(
            json.dumps(package_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        completed_at = datetime.now(timezone.utc)
        result: dict[str, Any] = {
            "status": "verified_render_canary",
            "canary_id": canary_id,
            "artifact_path": f"canaries/{canary_id}",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
            "topic": package.topic,
            "title": package.title,
            "strategy": CANARY_STRATEGY.key,
            "source_urls": package.source_urls,
            "source_max_age_hours": selection.max_age_hours,
            "voice": {
                "generator": settings.qwen_tts_model,
                "reviewer": settings.reviewer_model,
                "attempts": voice.attempts,
                "overall_score": voice.review.overall_score if voice.review else None,
                "metrics": voice.metrics.as_dict(),
                "contract": voice.voice_contract.as_dict(),
            },
            "video_qc": video_qc.as_dict(),
        }
        (destination / "canary-result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result
    except Exception as exc:
        failure = {
            "status": "canary_failed_closed",
            "canary_id": canary_id,
            "artifact_path": f"canaries/{canary_id}",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        (destination / "canary-failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        llama_log = research_dir / "llama-server.log"
        if llama_log.exists():
            _copy(llama_log, destination)
        voice_manifest = workdir / "voice-review-manifest.json"
        if voice_manifest.exists():
            _copy(voice_manifest, destination)
        qc_report = workdir / "video-qc-report.json"
        if qc_report.exists():
            _copy(qc_report, destination)
        return failure
    finally:
        shutil.rmtree(research_dir, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)
