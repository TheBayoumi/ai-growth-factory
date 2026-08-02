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
from .source_attributed_llm import generate_package
from .trend_ranking import align_primary_sources_to_trends
from .trend_sources import fetch_trend_snapshot
from .video_qc import verify_video_output
from .visual_pipeline import release_accelerator_memory, render_visual_plan
from .visual_prompt import construct_visual_plan
from .voice_pipeline import build_reviewed_narration
from .voice_policy import contract_for_strategy


CANARY_STRATEGY = Strategy(
    hook="practical",
    pacing="balanced",
    visual="cinematic_editorial",
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


def _copy_if_exists(path: Path, destination: Path, name: str | None = None) -> None:
    if path.is_file():
        _copy(path, destination, name)


def _copy_keyframes(workdir: Path, destination: Path) -> None:
    source = workdir / "visual-assets" / "keyframes"
    if not source.is_dir():
        return
    target = destination / "visual-keyframes"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.png")):
        _copy(path, target)


def _copy_visual_audit(workdir: Path, destination: Path) -> None:
    visual_root = workdir / "visual-assets"
    candidates = (
        visual_root / "visual-plan.json",
        visual_root / "visual-pipeline-manifest.json",
        visual_root / "keyframes" / "keyframe-manifest.json",
        visual_root / "scene-media" / "scene-media-manifest.json",
        visual_root / "render" / "animated-captions.ass",
        visual_root / "render" / "animated-captions.json",
        visual_root / "render" / "visual-composition-manifest.json",
        visual_root / "render" / "visual-compositor.log",
    )
    for path in candidates:
        _copy_if_exists(path, destination, path.name)
    _copy_keyframes(workdir, destination)


def run_production_canary(settings: Settings, output_root: Path) -> dict[str, Any]:
    """Execute live trend discovery plus the real voice, visual, caption, and QC stack."""
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
        source_publishers = set(selection.publishers)
        if selection.publisher_count < settings.min_primary_sources:
            raise RuntimeError(
                f"Canary found only {selection.publisher_count} primary publishers "
                f"within {selection.max_age_hours} hours; required {settings.min_primary_sources}"
            )

        trend_snapshot = fetch_trend_snapshot(
            max_age_hours=min(settings.max_source_age_hours, 72),
        )
        trend_alignment = align_primary_sources_to_trends(selection.items, trend_snapshot)
        sources = list(trend_alignment.ranked_sources or selection.items)
        trend_payload = trend_snapshot.as_dict()
        trend_payload["alignment"] = trend_alignment.as_dict()
        (destination / "trend-snapshot.json").write_text(
            json.dumps(trend_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        with managed_llama_server(settings, research_dir):
            package = generate_package(settings, sources, CANARY_STRATEGY)
            visual_plan = construct_visual_plan(
                settings,
                package,
                sources,
                CANARY_STRATEGY,
            )

        if len(set(package.source_publishers)) < settings.min_primary_sources:
            raise RuntimeError("Canary package did not retain enough primary publishers")

        voice_contract = contract_for_strategy(settings.voice_contract, CANARY_STRATEGY)
        voice = build_reviewed_narration(
            settings,
            package.narration,
            workdir,
            voice_contract=voice_contract,
        )
        release_accelerator_memory()
        visual = render_visual_plan(
            plan=visual_plan,
            package=package,
            segments=voice.segments,
            audio_path=voice.audio_path,
            workdir=workdir,
            output_width=settings.width,
            output_height=settings.height,
            output_fps=settings.fps,
        )
        scene_durations = _scene_durations(voice.segments, voice.metrics.duration_seconds)
        qc_path = workdir / "video-qc-report.json"
        video_qc = verify_video_output(
            settings,
            visual.video_path,
            visual.thumbnail_path,
            expected_duration=voice.metrics.duration_seconds,
            scene_durations=scene_durations,
            voice_manifest_path=voice.manifest_path,
            require_production_voice=True,
            report_path=qc_path,
        )

        _copy(visual.video_path, destination, "video.mp4")
        _copy(visual.thumbnail_path, destination, "thumbnail.png")
        _copy(voice.audio_path, destination, "narration.wav")
        _copy(voice.manifest_path, destination, "voice-review-manifest.json")
        _copy(qc_path, destination, "video-qc-report.json")
        _copy_visual_audit(workdir, destination)
        llama_log = research_dir / "llama-server.log"
        if llama_log.exists():
            _copy(llama_log, destination)

        package_payload = asdict(package)
        package_payload["strategy"] = asdict(CANARY_STRATEGY)
        package_payload["source_feed_publishers"] = sorted(source_publishers)
        package_payload["source_max_age_hours"] = selection.max_age_hours
        package_payload["trend_signal_count"] = len(trend_snapshot.items)
        package_payload["trend_provider_status"] = dict(trend_snapshot.provider_status)
        package_payload["trend_match_count"] = len(trend_alignment.matches)
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
            "trends": {
                "signal_count": len(trend_snapshot.items),
                "provider_status": dict(trend_snapshot.provider_status),
                "matched_primary_count": len(trend_alignment.matches),
                "artifact": "trend-snapshot.json",
            },
            "voice": {
                "generator": settings.qwen_tts_model,
                "reviewer": settings.reviewer_model,
                "attempts": voice.attempts,
                "overall_score": voice.review.overall_score if voice.review else None,
                "metrics": voice.metrics.as_dict(),
                "contract": voice.voice_contract.as_dict(),
            },
            "visuals": {
                "prompt_version": visual_plan.prompt_version,
                "image_model": visual_plan.image_model,
                "video_model": visual_plan.video_model,
                "wan_scene_count": sum(
                    scene.generation_mode == "wan_i2v" for scene in visual_plan.scenes
                ),
                "caption_rendering": "separate_animated_ass_layer",
                "captions_baked_into_generated_media": False,
                "director_input_sha256": visual_plan.director_input_sha256,
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
        _copy_visual_audit(workdir, destination)
        return failure
    finally:
        shutil.rmtree(research_dir, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)
