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
from .feeds import (
    SourceItem,
    fetch_diverse_recent,
    fetch_recent,
    hydrate_source_summaries,
    source_authority,
)
from .llm_runtime import managed_llama_server
from .policy import Strategy
from .source_attributed_llm import generate_package
from .trend_ranking import align_primary_sources_to_trends
from .trend_sources import fetch_trend_snapshot
from .video_qc import verify_video_output
from .visual_pipeline import release_accelerator_memory, render_visual_plan
from .visual_prompt import construct_visual_plan
from .voice_pipeline import VoiceGenerationError, build_reviewed_narration
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


def _copy_scene_media(workdir: Path, destination: Path) -> None:
    source = workdir / "visual-assets" / "scene-media"
    if not source.is_dir():
        return
    target = destination / "visual-scene-media"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.mp4")):
        _copy(path, target)


def _copy_voice_diagnostics(workdir: Path, destination: Path) -> None:
    """Persist the final reviewed assets even when voice generation raises.

    VoiceGenerationError is raised before a VoicePipelineResult exists. The manifest and
    WAVs still live in the temporary work directory, so they must be copied before the
    canary's finally block removes that directory.
    """
    manifest = workdir / "voice-review-manifest.json"
    if not manifest.is_file():
        return
    _copy(manifest, destination, "voice-review-manifest.json")

    segment_target = destination / "voice-segments"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    copied: set[Path] = set()
    for item in payload.get("segments") or []:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("audio_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        source = Path(raw_path)
        if source.is_file() and source not in copied:
            _copy(source, segment_target)
            copied.add(source)

    review_wavs = sorted(workdir.glob("voice-review-attempt-*.wav"))
    if review_wavs:
        _copy(review_wavs[-1], destination, "voice-review-final.wav")
    normalized_wavs = sorted(
        {
            *workdir.glob("voice-normalized-attempt-*.wav"),
            *workdir.glob("voice-tempo-normalized-attempt-*.wav"),
        }
    )
    if normalized_wavs:
        _copy(normalized_wavs[-1], destination, "narration-final-attempt.wav")


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
        visual_root / "render" / "background-music.wav",
    )
    for path in candidates:
        _copy_if_exists(path, destination, path.name)
    _copy_keyframes(workdir, destination)
    _copy_scene_media(workdir, destination)


def _write_package(
    destination: Path,
    package: Any,
    *,
    sources: list[SourceItem],
    source_publishers: set[str],
    source_max_age_hours: int,
    trend_snapshot: Any,
    trend_alignment: Any,
) -> Path:
    package_payload = asdict(package)
    package_payload["strategy"] = asdict(CANARY_STRATEGY)
    package_payload["source_feed_publishers"] = sorted(source_publishers)
    sources_by_url = {source.url: source for source in sources}
    package_payload["source_evidence"] = [
        {
            "url": source.url,
            "publisher": source.publisher,
            "author": source.author,
            "authority": source_authority(source),
            "title": source.title,
            "published_at": source.published_at.isoformat(),
            "summary": source.summary[:800],
        }
        for url in package.source_urls
        if (source := sources_by_url.get(url)) is not None
    ]
    package_payload["source_max_age_hours"] = source_max_age_hours
    package_payload["trend_signal_count"] = len(trend_snapshot.items)
    package_payload["trend_provider_status"] = dict(trend_snapshot.provider_status)
    package_payload["trend_match_count"] = len(trend_alignment.matches)
    output = destination / "package.json"
    output.write_text(
        json.dumps(package_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def _persist_generated_bundle(
    *,
    destination: Path,
    workdir: Path,
    package: Any | None,
    voice: Any | None,
    visual: Any | None,
    sources: list[SourceItem],
    source_publishers: set[str],
    source_max_age_hours: int | None,
    trend_snapshot: Any | None,
    trend_alignment: Any | None,
) -> None:
    if visual is not None:
        _copy_if_exists(visual.video_path, destination, "video.mp4")
        _copy_if_exists(visual.thumbnail_path, destination, "thumbnail.png")
        _copy_if_exists(visual.caption_path, destination, "animated-captions.ass")
    if voice is not None:
        _copy_if_exists(voice.audio_path, destination, "narration.wav")
        _copy_if_exists(voice.manifest_path, destination, "voice-review-manifest.json")
    _copy_voice_diagnostics(workdir, destination)
    if (
        package is not None
        and source_max_age_hours is not None
        and trend_snapshot is not None
        and trend_alignment is not None
    ):
        _write_package(
            destination,
            package,
            sources=sources,
            source_publishers=source_publishers,
            source_max_age_hours=source_max_age_hours,
            trend_snapshot=trend_snapshot,
            trend_alignment=trend_alignment,
        )
    _copy_visual_audit(workdir, destination)


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

    package = None
    voice = None
    visual = None
    selection = None
    trend_snapshot = None
    trend_alignment = None
    source_publishers: set[str] = set()
    sources: list[SourceItem] = []

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
        sources = hydrate_source_summaries(
            trend_alignment.ranked_sources or selection.items,
        )
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

        _persist_generated_bundle(
            destination=destination,
            workdir=workdir,
            package=package,
            voice=voice,
            visual=visual,
            sources=sources,
            source_publishers=source_publishers,
            source_max_age_hours=selection.max_age_hours,
            trend_snapshot=trend_snapshot,
            trend_alignment=trend_alignment,
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
        _copy(qc_path, destination, "video-qc-report.json")

        llama_log = research_dir / "llama-server.log"
        if llama_log.exists():
            _copy(llama_log, destination)

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
        failure: dict[str, Any] = {
            "status": "canary_failed_closed",
            "canary_id": canary_id,
            "artifact_path": f"canaries/{canary_id}",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(exc, VoiceGenerationError):
            failure["voice_manifest"] = "voice-review-manifest.json"
            failure["failed_segments"] = [
                {
                    "segment_id": item.segment_id,
                    "reason": item.reason,
                    "tts_instruction": item.tts_instruction,
                }
                for item in exc.failed_segments
            ]
        (destination / "canary-failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _persist_generated_bundle(
            destination=destination,
            workdir=workdir,
            package=package,
            voice=voice,
            visual=visual,
            sources=sources,
            source_publishers=source_publishers,
            source_max_age_hours=selection.max_age_hours if selection is not None else None,
            trend_snapshot=trend_snapshot,
            trend_alignment=trend_alignment,
        )
        llama_log = research_dir / "llama-server.log"
        if llama_log.exists():
            _copy(llama_log, destination)
        qc_report = workdir / "video-qc-report.json"
        if qc_report.exists():
            _copy(qc_report, destination)
        return failure
    finally:
        shutil.rmtree(research_dir, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)
