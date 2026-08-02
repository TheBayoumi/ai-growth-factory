from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .feeds import fetch_diverse_recent, fetch_recent
from .llm_runtime import managed_llama_server
from .policy import reward, select_strategy
from .source_attributed_llm import generate_package
from .trend_ranking import TrendAlignment, align_primary_sources_to_trends
from .trend_sources import TrendSnapshot, fetch_trend_snapshot
from .video_qc import verify_video_output
from .visual_pipeline import release_accelerator_memory, render_visual_plan
from .visual_prompt import construct_visual_plan
from .voice_pipeline import build_reviewed_narration
from .voice_policy import contract_for_strategy
from .youtube import YouTubeClient


def _persist_run_record(settings: Settings, record: dict[str, Any]) -> Path:
    settings.state_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = settings.state_root / "runs" / f"{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _persist_trend_audit(
    settings: Settings,
    snapshot: TrendSnapshot,
    alignment: TrendAlignment,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = settings.state_root / "trend-audits" / f"{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.as_dict()
    payload["alignment"] = alignment.as_dict()
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _persist_visual_audit(settings: Settings, workdir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = settings.state_root / "visual-audits" / timestamp
    destination.mkdir(parents=True, exist_ok=False)
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
    for source in candidates:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    keyframes = visual_root / "keyframes"
    if keyframes.is_dir():
        keyframe_destination = destination / "keyframes"
        keyframe_destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(keyframes.glob("*.png")):
            shutil.copy2(source, keyframe_destination / source.name)
    return destination


def run_factory(settings: Settings) -> dict[str, Any]:
    if not settings.publish_enabled:
        return {
            "status": "setup_required",
            "message": (
                "The local worker is installed but publishing is disabled until the local model, "
                "reviewer, open visual models, and YouTube authorization are configured."
            ),
            "setup": settings.setup_status,
        }

    youtube = YouTubeClient(settings)
    context = youtube.channel_context()
    recent = youtube.recent_videos(context)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    daily_tag = f"agfd-{today}"
    if youtube.already_published(recent, daily_tag):
        return {"status": "idempotent_skip", "reason": "Today's video already exists"}

    observations = youtube.observations(recent)
    mature = [observation for observation in observations if observation.age_hours >= 24]
    last_five = mature[-5:]
    if len(last_five) == 5:
        mean_reward = sum(reward(item) for item in last_five) / 5
        net_subscribers = sum(item.subscribers_gained - item.subscribers_lost for item in last_five)
        if mean_reward < 0.20 and net_subscribers < 0:
            result = {
                "status": "growth_pause",
                "mean_reward": round(mean_reward, 4),
                "net_subscribers": net_subscribers,
                "reason": "Five-post safety controller stopped publishing to prevent channel damage",
            }
            _persist_run_record(settings, result)
            return result

    selection = fetch_diverse_recent(
        max_age_hours=settings.max_source_age_hours,
        min_publishers=settings.min_primary_sources,
        fetcher=fetch_recent,
    )
    if selection.publisher_count < settings.min_primary_sources:
        result = {
            "status": "evidence_skip",
            "reason": "Not enough fresh primary-source publishers",
            "source_count": len(selection.items),
            "publisher_count": selection.publisher_count,
            "source_max_age_hours": selection.max_age_hours,
        }
        _persist_run_record(settings, result)
        return result

    trend_snapshot = fetch_trend_snapshot(
        max_age_hours=min(settings.max_source_age_hours, 72),
    )
    trend_alignment = align_primary_sources_to_trends(selection.items, trend_snapshot)
    sources = list(trend_alignment.ranked_sources or selection.items)
    trend_audit = _persist_trend_audit(settings, trend_snapshot, trend_alignment)

    seed_material = f"{today}|{context.channel_id}|{len(mature)}"
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    strategy = select_strategy(observations, seed)
    settings.work_root.mkdir(parents=True, exist_ok=True)
    research_workdir = Path(tempfile.mkdtemp(prefix="research-", dir=settings.work_root))
    try:
        with managed_llama_server(settings, research_workdir):
            package = generate_package(settings, sources, strategy)
            visual_plan = construct_visual_plan(settings, package, sources, strategy)
    finally:
        shutil.rmtree(research_workdir, ignore_errors=True)
    if len(set(package.source_publishers)) < settings.min_primary_sources:
        result = {
            "status": "evidence_skip",
            "reason": "Generated package did not use enough independent primary publishers",
            "source_max_age_hours": selection.max_age_hours,
            "trend_audit": str(trend_audit),
        }
        _persist_run_record(settings, result)
        return result

    workdir = Path(tempfile.mkdtemp(prefix="run-", dir=settings.work_root))
    succeeded = False
    try:
        voice_contract = contract_for_strategy(settings.voice_contract, strategy)
        voice = build_reviewed_narration(
            settings, package.narration, workdir, voice_contract=voice_contract
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
        ordered_segments = sorted(voice.segments, key=lambda item: item.segment_id)
        scene_durations = [
            (
                ordered_segments[index + 1].start_seconds
                if index + 1 < len(ordered_segments)
                else voice.metrics.duration_seconds
            )
            - segment.start_seconds
            for index, segment in enumerate(ordered_segments)
        ]
        video_qc = verify_video_output(
            settings,
            visual.video_path,
            visual.thumbnail_path,
            expected_duration=voice.metrics.duration_seconds,
            scene_durations=scene_durations,
            voice_manifest_path=voice.manifest_path,
            require_production_voice=True,
            report_path=workdir / "video-qc-report.json",
        )
        visual_audit = _persist_visual_audit(settings, workdir)
        video_id = youtube.upload(
            video_path=visual.video_path,
            thumbnail_path=visual.thumbnail_path,
            package=package,
            strategy=strategy,
            daily_tag=daily_tag,
        )
        result = {
            "status": "published",
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": package.title,
            "strategy": strategy.key,
            "strategy_tag": strategy.tag,
            "source_urls": package.source_urls,
            "source_max_age_hours": selection.max_age_hours,
            "mature_observations": len(mature),
            "trends": {
                "signal_count": len(trend_snapshot.items),
                "provider_status": dict(trend_snapshot.provider_status),
                "matched_primary_count": len(trend_alignment.matches),
                "audit_path": str(trend_audit),
            },
            "voice": {
                "generator": settings.qwen_tts_model,
                "reviewer": settings.reviewer_model if settings.reviewer_required else None,
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
                "audit_path": str(visual_audit),
            },
            "video_qc": video_qc.as_dict(),
        }
        record = _persist_run_record(settings, result)
        result["run_record"] = str(record)
        succeeded = True
        return result
    except Exception as exc:
        failure = {
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "workdir": str(workdir),
            "trend_audit": str(trend_audit),
        }
        try:
            failure["visual_audit"] = str(_persist_visual_audit(settings, workdir))
        except Exception:
            pass
        record = _persist_run_record(settings, failure)
        failure["run_record"] = str(record)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False)) from exc
    finally:
        if succeeded:
            shutil.rmtree(workdir, ignore_errors=True)
