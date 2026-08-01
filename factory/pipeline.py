from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .feeds import fetch_recent, publishers
from .llm_runtime import managed_llama_server
from .local_llm import generate_package
from .policy import reward, select_strategy
from .render import render_video
from .video_qc import verify_video_output
from .voice_pipeline import build_reviewed_narration
from .voice_policy import contract_for_strategy
from .youtube import YouTubeClient


def _record(settings: Settings, payload: dict) -> Path:
    output = settings.state_root / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def run_factory(settings: Settings) -> dict:
    if not settings.publish_enabled:
        return {"status": "setup_required", "setup": settings.setup_status}
    youtube = YouTubeClient(settings)
    context = youtube.channel_context()
    recent = youtube.recent_videos(context)
    daily_tag = "agfd-" + datetime.now(timezone.utc).strftime("%Y%m%d")
    if youtube.already_published(recent, daily_tag):
        return {"status": "idempotent_skip", "reason": "today's video already exists"}
    observations = youtube.observations(recent)
    mature = [item for item in observations if item.age_hours >= 24]
    if len(mature) >= 5:
        last = mature[-5:]
        if sum(reward(item) for item in last) / 5 < 0.20 and sum(item.subscribers_gained - item.subscribers_lost for item in last) < 0:
            result = {"status": "growth_pause", "reason": "five-post safety controller stopped publishing"}
            _record(settings, result)
            return result
    sources = fetch_recent(settings.max_source_age_hours)
    if len(publishers(sources)) < settings.min_primary_sources:
        result = {"status": "evidence_skip", "reason": "not enough fresh primary publishers"}
        _record(settings, result)
        return result
    seed = int(hashlib.sha256(f"{daily_tag}|{context.channel_id}|{len(mature)}".encode()).hexdigest()[:16], 16)
    strategy = select_strategy(observations, seed)
    settings.work_root.mkdir(parents=True, exist_ok=True)
    research = Path(tempfile.mkdtemp(prefix="research-", dir=settings.work_root))
    try:
        with managed_llama_server(settings, research):
            package = generate_package(settings, sources, strategy)
    finally:
        shutil.rmtree(research, ignore_errors=True)
    workdir = Path(tempfile.mkdtemp(prefix="run-", dir=settings.work_root))
    try:
        voice = build_reviewed_narration(settings, package.narration, workdir, contract_for_strategy(settings.voice_contract, strategy))
        video, thumbnail = render_video(settings, package, strategy, workdir, list(voice.segments))
        qc = verify_video_output(settings, video, thumbnail, voice.metrics.duration_seconds, voice.manifest_path)
        video_id = youtube.upload(video, thumbnail, package, strategy, daily_tag)
        result = {"status": "published", "video_id": video_id, "video_url": f"https://www.youtube.com/watch?v={video_id}", "title": package.title, "strategy": strategy.key, "source_urls": package.source_urls, "voice": {"attempts": voice.attempts, "metrics": voice.metrics.as_dict()}, "video_qc": qc.as_dict()}
        result["run_record"] = str(_record(settings, result))
        shutil.rmtree(workdir, ignore_errors=True)
        return result
    except Exception as exc:
        failure = {"status": "failed_closed", "error_type": type(exc).__name__, "error": str(exc), "workdir": str(workdir)}
        failure["run_record"] = str(_record(settings, failure))
        raise RuntimeError(json.dumps(failure)) from exc
