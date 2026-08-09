from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .image_generator import KeyframeAsset
from .models import NarrationSegment, Scene, VideoPackage
from .visual_prompt import SceneVisualPrompt, VisualPlan


_CHECKPOINT_SCHEMA = "vimax-hitl-checkpoint-v71"
_REQUIRED_CHECKPOINT_FILES = (
    "package.json",
    "narration.wav",
    "voice-review-manifest.json",
    "visual-plan.json",
    "vimax-plan.json",
    "keyframe-manifest.json",
    "production-preflight.json",
)
_ALLOWED_MACHINE_DISPOSITIONS = {"passed", "advisory_human_arbitration"}


class HumanCheckpointApprovalError(RuntimeError):
    """Raised before Wan when the human-approved preproduction checkpoint is not exact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanCheckpointApprovalError(f"invalid checkpoint JSON {Path(path).name}: {exc}") from exc
    if not isinstance(value, dict):
        raise HumanCheckpointApprovalError(f"checkpoint JSON must be an object: {Path(path).name}")
    return value


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_keyframe_set_digest(dossier: dict[str, Any]) -> str | None:
    records: list[str] = []
    for item in dossier.get("shots") or []:
        if not isinstance(item, dict):
            return None
        digest = str(item.get("keyframe_sha256") or "").strip().lower()
        if not digest:
            return None
        records.append(f"{int(item.get('shot_id', -1)):04d}:{digest}")
    if not records:
        return None
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def finalize_hitl_checkpoint_v71(checkpoint_dir: Path, *, code_sha: str) -> dict[str, Any]:
    """Seal the exact complete pre-Wan state that a human is asked to approve or reject."""
    root = Path(checkpoint_dir)
    code_sha = str(code_sha or "").strip()
    if not code_sha:
        raise HumanCheckpointApprovalError("HITL checkpoint requires the exact code SHA")
    dossier_path = root / "keyframe-human-review-dossier.json"
    dossier = _read_json(dossier_path)
    disposition = str(dossier.get("machine_review_disposition") or "").strip()
    if dossier.get("status") != "awaiting_human_keyframe_review" or disposition not in _ALLOWED_MACHINE_DISPOSITIONS:
        raise HumanCheckpointApprovalError(
            "only a complete keyframe set with passed or advisory machine review can enter human arbitration"
        )

    files: dict[str, str] = {}
    for name in _REQUIRED_CHECKPOINT_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise HumanCheckpointApprovalError(f"required HITL checkpoint file is missing: {name}")
        files[name] = _sha256(path)

    keyframe_dir = root / "visual-keyframes"
    keyframes = sorted(keyframe_dir.glob("scene-*-keyframe.png"))
    expected = int(dossier.get("expected_shots") or 0)
    if expected < 1 or len(keyframes) != expected or int(dossier.get("realized_keyframes") or 0) != expected:
        raise HumanCheckpointApprovalError(
            f"HITL checkpoint requires {expected} exact keyframes; found {len(keyframes)}"
        )
    keyframe_hashes = {path.name: _sha256(path) for path in keyframes}
    dossier_hashes = {
        str(item.get("keyframe")): str(item.get("keyframe_sha256"))
        for item in dossier.get("shots") or []
        if isinstance(item, dict) and item.get("keyframe")
    }
    for name, digest in keyframe_hashes.items():
        if dossier_hashes.get(name) != digest:
            raise HumanCheckpointApprovalError(f"human dossier keyframe digest mismatch: {name}")
    set_digest = _ordered_keyframe_set_digest(dossier)
    if not set_digest or set_digest != str(dossier.get("keyframe_set_sha256") or ""):
        raise HumanCheckpointApprovalError("human dossier ordered keyframe-set digest is missing or inconsistent")

    subject = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "code_sha": code_sha,
        "canary_id": root.name,
        "machine_keyframe_review_passed": bool(dossier.get("machine_keyframe_review_passed")),
        "machine_review_disposition": disposition,
        "keyframe_set_sha256": set_digest,
        "files": files,
        "keyframes": keyframe_hashes,
    }
    approval_subject_sha256 = _canonical_digest(subject)
    manifest = {
        **subject,
        "approval_subject_sha256": approval_subject_sha256,
        "human_approval_required": True,
        "human_verdict": None,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "hitl-checkpoint.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # The dossier is mutable review metadata and therefore intentionally outside the digest cycle.
    dossier["code_sha"] = code_sha
    dossier["approval_subject_sha256"] = approval_subject_sha256
    dossier["checkpoint_manifest"] = "hitl-checkpoint.json"
    dossier_path.write_text(json.dumps(dossier, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify_hitl_checkpoint_v71(
    checkpoint_dir: Path,
    *,
    approved_keyframe_sha256: str,
    approved_code_sha: str,
) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    manifest = _read_json(root / "hitl-checkpoint.json")
    if manifest.get("schema_version") != _CHECKPOINT_SCHEMA:
        raise HumanCheckpointApprovalError("unsupported or missing HITL checkpoint schema")
    if str(manifest.get("machine_review_disposition") or "") not in _ALLOWED_MACHINE_DISPOSITIONS:
        raise HumanCheckpointApprovalError("sealed HITL checkpoint was not eligible for human arbitration")
    expected_digest = str(approved_keyframe_sha256 or "").strip()
    expected_code = str(approved_code_sha or "").strip()
    if not expected_digest or not expected_code:
        raise HumanCheckpointApprovalError("temporal generation requires human-approved checkpoint digest and code SHA")
    if str(manifest.get("code_sha")) != expected_code:
        raise HumanCheckpointApprovalError("approved code SHA does not match the sealed HITL checkpoint")
    if str(manifest.get("approval_subject_sha256")) != expected_digest:
        raise HumanCheckpointApprovalError("human-approved checkpoint digest does not match the sealed checkpoint")

    files = manifest.get("files")
    keyframes = manifest.get("keyframes")
    if not isinstance(files, dict) or not isinstance(keyframes, dict):
        raise HumanCheckpointApprovalError("sealed HITL checkpoint is missing file digests")
    for name, digest in files.items():
        path = root / str(name)
        if not path.is_file() or _sha256(path) != str(digest):
            raise HumanCheckpointApprovalError(f"approved checkpoint file changed after review: {name}")
    for name, digest in keyframes.items():
        path = root / "visual-keyframes" / str(name)
        if not path.is_file() or _sha256(path) != str(digest):
            raise HumanCheckpointApprovalError(f"approved keyframe changed after review: {name}")
    return manifest


def _load_package(root: Path) -> VideoPackage:
    payload = _read_json(root / "package.json")
    scenes_raw = payload.get("scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise HumanCheckpointApprovalError("approved package has no scenes")
    return VideoPackage(
        topic=str(payload["topic"]),
        narration=str(payload["narration"]),
        title=str(payload["title"]),
        description=str(payload["description"]),
        tags=[str(value) for value in payload.get("tags") or []],
        thumbnail_text=str(payload.get("thumbnail_text") or ""),
        top_comment=str(payload.get("top_comment") or ""),
        scenes=[
            Scene(
                heading=str(item["heading"]),
                body=str(item["body"]),
                visual=str(item["visual"]),
                source_index=int(item.get("source_index", 0)),
            )
            for item in scenes_raw
        ],
        source_urls=[str(value) for value in payload.get("source_urls") or []],
        source_publishers=[str(value) for value in payload.get("source_publishers") or []],
    )


def _load_visual_plan(root: Path) -> VisualPlan:
    payload = _read_json(root / "visual-plan.json")
    scenes_raw = payload.get("scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise HumanCheckpointApprovalError("approved visual plan has no scenes")
    scenes = tuple(
        SceneVisualPrompt(
            scene_index=int(item["scene_index"]),
            source_index=int(item["source_index"]),
            role=str(item["role"]),
            generation_mode=str(item["generation_mode"]),
            image_prompt=str(item["image_prompt"]),
            motion_prompt=str(item["motion_prompt"]),
            negative_prompt=str(item["negative_prompt"]),
            continuity_anchor=str(item["continuity_anchor"]),
            caption_safe_zone=str(item["caption_safe_zone"]),
            seed=int(item["seed"]),
            duration_seconds=float(item["duration_seconds"]),
        )
        for item in scenes_raw
    )
    return VisualPlan(
        prompt_version=str(payload["prompt_version"]),
        global_style=str(payload["global_style"]),
        palette=str(payload["palette"]),
        lighting=str(payload["lighting"]),
        continuity_bible=str(payload["continuity_bible"]),
        image_model=str(payload["image_model"]),
        video_model=str(payload["video_model"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        fps=int(payload["fps"]),
        director_input_sha256=str(payload["director_input_sha256"]),
        scenes=scenes,
    )


def _load_segments(root: Path, package: VideoPackage) -> tuple[tuple[NarrationSegment, ...], dict[str, Any]]:
    manifest = _read_json(root / "voice-review-manifest.json")
    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise HumanCheckpointApprovalError("approved checkpoint has no reviewed narration segments")
    segment_root = root / "voice-segments"
    segments: list[NarrationSegment] = []
    for item in raw_segments:
        original = Path(str(item.get("audio_path") or ""))
        audio_path = segment_root / original.name
        if not audio_path.is_file():
            raise HumanCheckpointApprovalError(f"approved narration segment is missing: {original.name}")
        segments.append(
            NarrationSegment(
                segment_id=int(item["segment_id"]),
                text=str(item["text"]),
                instruction=str(item["instruction"]),
                audio_path=audio_path,
                start_seconds=float(item["start_seconds"]),
                end_seconds=float(item["end_seconds"]),
                attempt=int(item.get("attempt", 1)),
            )
        )
    ordered = tuple(sorted(segments, key=lambda value: value.segment_id))
    spoken = " ".join(segment.text for segment in ordered)
    if " ".join(spoken.split()) != " ".join(package.narration.split()):
        raise HumanCheckpointApprovalError("approved narration segments do not reconstruct the approved package narration")
    return ordered, manifest


def materialize_approved_keyframes_v71(plan: VisualPlan, output_dir: Path) -> tuple[KeyframeAsset, ...]:
    checkpoint = os.getenv("HITL_APPROVED_CHECKPOINT_DIR", "").strip()
    if not checkpoint:
        raise HumanCheckpointApprovalError("approved checkpoint directory is not configured")
    root = Path(checkpoint)
    manifest = _read_json(root / "keyframe-manifest.json")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list):
        raise HumanCheckpointApprovalError("approved keyframe manifest has no assets")
    by_index = {int(item["scene_index"]): item for item in raw_assets if isinstance(item, dict)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[KeyframeAsset] = []
    manifest_assets: list[dict[str, Any]] = []
    for scene in plan.scenes:
        item = by_index.get(scene.scene_index)
        if item is None:
            raise HumanCheckpointApprovalError(f"approved keyframe metadata missing scene {scene.scene_index}")
        source = root / "visual-keyframes" / f"scene-{scene.scene_index:02d}-keyframe.png"
        if not source.is_file() or _sha256(source) != str(item.get("sha256") or ""):
            raise HumanCheckpointApprovalError(f"approved keyframe digest mismatch for scene {scene.scene_index}")
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        asset = KeyframeAsset(
            scene_index=scene.scene_index,
            path=destination,
            model=str(item.get("model") or "approved-checkpoint"),
            seed=int(item.get("seed", scene.seed)),
            width=int(item.get("width", plan.width)),
            height=int(item.get("height", plan.height)),
            sha256=_sha256(destination),
            entropy=float(item.get("entropy", 0.0)),
            prompt=str(item.get("prompt") or ""),
            negative_prompt=str(item.get("negative_prompt") or ""),
            director_prompt=str(item.get("director_prompt") or scene.image_prompt),
            prompt_word_count=int(item.get("prompt_word_count", 0)),
            prompt_word_budget=int(item.get("prompt_word_budget", 0)),
            prompt_compiler_version=str(item.get("prompt_compiler_version") or ""),
            caption_zone_detail_before=float(item.get("caption_zone_detail_before", 0.0)),
            caption_zone_detail_after=float(item.get("caption_zone_detail_after", 0.0)),
            caption_zone_repaired=bool(item.get("caption_zone_repaired", False)),
        )
        assets.append(asset)
        manifest_assets.append(asset.as_dict())
    if len(assets) != len(plan.scenes):
        raise HumanCheckpointApprovalError("approved keyframe count does not match approved visual plan")
    copied_manifest = dict(manifest)
    copied_manifest["assets"] = manifest_assets
    copied_manifest["asset_cache"] = "human_approved_checkpoint_v71"
    copied_manifest["approved_checkpoint_canary_id"] = root.name
    (output_dir / "keyframe-manifest.json").write_text(
        json.dumps(copied_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return tuple(assets)


def install_production_hitl_checkpoint_v71() -> None:
    """Use byte-identical human-approved keyframes when a checkpoint resume is requested."""
    from . import visual_pipeline

    current = visual_pipeline.generate_keyframes
    if getattr(current, "_agf_v71", False):
        return

    def generate_keyframes_v71(plan: VisualPlan, output_dir: Path) -> tuple[KeyframeAsset, ...]:
        if os.getenv("HITL_REUSE_APPROVED_CHECKPOINT", "false").strip().casefold() in {"1", "true", "yes", "on"}:
            return materialize_approved_keyframes_v71(plan, output_dir)
        return current(plan, output_dir)

    generate_keyframes_v71._agf_v71 = True  # type: ignore[attr-defined]
    visual_pipeline.generate_keyframes = generate_keyframes_v71


def run_approved_checkpoint_canary_v71(
    settings: Any,
    output_root: Path,
    *,
    approved_canary_id: str,
    approved_keyframe_sha256: str,
    approved_code_sha: str,
) -> dict[str, Any]:
    """Resume only temporal/render stages from the exact human-approved Gate-1 state."""
    output_root = Path(output_root)
    checkpoint = output_root / str(approved_canary_id).strip()
    approval = verify_hitl_checkpoint_v71(
        checkpoint,
        approved_keyframe_sha256=approved_keyframe_sha256,
        approved_code_sha=approved_code_sha,
    )
    package = _load_package(checkpoint)
    plan = _load_visual_plan(checkpoint)
    segments, voice_manifest = _load_segments(checkpoint, package)
    narration_path = checkpoint / "narration.wav"
    if not narration_path.is_file():
        raise HumanCheckpointApprovalError("approved narration.wav is missing")

    os.environ["HITL_APPROVED_CHECKPOINT_DIR"] = str(checkpoint)
    os.environ["HITL_REUSE_APPROVED_CHECKPOINT"] = "true"
    install_production_hitl_checkpoint_v71()

    from . import canary
    from .production_human_review_handoff_v61 import write_human_review_dossier_v61
    from .video_qc import verify_video_output
    from .visual_pipeline import render_visual_plan

    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(f"{stamp}|{approved_canary_id}|{approved_keyframe_sha256}".encode()).hexdigest()[:8]
    canary_id = f"{stamp}-{suffix}"
    destination = output_root / canary_id
    destination.mkdir(parents=True, exist_ok=False)
    settings.work_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="hitl-resume-", dir=settings.work_root))
    try:
        visual = render_visual_plan(
            plan=plan,
            package=package,
            segments=segments,
            audio_path=narration_path,
            workdir=workdir,
            output_width=settings.width,
            output_height=settings.height,
            output_fps=settings.fps,
        )
        canary._copy(visual.video_path, destination, "video.mp4")
        canary._copy(visual.thumbnail_path, destination, "thumbnail.png")
        canary._copy(visual.caption_path, destination, "animated-captions.ass")
        canary._copy(narration_path, destination, "narration.wav")
        canary._copy(checkpoint / "voice-review-manifest.json", destination, "voice-review-manifest.json")
        for name in (
            "package.json",
            "trend-snapshot.json",
            "vimax-plan.json",
            "production-preflight.json",
            "static-preflight.json",
            "preflight-animatic.mp4",
            "hitl-checkpoint.json",
            "keyframe-human-review-dossier.json",
            "keyframe-contact-sheet.jpg",
        ):
            source = checkpoint / name
            if source.is_file():
                canary._copy(source, destination, name)
        canary._copy_visual_audit(workdir, destination)

        metrics = voice_manifest.get("metrics") if isinstance(voice_manifest.get("metrics"), dict) else {}
        expected_duration = float(metrics.get("duration_seconds") or 0.0)
        if expected_duration <= 0:
            raise HumanCheckpointApprovalError("approved voice manifest has no valid duration")
        qc_path = workdir / "video-qc-report.json"
        qc = verify_video_output(
            settings,
            visual.video_path,
            visual.thumbnail_path,
            expected_duration=expected_duration,
            scene_durations=canary._scene_durations(segments, expected_duration),
            voice_manifest_path=checkpoint / "voice-review-manifest.json",
            require_production_voice=True,
            report_path=qc_path,
        )
        canary._copy(qc_path, destination, "video-qc-report.json")
        dossier_path = write_human_review_dossier_v61(destination)
        final_dossier = _read_json(dossier_path)
        if not final_dossier.get("automated_precheck_passed"):
            raise HumanCheckpointApprovalError("final render failed machine evidence precheck before human MP4 review")

        completed_at = datetime.now(timezone.utc)
        result = {
            "status": "awaiting_human_final_review",
            "release_decision": "blocked_pending_human_review",
            "canary_id": canary_id,
            "artifact_path": f"canaries/{canary_id}",
            "approved_checkpoint_canary_id": approved_canary_id,
            "approved_checkpoint_sha256": approval["approval_subject_sha256"],
            "approved_code_sha": approved_code_sha,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
            "topic": package.topic,
            "title": package.title,
            "shot_count": len(plan.scenes),
            "temporal_shots": sum(scene.generation_mode == "wan_i2v" for scene in plan.scenes),
            "video_qc": qc.as_dict(),
            "human_review": {
                "status": "awaiting_human_review",
                "artifact": "human-review-dossier.json",
                "video": "video.mp4",
                "audio": "narration.wav",
            },
        }
        (destination / "canary-result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except Exception as exc:
        canary._copy_visual_audit(workdir, destination)
        failure = {
            "status": "canary_failed_closed",
            "canary_id": canary_id,
            "artifact_path": f"canaries/{canary_id}",
            "approved_checkpoint_canary_id": approved_canary_id,
            "approved_checkpoint_sha256": approved_keyframe_sha256,
            "approved_code_sha": approved_code_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        (destination / "canary-failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return failure
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
