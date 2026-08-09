from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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
    "editorial-timeline.json",
)
_ALLOWED_MACHINE_DISPOSITIONS = {"passed", "advisory_human_arbitration"}
_DECISION_SCHEMA = "vimax-hitl-decision-v71"
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


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
    code_sha = str(code_sha or "").strip().lower()
    if not _SHA40_RE.fullmatch(code_sha):
        raise HumanCheckpointApprovalError("HITL checkpoint requires an exact 40-character Git code SHA")
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
        "expected_shots": expected,
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



def record_hitl_decision_v71(
    checkpoint_dir: Path,
    *,
    approval_subject_sha256: str,
    code_sha: str,
    reviewer_kind: str,
    verdict: str,
    reviewed_shot_ids: Sequence[int],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Persist the explicit human/simulated-human verdict bound to the sealed checkpoint."""
    root = Path(checkpoint_dir)
    approval = verify_hitl_checkpoint_v71(
        root,
        approved_keyframe_sha256=approval_subject_sha256,
        approved_code_sha=code_sha,
    )
    reviewer = str(reviewer_kind or "").strip().lower()
    if reviewer not in {"human", "human_simulation"}:
        raise HumanCheckpointApprovalError("reviewer_kind must be human or human_simulation")
    clean_verdict = str(verdict or "").strip().lower()
    if clean_verdict not in {"approve", "reject"}:
        raise HumanCheckpointApprovalError("HITL verdict must be approve or reject")
    reviewed = sorted({int(value) for value in reviewed_shot_ids})
    expected = list(range(int(approval.get("expected_shots") or 0)))
    if any(value not in expected for value in reviewed):
        raise HumanCheckpointApprovalError("HITL decision contains an unknown shot ID")
    if clean_verdict == "approve" and reviewed != expected:
        raise HumanCheckpointApprovalError("approval requires explicit review of every sealed keyframe")
    clean_notes = [" ".join(str(value).split()).strip() for value in notes if str(value).strip()]
    payload: dict[str, Any] = {
        "schema_version": _DECISION_SCHEMA,
        "canary_id": root.name,
        "code_sha": str(approval["code_sha"]),
        "approval_subject_sha256": str(approval["approval_subject_sha256"]),
        "reviewer_kind": reviewer,
        "verdict": clean_verdict,
        "reviewed_shot_ids": reviewed,
        "notes": clean_notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["decision_sha256"] = _canonical_digest(payload)
    (root / "hitl-decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def require_approved_hitl_decision_v71(
    checkpoint_dir: Path,
    *,
    approval_subject_sha256: str,
    code_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless an explicit approve decision covers every sealed keyframe."""
    root = Path(checkpoint_dir)
    approval = verify_hitl_checkpoint_v71(
        root,
        approved_keyframe_sha256=approval_subject_sha256,
        approved_code_sha=code_sha,
    )
    decision = _read_json(root / "hitl-decision.json")
    if decision.get("schema_version") != _DECISION_SCHEMA:
        raise HumanCheckpointApprovalError("missing or unsupported HITL decision schema")
    stored = str(decision.get("decision_sha256") or "").strip().lower()
    unsigned = dict(decision)
    unsigned.pop("decision_sha256", None)
    if not stored or stored != _canonical_digest(unsigned):
        raise HumanCheckpointApprovalError("HITL decision digest is invalid")
    if str(decision.get("canary_id") or "") != root.name:
        raise HumanCheckpointApprovalError("HITL decision targets a different canary")
    if str(decision.get("code_sha") or "") != str(approval["code_sha"]):
        raise HumanCheckpointApprovalError("HITL decision targets a different code revision")
    if str(decision.get("approval_subject_sha256") or "") != str(approval["approval_subject_sha256"]):
        raise HumanCheckpointApprovalError("HITL decision targets a different sealed checkpoint")
    if str(decision.get("reviewer_kind") or "") not in {"human", "human_simulation"}:
        raise HumanCheckpointApprovalError("HITL decision reviewer kind is invalid")
    if str(decision.get("verdict") or "") != "approve":
        raise HumanCheckpointApprovalError("HITL decision has not approved temporal generation")
    expected = list(range(int(approval.get("expected_shots") or 0)))
    try:
        reviewed = sorted({int(value) for value in decision.get("reviewed_shot_ids") or []})
    except (TypeError, ValueError) as exc:
        raise HumanCheckpointApprovalError("HITL decision contains invalid reviewed shot IDs") from exc
    if reviewed != expected:
        raise HumanCheckpointApprovalError("HITL approval does not cover every sealed keyframe")
    return approval, decision


def _audio_duration(path: Path) -> float:
    if not Path(path).is_file():
        raise HumanCheckpointApprovalError("approved narration.wav is missing")
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / max(1, handle.getframerate())
    except (wave.Error, OSError) as exc:
        raise HumanCheckpointApprovalError("approved narration.wav is unreadable") from exc


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


def _load_segments(
    root: Path,
    package: VideoPackage,
    narration_path: Path,
) -> tuple[tuple[NarrationSegment, ...], dict[str, Any]]:
    manifest = _read_json(root / "voice-review-manifest.json")
    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise HumanCheckpointApprovalError("approved checkpoint has no reviewed narration segments")
    segments = tuple(
        NarrationSegment(
            segment_id=int(item["segment_id"]),
            text=str(item["text"]),
            instruction=str(item.get("instruction") or ""),
            audio_path=narration_path,
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
            attempt=int(item.get("attempt", 1)),
        )
        for item in raw_segments
        if isinstance(item, dict)
    )
    ordered = tuple(sorted(segments, key=lambda value: value.segment_id))
    if [item.segment_id for item in ordered] != list(range(len(ordered))):
        raise HumanCheckpointApprovalError("approved narration segment IDs are not contiguous")
    spoken = " ".join(segment.text for segment in ordered)
    if " ".join(spoken.split()) != " ".join(package.narration.split()):
        raise HumanCheckpointApprovalError("approved narration segments do not reconstruct the approved package narration")
    return ordered, manifest


def _load_editorial_timeline(
    root: Path,
    plan: VisualPlan,
    *,
    narration_duration: float,
) -> tuple[Any, ...]:
    from .editorial_timeline import ShotSpec
    from .video_profile import VideoProfile

    payload = _read_json(root / "editorial-timeline.json")
    profile = payload.get("profile")
    if profile != VideoProfile.from_env().as_dict():
        raise HumanCheckpointApprovalError("approved editorial timeline quality contract differs from runtime")
    raw_shots = payload.get("shots")
    if not isinstance(raw_shots, list) or len(raw_shots) != len(plan.scenes):
        raise HumanCheckpointApprovalError("approved editorial timeline does not match the reviewed visual plan")
    shots = tuple(
        ShotSpec(
            shot_id=int(item["shot_id"]),
            beat_index=int(item["beat_index"]),
            segment_id=int(item["segment_id"]),
            package_scene_index=int(item["package_scene_index"]),
            source_index=int(item["source_index"]),
            start_seconds=float(item["start_seconds"]),
            duration_seconds=float(item["duration_seconds"]),
            renderer=str(item["renderer"]),
            semantic_claim=str(item["semantic_claim"]),
            visual_direction=str(item["visual_direction"]),
            treatment=str(item["treatment"]),
            seed=int(item["seed"]),
        )
        for item in raw_shots
        if isinstance(item, dict)
    )
    if len(shots) != len(plan.scenes) or [item.shot_id for item in shots] != list(range(len(shots))):
        raise HumanCheckpointApprovalError("approved editorial timeline has malformed or non-contiguous shots")
    if any(item.renderer != "wan_i2v" for item in shots):
        raise HumanCheckpointApprovalError("approved editorial timeline contains non-temporal media")
    if any(scene.generation_mode != "wan_i2v" for scene in plan.scenes):
        raise HumanCheckpointApprovalError("approved visual plan contains non-temporal media")
    for shot, scene in zip(shots, plan.scenes, strict=True):
        if abs(float(shot.duration_seconds) - float(scene.duration_seconds)) > 1e-6:
            raise HumanCheckpointApprovalError(
                f"approved timeline duration differs from reviewed visual plan for shot {shot.shot_id}"
            )
        if int(shot.seed) != int(scene.seed) or int(shot.source_index) != int(scene.source_index):
            raise HumanCheckpointApprovalError(
                f"approved timeline identity differs from reviewed visual plan for shot {shot.shot_id}"
            )
    total = sum(float(item.duration_seconds) for item in shots)
    if abs(total - narration_duration) > 0.08:
        raise HumanCheckpointApprovalError(
            f"approved editorial timeline {total:.3f}s does not match narration {narration_duration:.3f}s"
        )
    if abs(float(payload.get("duration_seconds") or 0.0) - narration_duration) > 0.08:
        raise HumanCheckpointApprovalError("approved timeline manifest duration differs from narration")
    return shots


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
    """Resume only Wan/render stages from the exact approved 20-shot checkpoint.

    This intentionally bypasses ``render_visual_plan_v28`` because the persisted visual plan is
    already the expanded reviewed editorial plan. Feeding it back through v28 preflight would
    require source-plan/package cardinality and could rebuild a different timeline after HITL.
    """
    from . import canary, production_editorial_v28 as editorial_v28, video_generator, visual_pipeline
    from .production_human_review_handoff_v61 import write_human_review_dossier_v61
    from .video_profile import VideoProfile
    from .video_qc import verify_video_output

    output_root = Path(output_root)
    checkpoint = output_root / str(approved_canary_id).strip()
    if checkpoint.name != str(approved_canary_id).strip() or not checkpoint.is_dir():
        raise HumanCheckpointApprovalError("approved checkpoint canary ID is missing or unsafe")
    approval, decision = require_approved_hitl_decision_v71(
        checkpoint,
        approval_subject_sha256=approved_keyframe_sha256,
        code_sha=approved_code_sha,
    )
    package = _load_package(checkpoint)
    plan = _load_visual_plan(checkpoint)
    narration_path = checkpoint / "narration.wav"
    narration_duration = _audio_duration(narration_path)
    segments, voice_manifest = _load_segments(checkpoint, package, narration_path)
    shots = _load_editorial_timeline(
        checkpoint,
        plan,
        narration_duration=narration_duration,
    )
    preflight = _read_json(checkpoint / "production-preflight.json")
    profile = VideoProfile.from_env()
    if preflight.get("status") != "passed":
        raise HumanCheckpointApprovalError("approved exact editorial preflight did not pass")
    if preflight.get("quality_contract") != profile.as_dict():
        raise HumanCheckpointApprovalError("approved preflight quality contract differs from runtime")
    if int(preflight.get("shot_count") or 0) != len(shots):
        raise HumanCheckpointApprovalError("approved preflight shot count differs from reviewed timeline")
    if int(preflight.get("wan_shots") or 0) != len(shots):
        raise HumanCheckpointApprovalError("approved preflight is not an all-temporal Wan plan")
    if abs(float(preflight.get("narration_duration_seconds") or 0.0) - narration_duration) > 0.01:
        raise HumanCheckpointApprovalError("approved narration no longer matches exact preflight")

    os.environ["HITL_APPROVED_CHECKPOINT_DIR"] = str(checkpoint)
    os.environ["HITL_REUSE_APPROVED_CHECKPOINT"] = "true"
    install_production_hitl_checkpoint_v71()

    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(
        f"{stamp}|{approved_canary_id}|{approved_keyframe_sha256}".encode()
    ).hexdigest()[:8]
    canary_id = f"{stamp}-{suffix}"
    destination = output_root / canary_id
    destination.mkdir(parents=True, exist_ok=False)
    settings.work_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="hitl-resume-", dir=settings.work_root))
    try:
        visual_root = workdir / "visual-assets"
        keyframe_dir = visual_root / "keyframes"
        scene_media_dir = visual_root / "scene-media"
        render_dir = visual_root / "render"
        plan_path = visual_pipeline.persist_visual_plan(plan, visual_root / "visual-plan.json")
        timeline_path = visual_root / "editorial-timeline.json"
        shutil.copy2(checkpoint / "editorial-timeline.json", timeline_path)

        keyframes = materialize_approved_keyframes_v71(plan, keyframe_dir)
        visual_pipeline.release_accelerator_memory()
        scene_media = video_generator.generate_scene_media(plan, keyframes, scene_media_dir)
        visual_pipeline.release_accelerator_memory()
        if len(scene_media) != len(shots) or any(asset.media_type != "video" for asset in scene_media):
            raise HumanCheckpointApprovalError(
                "temporal resume produced missing or non-video scene media"
            )

        video_path, thumbnail_path, caption_path = editorial_v28._compose_editorial_video(
            media=scene_media,
            shots=shots,
            segments=segments,
            package=package,
            audio_path=narration_path,
            workdir=render_dir,
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
        )
        visual = visual_pipeline.VisualPipelineOutput(
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            caption_path=caption_path,
            visual_plan_path=plan_path,
            keyframes=keyframes,
            scene_media=scene_media,
        )
        pipeline_manifest = {
            **visual.as_dict(),
            "profile": profile.as_dict(),
            "editorial_timeline": str(timeline_path),
            "shot_count": len(shots),
            "source_asset_looping": False,
            "destructive_caption_matte": False,
            "preflight": {
                "status": "passed_before_human_review",
                "manifest": str(checkpoint / "production-preflight.json"),
                "animatic": str(checkpoint / "preflight-animatic.mp4"),
                "quality_contract_sha256": preflight.get("quality_contract_sha256"),
            },
            "hitl_resume": {
                "approved_checkpoint_canary_id": approved_canary_id,
                "approval_subject_sha256": approval["approval_subject_sha256"],
                "approved_code_sha": approved_code_sha,
                "decision_sha256": decision["decision_sha256"],
                "reviewer_kind": decision["reviewer_kind"],
            },
            "shots": [item.as_dict() for item in shots],
        }
        (visual_root / "visual-pipeline-manifest.json").write_text(
            json.dumps(pipeline_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        canary._copy(video_path, destination, "video.mp4")
        canary._copy(thumbnail_path, destination, "thumbnail.png")
        canary._copy(caption_path, destination, "animated-captions.ass")
        canary._copy(narration_path, destination, "narration.wav")
        canary._copy(checkpoint / "voice-review-manifest.json", destination, "voice-review-manifest.json")
        for name in (
            "package.json",
            "trend-snapshot.json",
            "vimax-plan.json",
            "production-preflight.json",
            "static-preflight.json",
            "preflight-animatic.mp4",
            "editorial-timeline.json",
            "hitl-checkpoint.json",
            "hitl-decision.json",
            "keyframe-human-review-dossier.json",
            "keyframe-contact-sheet.jpg",
        ):
            source = checkpoint / name
            if source.is_file():
                canary._copy(source, destination, name)
        canary._copy_visual_audit(workdir, destination)

        metrics = voice_manifest.get("metrics") if isinstance(voice_manifest.get("metrics"), dict) else {}
        manifest_duration = float(metrics.get("duration_seconds") or 0.0)
        if manifest_duration <= 0 or abs(manifest_duration - narration_duration) > 0.01:
            raise HumanCheckpointApprovalError("approved voice manifest duration differs from narration")
        qc_path = workdir / "video-qc-report.json"
        qc = verify_video_output(
            settings,
            video_path,
            thumbnail_path,
            expected_duration=narration_duration,
            scene_durations=[float(item.duration_seconds) for item in shots],
            voice_manifest_path=checkpoint / "voice-review-manifest.json",
            require_production_voice=True,
            report_path=qc_path,
        )
        canary._copy(qc_path, destination, "video-qc-report.json")
        dossier_path = write_human_review_dossier_v61(destination)
        final_dossier = _read_json(dossier_path)
        if not final_dossier.get("automated_precheck_passed"):
            raise HumanCheckpointApprovalError(
                "final render failed machine evidence precheck before human MP4 review"
            )

        completed_at = datetime.now(timezone.utc)
        result = {
            "status": "awaiting_human_final_review",
            "release_decision": "blocked_pending_human_review",
            "canary_id": canary_id,
            "artifact_path": f"canaries/{canary_id}",
            "approved_checkpoint_canary_id": approved_canary_id,
            "approved_checkpoint_sha256": approval["approval_subject_sha256"],
            "approved_code_sha": approved_code_sha,
            "hitl_decision_sha256": decision["decision_sha256"],
            "hitl_reviewer_kind": decision["reviewer_kind"],
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
            "topic": package.topic,
            "title": package.title,
            "shot_count": len(shots),
            "temporal_shots": len(scene_media),
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
