from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_INSTALLED = False


class HumanReviewPreparationError(RuntimeError):
    """Raised when a rendered canary cannot be handed to a human editor safely."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _shot_samples(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for item in shots:
        try:
            shot_id = int(item.get("shot_id"))
            start = float(item.get("start_seconds") or 0.0)
            duration = float(item.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        margin = min(0.25, duration * 0.12)
        samples.append(
            {
                "shot_id": shot_id,
                "start_seconds": round(start, 3),
                "duration_seconds": round(duration, 3),
                "review_timestamps_seconds": [
                    round(start + margin, 3),
                    round(start + duration * 0.50, 3),
                    round(start + max(margin, duration - margin), 3),
                ],
                "semantic_claim": str(item.get("semantic_claim") or ""),
                "renderer": str(item.get("renderer") or ""),
            }
        )
    return samples


def build_human_review_dossier_v61(canary_dir: Path) -> dict[str, Any]:
    """Prepare a deterministic handoff for a senior human video-editor review.

    This gate does not pretend to be the human.  It proves that the artifact is reviewable,
    records the exact temporal/editing evidence, and leaves subjective editorial criteria in a
    pending state for an actual frame/audio review before merge or publishing.
    """
    canary_dir = Path(canary_dir)
    pipeline = _read_json(canary_dir / "visual-pipeline-manifest.json")
    media = _read_json(canary_dir / "scene-media-manifest.json")
    composition = _read_json(canary_dir / "visual-composition-manifest.json")
    qc = _read_json(canary_dir / "video-qc-report.json")
    voice = _read_json(canary_dir / "voice-review-manifest.json")
    captions = _read_json(canary_dir / "animated-captions.json")
    vimax = _read_json(canary_dir / "vimax-plan.json")

    shots = composition.get("shots") or pipeline.get("shots") or []
    if not isinstance(shots, list):
        shots = []
    assets = media.get("assets") or composition.get("scene_media") or []
    if not isinstance(assets, list):
        assets = []

    media_types = [str(item.get("media_type") or "").strip().lower() for item in assets if isinstance(item, dict)]
    hashes = [str(item.get("sha256") or "") for item in assets if isinstance(item, dict)]
    expected_temporal = int(media.get("expected_temporal_shots") or len(shots) or 0)
    realized_temporal = int(media.get("realized_temporal_shots") or sum(value == "video" for value in media_types))
    qc_passed = bool(qc.get("passed"))
    voice_metrics = voice.get("metrics") if isinstance(voice.get("metrics"), dict) else {}
    try:
        voice_wpm = float(voice_metrics.get("estimated_wpm") or 0.0)
    except (TypeError, ValueError):
        voice_wpm = 0.0

    checks = {
        "video_mp4_present": (canary_dir / "video.mp4").is_file() and (canary_dir / "video.mp4").stat().st_size > 0,
        "narration_wav_present": (canary_dir / "narration.wav").is_file() and (canary_dir / "narration.wav").stat().st_size > 0,
        "vimax_plan_present": bool(vimax),
        "shot_timeline_present": bool(shots),
        "all_source_media_temporal": bool(assets) and all(value == "video" for value in media_types),
        "temporal_count_matches_plan": expected_temporal > 0 and realized_temporal == expected_temporal == len(shots),
        "no_media_reuse": bool(hashes) and all(hashes) and len(hashes) == len(set(hashes)),
        "source_asset_looping_disabled": composition.get("source_asset_looping") is False,
        "image_fallback_forbidden": media.get("image_fallback_allowed") is False,
        "digital_zoom_motion_forbidden": media.get("digital_zoom_motion_allowed") is False,
        "machine_video_qc_passed": qc_passed,
        "voice_pace_in_release_window": 138.0 <= voice_wpm <= 146.0,
        "caption_pixel_fit_passed": captions.get("all_rendered_cues_fit") is True,
    }
    automated_precheck_passed = all(checks.values())

    return {
        "schema_version": "human-editor-simulation-v61",
        "status": "awaiting_human_review" if automated_precheck_passed else "blocked_before_human_review",
        "release_decision": "blocked_pending_human_review" if automated_precheck_passed else "blocked_machine_evidence_incomplete",
        "manual_review_required": True,
        "automated_precheck_passed": automated_precheck_passed,
        "checks": checks,
        "evidence": {
            "video": "video.mp4",
            "audio": "narration.wav",
            "vimax_plan": "vimax-plan.json",
            "scene_media_manifest": "scene-media-manifest.json",
            "composition_manifest": "visual-composition-manifest.json",
            "video_qc": "video-qc-report.json",
            "voice_review": "voice-review-manifest.json",
            "captions": "animated-captions.json",
            "expected_temporal_shots": expected_temporal,
            "realized_temporal_shots": realized_temporal,
            "voice_wpm": round(voice_wpm, 3),
        },
        "shot_review_samples": _shot_samples([item for item in shots if isinstance(item, dict)]),
        "human_checklist": [
            {"criterion": "hook_and_first_10_seconds", "status": "pending_human", "question": "Does the opening stop the scroll and deliver at least four genuinely distinct, relevant shots without dead air or slideshow motion?"},
            {"criterion": "native_scene_action", "status": "pending_human", "question": "Does every shot contain meaningful subject/object/environment motion rather than a frozen pose, crop, zoom, or loop?"},
            {"criterion": "semantic_relevance", "status": "pending_human", "question": "Do the generated scenes visibly support the spoken claim instead of generic or off-topic technology imagery?"},
            {"criterion": "continuity_and_visual_grammar", "status": "pending_human", "question": "Do adjacent shots preserve a coherent visual world while varying composition, camera language, and action enough to avoid repetition?"},
            {"criterion": "transitions_and_edit_rhythm", "status": "pending_human", "question": "Are cuts/transitions motivated and clean, with no jarring fades, accidental freezes, duplicated end frames, or quality drops?"},
            {"criterion": "captions", "status": "pending_human", "question": "Are captions phrase-level, readable, synchronized, visually balanced, and free from distracting one-word churn or occlusion?"},
            {"criterion": "voice_and_sync", "status": "pending_human", "question": "Does narration sound natural and clean, with credible cadence, pronunciation, pauses, and audiovisual synchronization?"},
            {"criterion": "artifact_and_release_quality", "status": "pending_human", "question": "Is the final 9:16 MP4 free of flicker, morphing, malformed subjects, text artifacts, compression defects, black frames, or other publish-blocking defects?"},
        ],
        "human_verdict": None,
        "human_notes": [],
    }


def write_human_review_dossier_v61(canary_dir: Path) -> Path:
    payload = build_human_review_dossier_v61(canary_dir)
    output = Path(canary_dir) / "human-review-dossier.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def install_production_human_review_handoff_v61() -> None:
    """Attach a mandatory manual-editor handoff to every machine-verified canary."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import canary

    current = canary.run_production_canary
    if getattr(current, "_agf_v61", False):
        _INSTALLED = True
        return

    def run_with_human_handoff(settings: Any, output_root: Path) -> dict[str, Any]:
        result = current(settings, output_root)
        if result.get("status") != "verified_render_canary":
            return result
        canary_id = str(result.get("canary_id") or "").strip()
        if not canary_id:
            return result
        canary_dir = Path(output_root) / canary_id
        dossier_path = write_human_review_dossier_v61(canary_dir)
        dossier = _read_json(dossier_path)
        if not dossier.get("automated_precheck_passed"):
            failure = {
                **result,
                "status": "canary_failed_closed",
                "error_type": "HumanReviewPreparationError",
                "error": "machine evidence is incomplete or inconsistent before human review",
                "human_review": dossier,
            }
            (canary_dir / "canary-failure.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return failure

        updated = {
            **result,
            "human_review": {
                "status": "awaiting_human_review",
                "release_decision": "blocked_pending_human_review",
                "manual_review_required": True,
                "artifact": "human-review-dossier.json",
            },
        }
        (canary_dir / "canary-result.json").write_text(
            json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return updated

    run_with_human_handoff._agf_v61 = True  # type: ignore[attr-defined]
    canary.run_production_canary = run_with_human_handoff
    _INSTALLED = True
