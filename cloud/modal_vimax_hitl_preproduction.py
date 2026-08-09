from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import modal

from cloud.modal_app import (
    MODEL_CACHE,
    STATE_DIR,
    VIMAX_COMMIT,
    _prepare_runtime,
    app,
    factory_secrets,
    hf_cache,
    state_volume,
    worker_image,
)


_HITL_RESULT_PREFIX = "HITL_RESULT_JSON="
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


def _clean_code_sha(value: str) -> str:
    clean = str(value or "").strip().lower()
    if not _SHA40_RE.fullmatch(clean):
        raise ValueError("Gate-1 requires the exact 40-character PR code SHA")
    return clean


_REPAIR_ENV_KEYS = (
    "HITL_REPAIR_SOURCE_DIR",
    "HITL_REPAIR_APPROVAL_SUBJECT_SHA256",
    "HITL_REPAIR_SOURCE_CODE_SHA",
    "HITL_REPAIR_REJECTED_SHOTS",
    "HITL_REPAIR_REQUEST_SHA256",
)


def _configure_hitl_repair_request(code_sha: str) -> dict[str, Any] | None:
    """Load an optional checked-in rejected-checkpoint repair request into the remote runtime.

    Clearing the variables first is mandatory because Modal containers may be warm-reused between
    invocations; a future run without a request must never inherit an older repair checkpoint.
    """
    import hashlib

    for key in _REPAIR_ENV_KEYS:
        os.environ.pop(key, None)
    request_path = Path(__file__).resolve().parents[1] / ".github" / "verification" / "hitl-repair-request.json"
    if not request_path.is_file():
        return None
    raw = request_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HITL repair request JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "hitl-repair-request-v72":
        raise ValueError("unsupported HITL repair request schema")
    source_canary = str(payload.get("source_canary_id") or "").strip()
    source_digest = str(payload.get("source_approval_subject_sha256") or "").strip().lower()
    source_code = _clean_code_sha(str(payload.get("source_code_sha") or ""))
    if source_code == _clean_code_sha(code_sha):
        raise ValueError("repair source code SHA must be older than the current exact-head revision")
    if not source_canary or Path(source_canary).name != source_canary:
        raise ValueError("repair request contains an unsafe source canary ID")
    if not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ValueError("repair request approval subject must be a 64-character SHA-256")
    raw_rejected = payload.get("rejected_shot_ids")
    if not isinstance(raw_rejected, list) or not raw_rejected:
        raise ValueError("repair request must contain rejected_shot_ids")
    try:
        rejected = sorted({int(value) for value in raw_rejected})
    except (TypeError, ValueError) as exc:
        raise ValueError("repair request rejected_shot_ids must be integers") from exc
    if rejected[0] < 0:
        raise ValueError("repair request rejected shot IDs must be non-negative")
    request_sha = hashlib.sha256(raw).hexdigest()
    os.environ["HITL_REPAIR_SOURCE_DIR"] = str(Path(STATE_DIR) / "canaries" / source_canary)
    os.environ["HITL_REPAIR_APPROVAL_SUBJECT_SHA256"] = source_digest
    os.environ["HITL_REPAIR_SOURCE_CODE_SHA"] = source_code
    os.environ["HITL_REPAIR_REJECTED_SHOTS"] = ",".join(str(value) for value in rejected)
    os.environ["HITL_REPAIR_REQUEST_SHA256"] = request_sha
    return {
        "schema_version": payload["schema_version"],
        "source_canary_id": source_canary,
        "source_approval_subject_sha256": source_digest,
        "source_code_sha": source_code,
        "rejected_shot_ids": rejected,
        "request_sha256": request_sha,
    }


def _persist_gate1_result(result: dict[str, Any], *, code_sha: str) -> Path:
    """Persist a deterministic exact-SHA result pointer in addition to streamed stdout."""
    clean_sha = _clean_code_sha(code_sha)
    result_root = Path(STATE_DIR) / "hitl-results"
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / f"{clean_sha}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    canary_id = str(result.get("canary_id") or "").strip()
    if canary_id and Path(canary_id).name == canary_id:
        canary_dir = Path(STATE_DIR) / "canaries" / canary_id
        if canary_dir.is_dir():
            (canary_dir / "gate1-result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return path


def _structured_machine_failure(result: dict[str, Any], *, code_sha: str) -> dict[str, Any]:
    return {
        **result,
        "release_decision": "blocked_machine_failure",
        "approved_code_sha": _clean_code_sha(code_sha),
        "human_review_required": False,
        "planning_backend": "vimax_script2video",
        "human_gate": "gate1_machine_failure_before_human_review",
        "vimax_commit": VIMAX_COMMIT,
    }


def _emit_gate1_result(result: dict[str, Any]) -> None:
    """Stream exactly one machine-readable result marker into `modal run` logs."""
    print(
        _HITL_RESULT_PREFIX
        + json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        flush=True,
    )


@app.function(
    image=worker_image,
    gpu="A10",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=0),
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def prepare_vimax_keyframe_review(code_sha: str = "") -> dict[str, object]:
    """Run exact production through keyframes and always emit/return a structured Gate-1 result."""
    code_sha = _clean_code_sha(code_sha)
    os.environ["PUBLISH_ENABLED"] = "false"
    os.environ["VIMAX_PLANNER_ENABLED"] = "true"
    os.environ["VIDEO_RENDER_BACKEND"] = "remotion"
    os.environ["WAN22_DYNAMIC_FRAME_NUM"] = "true"
    os.environ["WAN22_MIN_FRAME_NUM"] = "41"
    os.environ["WAN22_MAX_FRAME_NUM"] = "81"
    os.environ["WAN22_MODEL_CPU_OFFLOAD"] = "true"
    os.environ["HITL_KEYFRAME_PREVIEW_ONLY"] = "true"
    os.environ["HITL_CODE_SHA"] = code_sha
    repair_request = _configure_hitl_repair_request(code_sha)
    _prepare_runtime()

    from factory.production_caption_scale_v67 import install_production_caption_scale_v67
    from factory.production_editorial_boundary_v65 import install_production_editorial_boundary_v65
    from factory.production_hitl_checkpoint_v71 import finalize_hitl_checkpoint_v71
    from factory.production_keyframe_human_gate_v63 import install_production_keyframe_human_gate_v63
    from factory.production_vimax_copy_integrity_v68 import install_production_vimax_copy_integrity_v68
    from factory.production_vimax_focused_copy_protocol_v69 import install_production_vimax_focused_copy_protocol_v69
    from factory.production_vimax_human_editorial_v66 import install_production_vimax_human_editorial_v66
    from factory.production_vimax_infrastructure_grammar_v62 import install_production_vimax_infrastructure_grammar_v62
    from factory.production_vimax_topic_editorial_v70 import install_production_vimax_topic_editorial_v70
    from factory.production_vimax_unified_storyboard_v64 import install_production_vimax_unified_storyboard_v64

    install_production_editorial_boundary_v65()
    install_production_caption_scale_v67()
    install_production_vimax_infrastructure_grammar_v62()
    install_production_vimax_unified_storyboard_v64()
    install_production_vimax_human_editorial_v66()
    install_production_vimax_focused_copy_protocol_v69()
    install_production_vimax_copy_integrity_v68()
    install_production_vimax_topic_editorial_v70()
    install_production_keyframe_human_gate_v63()

    from factory.canary import run_production_canary
    from factory.config import Settings

    result = run_production_canary(Settings.from_env(), Path(STATE_DIR) / "canaries")

    if result.get("error_type") == "HumanKeyframeReviewRequired":
        artifact_path = str(result.get("artifact_path") or "")
        checkpoint_dir = Path(STATE_DIR) / artifact_path
        try:
            sealed = finalize_hitl_checkpoint_v71(checkpoint_dir, code_sha=code_sha)
            final: dict[str, Any] = {
                "status": "awaiting_human_keyframe_review",
                "release_decision": "blocked_pending_human_keyframe_review",
                "canary_id": result.get("canary_id"),
                "artifact_path": artifact_path,
                "approval_subject_sha256": sealed["approval_subject_sha256"],
                "approved_code_sha": sealed["code_sha"],
                "checkpoint_manifest": "hitl-checkpoint.json",
                "planning_backend": "vimax_script2video",
                "media_contract": "all_native_temporal_v55_after_human_keyframe_approval",
                "render_backend": "remotion_after_human_keyframe_approval",
                "human_gate": "sealed_checkpoint_v71",
                "editorial_grammar": "topic_aware_human_editorial_v70",
                "focused_copy_protocol": "narration_only_v69",
                "copy_integrity": "finished_punctuation_v68",
                "caption_geometry": "resolution_proportional_v67",
                "editorial_boundary": "post_grounding_capacity_authority_and_consumer_copy_v66",
                "human_review_required": True,
                "repair_request": repair_request,
                "vimax_commit": VIMAX_COMMIT,
            }
        except Exception as exc:
            final = {
                **result,
                "status": "canary_failed_closed",
                "release_decision": "blocked_checkpoint_sealing_failure",
                "approved_code_sha": code_sha,
                "human_review_required": False,
                "error_type": type(exc).__name__,
                "error": f"Gate-1 produced review evidence but checkpoint sealing failed: {exc}",
                "vimax_commit": VIMAX_COMMIT,
            }
    elif result.get("status") == "verified_render_canary":
        final = {
            **result,
            "status": "canary_failed_closed",
            "release_decision": "blocked_gate1_protocol_violation",
            "approved_code_sha": code_sha,
            "human_review_required": False,
            "error_type": "Gate1ProtocolError",
            "error": "HITL preproduction unexpectedly reached a final render instead of stopping before Wan",
            "vimax_commit": VIMAX_COMMIT,
        }
    else:
        final = _structured_machine_failure(dict(result), code_sha=code_sha)
        if repair_request is not None:
            final["repair_request"] = repair_request

    _persist_gate1_result(final, code_sha=code_sha)
    state_volume.commit()
    hf_cache.commit()
    _emit_gate1_result(final)
    return final
