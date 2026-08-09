from __future__ import annotations

import json
import os
from pathlib import Path

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


@app.function(
    image=worker_image,
    cpu=2.0,
    memory=4096,
    timeout=10 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=0),
    volumes={STATE_DIR: state_volume},
)
def record_vimax_keyframe_review(
    approved_canary_id: str = "",
    approval_subject_sha256: str = "",
    approved_code_sha: str = "",
    reviewer_kind: str = "human_simulation",
    verdict: str = "",
    reviewed_shots: str = "",
    notes: str = "",
) -> dict[str, object]:
    """Persist an explicit HITL verdict before any temporal model can run."""
    _prepare_runtime()
    from factory.production_hitl_checkpoint_v71 import record_hitl_decision_v71

    clean_id = str(approved_canary_id or "").strip()
    checkpoint = Path(STATE_DIR) / "canaries" / Path(clean_id).name
    if checkpoint.name != clean_id or not checkpoint.is_dir():
        raise ValueError("approved canary ID is missing or unsafe")
    try:
        shot_ids = tuple(
            int(value.strip())
            for value in str(reviewed_shots or "").split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("reviewed_shots must be a comma-separated list of integer shot IDs") from exc
    decision = record_hitl_decision_v71(
        checkpoint,
        approval_subject_sha256=approval_subject_sha256,
        code_sha=approved_code_sha,
        reviewer_kind=reviewer_kind,
        verdict=verdict,
        reviewed_shot_ids=shot_ids,
        notes=tuple(value.strip() for value in str(notes or "").split("||") if value.strip()),
    )
    state_volume.commit()
    return decision


@app.function(
    image=worker_image,
    gpu="A10",
    cpu=8.0,
    memory=65536,
    timeout=110 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=0),
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def render_vimax_temporal_canary(
    approved_canary_id: str = "",
    approved_keyframe_sha256: str = "",
    approved_code_sha: str = "",
) -> dict[str, object]:
    """Resume the exact human-approved Gate-1 checkpoint through Wan, Remotion and final QC."""
    os.environ["PUBLISH_ENABLED"] = "false"
    os.environ["VIMAX_PLANNER_ENABLED"] = "true"
    os.environ["VIDEO_RENDER_BACKEND"] = "remotion"
    os.environ["WAN22_DYNAMIC_FRAME_NUM"] = "true"
    os.environ["WAN22_MIN_FRAME_NUM"] = "41"
    os.environ["WAN22_MAX_FRAME_NUM"] = "81"
    os.environ["WAN22_MODEL_CPU_OFFLOAD"] = "true"
    os.environ["HITL_KEYFRAME_PREVIEW_ONLY"] = "false"
    _prepare_runtime()

    from factory.production_caption_scale_v67 import install_production_caption_scale_v67
    from factory.production_editorial_boundary_v65 import install_production_editorial_boundary_v65
    from factory.production_hitl_checkpoint_v71 import (
        install_production_hitl_checkpoint_v71,
        run_approved_checkpoint_canary_v71,
    )
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
    install_production_hitl_checkpoint_v71()

    from factory.config import Settings

    result = run_approved_checkpoint_canary_v71(
        Settings.from_env(),
        Path(STATE_DIR) / "canaries",
        approved_canary_id=approved_canary_id,
        approved_keyframe_sha256=approved_keyframe_sha256,
        approved_code_sha=approved_code_sha,
    )
    state_volume.commit()
    hf_cache.commit()
    if result.get("status") != "awaiting_human_final_review":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return {
        **result,
        "planning_backend": "vimax_script2video_checkpoint_resume",
        "media_contract": "all_native_temporal_v55_from_approved_keyframes",
        "render_backend": "remotion",
        "validation_gpu": "A10",
        "editorial_grammar": "topic_aware_human_editorial_v70",
        "human_gate": "sealed_checkpoint_v71",
        "final_human_gate": "human_editor_simulation_v61",
        "vimax_commit": VIMAX_COMMIT,
    }
