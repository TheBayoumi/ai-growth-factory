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
    gpu="A10",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=0),
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def prepare_vimax_keyframe_review() -> dict[str, object]:
    """Run the real production stack only through machine-reviewed keyframes.

    The expected successful outcome is not a rendered video. It is an immutable canary directory
    containing the package, reviewed narration, ViMax plan, all keyframes, machine-review evidence,
    contact sheet and human-review dossier. Wan temporal inference is deliberately not entered
    until those exact keyframes are approved by a human editor.
    """
    os.environ["PUBLISH_ENABLED"] = "false"
    os.environ["VIMAX_PLANNER_ENABLED"] = "true"
    os.environ["VIDEO_RENDER_BACKEND"] = "remotion"
    os.environ["WAN22_DYNAMIC_FRAME_NUM"] = "true"
    os.environ["WAN22_MIN_FRAME_NUM"] = "41"
    os.environ["WAN22_MAX_FRAME_NUM"] = "81"
    os.environ["WAN22_MODEL_CPU_OFFLOAD"] = "true"
    os.environ["HITL_KEYFRAME_PREVIEW_ONLY"] = "true"
    _prepare_runtime()

    # Install after the legacy runtime so ViMax planning, preflight, generation, retry and review
    # share one exact filmable storyboard. v63 then stops the run after those keyframes exist.
    from factory.production_keyframe_human_gate_v63 import (
        install_production_keyframe_human_gate_v63,
    )
    from factory.production_vimax_infrastructure_grammar_v62 import (
        install_production_vimax_infrastructure_grammar_v62,
    )
    from factory.production_vimax_unified_storyboard_v64 import (
        install_production_vimax_unified_storyboard_v64,
    )

    install_production_vimax_infrastructure_grammar_v62()
    install_production_vimax_unified_storyboard_v64()
    install_production_keyframe_human_gate_v63()

    from factory.canary import run_production_canary
    from factory.config import Settings

    result = run_production_canary(Settings.from_env(), Path(STATE_DIR) / "canaries")
    state_volume.commit()
    hf_cache.commit()

    if result.get("error_type") == "HumanKeyframeReviewRequired":
        return {
            "status": "awaiting_human_keyframe_review",
            "release_decision": "blocked_pending_human_keyframe_review",
            "canary_id": result.get("canary_id"),
            "artifact_path": result.get("artifact_path"),
            "planning_backend": "vimax_script2video",
            "media_contract": "all_native_temporal_v55_after_human_keyframe_approval",
            "render_backend": "remotion_after_human_keyframe_approval",
            "human_gate": "pre_wan_keyframe_review_v63",
            "editorial_grammar": "unified_vimax_storyboard_v64",
            "vimax_commit": VIMAX_COMMIT,
        }
    if result.get("status") != "verified_render_canary":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    raise RuntimeError(
        "HITL preproduction unexpectedly reached a final render; keyframe preview gate did not stop before Wan"
    )
