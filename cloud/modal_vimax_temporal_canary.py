from __future__ import annotations

import os

import modal

from cloud.modal_app import (
    MODEL_CACHE,
    STATE_DIR,
    VIMAX_COMMIT,
    _prepare_runtime,
    _run_render_canary,
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
    timeout=110 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=0),
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def render_vimax_temporal_canary() -> dict[str, object]:
    """Validate the all-temporal ViMax -> Factory -> Remotion release path.

    This stage is used only after the exact keyframes have passed the pre-Wan human gate. The
    same package/editorial boundaries and human-approved storyboard grammar are reinstalled for
    the full temporal run. Publishing stays disabled.
    """
    os.environ["PUBLISH_ENABLED"] = "false"
    os.environ["VIMAX_PLANNER_ENABLED"] = "true"
    os.environ["VIDEO_RENDER_BACKEND"] = "remotion"
    os.environ["WAN22_DYNAMIC_FRAME_NUM"] = "true"
    os.environ["WAN22_MIN_FRAME_NUM"] = "41"
    os.environ["WAN22_MAX_FRAME_NUM"] = "81"
    os.environ["WAN22_MODEL_CPU_OFFLOAD"] = "true"
    os.environ["HITL_KEYFRAME_PREVIEW_ONLY"] = "false"
    _prepare_runtime()

    from factory.production_editorial_boundary_v65 import install_production_editorial_boundary_v65
    from factory.production_keyframe_human_gate_v63 import install_production_keyframe_human_gate_v63
    from factory.production_vimax_human_editorial_v66 import install_production_vimax_human_editorial_v66
    from factory.production_vimax_infrastructure_grammar_v62 import install_production_vimax_infrastructure_grammar_v62
    from factory.production_vimax_unified_storyboard_v64 import install_production_vimax_unified_storyboard_v64

    install_production_editorial_boundary_v65()
    install_production_vimax_infrastructure_grammar_v62()
    install_production_vimax_unified_storyboard_v64()
    install_production_vimax_human_editorial_v66()
    install_production_keyframe_human_gate_v63()

    result = _run_render_canary()
    return {
        **result,
        "planning_backend": "vimax_script2video",
        "media_contract": "all_native_temporal_v55",
        "render_backend": "remotion",
        "validation_gpu": "A10",
        "editorial_grammar": "human_editorial_v66",
        "editorial_boundary": "post_grounding_capacity_authority_and_consumer_copy_v66",
        "human_keyframe_dossier": "keyframe-human-review-dossier.json",
        "vimax_commit": VIMAX_COMMIT,
    }
