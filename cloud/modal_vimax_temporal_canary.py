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

    Scheduled production remains untouched. The current Modal workspace permits A10 compute without
    adding a payment method, so validation retains the proven A10 + CPU-offload memory strategy while
    still requiring twenty native temporal clips, 24 Wan sampling steps, and every publication gate.
    Publishing is always disabled.
    """
    os.environ["PUBLISH_ENABLED"] = "false"
    os.environ["VIMAX_PLANNER_ENABLED"] = "true"
    os.environ["VIDEO_RENDER_BACKEND"] = "remotion"
    os.environ["WAN22_DYNAMIC_FRAME_NUM"] = "true"
    os.environ["WAN22_MIN_FRAME_NUM"] = "41"
    os.environ["WAN22_MAX_FRAME_NUM"] = "81"
    os.environ["WAN22_MODEL_CPU_OFFLOAD"] = "true"
    _prepare_runtime()
    result = _run_render_canary()
    return {
        **result,
        "planning_backend": "vimax_script2video",
        "media_contract": "all_native_temporal_v55",
        "render_backend": "remotion",
        "validation_gpu": "A10",
        "vimax_commit": VIMAX_COMMIT,
    }
