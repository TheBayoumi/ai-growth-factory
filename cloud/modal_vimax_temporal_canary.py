from __future__ import annotations

import json
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
    gpu="L40S",
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

    Scheduled production remains untouched. This validation worker uses an L40S so the complete
    twenty-clip Wan pass can keep the 24-step quality setting without the legacy A10 CPU-offload
    bottleneck. Publishing is always disabled.
    """
    os.environ["PUBLISH_ENABLED"] = "false"
    os.environ["VIMAX_PLANNER_ENABLED"] = "true"
    os.environ["VIDEO_RENDER_BACKEND"] = "remotion"
    os.environ["WAN22_DYNAMIC_FRAME_NUM"] = "true"
    os.environ["WAN22_MIN_FRAME_NUM"] = "41"
    os.environ["WAN22_MAX_FRAME_NUM"] = "81"
    os.environ["WAN22_MODEL_CPU_OFFLOAD"] = "false"
    _prepare_runtime()
    result = _run_render_canary()
    return {
        **result,
        "planning_backend": "vimax_script2video",
        "media_contract": "all_native_temporal_v55",
        "render_backend": "remotion",
        "validation_gpu": "L40S",
        "vimax_commit": VIMAX_COMMIT,
    }


@app.local_entrypoint()
def main() -> None:
    result = render_vimax_temporal_canary.remote()
    print(json.dumps(result, ensure_ascii=False))
