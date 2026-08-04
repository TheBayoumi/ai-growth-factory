from __future__ import annotations

import os


_INSTALLED = False
_SDXL_CHECKPOINT = "sdxl_lightning_8step_unet.safetensors"
_SDXL_STEPS = "8"


def install_production_visual_runtime_v28() -> None:
    """Use the matched eight-step SDXL-Lightning checkpoint for semantic keyframes."""
    global _INSTALLED
    if _INSTALLED:
        return
    os.environ["VISUAL_SDXL_LIGHTNING_CHECKPOINT"] = _SDXL_CHECKPOINT
    os.environ["VISUAL_SDXL_LIGHTNING_STEPS"] = _SDXL_STEPS
    _INSTALLED = True
