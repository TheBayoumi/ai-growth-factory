from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = "ai-growth-factory"
MODEL_CACHE = "/cache/huggingface"
STATE_DIR = "/state"
WORK_DIR = "/tmp/ai-growth-factory"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("ai-growth-factory-model-cache", create_if_missing=True)
state_volume = modal.Volume.from_name("ai-growth-factory-state", create_if_missing=True)

worker_image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "sox",
        "libsox-fmt-all",
        "git",
        "libgl1",
        "libglib2.0-0",
        "fontconfig",
        "fonts-dejavu-core",
    )
    .run_commands(
        "python -m pip install --upgrade pip wheel setuptools",
        (
            "python -m pip install torch==2.8.0 torchaudio==2.8.0 "
            "torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128"
        ),
    )
    .pip_install(
        "requests==2.32.5",
        "Pillow==11.3.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "numpy==2.2.6",
        "numba==0.64.0",
        "soundfile>=0.12,<0.14",
        "qwen-tts==0.1.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "qwen-omni-utils[decord]>=0.0.8",
        "diffusers==0.39.0",
        "huggingface-hub[hf_xet]>=0.34,<2",
        "safetensors>=0.5,<1",
        "ftfy>=6.3,<7",
        "sentencepiece>=0.2,<1",
        "einops>=0.8,<1",
    )
    .run_commands(
        (
            "python -c \"import decord, imageio, imageio_ffmpeg, numba, numpy, torch, torchvision; "
            "from qwen_tts import Qwen3TTSModel; "
            "from qwen_omni_utils import process_mm_info; "
            "from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; "
            "from diffusers import AutoencoderKLWan, FluxPipeline, StableDiffusionXLPipeline, "
            "UNet2DConditionModel, WanImageToVideoPipeline; "
            "assert imageio.__version__ == '2.37.0', imageio.__version__; "
            "assert numpy.__version__ == '2.2.6', numpy.__version__; "
            "assert numba.__version__ == '0.64.0', numba.__version__; "
            "assert imageio_ffmpeg.get_ffmpeg_exe(); "
            "print('Voice, reviewer, image, and Wan2.2 visual runtime import preflight passed; imageio export available')\""
        ),
        "fc-cache -f -v >/dev/null",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE,
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "WORK_ROOT": WORK_DIR,
            "STATE_ROOT": STATE_DIR,
            "LLAMA_CPP_EXECUTABLE": "/app/llama-server",
            "LLAMA_CPP_BASE_URL": "http://127.0.0.1:8080/v1",
            "LLAMA_CPP_MANAGED": "true",
            "LLAMA_CPP_GPU_LAYERS": "99",
            "LLAMA_CPP_CONTEXT_TOKENS": "16384",
            "TTS_BACKEND": "qwen3",
            "QWEN_TTS_DEVICE": "cuda:0",
            "QWEN_TTS_DTYPE": "float32",
            "QWEN_TTS_ATTENTION": "sdpa",
            "REVIEWER_BACKEND": "qwen_omni",
            "REVIEWER_REQUIRED": "true",
            "QWEN_OMNI_DEVICE": "cuda:0",
            "QWEN_OMNI_DTYPE": "float16",
            "QWEN_OMNI_ATTENTION": "sdpa",
            "QWEN_OMNI_REVIEW_MODEL": "Qwen/Qwen2.5-Omni-3B",
            "QWEN_OMNI_MAX_NEW_TOKENS": "700",
            "AUDIO_WPM_TOLERANCE": "15",
            "VISUAL_IMAGE_BACKEND": "auto",
            "VISUAL_FLUX_MODEL": "black-forest-labs/FLUX.1-schnell",
            "VISUAL_FLUX_STEPS": "4",
            "VISUAL_SDXL_BASE_MODEL": "stabilityai/stable-diffusion-xl-base-1.0",
            "VISUAL_SDXL_LIGHTNING_REPO": "ByteDance/SDXL-Lightning",
            "VISUAL_SDXL_LIGHTNING_CHECKPOINT": "sdxl_lightning_4step_unet.safetensors",
            "VISUAL_SDXL_LIGHTNING_STEPS": "4",
            "WAN22_MODEL_ID": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "WAN22_FRAME_NUM": "81",
            "WAN22_SAMPLE_STEPS": "24",
            "WAN22_GUIDANCE_SCALE": "5.0",
            "VIDEO_WIDTH": "1080",
            "VIDEO_HEIGHT": "1920",
            "VIDEO_FPS": "30",
            "TARGET_SECONDS": "58",
            "PUBLISH_ENABLED": "false",
            "YOUTUBE_PRIVACY_STATUS": "private",
            "TIMEZONE_NAME": "Africa/Cairo",
        }
    )
    .add_local_python_source("factory")
)

factory_secrets = (
    [modal.Secret.from_name("ai-growth-factory-secrets")]
    if os.getenv("MODAL_USE_FACTORY_SECRET", "").strip().lower() == "true"
    else []
)


def _prepare_runtime() -> None:
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)
    from factory.production_runtime import install_production_runtime

    install_production_runtime()


@app.function(
    image=worker_image,
    gpu="A10",
    cpu=8.0,
    memory=65536,
    timeout=85 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
    schedule=modal.Cron("0 10 * * *", timezone="Africa/Cairo"),
)
def daily_factory() -> dict[str, object]:
    _prepare_runtime()
    from factory.config import Settings
    from factory.pipeline import run_factory

    result = run_factory(Settings.from_env())
    state_volume.commit()
    hf_cache.commit()
    return result


@app.function(
    image=worker_image,
    gpu="A10",
    cpu=8.0,
    memory=65536,
    timeout=85 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=0),
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def render_production_canary() -> dict[str, object]:
    _prepare_runtime()
    os.environ["PUBLISH_ENABLED"] = "false"
    from factory.canary import run_production_canary
    from factory.config import Settings

    result = run_production_canary(Settings.from_env(), Path(STATE_DIR) / "canaries")
    state_volume.commit()
    hf_cache.commit()
    if result.get("status") != "verified_render_canary":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


@app.function(
    image=worker_image,
    gpu="A10",
    cpu=8.0,
    memory=65536,
    timeout=85 * 60,
    max_containers=1,
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def run_canary() -> dict[str, object]:
    _prepare_runtime()
    os.environ["YOUTUBE_PRIVACY_STATUS"] = "private"
    from factory.config import Settings
    from factory.pipeline import run_factory

    result = run_factory(Settings.from_env())
    state_volume.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(canary: bool = False, render_canary: bool = False) -> None:
    if canary and render_canary:
        raise ValueError("Choose only one of --canary or --render-canary")
    if render_canary:
        result = render_production_canary.remote()
    elif canary:
        result = run_canary.remote()
    else:
        result = daily_factory.remote()
    print(json.dumps(result, ensure_ascii=False))
