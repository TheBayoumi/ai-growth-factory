from __future__ import annotations

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

# The llama.cpp CUDA image supplies /app/llama-server. Python environments are added
# on top. Qwen3-TTS and Qwen2.5-Omni intentionally share transformers 4.57.3.
worker_image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ffmpeg", "libsndfile1", "git")
    .run_commands(
        "python -m pip install --upgrade pip wheel setuptools",
        "python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "requests==2.32.5",
        "Pillow==11.3.1",
        "imageio-ffmpeg==0.6.0",
        "qwen-tts==0.1.1",
        "qwen-omni-utils>=0.0.8",
        "gptqmodel==2.0.0",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE,
            "HF_HUB_CACHE": MODEL_CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "WORK_ROOT": WORK_DIR,
            "STATE_ROOT": STATE_DIR,
            "LLAMA_CPP_EXECUTABLE": "/app/llama-server",
            "LLAMA_CPP_BASE_URL": "http://127.0.0.1:8080/v1",
            "LLAMA_CPP_MANAGED": "true",
            "LLAMA_CPP_GPU_LAYERS": "99",
            "LLAMA_CPP_CONTEXT_TOKENS": "16384",
            "TTS_BACKEND": "qwen3",
            "QWEN_TTS_DEVICE": "cuda:0",
            "QWEN_TTS_DTYPE": "float16",
            "QWEN_TTS_ATTENTION": "sdpa",
            "REVIEWER_BACKEND": "qwen_omni",
            "QWEN_OMNI_DEVICE": "cuda:0",
            "QWEN_OMNI_DTYPE": "float16",
            "QWEN_OMNI_ATTENTION": "sdpa",
            "QWEN_OMNI_REVIEW_MODEL": "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
            "PUBLISH_ENABLED": "true",
            "YOUTUBE_PRIVACY_STATUS": "private",
            "TIMEZONE_NAME": "Africa/Cairo",
        }
    )
    .add_local_python_source("factory")
)

factory_secret = modal.Secret.from_name("ai-growth-factory-secrets")


def _prepare_runtime() -> None:
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)


@app.function(
    image=worker_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=30 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
    secrets=[factory_secret],
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
    schedule=modal.Cron("0 10 * * *", timezone="Africa/Cairo"),
)
def daily_factory() -> dict[str, object]:
    """Run one private-first autonomous publication each day."""
    _prepare_runtime()
    from factory.config import Settings
    from factory.pipeline import run_factory

    result = run_factory(Settings.from_env())
    state_volume.commit()
    hf_cache.commit()
    return result


@app.function(
    image=worker_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=20 * 60,
    max_containers=1,
    secrets=[factory_secret],
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def run_canary() -> dict[str, object]:
    """Manual canary using the same production image and private upload policy."""
    _prepare_runtime()
    os.environ["YOUTUBE_PRIVACY_STATUS"] = "private"
    from factory.config import Settings
    from factory.pipeline import run_factory

    result = run_factory(Settings.from_env())
    state_volume.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(canary: bool = False) -> None:
    result = run_canary.remote() if canary else daily_factory.remote()
    print(result)
