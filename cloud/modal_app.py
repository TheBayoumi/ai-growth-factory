from __future__ import annotations

import json
import os
import traceback
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
        "clang",
        "build-essential",
        "pkg-config",
        "libpcre2-dev",
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
        "imageio-ffmpeg==0.6.0",
        "numpy==2.2.6",
        "soundfile>=0.12,<0.14",
        "qwen-tts==0.1.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "optimum==2.2.0",
        "qwen-omni-utils[decord]>=0.0.8",
    )
    # GPTQModel 5.7 exports the METHOD API used by Optimum. Modal image builders
    # do not expose a GPU or nvcc, so disable its optional compiled CUDA extension.
    # PyPcre is a transitive GPTQModel dependency and builds a native extension on
    # Python 3.12, so clang, pkg-config and PCRE2 development headers are installed
    # in the base image before this layer.
    .run_commands(
        "BUILD_CUDA_EXT=0 python -m pip install --no-build-isolation gptqmodel==5.7.0",
        "python -m pip install --force-reinstall numpy==2.2.6 numba==0.64.0",
        (
            "python -c \"from importlib.metadata import version; "
            "import decord, numba, numpy, torch, torchvision; "
            "from gptqmodel.quantization import METHOD; "
            "from optimum.gptq import GPTQQuantizer; "
            "from qwen_tts import Qwen3TTSModel; "
            "from qwen_omni_utils import process_mm_info; "
            "from transformers import Qwen2_5OmniForConditionalGeneration, "
            "Qwen2_5OmniProcessor; "
            "from transformers.utils import is_optimum_available; "
            "assert version('gptqmodel') == '5.7.0', version('gptqmodel'); "
            "assert version('optimum') == '2.2.0', version('optimum'); "
            "assert is_optimum_available(), 'Transformers cannot detect Optimum'; "
            "assert METHOD.GPTQ is not None, 'GPTQModel METHOD.GPTQ is unavailable'; "
            "assert GPTQQuantizer is not None, 'Optimum GPTQQuantizer is unavailable'; "
            "assert numpy.__version__ == '2.2.6', numpy.__version__; "
            "assert numba.__version__ == '0.64.0', numba.__version__; "
            "print('Qwen TTS and Optimum GPTQ integration preflight passed')\""
        ),
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
            "QWEN_TTS_DTYPE": "float32",
            "QWEN_TTS_ATTENTION": "sdpa",
            "REVIEWER_BACKEND": "qwen_omni",
            "REVIEWER_REQUIRED": "true",
            "QWEN_OMNI_DEVICE": "cuda:0",
            "QWEN_OMNI_DTYPE": "float16",
            "QWEN_OMNI_ATTENTION": "sdpa",
            "QWEN_OMNI_REVIEW_MODEL": "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
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

# Verification deployments do not require account secrets. Production publishing
# opts into the named secret by setting MODAL_USE_FACTORY_SECRET=true at deploy time.
factory_secrets = (
    [modal.Secret.from_name("ai-growth-factory-secrets")]
    if os.getenv("MODAL_USE_FACTORY_SECRET", "").strip().lower() == "true"
    else []
)


def _prepare_runtime() -> None:
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)


@app.function(
    image=worker_image,
    gpu="T4",
    cpu=1.0,
    memory=4096,
    timeout=5 * 60,
    max_containers=1,
)
def reviewer_runtime_probe() -> dict[str, object]:
    """Verify the production imports and return only JSON-safe primitives."""
    _prepare_runtime()
    try:
        from importlib.metadata import version as package_version

        import decord
        import numba
        import numpy
        import torch
        import torchaudio
        import torchvision
        import transformers
        from gptqmodel.quantization import METHOD
        from optimum.gptq import GPTQQuantizer
        from qwen_tts import Qwen3TTSModel
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
        from transformers.utils import is_optimum_available
        from transformers.utils.quantization_config import GPTQConfig

        if numpy.__version__ != "2.2.6":
            raise RuntimeError(f"Unexpected NumPy version: {numpy.__version__}")
        if numba.__version__ != "0.64.0":
            raise RuntimeError(f"Unexpected Numba version: {numba.__version__}")
        if package_version("gptqmodel") != "5.7.0":
            raise RuntimeError(
                f"Unexpected GPTQModel version: {package_version('gptqmodel')}"
            )
        if package_version("optimum") != "2.2.0":
            raise RuntimeError(
                f"Unexpected Optimum version: {package_version('optimum')}"
            )
        if not is_optimum_available():
            raise RuntimeError("Transformers cannot detect the Optimum installation")
        if METHOD.GPTQ is None or GPTQQuantizer is None:
            raise RuntimeError("GPTQModel METHOD or Optimum GPTQQuantizer is unavailable")

        quantizer = GptqHfQuantizer(GPTQConfig(bits=4))
        quantizer.validate_environment()

        del quantizer
        del METHOD
        del GPTQQuantizer
        del Qwen3TTSModel
        del process_mm_info
        del Qwen2_5OmniForConditionalGeneration
        del Qwen2_5OmniProcessor
        probe = {
            "ok": True,
            "gptq_environment_valid": True,
            "gptq_method_api": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_capability": [
                int(value) for value in torch.cuda.get_device_capability(0)
            ]
            if torch.cuda.is_available()
            else None,
            "gpu_name": str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available()
            else None,
            "versions": {
                "torch": str(torch.__version__),
                "torchaudio": str(torchaudio.__version__),
                "torchvision": str(torchvision.__version__),
                "transformers": str(transformers.__version__),
                "gptqmodel": str(package_version("gptqmodel")),
                "optimum": str(package_version("optimum")),
                "numpy": str(numpy.__version__),
                "numba": str(numba.__version__),
                "decord": str(decord.__version__),
            },
        }
        # Modal serializes return values for the local GitHub runner. Round-trip
        # through JSON so framework-specific objects can never leak to the client.
        return json.loads(json.dumps(probe, default=str))
    except Exception as exc:
        raise RuntimeError(
            "Qwen reviewer T4 runtime probe failed: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        ) from exc


@app.function(
    image=worker_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=30 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
    schedule=modal.Cron("0 10 * * *", timezone="Africa/Cairo"),
)
def daily_factory() -> dict[str, object]:
    """Run one private-first autonomous publication each day when secrets enable it."""
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
    timeout=30 * 60,
    max_containers=1,
    retries=modal.Retries(max_retries=0),
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def render_production_canary() -> dict[str, object]:
    """Run the real generation/review/render stack and export artifacts without publishing."""
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
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=20 * 60,
    max_containers=1,
    secrets=factory_secrets,
    volumes={MODEL_CACHE: hf_cache, STATE_DIR: state_volume},
)
def run_canary() -> dict[str, object]:
    """Run the publication path with private visibility after owner credentials exist."""
    _prepare_runtime()
    os.environ["YOUTUBE_PRIVACY_STATUS"] = "private"
    from factory.config import Settings
    from factory.pipeline import run_factory

    result = run_factory(Settings.from_env())
    state_volume.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(
    canary: bool = False,
    render_canary: bool = False,
    reviewer_probe: bool = False,
) -> None:
    selected = sum((canary, render_canary, reviewer_probe))
    if selected > 1:
        raise ValueError("Choose only one of --canary, --render-canary, or --reviewer-probe")
    if reviewer_probe:
        result = reviewer_runtime_probe.remote()
    elif render_canary:
        result = render_production_canary.remote()
    elif canary:
        result = run_canary.remote()
    else:
        result = daily_factory.remote()
    print(json.dumps(result, ensure_ascii=False))
