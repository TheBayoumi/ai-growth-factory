from __future__ import annotations

import json
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from .config import Settings
from .local_llm import healthcheck


class LLMRuntimeError(RuntimeError):
    pass


def _is_healthy(settings: Settings) -> bool:
    try:
        return bool(healthcheck(settings).get("ok"))
    except Exception:
        return False


def _command(settings: Settings) -> list[str]:
    parsed = urlparse(settings.llm_base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise LLMRuntimeError("Managed llama.cpp requires a localhost LLAMA_CPP_BASE_URL")
    port = parsed.port or 8080
    executable = shutil.which(settings.llm_executable)
    if not executable:
        candidate = Path(settings.llm_executable).expanduser()
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        raise LLMRuntimeError(
            f"Could not find {settings.llm_executable}. Install llama.cpp or set LLAMA_CPP_EXECUTABLE."
        )
    return [
        executable,
        "-hf",
        settings.llm_hf_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(settings.llm_context_tokens),
        "-ngl",
        str(settings.llm_gpu_layers),
        "--jinja",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
    ]


@contextmanager
def managed_llama_server(settings: Settings, log_dir: Path) -> Iterator[None]:
    if _is_healthy(settings):
        yield
        return
    if not settings.llm_managed:
        raise LLMRuntimeError(
            f"llama.cpp is not reachable at {settings.llm_base_url} and LLAMA_CPP_MANAGED is false"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llama-server.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            _command(settings),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + settings.llm_startup_timeout_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise LLMRuntimeError(
                        f"llama.cpp exited during startup with code {process.returncode}; log: {log_path}"
                    )
                if _is_healthy(settings):
                    yield
                    return
                time.sleep(1.5)
            raise LLMRuntimeError(
                f"llama.cpp did not become ready within {settings.llm_startup_timeout_seconds}s; log: {log_path}"
            )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
