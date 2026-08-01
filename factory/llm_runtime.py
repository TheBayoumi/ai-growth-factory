from __future__ import annotations

import json
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import requests

from .config import Settings


@contextmanager
def managed_llama_server(settings: Settings, log_dir: Path) -> Iterator[None]:
    def healthy() -> bool:
        try:
            return requests.get(f"{settings.llm_base_url}/models", timeout=5).ok
        except Exception:
            return False

    if healthy():
        yield
        return
    if not settings.llm_managed:
        raise RuntimeError("llama.cpp is unavailable and managed mode is disabled")
    executable = shutil.which(settings.llm_executable) or settings.llm_executable
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "llama-server.log").open("ab") as log:
        process = subprocess.Popen([
            executable, "-hf", settings.llm_hf_model, "--host", "127.0.0.1", "--port", "8080",
            "-c", str(settings.llm_context_tokens), "-ngl", str(settings.llm_gpu_layers), "--jinja",
            "--chat-template-kwargs", json.dumps({"enable_thinking": False}),
        ], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + settings.llm_startup_timeout_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("llama.cpp exited during startup")
                if healthy():
                    yield
                    return
                time.sleep(1.5)
            raise RuntimeError("llama.cpp startup timed out")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
