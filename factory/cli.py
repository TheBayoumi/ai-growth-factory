from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import Settings
from .local_llm import healthcheck
from .pipeline import run_factory
from .voice_pipeline import build_reviewed_narration


def _doctor(settings: Settings, *, load_voice: bool) -> dict[str, object]:
    checks: dict[str, object] = {
        "version": __version__,
        "configuration": "ok",
        "ffmpeg": shutil.which("ffmpeg") or "bundled through imageio-ffmpeg",
        "setup": settings.setup_status,
    }
    try:
        checks["local_llm"] = healthcheck(settings)
    except Exception as exc:
        checks["local_llm"] = {"ok": False, "error": str(exc)}
    try:
        import torch
        import qwen_tts  # noqa: F401
        import soundfile  # noqa: F401

        checks["qwen3_tts"] = {
            "ok": True,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": settings.qwen_tts_device,
            "model": settings.qwen_tts_model,
        }
        if load_voice:
            from .qwen_tts import Qwen3TTS

            provider = Qwen3TTS(settings)
            provider._load()
            provider.unload()
            checks["qwen3_tts"]["model_load"] = "ok"
    except Exception as exc:
        checks["qwen3_tts"] = {"ok": False, "error": str(exc)}
    reviewer_configured = True
    reviewer_error = None
    if settings.reviewer_required and settings.reviewer_backend == "openai":
        reviewer_configured = bool(settings.openai_api_key)
    elif settings.reviewer_required and settings.reviewer_backend == "qwen_omni":
        try:
            import qwen_omni_utils  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            reviewer_configured = False
            reviewer_error = str(exc)
    checks["reviewer"] = {
        "required": settings.reviewer_required,
        "backend": settings.reviewer_backend,
        "configured": reviewer_configured,
        "model": settings.reviewer_model,
        "error": reviewer_error,
    }
    checks["youtube"] = {"configured": bool(settings.youtube_oauth)}
    checks["ok"] = bool(
        isinstance(checks["local_llm"], dict)
        and checks["local_llm"].get("ok")
        and isinstance(checks["qwen3_tts"], dict)
        and checks["qwen3_tts"].get("ok")
        and (not settings.reviewer_required or reviewer_configured)
    )
    return checks


def _voice_test(settings: Settings, script_path: Path, output_dir: Path, no_review: bool) -> dict[str, object]:
    narration = script_path.read_text(encoding="utf-8").strip()
    if not narration:
        raise ValueError("The voice-test script is empty")
    effective = replace(settings, reviewer_required=False) if no_review else settings
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_reviewed_narration(effective, narration, output_dir)
    return {
        "status": "approved",
        "audio_path": str(result.audio_path),
        "manifest_path": str(result.manifest_path),
        "attempts": result.attempts,
        "metrics": result.metrics.as_dict(),
        "review": result.review.as_dict() if result.review else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-growth-factory")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Validate local services and credentials")
    doctor.add_argument("--load-voice-model", action="store_true")
    sub.add_parser("run", help="Execute the complete autonomous pipeline once")
    voice = sub.add_parser("voice-test", help="Generate and review narration from a text file")
    voice.add_argument("script", type=Path)
    voice.add_argument("--output-dir", type=Path)
    voice.add_argument("--no-review", action="store_true")
    args = parser.parse_args(argv)
    try:
        # The feature-gated ViMax and Remotion adapters must be installed before execution.
        # The installer also repairs already-imported CLI module bindings deterministically.
        from .production_runtime import install_production_runtime

        install_production_runtime()
        settings = Settings.from_env()
        if args.command == "doctor":
            result = _doctor(settings, load_voice=args.load_voice_model)
        elif args.command == "run":
            result = run_factory(settings)
        else:
            output = args.output_dir or Path(tempfile.mkdtemp(prefix="voice-test-"))
            result = _voice_test(settings, args.script, output, args.no_review)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok", True) else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
