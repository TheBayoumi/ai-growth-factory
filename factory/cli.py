from __future__ import annotations

import argparse
import json
import shutil
import sys

from . import __version__
from .config import Settings
from .local_llm import healthcheck
from .pipeline import run_factory


def doctor(settings: Settings) -> dict:
    checks = {"version": __version__, "ffmpeg": shutil.which("ffmpeg") or "imageio-ffmpeg", "setup": settings.setup_status}
    try:
        checks["local_llm"] = healthcheck(settings)
    except Exception as exc:
        checks["local_llm"] = {"ok": False, "error": str(exc)}
    checks["ok"] = bool(checks["local_llm"].get("ok"))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-growth-factory")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("run")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        result = doctor(settings) if args.command == "doctor" else run_factory(settings)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok", True) else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
