from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PINNED_VIMAX_COMMIT = "05a48943878312d88fe5a016c12a9654940ecc43"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViMax Script2Video text planning only")
    parser.add_argument("--vimax-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


async def _run(request: dict[str, Any], *, vimax_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(vimax_root))
    try:
        from langchain.chat_models import init_chat_model
        from pipelines.script2video_pipeline import Script2VideoPipeline
        from utils.provider_presets import resolve_chat_model_config
    except Exception as exc:
        raise RuntimeError(f"could not import pinned ViMax planner: {exc}") from exc

    chat_config = request.get("chat_model")
    if not isinstance(chat_config, dict):
        raise ValueError("request.chat_model must be an object")
    chat_model = init_chat_model(**resolve_chat_model_config(dict(chat_config)))
    working_dir = Path(str(request["working_dir"])).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Script2VideoPipeline(
        chat_model=chat_model,
        image_generator=None,
        video_generator=None,
        working_dir=str(working_dir),
    )
    planned = await pipeline.plan_text_artifacts(
        script=str(request["script"]),
        user_requirement=str(request["user_requirement"]),
        style=str(request["style"]),
        progress=None,
        quiet=True,
    )
    return {
        "status": "planned",
        "schema_version": "agf-vimax-plan-v1",
        "vimax_commit": PINNED_VIMAX_COMMIT,
        "characters": _dump(planned["characters"]),
        "storyboard": _dump(planned["storyboard"]),
        "shot_descriptions": _dump(planned["shot_descriptions"]),
        "camera_tree": _dump(planned["camera_tree"]),
        "working_dir": str(working_dir),
    }


def main() -> None:
    args = _args()
    vimax_root = Path(args.vimax_root).resolve()
    request_path = Path(args.request).resolve()
    output_path = Path(args.output).resolve()
    if not (vimax_root / "pipelines" / "script2video_pipeline.py").is_file():
        raise SystemExit(f"invalid ViMax root: {vimax_root}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("vimax_commit") != PINNED_VIMAX_COMMIT:
        raise SystemExit("request ViMax commit does not match pinned adapter")
    payload = asyncio.run(_run(request, vimax_root=vimax_root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
