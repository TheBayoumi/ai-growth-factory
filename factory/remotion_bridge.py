from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .remotion_contract import (
    RemotionContractError,
    RemotionRenderSpec,
    render_manifest_payload,
)


REMOTION_VERSION = "4.0.500"


class RemotionRenderError(RuntimeError):
    """The Remotion subprocess did not produce a verified production artifact."""


def _safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if not suffix or len(suffix) > 8 or not suffix[1:].isalnum():
        raise RemotionContractError(f"unsupported media extension: {path.name}")
    return suffix


def _copy_asset(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise RemotionContractError(f"asset does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def stage_render_assets(
    *,
    spec: RemotionRenderSpec,
    stage_root: Path,
) -> tuple[RemotionRenderSpec, Path]:
    """Copy untrusted absolute paths into one isolated Remotion public directory."""
    spec.validate(require_files=True)
    public_root = stage_root / "public"
    asset_root = public_root / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)

    audio_source = Path(spec.audio_path)
    audio_name = f"narration{_safe_suffix(audio_source)}"
    _copy_asset(audio_source, asset_root / audio_name)

    background_music_name: str | None = None
    if spec.background_music_path:
        music_source = Path(spec.background_music_path)
        background_music_name = f"background-music{_safe_suffix(music_source)}"
        _copy_asset(music_source, asset_root / background_music_name)

    shot_media: dict[int, str] = {}
    shot_keyframes: dict[int, str | None] = {}
    for shot in spec.shots:
        media_source = Path(shot.media_path)
        media_name = f"shot-{shot.shot_id:03d}-media{_safe_suffix(media_source)}"
        _copy_asset(media_source, asset_root / media_name)
        shot_media[shot.shot_id] = f"assets/{media_name}"

        if shot.keyframe_path:
            keyframe_source = Path(shot.keyframe_path)
            keyframe_name = (
                f"shot-{shot.shot_id:03d}-keyframe{_safe_suffix(keyframe_source)}"
            )
            _copy_asset(keyframe_source, asset_root / keyframe_name)
            shot_keyframes[shot.shot_id] = f"assets/{keyframe_name}"
        else:
            shot_keyframes[shot.shot_id] = None

    staged = spec.with_staged_paths(
        audio_path=f"assets/{audio_name}",
        background_music_path=(
            f"assets/{background_music_name}" if background_music_name else None
        ),
        shot_media=shot_media,
        shot_keyframes=shot_keyframes,
    )
    staged_spec_path = stage_root / "render-spec.json"
    staged.write_json(staged_spec_path)
    return staged, staged_spec_path


def _renderer_dir() -> Path:
    configured = os.getenv("REMOTION_RENDERER_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "renderer").resolve()


def render_with_remotion(
    *,
    spec: RemotionRenderSpec,
    output_path: Path,
    workdir: Path,
    timeout_seconds: int = 1200,
) -> tuple[Path, Path, Path]:
    """Stage media, invoke the pinned Node renderer, and persist a digest-bound manifest."""
    renderer_dir = _renderer_dir()
    package_path = renderer_dir / "package.json"
    render_script = renderer_dir / "dist" / "render.js"
    if not package_path.is_file():
        raise RemotionRenderError(f"Remotion package is missing: {package_path}")
    if not render_script.is_file():
        raise RemotionRenderError(
            f"Remotion renderer is not built: {render_script}; run npm ci && npm run build"
        )

    workdir.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix="remotion-stage-", dir=workdir))
    staged_spec, staged_spec_path = stage_render_assets(spec=spec, stage_root=stage_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "remotion-render.log"
    manifest_path = workdir / "remotion-render-manifest.json"

    command = [
        "node",
        str(render_script),
        "--spec",
        str(staged_spec_path),
        "--public-dir",
        str(stage_root / "public"),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=renderer_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        env={**os.environ, "REMOTION_VERSION_EXPECTED": REMOTION_VERSION},
    )
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RemotionRenderError(
            f"Remotion exited with {completed.returncode}: {completed.stderr[-4000:]}"
        )
    if not output_path.is_file() or output_path.stat().st_size < 500_000:
        raise RemotionRenderError("Remotion produced no usable MP4")

    payload: dict[str, Any] = render_manifest_payload(
        spec=staged_spec,
        output_path=output_path,
        staged_spec_path=staged_spec_path,
        renderer_version=REMOTION_VERSION,
    )
    payload["command"] = command
    payload["log"] = str(log_path)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path, manifest_path, log_path
