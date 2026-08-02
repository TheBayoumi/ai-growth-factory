from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .image_generator import KeyframeAsset
from .visual_prompt import SceneVisualPrompt, VisualPlan


class VideoGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneMediaAsset:
    scene_index: int
    media_type: str
    path: Path
    keyframe_path: Path
    model: str
    seed: int
    prompt: str
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["keyframe_path"] = str(self.keyframe_path)
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _keyframe_by_scene(keyframes: tuple[KeyframeAsset, ...]) -> dict[int, KeyframeAsset]:
    result = {asset.scene_index: asset for asset in keyframes}
    if len(result) != len(keyframes):
        raise VideoGenerationError("Duplicate keyframe scene index")
    return result


class Wan22Animator:
    """Invoke the official Wan2.2 TI2V-5B reference implementation.

    The official project supports 704x1280 TI2V generation on a 24 GB GPU with
    model offloading, dtype conversion, and the T5 encoder on CPU. This wrapper
    intentionally uses that reference CLI instead of an unverified custom port.
    """

    def __init__(self, plan: VisualPlan) -> None:
        self.plan = plan
        self.repo = Path(os.getenv("WAN22_REPO", "/opt/Wan2.2")).expanduser()
        self.checkpoint = Path(
            os.getenv(
                "WAN22_CHECKPOINT_DIR",
                "/cache/huggingface/Wan-AI/Wan2.2-TI2V-5B",
            )
        ).expanduser()
        self.python = os.getenv("WAN22_PYTHON", "python").strip()
        self.steps = int(os.getenv("WAN22_SAMPLE_STEPS", "30"))
        self.frame_num = int(os.getenv("WAN22_FRAME_NUM", "81"))
        self.timeout_seconds = int(os.getenv("WAN22_TIMEOUT_SECONDS", "1800"))
        if self.frame_num < 17 or (self.frame_num - 1) % 4 != 0:
            raise VideoGenerationError("WAN22_FRAME_NUM must be 4n+1 and at least 17")
        if not 8 <= self.steps <= 60:
            raise VideoGenerationError("WAN22_SAMPLE_STEPS must be between 8 and 60")

    def _validate_runtime(self) -> None:
        generate_py = self.repo / "generate.py"
        if not generate_py.is_file():
            raise VideoGenerationError(
                f"Official Wan2.2 repository is missing at {self.repo}"
            )
        if not self.checkpoint.is_dir():
            raise VideoGenerationError(
                f"Wan2.2 checkpoint is missing at {self.checkpoint}"
            )

    def animate(
        self,
        scene: SceneVisualPrompt,
        keyframe: KeyframeAsset,
        output: Path,
    ) -> Path:
        self._validate_runtime()
        output.parent.mkdir(parents=True, exist_ok=True)
        prompt = (
            scene.motion_prompt
            + " Visual content and composition must remain faithful to this initial keyframe: "
            + scene.image_prompt
            + " Avoid: "
            + scene.negative_prompt
        )
        command = [
            self.python,
            str(self.repo / "generate.py"),
            "--task",
            "ti2v-5B",
            "--size",
            "704*1280",
            "--ckpt_dir",
            str(self.checkpoint),
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--image",
            str(keyframe.path),
            "--prompt",
            prompt,
            "--frame_num",
            str(self.frame_num),
            "--sample_steps",
            str(self.steps),
            "--base_seed",
            str(scene.seed),
            "--save_file",
            str(output),
        ]
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        completed = subprocess.run(
            command,
            cwd=self.repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        log_path = output.with_suffix(".wan.log")
        log_path.write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise VideoGenerationError(
                f"Wan2.2 failed for scene {scene.scene_index}: {completed.stderr[-2200:]}"
            )
        if not output.is_file() or output.stat().st_size < 250_000:
            raise VideoGenerationError(
                f"Wan2.2 produced no usable clip for scene {scene.scene_index}"
            )
        return output


def generate_scene_media(
    plan: VisualPlan,
    keyframes: tuple[KeyframeAsset, ...],
    output_dir: Path,
) -> tuple[SceneMediaAsset, ...]:
    """Generate Wan clips for hero scenes and preserve keyframes for supporting scenes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_map = _keyframe_by_scene(keyframes)
    animator = Wan22Animator(plan)
    assets: list[SceneMediaAsset] = []

    for scene in plan.scenes:
        keyframe = keyframe_map.get(scene.scene_index)
        if keyframe is None:
            raise VideoGenerationError(f"Missing keyframe for scene {scene.scene_index}")
        if scene.generation_mode == "wan_i2v":
            path = output_dir / f"scene-{scene.scene_index:02d}-wan.mp4"
            animator.animate(scene, keyframe, path)
            media_type = "video"
            model = plan.video_model
            prompt = scene.motion_prompt
        elif scene.generation_mode == "image":
            path = keyframe.path
            media_type = "image"
            model = keyframe.model
            prompt = scene.image_prompt
        else:
            raise VideoGenerationError(
                f"Unsupported generation mode: {scene.generation_mode}"
            )
        assets.append(
            SceneMediaAsset(
                scene_index=scene.scene_index,
                media_type=media_type,
                path=path,
                keyframe_path=keyframe.path,
                model=model,
                seed=scene.seed,
                prompt=prompt,
                sha256=_sha256(path),
            )
        )

    if sum(asset.media_type == "video" for asset in assets) != 3:
        raise VideoGenerationError("Production visual plan must contain exactly three Wan clips")

    manifest = {
        "video_backend": "wan22_ti2v_official",
        "video_model": plan.video_model,
        "frame_num": animator.frame_num,
        "sample_steps": animator.steps,
        "offload_model": True,
        "convert_model_dtype": True,
        "t5_cpu": True,
        "assets": [asset.as_dict() for asset in assets],
    }
    (output_dir / "scene-media-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tuple(assets)
