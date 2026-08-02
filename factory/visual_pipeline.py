from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .image_generator import KeyframeAsset, generate_keyframes
from .models import NarrationSegment, VideoPackage
from .video_generator import SceneMediaAsset, generate_scene_media
from .visual_compositor import compose_platform_video
from .visual_prompt import VisualPlan


@dataclass(frozen=True)
class VisualPipelineOutput:
    video_path: Path
    thumbnail_path: Path
    caption_path: Path
    visual_plan_path: Path
    keyframes: tuple[KeyframeAsset, ...]
    scene_media: tuple[SceneMediaAsset, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "thumbnail_path": str(self.thumbnail_path),
            "caption_path": str(self.caption_path),
            "visual_plan_path": str(self.visual_plan_path),
            "keyframes": [asset.as_dict() for asset in self.keyframes],
            "scene_media": [asset.as_dict() for asset in self.scene_media],
        }


def release_accelerator_memory() -> None:
    """Release sequential model allocations before the next production stage."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        return


def persist_visual_plan(plan: VisualPlan, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def render_visual_plan(
    *,
    plan: VisualPlan,
    package: VideoPackage,
    segments: Sequence[NarrationSegment],
    audio_path: Path,
    workdir: Path,
    output_width: int = 1080,
    output_height: int = 1920,
    output_fps: int = 30,
) -> VisualPipelineOutput:
    """Generate text-free media, animate hero scenes, and add captions separately."""
    visual_root = workdir / "visual-assets"
    keyframe_dir = visual_root / "keyframes"
    scene_media_dir = visual_root / "scene-media"
    render_dir = visual_root / "render"
    plan_path = persist_visual_plan(plan, visual_root / "visual-plan.json")

    keyframes = generate_keyframes(plan, keyframe_dir)
    release_accelerator_memory()
    scene_media = generate_scene_media(plan, keyframes, scene_media_dir)
    release_accelerator_memory()
    video_path, thumbnail_path, caption_path = compose_platform_video(
        media=scene_media,
        segments=segments,
        package=package,
        audio_path=audio_path,
        workdir=render_dir,
        width=output_width,
        height=output_height,
        fps=output_fps,
    )
    output = VisualPipelineOutput(
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        caption_path=caption_path,
        visual_plan_path=plan_path,
        keyframes=keyframes,
        scene_media=scene_media,
    )
    (visual_root / "visual-pipeline-manifest.json").write_text(
        json.dumps(output.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output
