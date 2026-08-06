from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "vimax-remotion-v1"


class RemotionContractError(ValueError):
    """The cross-language render contract is incomplete or internally inconsistent."""


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CameraSpec:
    shot_size: str
    angle: str
    movement: str

    def validate(self) -> None:
        if not _clean(self.shot_size):
            raise RemotionContractError("camera.shot_size must not be empty")
        if not _clean(self.angle):
            raise RemotionContractError("camera.angle must not be empty")
        if not _clean(self.movement):
            raise RemotionContractError("camera.movement must not be empty")


@dataclass(frozen=True)
class CaptionCue:
    cue_id: int
    start_frame: int
    end_frame: int
    text: str

    def validate(self, *, total_frames: int) -> None:
        if self.cue_id < 0:
            raise RemotionContractError("caption cue_id must be non-negative")
        if not 0 <= self.start_frame < self.end_frame <= total_frames:
            raise RemotionContractError(
                f"caption {self.cue_id} frames are outside the composition: "
                f"{self.start_frame}-{self.end_frame}/{total_frames}"
            )
        if not _clean(self.text):
            raise RemotionContractError(f"caption {self.cue_id} text must not be empty")


@dataclass(frozen=True)
class RenderShot:
    shot_id: int
    start_frame: int
    duration_in_frames: int
    semantic_claim: str
    purpose: str
    first_frame_prompt: str
    last_frame_prompt: str
    motion_prompt: str
    camera: CameraSpec
    renderer: str
    media_path: str
    keyframe_path: str | None
    source_index: int
    seed: int
    reference_assets: tuple[str, ...] = ()

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.duration_in_frames

    def validate(self, *, total_frames: int) -> None:
        if self.shot_id < 0:
            raise RemotionContractError("shot_id must be non-negative")
        if self.start_frame < 0 or self.duration_in_frames <= 0:
            raise RemotionContractError(
                f"shot {self.shot_id} has invalid frame range "
                f"{self.start_frame}+{self.duration_in_frames}"
            )
        if self.end_frame > total_frames:
            raise RemotionContractError(
                f"shot {self.shot_id} extends beyond composition: "
                f"{self.end_frame}>{total_frames}"
            )
        if self.renderer not in {"image_motion", "video_clip"}:
            raise RemotionContractError(
                f"shot {self.shot_id} has unsupported renderer {self.renderer!r}"
            )
        if not _clean(self.semantic_claim):
            raise RemotionContractError(f"shot {self.shot_id} semantic_claim is empty")
        if not _clean(self.first_frame_prompt):
            raise RemotionContractError(f"shot {self.shot_id} first_frame_prompt is empty")
        if not _clean(self.motion_prompt):
            raise RemotionContractError(f"shot {self.shot_id} motion_prompt is empty")
        if self.source_index < 0:
            raise RemotionContractError(f"shot {self.shot_id} source_index is negative")
        self.camera.validate()


@dataclass(frozen=True)
class RemotionRenderSpec:
    schema_version: str
    width: int
    height: int
    fps: int
    duration_in_frames: int
    audio_path: str
    background_music_path: str | None
    title: str
    source_label: str
    shots: tuple[RenderShot, ...]
    captions: tuple[CaptionCue, ...]
    transition_frames: int = 5

    def validate(self, *, require_files: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RemotionContractError(
                f"schema_version must be {SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if self.width < 360 or self.height < 640:
            raise RemotionContractError("render dimensions are below the production minimum")
        if self.width % 2 or self.height % 2:
            raise RemotionContractError("render dimensions must be even for yuv420p")
        if not 12 <= self.fps <= 60:
            raise RemotionContractError("fps must be between 12 and 60")
        if self.duration_in_frames <= 0:
            raise RemotionContractError("duration_in_frames must be positive")
        if not 0 <= self.transition_frames <= max(1, self.fps // 2):
            raise RemotionContractError("transition_frames is outside the supported range")
        if not self.shots:
            raise RemotionContractError("render spec must contain at least one shot")

        expected_start = 0
        for expected_id, shot in enumerate(self.shots):
            shot.validate(total_frames=self.duration_in_frames)
            if shot.shot_id != expected_id:
                raise RemotionContractError(
                    f"shot IDs must be contiguous; expected {expected_id}, got {shot.shot_id}"
                )
            if shot.start_frame != expected_start:
                raise RemotionContractError(
                    f"shot {shot.shot_id} starts at {shot.start_frame}; expected {expected_start}"
                )
            expected_start = shot.end_frame
        if expected_start != self.duration_in_frames:
            raise RemotionContractError(
                f"shots end at frame {expected_start}; composition ends at "
                f"{self.duration_in_frames}"
            )

        previous_start = -1
        for expected_id, cue in enumerate(self.captions):
            cue.validate(total_frames=self.duration_in_frames)
            if cue.cue_id != expected_id:
                raise RemotionContractError(
                    f"caption IDs must be contiguous; expected {expected_id}, got {cue.cue_id}"
                )
            if cue.start_frame < previous_start:
                raise RemotionContractError("captions must be ordered by start_frame")
            previous_start = cue.start_frame

        if require_files:
            audio = Path(self.audio_path)
            if not audio.is_file():
                raise RemotionContractError(f"narration audio does not exist: {audio}")
            if self.background_music_path and not Path(self.background_music_path).is_file():
                raise RemotionContractError(
                    f"background music does not exist: {self.background_music_path}"
                )
            for shot in self.shots:
                media = Path(shot.media_path)
                if not media.is_file():
                    raise RemotionContractError(
                        f"shot {shot.shot_id} media does not exist: {media}"
                    )
                if shot.keyframe_path and not Path(shot.keyframe_path).is_file():
                    raise RemotionContractError(
                        f"shot {shot.shot_id} keyframe does not exist: {shot.keyframe_path}"
                    )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shots"] = [asdict(item) for item in self.shots]
        payload["captions"] = [asdict(item) for item in self.captions]
        return payload

    def sha256(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def write_json(self, path: Path) -> Path:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def with_staged_paths(
        self,
        *,
        audio_path: str,
        background_music_path: str | None,
        shot_media: dict[int, str],
        shot_keyframes: dict[int, str | None],
    ) -> "RemotionRenderSpec":
        shots = tuple(
            replace(
                shot,
                media_path=shot_media[shot.shot_id],
                keyframe_path=shot_keyframes.get(shot.shot_id),
                reference_assets=(),
            )
            for shot in self.shots
        )
        staged = replace(
            self,
            audio_path=audio_path,
            background_music_path=background_music_path,
            shots=shots,
        )
        staged.validate()
        return staged


def _camera_from_text(*values: str) -> CameraSpec:
    text = " ".join(values).casefold()
    movement = "subtle_push_in"
    for needle, selected in (
        ("dolly out", "dolly_out"),
        ("pull out", "dolly_out"),
        ("zoom out", "dolly_out"),
        ("dolly in", "dolly_in"),
        ("push in", "dolly_in"),
        ("zoom in", "dolly_in"),
        ("pan left", "pan_left"),
        ("pan right", "pan_right"),
        ("tilt up", "tilt_up"),
        ("tilt down", "tilt_down"),
        ("static", "static"),
        ("locked", "static"),
    ):
        if needle in text:
            movement = selected
            break

    shot_size = "medium"
    for needle, selected in (
        ("extreme close", "extreme_close_up"),
        ("close-up", "close_up"),
        ("close up", "close_up"),
        ("wide", "wide"),
        ("establish", "establishing"),
        ("macro", "macro"),
        ("over-the-shoulder", "over_the_shoulder"),
    ):
        if needle in text:
            shot_size = selected
            break

    angle = "eye_level"
    for needle, selected in (
        ("low-angle", "low_angle"),
        ("low angle", "low_angle"),
        ("high-angle", "high_angle"),
        ("high angle", "high_angle"),
        ("aerial", "aerial"),
        ("top-down", "top_down"),
        ("top down", "top_down"),
    ):
        if needle in text:
            angle = selected
            break
    return CameraSpec(shot_size=shot_size, angle=angle, movement=movement)


def build_remotion_render_spec(
    *,
    shots: Sequence[Any],
    media: Sequence[Any],
    segments: Sequence[Any],
    package: Any,
    audio_path: Path,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    background_music_path: Path | None = None,
    caption_cues: Sequence[Any] | None = None,
) -> RemotionRenderSpec:
    """Translate the existing or ViMax editorial plan into one frame-authoritative spec."""
    ordered_shots = sorted(shots, key=lambda item: int(item.shot_id))
    ordered_media = sorted(media, key=lambda item: int(item.scene_index))
    if len(ordered_shots) != len(ordered_media):
        raise RemotionContractError("every editorial shot requires one media asset")
    media_by_scene = {int(item.scene_index): item for item in ordered_media}
    if len(media_by_scene) != len(ordered_media):
        raise RemotionContractError("scene media indices must be unique")

    total_frames = round(float(duration_seconds) * fps)
    if total_frames <= 0:
        raise RemotionContractError("audio duration produced no frames")

    starts = [round(float(item.start_seconds) * fps) for item in ordered_shots]
    if starts and starts[0] != 0:
        raise RemotionContractError(f"first shot starts at frame {starts[0]}, expected 0")
    boundaries = [*starts, total_frames]
    render_shots: list[RenderShot] = []
    for index, source_shot in enumerate(ordered_shots):
        asset = media_by_scene.get(index)
        if asset is None:
            raise RemotionContractError(f"missing media for shot {index}")
        duration_frames = boundaries[index + 1] - boundaries[index]
        if duration_frames <= 0:
            raise RemotionContractError(f"shot {index} rounds to zero frames")
        visual_direction = _clean(getattr(source_shot, "visual_direction", ""))
        treatment = _clean(getattr(source_shot, "treatment", ""))
        prompt = _clean(getattr(asset, "director_prompt", "")) or _clean(
            getattr(asset, "prompt", "")
        )
        motion = visual_direction or treatment or "Controlled documentary camera motion"
        camera = _camera_from_text(visual_direction, treatment, motion)
        media_type = _clean(getattr(asset, "media_type", "image"))
        renderer = "video_clip" if media_type == "video" else "image_motion"
        keyframe = getattr(asset, "keyframe_path", None)
        render_shots.append(
            RenderShot(
                shot_id=index,
                start_frame=boundaries[index],
                duration_in_frames=duration_frames,
                semantic_claim=_clean(getattr(source_shot, "semantic_claim", "")),
                purpose=_clean(getattr(source_shot, "treatment", "support_claim"))
                or "support_claim",
                first_frame_prompt=prompt,
                last_frame_prompt=prompt,
                motion_prompt=motion,
                camera=camera,
                renderer=renderer,
                media_path=str(Path(asset.path).resolve()),
                keyframe_path=(str(Path(keyframe).resolve()) if keyframe else None),
                source_index=int(getattr(source_shot, "source_index", 0)),
                seed=int(getattr(source_shot, "seed", getattr(asset, "seed", 0))),
            )
        )

    caption_source = (
        list(caption_cues)
        if caption_cues is not None
        else sorted(segments, key=lambda item: int(item.segment_id))
    )
    captions: list[CaptionCue] = []
    for cue_id, cue in enumerate(caption_source):
        start_seconds = getattr(cue, "start_seconds", 0.0)
        end_seconds = getattr(cue, "end_seconds", 0.0)
        start = max(0, round(float(start_seconds) * fps))
        end = min(total_frames, max(start + 1, round(float(end_seconds) * fps)))
        captions.append(
            CaptionCue(
                cue_id=cue_id,
                start_frame=start,
                end_frame=end,
                text=_clean(getattr(cue, "text", "")),
            )
        )

    source_publishers = [
        _clean(value) for value in getattr(package, "source_publishers", []) if _clean(value)
    ]
    source_label = " • ".join(dict.fromkeys(source_publishers))
    spec = RemotionRenderSpec(
        schema_version=SCHEMA_VERSION,
        width=width,
        height=height,
        fps=fps,
        duration_in_frames=total_frames,
        audio_path=str(audio_path.resolve()),
        background_music_path=(
            str(background_music_path.resolve()) if background_music_path else None
        ),
        title=_clean(getattr(package, "title", "")),
        source_label=source_label,
        shots=tuple(render_shots),
        captions=tuple(captions),
    )
    spec.validate(require_files=True)
    return spec


def render_manifest_payload(
    *,
    spec: RemotionRenderSpec,
    output_path: Path,
    staged_spec_path: Path,
    renderer_version: str,
) -> dict[str, Any]:
    return {
        "renderer": "remotion",
        "renderer_version": renderer_version,
        "schema_version": spec.schema_version,
        "render_spec_sha256": spec.sha256(),
        "staged_render_spec": str(staged_spec_path),
        "output": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "width": spec.width,
        "height": spec.height,
        "fps": spec.fps,
        "duration_in_frames": spec.duration_in_frames,
        "shot_count": len(spec.shots),
        "caption_count": len(spec.captions),
        "source_asset_looping": False,
        "destructive_caption_matte": False,
    }
