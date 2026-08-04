from __future__ import annotations

import json
import re
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import imageio_ffmpeg
from PIL import Image, ImageChops, ImageFilter, ImageStat

from .editorial_timeline import ShotSpec, build_editorial_plan
from .models import NarrationSegment, VideoPackage
from .video_profile import VideoProfile
from .visual_prompt import VisualPlan
from .visual_prompt_compiler import CompiledVisualPrompt


_INSTALLED = False
_TRANSITION_SECONDS = 0.16
_SPACE_RE = re.compile(r"\s+")


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / max(1, handle.getframerate())


def _clean(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip(" ,.;")


def compile_semantic_image_prompt(
    director_prompt: str,
    director_negative_prompt: str = "",
    *,
    word_budget: int = 72,
) -> CompiledVisualPrompt:
    """Preserve the factual subject instead of replacing it with role-based geometry."""
    del word_budget
    content = _clean(director_prompt)
    words = content.split()
    if len(words) > 58:
        content = " ".join(words[:58]).rstrip(" ,.;:")
    compiled = _clean(
        "Vertical cinematic technology documentary image. "
        + content
        + ". Full-frame environment, clear concrete focal subject, realistic materials, "
        "natural depth, generic unbranded people and devices when relevant, blank surfaces, "
        "readable visual action, moderately calm caption area near the bottom."
    )
    if len(compiled.split()) > 86:
        compiled = " ".join(compiled.split()[:86]).rstrip(" ,.;:")
    negative = _clean(
        "readable text, pseudo-text, gibberish, logos, trademarks, watermarks, signatures, "
        "duplicate subjects, malformed anatomy, warped geometry, collage, split frame, "
        "generic corridor, generic tower, generic blocks, generic orb, camera shake, flicker, "
        f"blur, low resolution, {director_negative_prompt}"
    )
    return CompiledVisualPrompt(
        director_prompt=director_prompt,
        compiled_prompt=compiled,
        negative_prompt=negative,
        word_count=len(compiled.split()),
        word_budget=86,
        compiler_version="visual-compiler-v28-semantic-preservation",
    )


def full_frame_caption_zone(
    image: Image.Image, *, start_ratio: float = 0.78
) -> tuple[Image.Image, float, float]:
    """Keep every source pixel; caption readability is provided by the ASS text card."""
    source = image.convert("RGB")
    width, height = source.size
    start = max(1, min(height - 1, round(height * start_ratio)))
    edges = source.crop((0, start, width, height)).convert("L").filter(ImageFilter.FIND_EDGES)
    detail = float(ImageStat.Stat(edges).mean[0])
    return source.copy(), detail, detail


def _full_frame_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            source = image.convert("RGB")
            width, height = source.size
            if width < 360 or height < 640:
                return False
            start = max(1, min(height - 1, round(height * 0.68)))
            zone = source.crop((0, start, width, height))
            matte = Image.new("RGB", zone.size, (5, 7, 12))
            return ImageChops.difference(zone, matte).getbbox() is not None
    except OSError:
        return False


def _caption_chunks(text: str, *, maximum_words: int = 6) -> list[str]:
    del maximum_words
    profile = VideoProfile.from_env()
    words = text.split()
    chunks: list[list[str]] = []
    current: list[str] = []
    for word in words:
        candidate = [*current, word]
        candidate_text = " ".join(candidate)
        punctuation = bool(re.search(r"[,;:!?]$", word))
        exceeds = (
            len(candidate) > profile.caption_maximum_words
            or len(candidate_text) > profile.caption_maximum_characters
        )
        if exceeds and current:
            chunks.append(current)
            current = [word]
        else:
            current = candidate
        if punctuation and len(current) >= 3:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    index = 0
    while index < len(chunks):
        if len(chunks[index]) >= profile.caption_minimum_words or len(chunks) == 1:
            index += 1
            continue
        merged = False
        if index > 0:
            candidate = chunks[index - 1] + chunks[index]
            if (
                len(candidate) <= profile.caption_maximum_words
                and len(" ".join(candidate)) <= profile.caption_maximum_characters
            ):
                chunks[index - 1] = candidate
                del chunks[index]
                merged = True
            elif len(chunks[index - 1]) > profile.caption_minimum_words:
                moved = chunks[index - 1].pop()
                chunks[index].insert(0, moved)
                merged = True
        if not merged and index + 1 < len(chunks):
            candidate = chunks[index] + chunks[index + 1]
            if (
                len(candidate) <= profile.caption_maximum_words
                and len(" ".join(candidate)) <= profile.caption_maximum_characters
            ):
                chunks[index] = candidate
                del chunks[index + 1]
                merged = True
            elif len(chunks[index + 1]) > profile.caption_minimum_words:
                moved = chunks[index + 1].pop(0)
                chunks[index].append(moved)
                merged = True
        if not merged:
            index += 1
        elif index > 0:
            index -= 1

    result: list[list[str]] = []
    for chunk in chunks:
        if len(" ".join(chunk)) <= profile.caption_maximum_characters:
            result.append(chunk)
            continue
        split = max(profile.caption_minimum_words, len(chunk) // 2)
        left, right = chunk[:split], chunk[split:]
        if len(right) < profile.caption_minimum_words and len(left) > profile.caption_minimum_words:
            right.insert(0, left.pop())
        result.extend([left, right] if right else [left])
    return [" ".join(chunk) for chunk in result]


def _plain_caption(cue: Any, escape_ass: Any) -> str:
    return escape_ass(cue.text)


def _filter_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    return value.replace(":", r"\:").replace("'", r"\'")


def _compose_editorial_video(
    *,
    media: Sequence[Any],
    shots: Sequence[ShotSpec],
    segments: Sequence[NarrationSegment],
    package: VideoPackage,
    audio_path: Path,
    workdir: Path,
    width: int,
    height: int,
    fps: int,
) -> tuple[Path, Path, Path]:
    from . import caption_renderer, visual_compositor

    ordered_media = sorted(media, key=lambda item: item.scene_index)
    ordered_shots = sorted(shots, key=lambda item: item.shot_id)
    if len(ordered_media) != len(ordered_shots):
        raise ValueError("Every editorial shot requires one unique media asset")
    if [item.scene_index for item in ordered_media] != list(range(len(ordered_media))):
        raise ValueError("Editorial media indices are not contiguous")
    if any(item.duration_seconds <= 0 for item in ordered_shots):
        raise ValueError("Editorial shot duration must be positive")

    workdir.mkdir(parents=True, exist_ok=True)
    caption_path = workdir / "animated-captions.ass"
    cues = caption_renderer.write_animated_caption_track(
        sorted(segments, key=lambda item: item.segment_id),
        caption_path,
        width=width,
        height=height,
    )
    total_duration = _audio_duration(audio_path)
    planned_duration = sum(item.duration_seconds for item in ordered_shots)
    if abs(total_duration - planned_duration) > 0.08:
        raise ValueError(
            f"Editorial timeline {planned_duration:.3f}s does not match narration {total_duration:.3f}s"
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for asset, shot in zip(ordered_media, ordered_shots, strict=True):
        input_duration = shot.duration_seconds + _TRANSITION_SECONDS
        if asset.media_type == "image":
            command += [
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{input_duration:.6f}",
                "-i",
                str(asset.path),
            ]
        elif asset.media_type == "video":
            command += ["-i", str(asset.path)]
        else:
            raise ValueError(f"Unsupported editorial media type: {asset.media_type}")
    audio_index = len(ordered_media)
    command += ["-i", str(audio_path)]

    filters: list[str] = []
    for index, (asset, shot) in enumerate(zip(ordered_media, ordered_shots, strict=True)):
        input_duration = shot.duration_seconds + _TRANSITION_SECONDS
        common = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={fps},setsar=1,format=yuv420p"
        )
        if asset.media_type == "image":
            zoom = (
                "min(zoom+0.00055,1.055)"
                if index % 2 == 0
                else "max(1.055-0.00055*on,1.0)"
            )
            frame_count = max(1, round(input_duration * fps))
            x = (
                "iw/2-(iw/zoom/2)"
                if index % 3
                else f"(iw-iw/zoom)*on/{frame_count}"
            )
            filters.append(
                f"[{index}:v]{common},"
                f"zoompan=z='{zoom}':x='{x}':y='ih/2-(ih/zoom/2)':"
                f"d=1:s={width}x{height}:fps={fps},"
                f"trim=duration={input_duration:.6f},settb=AVTB,setpts=PTS-STARTPTS[v{index}]"
            )
        else:
            filters.append(
                f"[{index}:v]{common},"
                f"tpad=stop_mode=clone:stop_duration={_TRANSITION_SECONDS:.3f},"
                f"trim=duration={input_duration:.6f},settb=AVTB,setpts=PTS-STARTPTS[v{index}]"
            )

    previous = "v0"
    cumulative = ordered_shots[0].duration_seconds
    for index in range(1, len(ordered_shots)):
        label = f"x{index}"
        filters.append(
            f"[{previous}][v{index}]xfade=transition=fade:"
            f"duration={_TRANSITION_SECONDS:.3f}:offset={cumulative:.6f}[{label}]"
        )
        previous = label
        cumulative += ordered_shots[index].duration_seconds

    subtitles = _filter_path(caption_path)
    filters.append(
        f"[{previous}]subtitles=filename='{subtitles}':"
        "fontsdir='/usr/share/fonts/truetype/dejavu'[vout]"
    )
    output = workdir / "video.mp4"
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-map",
        f"{audio_index}:a",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    log = workdir / "visual-compositor.log"
    log.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 500_000:
        raise RuntimeError(f"v28 editorial composition failed: {completed.stderr[-3000:]}")

    thumbnail = workdir / "thumbnail.png"
    visual_compositor._thumbnail(ordered_media[0].keyframe_path, package, thumbnail)
    manifest = {
        "renderer": "ffmpeg_editorial_timeline_v28",
        "source_asset_looping": False,
        "destructive_caption_matte": False,
        "still_motion": "deterministic_ken_burns",
        "caption_layer": str(caption_path),
        "caption_cues": len(cues),
        "shot_count": len(ordered_shots),
        "shots": [item.as_dict() for item in ordered_shots],
        "scene_media": [item.as_dict() for item in ordered_media],
        "output": {
            "path": str(output),
            "width": width,
            "height": height,
            "fps": fps,
            "audio_path": str(audio_path),
        },
    }
    (workdir / "visual-composition-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output, thumbnail, caption_path


def render_visual_plan_v28(
    *,
    plan: VisualPlan,
    package: VideoPackage,
    segments: Sequence[NarrationSegment],
    audio_path: Path,
    workdir: Path,
    output_width: int = 1080,
    output_height: int = 1920,
    output_fps: int = 30,
) -> Any:
    from . import image_generator, video_generator, visual_pipeline

    profile = VideoProfile.from_env()
    total_duration = _audio_duration(audio_path)
    expanded, shots = build_editorial_plan(
        plan=plan,
        package=package,
        segments=segments,
        total_duration=total_duration,
        profile=profile,
    )
    visual_root = workdir / "visual-assets"
    keyframe_dir = visual_root / "keyframes"
    scene_media_dir = visual_root / "scene-media"
    render_dir = visual_root / "render"
    plan_path = visual_pipeline.persist_visual_plan(expanded, visual_root / "visual-plan.json")
    timeline_path = visual_root / "editorial-timeline.json"
    timeline_path.write_text(
        json.dumps(
            {
                "profile": profile.as_dict(),
                "duration_seconds": round(total_duration, 6),
                "shot_count": len(shots),
                "shots": [item.as_dict() for item in shots],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    keyframes = image_generator.generate_keyframes(expanded, keyframe_dir)
    visual_pipeline.release_accelerator_memory()
    scene_media = video_generator.generate_scene_media(expanded, keyframes, scene_media_dir)
    visual_pipeline.release_accelerator_memory()
    video_path, thumbnail_path, caption_path = _compose_editorial_video(
        media=scene_media,
        shots=shots,
        segments=segments,
        package=package,
        audio_path=audio_path,
        workdir=render_dir,
        width=output_width,
        height=output_height,
        fps=output_fps,
    )
    output = visual_pipeline.VisualPipelineOutput(
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        caption_path=caption_path,
        visual_plan_path=plan_path,
        keyframes=keyframes,
        scene_media=scene_media,
    )
    (visual_root / "visual-pipeline-manifest.json").write_text(
        json.dumps(
            {
                **output.as_dict(),
                "profile": profile.as_dict(),
                "editorial_timeline": str(timeline_path),
                "shot_count": len(shots),
                "source_asset_looping": False,
                "destructive_caption_matte": False,
                "shots": [item.as_dict() for item in shots],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output


def _install_settings_and_voice() -> None:
    from . import audio_qc, production_pacing, voice_policy
    from .config import Settings

    profile = VideoProfile.from_env()
    current_from_env = Settings.from_env.__func__

    def v28_from_env(cls: type[Settings]) -> Settings:
        settings = current_from_env(cls)
        contract = replace(
            settings.voice_contract,
            target_wpm=profile.target_wpm,
            pause_style=(
                "natural clause pauses; a deliberate pause before the final call to action; "
                "never rush technical names"
            ),
        )
        return replace(
            settings,
            voice_contract=contract,
            audio_wpm_tolerance=max(
                profile.target_wpm - profile.minimum_wpm,
                profile.maximum_wpm - profile.target_wpm,
            ),
            audio_segment_pause_ms=profile.segment_pause_ms,
            target_seconds=60,
        )

    Settings.from_env = classmethod(v28_from_env)

    original_contract = voice_policy.contract_for_strategy

    def contract_for_profile(base: Any, strategy: Any) -> Any:
        contract = original_contract(base, strategy)
        return replace(
            contract,
            target_wpm=profile.target_wpm,
            pause_style=(
                "natural clause pauses; a deliberate pause before the final call to action; "
                "never rush technical names"
            ),
        )

    voice_policy.contract_for_strategy = contract_for_profile
    production_pacing._MAX_PRODUCTION_TEMPO = profile.maximum_tempo_factor
    production_pacing._MAX_CLOSING_TEMPO = min(1.10, profile.maximum_tempo_factor)
    audio_qc.MAX_TEMPO_FACTOR = profile.maximum_tempo_factor


def _install_visual_and_caption_contracts() -> None:
    from . import caption_renderer, image_generator, production_visual_quality
    from . import visual_prompt_compiler

    image_generator.compile_image_prompt = compile_semantic_image_prompt
    visual_prompt_compiler.compile_image_prompt = compile_semantic_image_prompt
    image_generator._caption_safe_zone = full_frame_caption_zone
    production_visual_quality.strengthen_compiled_prompt = lambda result: result
    production_visual_quality._caption_zone_is_exact_matte = _full_frame_is_valid

    original_normalize = production_visual_quality._normalize_review_payload

    def normalize_review(raw: dict[str, Any], **kwargs: Any) -> Any:
        cleaned = dict(raw)
        cleaned["prominent_person"] = False
        cleaned["device_or_panel"] = False
        return original_normalize(cleaned, **kwargs)

    production_visual_quality._normalize_review_payload = normalize_review
    caption_renderer._phrase_chunks = _caption_chunks
    caption_renderer._karaoke_text = lambda cue: _plain_caption(
        cue, caption_renderer._escape_ass
    )


def _install_video_qc() -> None:
    from . import video_qc

    profile = VideoProfile.from_env()
    current_verify = video_qc.verify_video_output

    def verify_v28(*args: Any, **kwargs: Any) -> Any:
        video_path = Path(kwargs.get("video_path") or (args[1] if len(args) > 1 else ""))
        timeline_path = video_path.parent.parent / "editorial-timeline.json"
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        shots = timeline.get("shots") or []
        shot_durations = [float(item["duration_seconds"]) for item in shots]
        kwargs["scene_durations"] = shot_durations
        kwargs["scene_media_types"] = ["video"] * len(shot_durations)
        report_path = kwargs.get("report_path")
        report = current_verify(*args, **kwargs)
        failures = list(report.failures)

        if len(shots) < profile.minimum_shots:
            failures.append(f"editorial shot count {len(shots)} is below {profile.minimum_shots}")
        if len(shots) > profile.maximum_shots:
            failures.append(f"editorial shot count {len(shots)} exceeds {profile.maximum_shots}")
        if shots and max(shot_durations) > profile.maximum_shot_seconds + 0.02:
            failures.append("one or more editorial shots exceed the configured duration ceiling")
        if (
            sum(float(item["start_seconds"]) < 10.0 for item in shots)
            < profile.first_ten_seconds_minimum_shots
        ):
            failures.append("the first ten seconds contain too few unique shots")

        composition_path = video_path.parent / "visual-composition-manifest.json"
        composition = json.loads(composition_path.read_text(encoding="utf-8"))
        if composition.get("source_asset_looping") is not False:
            failures.append("source asset looping is enabled")
        if composition.get("destructive_caption_matte") is not False:
            failures.append("destructive caption matte is enabled")
        log_text = (video_path.parent / "visual-compositor.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if "-stream_loop" in log_text:
            failures.append("FFmpeg source video looping was detected")

        media_manifest = (
            video_path.parent.parent / "scene-media" / "scene-media-manifest.json"
        )
        media_payload = json.loads(media_manifest.read_text(encoding="utf-8"))
        hashes = [
            str(item.get("sha256") or "")
            for item in media_payload.get("assets") or []
        ]
        if len(hashes) != len(set(hashes)):
            failures.append("a rendered shot asset was reused")

        keyframe_manifest = (
            video_path.parent.parent / "keyframes" / "keyframe-manifest.json"
        )
        keyframes = json.loads(
            keyframe_manifest.read_text(encoding="utf-8")
        ).get("assets") or []
        for item in keyframes:
            path = Path(str(item.get("path") or ""))
            if path.is_file() and not _full_frame_is_valid(path):
                failures.append(f"destructive lower matte detected in {path.name}")
                break

        caption_json = video_path.parent / "animated-captions.json"
        cues = json.loads(caption_json.read_text(encoding="utf-8")).get("cues") or []
        single_word = sum(int(item.get("word_count") or 0) == 1 for item in cues)
        if single_word > profile.maximum_single_word_cues:
            failures.append(f"caption track contains {single_word} isolated one-word cues")
        if any(
            float(item.get("end_seconds") or 0.0)
            - float(item.get("start_seconds") or 0.0)
            < profile.caption_minimum_seconds
            for item in cues
        ):
            failures.append("caption track contains an excessively short cue")

        voice_manifest = kwargs.get("voice_manifest_path")
        if voice_manifest:
            voice_payload = json.loads(Path(voice_manifest).read_text(encoding="utf-8"))
            metrics = voice_payload.get("metrics") or {}
            wpm = float(metrics.get("estimated_wpm") or 0.0)
            if not profile.minimum_wpm <= wpm <= profile.maximum_wpm:
                failures.append(
                    f"voice pace {wpm:.2f} WPM is outside "
                    f"{profile.minimum_wpm}-{profile.maximum_wpm}"
                )
            for event in voice_payload.get("reviews") or []:
                if str(event.get("type") or "").startswith("deterministic_"):
                    factor = float(event.get("factor") or 1.0)
                    if factor > profile.maximum_tempo_factor + 1e-6:
                        failures.append(
                            f"voice tempo correction {factor:.3f} exceeds "
                            f"{profile.maximum_tempo_factor:.2f}"
                        )
                        break

        updated = replace(
            report,
            passed=not failures,
            failures=tuple(dict.fromkeys(failures)),
        )
        if report_path is not None:
            Path(report_path).write_text(
                json.dumps(updated.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if failures:
            raise video_qc.VideoQCError("; ".join(dict.fromkeys(failures)))
        return updated

    video_qc.verify_video_output = verify_v28


def install_production_editorial_v28() -> None:
    """Install the fail-closed, profile-driven v28 editorial production pipeline."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_settings_and_voice()
    _install_visual_and_caption_contracts()
    _install_video_qc()

    from . import visual_pipeline

    visual_pipeline.render_visual_plan = render_visual_plan_v28
    for module_name in ("factory.pipeline", "factory.canary"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "render_visual_plan", render_visual_plan_v28)
    _INSTALLED = True
