from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

from factory.audio_qc import analyze_audio, concatenate_segments, normalize_audio, split_narration
from factory.config import Settings
from factory.models import NarrationSegment, Scene, VideoPackage
from factory.policy import Strategy
from factory.render import render_video
from factory.video_qc import verify_video_output


NARRATION = (
    "This pipeline now creates a complete AI news video without paying for a hosted generation model. "
    "It collects recent primary sources, rejects stale or weak evidence, and gives the script model only the material it is allowed to cite. "
    "The script model then produces six short scenes, each designed around one claim instead of filling the video with generic AI hype. "
    "Qwen three TTS generates the narration in segments, while objective audio checks catch clipping, dead air, weak volume, and pacing problems. "
    "An audio reviewer can reject one weak segment without forcing the system to regenerate the entire voice track or waste another GPU run. "
    "Only after the voice, captions, thumbnail, duration, codecs, and sampled frames pass verification can the private YouTube canary be uploaded."
)

SCENES = [
    Scene("THE COST RESET", "Hosted generation is no longer required for the proof stage.", "cost meter", source_index=1),
    Scene("EVIDENCE FIRST", "Fresh primary sources are filtered before the script model sees them.", "source graph", source_index=0),
    Scene("SIX FOCUSED SCENES", "Every scene carries one claim, one explanation, and one visual purpose.", "scene timeline", source_index=0),
    Scene("VOICE QUALITY GATES", "Narration is checked for clipping, silence, loudness, and speaking pace.", "audio meter", source_index=0),
    Scene("REPAIR ONLY THE FAILURE", "A rejected segment is regenerated without discarding approved narration.", "repair loop", source_index=0),
    Scene("VERIFY BEFORE UPLOAD", "The final MP4 and thumbnail must pass objective output checks.", "checklist", source_index=1),
]


def espeak_segment(text: str, output: Path, *, speed: int, pitch: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "espeak",
        "-v",
        "en-us",
        "-s",
        str(speed),
        "-p",
        str(pitch),
        "-a",
        "175",
        "-w",
        str(output),
        text,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(completed.stderr)
    return output


def create_audio(root: Path, speed: int, pitch: int, settings: Settings) -> tuple[list[NarrationSegment], float]:
    texts = split_narration(NARRATION, 6)
    segments: list[NarrationSegment] = []
    for segment_id, text in enumerate(texts):
        audio = espeak_segment(
            text,
            root / "raw-segments" / f"segment-{segment_id}.wav",
            speed=speed,
            pitch=pitch,
        )
        segments.append(
            NarrationSegment(
                segment_id=segment_id,
                text=text,
                instruction="mechanical render fixture only; forbidden for perceptual or publication review",
                audio_path=audio,
            )
        )
    raw, timed = concatenate_segments(segments, root / "voice-raw.wav", pause_ms=130)
    normalized = normalize_audio(raw, root / "voice.wav", target_lufs=-16.0, peak_dbfs=-1.0)
    with wave.open(str(normalized), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    metrics = analyze_audio(normalized, narration=NARRATION, settings=settings, target_wpm=speed)
    (root / "audio-metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2), encoding="utf-8"
    )
    return timed, duration


def main() -> None:
    if os.environ.get("ALLOW_MECHANICAL_RENDER_FIXTURE", "").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError(
            "This script uses eSpeak only to exercise timing. Set "
            "ALLOW_MECHANICAL_RENDER_FIXTURE=true explicitly; never present its output as a canary."
        )
    destination = Path(os.environ.get("FIXTURE_ROOT", "/mnt/data/ai-growth-factory-render-fixtures-1.3.1"))
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "NOT_A_PRODUCTION_CANARY.txt").write_text(
        "Mechanical eSpeak narration. Valid only for render timing and temporal-stability tests. "
        "It must never be used for voice-quality review, publication, or user approval.\n",
        encoding="utf-8",
    )
    os.environ.update({"VIDEO_WIDTH": "720", "VIDEO_HEIGHT": "1280", "VIDEO_FPS": "24"})
    settings = Settings.from_env()
    variants = [
        ("dashboard-practical", Strategy("practical", "balanced", "dashboard", "55-62", "subscribe"), 156, 46),
        ("kinetic-breaking", Strategy("breaking", "fast", "kinetic", "55-62", "subscribe"), 166, 52),
        ("cinematic-contrarian", Strategy("contrarian", "balanced", "cinematic", "63-72", "newsletter"), 151, 42),
    ]
    summary = []
    for name, strategy, speed, pitch in variants:
        root = destination / name
        root.mkdir(parents=True, exist_ok=True)
        timed, duration = create_audio(root, speed, pitch, settings)
        package = VideoPackage(
            topic="Free-first AI video automation",
            narration=NARRATION,
            title="The AI Video Pipeline That Verifies Itself",
            description="A visual render fixture. The mechanical narration is invalid for perceptual or publication review.",
            tags=["AI automation", "open source AI", "Qwen", "creator tools"],
            thumbnail_text="VISUAL FIXTURE ONLY",
            top_comment="Which stage should be optimized next?",
            scenes=SCENES,
            source_urls=["https://github.com/QwenLM", "https://modal.com"],
            source_publishers=["Qwen", "Modal"],
        )
        video, thumbnail = render_video(settings, package, strategy, root, segments=timed)
        report = verify_video_output(
            settings,
            video,
            thumbnail,
            expected_duration=duration,
            report_path=root / "video-qc-report.json",
        )
        summary.append(
            {
                "variant": name,
                "video": str(video),
                "thumbnail": str(thumbnail),
                "duration_seconds": report.duration_seconds,
                "size_bytes": report.file_size_bytes,
                "frame_luminance": report.frame_luminance,
                "frame_contrast": report.frame_contrast,
                "frame_differences": report.frame_differences,
                "passed": report.passed,
            }
        )
    (destination / "render-fixture-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
