import json
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from factory.config import Settings
from factory.models import NarrationSegment, Scene, VideoPackage
from factory.policy import Strategy
from factory.render import render_video
from factory.video_qc import (
    VideoQCError,
    _is_hold_jump_stutter,
    verify_video_output,
)


class VideoQCTests(unittest.TestCase):
    def _render(self, root: Path):
        package = VideoPackage(
            topic="Local AI",
            narration="Narration",
            title="Local AI explained",
            description="Description",
            tags=["AI"],
            thumbnail_text="LOCAL AI CHANGED",
            top_comment="Question",
            scenes=[
                Scene(f"Signal {index + 1}", "Evidence-backed explanation.", "Procedural visual")
                for index in range(6)
            ],
            source_urls=["https://example.com/a", "https://example.com/b"],
            source_publishers=["A", "B"],
        )
        strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")
        rate = 24000
        duration = 18
        samples = array(
            "h",
            (
                int(6200 * math.sin(2 * math.pi * (170 + (index // rate) * 9) * index / rate))
                for index in range(duration * rate)
            ),
        )
        voice = root / "voice.wav"
        with wave.open(str(voice), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(samples.tobytes())
        segments = []
        per = duration / 6
        for index in range(6):
            segments.append(
                NarrationSegment(
                    segment_id=index,
                    text=f"This is synchronized narration segment {index + 1} with useful technical context.",
                    instruction="test",
                    audio_path=voice,
                    start_seconds=index * per,
                    end_seconds=(index + 1) * per,
                )
            )
        video, thumbnail = render_video(
            Settings.from_env(), package, strategy, root, segments=segments
        )
        return video, thumbnail, duration, [per] * 6

    def _valid_voice_manifest(self, root: Path) -> Path:
        manifest = root / "voice-review-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generator": {
                        "backend": "qwen3",
                        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    },
                    "reviewer": {
                        "required": True,
                        "backend": "qwen_omni",
                        "model": "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
                    },
                    "metrics": {"passed": True},
                    "reviews": [
                        {
                            "type": "model_review",
                            "decision": "approve",
                            "overall_score": 0.93,
                            "scores": {
                                "naturalness": 0.91,
                                "pronunciation": 0.96,
                                "script_fidelity": 0.99,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_rendered_canary_passes_post_render_verification(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {"VIDEO_WIDTH": "360", "VIDEO_HEIGHT": "640", "VIDEO_FPS": "15"},
            clear=True,
        ):
            root = Path(temporary)
            video, thumbnail, duration, scene_durations = self._render(root)
            report = verify_video_output(
                Settings.from_env(),
                video,
                thumbnail,
                expected_duration=duration,
                scene_durations=scene_durations,
            )
            self.assertTrue(report.passed)
            self.assertEqual(report.video_codec, "h264")
            self.assertEqual(report.audio_codec, "aac")
            self.assertGreater(min(report.frame_contrast), 18)
            self.assertEqual(report.temporal_stutter_windows, 0)

    def test_wrong_expected_duration_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {"VIDEO_WIDTH": "360", "VIDEO_HEIGHT": "640", "VIDEO_FPS": "15"},
            clear=True,
        ):
            root = Path(temporary)
            video, thumbnail, _, scene_durations = self._render(root)
            with self.assertRaises(VideoQCError):
                verify_video_output(
                    Settings.from_env(),
                    video,
                    thumbnail,
                    expected_duration=40,
                    scene_durations=scene_durations,
                )

    def test_hold_jump_pattern_is_detected(self):
        self.assertTrue(
            _is_hold_jump_stutter([0.0, 0.01, 2.4, 0.0, 0.0, 1.9, 0.02, 2.2])
        )
        self.assertFalse(_is_hold_jump_stutter([0.0] * 12))
        self.assertFalse(_is_hold_jump_stutter([0.6] * 12))

    def test_production_voice_requires_qwen_and_approved_review(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {"VIDEO_WIDTH": "360", "VIDEO_HEIGHT": "640", "VIDEO_FPS": "15"},
            clear=True,
        ):
            root = Path(temporary)
            video, thumbnail, duration, scene_durations = self._render(root)
            manifest = self._valid_voice_manifest(root)
            report = verify_video_output(
                Settings.from_env(),
                video,
                thumbnail,
                expected_duration=duration,
                scene_durations=scene_durations,
                voice_manifest_path=manifest,
                require_production_voice=True,
            )
            self.assertTrue(report.production_voice_verified)
            self.assertIn("qwen3", report.voice_provenance)

    def test_espeak_or_missing_review_cannot_pass_as_production_voice(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {"VIDEO_WIDTH": "360", "VIDEO_HEIGHT": "640", "VIDEO_FPS": "15"},
            clear=True,
        ):
            root = Path(temporary)
            video, thumbnail, duration, scene_durations = self._render(root)
            manifest = root / "voice-review-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "generator": {"backend": "espeak", "model": "espeak"},
                        "metrics": {"passed": True},
                        "reviews": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(VideoQCError) as context:
                verify_video_output(
                    Settings.from_env(),
                    video,
                    thumbnail,
                    expected_duration=duration,
                    scene_durations=scene_durations,
                    voice_manifest_path=manifest,
                    require_production_voice=True,
                )
            message = str(context.exception)
            self.assertIn("not approved Qwen3-TTS", message)
            self.assertIn("perceptual voice review is missing", message)


if __name__ == "__main__":
    unittest.main()
