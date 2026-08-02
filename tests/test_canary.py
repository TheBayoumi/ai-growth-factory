import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from factory.canary import run_production_canary
from factory.config import Settings
from factory.feeds import SourceItem
from factory.models import AudioMetrics, NarrationSegment, Scene, VideoPackage, VoiceContract


class CanaryTests(unittest.TestCase):
    def test_real_stack_contract_exports_reviewed_visual_artifacts_without_youtube(self):
        sources = [
            SourceItem("A", "One", "https://a.example", "Summary", datetime.now(timezone.utc)),
            SourceItem("B", "Two", "https://b.example", "Summary", datetime.now(timezone.utc)),
        ]
        package = VideoPackage(
            topic="Current AI change",
            narration=" ".join(["word"] * 145),
            title="Verified AI Change",
            description="Description\nhttps://a.example\nhttps://b.example",
            tags=["AI"],
            thumbnail_text="AI CHANGE",
            top_comment="What will you test?",
            scenes=[Scene("Head", "Body", "Visual", index % 2) for index in range(6)],
            source_urls=[item.url for item in sources],
            source_publishers=[item.publisher for item in sources],
        )
        metrics = AudioMetrics(60, 24000, 1, -2, -18, 0, 0.05, 0.2, 155, 0, True)
        visual_plan = SimpleNamespace(
            prompt_version="visual-director-v1",
            image_model="ByteDance/SDXL-Lightning",
            video_model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            director_input_sha256="a" * 64,
            scenes=tuple(
                SimpleNamespace(generation_mode="wan_i2v" if index in {0, 2, 4} else "image")
                for index in range(6)
            ),
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "PUBLISH_ENABLED": "false",
                "WORK_ROOT": str(Path(temporary) / "work"),
                "STATE_ROOT": str(Path(temporary) / "state"),
            },
            clear=True,
        ), patch("factory.canary.fetch_recent", return_value=sources), patch(
            "factory.canary.managed_llama_server", return_value=nullcontext()
        ), patch("factory.canary.generate_package", return_value=package), patch(
            "factory.canary.construct_visual_plan", return_value=visual_plan
        ) as construct_visual, patch(
            "factory.canary.build_reviewed_narration"
        ) as voice, patch("factory.canary.render_visual_plan") as render_visual, patch(
            "factory.canary.verify_video_output"
        ) as verify:
            base = Path(temporary)
            audio = base / "voice.wav"
            manifest = base / "voice-review-manifest.json"
            video = base / "video.mp4"
            thumbnail = base / "thumbnail.png"
            captions = base / "animated-captions.ass"
            for path in (audio, manifest, video, thumbnail, captions):
                path.write_bytes(b"verified-data")
            segments = tuple(
                NarrationSegment(
                    segment_id=index,
                    text="word",
                    instruction="instruction",
                    audio_path=audio,
                    start_seconds=index * 10,
                    end_seconds=(index + 1) * 10,
                )
                for index in range(6)
            )
            voice.return_value = SimpleNamespace(
                audio_path=audio,
                manifest_path=manifest,
                metrics=metrics,
                review=SimpleNamespace(overall_score=0.94),
                attempts=1,
                voice_contract=VoiceContract(),
                segments=segments,
            )
            render_visual.return_value = SimpleNamespace(
                video_path=video,
                thumbnail_path=thumbnail,
                caption_path=captions,
                visual_plan_path=base / "visual-plan.json",
                keyframes=(),
                scene_media=(),
            )

            def fake_verify(*args, **kwargs):
                kwargs["report_path"].write_bytes(b"verified-data")
                return SimpleNamespace(as_dict=lambda: {"passed": True})

            verify.side_effect = fake_verify
            output_root = base / "artifacts"
            result = run_production_canary(Settings.from_env(), output_root)

            self.assertEqual(result["status"], "verified_render_canary")
            self.assertIn("canaries/", result["artifact_path"])
            self.assertEqual(result["visuals"]["wan_scene_count"], 3)
            self.assertFalse(result["visuals"]["captions_baked_into_generated_media"])
            construct_visual.assert_called_once()
            render_visual.assert_called_once()
            verify.assert_called_once()
            artifact_dir = output_root / result["canary_id"]
            self.assertTrue((artifact_dir / "video.mp4").exists())
            self.assertTrue((artifact_dir / "narration.wav").exists())
            self.assertTrue((artifact_dir / "video-qc-report.json").exists())
            self.assertTrue((artifact_dir / "canary-result.json").exists())


if __name__ == "__main__":
    unittest.main()
