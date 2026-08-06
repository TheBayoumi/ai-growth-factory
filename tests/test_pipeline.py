import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from factory.config import Settings
from factory.feeds import SourceItem
from factory.models import AudioMetrics, NarrationSegment, Scene, VideoPackage, VoiceContract
from factory.pipeline import run_factory
from factory.policy import Strategy


class FakeYouTube:
    def __init__(self, settings):
        self.settings = settings
        self.upload = Mock(return_value="video123")

    def channel_context(self):
        return SimpleNamespace(channel_id="channel", uploads_playlist="uploads")

    def recent_videos(self, context):
        del context
        return []

    def observations(self, recent):
        del recent
        return []

    def already_published(self, recent, daily_tag):
        del recent, daily_tag
        return False


class PipelineTests(unittest.TestCase):
    def test_strategy_contract_reaches_voice_visual_pipeline_and_upload(self):
        sources = [
            SourceItem("A", "One", "https://a.example", "Summary", datetime.now(timezone.utc)),
            SourceItem("B", "Two", "https://b.example", "Summary", datetime.now(timezone.utc)),
        ]
        package = VideoPackage(
            topic="Topic",
            narration=" ".join(["word"] * 140),
            title="Title",
            description="Description",
            tags=["AI"],
            thumbnail_text="AI CHANGE",
            top_comment="Question",
            scenes=[Scene("Head", "Body", "Visual", index % 2) for index in range(6)],
            source_urls=[item.url for item in sources],
            source_publishers=[item.publisher for item in sources],
        )
        strategy = Strategy("breaking", "fast", "kinetic", "55-62", "subscribe")
        metrics = AudioMetrics(60, 24000, 1, -2, -18, 0, 0.05, 0.2, 165, 0, True)
        fake_youtube = FakeYouTube(None)
        visual_plan = SimpleNamespace(
            prompt_version="visual-director-v1",
            image_model="ByteDance/SDXL-Lightning",
            video_model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            director_input_sha256="b" * 64,
            scenes=tuple(
                SimpleNamespace(generation_mode="wan_i2v" if index in {0, 2, 4} else "image")
                for index in range(6)
            ),
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "PUBLISH_ENABLED": "true",
                "OPENAI_API_KEY": "review-key",
                "YOUTUBE_OAUTH_JSON": '{"client_id":"a","client_secret":"b","refresh_token":"c"}',
                "WORK_ROOT": str(Path(temporary) / "work"),
                "STATE_ROOT": str(Path(temporary) / "state"),
            },
            clear=True,
        ), patch("factory.pipeline.YouTubeClient", return_value=fake_youtube), patch(
            "factory.pipeline.fetch_recent", return_value=sources
        ), patch("factory.pipeline.select_strategy", return_value=strategy), patch(
            "factory.pipeline.managed_llama_server", return_value=nullcontext()
        ), patch("factory.pipeline.generate_package", return_value=package), patch(
            "factory.pipeline.construct_visual_plan", return_value=visual_plan
        ) as construct_visual, patch(
            "factory.pipeline.validate_static_editorial_preflight"
        ) as static_preflight, patch(
            "factory.pipeline.build_reviewed_narration"
        ) as voice, patch("factory.pipeline.render_visual_plan") as render_visual, patch(
            "factory.pipeline.verify_video_output"
        ) as verify:
            audio = Path(temporary) / "voice.wav"
            manifest = Path(temporary) / "manifest.json"
            video = Path(temporary) / "video.mp4"
            thumbnail = Path(temporary) / "thumbnail.jpg"
            captions = Path(temporary) / "captions.ass"
            for path in (audio, manifest, video, thumbnail, captions):
                path.write_bytes(b"data")
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
            voice.side_effect = lambda settings, narration, workdir, voice_contract: SimpleNamespace(
                audio_path=audio,
                manifest_path=manifest,
                metrics=metrics,
                review=SimpleNamespace(overall_score=0.94),
                attempts=1,
                voice_contract=voice_contract,
                segments=segments,
            )
            render_visual.return_value = SimpleNamespace(
                video_path=video,
                thumbnail_path=thumbnail,
                caption_path=captions,
                visual_plan_path=Path(temporary) / "visual-plan.json",
                keyframes=(),
                scene_media=(),
            )
            verify.return_value = SimpleNamespace(as_dict=lambda: {"passed": True})
            static_preflight.return_value = None
            construct_visual.side_effect = lambda *args, **kwargs: (
                kwargs["plan_validator"](visual_plan) or visual_plan
            )
            result = run_factory(Settings.from_env())
        self.assertEqual(result["status"], "published")
        contract = voice.call_args.kwargs["voice_contract"]
        self.assertIsInstance(contract, VoiceContract)
        self.assertGreater(contract.target_wpm, 155)
        self.assertIn("urgent", contract.baseline_style)
        fake_youtube.upload.assert_called_once()
        construct_visual.assert_called_once()
        static_preflight.assert_called_once()
        render_visual.assert_called_once()
        self.assertIn("segments", render_visual.call_args.kwargs)
        self.assertEqual(render_visual.call_args.kwargs["audio_path"], audio)
        verify.assert_called_once()
        self.assertEqual(result["voice"]["contract"]["target_wpm"], contract.target_wpm)
        self.assertEqual(result["visuals"]["wan_scene_count"], 3)
        self.assertFalse(result["visuals"]["captions_baked_into_generated_media"])


if __name__ == "__main__":
    unittest.main()
