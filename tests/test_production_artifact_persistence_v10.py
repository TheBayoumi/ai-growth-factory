import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from factory.canary import _persist_generated_bundle
from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
from factory.visual_compositor import _thumbnail


class ProductionArtifactPersistenceV10Tests(unittest.TestCase):
    @staticmethod
    def _package() -> VideoPackage:
        return VideoPackage(
            topic="AI support update",
            narration="A production-length narration is not required for this persistence test.",
            title="AI Support Changed This Week",
            description="Source-grounded description.",
            tags=["AI", "support"],
            thumbnail_text="AI SUPPORT CHANGED",
            top_comment="What should be tested next?",
            scenes=[Scene("Heading", "Body", "Visual", 0)],
            source_urls=["https://example.com/source"],
            source_publishers=["Publisher"],
        )

    def test_thumbnail_is_a_dedicated_1280_by_720_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframe = root / "keyframe.png"
            output = root / "thumbnail.png"
            Image.new("RGB", (704, 1280), (22, 88, 140)).save(keyframe)

            _thumbnail(keyframe, self._package(), output)

            with Image.open(output) as image:
                self.assertEqual(image.size, (1280, 720))
                self.assertGreater(image.convert("L").getextrema()[1], 40)

    def test_failed_qc_bundle_keeps_video_audio_thumbnail_package_and_wan_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workdir = root / "work"
            destination = root / "artifact"
            render = workdir / "visual-assets" / "render"
            media = workdir / "visual-assets" / "scene-media"
            keyframes = workdir / "visual-assets" / "keyframes"
            render.mkdir(parents=True)
            media.mkdir(parents=True)
            keyframes.mkdir(parents=True)
            destination.mkdir()

            video = render / "video.mp4"
            thumbnail = render / "thumbnail.png"
            caption = render / "animated-captions.ass"
            audio = workdir / "voice.wav"
            voice_manifest = workdir / "voice-review-manifest.json"
            for path, payload in (
                (video, b"video" * 100),
                (thumbnail, b"thumbnail"),
                (caption, b"captions"),
                (audio, b"audio"),
                (voice_manifest, b"{}"),
                (media / "scene-00-wan.mp4", b"wan" * 100),
                (keyframes / "scene-00-keyframe.png", b"png"),
            ):
                path.write_bytes(payload)

            visual = SimpleNamespace(
                video_path=video,
                thumbnail_path=thumbnail,
                caption_path=caption,
            )
            voice = SimpleNamespace(audio_path=audio, manifest_path=voice_manifest)
            snapshot = SimpleNamespace(items=[1, 2], provider_status={"github": "ok"})
            alignment = SimpleNamespace(matches=[1])

            _persist_generated_bundle(
                destination=destination,
                workdir=workdir,
                package=self._package(),
                voice=voice,
                visual=visual,
                sources=[
                    SourceItem(
                        "Publisher",
                        "AI support update",
                        "https://example.com/source",
                        "Primary evidence",
                        datetime.now(timezone.utc),
                    )
                ],
                source_publishers={"Publisher"},
                source_max_age_hours=24,
                trend_snapshot=snapshot,
                trend_alignment=alignment,
            )

            required = (
                destination / "video.mp4",
                destination / "thumbnail.png",
                destination / "narration.wav",
                destination / "package.json",
                destination / "voice-review-manifest.json",
                destination / "visual-scene-media" / "scene-00-wan.mp4",
                destination / "visual-keyframes" / "scene-00-keyframe.png",
            )
            self.assertTrue(all(path.is_file() for path in required), required)
            package_payload = (destination / "package.json").read_text(encoding="utf-8")
            self.assertIn('"source_evidence"', package_payload)


if __name__ == "__main__":
    unittest.main()
