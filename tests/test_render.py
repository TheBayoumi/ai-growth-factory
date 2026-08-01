import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from factory.config import Settings
from factory.models import Scene, VideoPackage
from factory.policy import Strategy
from factory.render import render_video


class RenderTests(unittest.TestCase):
    def test_real_ffmpeg_render_uses_reviewed_voice_file(self):
        package = VideoPackage(
            topic="AI release",
            narration="Narration",
            title="AI release explained",
            description="Description",
            tags=["AI"],
            thumbnail_text="AI CHANGED",
            top_comment="Question",
            scenes=[
                Scene(f"Scene {index}", "Evidence-backed explanation.", "Procedural visual")
                for index in range(6)
            ],
            source_urls=["https://example.com/a", "https://example.com/b"],
            source_publishers=["A", "B"],
        )
        strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {"VIDEO_WIDTH": "360", "VIDEO_HEIGHT": "640", "VIDEO_FPS": "15"},
            clear=True,
        ):
            workdir = Path(temporary)
            samples = array(
                "h",
                (
                    int(6500 * math.sin(2 * math.pi * 180 * index / 24000))
                    for index in range(18 * 24000)
                ),
            )
            with wave.open(str(workdir / "voice.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(samples.tobytes())
            video, thumbnail = render_video(Settings.from_env(), package, strategy, workdir)
            self.assertGreater(video.stat().st_size, 100_000)
            self.assertGreater(thumbnail.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
