import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from factory.caption_renderer import build_caption_cues, write_animated_caption_track
from factory.image_generator import selected_image_backend
from factory.models import NarrationSegment, Scene, VideoPackage
from factory.visual_prompt import VisualPromptError, _validate_and_normalize


class AutonomousVisualPromptTests(unittest.TestCase):
    def setUp(self):
        self.package = VideoPackage(
            topic="Reliable tool calling",
            narration="A source-grounded production narration.",
            title="OpenAI Realtime API Adds Reliable Tool Calling",
            description="Evidence-backed description",
            tags=["OpenAI", "Realtime API"],
            thumbnail_text="RELIABLE TOOL CALLING",
            top_comment="What would you test first?",
            scenes=[
                Scene(
                    heading=f"Scene {index + 1}",
                    body=f"Distinct evidence point {index + 1}",
                    visual="abstract technology mechanism",
                    source_index=index % 2,
                )
                for index in range(6)
            ],
            source_urls=["https://example.com/a", "https://example.com/b"],
            source_publishers=["OpenAI", "Microsoft Research"],
        )
        self.settings = SimpleNamespace(target_seconds=58)

    def _raw_plan(self):
        modes = ["wan_i2v", "image", "wan_i2v", "image", "wan_i2v", "image"]
        roles = ["hook", "evidence", "mechanism", "comparison", "implication", "cta"]
        return {
            "global_style": "premium editorial technology documentary with tactile materials",
            "palette": "deep blue graphite cyan and restrained warm highlights",
            "lighting": "cinematic volumetric side light with controlled contrast",
            "continuity_bible": "repeat one translucent data ribbon and the same lens language across scenes",
            "scenes": [
                {
                    "scene_index": index,
                    "source_index": index % 2,
                    "role": roles[index],
                    "generation_mode": modes[index],
                    "image_prompt": (
                        "A premium editorial visualization of a verified technology workflow inside a quiet architectural environment. "
                        "A translucent data ribbon connects precise mechanical stages while realistic materials, controlled depth, "
                        "cinematic side lighting, and a deliberate vertical composition communicate the supplied factual scene without branding."
                    ),
                    "motion_prompt": (
                        "A slow controlled camera push while the data ribbon illuminates sequentially and subtle environmental reflections change."
                    ),
                    "negative_prompt": "blur, deformation, text, logo, watermark, camera shake",
                    "continuity_anchor": "translucent cyan data ribbon",
                }
                for index in range(6)
            ],
        }

    def test_visual_plan_enforces_wan_routing_and_caption_safe_prompts(self):
        plan = _validate_and_normalize(
            self._raw_plan(),
            package=self.package,
            settings=self.settings,
            director_input_sha256="a" * 64,
        )
        self.assertEqual(sum(scene.generation_mode == "wan_i2v" for scene in plan.scenes), 3)
        self.assertEqual(plan.scenes[0].generation_mode, "wan_i2v")
        for scene in plan.scenes:
            self.assertIn("lower 32 percent", scene.image_prompt)
            self.assertIn("No text", scene.image_prompt)
            self.assertIn("No camera shake", scene.motion_prompt)
            self.assertEqual(scene.source_index, self.package.scenes[scene.scene_index].source_index)
            self.assertEqual(scene.caption_safe_zone, "lower_32_percent")

    def test_visual_plan_rejects_source_reassignment(self):
        raw = self._raw_plan()
        raw["scenes"][2]["source_index"] = 0
        with self.assertRaisesRegex(VisualPromptError, "source_index changed"):
            _validate_and_normalize(
                raw,
                package=self.package,
                settings=self.settings,
                director_input_sha256="b" * 64,
            )

    def test_image_backend_routes_without_hidden_token_requirement(self):
        with patch.dict(os.environ, {"VISUAL_IMAGE_BACKEND": "auto"}, clear=True):
            self.assertEqual(selected_image_backend(), "sdxl_lightning")
        with patch.dict(
            os.environ,
            {"VISUAL_IMAGE_BACKEND": "auto", "HF_TOKEN": "configured"},
            clear=True,
        ):
            self.assertEqual(selected_image_backend(), "flux")

    def test_animated_captions_are_separate_short_phrase_cues(self):
        segments = [
            NarrationSegment(
                segment_id=0,
                text="OpenAI released a concrete update that changes reliable tool calling in production voice agents.",
                instruction="narrate",
                audio_path=Path("segment.wav"),
                start_seconds=0.0,
                end_seconds=5.0,
            )
        ]
        cues = build_caption_cues(segments)
        self.assertGreaterEqual(len(cues), 2)
        self.assertTrue(all(cue.word_count <= 7 for cue in cues))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "captions.ass"
            written = write_animated_caption_track(segments, output)
            content = output.read_text(encoding="utf-8")
            self.assertEqual(len(written), len(cues))
            self.assertIn(r"\kf", content)
            self.assertIn(r"\fad", content)
            manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rendering"], "separate_animated_caption_layer")


if __name__ == "__main__":
    unittest.main()
