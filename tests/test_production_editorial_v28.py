from __future__ import annotations

import json
import tempfile
import unittest
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from factory.config import Settings
from factory.editorial_timeline import ShotSpec
from factory.feeds import SourceItem
from factory.models import NarrationSegment, Scene, VideoPackage
from factory.policy import Strategy
from factory.production_caption_layout_v32 import write_pixel_fitted_caption_track
from factory.production_editorial_v28 import (
    EditorialPreflightResult,
    ProductionPreflightError,
    _caption_chunks,
    _plain_caption,
    _projected_narration_segments,
    compile_semantic_image_prompt,
    full_frame_caption_zone,
    render_visual_plan_v28,
    validate_static_editorial_preflight,
)
from factory.video_profile import VideoProfile
from factory.visual_prompt import SceneVisualPrompt, VisualPlan, construct_visual_plan


class ProductionEditorialV28Tests(unittest.TestCase):
    def _package(self, word_count: int = 136, scene_count: int = 6) -> VideoPackage:
        return VideoPackage(
            topic="Liquid AI local agents",
            narration=" ".join(f"word{index}" for index in range(word_count)),
            title="Liquid AI Releases a Measured Local Agent Model",
            description="Source-grounded description",
            tags=["AI"],
            thumbnail_text="LOCAL AGENTS",
            top_comment="Which measurement matters?",
            scenes=[
                Scene(
                    heading=f"Evidence {index}",
                    body=f"Measured evidence body {index}",
                    visual=f"Literal workflow {index}",
                    source_index=0,
                )
                for index in range(scene_count)
            ],
            source_urls=["https://example.com/source"],
            source_publishers=["Liquid AI"],
        )

    def _plan(self, scene_count: int = 6) -> VisualPlan:
        return VisualPlan(
            prompt_version="test-plan",
            global_style="factual documentary",
            palette="neutral blue",
            lighting="natural task light",
            continuity_bible="literal unbranded workflows",
            image_model="test-image-model",
            video_model="test-video-model",
            width=704,
            height=1280,
            fps=24,
            director_input_sha256="a" * 64,
            scenes=tuple(
                SceneVisualPrompt(
                    scene_index=index,
                    source_index=0,
                    role="hook" if index == 0 else "evidence",
                    generation_mode="wan_i2v" if index in {0, 2, 4} else "image",
                    image_prompt=f"Literal measured local agent workflow {index}",
                    motion_prompt="One restrained physical action",
                    negative_prompt="text, logo, shake",
                    continuity_anchor=f"workflow-{index}",
                    caption_safe_zone="lower_32_percent",
                    seed=index + 1,
                    duration_seconds=3.0,
                )
                for index in range(scene_count)
            ),
        )

    def test_compiler_preserves_concrete_semantics(self) -> None:
        result = compile_semantic_image_prompt(
            "Researchers collaborate in a shared workspace using unbranded computers and tools"
        )
        lowered = result.compiled_prompt.casefold()
        self.assertIn("researchers", lowered)
        self.assertIn("shared workspace", lowered)
        self.assertNotIn("modular bridge", lowered)
        self.assertNotIn("geometric modules converge", lowered)
        self.assertEqual(result.compiler_version, "visual-compiler-v28-semantic-preservation")

    def test_caption_zone_does_not_destroy_pixels(self) -> None:
        image = Image.new("RGB", (100, 200), (22, 44, 66))
        image.putpixel((50, 190), (200, 100, 50))
        result, before, after = full_frame_caption_zone(image)
        self.assertEqual(result.tobytes(), image.tobytes())
        self.assertEqual(before, after)
        self.assertEqual(result.getpixel((50, 190)), (200, 100, 50))

    def test_caption_chunks_avoid_one_word_churn(self) -> None:
        chunks = _caption_chunks(
            "This framework helps researchers build shared infrastructure, test ideas, and collaborate efficiently."
        )
        self.assertTrue(all(len(chunk.split()) >= 2 for chunk in chunks))
        self.assertTrue(all(len(chunk.split()) <= 5 for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 34 for chunk in chunks))

    def test_caption_text_is_phrase_level_not_karaoke(self) -> None:
        cue = SimpleNamespace(text="shared research infrastructure")
        rendered = _plain_caption(cue, lambda value: value)
        self.assertEqual(rendered, cue.text)
        self.assertNotIn("\\kf", rendered)

    def test_caption_layout_proves_actual_libass_pixels_are_inside_safe_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments = (
                NarrationSegment(
                    0,
                    "Liquid AI released an on-device model with multi-step tool calling.",
                    "test",
                    root / "segment.wav",
                    0.0,
                    5.0,
                ),
            )
            output = root / "captions.ass"

            write_pixel_fitted_caption_track(segments, output)
            manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

            self.assertTrue(manifest["all_cues_fit"])
            self.assertTrue(manifest["all_rendered_cues_fit"])
            self.assertTrue(
                all("rendered_bbox_pixels" in cue for cue in manifest["cues"])
            )

    def test_projected_timing_rejects_bad_duration_before_tts(self) -> None:
        settings = Settings.from_env()
        profile = VideoProfile()
        segments, duration = _projected_narration_segments(
            settings,
            self._package(),
            profile,
        )
        self.assertGreaterEqual(duration, profile.minimum_video_seconds)
        self.assertLessEqual(duration, profile.maximum_video_seconds)
        self.assertGreaterEqual(len(segments), 1)
        self.assertLessEqual(len(segments), settings.narration_segments)

        with self.assertRaisesRegex(ProductionPreflightError, "Projected narration duration"):
            _projected_narration_segments(
                settings,
                self._package(word_count=80),
                profile,
            )

    def test_visual_director_repairs_a_plan_that_fails_executable_preflight(self) -> None:
        settings = Settings.from_env()
        package = self._package()
        plan = self._plan()
        source = SourceItem(
            "Liquid AI",
            "Measured local agent model",
            package.source_urls[0],
            "Concrete measured evidence for the local agent model.",
            datetime.now(timezone.utc),
        )
        validator_calls: list[int] = []

        def validator(_plan: VisualPlan) -> None:
            validator_calls.append(len(validator_calls) + 1)
            if len(validator_calls) == 1:
                raise ProductionPreflightError("repeated executable environment")

        with patch(
            "factory.visual_prompt._chat_visual_director",
            return_value={},
        ) as director, patch(
            "factory.visual_prompt._validate_and_normalize",
            return_value=plan,
        ), patch("factory.visual_prompt.time.sleep"):
            result = construct_visual_plan(
                settings,
                package,
                [source],
                Strategy("practical", "balanced", "editorial", "55-62", "subscribe"),
                plan_validator=validator,
            )

        self.assertIs(result, plan)
        self.assertEqual(validator_calls, [1, 2])
        self.assertEqual(director.call_count, 2)
        self.assertIn("repeated executable environment", director.call_args.args[1])

    def test_static_preflight_persists_the_frozen_contract(self) -> None:
        package = self._package()
        plan = self._plan()
        shots = tuple(
            ShotSpec(
                shot_id=index,
                beat_index=index,
                segment_id=index,
                package_scene_index=index,
                source_index=0,
                start_seconds=float(index),
                duration_seconds=2.0,
                renderer="wan_i2v",
                semantic_claim=f"claim {index}",
                visual_direction=f"workflow {index}",
                treatment="literal",
                seed=index + 10,
            )
            for index in range(3)
        )

        def write_captions(_segments, path: Path, **_kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("captions", encoding="utf-8")
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "all_cues_fit": True,
                        "all_rendered_cues_fit": True,
                    }
                ),
                encoding="utf-8",
            )
            return ()

        with tempfile.TemporaryDirectory() as temporary, patch(
            "factory.production_editorial_v28.build_editorial_plan",
            return_value=(plan, shots),
        ), patch(
            "factory.production_visual_convergence_v41.validate_editorial_contract_diversity_v41",
            return_value={"field-mobile": 2, "healthcare": 1},
        ), patch(
            "factory.caption_renderer.write_animated_caption_track",
            side_effect=write_captions,
        ):
            payload = validate_static_editorial_preflight(
                settings=Settings.from_env(),
                plan=plan,
                package=package,
                workdir=Path(temporary),
                attempt=1,
            )
            persisted = json.loads(
                (
                    Path(temporary)
                    / "visual-assets"
                    / "preflight"
                    / "static-preflight.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(
            persisted["quality_contract_sha256"],
            payload["quality_contract_sha256"],
        )
        self.assertTrue(persisted["checks"]["rendered_caption_bounds"])

    def test_exact_preflight_runs_before_any_visual_generator(self) -> None:
        plan = self._plan(scene_count=1)
        package = self._package(scene_count=1)
        shot = ShotSpec(
            shot_id=0,
            beat_index=0,
            segment_id=0,
            package_scene_index=0,
            source_index=0,
            start_seconds=0.0,
            duration_seconds=1.0,
            renderer="wan_i2v",
            semantic_claim="measured claim",
            visual_direction="literal workflow",
            treatment="literal",
            seed=7,
        )
        events: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "voice.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(b"\x00\x00" * 24000)
            manifest = root / "production-preflight.json"
            manifest.write_text("{}", encoding="utf-8")
            animatic = root / "animatic.mp4"
            animatic.write_bytes(b"animatic")
            video = root / "video.mp4"
            thumbnail = root / "thumbnail.png"
            captions = root / "captions.ass"
            for path in (video, thumbnail, captions):
                path.write_bytes(b"output")
            preflight = EditorialPreflightResult(
                plan=plan,
                shots=(shot,),
                environment_families={"controlled-test": 1},
                quality_contract_sha256="b" * 64,
                manifest_path=manifest,
                animatic_path=animatic,
            )
            with patch(
                "factory.production_editorial_v28.validate_exact_editorial_preflight",
                side_effect=lambda **_kwargs: (events.append("preflight") or preflight),
            ), patch(
                "factory.visual_pipeline.generate_keyframes",
                side_effect=lambda *_args, **_kwargs: (events.append("keyframes") or ()),
            ), patch(
                "factory.video_generator.generate_scene_media",
                return_value=(),
            ), patch(
                "factory.visual_pipeline.release_accelerator_memory"
            ), patch(
                "factory.production_editorial_v28._compose_editorial_video",
                return_value=(video, thumbnail, captions),
            ):
                render_visual_plan_v28(
                    plan=plan,
                    package=package,
                    segments=(NarrationSegment(0, "word", "test", audio, 0.0, 1.0),),
                    audio_path=audio,
                    workdir=root / "work",
                )

        self.assertEqual(events[:2], ["preflight", "keyframes"])


if __name__ == "__main__":
    unittest.main()
