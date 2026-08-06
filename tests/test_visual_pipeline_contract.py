import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VisualPipelineContractTests(unittest.TestCase):
    def test_canary_and_publisher_use_the_same_visual_pipeline(self):
        canary = (ROOT / "factory" / "canary.py").read_text(encoding="utf-8")
        pipeline = (ROOT / "factory" / "pipeline.py").read_text(encoding="utf-8")
        for source in (canary, pipeline):
            self.assertIn("construct_visual_plan", source)
            self.assertIn("render_visual_plan", source)
            self.assertNotIn("render_video(", source)

    def test_paid_visual_inference_is_blocked_by_static_and_animatic_preflight(self):
        canary = (ROOT / "factory" / "canary.py").read_text(encoding="utf-8")
        pipeline = (ROOT / "factory" / "pipeline.py").read_text(encoding="utf-8")
        editorial = (ROOT / "factory" / "production_editorial_v28.py").read_text(
            encoding="utf-8"
        )
        for source in (canary, pipeline):
            self.assertLess(
                source.index("validate_static_editorial_preflight"),
                source.index("build_reviewed_narration"),
            )
        self.assertLess(
            editorial.index("validate_exact_editorial_preflight("),
            editorial.index("visual_pipeline.generate_keyframes", editorial.index("def render_visual_plan_v28")),
        )
        self.assertIn("deterministic-preflight-placeholder", editorial)
        self.assertIn("exact_before_visual_inference", editorial)
        self.assertIn(
            "animatic_width, animatic_height, animatic_fps = 720, 1280, 30",
            editorial,
        )

    def test_final_duration_gate_reads_the_completed_qc_report(self):
        editorial = (ROOT / "factory" / "production_editorial_v28.py").read_text(
            encoding="utf-8"
        )
        verifier = editorial.index("def verify_v28")
        report = editorial.index("report = current_verify", verifier)
        duration = editorial.index("report.duration_seconds", report)
        self.assertLess(report, duration)

    def test_generated_media_and_captions_are_separate_layers(self):
        compositor = (ROOT / "factory" / "visual_compositor.py").read_text(encoding="utf-8")
        captioner = (ROOT / "factory" / "caption_renderer.py").read_text(encoding="utf-8")
        prompt = (ROOT / "factory" / "visual_prompt.py").read_text(encoding="utf-8")
        self.assertIn("captions_baked_into_generated_media", compositor)
        self.assertIn("subtitles=filename", compositor)
        self.assertIn("separate_animated_caption_layer", captioner)
        self.assertIn("No text, letters, numbers, captions", prompt)
        self.assertIn("lower 32 percent", prompt)

    def test_open_model_backends_are_source_controlled(self):
        image_generator = (ROOT / "factory" / "image_generator.py").read_text(encoding="utf-8")
        video_generator = (ROOT / "factory" / "video_generator.py").read_text(encoding="utf-8")
        self.assertIn("black-forest-labs/FLUX.1-schnell", image_generator)
        self.assertIn("ByteDance/SDXL-Lightning", image_generator)
        self.assertIn("Wan-AI/Wan2.2-TI2V-5B-Diffusers", video_generator)
        self.assertIn("WanImageToVideoPipeline", video_generator)
        self.assertNotIn("procedural visual direction", video_generator)

    def test_modal_worker_contains_visual_runtime_and_offload_capacity(self):
        source = (ROOT / "cloud" / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn('"diffusers==0.39.0"', source)
        self.assertIn('"fonts-dejavu-core"', source)
        self.assertIn('"VISUAL_IMAGE_BACKEND": "auto"', source)
        self.assertIn('"WAN22_MODEL_ID": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"', source)
        self.assertIn("memory=65536", source)
        self.assertIn("timeout=85 * 60", source)

    def test_visual_audit_persists_prompts_seeds_and_keyframes(self):
        canary = (ROOT / "factory" / "canary.py").read_text(encoding="utf-8")
        self.assertIn("visual-plan.json", canary)
        self.assertIn("keyframe-manifest.json", canary)
        self.assertIn("scene-media-manifest.json", canary)
        self.assertIn("animated-captions.ass", canary)
        self.assertIn("visual-keyframes", canary)


if __name__ == "__main__":
    unittest.main()
