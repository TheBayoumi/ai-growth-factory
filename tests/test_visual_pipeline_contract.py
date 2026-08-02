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
