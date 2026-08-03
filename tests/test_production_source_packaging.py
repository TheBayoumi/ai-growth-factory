import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionSourcePackagingTests(unittest.TestCase):
    def test_every_production_runtime_module_exists(self):
        required = (
            "source_index_repair.py",
            "production_content.py",
            "production_source_publishers.py",
            "production_narration_length.py",
            "production_relationship_grounding.py",
            "production_pacing.py",
            "production_reviewer_feedback.py",
            "production_voice_repair.py",
            "production_visual_routing.py",
            "production_visual_semantics.py",
            "production_video_qc.py",
            "production_renderer.py",
            "production_runtime.py",
            "trend_sources.py",
            "trend_ranking.py",
            "visual_prompt.py",
            "visual_prompt_compiler.py",
            "image_generator.py",
            "video_generator.py",
            "caption_renderer.py",
            "visual_compositor.py",
            "visual_pipeline.py",
        )
        missing = [name for name in required if not (ROOT / "factory" / name).is_file()]
        self.assertEqual(missing, [])

    def test_gpu_workflow_imports_runtime_before_deployment(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        production_preflight = (
            "from factory.production_runtime import install_production_runtime; "
            "install_production_runtime()"
        )
        visual_preflight = (
            "from factory.visual_prompt import construct_visual_plan; "
            "from factory.caption_renderer import write_animated_caption_track; "
            "from factory.visual_pipeline import render_visual_plan"
        )
        self.assertIn(production_preflight, workflow)
        self.assertIn(visual_preflight, workflow)
        self.assertLess(
            workflow.index("Production runtime import preflight passed"),
            workflow.index("Deploy A10 production worker"),
        )
        self.assertLess(
            workflow.index("Autonomous visual pipeline import preflight passed"),
            workflow.index("Deploy A10 production worker"),
        )

    def test_modal_worker_pins_and_preflights_wan_exporter(self):
        source = (ROOT / "cloud" / "modal_app.py").read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"imageio==2.37.0"', source)
        self.assertIn('"imageio==2.37.0"', project)
        self.assertIn("import decord, imageio, imageio_ffmpeg", source)
        self.assertIn("imageio_ffmpeg.get_ffmpeg_exe", source)

    def test_runtime_installs_source_visual_and_media_aware_policies(self):
        source = (ROOT / "factory" / "production_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("install_production_source_publisher_canonicalization", source)
        self.assertIn("install_production_visual_semantics", source)
        self.assertIn("install_production_video_qc", source)
        self.assertLess(
            source.index("install_production_source_publisher_canonicalization()"),
            source.index("install_production_content_gate()"),
        )
        self.assertLess(
            source.index("install_production_visual_routing()"),
            source.index("install_production_visual_semantics()"),
        )


if __name__ == "__main__":
    unittest.main()
