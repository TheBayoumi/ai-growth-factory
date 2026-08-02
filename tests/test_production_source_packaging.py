import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionSourcePackagingTests(unittest.TestCase):
    def test_every_production_runtime_module_exists(self):
        required = (
            "source_index_repair.py",
            "production_content.py",
            "production_pacing.py",
            "production_voice_repair.py",
            "production_renderer.py",
            "production_runtime.py",
            "visual_prompt.py",
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


if __name__ == "__main__":
    unittest.main()
