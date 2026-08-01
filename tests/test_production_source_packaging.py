import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionSourcePackagingTests(unittest.TestCase):
    def test_every_production_runtime_module_exists(self):
        required = (
            "source_index_repair.py",
            "production_content.py",
            "production_pacing.py",
            "production_renderer.py",
            "production_runtime.py",
        )
        missing = [name for name in required if not (ROOT / "factory" / name).is_file()]
        self.assertEqual(missing, [])

    def test_gpu_workflow_imports_runtime_before_deployment(self):
        workflow = (
            ROOT / ".github" / "workflows" / "modal-production-verification.yml"
        ).read_text(encoding="utf-8")
        preflight = (
            "from factory.production_runtime import install_production_runtime; "
            "install_production_runtime()"
        )
        self.assertIn(preflight, workflow)
        self.assertLess(
            workflow.index("Production runtime import preflight passed"),
            workflow.index("Deploy A10 production worker"),
        )


if __name__ == "__main__":
    unittest.main()
