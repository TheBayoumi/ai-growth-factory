import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalOptimumRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_optimum_is_installed_before_gptq_model(self):
        optimum = '"optimum==1.27.0"'
        gptq = 'python -m pip install --no-build-isolation gptqmodel==2.0.0'
        self.assertIn(optimum, self.source)
        self.assertIn(gptq, self.source)
        self.assertLess(self.source.index(optimum), self.source.index(gptq))

    def test_gpu_probe_reports_optimum_version(self):
        self.assertIn('package_version("optimum")', self.source)
        self.assertIn('"optimum": str(package_version("optimum"))', self.source)
        self.assertIn(
            "Qwen TTS, Optimum, and Omni runtime import preflight passed",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
