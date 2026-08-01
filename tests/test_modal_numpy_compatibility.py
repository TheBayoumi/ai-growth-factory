import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalNumPyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_numpy_and_numba_are_repinned_after_gptq_install(self):
        gptq = 'python -m pip install --no-build-isolation gptqmodel==5.7.0'
        compatibility = (
            'python -m pip install --force-reinstall '
            'numpy==2.2.6 numba==0.64.0'
        )
        self.assertIn(gptq, self.source)
        self.assertNotIn('gptqmodel==2.0.0', self.source)
        self.assertIn(compatibility, self.source)
        self.assertLess(self.source.index(gptq), self.source.index(compatibility))

    def test_runtime_probe_imports_qwen_tts_after_compatibility_pin(self):
        compatibility = 'numpy==2.2.6 numba==0.64.0'
        gptq_api = 'from gptqmodel.quantization import METHOD'
        optimum_api = 'from optimum.gptq import GPTQQuantizer'
        preflight = 'from qwen_tts import Qwen3TTSModel'
        self.assertLess(self.source.index(compatibility), self.source.index(gptq_api))
        self.assertLess(self.source.index(gptq_api), self.source.index(preflight))
        self.assertLess(self.source.index(optimum_api), self.source.index(preflight))
        self.assertIn('"numba": str(numba.__version__)', self.source)
        self.assertIn('"gptqmodel": str(package_version("gptqmodel"))', self.source)
        self.assertNotIn('"numba": numba.__version__', self.source)

    def test_production_vertical_output_is_full_hd(self):
        self.assertIn('"VIDEO_WIDTH": "1080"', self.source)
        self.assertIn('"VIDEO_HEIGHT": "1920"', self.source)
        self.assertIn('"VIDEO_FPS": "30"', self.source)


if __name__ == "__main__":
    unittest.main()
