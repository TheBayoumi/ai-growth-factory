import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class GPTQModelImageBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_optional_cuda_extension_is_disabled_in_cpu_image_builder(self):
        command = (
            "BUILD_CUDA_EXT=0 python -m pip install --no-build-isolation "
            "gptqmodel==5.7.0"
        )
        self.assertIn(command, self.source)
        self.assertNotIn(
            '"python -m pip install --no-build-isolation gptqmodel==5.7.0"',
            self.source,
        )

    def test_t4_probe_validates_method_and_transformers_quantizer(self):
        self.assertIn("from gptqmodel.quantization import METHOD", self.source)
        self.assertIn("from optimum.gptq import GPTQQuantizer", self.source)
        self.assertIn("GptqHfQuantizer(GPTQConfig(bits=4))", self.source)
        self.assertIn("quantizer.validate_environment()", self.source)
        self.assertIn('"gptq_environment_valid": True', self.source)
        self.assertIn('"gptq_method_api": True', self.source)


if __name__ == "__main__":
    unittest.main()
