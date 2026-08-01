import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalProbeSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_framework_versions_are_cast_to_plain_strings(self):
        for expression in (
            "str(torch.__version__)",
            "str(torchaudio.__version__)",
            "str(torchvision.__version__)",
            "str(transformers.__version__)",
            "str(numpy.__version__)",
            "str(numba.__version__)",
            "str(decord.__version__)",
        ):
            with self.subTest(expression=expression):
                self.assertIn(expression, self.source)

    def test_probe_result_is_json_round_tripped_before_modal_serialization(self):
        self.assertIn(
            "return json.loads(json.dumps(probe, default=str))",
            self.source,
        )
        self.assertIn("int(value) for value in torch.cuda.get_device_capability(0)", self.source)


if __name__ == "__main__":
    unittest.main()
