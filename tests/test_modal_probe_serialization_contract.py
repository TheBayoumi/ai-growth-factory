import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalProbeSerializationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_framework_version_objects_are_converted_to_plain_strings(self):
        for expression in (
            '"torch": str(torch.__version__)',
            '"torchaudio": str(torchaudio.__version__)',
            '"torchvision": str(torchvision.__version__)',
            '"transformers": str(transformers.__version__)',
            '"numpy": str(numpy.__version__)',
            '"numba": str(numba.__version__)',
            '"decord": str(decord.__version__)',
        ):
            with self.subTest(expression=expression):
                self.assertIn(expression, self.source)

    def test_probe_forces_json_round_trip_before_modal_serialization(self):
        self.assertIn(
            "return json.loads(json.dumps(result, ensure_ascii=False))",
            self.source,
        )
        self.assertIn(
            "[int(value) for value in torch.cuda.get_device_capability(0)]",
            self.source,
        )
        self.assertNotIn('"torch": torch.__version__', self.source)


if __name__ == "__main__":
    unittest.main()
