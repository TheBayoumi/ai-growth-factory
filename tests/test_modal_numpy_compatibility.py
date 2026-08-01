import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalNativeOmniCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_native_omni_uses_pinned_numpy_and_numba(self):
        self.assertIn('"numpy==2.2.6"', self.source)
        self.assertIn('"numba==0.64.0"', self.source)
        self.assertNotIn("gptqmodel", self.source.lower())
        self.assertNotIn("optimum", self.source.lower())

    def test_image_preflight_imports_real_tts_and_native_omni(self):
        self.assertIn("from qwen_tts import Qwen3TTSModel", self.source)
        self.assertIn(
            "from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor",
            self.source,
        )
        self.assertIn("Qwen TTS and native Omni runtime import preflight passed", self.source)

    def test_production_resource_and_video_contract(self):
        self.assertIn('gpu="A10"', self.source)
        self.assertIn("memory=24576", self.source)
        self.assertIn('"QWEN_OMNI_REVIEW_MODEL": "Qwen/Qwen2.5-Omni-3B"', self.source)
        self.assertIn('"VIDEO_WIDTH": "1080"', self.source)
        self.assertIn('"VIDEO_HEIGHT": "1920"', self.source)
        self.assertIn('"VIDEO_FPS": "30"', self.source)


if __name__ == "__main__":
    unittest.main()
