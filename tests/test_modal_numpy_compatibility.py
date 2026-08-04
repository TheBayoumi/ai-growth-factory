import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalNativeOmniCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_quantized_omni_preserves_pinned_numpy_and_numba(self):
        self.assertIn('"numpy==2.2.6"', self.source)
        self.assertIn('"numba==0.64.0"', self.source)
        self.assertIn("optimum==1.27.0", self.source)
        self.assertIn("gptqmodel==5.7.0 --no-build-isolation", self.source)

    def test_image_preflight_imports_voice_reviewer_and_visual_models(self):
        self.assertIn("from qwen_tts import Qwen3TTSModel", self.source)
        self.assertIn("import decord, gptqmodel", self.source)
        self.assertIn("numpy, optimum, torch", self.source)
        self.assertIn("Qwen2_5OmniForConditionalGeneration", self.source)
        self.assertIn("Qwen2_5OmniProcessor", self.source)
        self.assertIn("AutoencoderKLWan", self.source)
        self.assertIn("WanImageToVideoPipeline", self.source)
        self.assertIn("FluxPipeline", self.source)
        self.assertIn("StableDiffusionXLPipeline", self.source)
        self.assertIn(
            "Voice, GPTQ reviewer, image, and Wan2.2 runtime import preflight passed",
            self.source,
        )

    def test_production_resource_and_video_contract(self):
        self.assertIn('gpu="A10"', self.source)
        self.assertIn("memory=65536", self.source)
        self.assertIn("timeout=85 * 60", self.source)
        self.assertIn(
            '"QWEN_OMNI_REVIEW_MODEL": "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"',
            self.source,
        )
        self.assertIn('"WAN22_MODEL_ID": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"', self.source)
        self.assertIn('"VIDEO_WIDTH": "1080"', self.source)
        self.assertIn('"VIDEO_HEIGHT": "1920"', self.source)
        self.assertIn('"VIDEO_FPS": "30"', self.source)


if __name__ == "__main__":
    unittest.main()
