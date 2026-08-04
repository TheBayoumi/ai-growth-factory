from __future__ import annotations

import unittest
from pathlib import Path


class ModalReviewerRuntimeV28Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path("cloud/modal_app.py").read_text(encoding="utf-8")
        cls.runtime = Path("factory/production_voice_runtime_v28.py").read_text(
            encoding="utf-8"
        )
        cls.loader = Path("factory/production_qwen_omni_bitsandbytes_v28.py").read_text(
            encoding="utf-8"
        )

    def test_bitsandbytes_runtime_is_wheel_pinned(self) -> None:
        self.assertIn('"bitsandbytes==0.50.0"', self.source)
        self.assertNotIn("gptqmodel", self.source.casefold())
        self.assertNotIn("optimum==", self.source)
        self.assertNotIn("nvidia-smi", self.source)
        self.assertNotIn("nvcc", self.source.casefold())

    def test_image_build_imports_4bit_transformers_runtime(self) -> None:
        self.assertIn("import bitsandbytes, decord", self.source)
        self.assertIn("from transformers import BitsAndBytesConfig", self.source)
        self.assertIn("version('bitsandbytes') == '0.50.0'", self.source)
        self.assertIn(
            "Voice, bitsandbytes 4-bit Omni reviewer, image, and Wan2.2 runtime preflight passed",
            self.source,
        )

    def test_modal_and_runtime_use_same_base_7b_reviewer(self) -> None:
        expected = "Qwen/Qwen2.5-Omni-7B"
        self.assertIn(f'"QWEN_OMNI_REVIEW_MODEL": "{expected}"', self.source)
        self.assertIn(expected, self.runtime)
        self.assertNotIn("GPTQ", self.runtime)

    def test_loader_uses_nf4_nested_quantization(self) -> None:
        self.assertIn("load_in_4bit=True", self.loader)
        self.assertIn('bnb_4bit_quant_type="nf4"', self.loader)
        self.assertIn("bnb_4bit_compute_dtype=torch.float16", self.loader)
        self.assertIn("bnb_4bit_use_double_quant=True", self.loader)
        self.assertIn('"device_map": "auto"', self.loader)
        self.assertIn("disable_talker()", self.loader)

    def test_publishing_remains_disabled(self) -> None:
        self.assertIn('"PUBLISH_ENABLED": "false"', self.source)


if __name__ == "__main__":
    unittest.main()
