from __future__ import annotations

import unittest
from pathlib import Path


class ModalReviewerRuntimeV28Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path("cloud/modal_app.py").read_text(encoding="utf-8")

    def test_gptq_runtime_dependencies_are_pinned(self) -> None:
        self.assertIn("optimum==1.27.0", self.source)
        self.assertIn("gptqmodel==5.7.0 --no-build-isolation", self.source)
        self.assertIn("python3-dev", self.source)
        self.assertIn("ninja-build", self.source)

    def test_image_build_imports_gptq_runtime(self) -> None:
        self.assertIn("import decord, gptqmodel", self.source)
        self.assertIn("numpy, optimum, torch", self.source)
        self.assertIn("version('optimum') == '1.27.0'", self.source)
        self.assertIn("version('gptqmodel') == '5.7.0'", self.source)

    def test_modal_and_runtime_use_same_quantized_reviewer(self) -> None:
        expected = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
        self.assertIn(f'"QWEN_OMNI_REVIEW_MODEL": "{expected}"', self.source)
        runtime = Path("factory/production_voice_runtime_v28.py").read_text(encoding="utf-8")
        self.assertIn(expected, runtime)

    def test_publishing_remains_disabled(self) -> None:
        self.assertIn('"PUBLISH_ENABLED": "false"', self.source)


if __name__ == "__main__":
    unittest.main()
