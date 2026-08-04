from __future__ import annotations

import unittest

from factory.production_qwen_omni_bitsandbytes_v28 import (
    build_omni_quantization_config_v28,
)


class _FakeTorch:
    float16 = object()


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class ProductionQwenOmniBitsAndBytesV28Tests(unittest.TestCase):
    def test_nf4_double_quant_configuration_is_exact(self) -> None:
        config = build_omni_quantization_config_v28(
            _FakeTorch,
            _FakeBitsAndBytesConfig,
        )
        self.assertEqual(
            config.kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": _FakeTorch.float16,
                "bnb_4bit_use_double_quant": True,
            },
        )

    def test_loader_source_forbids_gptq_fallback(self) -> None:
        source = __import__(
            "pathlib"
        ).Path("factory/production_qwen_omni_bitsandbytes_v28.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("gptqmodel", source.casefold())
        self.assertNotIn("optimum", source.casefold())
        self.assertIn("low_cpu_mem_usage", source)
        self.assertIn("torch.cuda.is_available", source)
        self.assertIn("disable_talker()", source)


if __name__ == "__main__":
    unittest.main()
