from __future__ import annotations

from typing import Any


_INSTALLED = False


def build_omni_quantization_config_v28(torch: Any, config_type: Any) -> Any:
    """Build the fail-closed 4-bit reviewer configuration.

    Qwen2.5-Omni-7B is loaded from its authoritative base checkpoint and quantized at load
    time. NF4 plus nested quantization keeps the reviewer inside the A10 memory envelope
    without requiring a CUDA source build in the Modal image-construction environment.
    """
    return config_type(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def install_production_qwen_omni_bitsandbytes_v28() -> None:
    """Load the 7B audio reviewer through Transformers' bitsandbytes integration."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import qwen_omni_reviewer
    from .reviewer import ReviewerError

    def load_4bit_reviewer(self: Any) -> None:
        if self._model is not None or self._segment_inference is not None:
            return
        try:
            import bitsandbytes  # noqa: F401
            import torch
            from qwen_omni_utils import process_mm_info
            from transformers import (
                BitsAndBytesConfig,
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as exc:
            raise ReviewerError(
                "Qwen Omni 7B reviewer dependencies are missing. The production image must "
                "include bitsandbytes and the Qwen Omni Transformers runtime."
            ) from exc

        if not torch.cuda.is_available():
            raise ReviewerError("Qwen Omni 7B reviewer requires an available CUDA device")
        if not str(self.settings.qwen_omni_device).startswith("cuda"):
            raise ReviewerError(
                "QWEN_OMNI_DEVICE must target CUDA for the production 7B reviewer"
            )

        quantization_config = build_omni_quantization_config_v28(
            torch,
            BitsAndBytesConfig,
        )
        kwargs: dict[str, Any] = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if self.settings.qwen_omni_attention:
            kwargs["attn_implementation"] = self.settings.qwen_omni_attention

        try:
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.settings.qwen_omni_model,
                **kwargs,
            )
            self._model.disable_talker()
            self._processor = Qwen2_5OmniProcessor.from_pretrained(
                self.settings.qwen_omni_model
            )
            self._process_mm_info = process_mm_info
        except Exception as exc:
            self._model = None
            self._processor = None
            self._process_mm_info = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                raise ReviewerError(
                    f"Could not load the bitsandbytes Qwen Omni 7B reviewer: {exc}"
                ) from exc

    qwen_omni_reviewer.QwenOmniReviewer._load = load_4bit_reviewer
    _INSTALLED = True
