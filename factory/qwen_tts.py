from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings


class Qwen3TTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None
        self._soundfile: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        if self.settings.qwen_tts_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Qwen3-TTS requires CUDA in the configured production worker")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.settings.qwen_tts_dtype]
        self._model = Qwen3TTSModel.from_pretrained(self.settings.qwen_tts_model, device_map=self.settings.qwen_tts_device, dtype=dtype, attn_implementation=self.settings.qwen_tts_attention)
        self._soundfile = sf

    def generate(self, text: str, instruction: str, output_path: Path, seed: int) -> Path:
        self._load()
        import torch

        torch.manual_seed(seed)
        if self.settings.qwen_tts_mode == "custom_voice":
            wavs, rate = self._model.generate_custom_voice(text=text, language=self.settings.qwen_tts_language, speaker=self.settings.qwen_tts_speaker, instruct=instruction, do_sample=True, top_p=0.92, temperature=0.78)
        else:
            prompt = self._model.create_voice_clone_prompt(ref_audio=str(self.settings.qwen_ref_audio), ref_text=self.settings.qwen_ref_text, x_vector_only_mode=False)
            wavs, rate = self._model.generate_voice_clone(text=text, language=self.settings.qwen_tts_language, voice_clone_prompt=prompt, do_sample=True, top_p=0.92, temperature=0.78)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._soundfile.write(str(output_path), wavs[0], rate, subtype="PCM_16")
        if output_path.stat().st_size < 1000:
            raise RuntimeError("Qwen3-TTS produced empty audio")
        return output_path

    def unload(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
