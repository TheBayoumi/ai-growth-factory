from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings


class QwenTTSError(RuntimeError):
    pass


class Qwen3TTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None
        self._clone_prompt: Any = None
        self._soundfile: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import soundfile as sf
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise QwenTTSError(
                "Qwen3-TTS dependencies are missing. Install requirements-voice.txt in a Python 3.12 environment."
            ) from exc
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.settings.qwen_tts_device.startswith("cuda") and not torch.cuda.is_available():
            raise QwenTTSError(
                f"QWEN_TTS_DEVICE={self.settings.qwen_tts_device} but CUDA is not available"
            )
        kwargs: dict[str, Any] = {
            "device_map": self.settings.qwen_tts_device,
            "dtype": dtype_map[self.settings.qwen_tts_dtype],
            "attn_implementation": self.settings.qwen_tts_attention,
        }
        try:
            self._model = Qwen3TTSModel.from_pretrained(self.settings.qwen_tts_model, **kwargs)
        except Exception as exc:
            raise QwenTTSError(f"Could not load Qwen3-TTS model: {exc}") from exc
        self._soundfile = sf
        if self.settings.qwen_tts_mode == "voice_clone":
            assert self.settings.qwen_ref_audio is not None
            try:
                self._clone_prompt = self._model.create_voice_clone_prompt(
                    ref_audio=str(self.settings.qwen_ref_audio),
                    ref_text=self.settings.qwen_ref_text,
                    x_vector_only_mode=False,
                )
            except Exception as exc:
                raise QwenTTSError(f"Could not build reusable voice clone prompt: {exc}") from exc

    def generate(
        self,
        *,
        text: str,
        instruction: str,
        output_path: Path,
        seed: int,
    ) -> Path:
        self._load()
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if self.settings.qwen_tts_mode == "custom_voice":
                wavs, sample_rate = self._model.generate_custom_voice(
                    text=text,
                    language=self.settings.qwen_tts_language,
                    speaker=self.settings.qwen_tts_speaker,
                    instruct=instruction,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.78,
                )
            else:
                wavs, sample_rate = self._model.generate_voice_clone(
                    text=text,
                    language=self.settings.qwen_tts_language,
                    voice_clone_prompt=self._clone_prompt,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.78,
                )
        except Exception as exc:
            raise QwenTTSError(f"Qwen3-TTS generation failed: {exc}") from exc
        if not wavs:
            raise QwenTTSError("Qwen3-TTS returned no waveform")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._soundfile.write(str(output_path), wavs[0], sample_rate, subtype="PCM_16")
        if not output_path.exists() or output_path.stat().st_size < 1_000:
            raise QwenTTSError("Qwen3-TTS produced an empty audio file")
        return output_path

    def unload(self) -> None:
        self._model = None
        self._clone_prompt = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
