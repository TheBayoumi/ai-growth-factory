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
        self._effective_dtype: str | None = None

    def _resolve_dtype(self, torch: Any) -> Any:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        requested = self.settings.qwen_tts_dtype
        resolved = dtype_map[requested]
        self._effective_dtype = requested

        if not self.settings.qwen_tts_device.startswith("cuda"):
            return resolved

        if requested == "float16":
            try:
                capability = torch.cuda.get_device_capability(self.settings.qwen_tts_device)
                major = int(capability[0])
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                raise QwenTTSError(
                    "Could not determine CUDA capability for safe Qwen3-TTS dtype selection"
                ) from exc
            if major < 8:
                self._effective_dtype = "float32"
                return torch.float32

        if requested == "bfloat16":
            try:
                supported = bool(torch.cuda.is_bf16_supported())
            except AttributeError:
                supported = False
            if not supported:
                self._effective_dtype = "float32"
                return torch.float32

        return resolved

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
        if self.settings.qwen_tts_device.startswith("cuda") and not torch.cuda.is_available():
            raise QwenTTSError(
                f"QWEN_TTS_DEVICE={self.settings.qwen_tts_device} but CUDA is not available"
            )
        kwargs: dict[str, Any] = {
            "device_map": self.settings.qwen_tts_device,
            "dtype": self._resolve_dtype(torch),
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
            generation_kwargs = {"max_new_tokens": 2048}
            if self.settings.qwen_tts_mode == "custom_voice":
                wavs, sample_rate = self._model.generate_custom_voice(
                    text=text,
                    language=self.settings.qwen_tts_language,
                    speaker=self.settings.qwen_tts_speaker,
                    instruct=instruction,
                    **generation_kwargs,
                )
            else:
                wavs, sample_rate = self._model.generate_voice_clone(
                    text=text,
                    language=self.settings.qwen_tts_language,
                    voice_clone_prompt=self._clone_prompt,
                    **generation_kwargs,
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
