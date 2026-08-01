import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from factory.config import Settings
from factory.qwen_tts import Qwen3TTS, QwenTTSError


class FakeTensorModule(types.ModuleType):
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(
        self,
        cuda_available: bool = True,
        capability: tuple[int, int] = (8, 0),
        bf16_supported: bool = True,
    ) -> None:
        super().__init__("torch")
        self.cuda = Mock()
        self.cuda.is_available.return_value = cuda_available
        self.cuda.get_device_capability.return_value = capability
        self.cuda.is_bf16_supported.return_value = bf16_supported
        self.manual_seed = Mock()


class QwenTTSTests(unittest.TestCase):
    def settings(self, extra=None):
        with patch.dict("os.environ", extra or {}, clear=True):
            return Settings.from_env()

    @staticmethod
    def runtime_modules(model: Mock, torch: FakeTensorModule):
        qwen_module = types.ModuleType("qwen_tts")
        qwen_module.Qwen3TTSModel = Mock()
        qwen_module.Qwen3TTSModel.from_pretrained.return_value = model
        soundfile = types.ModuleType("soundfile")
        return qwen_module, soundfile

    def test_custom_voice_uses_seed_instruction_and_checkpoint_sampling_defaults(self):
        model = Mock()
        model.generate_custom_voice.return_value = ([[0.0] * 1200], 24000)
        torch = FakeTensorModule()
        qwen_module, soundfile = self.runtime_modules(model, torch)

        def write(path, wav, sample_rate, subtype):
            del wav, sample_rate, subtype
            Path(path).write_bytes(b"RIFF" + b"0" * 1500)

        soundfile.write = Mock(side_effect=write)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules,
            {"torch": torch, "qwen_tts": qwen_module, "soundfile": soundfile},
        ):
            output = Path(temporary) / "voice.wav"
            Qwen3TTS(self.settings()).generate(
                text="Exact transcript.",
                instruction="Increase energy by fifteen percent.",
                output_path=output,
                seed=123,
            )
            self.assertTrue(output.exists())

        torch.manual_seed.assert_called_once_with(123)
        torch.cuda.manual_seed_all.assert_called_once_with(123)
        model.generate_custom_voice.assert_called_once()
        kwargs = model.generate_custom_voice.call_args.kwargs
        self.assertEqual(kwargs["instruct"], "Increase energy by fifteen percent.")
        self.assertEqual(kwargs["max_new_tokens"], 2048)
        self.assertNotIn("do_sample", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("temperature", kwargs)

    def test_turing_float16_is_promoted_to_float32_before_model_load(self):
        model = Mock()
        torch = FakeTensorModule(capability=(7, 5))
        qwen_module, soundfile = self.runtime_modules(model, torch)

        with patch.dict(
            sys.modules,
            {"torch": torch, "qwen_tts": qwen_module, "soundfile": soundfile},
        ):
            engine = Qwen3TTS(self.settings({"QWEN_TTS_DTYPE": "float16"}))
            engine._load()

        kwargs = qwen_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["dtype"], torch.float32)
        self.assertEqual(engine._effective_dtype, "float32")
        torch.cuda.get_device_capability.assert_called_once_with("cuda:0")

    def test_ampere_float16_remains_float16(self):
        model = Mock()
        torch = FakeTensorModule(capability=(8, 0))
        qwen_module, soundfile = self.runtime_modules(model, torch)

        with patch.dict(
            sys.modules,
            {"torch": torch, "qwen_tts": qwen_module, "soundfile": soundfile},
        ):
            engine = Qwen3TTS(self.settings({"QWEN_TTS_DTYPE": "float16"}))
            engine._load()

        kwargs = qwen_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["dtype"], torch.float16)
        self.assertEqual(engine._effective_dtype, "float16")

    def test_unsupported_bfloat16_is_promoted_to_float32(self):
        model = Mock()
        torch = FakeTensorModule(capability=(7, 5), bf16_supported=False)
        qwen_module, soundfile = self.runtime_modules(model, torch)

        with patch.dict(
            sys.modules,
            {"torch": torch, "qwen_tts": qwen_module, "soundfile": soundfile},
        ):
            engine = Qwen3TTS(self.settings({"QWEN_TTS_DTYPE": "bfloat16"}))
            engine._load()

        kwargs = qwen_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["dtype"], torch.float32)
        self.assertEqual(engine._effective_dtype, "float32")

    def test_cuda_configuration_fails_when_cuda_is_unavailable(self):
        qwen_module = types.ModuleType("qwen_tts")
        qwen_module.Qwen3TTSModel = Mock()
        soundfile = types.ModuleType("soundfile")
        with patch.dict(
            sys.modules,
            {"torch": FakeTensorModule(False), "qwen_tts": qwen_module, "soundfile": soundfile},
        ):
            with self.assertRaisesRegex(QwenTTSError, "CUDA is not available"):
                Qwen3TTS(self.settings())._load()

    def test_missing_cuda_capability_fails_before_unsafe_float16_load(self):
        model = Mock()
        torch = FakeTensorModule()
        torch.cuda.get_device_capability.side_effect = AttributeError("capability unavailable")
        qwen_module, soundfile = self.runtime_modules(model, torch)

        with patch.dict(
            sys.modules,
            {"torch": torch, "qwen_tts": qwen_module, "soundfile": soundfile},
        ):
            with self.assertRaisesRegex(QwenTTSError, "CUDA capability"):
                Qwen3TTS(self.settings({"QWEN_TTS_DTYPE": "float16"}))._load()


if __name__ == "__main__":
    unittest.main()
