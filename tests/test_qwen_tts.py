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

    def __init__(self, cuda_available=True):
        super().__init__("torch")
        self.cuda = Mock()
        self.cuda.is_available.return_value = cuda_available
        self.manual_seed = Mock()


class QwenTTSTests(unittest.TestCase):
    def settings(self, extra=None):
        with patch.dict("os.environ", extra or {}, clear=True):
            return Settings.from_env()

    def test_custom_voice_passes_reviewer_instruction_and_seed(self):
        model = Mock()
        model.generate_custom_voice.return_value = ([[0.0] * 1200], 24000)
        qwen_module = types.ModuleType("qwen_tts")
        qwen_module.Qwen3TTSModel = Mock()
        qwen_module.Qwen3TTSModel.from_pretrained.return_value = model
        soundfile = types.ModuleType("soundfile")

        def write(path, wav, sample_rate, subtype):
            del wav, sample_rate, subtype
            Path(path).write_bytes(b"RIFF" + b"0" * 1500)

        soundfile.write = Mock(side_effect=write)
        torch = FakeTensorModule()
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
        model.generate_custom_voice.assert_called_once()
        self.assertEqual(
            model.generate_custom_voice.call_args.kwargs["instruct"],
            "Increase energy by fifteen percent.",
        )

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


if __name__ == "__main__":
    unittest.main()
