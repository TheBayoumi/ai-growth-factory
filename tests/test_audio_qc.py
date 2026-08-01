import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from factory.audio_qc import analyze_audio, split_narration
from factory.config import Settings


def write_tone(path: Path, duration: float, sample_rate: int = 24000, amplitude: int = 7000) -> None:
    samples = array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(int(duration * sample_rate))
        ),
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


class AudioQCTests(unittest.TestCase):
    def test_split_narration_balances_sentences(self):
        text = "One short sentence. This sentence contains a few more useful words. Final sentence closes it."
        segments = split_narration(text, 3)
        self.assertEqual(len(segments), 3)
        self.assertEqual(" ".join(segments), text)

    def test_clean_tone_passes_relaxed_objective_gate(self):
        narration = " ".join(["word"] * 31)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "VOICE_CONTRACT_JSON": '{"target_wpm":155}',
                "AUDIO_PEAK_LIMIT_DBFS": "-1",
                "AUDIO_MIN_RMS_DBFS": "-40",
                "AUDIO_WPM_TOLERANCE": "10",
            },
            clear=True,
        ):
            settings = Settings.from_env()
            path = Path(temporary) / "voice.wav"
            write_tone(path, 12.0)
            metrics = analyze_audio(path, narration=narration, settings=settings)
            self.assertTrue(metrics.passed, metrics.failures)
            self.assertAlmostEqual(metrics.estimated_wpm, 155.0, places=1)


if __name__ == "__main__":
    unittest.main()
