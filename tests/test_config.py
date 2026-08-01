import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.config import Settings


class ConfigTests(unittest.TestCase):
    def test_setup_mode_needs_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
            self.assertFalse(settings.publish_enabled)
            self.assertFalse(settings.setup_status["youtube"])
            self.assertEqual(settings.qwen_tts_model, "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
            self.assertEqual(settings.reviewer_backend, "qwen_omni")
            self.assertEqual(settings.reviewer_model, "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4")

    def test_publish_mode_fails_closed(self):
        with patch.dict(os.environ, {"PUBLISH_ENABLED": "true"}, clear=True):
            with self.assertRaisesRegex(ValueError, "missing"):
                Settings.from_env()

    def test_valid_publish_configuration(self):
        env = {
            "PUBLISH_ENABLED": "true",
            "OPENAI_API_KEY": "review-key",
            "YOUTUBE_OAUTH_JSON": '{"client_id":"a","client_secret":"b","refresh_token":"c"}',
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
            self.assertTrue(settings.publish_enabled)
            self.assertTrue(settings.reviewer_required)

    def test_voice_contract_is_validated(self):
        env = {"VOICE_CONTRACT_JSON": '{"target_wpm":250}'}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "target_wpm"):
                Settings.from_env()

    def test_clone_mode_requires_owned_reference(self):
        with patch.dict(os.environ, {"QWEN_TTS_MODE": "voice_clone"}, clear=True):
            with self.assertRaisesRegex(ValueError, "QWEN_REF_AUDIO"):
                Settings.from_env()

    def test_clone_mode_accepts_reference_and_transcript(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.wav"
            reference.write_bytes(b"RIFF")
            env = {
                "QWEN_TTS_MODE": "voice_clone",
                "QWEN_REF_AUDIO": str(reference),
                "QWEN_REF_TEXT": "This is my authorized reference recording.",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env()
                self.assertEqual(settings.qwen_tts_mode, "voice_clone")


if __name__ == "__main__":
    unittest.main()
