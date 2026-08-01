from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from factory.config import Settings


class ConfigTests(unittest.TestCase):
    def test_defaults_are_private_and_open_weight(self) -> None:
        with patch.dict(os.environ, {"PUBLISH_ENABLED": "false"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.privacy_status, "private")
        self.assertEqual(settings.reviewer_backend, "qwen_omni")
        self.assertFalse(settings.publish_enabled)

    def test_publish_requires_oauth(self) -> None:
        with patch.dict(os.environ, {"PUBLISH_ENABLED": "true"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
