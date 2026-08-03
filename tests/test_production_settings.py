from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from factory.config import Settings
from factory.production_settings import install_production_settings


class ProductionSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_production_settings()

    def test_single_authority_value_is_accepted_after_runtime_install(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MIN_PRIMARY_SOURCES": "1",
                "PUBLISH_ENABLED": "false",
                "REVIEWER_BACKEND": "disabled",
            },
            clear=False,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.min_primary_sources, 1)

    def test_zero_still_fails_closed(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MIN_PRIMARY_SOURCES": "0",
                "PUBLISH_ENABLED": "false",
                "REVIEWER_BACKEND": "disabled",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "MIN_PRIMARY_SOURCES must be between 2 and 6"):
                Settings.from_env()

    def test_legacy_two_source_value_remains_valid(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MIN_PRIMARY_SOURCES": "2",
                "PUBLISH_ENABLED": "false",
                "REVIEWER_BACKEND": "disabled",
            },
            clear=False,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.min_primary_sources, 2)


if __name__ == "__main__":
    unittest.main()
