from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import factory.production_scene_metadata as scene_metadata
from factory import local_llm
from factory.local_llm import LocalLLMError
from factory.production_scene_metadata import enforce_production_scene_metadata


class ProductionSceneMetadataTests(unittest.TestCase):
    @staticmethod
    def package() -> dict[str, object]:
        return {
            "scenes": [
                {
                    "heading": f"Scene {index}",
                    "body": "A complete evidence-backed statement for this scene.",
                    "visual": "One coherent physical composition with stable geometry.",
                    "source_index": 0,
                }
                for index in range(6)
            ]
        }

    def test_complete_scene_metadata_is_preserved_and_whitespace_normalized(self) -> None:
        raw = self.package()
        raw["scenes"][5]["body"] = (
            "The framework supports efficient and accessible research for the broader   community."
        )

        corrected = enforce_production_scene_metadata(raw)

        self.assertEqual(
            corrected["scenes"][5]["body"],
            "The framework supports efficient and accessible research for the broader community.",
        )
        self.assertEqual(raw["scenes"][5]["body"].count("   "), 1)

    def test_overlong_scene_body_fails_closed_instead_of_slicing_a_word(self) -> None:
        raw = self.package()
        raw["scenes"][5]["body"] = (
            "The framework focuses on scalability and performance so researchers can explore "
            "many applications while preserving efficiency and accessibility for the broader "
            "community worldwide."
        )

        with self.assertRaisesRegex(
            LocalLLMError,
            r"Scene 5 body exceeds 18-word limit: 23",
        ):
            enforce_production_scene_metadata(raw)

        self.assertTrue(str(raw["scenes"][5]["body"]).endswith("community worldwide."))

    def test_eighteen_long_words_cannot_cross_legacy_character_boundary(self) -> None:
        raw = self.package()
        raw["scenes"][1]["body"] = " ".join(["abcdefghijk"] * 18)

        with self.assertRaisesRegex(
            LocalLLMError,
            r"Scene 1 body exceeds 180-character limit: 215",
        ):
            enforce_production_scene_metadata(raw)

    def test_overlong_heading_and_visual_are_rejected(self) -> None:
        raw = self.package()
        raw["scenes"][0]["heading"] = "One two three four five six"
        with self.assertRaisesRegex(LocalLLMError, "heading exceeds 5-word limit"):
            enforce_production_scene_metadata(raw)

        raw = self.package()
        raw["scenes"][0]["visual"] = "x" * 401
        with self.assertRaisesRegex(LocalLLMError, "visual exceeds 400-character limit"):
            enforce_production_scene_metadata(raw)

    def test_installed_contract_runs_before_legacy_package_parser(self) -> None:
        raw = self.package()
        raw["scenes"][2]["body"] = "word " * 19
        original = Mock(return_value="parsed")
        previous_installed = scene_metadata._INSTALLED

        with patch.object(local_llm, "_package_from_raw", original):
            scene_metadata._INSTALLED = False
            try:
                scene_metadata.install_production_scene_metadata()
                with self.assertRaisesRegex(LocalLLMError, "Scene 2 body exceeds 18-word limit"):
                    local_llm._package_from_raw(None, None, raw)
                original.assert_not_called()
            finally:
                scene_metadata._INSTALLED = previous_installed


if __name__ == "__main__":
    unittest.main()
