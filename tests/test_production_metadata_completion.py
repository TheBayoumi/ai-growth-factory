import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from factory.config import Settings
from factory.feeds import SourceItem
from factory.local_llm import LocalLLMError, generate_package
from factory.policy import Strategy


ROOT = Path(__file__).resolve().parents[1]


class ProductionMetadataCompletionTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.sources = [
            SourceItem(
                "OpenAI",
                "Release A",
                "https://a.example/news",
                "Primary details",
                now,
            ),
            SourceItem(
                "NVIDIA",
                "Release B",
                "https://b.example/news",
                "Hardware context",
                now,
            ),
        ]
        self.strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")

    @staticmethod
    def core_package() -> dict:
        return {
            "topic": "A supported AI development",
            "narration": " ".join(f"word{index}" for index in range(145)),
            "title": "What Changed in AI Engineering",
            "description": "Evidence-based summary.",
            "source_urls": ["https://a.example/news", "https://b.example/news"],
            "source_publishers": ["OpenAI", "NVIDIA"],
            "scenes": [
                {
                    "heading": f"Scene {index}",
                    "body": "A concise evidence-backed point.",
                    "visual": "Procedural data card animation.",
                    "source_index": index % 2,
                }
                for index in range(6)
            ],
        }

    def test_missing_display_metadata_is_completed_without_an_llm_retry(self):
        raw = self.core_package()
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", return_value=raw
        ) as chat:
            package = generate_package(Settings.from_env(), self.sources, self.strategy)

        self.assertEqual(chat.call_count, 1)
        self.assertGreaterEqual(len(package.tags), 8)
        self.assertLessEqual(len(package.tags), 14)
        self.assertEqual(len(package.thumbnail_text.split()), 4)
        self.assertLessEqual(len(package.thumbnail_text), 45)
        self.assertEqual(
            package.top_comment,
            "What would you test first? Subscribe for evidence-backed AI updates.",
        )

    def test_partial_display_metadata_is_sanitized_and_completed(self):
        raw = self.core_package()
        raw["tags"] = ["AI", "ai", "", "engineering"]
        raw["thumbnail_text"] = "AI"
        raw["top_comment"] = ""
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", return_value=raw
        ):
            package = generate_package(Settings.from_env(), self.sources, self.strategy)

        self.assertEqual(len({tag.casefold() for tag in package.tags}), len(package.tags))
        self.assertGreaterEqual(len(package.tags), 8)
        self.assertGreaterEqual(len(package.thumbnail_text.split()), 2)
        self.assertLessEqual(len(package.thumbnail_text.split()), 5)

    def test_missing_factual_core_field_still_fails_closed_after_three_attempts(self):
        raw = self.core_package()
        del raw["description"]
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", side_effect=[raw, raw, raw]
        ) as chat:
            with self.assertRaisesRegex(LocalLLMError, "Package missing keys: description"):
                generate_package(Settings.from_env(), self.sources, self.strategy)
        self.assertEqual(chat.call_count, 3)

    def test_modal_numpy_pin_satisfies_gptq_and_numba_ranges(self):
        source = (ROOT / "cloud" / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn('"numpy==2.2.6"', source)
        self.assertIn("numpy==2.2.6 numba==0.64.0", source)
        self.assertIn("gptqmodel==5.7.0", source)
        self.assertNotIn('"numpy==2.0.0"', source)
        self.assertNotIn('"numpy==2.4.6"', source)


if __name__ == "__main__":
    unittest.main()
