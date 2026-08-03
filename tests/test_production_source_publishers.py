from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from factory.config import Settings
from factory.feeds import SourceItem
from factory.local_llm import LocalLLMError, _package_from_raw
from factory.production_source_publishers import canonicalize_source_publishers


class ProductionSourcePublisherTests(unittest.TestCase):
    def setUp(self) -> None:
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
                "Independent context",
                now,
            ),
        ]

    @staticmethod
    def raw_package() -> dict[str, object]:
        return {
            "topic": "A supported AI development",
            "narration": " ".join(f"word{index}" for index in range(150)),
            "title": "What changed in AI",
            "description": "Evidence-based summary.",
            "tags": ["AI", "engineering"] * 4,
            "thumbnail_text": "WHAT CHANGED",
            "top_comment": "What would you test first?",
            "source_urls": ["https://a.example/news", "https://b.example/news"],
            "source_publishers": ["wrong"],
            "scenes": [
                {
                    "heading": f"Scene {index}",
                    "body": "A concise evidence-backed point.",
                    "visual": "Procedural abstract editorial visual.",
                    "source_index": index % 2,
                }
                for index in range(6)
            ],
        }

    def test_derives_one_for_one_publishers_from_selected_urls(self) -> None:
        raw = self.raw_package()
        corrected = canonicalize_source_publishers(raw, self.sources)

        self.assertEqual(corrected["source_publishers"], ["OpenAI", "NVIDIA"])
        self.assertEqual(raw["source_publishers"], ["wrong"])

        with patch.dict("os.environ", {}, clear=True):
            package = _package_from_raw(Settings.from_env(), self.sources, corrected)
        self.assertEqual(package.source_publishers, ["OpenAI", "NVIDIA"])

    def test_missing_model_publisher_metadata_is_completed(self) -> None:
        raw = self.raw_package()
        raw.pop("source_publishers")

        corrected = canonicalize_source_publishers(raw, self.sources)

        self.assertEqual(corrected["source_publishers"], ["OpenAI", "NVIDIA"])
        with patch.dict("os.environ", {}, clear=True):
            package = _package_from_raw(Settings.from_env(), self.sources, corrected)
        self.assertEqual(package.source_publishers, ["OpenAI", "NVIDIA"])

    def test_unknown_url_still_fails_closed(self) -> None:
        raw = self.raw_package()
        raw["source_urls"] = ["https://a.example/news", "https://invented.example/post"]
        corrected = canonicalize_source_publishers(raw, self.sources)

        self.assertEqual(corrected["source_publishers"], ["OpenAI", ""])
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(LocalLLMError, "not supplied"):
                _package_from_raw(Settings.from_env(), self.sources, corrected)

    def test_canonicalization_does_not_weaken_publisher_diversity(self) -> None:
        now = datetime.now(timezone.utc)
        same_publisher_sources = [
            SourceItem("OpenAI", "A", "https://a.example/news", "A", now),
            SourceItem("OpenAI", "B", "https://b.example/news", "B", now),
        ]
        corrected = canonicalize_source_publishers(self.raw_package(), same_publisher_sources)

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(LocalLLMError, "independent primary publisher"):
                _package_from_raw(Settings.from_env(), same_publisher_sources, corrected)

    def test_conflicting_catalog_publishers_fail_closed(self) -> None:
        now = datetime.now(timezone.utc)
        conflicting = [
            SourceItem("Publisher A", "A", "https://same.example", "A", now),
            SourceItem("Publisher B", "B", "https://same.example", "B", now),
        ]
        raw = self.raw_package()
        raw["source_urls"] = ["https://same.example", "https://a.example/news"]

        with self.assertRaisesRegex(LocalLLMError, "conflicting publishers"):
            canonicalize_source_publishers(raw, conflicting)


if __name__ == "__main__":
    unittest.main()
