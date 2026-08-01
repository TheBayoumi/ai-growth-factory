import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from factory.config import Settings
from factory.feeds import SourceItem
from factory.local_llm import (
    LocalLLMError,
    _balanced_source_candidates,
    generate_package,
)
from factory.policy import Strategy


class PublisherDiversityTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.sources = [
            SourceItem("OpenAI", "OpenAI one", "https://openai.example/1", "Primary details", now),
            SourceItem("OpenAI", "OpenAI two", "https://openai.example/2", "More details", now),
            SourceItem("NVIDIA", "NVIDIA one", "https://nvidia.example/1", "Hardware context", now),
            SourceItem("Google AI", "Google one", "https://google.example/1", "Platform context", now),
        ]
        self.strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")

    @staticmethod
    def package(urls: list[str], publishers: list[str]) -> dict:
        narration = " ".join(f"word{index}" for index in range(145))
        return {
            "topic": "A supported AI development",
            "narration": narration,
            "title": "What changed in AI",
            "description": "Evidence-based summary.",
            "tags": ["AI", "engineering"] * 4,
            "thumbnail_text": "WHAT CHANGED",
            "top_comment": "What would you test first?",
            "source_urls": urls,
            "source_publishers": publishers,
            "scenes": [
                {
                    "heading": f"Scene {index}",
                    "body": "A concise evidence-backed point.",
                    "visual": "Procedural data card animation.",
                    "source_index": index % len(urls),
                }
                for index in range(6)
            ],
        }

    def test_balanced_candidates_prevent_one_publisher_from_dominating_front(self):
        balanced = _balanced_source_candidates(self.sources, limit=4)

        self.assertEqual(
            [source.publisher for source in balanced],
            ["OpenAI", "NVIDIA", "Google AI", "OpenAI"],
        )
        self.assertEqual(balanced[0].url, "https://openai.example/1")
        self.assertEqual(balanced[3].url, "https://openai.example/2")

    def test_balanced_candidates_validate_limit(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            _balanced_source_candidates(self.sources, limit=0)

    def test_initial_prompt_has_exact_distinct_publisher_contract(self):
        valid = self.package(
            ["https://openai.example/1", "https://nvidia.example/1"],
            ["OpenAI", "NVIDIA"],
        )
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", return_value=valid
        ) as chat:
            generate_package(Settings.from_env(), self.sources, self.strategy)

        prompt = chat.call_args.args[1]
        self.assertIn("at least 2 DISTINCT supplied publishers", prompt)
        self.assertIn("Multiple URLs from one publisher still count as one publisher", prompt)
        self.assertIn("PUBLISHER SOURCE OPTIONS", prompt)
        self.assertLess(prompt.index('"publisher": "NVIDIA"'), prompt.index('"url": "https://openai.example/2"'))

    def test_same_publisher_package_gets_precise_repair_and_can_recover(self):
        invalid = self.package(
            ["https://openai.example/1", "https://openai.example/2"],
            ["OpenAI", "OpenAI"],
        )
        corrected = self.package(
            ["https://openai.example/1", "https://nvidia.example/1"],
            ["OpenAI", "NVIDIA"],
        )

        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", side_effect=[invalid, corrected]
        ) as chat:
            package = generate_package(Settings.from_env(), self.sources, self.strategy)

        self.assertEqual(package.source_publishers, ["OpenAI", "NVIDIA"])
        self.assertEqual(chat.call_count, 2)
        repair_prompt = chat.call_args_list[1].args[1]
        self.assertIn("Package used 1 independent primary publisher(s); required 2", repair_prompt)
        self.assertIn('currently selected distinct publishers are: ["OpenAI"]', repair_prompt)
        self.assertIn('"publisher": "NVIDIA"', repair_prompt)
        self.assertIn("Never substitute a URL while retaining an unsupported claim", repair_prompt)

    def test_repeated_same_publisher_output_fails_after_three_attempts(self):
        invalid = self.package(
            ["https://openai.example/1", "https://openai.example/2"],
            ["OpenAI", "OpenAI"],
        )

        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", side_effect=[invalid, invalid, invalid]
        ) as chat:
            with self.assertRaisesRegex(
                LocalLLMError,
                "Package used 1 independent primary publisher",
            ):
                generate_package(Settings.from_env(), self.sources, self.strategy)
        self.assertEqual(chat.call_count, 3)

    def test_duplicate_source_urls_fail_closed(self):
        invalid = self.package(
            ["https://openai.example/1", "https://openai.example/1"],
            ["OpenAI", "OpenAI"],
        )

        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", side_effect=[invalid, invalid, invalid]
        ):
            with self.assertRaisesRegex(LocalLLMError, "must not contain duplicates"):
                generate_package(Settings.from_env(), self.sources, self.strategy)


if __name__ == "__main__":
    unittest.main()
