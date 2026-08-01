import copy
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from factory.config import Settings
from factory.feeds import SourceItem
from factory.local_llm import LocalLLMError, generate_package
from factory.policy import Strategy


class SceneSourceIndexRepairTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.sources = [
            SourceItem(
                f"Publisher {index}",
                f"Release {index}",
                f"https://source{index}.example/release",
                f"Evidence summary for scene {index}",
                now,
            )
            for index in range(5)
        ]
        self.strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")

    def package(self) -> dict:
        return {
            "topic": "A supported AI development",
            "narration": " ".join(f"word{index}" for index in range(145)),
            "title": "A verified AI update",
            "description": "Evidence-backed explanation.",
            "tags": ["AI", "Engineering", "Research", "Models", "Tools", "News", "Data", "Testing"],
            "thumbnail_text": "VERIFIED AI UPDATE",
            "top_comment": "What would you test first?",
            "source_urls": [item.url for item in self.sources],
            "source_publishers": [item.publisher for item in self.sources],
            "scenes": [
                {
                    "heading": f"Scene {index}",
                    "body": f"Evidence-backed point {index}.",
                    "visual": f"Procedural visual {index}.",
                    "source_index": index if index < 5 else 5,
                }
                for index in range(6)
            ],
        }

    def test_targeted_repair_changes_only_source_indices(self):
        invalid = self.package()
        candidate = copy.deepcopy(invalid)
        candidate["topic"] = "MALICIOUS TOPIC CHANGE"
        candidate["narration"] = "MALICIOUS NARRATION CHANGE"
        candidate["source_urls"] = ["https://invented.example/"]
        candidate["scenes"][0]["heading"] = "MALICIOUS HEADING CHANGE"
        corrected_indices = [0, 1, 2, 3, 4, 0]
        for scene, source_index in zip(candidate["scenes"], corrected_indices, strict=True):
            scene["source_index"] = source_index

        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", side_effect=[invalid, candidate]
        ) as chat:
            package = generate_package(Settings.from_env(), self.sources, self.strategy)

        self.assertEqual(chat.call_count, 2)
        self.assertEqual(package.topic, invalid["topic"])
        self.assertEqual(package.narration, invalid["narration"])
        self.assertEqual(package.source_urls, invalid["source_urls"])
        self.assertEqual(
            [scene.heading for scene in package.scenes],
            [scene["heading"] for scene in invalid["scenes"]],
        )
        self.assertEqual(
            [scene.source_index for scene in package.scenes],
            corrected_indices,
        )
        repair_prompt = chat.call_args_list[1].args[1]
        self.assertIn("Valid ZERO-BASED source_index values are 0 through 4", repair_prompt)
        self.assertIn("Preserve every field", repair_prompt)

    def test_targeted_repair_fails_closed_after_three_invalid_attempts(self):
        invalid = self.package()
        repeated_invalid = copy.deepcopy(invalid)
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat",
            side_effect=[invalid, repeated_invalid, repeated_invalid, repeated_invalid],
        ) as chat:
            with self.assertRaisesRegex(
                LocalLLMError,
                "Scene source_index repair failed after 3 targeted attempts",
            ):
                generate_package(Settings.from_env(), self.sources, self.strategy)
        self.assertEqual(chat.call_count, 4)


if __name__ == "__main__":
    unittest.main()
