import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from factory import local_llm
from factory.config import Settings
from factory.feeds import SourceItem
from factory.policy import Strategy
from factory.source_attributed_llm import (
    _repair_scene_attribution,
    generate_package,
)


class SourceAttributedLLMTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.sources = [
            SourceItem(
                "OpenAI",
                "Release A",
                "https://a.example/news",
                "Source A supports the opening and practical implications.",
                now,
            ),
            SourceItem(
                "NVIDIA",
                "Release B",
                "https://b.example/news",
                "Source B supports the technical mechanism and limitations.",
                now,
            ),
            SourceItem(
                "Google",
                "Release C",
                "https://c.example/news",
                "Source C supports deployment details and caveats.",
                now,
            ),
        ]
        self.source_urls = [source.url for source in self.sources]
        self.strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")

    @staticmethod
    def _response(payload: dict) -> Mock:
        response = Mock(status_code=200, text="")
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        return response

    def _raw_package(self, indices: list[int], source_count: int = 2) -> dict:
        urls = self.source_urls[:source_count]
        publishers = [source.publisher for source in self.sources[:source_count]]
        return {
            "topic": "A supported AI development",
            "narration": " ".join(f"word{index}" for index in range(145)),
            "title": "What changed in AI",
            "description": "Evidence-based summary.",
            "tags": [
                "AI",
                "engineering",
                "models",
                "deployment",
                "research",
                "tools",
                "technology",
                "update",
            ],
            "thumbnail_text": "WHAT CHANGED",
            "top_comment": "What would you test first?",
            "source_urls": urls,
            "source_publishers": publishers,
            "scenes": [
                {
                    "heading": f"Scene {scene_id}",
                    "body": f"Evidence-backed claim {scene_id}.",
                    "visual": "Procedural evidence card.",
                    "source_index": indices[scene_id],
                }
                for scene_id in range(6)
            ],
        }

    def test_exact_selected_urls_are_mapped_to_internal_indices(self):
        scenes = self._raw_package([0, 1, 0, 1, 0, 5])["scenes"]
        payload = {
            "assignments": [
                {
                    "scene_id": scene_id,
                    "source_url": self.source_urls[scene_id % 2],
                }
                for scene_id in range(6)
            ],
            "unsupported_scene_ids": [],
        }
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.source_attributed_llm.requests.post",
            return_value=self._response(payload),
        ) as post:
            indices = _repair_scene_attribution(
                Settings.from_env(),
                scenes,
                self.source_urls[:2],
                self.sources,
            )

        self.assertEqual(indices, [0, 1, 0, 1, 0, 1])
        request_payload = post.call_args.kwargs["json"]
        source_enum = request_payload["response_format"]["schema"]["properties"][
            "assignments"
        ]["items"]["properties"]["source_url"]["enum"]
        self.assertEqual(source_enum, self.source_urls[:2])
        prompt = request_payload["messages"][1]["content"]
        self.assertIn("Do not use numeric source positions", prompt)

    def test_unknown_url_is_rejected_without_clamping(self):
        scenes = self._raw_package([0, 1, 0, 1, 0, 5])["scenes"]
        payload = {
            "assignments": [
                {
                    "scene_id": scene_id,
                    "source_url": (
                        "https://invented.example/post"
                        if scene_id == 5
                        else self.source_urls[scene_id % 2]
                    ),
                }
                for scene_id in range(6)
            ],
            "unsupported_scene_ids": [],
        }
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.source_attributed_llm.requests.post",
            return_value=self._response(payload),
        ) as post:
            with self.assertRaisesRegex(
                local_llm.LocalLLMError,
                r"Exact scene-source attribution failed after 2 attempts: .*unselected source URL",
            ):
                _repair_scene_attribution(
                    Settings.from_env(),
                    scenes,
                    self.source_urls[:2],
                    self.sources,
                )
        self.assertEqual(post.call_count, 2)

    def test_unsupported_scene_fails_closed_immediately(self):
        scenes = self._raw_package([0, 1, 0, 1, 0, 5])["scenes"]
        payload = {
            "assignments": [
                {
                    "scene_id": scene_id,
                    "source_url": self.source_urls[scene_id % 2],
                }
                for scene_id in range(6)
            ],
            "unsupported_scene_ids": [4],
        }
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.source_attributed_llm.requests.post",
            return_value=self._response(payload),
        ) as post:
            with self.assertRaisesRegex(
                local_llm.LocalLLMError,
                r"do not support scene\(s\): 4",
            ):
                _repair_scene_attribution(
                    Settings.from_env(),
                    scenes,
                    self.source_urls[:2],
                    self.sources,
                )
        self.assertEqual(post.call_count, 1)

    def test_invalid_numeric_attribution_is_repaired_at_package_boundary(self):
        raw = self._raw_package([0, 1, 0, 1, 0, 5])

        def fake_generate(settings, sources, strategy):
            del strategy
            return local_llm._package_from_raw(settings, sources, raw)

        with patch.dict("os.environ", {}, clear=True), patch.object(
            local_llm,
            "generate_package",
            side_effect=fake_generate,
        ), patch(
            "factory.source_attributed_llm._repair_scene_attribution",
            return_value=[0, 1, 0, 1, 0, 1],
        ) as repair:
            package = generate_package(
                Settings.from_env(),
                self.sources,
                self.strategy,
            )

        self.assertEqual(
            [scene.source_index for scene in package.scenes],
            [0, 1, 0, 1, 0, 1],
        )
        repair.assert_called_once()

    def test_all_nonzero_zero_based_indices_are_not_shifted(self):
        raw = self._raw_package([1, 2, 1, 2, 1, 2], source_count=3)

        def fake_generate(settings, sources, strategy):
            del strategy
            return local_llm._package_from_raw(settings, sources, raw)

        with patch.dict("os.environ", {}, clear=True), patch.object(
            local_llm,
            "generate_package",
            side_effect=fake_generate,
        ), patch(
            "factory.source_attributed_llm._repair_scene_attribution"
        ) as repair:
            package = generate_package(
                Settings.from_env(),
                self.sources,
                self.strategy,
            )

        self.assertEqual(
            [scene.source_index for scene in package.scenes],
            [1, 2, 1, 2, 1, 2],
        )
        repair.assert_not_called()

    def test_package_and_normalizer_are_restored_when_generation_raises(self):
        original_package = local_llm._package_from_raw
        original_normalizer = local_llm._normalize_scene_source_indices
        with patch.dict("os.environ", {}, clear=True), patch.object(
            local_llm,
            "generate_package",
            side_effect=RuntimeError("generation failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                generate_package(
                    Settings.from_env(),
                    self.sources,
                    self.strategy,
                )
        self.assertIs(local_llm._package_from_raw, original_package)
        self.assertIs(local_llm._normalize_scene_source_indices, original_normalizer)


if __name__ == "__main__":
    unittest.main()
