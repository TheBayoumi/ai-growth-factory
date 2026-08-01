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
        ]
        self.source_urls = [source.url for source in self.sources]
        self.strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")
        self.scenes = [
            {
                "heading": f"Scene {scene_id}",
                "body": f"Evidence-backed claim {scene_id}.",
                "visual": "Procedural evidence card.",
                "source_index": 5 if scene_id == 5 else scene_id % 2,
            }
            for scene_id in range(6)
        ]

    @staticmethod
    def _response(payload: dict) -> Mock:
        response = Mock(status_code=200, text="")
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }
        return response

    def test_exact_selected_urls_are_mapped_to_internal_indices(self):
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
                self.scenes,
                self.source_urls,
                self.sources,
            )

        self.assertEqual(indices, [0, 1, 0, 1, 0, 1])
        request_payload = post.call_args.kwargs["json"]
        source_enum = request_payload["response_format"]["schema"]["properties"][
            "assignments"
        ]["items"]["properties"]["source_url"]["enum"]
        self.assertEqual(source_enum, self.source_urls)
        prompt = request_payload["messages"][1]["content"]
        self.assertIn("Do not use numeric source positions", prompt)
        self.assertIn(self.source_urls[0], prompt)
        self.assertIn(self.source_urls[1], prompt)

    def test_unknown_url_is_rejected_without_clamping(self):
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
                "unselected source URL",
            ):
                _repair_scene_attribution(
                    Settings.from_env(),
                    self.scenes,
                    self.source_urls,
                    self.sources,
                )
        self.assertEqual(post.call_count, 2)

    def test_unsupported_scene_fails_closed_immediately(self):
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
                "do not support scene\(s\): 4",
            ):
                _repair_scene_attribution(
                    Settings.from_env(),
                    self.scenes,
                    self.source_urls,
                    self.sources,
                )
        self.assertEqual(post.call_count, 1)

    def test_wrapper_repairs_invalid_indices_and_restores_normalizer(self):
        original_normalizer = local_llm._normalize_scene_source_indices

        def fake_generate(settings, sources, strategy):
            del settings, strategy
            return local_llm._normalize_scene_source_indices(
                self.scenes,
                self.source_urls,
                sources,
            )

        with patch.dict("os.environ", {}, clear=True), patch.object(
            local_llm,
            "generate_package",
            side_effect=fake_generate,
        ), patch(
            "factory.source_attributed_llm._repair_scene_attribution",
            return_value=[0, 1, 0, 1, 0, 1],
        ) as repair:
            result = generate_package(
                Settings.from_env(),
                self.sources,
                self.strategy,
            )

        self.assertEqual(result, [0, 1, 0, 1, 0, 1])
        repair.assert_called_once()
        self.assertIs(local_llm._normalize_scene_source_indices, original_normalizer)

    def test_valid_indices_do_not_trigger_attribution_request(self):
        valid_scenes = [dict(scene, source_index=scene_id % 2) for scene_id, scene in enumerate(self.scenes)]
        original_normalizer = local_llm._normalize_scene_source_indices

        def fake_generate(settings, sources, strategy):
            del settings, strategy
            return local_llm._normalize_scene_source_indices(
                valid_scenes,
                self.source_urls,
                sources,
            )

        with patch.dict("os.environ", {}, clear=True), patch.object(
            local_llm,
            "generate_package",
            side_effect=fake_generate,
        ), patch(
            "factory.source_attributed_llm._repair_scene_attribution"
        ) as repair:
            result = generate_package(
                Settings.from_env(),
                self.sources,
                self.strategy,
            )

        self.assertEqual(result, [0, 1, 0, 1, 0, 1])
        repair.assert_not_called()
        self.assertIs(local_llm._normalize_scene_source_indices, original_normalizer)

    def test_normalizer_is_restored_when_generation_raises(self):
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
        self.assertIs(local_llm._normalize_scene_source_indices, original_normalizer)


if __name__ == "__main__":
    unittest.main()
