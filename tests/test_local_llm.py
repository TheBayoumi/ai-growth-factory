import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from factory.config import Settings
from factory.feeds import SourceItem
from factory.local_llm import LocalLLMError, _chat, _extract_json, generate_package
from factory.policy import Strategy


class LocalLLMTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            SourceItem("OpenAI", "Release A", "https://a.example/news", "Primary details", datetime.now(timezone.utc)),
            SourceItem("NVIDIA", "Release B", "https://b.example/news", "Hardware context", datetime.now(timezone.utc)),
        ]
        self.strategy = Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")

    @staticmethod
    def package():
        narration = " ".join(f"word{index}" for index in range(140))
        return {
            "topic": "A supported AI development",
            "narration": narration,
            "title": "What changed in AI",
            "description": "Evidence-based summary.",
            "tags": ["AI", "engineering"] * 4,
            "thumbnail_text": "WHAT CHANGED",
            "top_comment": "What would you test first?",
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

    def test_extracts_embedded_json(self):
        self.assertEqual(_extract_json('prefix {"ok": true} suffix'), {"ok": True})

    def test_chat_uses_json_mode_and_auth(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        with patch.dict(
            "os.environ",
            {"LLAMA_CPP_API_KEY": "secret", "LLAMA_CPP_MODEL": "qwen"},
            clear=True,
        ), patch("factory.local_llm.requests.post", return_value=response) as post:
            result = _chat(Settings.from_env(), "prompt", attempts=1)
        self.assertTrue(result["ok"])
        request = post.call_args
        self.assertEqual(request.kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(request.kwargs["json"]["response_format"]["type"], "json_object")
        self.assertEqual(request.kwargs["json"]["model"], "qwen")

    def test_package_accepts_only_supplied_primary_sources(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", return_value=self.package()
        ):
            package = generate_package(Settings.from_env(), self.sources, self.strategy)
        self.assertEqual(package.source_publishers, ["OpenAI", "NVIDIA"])
        self.assertIn("https://a.example/news", package.description)
        self.assertEqual(len(package.scenes), 6)

    def test_package_repairs_undersized_narration(self):
        undersized = self.package()
        undersized["narration"] = " ".join(f"short{index}" for index in range(96))
        corrected = self.package()
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", side_effect=[undersized, corrected]
        ) as chat:
            package = generate_package(Settings.from_env(), self.sources, self.strategy)
        self.assertEqual(len(package.narration.split()), 140)
        self.assertEqual(chat.call_count, 2)
        repair_prompt = chat.call_args_list[1].args[1]
        self.assertIn("Narration word count outside quality gate: 96", repair_prompt)
        self.assertIn("135-175 whitespace-separated words", repair_prompt)

    def test_package_stops_after_three_invalid_content_attempts(self):
        undersized = self.package()
        undersized["narration"] = "too short"
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", side_effect=[undersized, undersized, undersized]
        ) as chat:
            with self.assertRaisesRegex(LocalLLMError, "failed after 3 attempts"):
                generate_package(Settings.from_env(), self.sources, self.strategy)
        self.assertEqual(chat.call_count, 3)

    def test_package_rejects_scene_source_index_out_of_range(self):
        raw = self.package()
        raw["scenes"][2]["source_index"] = 9
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", return_value=raw
        ):
            with self.assertRaisesRegex(LocalLLMError, "source_index out of range"):
                generate_package(Settings.from_env(), self.sources, self.strategy)

    def test_package_rejects_hallucinated_source(self):
        raw = self.package()
        raw["source_urls"][1] = "https://invented.example/post"
        with patch.dict("os.environ", {}, clear=True), patch(
            "factory.local_llm._chat", return_value=raw
        ):
            with self.assertRaisesRegex(LocalLLMError, "not supplied"):
                generate_package(Settings.from_env(), self.sources, self.strategy)


if __name__ == "__main__":
    unittest.main()
