import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from factory.config import Settings
from factory.models import AudioMetrics, NarrationSegment, VoiceContract
from factory.reviewer import OpenAIRealtimeReviewer, ReviewerError


PASSING = {
    "decision": "approve",
    "overall_score": 0.94,
    "scores": {
        "script_fidelity": 1.0,
        "naturalness": 0.92,
        "authority": 0.93,
        "engagement": 0.90,
        "pronunciation": 0.98,
        "pace": 0.91,
        "pause_quality": 0.91,
        "emotional_match": 0.90,
        "audio_artifacts": 0.98,
    },
    "failed_segments": [],
    "summary": "Publication quality.",
}


class ReviewerAPITests(unittest.TestCase):
    def settings(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "review-key"}, clear=True):
            return Settings.from_env()

    @staticmethod
    def metrics():
        return AudioMetrics(3, 24000, 1, -2, -18, 0, 0.05, 0.2, 150, 0, True)

    def test_posts_audio_as_text_only_review_and_disables_storage(self):
        response = Mock(status_code=200, text="")
        response.json.return_value = {"output_text": __import__("json").dumps(PASSING)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.wav"
            path.write_bytes(b"RIFF" + b"0" * 2000)
            segment = NarrationSegment(0, "Exact words.", "Speak clearly.", path, 0, 3)
            with patch("factory.reviewer.requests.post", return_value=response) as post:
                review = OpenAIRealtimeReviewer(self.settings()).review(
                    audio_path=path,
                    narration="Exact words.",
                    contract=VoiceContract(),
                    segments=[segment],
                    metrics=self.metrics(),
                    attempt=1,
                )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gpt-realtime-2.1")
        self.assertFalse(payload["store"])
        audio = payload["input"][0]["content"][1]
        self.assertEqual(audio["type"], "input_audio")
        self.assertEqual(audio["input_audio"]["format"], "wav")
        self.assertEqual(review.decision, "approve")

    def test_retries_one_transient_transport_failure(self):
        good = Mock(status_code=200, text="")
        good.json.return_value = {"output_text": __import__("json").dumps(PASSING)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.wav"
            path.write_bytes(b"RIFF" + b"0" * 2000)
            segment = NarrationSegment(0, "Text.", "Instruction.", path)
            with patch(
                "factory.reviewer.requests.post", side_effect=[OSError("network"), good]
            ) as post, patch("factory.reviewer.time.sleep"):
                review = OpenAIRealtimeReviewer(self.settings()).review(
                    audio_path=path,
                    narration="Text.",
                    contract=VoiceContract(),
                    segments=[segment],
                    metrics=self.metrics(),
                    attempt=1,
                )
        self.assertEqual(post.call_count, 2)
        self.assertEqual(review.decision, "approve")

    def test_rejects_oversized_audio_before_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "huge.wav"
            path.write_bytes(b"0" * (15 * 1024 * 1024 + 1))
            with patch("factory.reviewer.requests.post") as post:
                with self.assertRaisesRegex(ReviewerError, "15 MiB"):
                    OpenAIRealtimeReviewer(self.settings()).review(
                        audio_path=path,
                        narration="Text.",
                        contract=VoiceContract(),
                        segments=[],
                        metrics=self.metrics(),
                        attempt=1,
                    )
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
