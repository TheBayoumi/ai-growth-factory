import unittest
from datetime import datetime, timedelta, timezone

from factory.feeds import SourceItem
from factory.trend_ranking import align_primary_sources_to_trends
from factory.trend_sources import TrendSnapshot


NOW = datetime(2026, 8, 2, 1, 0, tzinfo=timezone.utc)


class TrendRankingTests(unittest.TestCase):
    def test_matching_official_source_is_ranked_first_without_exposing_trend_as_evidence(self):
        primary = [
            SourceItem(
                "Microsoft Research",
                "Mage Flow advances text-to-image generation",
                "https://microsoft.com/research/mage-flow",
                "Microsoft describes the Mage Flow architecture and evaluation.",
                NOW - timedelta(hours=8),
            ),
            SourceItem(
                "NVIDIA",
                "New GPU scheduling guide",
                "https://nvidia.com/blog/gpu-scheduling",
                "A guide to scheduling GPU jobs.",
                NOW - timedelta(hours=1),
            ),
        ]
        trend = SourceItem(
            "Hugging Face Trending",
            "microsoft/Mage-Flow — text-to-image",
            "https://huggingface.co/microsoft/Mage-Flow",
            "Trending text-to-image model with strong current attention.",
            NOW,
            "trend",
            374.0,
        )
        snapshot = TrendSnapshot((trend,), (("hugging_face", "ok:1"),), NOW)

        alignment = align_primary_sources_to_trends(primary, snapshot, now=NOW)

        self.assertEqual(alignment.ranked_sources[0].url, primary[0].url)
        self.assertTrue(all(item.is_primary for item in alignment.ranked_sources))
        self.assertNotIn(trend.url, [item.url for item in alignment.ranked_sources])
        self.assertEqual(alignment.matches[0].trend_url, trend.url)
        self.assertIn("mage", alignment.matches[0].overlap_terms)
        self.assertIn("flow", alignment.matches[0].overlap_terms)

    def test_no_trend_match_falls_back_to_primary_recency(self):
        older = SourceItem(
            "OpenAI",
            "Safety update",
            "https://openai.com/safety",
            "A safety update.",
            NOW - timedelta(hours=10),
        )
        newer = SourceItem(
            "Google AI",
            "Efficiency update",
            "https://blog.google/efficiency",
            "An efficiency update.",
            NOW - timedelta(hours=2),
        )
        unrelated = SourceItem(
            "Product Hunt",
            "Calendar app",
            "https://producthunt.com/calendar",
            "A calendar product.",
            NOW,
            "trend",
            100.0,
        )
        snapshot = TrendSnapshot((unrelated,), (("product_hunt", "ok:1"),), NOW)

        alignment = align_primary_sources_to_trends([older, newer], snapshot, now=NOW)

        self.assertEqual([item.url for item in alignment.ranked_sources], [newer.url, older.url])
        self.assertEqual(alignment.matches, ())

    def test_trend_items_passed_as_primary_input_are_excluded(self):
        trend = SourceItem(
            "GitHub Trending",
            "agent-runtime",
            "https://github.com/example/agent-runtime",
            "AI runtime",
            NOW,
            "trend",
            10.0,
        )
        snapshot = TrendSnapshot((trend,), (("github", "ok:1"),), NOW)
        alignment = align_primary_sources_to_trends([trend], snapshot, now=NOW)
        self.assertEqual(alignment.ranked_sources, ())


if __name__ == "__main__":
    unittest.main()
