import unittest
from datetime import datetime, timedelta, timezone

from factory.feeds import SourceItem
from factory.trend_sources import (
    fetch_trend_snapshot,
    parse_github_trending,
    parse_hacker_news_items,
    parse_hugging_face_models,
    parse_product_hunt_feed,
)


NOW = datetime(2026, 8, 2, 1, 0, tzinfo=timezone.utc)


class TrendSourceParserTests(unittest.TestCase):
    def test_product_hunt_keeps_only_ai_launches_as_trend_signals(self):
        feed = b"""<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns='http://www.w3.org/2005/Atom'>
          <entry><title>Agent Desk</title><link href='https://producthunt.com/posts/agent-desk'/>
          <updated>2026-08-02T00:00:00Z</updated><summary>AI agents for support teams</summary></entry>
          <entry><title>Water Bottle</title><link href='https://producthunt.com/posts/bottle'/>
          <updated>2026-08-02T00:00:00Z</updated><summary>A steel bottle</summary></entry>
        </feed>"""
        items = parse_product_hunt_feed(feed)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_kind, "trend")
        self.assertIn("not independent product verification", items[0].summary)

    def test_hugging_face_metrics_are_discovery_only(self):
        items = parse_hugging_face_models(
            [
                {
                    "id": "microsoft/Mage-Flow",
                    "pipeline_tag": "text-to-image",
                    "likes": 204,
                    "downloads": 891,
                    "trendingScore": 374,
                    "lastModified": "2026-08-02T00:00:00Z",
                    "tags": ["diffusers", "image-generation"],
                }
            ],
            now=NOW,
        )
        self.assertEqual(items[0].url, "https://huggingface.co/microsoft/Mage-Flow")
        self.assertEqual(items[0].trend_score, 374.0)
        self.assertIn("not proof of model quality", items[0].summary)

    def test_github_parser_extracts_ai_repo_and_daily_velocity(self):
        page = """
        <article class="Box-row">
          <h2><a href="/example/agent-runtime">example / agent-runtime</a></h2>
          <p>An agentic AI runtime for reliable tool execution.</p>
          <span>321 stars today</span>
        </article>
        <article class="Box-row">
          <h2><a href="/example/css-kit">example / css-kit</a></h2>
          <p>A collection of stylesheets.</p><span>900 stars today</span>
        </article>
        """
        items = parse_github_trending(page, now=NOW)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "example/agent-runtime")
        self.assertEqual(items[0].trend_score, 321.0)

    def test_hacker_news_parser_filters_age_and_topic(self):
        payloads = [
            {
                "id": 1,
                "type": "story",
                "title": "New multimodal model released",
                "url": "https://example.com/model",
                "time": int((NOW - timedelta(hours=2)).timestamp()),
                "score": 100,
                "descendants": 20,
            },
            {
                "id": 2,
                "type": "story",
                "title": "A database migration guide",
                "url": "https://example.com/database",
                "time": int((NOW - timedelta(hours=2)).timestamp()),
                "score": 500,
            },
        ]
        items = parse_hacker_news_items(payloads, now=NOW, max_age_hours=24)
        self.assertEqual([item.url for item in items], ["https://news.ycombinator.com/item?id=1"])
        self.assertEqual(items[0].trend_score, 110.0)

    def test_snapshot_fails_soft_per_provider_and_deduplicates(self):
        signal = SourceItem(
            "GitHub Trending",
            "agent-runtime",
            "https://github.com/example/agent-runtime",
            "AI agent runtime",
            datetime.now(timezone.utc),
            "trend",
            50.0,
        )

        def good(*, timeout_seconds):
            del timeout_seconds
            return [signal, signal]

        def broken(*, timeout_seconds):
            del timeout_seconds
            raise RuntimeError("network down")

        snapshot = fetch_trend_snapshot(
            max_age_hours=24,
            fetchers=(("good", good), ("broken", broken)),
        )
        self.assertEqual(len(snapshot.items), 1)
        self.assertEqual(dict(snapshot.provider_status)["good"], "ok:2")
        self.assertEqual(dict(snapshot.provider_status)["broken"], "error:RuntimeError")


if __name__ == "__main__":
    unittest.main()
