import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from factory.feeds import (
    SourceItem,
    _clean,
    _date,
    _link,
    _parse,
    fetch_recent,
    publishers,
)


class FeedTests(unittest.TestCase):
    def test_clean_removes_markup_and_unescapes_entities(self):
        self.assertEqual(_clean("<p>AI &amp; ML</p>\nNext"), "AI & ML  Next")
        self.assertEqual(_clean(None), "")

    def test_date_accepts_rfc_iso_naive_and_empty_values(self):
        rfc = _date("Mon, 20 Jul 2026 10:00:00 GMT")
        iso = _date("2026-07-20T12:00:00+02:00")
        naive = _date("2026-07-20T10:00:00")
        empty = _date("")

        self.assertEqual(rfc, datetime(2026, 7, 20, 10, tzinfo=timezone.utc))
        self.assertEqual(iso, datetime(2026, 7, 20, 10, tzinfo=timezone.utc))
        self.assertEqual(naive.tzinfo, timezone.utc)
        self.assertEqual(empty.tzinfo, timezone.utc)

    def test_parse_supports_rss_atom_and_identifier_fallback(self):
        rss = b"""
        <rss><channel>
          <item>
            <title>RSS &amp; item</title>
            <link>https://example.com/rss</link>
            <description><![CDATA[<p>RSS summary</p>]]></description>
            <pubDate>Mon, 20 Jul 2026 10:00:00 GMT</pubDate>
          </item>
        </channel></rss>
        """
        atom = b"""
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Atom item</title>
            <link rel="alternate" href="https://example.com/atom" />
            <summary>Atom summary</summary>
            <updated>2026-07-20T10:00:00Z</updated>
          </entry>
          <entry>
            <title>ID fallback</title>
            <id>https://example.com/id</id>
            <content>Fallback content</content>
            <published>2026-07-20T09:00:00Z</published>
          </entry>
          <entry><title>Missing URL</title></entry>
        </feed>
        """

        rss_items = _parse(rss, "RSS Publisher")
        atom_items = _parse(atom, "Atom Publisher")

        self.assertEqual(len(rss_items), 1)
        self.assertEqual(rss_items[0].title, "RSS & item")
        self.assertEqual(rss_items[0].summary, "RSS summary")
        self.assertEqual(rss_items[0].url, "https://example.com/rss")
        self.assertEqual([item.url for item in atom_items], [
            "https://example.com/atom",
            "https://example.com/id",
        ])

    def test_link_accepts_text_and_ignores_non_alternate_href(self):
        import xml.etree.ElementTree as ElementTree

        node = ElementTree.fromstring(
            '<entry><link rel="enclosure" href="ignored"/><link>https://example.com/text</link></entry>'
        )
        self.assertEqual(_link(node), "https://example.com/text")

    def test_source_fingerprint_is_stable_and_content_sensitive(self):
        published = datetime(2026, 7, 20, tzinfo=timezone.utc)
        first = SourceItem("Publisher", "Title", "https://example.com", "A", published)
        same = SourceItem("Publisher", "Title", "https://example.com", "B", published)
        changed = SourceItem("Publisher", "Other", "https://example.com", "A", published)

        self.assertEqual(first.fingerprint, same.fingerprint)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        self.assertEqual(len(first.fingerprint), 16)

    def test_fetch_recent_filters_age_future_duplicates_and_feed_errors(self):
        now = datetime.now(timezone.utc)
        fresh_old = SourceItem(
            "Publisher",
            "Fresh old",
            "https://example.com/old-fresh",
            "",
            now - timedelta(hours=2),
        )
        fresh_new = SourceItem(
            "Publisher",
            "Fresh new",
            "https://example.com/new-fresh",
            "",
            now - timedelta(minutes=5),
        )
        duplicate = SourceItem(
            "Publisher",
            "Fresh new",
            "https://example.com/new-fresh",
            "Duplicate payload",
            now - timedelta(minutes=4),
        )
        stale = SourceItem(
            "Publisher",
            "Stale",
            "https://example.com/stale",
            "",
            now - timedelta(hours=25),
        )
        too_future = SourceItem(
            "Publisher",
            "Future",
            "https://example.com/future",
            "",
            now + timedelta(minutes=20),
        )
        response = Mock(content=b"feed")
        response.raise_for_status.return_value = None

        with patch("factory.feeds.FEEDS", (("One", "https://one"), ("Two", "https://two"))), patch(
            "factory.feeds.requests.get", side_effect=[response, OSError("offline")]
        ) as get, patch(
            "factory.feeds._parse",
            return_value=[fresh_old, fresh_new, duplicate, stale, too_future],
        ):
            result = fetch_recent(max_age_hours=24, timeout_seconds=3.5)

        self.assertEqual([item.title for item in result], ["Fresh new", "Fresh old"])
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["timeout"], 3.5)
        self.assertIn("AIGrowthFactory", get.call_args_list[0].kwargs["headers"]["User-Agent"])

    def test_publishers_returns_unique_names(self):
        published = datetime(2026, 7, 20, tzinfo=timezone.utc)
        items = [
            SourceItem("One", "A", "https://a", "", published),
            SourceItem("One", "B", "https://b", "", published),
            SourceItem("Two", "C", "https://c", "", published),
        ]
        self.assertEqual(publishers(items), {"One", "Two"})


if __name__ == "__main__":
    unittest.main()
