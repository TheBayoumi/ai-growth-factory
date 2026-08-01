import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, call

from factory.feeds import SourceItem, fetch_diverse_recent


def _source(publisher: str, index: int) -> SourceItem:
    return SourceItem(
        publisher=publisher,
        title=f"Story {index}",
        url=f"https://{publisher.lower()}.example/{index}",
        summary="Primary-source summary",
        published_at=datetime.now(timezone.utc),
    )


class SourceSelectionTests(unittest.TestCase):
    def test_primary_window_stops_when_diverse(self):
        fetcher = Mock(return_value=[_source("Alpha", 1), _source("Beta", 2)])

        selection = fetch_diverse_recent(
            max_age_hours=48,
            min_publishers=2,
            fetcher=fetcher,
        )

        self.assertEqual(selection.max_age_hours, 48)
        self.assertEqual(selection.publisher_count, 2)
        fetcher.assert_called_once_with(max_age_hours=48, timeout_seconds=10.0)

    def test_sparse_primary_window_expands_to_seven_days(self):
        fetcher = Mock(
            side_effect=[
                [_source("Alpha", 1)],
                [_source("Alpha", 1), _source("Beta", 2)],
            ]
        )

        selection = fetch_diverse_recent(
            max_age_hours=48,
            min_publishers=2,
            fetcher=fetcher,
        )

        self.assertEqual(selection.max_age_hours, 168)
        self.assertEqual(selection.publishers, frozenset({"Alpha", "Beta"}))
        self.assertEqual(
            fetcher.call_args_list,
            [
                call(max_age_hours=48, timeout_seconds=10.0),
                call(max_age_hours=168, timeout_seconds=10.0),
            ],
        )

    def test_fallback_never_exceeds_seven_days(self):
        fetcher = Mock(
            side_effect=[
                [_source("Alpha", 1)],
                [_source("Alpha", 1), _source("Alpha", 2)],
            ]
        )

        selection = fetch_diverse_recent(
            max_age_hours=48,
            min_publishers=2,
            fallback_max_age_hours=500,
            fetcher=fetcher,
        )

        self.assertEqual(selection.max_age_hours, 168)
        self.assertEqual(fetcher.call_args_list[-1], call(max_age_hours=168, timeout_seconds=10.0))

    def test_returns_best_available_without_weakening_requirement(self):
        fetcher = Mock(
            side_effect=[
                [_source("Alpha", 1)],
                [_source("Alpha", 1), _source("Alpha", 2)],
            ]
        )

        selection = fetch_diverse_recent(
            max_age_hours=48,
            min_publishers=2,
            fetcher=fetcher,
        )

        self.assertEqual(selection.max_age_hours, 168)
        self.assertEqual(selection.publisher_count, 1)
        self.assertEqual(len(selection.items), 2)
        self.assertLess(selection.publisher_count, 2)

    def test_rejects_invalid_selection_contract(self):
        with self.assertRaisesRegex(ValueError, "max_age_hours"):
            fetch_diverse_recent(max_age_hours=0, min_publishers=2)
        with self.assertRaisesRegex(ValueError, "min_publishers"):
            fetch_diverse_recent(max_age_hours=48, min_publishers=0)


if __name__ == "__main__":
    unittest.main()
