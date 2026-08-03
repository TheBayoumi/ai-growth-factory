from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

from factory.config import Settings
from factory.feeds import SourceItem
from factory.local_llm import LocalLLMError
from factory.models import Scene, VideoPackage
from factory.policy import Strategy
from factory.production_single_story_selection import (
    rewrite_prompt_for_single_authority,
    select_story_candidates,
)


def _source(title: str, url: str) -> SourceItem:
    return SourceItem(
        publisher="Official Publisher",
        title=title,
        url=url,
        summary=f"Official evidence for {title}",
        published_at=datetime.now(timezone.utc),
    )


class ProductionSingleStorySelectionTests(unittest.TestCase):
    def test_candidate_order_prefers_concrete_title_inside_near_position_pair(self) -> None:
        generic = _source("Building useful artificial intelligence", "https://example.com/generic")
        concrete = _source("EvoLib 2.0: Evolving agent memory", "https://example.com/evolib")
        later = _source("Another general update", "https://example.com/later")

        selected = select_story_candidates([generic, concrete, later], limit=3)

        self.assertEqual(selected[0].url, concrete.url)
        self.assertEqual({item.url for item in selected}, {generic.url, concrete.url, later.url})

    def test_prompt_requires_exactly_one_authoritative_url(self) -> None:
        prompt = (
            "Select one current AI development that can be responsibly explained using at least 1 DISTINCT supplied publishers. "
            "A second publisher may provide context rather than independent confirmation, but do not imply independent confirmation when it is not present.\n"
            "- source_urls: 2-5 UNIQUE URLs copied exactly from the supplied entries and spanning at least 1 distinct publishers\n"
            "- Choose source_urls from at least 1 different rows in PUBLISHER SOURCE OPTIONS.\n"
            "- Multiple URLs from one publisher still count as one publisher.\n"
            "If the sources cannot support one coherent package across that many publishers, return skip_reason rather than weakening attribution."
        )

        rewritten = rewrite_prompt_for_single_authority(prompt)

        self.assertIn("ONE supplied authoritative article", rewritten)
        self.assertIn("exactly 1 UNIQUE URL", rewritten)
        self.assertIn("Every factual claim and all six scenes", rewritten)
        self.assertNotIn("A second publisher", rewritten)
        self.assertNotIn("2-5 UNIQUE URLs", rewritten)

    def test_empty_candidate_set_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "received no articles"):
            select_story_candidates([])


if __name__ == "__main__":
    unittest.main()
