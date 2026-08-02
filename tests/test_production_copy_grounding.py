import unittest
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
from factory.production_content import _ground_generic_copy, _validate_publishable_content


class ProductionCopyGroundingTests(unittest.TestCase):
    def test_generic_slogan_is_replaced_with_validated_source_subject(self):
        source = SourceItem(
            publisher="OpenAI",
            title="Realtime API Adds Reliable Tool Calling",
            url="https://example.com/realtime-tools",
            summary="The Realtime API now supports more reliable tool calls.",
            published_at=datetime.now(timezone.utc),
        )
        narration = (
            "OpenAI released Realtime API updates for reliable tool calling. "
            "The change gives voice agents a clearer way to invoke approved tools. "
            "Developers can connect live conversations to structured actions with fewer ambiguous handoffs. "
            "That matters because spoken requests often combine context, corrections, and follow-up instructions. "
            "The updated interface keeps tool results inside the conversation while preserving explicit control. "
            "Teams can now test failures, retries, and confirmations against a concrete API contract. "
            "Evaluators also gain repeatable checkpoints for permission handling, malformed arguments, and delayed tool responses before a customer ever sees them. "
            "This is not just shaping the future; it creates a measurable workflow for production voice agents. "
            "The practical question is whether latency and reliability hold. "
            "Early deployments should track successful calls, correction rates, and user confirmations instead of relying on polished demos alone. "
            "Follow for the next verified deployment result and the exact numbers that matter."
        )
        package = VideoPackage(
            topic="Realtime API tool calling",
            narration=narration,
            title="OpenAI Realtime API Is Shaping the Future",
            description="A focused explanation. https://example.com/realtime-tools",
            tags=["OpenAI", "Realtime API", "voice agents", "tool calling"],
            thumbnail_text="Realtime Tool Calling",
            top_comment="Where would you use this?",
            scenes=[
                Scene(f"Scene {index + 1}", f"Distinct evidence point {index + 1}.", "diagram", 0)
                for index in range(6)
            ],
            source_urls=[source.url],
            source_publishers=[source.publisher],
        )

        grounded = _ground_generic_copy(package, [source])

        self.assertNotIn("shaping the future", grounded.title.casefold())
        self.assertNotIn("shaping the future", grounded.narration.casefold())
        self.assertIn("Realtime API", grounded.title)
        _validate_publishable_content(grounded, [source])


if __name__ == "__main__":
    unittest.main()
