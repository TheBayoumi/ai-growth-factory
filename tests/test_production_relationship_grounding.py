import unittest
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.models import Scene, VideoPackage
from factory.production_content import _validate_publishable_content
from factory.production_relationship_grounding import ground_unsupported_relationships


class ProductionRelationshipGroundingTests(unittest.TestCase):
    def test_invented_collaboration_becomes_independent_source_context(self):
        now = datetime.now(timezone.utc)
        sources = [
            SourceItem(
                "OpenAI",
                "Realtime API Adds Reliable Tool Calling",
                "https://example.com/openai",
                (
                    "The Realtime API now supports more reliable tool calls with structured "
                    "arguments, explicit confirmation, retry handling, permission checks, and "
                    "observable results inside live voice conversations. Developers can evaluate "
                    "latency, malformed requests, delayed responses, correction rates, and failure "
                    "recovery against a concrete production interface."
                ),
                now,
            ),
            SourceItem(
                "Microsoft Research",
                "EvoLib Evaluates Automated Library Improvement",
                "https://example.com/microsoft",
                (
                    "EvoLib studies automated improvement of software libraries through controlled "
                    "tasks, repeatable evaluations, measured checkpoints, and explicit regression "
                    "tracking. The research separates proposed edits from verified outcomes and "
                    "records failures, retries, and reproducibility evidence before claiming that "
                    "an automated change improves a real workflow."
                ),
                now,
            ),
        ]
        narration = (
            "The collaboration between OpenAI and Microsoft Research changes how teams deploy artificial intelligence systems. "
            "OpenAI documented reliable tool calling for realtime voice agents and described a clearer contract for approved actions. "
            "Microsoft Research separately described EvoLib, an evaluation framework for improving software libraries through measured iterations. "
            "These reports address different technical questions, so each claim must remain tied to its own primary source. "
            "Before adoption, engineers should reproduce the relevant behavior on a controlled task and record latency, failures, corrections, and repeatability. "
            "The result should be compared with the current workflow instead of being accepted from a polished announcement. "
            "This evidence-first process keeps the practical decision grounded in observed behavior. "
            "Follow for the next verified production result and the exact measurements that matter."
        )
        package = VideoPackage(
            topic="AI production evaluation",
            narration=narration,
            title="The Collaboration Between OpenAI and Microsoft Research",
            description="Separate source context. https://example.com/openai https://example.com/microsoft",
            tags=["OpenAI", "Microsoft Research", "AI engineering"],
            thumbnail_text="AI PRODUCTION TEST",
            top_comment="What would you measure first?",
            scenes=[
                Scene(
                    f"Scene {index + 1}",
                    f"Distinct source-grounded evidence point {index + 1}.",
                    "Two independent evidence streams remain visually separate.",
                    index % 2,
                )
                for index in range(6)
            ],
            source_urls=[source.url for source in sources],
            source_publishers=[source.publisher for source in sources],
        )

        repaired = ground_unsupported_relationships(package, sources)
        lowered = f"{repaired.title} {repaired.narration}".casefold()

        self.assertNotIn("collaboration between", lowered)
        self.assertIn("separate primary-source context", repaired.narration)
        self.assertIn("evaluated independently", repaired.narration)
        self.assertIn("OpenAI", repaired.title)
        self.assertGreaterEqual(len(repaired.narration.split()), 130)
        self.assertLessEqual(len(repaired.narration.split()), 140)
        _validate_publishable_content(repaired, sources)

    def test_explicit_relationship_in_source_evidence_is_preserved(self):
        now = datetime.now(timezone.utc)
        sources = [
            SourceItem(
                "Publisher A",
                "Publisher A Announces Collaboration With Publisher B",
                "https://example.com/a",
                "The companies announced a formal collaboration.",
                now,
            ),
            SourceItem(
                "Publisher B",
                "Publisher B Confirms Joint Project",
                "https://example.com/b",
                "The joint project was confirmed.",
                now,
            ),
        ]
        package = VideoPackage(
            topic="Formal project",
            narration="The collaboration between Publisher A and Publisher B was announced in their primary sources.",
            title="Publisher A and Publisher B Collaboration",
            description="Description",
            tags=["project"],
            thumbnail_text="FORMAL PROJECT",
            top_comment="Question",
            scenes=[Scene("Head", "Body", "Visual", index % 2) for index in range(6)],
            source_urls=[source.url for source in sources],
            source_publishers=[source.publisher for source in sources],
        )

        self.assertIs(ground_unsupported_relationships(package, sources), package)


if __name__ == "__main__":
    unittest.main()
