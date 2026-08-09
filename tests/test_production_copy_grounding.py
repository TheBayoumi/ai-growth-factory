import unittest
from dataclasses import replace
from datetime import datetime, timezone

from factory.feeds import SourceItem
from factory.local_llm import LocalLLMError
from factory.models import Scene, VideoPackage
from factory.production_content import (
    _ground_generic_copy,
    _validate_author_metadata_grounding,
    _validate_evidence_specificity,
    _validate_publishable_content,
    _validate_release_authority,
    _validate_source_relationships,
)


class ProductionCopyGroundingTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.openai = SourceItem(
            publisher="OpenAI",
            title="Realtime API Adds Reliable Tool Calling",
            url="https://example.com/realtime-tools",
            summary=(
                "The Realtime API now supports more reliable tool calls, explicit argument "
                "validation, retry handling, confirmation checkpoints, and structured results "
                "inside live voice conversations. The update gives developers concrete controls "
                "for permissions, malformed requests, delayed responses, and observable failure "
                "handling in production voice-agent workflows."
            ),
            published_at=now,
        )
        self.microsoft = SourceItem(
            publisher="Microsoft Research",
            title="EvoLib Evaluates Automated Library Improvement",
            url="https://example.com/evolib",
            summary=(
                "EvoLib studies automated improvement of software libraries with repeatable "
                "evaluation tasks, controlled comparisons, and measured checkpoints. The research "
                "separates proposed changes from verified outcomes and records failures, retries, "
                "and regressions so teams can judge whether automated library edits improve a "
                "real software workflow."
            ),
            published_at=now,
        )

    @staticmethod
    def _narration(slogan: str = "shaping the future") -> str:
        return (
            "OpenAI released Realtime API updates for reliable tool calling. "
            "The change gives voice agents a clearer way to invoke approved tools. "
            "Developers can connect live conversations to structured actions with fewer ambiguous handoffs. "
            "That matters because spoken requests often combine context, corrections, and follow-up instructions. "
            "The updated interface keeps tool results inside the conversation while preserving explicit control. "
            "Teams can now test failures, retries, and confirmations against a concrete API contract. "
            "Evaluators gain checkpoints for permissions, malformed arguments, and delayed responses. "
            f"This is not just {slogan}; it creates a measurable workflow for production voice agents. "
            "The practical question is whether latency and reliability hold. "
            "Early deployments should track successful calls, correction rates, and user confirmations instead of relying on polished demos alone. "
            "Follow for the next verified deployment result and the exact numbers that matter."
        )

    def _package(
        self,
        *,
        narration: str,
        sources: list[SourceItem] | None = None,
    ) -> VideoPackage:
        selected = sources or [self.openai]
        return VideoPackage(
            topic="Realtime API tool calling",
            narration=narration,
            title="OpenAI Realtime API Is Shaping the Future",
            description="A focused explanation. " + " ".join(source.url for source in selected),
            tags=["OpenAI", "Realtime API", "voice agents", "tool calling"],
            thumbnail_text="Realtime Tool Calling",
            top_comment="Where would you use this?",
            scenes=[
                Scene(
                    f"Scene {index + 1}",
                    f"Distinct evidence point {index + 1}.",
                    "diagram",
                    index % len(selected),
                )
                for index in range(6)
            ],
            source_urls=[source.url for source in selected],
            source_publishers=[source.publisher for source in selected],
        )

    def test_generic_slogan_is_replaced_with_validated_source_subject(self):
        package = self._package(narration=self._narration())

        grounded = _ground_generic_copy(package, [self.openai])

        self.assertNotIn("shaping the future", grounded.title.casefold())
        self.assertNotIn("shaping the future", grounded.narration.casefold())
        self.assertIn("Realtime API", grounded.title)
        _validate_publishable_content(grounded, [self.openai])

    def test_future_of_work_replacement_cannot_corrupt_larger_words(self):
        package = self._package(
            narration=self._narration("reshaping the future of work")
        )

        grounded = _ground_generic_copy(package, [self.openai])
        lowered = grounded.narration.casefold()

        self.assertIn("changing how people work", lowered)
        self.assertNotIn("rechanging", lowered)
        self.assertNotIn("is used of", lowered)
        self.assertNotIn("shaping the future", lowered)
        _validate_publishable_content(grounded, [self.openai])

    def test_unsupported_cross_source_collaboration_is_rejected(self):
        narration = self._narration("changing current practice").replace(
            "OpenAI released Realtime API updates for reliable tool calling.",
            "OpenAI and Microsoft Research worked together on Realtime API updates.",
        )
        package = self._package(
            narration=narration,
            sources=[self.openai, self.microsoft],
        )
        grounded = _ground_generic_copy(package, [self.openai, self.microsoft])

        with self.assertRaisesRegex(
            LocalLLMError,
            "unsupported cross-source relationship",
        ):
            _validate_publishable_content(grounded, [self.openai, self.microsoft])

    def test_scene_copy_cannot_hide_unsupported_partnership(self):
        package = self._package(
            narration=self._narration("changing current practice"),
            sources=[self.openai, self.microsoft],
        )
        package = replace(
            package,
            scenes=[
                replace(scene, body="The project is a result of strategic partnerships and technical expertise.")
                if index == 3 else scene
                for index, scene in enumerate(package.scenes)
            ],
        )
        with self.assertRaisesRegex(LocalLLMError, "unsupported cross-source relationship"):
            _validate_source_relationships(package, [self.openai, self.microsoft])

    def test_personal_article_author_cannot_be_promoted_to_project_lead(self):
        source = SourceItem(
            publisher="NVIDIA",
            author="Rev Lebaredian",
            title="Firebird Launches CIS Region's Largest AI Factory in Armenia",
            url="https://example.com/firebird",
            summary=(
                "Firebird launched a large AI factory in Armenia using NVIDIA accelerated computing "
                "and Dell Technologies high-performance infrastructure for AI workloads and research. "
                "Government and industry leaders attended the launch event and discussed regional capacity."
            ),
            published_at=datetime.now(timezone.utc),
        )
        package = replace(
            self._package(narration=self._narration("changing current practice")),
            title="Firebird Launches Armenia AI Factory",
            narration=self._narration("changing current practice").replace(
                "OpenAI released Realtime API updates for reliable tool calling.",
                "Firebird launched a large AI factory in Armenia. The project is led by Rev Lebaredian.",
            ),
            source_urls=[source.url],
            source_publishers=[source.publisher],
        )
        with self.assertRaisesRegex(LocalLLMError, "author/byline metadata"):
            _validate_author_metadata_grounding(package, [source])

    def test_personal_author_is_allowed_when_factual_evidence_mentions_person(self):
        source = SourceItem(
            publisher="Example News",
            author="Jane Smith",
            title="Project Atlas Names Jane Smith as Technical Lead",
            url="https://example.com/atlas",
            summary=(
                "Jane Smith is the technical lead for Project Atlas and oversees the engineering rollout. "
                "The project uses controlled validation milestones and documented infrastructure tests."
            ),
            published_at=datetime.now(timezone.utc),
        )
        package = replace(
            self._package(narration=self._narration("changing current practice")),
            title="Project Atlas Names Jane Smith Lead",
            narration=self._narration("changing current practice").replace(
                "OpenAI released Realtime API updates for reliable tool calling.",
                "Project Atlas named Jane Smith as its technical lead for the engineering rollout.",
            ),
            source_urls=[source.url],
            source_publishers=[source.publisher],
        )
        _validate_author_metadata_grounding(package, [source])

    def test_hosting_platform_cannot_claim_liquid_ai_release(self):
        source = SourceItem(
            publisher="Hugging Face",
            title="Deploy local agents everywhere with LFM2.5-2.6B",
            url="https://huggingface.co/blog/LiquidAI/lfm2-5-2-6b",
            summary="Liquid AI released LFM2.5-2.6B for on-device agents.",
            published_at=datetime.now(timezone.utc),
        )
        package = replace(
            self._package(narration=self._narration("changing current practice")),
            title="Hugging Face Releases LFM2.5-2.6B Model",
            narration=self._narration("changing current practice").replace(
                "OpenAI released Realtime API updates for reliable tool calling.",
                "Hugging Face has launched LFM2.5-2.6B for local deployment.",
            ),
            source_urls=[source.url],
            source_publishers=[source.publisher],
        )

        with self.assertRaisesRegex(LocalLLMError, "source authority is 'Liquid AI'"):
            _validate_release_authority(package, [source])

    def test_measured_source_cannot_be_reduced_to_generic_trend_copy(self):
        source = SourceItem(
            publisher="Hugging Face",
            title="Deploy local agents everywhere with LFM2.5-2.6B",
            url="https://huggingface.co/blog/LiquidAI/lfm2-5-2-6b",
            summary=(
                "The model supports a 128K context window, uses under 2.5GB of memory, "
                "and reaches 220 tokens per second on an M5 Max. Liquid AI designed it "
                "for native tool calling, multi-step agent workflows, private on-device "
                "execution, and broad support across CPU and GPU inference runtimes."
            ),
            published_at=datetime.now(timezone.utc),
        )
        package = replace(
            self._package(narration=self._narration("changing current practice")),
            title="Liquid AI Releases LFM2.5-2.6B",
            narration=self._narration("changing current practice").replace(
                "OpenAI released Realtime API updates for reliable tool calling.",
                "Liquid AI released LFM2.5-2.6B for efficient local deployment.",
            ),
            source_urls=[source.url],
            source_publishers=[source.publisher],
        )

        with self.assertRaisesRegex(LocalLLMError, "measured source evidence"):
            _validate_evidence_specificity(package, [source])

    def test_empty_selected_summary_is_rejected_before_tts(self):
        source = replace(self.openai, summary="")
        package = self._package(
            narration=self._narration("changing current practice"),
            sources=[source],
        )
        with self.assertRaisesRegex(LocalLLMError, "evidence is too thin"):
            _validate_evidence_specificity(package, [source])


if __name__ == "__main__":
    unittest.main()
