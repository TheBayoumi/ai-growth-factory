from __future__ import annotations

import unittest
from pathlib import Path

from factory.production_visual_storyboard_v30 import (
    _prominent_text_evidence,
    compile_storyboard_prompt_v30,
    scene_for_attempt_v30,
)
from factory.visual_prompt import SceneVisualPrompt
from factory.visual_storyboard_v30 import (
    classify_claim,
    storyboard_categories,
    storyboard_for,
)


NVIDIA_CLAIMS = (
    "NVIDIA is joining the NSF's State and Regional AI Hubs program to boost AI research and education.",
    "The initiative aims to provide advanced computing resources and expertise for AI studies.",
    "This partnership supports the Genesis Mission by expanding access to AI tools and knowledge.",
    "NVIDIA's involvement helps create a stronger foundation for AI innovation across the US.",
    "The program focuses on making AI research more accessible and effective for educators and students.",
    "The program will support state and multistate groups by providing computing, data, software, and expertise needed for AI-enabled research and education.",
    "Before adoption, read the linked source and test the claim on a controlled task.",
)


def director_prompt(claim: str, direction: str = "Computer screen with AI software") -> str:
    return (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        f"{claim}. Supporting source-grounded visual direction: {direction}. "
        "Shot treatment: wide contextual view showing the surrounding workflow."
    )


class VisualStoryboardV30Tests(unittest.TestCase):
    def test_claim_classifier_covers_customizable_story_categories(self) -> None:
        expected = (
            "partnership_hub",
            "compute_resources",
            "access_knowledge",
            "regional_network",
            "education",
            "expertise_support",
            "controlled_test",
        )
        observed = tuple(classify_claim(claim) for claim in NVIDIA_CLAIMS)
        self.assertEqual(observed, expected)
        self.assertTrue(set(expected).issubset(set(storyboard_categories())))

    def test_repeated_claims_receive_distinct_environments_and_actions(self) -> None:
        prompt = director_prompt(NVIDIA_CLAIMS[1])
        frames = [storyboard_for(prompt + f". V30 STORYBOARD: shot-{index}", index) for index in range(3)]
        self.assertEqual(len({frame.environment for frame in frames}), 3)
        self.assertEqual(len({frame.action for frame in frames}), 3)
        self.assertEqual(len({frame.camera for frame in frames}), 3)

    def test_final_prompt_ignores_literal_source_direction(self) -> None:
        prompt = director_prompt(
            NVIDIA_CLAIMS[3],
            "US map with AI icons and NVIDIA logo",
        ) + ". V30 STORYBOARD: shot-8"
        compiled = compile_storyboard_prompt_v30(prompt)
        lowered = compiled.compiled_prompt.casefold()
        for forbidden in ("us map", "ai icons", "nvidia", "logo", "screen", "monitor", "display"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("one continuous photorealistic", lowered)
        self.assertIn("research", lowered)
        self.assertLessEqual(compiled.word_count, 68)

    def test_storyboard_prompts_create_cross_environment_variety(self) -> None:
        prompts = [
            compile_storyboard_prompt_v30(
                director_prompt(claim) + f". V30 STORYBOARD: shot-{index}"
            ).compiled_prompt
            for index, claim in enumerate(NVIDIA_CLAIMS)
        ]
        combined = " ".join(prompts).casefold()
        self.assertIn("data-center aisle", combined)
        self.assertIn("community technology classroom", combined)
        self.assertIn("logistics bay", combined)
        self.assertIn("controlled robotics test cell", combined)
        self.assertNotIn("old-fashioned laboratory", combined)

    def test_retry_keeps_story_identity_and_replaces_source_direction(self) -> None:
        source = SceneVisualPrompt(
            scene_index=8,
            source_index=0,
            role="implication",
            generation_mode="image",
            image_prompt=director_prompt(NVIDIA_CLAIMS[3], "US map with AI icons"),
            motion_prompt="controlled motion",
            negative_prompt="legacy",
            continuity_anchor="anchor",
            caption_safe_zone="lower",
            seed=19,
            duration_seconds=3.0,
        )
        first = scene_for_attempt_v30(source, scene_index=8, attempt=1)
        retry = scene_for_attempt_v30(
            source,
            scene_index=8,
            attempt=2,
            repair="The image contains readable text on a computer screen",
        )
        self.assertIn("regional_network", first.image_prompt)
        self.assertIn("regional_network", retry.image_prompt)
        self.assertEqual(first.image_prompt.count("V30 STORYBOARD:"), 1)
        self.assertEqual(retry.image_prompt.count("V30 STORYBOARD:"), 1)
        self.assertEqual(retry.image_prompt.count("V30 REPAIR:"), 1)
        self.assertIn("blank unmarked hardware", retry.image_prompt)
        self.assertNotEqual(first.seed, retry.seed)

    def test_text_rejection_requires_prominent_located_high_confidence_evidence(self) -> None:
        evidence = [
            {
                "kind": "readable",
                "content": "AI LAB",
                "bbox": [0.1, 0.1, 0.4, 0.3],
                "prominence": "prominent",
                "confidence": 0.96,
            },
            {
                "kind": "readable",
                "content": "7",
                "bbox": [0.5, 0.5, 0.51, 0.51],
                "prominence": "incidental",
                "confidence": 0.99,
            },
        ]
        accepted = _prominent_text_evidence(evidence)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["content"], "AI LAB")
        self.assertEqual(_prominent_text_evidence([]), [])
        self.assertEqual(
            _prominent_text_evidence(
                [
                    {
                        "kind": "readable",
                        "content": "AI LAB",
                        "bbox": [0.1, 0.1, 0.4, 0.3],
                        "prominence": "prominent",
                        "confidence": 0.60,
                    }
                ]
            ),
            [],
        )

    def test_reviewer_does_not_inherit_literal_map_or_page_authority(self) -> None:
        source = Path("factory/production_visual_storyboard_v30.py").read_text(encoding="utf-8")
        self.assertIn("The ONLY required storyboard target", source)
        self.assertIn("Never require a US map", source)
        self.assertNotIn("super().review", source)
        self.assertIn("text_evidence", source)
        self.assertIn("confidence >= 0.85", source)

    def test_runtime_installs_v30_last(self) -> None:
        source = Path("factory/production_runtime.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_production_visual_prompt_cleanup_v29()"),
            source.index("install_production_visual_storyboard_v30()"),
        )
        self.assertLess(
            source.index("install_production_visual_storyboard_v30()"),
            source.index("image_generator.generate_keyframes = visual_pipeline.generate_keyframes"),
        )


if __name__ == "__main__":
    unittest.main()
