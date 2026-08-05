from __future__ import annotations

import unittest
from pathlib import Path

from factory.production_visual_subject_authority_v31 import (
    _compact_negative,
    _words,
    compile_subject_first_prompt_v31,
    scene_for_attempt_v31,
    validate_clip_windows,
)
from factory.visual_prompt import SceneVisualPrompt
from factory.visual_storyboard_v30 import classify_claim, storyboard_for


ORCHARD_CLAIMS = (
    "Microsoft Research released an open-source framework to help researchers train and evaluate AI agents across various tasks.",
    "The tool enables reuse of infrastructure so smaller models can achieve strong performance.",
    "The framework supports scalable agentic AI and complex AI systems.",
    "Shared infrastructure reduces the complexity of developing and testing AI agents.",
    "This advancement could lead to more efficient and effective AI research.",
    "Researchers can focus on innovation rather than infrastructure.",
    "The framework is available for public use, encouraging collaboration and innovation.",
    "Sources: Microsoft Research blog post.",
    "Before adoption, read the linked source and test the claim on a controlled task.",
)


def director_prompt(claim: str, scene_index: int) -> str:
    return (
        "Factual technology documentary shot synchronized to this exact spoken sentence: "
        f"{claim}. Supporting source-grounded visual direction: a legacy screen request. "
        "Shot treatment: wide contextual view showing the surrounding workflow. "
        f"V30 STORYBOARD: shot-{scene_index}"
    )


def contains_sequence(haystack: list[str], needle: list[str]) -> int:
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


class _FakeTokenizer:
    model_max_length = 77

    def __call__(self, value: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(value.split()) + 2))}


class _FakePipeline:
    tokenizer = _FakeTokenizer()
    tokenizer_2 = _FakeTokenizer()


class VisualSubjectAuthorityV31Tests(unittest.TestCase):
    def test_orchard_claims_route_to_multiple_concrete_storyboards(self) -> None:
        observed = tuple(classify_claim(claim) for claim in ORCHARD_CLAIMS)
        self.assertEqual(
            observed,
            (
                "controlled_test",
                "compute_resources",
                "regional_network",
                "compute_resources",
                "expertise_support",
                "expertise_support",
                "partnership_hub",
                "controlled_test",
                "controlled_test",
            ),
        )
        self.assertGreaterEqual(len(set(observed)), 5)

    def test_failed_canary_subjects_and_actions_are_complete_before_environment(self) -> None:
        cases = (
            (ORCHARD_CLAIMS[2], 5),
            (ORCHARD_CLAIMS[4], 11),
            (ORCHARD_CLAIMS[6], 14),
        )
        for claim, scene_index in cases:
            prompt = director_prompt(claim, scene_index)
            frame = storyboard_for(prompt, scene_index)
            compiled = compile_subject_first_prompt_v31(prompt)
            prompt_words = [word.casefold() for word in _words(compiled.compiled_prompt)]
            subject_words = [word.casefold() for word in _words(frame.subject)]
            action_words = [word.casefold() for word in _words(frame.action)]
            environment_words = [word.casefold() for word in _words(frame.environment)]
            subject_at = contains_sequence(prompt_words, subject_words)
            action_at = contains_sequence(prompt_words, action_words)
            environment_at = contains_sequence(prompt_words, environment_words)
            self.assertGreaterEqual(subject_at, 0)
            self.assertGreater(action_at, subject_at)
            self.assertGreater(environment_at, action_at)
            self.assertLessEqual(compiled.word_count, 52)
            self.assertEqual(compiled.compiler_version, "visual-compiler-v31-subject-first-clip-safe")

    def test_retry_feedback_changes_the_executable_prompt_instead_of_only_the_seed(self) -> None:
        source = SceneVisualPrompt(
            scene_index=11,
            source_index=0,
            role="evidence",
            generation_mode="image",
            image_prompt=director_prompt(ORCHARD_CLAIMS[4], 11),
            motion_prompt="controlled motion",
            negative_prompt="legacy",
            continuity_anchor="anchor",
            caption_safe_zone="lower",
            seed=41,
            duration_seconds=3.0,
        )
        first = scene_for_attempt_v31(source, scene_index=11, attempt=1)
        retry = scene_for_attempt_v31(
            source,
            scene_index=11,
            attempt=2,
            repair="The server room has no people or equipment installation",
        )
        first_prompt = compile_subject_first_prompt_v31(first.image_prompt)
        retry_prompt = compile_subject_first_prompt_v31(retry.image_prompt)
        self.assertNotEqual(first_prompt.compiled_prompt, retry_prompt.compiled_prompt)
        self.assertIn("one primary subject", retry.image_prompt)
        self.assertIn("one clear physical action", retry.image_prompt)
        self.assertIn("one primary subject", retry_prompt.compiled_prompt)
        self.assertNotIn("Every named adult", retry.image_prompt)
        self.assertEqual(retry.image_prompt.count("V30 STORYBOARD:"), 1)
        self.assertEqual(retry.image_prompt.count("V31 REPAIR:"), 1)
        self.assertNotEqual(first.seed, retry.seed)

    def test_positive_and_negative_prompts_fit_the_runtime_clip_contract(self) -> None:
        compiled = compile_subject_first_prompt_v31(director_prompt(ORCHARD_CLAIMS[2], 5))
        validate_clip_windows(_FakePipeline(), compiled.compiled_prompt, compiled.negative_prompt)
        self.assertLessEqual(len(_words(compiled.negative_prompt)), 52)
        for required in (
            "readable text",
            "pseudo-text",
            "printed label",
            "vacant scene",
            "malformed anatomy",
            "warped equipment",
        ):
            self.assertIn(required, _compact_negative())
        self.assertNotIn("absent people", _compact_negative())

    def test_clip_overflow_fails_before_sdxl_inference(self) -> None:
        with self.assertRaisesRegex(Exception, "SDXL CLIP window overflow"):
            validate_clip_windows(
                _FakePipeline(),
                " ".join(["subject"] * 80),
                _compact_negative(),
            )

    def test_runtime_installs_v31_before_v34_and_pipeline_binding(self) -> None:
        source = Path("factory/production_runtime.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_production_visual_storyboard_v30()"),
            source.index("install_production_visual_subject_authority_v31()"),
        )
        self.assertLess(
            source.index("install_production_visual_subject_authority_v31()"),
            source.index("install_production_visual_atomic_storyboard_v34()"),
        )
        self.assertLess(
            source.index("install_production_visual_atomic_storyboard_v34()"),
            source.index("image_generator.generate_keyframes = visual_pipeline.generate_keyframes"),
        )


if __name__ == "__main__":
    unittest.main()
