from __future__ import annotations

import re
import unittest

from factory.production_object_visuals import compile_object_only_image_prompt


class ProductionObjectVisualTests(unittest.TestCase):
    def test_all_scene_roles_compile_to_distinct_short_object_only_prompts(self) -> None:
        prompts = {
            role: compile_object_only_image_prompt(
                f"Factual visual: Orchard framework. composition grammar {grammar}. "
                "A researcher works on a computer screen with documents and labels."
            ).compiled_prompt
            for role, grammar in {
                "hook": "foreground consequence",
                "evidence": "grouped physical evidence",
                "mechanism": "directional mechanism",
                "comparison": "side-by-side comparison",
                "implication": "wide environmental implication",
                "cta": "human-scale decision",
            }.items()
        }

        self.assertEqual(len(set(prompts.values())), 6)
        forbidden = re.compile(
            r"\b(?:no|not|without|text|letter|number|symbol|caption|logo|watermark|"
            r"person|people|human|researcher|face|portrait|body|hand|phone|device|screen|"
            r"monitor|tablet|laptop|poster|book|document|sign|frame|panel|grid|collage|"
            r"interface|dashboard|computer)\b",
            re.IGNORECASE,
        )
        for prompt in prompts.values():
            self.assertLessEqual(len(prompt.split()), 36)
            self.assertLessEqual(len(prompt), 240)
            self.assertIsNone(forbidden.search(prompt), prompt)
            self.assertIn("empty dark lower third", prompt)
            self.assertIn("unmarked matte forms", prompt)

        comparison = prompts["comparison"]
        self.assertIn("One modular column", comparison)
        self.assertIn("rough dark blocks", comparison)
        self.assertIn("aligned glowing blocks", comparison)
        self.assertNotIn("stand in clear contrast", comparison)
        self.assertLessEqual(len(comparison.split()), 36)
        self.assertLessEqual(len(comparison), 240)

    def test_prohibited_director_vocabulary_never_reaches_positive_prompt(self) -> None:
        compiled = compile_object_only_image_prompt(
            "Factual visual: A researcher with a laptop, books, framed panels, labels, "
            "screens, portraits, and a collage. composition grammar foreground consequence."
        )

        positive = compiled.compiled_prompt.casefold()
        for term in (
            "researcher",
            "laptop",
            "book",
            "frame",
            "panel",
            "label",
            "screen",
            "portrait",
            "collage",
        ):
            self.assertNotIn(term, positive)
        self.assertIn("people", compiled.negative_prompt)
        self.assertIn("pseudo-text", compiled.negative_prompt)

    def test_compiler_uses_positive_descriptions_instead_of_negation(self) -> None:
        compiled = compile_object_only_image_prompt(
            "Factual visual: Strong performance. composition grammar directional mechanism."
        )

        self.assertNotRegex(compiled.compiled_prompt.casefold(), r"\b(?:no|not|without)\b")
        self.assertIn("Precision components channel light", compiled.compiled_prompt)
        self.assertEqual(compiled.compiler_version, "visual-compiler-v8-coherent-clip-budget")

    def test_comparison_negative_prompt_blocks_split_layouts(self) -> None:
        compiled = compile_object_only_image_prompt(
            "Factual visual: Compare configurations. composition grammar side-by-side comparison."
        )

        self.assertIn("split scene", compiled.negative_prompt)
        self.assertIn("diptych", compiled.negative_prompt)
        self.assertIn("multiple rooms", compiled.negative_prompt)


if __name__ == "__main__":
    unittest.main()
