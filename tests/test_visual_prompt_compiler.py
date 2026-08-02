import unittest

from factory.visual_prompt_compiler import compile_image_prompt, compile_motion_prompt


class VisualPromptCompilerTests(unittest.TestCase):
    def test_sdxl_prompt_is_short_critical_first_and_text_resistant(self):
        director = (
            "A scientist holds a tablet while multiple dashboard screens show labeled charts. "
            "A large user interface explains the model architecture with readable numbers. "
            "The room has cinematic teal lighting and realistic materials. "
            "Keep the lower 32 percent quiet for separate animated captions. No text or logos."
        )
        compiled = compile_image_prompt(director, "watermark, text, logo, watermark")

        self.assertLessEqual(compiled.word_count, 44)
        self.assertLessEqual(len(compiled.compiled_prompt), 320)
        self.assertTrue(compiled.compiled_prompt.startswith("Text-free cinematic editorial image"))
        self.assertIn("Subject high in frame", compiled.compiled_prompt)
        self.assertIn("Dark empty lower third reserved for captions", compiled.compiled_prompt)
        self.assertTrue(compiled.compiled_prompt.endswith("realistic light"))
        self.assertNotIn("tablet", compiled.compiled_prompt.lower())
        self.assertNotIn("dashboard", compiled.compiled_prompt.lower())
        self.assertNotIn("user interface explains", compiled.compiled_prompt.lower())
        self.assertEqual(compiled.director_prompt, director)
        self.assertEqual(compiled.compiler_version, "visual-compiler-v4")
        negative_tokens = [part.strip() for part in compiled.negative_prompt.split(",")]
        self.assertEqual(len(negative_tokens), len(set(negative_tokens)))
        self.assertIn("screens", negative_tokens)
        self.assertIn("text", negative_tokens)

    def test_caption_safe_suffix_survives_an_extremely_long_director_prompt(self):
        director = " ".join(
            [
                "A detailed cinematic research environment with complex materials and lighting"
                for _ in range(20)
            ]
        )
        compiled = compile_image_prompt(director)

        self.assertLessEqual(compiled.word_count, 44)
        self.assertLessEqual(len(compiled.compiled_prompt), 320)
        self.assertIn("Dark empty lower third reserved for captions", compiled.compiled_prompt)
        self.assertTrue(compiled.compiled_prompt.endswith("realistic light"))

    def test_motion_prompt_does_not_reinject_long_image_description(self):
        motion = (
            "The central light sculpture rotates slowly while particles move along one path. "
            "The camera pushes forward and a dashboard appears with new labels."
        )
        compiled = compile_motion_prompt(motion)

        self.assertLessEqual(compiled.word_count, 48)
        self.assertTrue(compiled.compiled_motion_prompt.startswith("Locked camera"))
        self.assertIn("Preserve keyframe composition", compiled.compiled_motion_prompt)
        self.assertIn("No text", compiled.compiled_motion_prompt)
        self.assertNotIn("dashboard appears", compiled.compiled_motion_prompt.lower())
        self.assertNotIn("camera pushes", compiled.compiled_motion_prompt.lower())
        self.assertEqual(compiled.director_motion_prompt, motion)

    def test_small_invalid_word_budgets_are_rejected(self):
        with self.assertRaises(ValueError):
            compile_image_prompt("scene", word_budget=20)
        with self.assertRaises(ValueError):
            compile_motion_prompt("motion", word_budget=20)


if __name__ == "__main__":
    unittest.main()
