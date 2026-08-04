from __future__ import annotations

import unittest
from types import SimpleNamespace

from PIL import Image

from factory.production_editorial_v28 import (
    _caption_chunks,
    _plain_caption,
    compile_semantic_image_prompt,
    full_frame_caption_zone,
)


class ProductionEditorialV28Tests(unittest.TestCase):
    def test_compiler_preserves_concrete_semantics(self) -> None:
        result = compile_semantic_image_prompt(
            "Researchers collaborate in a shared workspace using unbranded computers and tools"
        )
        lowered = result.compiled_prompt.casefold()
        self.assertIn("researchers", lowered)
        self.assertIn("shared workspace", lowered)
        self.assertNotIn("modular bridge", lowered)
        self.assertNotIn("geometric modules converge", lowered)
        self.assertEqual(result.compiler_version, "visual-compiler-v28-semantic-preservation")

    def test_caption_zone_does_not_destroy_pixels(self) -> None:
        image = Image.new("RGB", (100, 200), (22, 44, 66))
        image.putpixel((50, 190), (200, 100, 50))
        result, before, after = full_frame_caption_zone(image)
        self.assertEqual(result.tobytes(), image.tobytes())
        self.assertEqual(before, after)
        self.assertEqual(result.getpixel((50, 190)), (200, 100, 50))

    def test_caption_chunks_avoid_one_word_churn(self) -> None:
        chunks = _caption_chunks(
            "This framework helps researchers build shared infrastructure, test ideas, and collaborate efficiently."
        )
        self.assertTrue(all(len(chunk.split()) >= 2 for chunk in chunks))
        self.assertTrue(all(len(chunk.split()) <= 5 for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 34 for chunk in chunks))

    def test_caption_text_is_phrase_level_not_karaoke(self) -> None:
        cue = SimpleNamespace(text="shared research infrastructure")
        rendered = _plain_caption(cue, lambda value: value)
        self.assertEqual(rendered, cue.text)
        self.assertNotIn("\\kf", rendered)


if __name__ == "__main__":
    unittest.main()
