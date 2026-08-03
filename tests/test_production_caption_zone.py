from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from factory.production_caption_zone import production_caption_safe_zone


class ProductionCaptionZoneTests(unittest.TestCase):
    @staticmethod
    def _busy_image() -> Image.Image:
        image = Image.new("RGB", (704, 1280), (40, 70, 110))
        draw = ImageDraw.Draw(image)
        for y in range(0, 1280, 16):
            for x in range(0, 704, 16):
                value = 230 if (x // 16 + y // 16) % 2 else 20
                draw.rectangle((x, y, x + 15, y + 15), fill=(value, 80, 255 - value))
        return image

    def test_upper_sixty_percent_is_preserved_exactly(self) -> None:
        source = self._busy_image()
        repaired, _before, _after = production_caption_safe_zone(source)
        boundary = round(source.height * 0.60)

        self.assertEqual(
            source.crop((0, 0, source.width, boundary)).tobytes(),
            repaired.crop((0, 0, repaired.width, boundary)).tobytes(),
        )

    def test_entire_lower_thirty_two_percent_is_subject_free(self) -> None:
        source = self._busy_image()
        repaired, before, after = production_caption_safe_zone(source)
        matte_start = round(source.height * 0.68)
        lower = repaired.crop((0, matte_start, source.width, source.height))

        self.assertGreater(before, 1.0)
        self.assertLessEqual(after, 0.05)
        self.assertEqual(lower.getbbox(), (0, 0, lower.width, lower.height))
        self.assertEqual(set(lower.getdata()), {(5, 7, 12)})

    def test_transition_is_gradual_not_an_abrupt_overlay(self) -> None:
        source = Image.new("RGB", (100, 100), (100, 120, 140))
        repaired, _before, _after = production_caption_safe_zone(source)

        unchanged = repaired.getpixel((50, 59))
        transition_middle = repaired.getpixel((50, 64))
        matte = repaired.getpixel((50, 68))

        self.assertEqual(unchanged, (100, 120, 140))
        self.assertNotEqual(transition_middle, unchanged)
        self.assertNotEqual(transition_middle, matte)
        self.assertEqual(matte, (5, 7, 12))


if __name__ == "__main__":
    unittest.main()
