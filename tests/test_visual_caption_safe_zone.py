import unittest

from PIL import Image, ImageDraw

from factory.image_generator import _caption_safe_zone, _detail_score


class VisualCaptionSafeZoneTests(unittest.TestCase):
    def test_lower_third_detail_is_feathered_without_changing_dimensions(self):
        image = Image.new("RGB", (320, 568), (80, 100, 120))
        draw = ImageDraw.Draw(image)
        start_y = round(image.height * 0.68)
        for y in range(start_y, image.height, 8):
            draw.line((0, y, image.width, y), fill=(245, 245, 245), width=3)
        for x in range(0, image.width, 8):
            draw.line((x, start_y, x, image.height), fill=(10, 10, 10), width=3)

        repaired, before, after = _caption_safe_zone(image)

        self.assertEqual(repaired.size, image.size)
        self.assertGreater(before, 4.0)
        self.assertLess(after, before * 0.94)
        upper_original = image.crop((0, 0, image.width, start_y - 2))
        upper_repaired = repaired.crop((0, 0, image.width, start_y - 2))
        self.assertEqual(list(upper_original.getdata()), list(upper_repaired.getdata()))

    def test_detail_score_detects_busy_region(self):
        quiet = Image.new("RGB", (100, 100), (20, 20, 20))
        busy = quiet.copy()
        draw = ImageDraw.Draw(busy)
        for index in range(0, 100, 5):
            draw.line((0, index, 99, index), fill=(240, 240, 240), width=2)
        self.assertGreater(_detail_score(busy), _detail_score(quiet))


if __name__ == "__main__":
    unittest.main()
