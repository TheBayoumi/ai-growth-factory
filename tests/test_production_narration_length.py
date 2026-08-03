import re
import unittest

from factory.production_narration_length import stabilize_production_narration


class ProductionNarrationLengthTests(unittest.TestCase):
    @staticmethod
    def _words(count: int) -> str:
        return " ".join(f"word{index}" for index in range(count - 8)) + " Follow for the next verified production result today."

    def test_ninety_two_word_draft_reaches_production_range(self):
        raw = {"narration": self._words(92), "title": "Title"}

        corrected = stabilize_production_narration(raw)
        narration = corrected["narration"]
        count = len(narration.split())

        self.assertGreaterEqual(count, 135)
        self.assertLessEqual(count, 155)
        self.assertTrue(narration.endswith("Follow for the next verified production result today."))
        self.assertIn("Before adoption, read the linked source", narration)
        self.assertIn("Track latency, failure rate", narration)
        self.assertEqual(raw["narration"], self._words(92))

    def test_very_short_draft_remains_fail_closed(self):
        raw = {"narration": self._words(79)}

        self.assertIs(stabilize_production_narration(raw), raw)

    def test_already_valid_draft_is_unchanged(self):
        raw = {"narration": self._words(140)}

        self.assertIs(stabilize_production_narration(raw), raw)

    def test_evidence_guidance_is_not_duplicated(self):
        narration = self._words(105) + (
            " Before adoption, read the linked source and test the claim on a controlled task."
        )
        raw = {"narration": narration}

        corrected = stabilize_production_narration(raw)

        self.assertEqual(
            corrected["narration"].casefold().count("before adoption"),
            1,
        )
        self.assertGreaterEqual(len(corrected["narration"].split()), 135)
        self.assertLessEqual(len(corrected["narration"].split()), 155)

    def test_125_word_draft_uses_one_short_spoken_sentence(self):
        raw = {"narration": self._words(125)}

        corrected = stabilize_production_narration(raw)
        added = "Before adoption, read the linked source and test the claim on a controlled task."

        self.assertIn(added, corrected["narration"])
        self.assertEqual(len(added.split()), 14)
        sentences = [
            value.strip()
            for value in re.split(r"(?<=[.!?])\s+", corrected["narration"])
            if value.strip()
        ]
        self.assertIn(added, sentences)
        self.assertGreaterEqual(len(corrected["narration"].split()), 135)
        self.assertLessEqual(len(corrected["narration"].split()), 155)


if __name__ == "__main__":
    unittest.main()
