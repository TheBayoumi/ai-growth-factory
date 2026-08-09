from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from factory.production_voice_orphan_recovery_v50 import (
    copy_voice_failure_diagnostics_v50,
    repair_short_voice_orphans_v50,
    split_narration_for_voice_v50,
)


HSP_NARRATION = (
    "HSP GRUPPE, a European tax advisory network, has integrated ChatGPT Enterprise to "
    "enhance productivity and client service. The firm uses AI to process information faster, "
    "improve work quality, and create more capacity for advisory work. Over six months, "
    "500,000+ ChatGPT conversations were processed, with 98.6% of employees reporting higher "
    "productivity. The AI is embedded across tax advisory, legal research, and client "
    "communication. HSP views AI as part of their operating model, not just a productivity "
    "tool. The firm focuses on adoption, governance, and continuous learning through monthly "
    "AI forums. HSP GRUPPE has been digitizing processes for over two decades, using AI to "
    "rethink how professional work evolves. The firm emphasizes continuous improvement and "
    "data protection in AI integration. Before adoption, read the linked source and test the "
    "claim on a controlled task."
)


class VoiceOrphanRecoveryV50Tests(unittest.TestCase):
    def test_hsp_canary_narration_has_no_subminimum_voice_orphan(self) -> None:
        segments = split_narration_for_voice_v50(HSP_NARRATION, 6)

        self.assertTrue(all(len(item.split()) >= 12 for item in segments))
        self.assertTrue(all(len(item.split()) <= 26 for item in segments))
        self.assertEqual(" ".join(segments), HSP_NARRATION)
        merged = [
            item
            for item in segments
            if "continuous improvement and data protection" in item
        ]
        self.assertEqual(len(merged), 1)
        self.assertIn("Before adoption", merged[0])
        self.assertEqual(len(merged[0].split()), 25)

    def test_balanced_fallback_repairs_orphan_when_bounded_merge_cannot_fit(self) -> None:
        first = " ".join(f"first{i}" for i in range(24))
        orphan = " ".join(f"short{i}" for i in range(5))
        last = " ".join(f"last{i}" for i in range(24))
        original = f"{first} {orphan} {last}"

        repaired = repair_short_voice_orphans_v50(
            [first, orphan, last],
            minimum_words=12,
            maximum_words=24,
            merge_overflow_words=2,
        )

        self.assertTrue(all(12 <= len(item.split()) <= 26 for item in repaired))
        self.assertEqual(" ".join(repaired), original)

    def test_sentence_preserving_merge_is_preferred_over_mid_sentence_repartition(self) -> None:
        previous = " ".join(f"previous{i}" for i in range(18)) + "."
        orphan = " ".join(f"orphan{i}" for i in range(11)) + "."
        following = " ".join(f"following{i}" for i in range(14)) + "."

        repaired = repair_short_voice_orphans_v50(
            [previous, orphan, following],
            minimum_words=12,
            maximum_words=24,
            merge_overflow_words=2,
        )

        self.assertEqual(len(repaired), 2)
        self.assertEqual(repaired[0], previous)
        self.assertEqual(repaired[1], f"{orphan} {following}")
        self.assertEqual(len(repaired[1].split()), 25)

    def test_voice_failure_diagnostics_are_persisted_for_failed_canaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "artifact"
            calibration = root / "voice-calibration-failure.json"
            micro = root / "voice-micro-clause-failure.json"
            calibration.write_text('{"status":"failed"}', encoding="utf-8")
            micro.write_text('{"text":"exact transcript"}', encoding="utf-8")

            copied = copy_voice_failure_diagnostics_v50(root, destination)

            self.assertEqual({path.name for path in copied}, {calibration.name, micro.name})
            self.assertEqual(
                (destination / micro.name).read_text(encoding="utf-8"),
                micro.read_text(encoding="utf-8"),
            )

    def test_orphan_repair_never_changes_transcript_tokens(self) -> None:
        segments = [
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
            "one two three four five six seven eight nine ten eleven.",
            "red blue green black white gray orange purple silver gold bronze copper zinc.",
        ]
        original = " ".join(segments)
        repaired = repair_short_voice_orphans_v50(segments)
        self.assertEqual(" ".join(repaired), original)


if __name__ == "__main__":
    unittest.main()
