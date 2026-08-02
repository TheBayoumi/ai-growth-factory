import unittest
from pathlib import Path

from factory.production_renderer import _balanced_caption_chunks


ROOT = Path(__file__).resolve().parents[1]


class PublishableProductionRuntimeTests(unittest.TestCase):
    def test_caption_chunks_are_short_and_complete(self):
        text = (
            "Echoverse trains computer-use agents inside changing interfaces "
            "instead of replaying one fixed benchmark task"
        )
        chunks = _balanced_caption_chunks(text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks), 3)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk.split()) <= 9 for chunk in chunks))

    def test_modal_runtime_activates_publishable_policies(self):
        source = (ROOT / "cloud" / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn("install_production_runtime", source)
        self.assertIn('"AUDIO_WPM_TOLERANCE": "15"', source)
        self.assertIn('"VIDEO_WIDTH": "1080"', source)
        self.assertIn('"VIDEO_HEIGHT": "1920"', source)
        self.assertIn('"VIDEO_FPS": "30"', source)

    def test_production_runtime_has_no_camera_motion(self):
        source = (ROOT / "factory" / "production_renderer.py").read_text(encoding="utf-8")
        self.assertIn("xfade=transition=fade", source)
        self.assertNotIn("zoompan", source)
        self.assertNotIn("crop", source)

    def test_content_gate_rejects_generic_copy_phrases(self):
        source = (ROOT / "factory" / "production_content.py").read_text(encoding="utf-8")
        self.assertIn('"ai advancements"', source)
        self.assertIn('"shaping the future"', source)
        self.assertIn("130-155 words", source)
        self.assertIn("ONE coherent, current story", source)

    def test_pacing_allows_bounded_pitch_preserving_segment_correction(self):
        pacing = (ROOT / "factory" / "production_pacing.py").read_text(encoding="utf-8")
        pipeline = (ROOT / "factory" / "voice_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("_MAX_PRODUCTION_TEMPO = 1.45", pacing)
        self.assertIn("voice_pipeline.tempo_correction_factor", pacing)
        self.assertIn("_pace_correct_segment_assets", pipeline)
        self.assertIn("deterministic_segment_tempo_correction", pipeline)
        self.assertIn("already-compliant segments", pacing)


if __name__ == "__main__":
    unittest.main()
