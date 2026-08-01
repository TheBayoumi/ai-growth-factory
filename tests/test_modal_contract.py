import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_modal_source_is_valid_python(self):
        ast.parse(self.source, filename=str(MODAL_APP))

    def test_free_first_resource_and_schedule_contract(self):
        required = (
            'gpu="T4"',
            "max_containers=1",
            'modal.Cron("0 10 * * *", timezone="Africa/Cairo")',
            '"REVIEWER_BACKEND": "qwen_omni"',
            '"YOUTUBE_PRIVACY_STATUS": "private"',
            '"PUBLISH_ENABLED": "true"',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_openai_is_not_a_modal_runtime_dependency(self):
        self.assertNotIn("OPENAI_API_KEY", self.source)
        self.assertNotIn("gpt-realtime", self.source.lower())

    def test_model_and_state_volumes_are_persistent(self):
        self.assertIn('Volume.from_name("ai-growth-factory-model-cache"', self.source)
        self.assertIn('Volume.from_name("ai-growth-factory-state"', self.source)
        self.assertIn("state_volume.commit()", self.source)
        self.assertIn("hf_cache.commit()", self.source)


if __name__ == "__main__":
    unittest.main()
