import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"
MODAL_WORKFLOW = ROOT / ".github" / "workflows" / "modal-production-verification.yml"


class ModalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")
        cls.workflow = MODAL_WORKFLOW.read_text(encoding="utf-8")

    def test_modal_source_is_valid_python(self):
        ast.parse(self.source, filename=str(MODAL_APP))

    def test_free_first_resource_and_schedule_contract(self):
        required = (
            'gpu="T4"',
            "max_containers=1",
            'modal.Cron("0 10 * * *", timezone="Africa/Cairo")',
            '"REVIEWER_BACKEND": "qwen_omni"',
            '"YOUTUBE_PRIVACY_STATUS": "private"',
            '"PUBLISH_ENABLED": "false"',
            'def render_production_canary()',
            'MODAL_USE_FACTORY_SECRET',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_gptq_build_uses_preinstalled_torch(self):
        self.assertIn(
            '"python -m pip install --no-build-isolation gptqmodel==2.0.0"',
            self.source,
        )
        pip_install_section = self.source.split(".pip_install(", 1)[1].split(
            ")\n    # gptqmodel", 1
        )[0]
        self.assertNotIn("gptqmodel", pip_install_section)

    def test_openai_is_not_a_modal_runtime_dependency(self):
        self.assertNotIn("OPENAI_API_KEY", self.source)
        self.assertNotIn("gpt-realtime", self.source.lower())

    def test_model_and_state_volumes_are_persistent(self):
        self.assertIn('Volume.from_name("ai-growth-factory-model-cache"', self.source)
        self.assertIn('Volume.from_name("ai-growth-factory-state"', self.source)
        self.assertIn("state_volume.commit()", self.source)
        self.assertIn("hf_cache.commit()", self.source)

    def test_ci_uses_persistent_modal_secrets_not_browser_auth(self):
        self.assertIn('MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}', self.workflow)
        self.assertIn('MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}', self.workflow)
        self.assertIn("environment: modal-production", self.workflow)
        self.assertNotIn("modal token new", self.workflow)
        self.assertNotIn("Authorize Modal in browser", self.workflow)

    def test_canary_step_outcome_is_enforced_after_artifact_collection(self):
        self.assertIn("continue-on-error: true", self.workflow)
        self.assertIn('CANARY_STEP_OUTCOME: ${{ steps.canary.outcome }}', self.workflow)
        self.assertIn('if [ "${CANARY_STEP_OUTCOME}" != "success" ]', self.workflow)
        self.assertIn("canary-failure.json", self.workflow)


if __name__ == "__main__":
    unittest.main()
