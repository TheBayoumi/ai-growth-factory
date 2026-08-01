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

    def test_t4_voice_runtime_uses_stable_precision_and_complete_audio_tools(self):
        self.assertIn('"QWEN_TTS_DEVICE": "cuda:0"', self.source)
        self.assertIn('"QWEN_TTS_DTYPE": "float32"', self.source)
        self.assertIn('"QWEN_TTS_ATTENTION": "sdpa"', self.source)
        self.assertIn('"sox"', self.source)
        self.assertIn('"libsox-fmt-all"', self.source)

    def test_qwen_omni_runtime_is_complete_and_preflighted(self):
        self.assertIn("torchvision==0.23.0", self.source)
        self.assertIn('"numpy==2.0.0"', self.source)
        self.assertIn('"qwen-omni-utils[decord]>=0.0.8"', self.source)
        self.assertIn("import decord, torch, torchvision", self.source)
        self.assertIn("Qwen2_5OmniForConditionalGeneration", self.source)
        self.assertIn("Qwen2_5OmniProcessor", self.source)
        self.assertIn("Qwen Omni runtime import preflight passed", self.source)
        self.assertLess(
            self.source.index("Qwen Omni runtime import preflight passed"),
            self.source.index(".env("),
        )

    def test_real_t4_reviewer_probe_runs_before_expensive_canary(self):
        self.assertIn("def reviewer_runtime_probe()", self.source)
        self.assertIn("from qwen_tts import Qwen3TTSModel", self.source)
        self.assertIn("Qwen reviewer T4 runtime probe failed", self.source)
        self.assertIn("--reviewer-probe", self.workflow)
        probe_index = self.workflow.index("Probe T4 reviewer runtime")
        canary_index = self.workflow.index("Run real render-only production canary")
        self.assertLess(probe_index, canary_index)
        self.assertIn("modal-reviewer-probe.txt", self.workflow)

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

    def test_credentials_fail_before_expensive_setup(self):
        credentials_index = self.workflow.index("Validate persistent Modal credentials")
        python_index = self.workflow.index("Set up Python")
        install_index = self.workflow.index("Install system dependencies")
        self.assertLess(credentials_index, python_index)
        self.assertLess(credentials_index, install_index)
        self.assertIn("Verify Modal credential pair", self.workflow)
        self.assertIn("modal token info", self.workflow)

    def test_artifact_steps_do_not_run_when_canary_was_skipped(self):
        guard = "steps.canary.outcome == 'success' || steps.canary.outcome == 'failure'"
        self.assertGreaterEqual(self.workflow.count(guard), 3)

    def test_modal_volume_download_materializes_directory_contents(self):
        self.assertIn("rm -rf production-canary", self.workflow)
        self.assertIn("mkdir -p production-canary", self.workflow)
        self.assertIn(
            'modal volume get --force ai-growth-factory-state "canaries/${canary_id}/" production-canary/',
            self.workflow,
        )
        self.assertIn("find production-canary -type f -print -quit", self.workflow)
        self.assertIn("find production-canary -type f -print | sort", self.workflow)
        self.assertNotIn(
            'modal volume get --force ai-growth-factory-state "canaries/${canary_id}" production-canary\n',
            self.workflow,
        )

    def test_enforcement_resolves_nested_modal_export_directory(self):
        self.assertIn("find production-canary -name canary-failure.json", self.workflow)
        self.assertIn("find production-canary -name canary-result.json", self.workflow)
        self.assertIn("result_path.parent", self.workflow)

    def test_canary_step_outcome_is_enforced_after_artifact_collection(self):
        self.assertIn("continue-on-error: true", self.workflow)
        self.assertIn('CANARY_STEP_OUTCOME: ${{ steps.canary.outcome }}', self.workflow)
        self.assertIn('if [ "${CANARY_STEP_OUTCOME}" != "success" ]', self.workflow)
        self.assertIn("canary-failure.json", self.workflow)


if __name__ == "__main__":
    unittest.main()
