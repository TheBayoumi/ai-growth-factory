import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DUPLICATE_PRODUCTION_WORKFLOW = ROOT / ".github" / "workflows" / "modal-production-verification.yml"
REVIEWER_REQUIREMENTS = ROOT / "requirements-reviewer.txt"


class ModalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")
        cls.workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        cls.reviewer_requirements = REVIEWER_REQUIREMENTS.read_text(encoding="utf-8")

    def test_modal_source_is_valid_python(self):
        ast.parse(self.source, filename=str(MODAL_APP))

    def test_production_uses_a10_4bit_omni_and_visual_offload_capacity(self):
        self.assertIn('gpu="A10"', self.source)
        self.assertNotIn('gpu="T4"', self.source)
        self.assertIn(
            '"QWEN_OMNI_REVIEW_MODEL": "Qwen/Qwen2.5-Omni-7B"',
            self.source,
        )
        self.assertIn('"QWEN_OMNI_DTYPE": "float16"', self.source)
        self.assertIn('"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"', self.source)
        self.assertIn("memory=65536", self.source)
        self.assertIn("timeout=85 * 60", self.source)

    def test_quantized_reviewer_has_pinned_wheel_runtime(self):
        self.assertIn('"bitsandbytes==0.50.0"', self.source)
        self.assertNotIn("gptqmodel", self.source.casefold())
        self.assertNotIn("optimum==", self.source)
        self.assertIn("BitsAndBytesConfig", self.source)
        self.assertIn("Qwen2_5OmniForConditionalGeneration", self.source)
        self.assertIn("Qwen2_5OmniProcessor", self.source)
        self.assertIn(
            "Voice, bitsandbytes 4-bit Omni reviewer, image, and Wan2.2 runtime preflight passed",
            self.source,
        )

    def test_voice_visual_runtime_and_output_contract(self):
        for token in (
            '"QWEN_TTS_DEVICE": "cuda:0"',
            '"QWEN_TTS_DTYPE": "float32"',
            '"QWEN_TTS_ATTENTION": "sdpa"',
            '"MIN_PRIMARY_SOURCES": "1"',
            '"VISUAL_IMAGE_BACKEND": "auto"',
            '"WAN22_MODEL_ID": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"',
            '"VIDEO_WIDTH": "1080"',
            '"VIDEO_HEIGHT": "1920"',
            '"VIDEO_FPS": "30"',
            '"fonts-dejavu-core"',
            '"sox"',
            '"libsox-fmt-all"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_schedule_and_persistent_volumes_remain_enabled(self):
        self.assertIn('modal.Cron("0 10 * * *", timezone="Africa/Cairo")', self.source)
        self.assertIn("max_containers=1", self.source)
        self.assertIn('Volume.from_name("ai-growth-factory-model-cache"', self.source)
        self.assertIn('Volume.from_name("ai-growth-factory-state"', self.source)
        self.assertIn("state_volume.commit()", self.source)
        self.assertIn("hf_cache.commit()", self.source)

    def test_paid_production_review_is_manual_and_follows_local_test_gate(self):
        self.assertIn("production-verification:", self.workflow)
        self.assertIn("needs: test", self.workflow)
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn("run_production_review:", self.workflow)
        self.assertIn("default: false", self.workflow)
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && inputs.run_production_review",
            self.workflow,
        )
        self.assertNotIn("pull_request:", self.workflow)
        self.assertNotIn('python-version: ["3.12", "3.13"]', self.workflow)
        deploy_workflow = (ROOT / ".github" / "workflows" / "modal-deploy.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("workflow_run:", deploy_workflow)
        self.assertIn("workflow_dispatch:", deploy_workflow)
        self.assertFalse(DUPLICATE_PRODUCTION_WORKFLOW.exists())

    def test_production_workflow_uses_persistent_modal_secrets(self):
        self.assertIn('MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}', self.workflow)
        self.assertIn('MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}', self.workflow)
        self.assertIn("environment: modal-production", self.workflow)
        self.assertNotIn("modal token new", self.workflow)
        self.assertNotIn("Authorize Modal in browser", self.workflow)

    def test_production_runs_from_the_exact_manually_requested_ref(self):
        self.assertIn("Check out exact requested ref", self.workflow)
        self.assertIn('ref: ${{ inputs.ref }}', self.workflow)
        self.assertIn("REVIEW_SHA=$(git rev-parse HEAD)", self.workflow)
        self.assertIn("Production runtime import preflight passed", self.workflow)
        self.assertIn("Deploy exact requested ref to isolated Modal tag", self.workflow)
        self.assertIn('modal deploy cloud/modal_app.py --tag "${REVIEW_SHA}"', self.workflow)
        self.assertIn("Generate exact pre-merge production video", self.workflow)
        self.assertNotIn("github.event.pull_request", self.workflow)
        test_index = self.workflow.index("Unit and integration tests")
        production_index = self.workflow.index("production-verification:")
        self.assertLess(test_index, production_index)

    def test_output_bundle_is_downloaded_and_profile_contract_is_enforced(self):
        self.assertIn("rm -rf production-video", self.workflow)
        self.assertIn("mkdir -p production-video", self.workflow)
        self.assertIn(
            'modal volume get --force ai-growth-factory-state "canaries/${canary_id}/" production-video/',
            self.workflow,
        )
        for name in (
            "video.mp4",
            "narration.wav",
            "thumbnail.png",
            "voice-review-manifest.json",
            "video-qc-report.json",
            "package.json",
            "visual-plan.json",
            "keyframe-manifest.json",
            "scene-media-manifest.json",
            "animated-captions.ass",
            "animated-captions.json",
            "visual-composition-manifest.json",
            "visual-pipeline-manifest.json",
            "visual-compositor.log",
            "background-music.wav",
        ):
            with self.subTest(name=name):
                self.assertIn(f'"{name}"', self.workflow)
        self.assertIn('contract = composition.get("editorial_contract") or {}', self.workflow)
        self.assertIn('required_wan_shots = int(contract["wan_shots"])', self.workflow)
        self.assertIn('actual_wan_shots != required_wan_shots', self.workflow)
        self.assertIn('realized_shot_count', self.workflow)
        self.assertIn('realized_wan_shots', self.workflow)
        self.assertIn('video_qc.get("width") != 1080', self.workflow)
        self.assertIn('video_qc.get("height") != 1920', self.workflow)
        self.assertNotIn('visuals.get("wan_scene_count") != 3', self.workflow)

    def test_failed_generation_still_uploads_diagnostics(self):
        guard = "steps.production.outcome == 'success' || steps.production.outcome == 'failure'"
        self.assertGreaterEqual(self.workflow.count(guard), 3)
        self.assertIn("continue-on-error: true", self.workflow)
        self.assertIn('PRODUCTION_STEP_OUTCOME: ${{ steps.production.outcome }}', self.workflow)
        self.assertIn("canary-failure.json", self.workflow)


if __name__ == "__main__":
    unittest.main()
