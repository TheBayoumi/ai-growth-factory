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

    def test_production_uses_a10_native_omni_and_visual_offload_capacity(self):
        self.assertIn('gpu="A10"', self.source)
        self.assertNotIn('gpu="T4"', self.source)
        self.assertIn('"QWEN_OMNI_REVIEW_MODEL": "Qwen/Qwen2.5-Omni-3B"', self.source)
        self.assertIn('"QWEN_OMNI_DTYPE": "float16"', self.source)
        self.assertIn('"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"', self.source)
        self.assertIn("memory=65536", self.source)
        self.assertIn("timeout=85 * 60", self.source)

    def test_native_reviewer_has_no_gptq_or_optimum_dependency(self):
        combined = self.source + "\n" + self.reviewer_requirements
        self.assertNotIn("gptqmodel", combined.lower())
        self.assertNotIn("optimum", combined.lower())
        self.assertIn("Qwen2_5OmniForConditionalGeneration", self.source)
        self.assertIn("Qwen2_5OmniProcessor", self.source)
        self.assertIn(
            "Voice, reviewer, image, and Wan2.2 visual runtime import preflight passed",
            self.source,
        )

    def test_voice_visual_runtime_and_output_contract(self):
        for token in (
            '"QWEN_TTS_DEVICE": "cuda:0"',
            '"QWEN_TTS_DTYPE": "float32"',
            '"QWEN_TTS_ATTENTION": "sdpa"',
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

    def test_production_is_gated_on_both_merge_ci_matrix_jobs(self):
        self.assertIn("production-verification:", self.workflow)
        self.assertIn("needs: test", self.workflow)
        self.assertIn('python-version: ["3.12", "3.13"]', self.workflow)
        self.assertIn("github.event.pull_request.merged == true", self.workflow)
        self.assertIn("startsWith(github.event.pull_request.head.ref, 'verify/modal-gpu-')", self.workflow)
        self.assertFalse(DUPLICATE_PRODUCTION_WORKFLOW.exists())

    def test_production_workflow_uses_persistent_modal_secrets(self):
        self.assertIn('MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}', self.workflow)
        self.assertIn('MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}', self.workflow)
        self.assertIn("environment: modal-production", self.workflow)
        self.assertNotIn("modal token new", self.workflow)
        self.assertNotIn("Authorize Modal in browser", self.workflow)

    def test_production_runs_after_verified_checkout(self):
        self.assertIn("Check out CI-verified main", self.workflow)
        self.assertIn("Autonomous visual pipeline import preflight passed", self.workflow)
        self.assertIn("Deploy A10 production worker", self.workflow)
        self.assertIn("Generate real platform-ready video and capture result", self.workflow)
        test_index = self.workflow.index("Unit and integration tests")
        production_index = self.workflow.index("production-verification:")
        self.assertLess(test_index, production_index)

    def test_output_bundle_is_downloaded_and_enforced(self):
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
        ):
            with self.subTest(name=name):
                self.assertIn(f'"{name}"', self.workflow)
        self.assertIn('video_qc.get("width") != 1080', self.workflow)
        self.assertIn('video_qc.get("height") != 1920', self.workflow)
        self.assertIn('visuals.get("wan_scene_count") != 3', self.workflow)
        self.assertIn('captions_baked_into_generated_media', self.workflow)

    def test_failed_generation_still_uploads_diagnostics(self):
        guard = "steps.production.outcome == 'success' || steps.production.outcome == 'failure'"
        self.assertGreaterEqual(self.workflow.count(guard), 3)
        self.assertIn("continue-on-error: true", self.workflow)
        self.assertIn('PRODUCTION_STEP_OUTCOME: ${{ steps.production.outcome }}', self.workflow)
        self.assertIn("canary-failure.json", self.workflow)


if __name__ == "__main__":
    unittest.main()
