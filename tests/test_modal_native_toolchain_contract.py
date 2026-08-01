import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalNativeToolchainContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_gptq_dependency_build_has_required_native_packages(self):
        for package in (
            '"clang"',
            '"build-essential"',
            '"pkg-config"',
            '"cmake"',
            '"ninja-build"',
            '"libpcre2-dev"',
        ):
            with self.subTest(package=package):
                self.assertIn(package, self.source)

    def test_cpu_safe_gptq_install_remains_enabled(self):
        self.assertIn(
            "BUILD_CUDA_EXT=0 python -m pip install --no-build-isolation "
            "gptqmodel==5.7.0",
            self.source,
        )
        self.assertIn("quantizer.validate_environment()", self.source)


if __name__ == "__main__":
    unittest.main()
