import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "cloud" / "modal_app.py"


class ModalNativeBuildDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")

    def test_pypcre_native_dependencies_precede_gptqmodel_install(self):
        required = (
            '"clang"',
            '"build-essential"',
            '"pkg-config"',
            '"libpcre2-dev"',
        )
        for package in required:
            with self.subTest(package=package):
                self.assertIn(package, self.source)

        apt_index = self.source.index('.apt_install(')
        gptq_index = self.source.index('gptqmodel==5.7.0')
        self.assertLess(apt_index, gptq_index)

    def test_cpu_only_image_build_still_disables_optional_cuda_extension(self):
        self.assertIn(
            'BUILD_CUDA_EXT=0 python -m pip install --no-build-isolation gptqmodel==5.7.0',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
