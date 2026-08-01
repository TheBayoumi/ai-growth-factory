import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionRuntimeAttributionContractTests(unittest.TestCase):
    def test_production_runtime_does_not_install_deleted_heuristic(self):
        runtime = (ROOT / "factory" / "production_runtime.py").read_text(encoding="utf-8")
        modal = (ROOT / "cloud" / "modal_app.py").read_text(encoding="utf-8")

        self.assertNotIn("source_index_repair", runtime)
        self.assertNotIn("install_source_index_repair", runtime)
        self.assertIn("source_attributed_llm", runtime)
        self.assertIn("install_production_runtime", modal)


if __name__ == "__main__":
    unittest.main()
