from __future__ import annotations

import unittest
from pathlib import Path


class ModalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path("cloud/modal_app.py").read_text(encoding="utf-8")

    def test_t4_is_bounded(self) -> None:
        self.assertIn('gpu="T4"', self.source)
        self.assertIn("max_containers=1", self.source)
        self.assertIn("timeout=30 * 60", self.source)

    def test_daily_cairo_schedule(self) -> None:
        self.assertIn('modal.Cron("0 10 * * *", timezone="Africa/Cairo")', self.source)


if __name__ == "__main__":
    unittest.main()
