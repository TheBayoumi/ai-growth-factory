from __future__ import annotations

import unittest
from pathlib import Path


class SourceTreeTests(unittest.TestCase):
    def test_no_archive_bootstrap_files(self) -> None:
        forbidden = list(Path(".").glob("source.zip*")) + list(Path(".").glob("*.b64.part*"))
        self.assertEqual(forbidden, [])

    def test_core_source_exists(self) -> None:
        for path in ("pyproject.toml", "factory/pipeline.py", "factory/render.py", "factory/youtube.py", "cloud/modal_app.py"):
            self.assertTrue(Path(path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
