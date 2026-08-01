from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".wav", ".mp3", ".mp4", ".mov", ".mkv", ".webm", ".gguf", ".safetensors", ".pt", ".pth"}
PATTERNS = [re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), re.compile(r"\b1//[A-Za-z0-9_-]{20,}\b")]


def main() -> int:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    failures = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = ROOT / raw.decode()
        if path.suffix.lower() in FORBIDDEN:
            failures.append(f"forbidden tracked artifact: {path.relative_to(ROOT)}")
        if path.is_file() and b"\0" not in path.read_bytes()[:8192]:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in PATTERNS):
                failures.append(f"potential credential in {path.relative_to(ROOT)}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Repository preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
