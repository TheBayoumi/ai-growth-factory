from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "dist", "build", "__pycache__"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".json", ".yml", ".yaml", ".sh", ".ps1", ".service", ".timer"
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Google OAuth secret": re.compile(r'"client_secret"\s*:\s*"(?!\.\.\.)[^"\n]{8,}"'),
    "Modal token": re.compile(r"ak-[A-Za-z0-9_-]{20,}"),
}
FORBIDDEN_TRACKED_SUFFIXES = {".wav", ".mp3", ".mp4", ".mov", ".mkv", ".webm"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def main() -> int:
    failures: list[str] = []
    for path in iter_files():
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES:
            failures.append(f"generated media must not be committed: {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".env.example", ".gitignore", "LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"possible {label} in {rel}")
    if failures:
        print("Repository preflight failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Repository preflight passed: no generated media or obvious credentials found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
