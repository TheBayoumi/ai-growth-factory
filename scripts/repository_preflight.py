from __future__ import annotations

import json
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
EXECUTABLE_CONFIG_SUFFIXES = {".py", ".toml", ".json", ".yml", ".yaml", ".sh", ".ps1"}
FORBIDDEN_VERCEL_PATTERNS = {
    "Vercel CLI deployment": re.compile(
        r"(?im)(?:^|[\s;&|])(?:npx\s+)?vercel\s+(?:deploy\b|--prod\b)"
    ),
    "Vercel deployment API/MCP call": re.compile(r"\bdeploy_to_vercel\b"),
    "Vercel deployment token": re.compile(r"\bVERCEL_TOKEN\b"),
    "Vercel project linkage": re.compile(r"\bVERCEL_(?:ORG_ID|PROJECT_ID)\b"),
}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def validate_vercel_kill_switch(failures: list[str]) -> None:
    policy_path = ROOT / "vercel.json"
    if not policy_path.is_file():
        failures.append("vercel.json kill-switch is missing")
        return
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.append(f"vercel.json kill-switch is unreadable: {exc}")
        return
    deployment_enabled = (payload.get("git") or {}).get("deploymentEnabled")
    if deployment_enabled is not False:
        failures.append("vercel.json must set git.deploymentEnabled=false")


def main() -> int:
    failures: list[str] = []
    validate_vercel_kill_switch(failures)

    for path in iter_files():
        rel = path.relative_to(ROOT)
        if ".vercel" in rel.parts:
            failures.append(f"tracked Vercel project linkage is forbidden: {rel}")
            continue
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
        if path.suffix.lower() in EXECUTABLE_CONFIG_SUFFIXES and path.name != "vercel.json":
            for label, pattern in FORBIDDEN_VERCEL_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label} is forbidden by Modal-only policy: {rel}")

    if failures:
        print("Repository preflight failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        "Repository preflight passed: no generated media or obvious credentials found; "
        "Vercel deployment is disabled and Modal-only policy is intact."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
