#!/usr/bin/env bash
set -euo pipefail
repository="${1:-TheBayoumi/ai-growth-factory}"
visibility="${2:-private}"
command -v gh >/dev/null || { echo "GitHub CLI (gh) is required." >&2; exit 1; }
python scripts/repository_preflight.py
flag="--private"
[[ "$visibility" == "public" ]] && flag="--public"
gh repo create "$repository" "$flag" --source . --remote origin --push
echo "Created and pushed https://github.com/$repository"
