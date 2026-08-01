#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v python3.12 >/dev/null || { echo "Python 3.12 is required." >&2; exit 1; }
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt -r requirements-voice.txt -r requirements-reviewer.txt
.venv/bin/python -m pip install -e .
if ! command -v llama-server >/dev/null; then
  echo "llama-server is missing. Install llama.cpp, then rerun doctor."
fi
[[ -f .env ]] || cp .env.example .env
[[ -f voice_contract.json ]] || cp voice_contract.example.json voice_contract.json
mkdir -p work state logs
printf 'Installed. Edit %s/.env, then run scripts/doctor_linux.sh\n' "$ROOT"
