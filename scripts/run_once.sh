#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python -m factory run 2>&1 | tee "logs/run-$STAMP.log"
