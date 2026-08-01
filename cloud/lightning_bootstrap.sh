#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[voice,reviewer]'
echo 'Lightning workspace prepared. Use it for manual canaries and debugging, not the daily production cron.'
