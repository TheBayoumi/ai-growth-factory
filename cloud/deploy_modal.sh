#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade 'modal>=1.0,<2'
modal setup
cat <<'EOF'
Before deployment, set the Modal workspace budget to $30/month.

Create the production secret once, using the JSON as one shell argument:

modal secret create ai-growth-factory-secrets \
  YOUTUBE_OAUTH_JSON='{"client_id":"...","client_secret":"...","refresh_token":"..."}' \
  PUBLISH_ENABLED=true \
  YOUTUBE_PRIVACY_STATUS=private

Then deploy:
  modal deploy cloud/modal_app.py

Run a private canary manually:
  modal run cloud/modal_app.py --canary
EOF
