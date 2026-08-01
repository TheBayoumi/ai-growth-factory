$ErrorActionPreference = "Stop"
python -m pip install --upgrade "modal>=1.0,<2"
modal setup
Write-Host @"
Before deployment, set the Modal workspace budget to `$30/month.

Create the production secret once:

modal secret create ai-growth-factory-secrets `
  YOUTUBE_OAUTH_JSON='{"client_id":"...","client_secret":"...","refresh_token":"..."}' `
  PUBLISH_ENABLED=true `
  YOUTUBE_PRIVACY_STATUS=private

Then deploy:
  modal deploy cloud/modal_app.py

Run a private canary:
  modal run cloud/modal_app.py --canary
"@
