$ErrorActionPreference = "Stop"
python -m pip install --upgrade "modal>=1.0,<2"
modal setup
Write-Host "Set the Modal workspace budget, create ai-growth-factory-secrets, then run: modal deploy cloud/modal_app.py"
