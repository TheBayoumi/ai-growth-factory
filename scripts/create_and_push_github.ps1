param(
    [string]$Repository = "TheBayoumi/ai-growth-factory",
    [ValidateSet("private", "public")][string]$Visibility = "private"
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required. Install it, authenticate with 'gh auth login', and rerun."
}
python scripts/repository_preflight.py
if ($LASTEXITCODE -ne 0) { throw "Repository preflight failed." }
$visibilityFlag = if ($Visibility -eq "private") { "--private" } else { "--public" }
& gh repo create $Repository $visibilityFlag --source . --remote origin --push
if ($LASTEXITCODE -ne 0) { throw "GitHub repository creation or push failed." }
Write-Host "Created and pushed https://github.com/$Repository"
