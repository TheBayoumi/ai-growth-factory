$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

Write-Host "[1/5] Checking Python 3.12..."
$Python = $null
try {
    & py -3.12 --version | Out-Null
    $Python = "py -3.12"
} catch {
    throw "Python 3.12 is required. Install it, then rerun this script."
}

Write-Host "[2/5] Creating isolated environment..."
if (-not (Test-Path ".venv")) {
    & py -3.12 -m venv .venv
}
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r requirements.txt -r requirements-voice.txt -r requirements-reviewer.txt
& $VenvPython -m pip install -e .

Write-Host "[3/5] Checking llama.cpp..."
$Llama = Get-Command llama-server -ErrorAction SilentlyContinue
if (-not $Llama) {
    Write-Host "llama-server was not found. Installing llama.cpp with WinGet..."
    winget install llama.cpp --accept-package-agreements --accept-source-agreements
}

Write-Host "[4/5] Creating configuration..."
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}
if (-not (Test-Path "voice_contract.json")) {
    Copy-Item "voice_contract.example.json" "voice_contract.json"
}
New-Item -ItemType Directory -Force -Path "work", "state", "logs" | Out-Null

Write-Host "[5/5] Installation complete."
Write-Host "Edit $Root\.env, add YOUTUBE_OAUTH_JSON, then run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\doctor_windows.ps1"
Write-Host "The first doctor/run downloads the Qwen models into the local Hugging Face/llama.cpp cache."
