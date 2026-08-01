$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts\install_windows.ps1 first." }
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Log = Join-Path $Root "logs\run-$Stamp.log"
try {
    & $Python -m factory run 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { throw "Factory exited with code $LASTEXITCODE. Log: $Log" }
} catch {
    $_ | Out-String | Add-Content $Log
    throw
}
