param(
    [Parameter(Mandatory=$true)][string]$Script,
    [switch]$NoReview
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Output = Join-Path $Root ("voice-tests\" + (Get-Date -Format "yyyyMMdd-HHmmss"))
$Args = @("-m", "factory", "voice-test", $Script, "--output-dir", $Output)
if ($NoReview) { $Args += "--no-review" }
& $Python @Args
exit $LASTEXITCODE
