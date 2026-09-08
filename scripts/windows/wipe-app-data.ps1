# Delete exo Windows app data (settings, logs, pid, models under LOCALAPPDATA).
# Does not uninstall the repo or touch %USERPROFILE%\.ollama / LM Studio libraries.

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Confirm
)

$ErrorActionPreference = 'Continue'
$Targets = @(
    (Join-Path $env:LOCALAPPDATA 'exo'),
    (Join-Path $HOME '.exo')
)

if ($DryRun) {
    Write-Host '[dry-run] Would remove:'
    $Targets | ForEach-Object { Write-Host "  $_" }
    exit 0
}

if (-not $Confirm) {
    Write-Host 'This deletes exo app data under LOCALAPPDATA\exo and %USERPROFILE%\.exo'
    Write-Host 'Type "wipe" to confirm:'
    $answer = Read-Host '>'
    if ($answer -ne 'wipe') {
        Write-Host 'Aborted.'
        exit 130
    }
}

foreach ($p in $Targets) {
    if (Test-Path -LiteralPath $p) {
        Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "removed $p"
    }
}

Write-Host 'Done. Restart exo with .\scripts\windows\run.ps1'
