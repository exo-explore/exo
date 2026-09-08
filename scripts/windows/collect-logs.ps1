# Copy exo logs into a timestamped folder next to this script.
# Inspired by PAIR's collect-logs helper (sanitized shareable bundle).

[CmdletBinding()]
param(
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'

$LogRoot = Join-Path $env:LOCALAPPDATA 'exo\exo_log'
if (-not (Test-Path -LiteralPath $LogRoot)) {
    $LogRoot = Join-Path $HOME '.exo\exo_log'
}

if (-not $OutDir) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutDir = Join-Path (Get-Location) "exo-logs-$stamp"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $LogRoot)) {
    Write-Warning "No exo log directory found at $LogRoot"
    exit 0
}

Copy-Item -LiteralPath $LogRoot -Destination (Join-Path $OutDir 'exo_log') -Recurse -Force
Write-Host "Wrote logs to $OutDir"
