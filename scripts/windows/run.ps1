# Start exo on Windows from a source checkout.
#
# PAIR ships a single Start-menu launcher that brings up the local cluster
# process. This script is the same idea for exo: frozen deps, Python 3.13,
# mlx-cpu extra already installed via `uv sync --extra mlx-cpu`.
#
# Usage:
#   .\scripts\windows\run.ps1
#   .\scripts\windows\run.ps1 --api-port 52415

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error 'uv is not on PATH. Install https://docs.astral.sh/uv/ and re-run.'
}

& uv run --python 3.13 --frozen exo @args
exit $LASTEXITCODE
