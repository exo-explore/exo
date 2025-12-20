#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Push exo to Android device via ADB for testing in Termux.

.DESCRIPTION
    This script pushes the exo Python source and helper scripts to an Android
    device running Termux. It sets up the environment for quick testing without
    needing to clone from git on the device.

.PARAMETER DeviceSerial
    Specific device serial if multiple devices connected (adb devices to list).

.PARAMETER TermuxPath
    Path inside Termux storage. Default: /data/data/com.termux/files/home/exo

.PARAMETER SharedStorage
    If set, uses shared storage (/sdcard/exo) which is easier to access but
    requires termux-setup-storage.

.EXAMPLE
    .\adb_push.ps1
    .\adb_push.ps1 -SharedStorage
    .\adb_push.ps1 -DeviceSerial "emulator-5554"
#>

param(
    [string]$DeviceSerial = "",
    [string]$TermuxPath = "/data/data/com.termux/files/home/exo",
    [switch]$SharedStorage = $false
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Header { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "⚠ $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "✗ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "ℹ $msg" -ForegroundColor Blue }

# Get script and exo directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExoDir = Split-Path -Parent $ScriptDir

Write-Host ""
Write-Host "╔═══════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║     exo ADB Push Script                   ║" -ForegroundColor Green
Write-Host "║     Push exo to Android/Termux            ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Check if adb is available
Write-Header "Checking ADB"
try {
    $adbVersion = & adb version 2>&1
    Write-Success "ADB found: $($adbVersion[0])"
} catch {
    Write-Error "ADB not found in PATH"
    Write-Info "Install Android SDK Platform Tools or add to PATH"
    exit 1
}

# Build ADB command with optional device serial
$adbCmd = if ($DeviceSerial) { "adb -s $DeviceSerial" } else { "adb" }

# Check for connected devices
Write-Header "Checking Connected Devices"
$devices = & adb devices 2>&1 | Select-Object -Skip 1 | Where-Object { $_ -match '\tdevice$' }
if (-not $devices) {
    Write-Error "No devices connected"
    Write-Info "1. Enable USB Debugging on your Android device"
    Write-Info "2. Connect via USB and authorize the connection"
    Write-Info "3. Or use 'adb connect <ip>:5555' for wireless debugging"
    exit 1
}
Write-Success "Found device(s):"
$devices | ForEach-Object { Write-Host "  $_" }

# Determine target path
if ($SharedStorage) {
    $TargetPath = "/sdcard/exo"
    Write-Info "Using shared storage: $TargetPath"
    Write-Warning "Run 'termux-setup-storage' in Termux first!"
} else {
    $TargetPath = $TermuxPath
    Write-Info "Using Termux home: $TargetPath"
}

# Files and directories to push
$PushItems = @(
    @{ Source = "src/exo"; Dest = "$TargetPath/src/exo" },
    @{ Source = "scripts"; Dest = "$TargetPath/scripts" },
    @{ Source = "pyproject.toml"; Dest = "$TargetPath/pyproject.toml" },
    @{ Source = "README.md"; Dest = "$TargetPath/README.md" }
)

# Create termux install helper script
$InstallScript = @"
#!/data/data/com.termux/files/usr/bin/bash
# Auto-generated install script for exo in Termux
# Run this after adb push to set up the environment

set -e

echo "Setting up exo in Termux..."

# Navigate to exo directory
cd ~/exo 2>/dev/null || cd /sdcard/exo || {
    echo "Error: exo directory not found"
    exit 1
}

# If using shared storage, copy to home
if [[ "\$PWD" == /sdcard/* ]]; then
    echo "Copying from shared storage to Termux home..."
    mkdir -p ~/exo
    cp -r /sdcard/exo/* ~/exo/
    cd ~/exo
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e . 2>/dev/null || {
    echo "Installing core dependencies first..."
    pip install aiofiles aiohttp pydantic fastapi psutil loguru rich huggingface-hub
}

# Check if llama-cpp-python is installed
if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo ""
    echo "llama-cpp-python not installed."
    echo "Run: ./scripts/termux_setup.sh"
    echo "Or: pip install llama-cpp-python (10-20 min build)"
fi

echo ""
echo "Done! Run exo with:"
echo "  cd ~/exo && python3 -m exo"
"@

Write-Header "Pushing exo to Device"

# Create target directory on device
Write-Info "Creating target directory..."
& cmd /c "$adbCmd shell mkdir -p $TargetPath/src 2>nul"

# Push each item
foreach ($item in $PushItems) {
    $sourcePath = Join-Path $ExoDir $item.Source
    if (Test-Path $sourcePath) {
        Write-Info "Pushing $($item.Source)..."
        & cmd /c "$adbCmd push `"$sourcePath`" `"$($item.Dest)`" 2>nul" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Pushed $($item.Source)"
        } else {
            Write-Warning "Failed to push $($item.Source)"
        }
    } else {
        Write-Warning "Not found: $sourcePath"
    }
}

# Write and push install helper script
$TempScript = Join-Path $env:TEMP "termux_install.sh"
$InstallScript | Out-File -FilePath $TempScript -Encoding UTF8 -NoNewline
& cmd /c "$adbCmd push `"$TempScript`" `"$TargetPath/install.sh`" 2>nul" | Out-Null
Remove-Item $TempScript -ErrorAction SilentlyContinue
Write-Success "Created install.sh helper"

Write-Header "Push Complete!"

Write-Host ""
Write-Host "Files pushed to: $TargetPath" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps in Termux:" -ForegroundColor Cyan
Write-Host ""
if ($SharedStorage) {
    Write-Host "  1. cp -r /sdcard/exo ~/exo"
    Write-Host "  2. cd ~/exo"
} else {
    Write-Host "  1. cd ~/exo"
}
Write-Host "  2. chmod +x install.sh scripts/*.sh"
Write-Host "  3. ./install.sh"
Write-Host ""
Write-Host "Or for full setup with llama-cpp-python:" -ForegroundColor Cyan
Write-Host "  ./scripts/termux_setup.sh"
Write-Host ""

