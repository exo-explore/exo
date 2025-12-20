#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Cross-compile exo Rust bindings for Android/Termux.

.DESCRIPTION
    This script sets up and runs cross-compilation of the exo_pyo3_bindings
    Rust crate for Android aarch64 target. The resulting wheel can be pushed
    to Android devices via ADB or published as a release.

.PARAMETER Setup
    Run first-time setup (install Rust target, Android NDK).

.PARAMETER Build
    Build the wheel for Android.

.PARAMETER NdkPath
    Path to Android NDK. Default: auto-detect from ANDROID_NDK_HOME.

.EXAMPLE
    .\cross_compile_android.ps1 -Setup
    .\cross_compile_android.ps1 -Build
#>

param(
    [switch]$Setup = $false,
    [switch]$Build = $false,
    [string]$NdkPath = ""
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Header { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "⚠ $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "✗ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "ℹ $msg" -ForegroundColor Blue }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExoDir = Split-Path -Parent $ScriptDir
$RustBindingsDir = Join-Path $ExoDir "rust\exo_pyo3_bindings"

Write-Host ""
Write-Host "╔═══════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║  exo Android Cross-Compilation            ║" -ForegroundColor Magenta
Write-Host "║  Build Rust bindings for Termux           ║" -ForegroundColor Magenta
Write-Host "╚═══════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

if (-not $Setup -and -not $Build) {
    Write-Host "Usage:"
    Write-Host "  .\cross_compile_android.ps1 -Setup   # First-time setup"
    Write-Host "  .\cross_compile_android.ps1 -Build   # Build wheel"
    Write-Host ""
    exit 0
}

# Find Android NDK
function Find-AndroidNdk {
    $possiblePaths = @(
        $env:ANDROID_NDK_HOME,
        $env:NDK_HOME,
        "$env:LOCALAPPDATA\Android\Sdk\ndk\*",
        "$env:USERPROFILE\AppData\Local\Android\Sdk\ndk\*",
        "C:\Android\ndk\*"
    )

    foreach ($path in $possiblePaths) {
        if ($path -and (Test-Path $path)) {
            $resolved = Get-Item $path | Sort-Object Name -Descending | Select-Object -First 1
            if ($resolved -and (Test-Path "$resolved\toolchains")) {
                return $resolved.FullName
            }
        }
    }
    return $null
}

if ($Setup) {
    Write-Header "Setting Up Cross-Compilation Environment"

    # Check Rust
    Write-Info "Checking Rust installation..."
    try {
        $rustVersion = & rustup --version 2>&1
        Write-Success "Rustup found"
    } catch {
        Write-Error "Rustup not found. Install from https://rustup.rs"
        exit 1
    }

    # Add Android target
    Write-Info "Adding aarch64-linux-android target..."
    & rustup target add aarch64-linux-android
    Write-Success "Added aarch64-linux-android target"

    # Check for maturin
    Write-Info "Checking maturin..."
    try {
        $maturinVersion = & maturin --version 2>&1
        Write-Success "Maturin found: $maturinVersion"
    } catch {
        Write-Info "Installing maturin..."
        & pip install maturin
    }

    # Check for Android NDK
    Write-Info "Checking for Android NDK..."
    $ndk = if ($NdkPath) { $NdkPath } else { Find-AndroidNdk }

    if ($ndk) {
        Write-Success "Found NDK: $ndk"
    } else {
        Write-Warning "Android NDK not found!"
        Write-Info "To install:"
        Write-Host "  1. Install Android Studio"
        Write-Host "  2. SDK Manager -> SDK Tools -> NDK"
        Write-Host "  3. Set ANDROID_NDK_HOME environment variable"
        Write-Host ""
        Write-Host "Or download directly from:"
        Write-Host "  https://developer.android.com/ndk/downloads"
    }

    # Create cargo config for cross-compilation
    $cargoConfigDir = Join-Path $ExoDir ".cargo"
    $cargoConfigFile = Join-Path $cargoConfigDir "config.toml"

    if (-not (Test-Path $cargoConfigDir)) {
        New-Item -ItemType Directory -Path $cargoConfigDir | Out-Null
    }

    if ($ndk) {
        $ndkToolchain = Join-Path $ndk "toolchains\llvm\prebuilt\windows-x86_64\bin"
        $linker = Join-Path $ndkToolchain "aarch64-linux-android24-clang.cmd"

        $cargoConfig = @"
# Auto-generated for Android cross-compilation
# Android NDK: $ndk

[target.aarch64-linux-android]
linker = "$($linker -replace '\\', '\\\\')"

[env]
CC_aarch64-linux-android = "$($ndkToolchain -replace '\\', '\\\\')/aarch64-linux-android24-clang.cmd"
CXX_aarch64-linux-android = "$($ndkToolchain -replace '\\', '\\\\')/aarch64-linux-android24-clang++.cmd"
AR_aarch64-linux-android = "$($ndkToolchain -replace '\\', '\\\\')/llvm-ar.exe"
"@
        $cargoConfig | Out-File -FilePath $cargoConfigFile -Encoding UTF8
        Write-Success "Created .cargo/config.toml"
    }

    Write-Header "Setup Complete"
    Write-Host ""
    Write-Host "Next: .\cross_compile_android.ps1 -Build" -ForegroundColor Green
}

if ($Build) {
    Write-Header "Building for Android aarch64"

    # Verify setup
    $target = "aarch64-linux-android"
    $installedTargets = & rustup target list --installed 2>&1
    if ($installedTargets -notcontains $target) {
        Write-Error "Target $target not installed. Run with -Setup first."
        exit 1
    }

    # Find NDK
    $ndk = if ($NdkPath) { $NdkPath } else { Find-AndroidNdk }
    if (-not $ndk) {
        Write-Error "Android NDK not found. Set ANDROID_NDK_HOME or use -NdkPath"
        exit 1
    }

    # Set environment variables
    $ndkToolchain = Join-Path $ndk "toolchains\llvm\prebuilt\windows-x86_64\bin"
    $env:CC_aarch64_linux_android = Join-Path $ndkToolchain "aarch64-linux-android24-clang.cmd"
    $env:CXX_aarch64_linux_android = Join-Path $ndkToolchain "aarch64-linux-android24-clang++.cmd"
    $env:AR_aarch64_linux_android = Join-Path $ndkToolchain "llvm-ar.exe"
    $env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = $env:CC_aarch64_linux_android

    Write-Info "NDK: $ndk"
    Write-Info "Toolchain: $ndkToolchain"

    # Build with maturin
    Push-Location $RustBindingsDir
    try {
        Write-Info "Building with maturin..."
        & maturin build --release --target $target

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Build successful!"

            # Find the wheel
            $wheelDir = Join-Path $ExoDir "target\wheels"
            $wheel = Get-ChildItem "$wheelDir\*.whl" -ErrorAction SilentlyContinue | 
                     Where-Object { $_.Name -match "android" } |
                     Sort-Object LastWriteTime -Descending | 
                     Select-Object -First 1

            if ($wheel) {
                Write-Header "Output"
                Write-Host "Wheel: $($wheel.FullName)" -ForegroundColor Green
                Write-Host ""
                Write-Host "To push to device:" -ForegroundColor Cyan
                Write-Host "  adb push `"$($wheel.FullName)`" /sdcard/"
                Write-Host ""
                Write-Host "Then in Termux:" -ForegroundColor Cyan
                Write-Host "  pip install /sdcard/$($wheel.Name)"
            }
        } else {
            Write-Error "Build failed"
        }
    } finally {
        Pop-Location
    }
}

