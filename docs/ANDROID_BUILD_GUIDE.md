# Android/Termux Build & Deployment Guide

> **Building and deploying exo to Android devices via Termux**

This guide covers cross-compiling exo for Android and deploying it via ADB for testing.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start: ADB Push for Testing](#quick-start-adb-push-for-testing)
3. [Cross-Compilation Setup](#cross-compilation-setup)
4. [Building Pre-compiled Wheels](#building-pre-compiled-wheels)
5. [CI/CD for Android Releases](#cicd-for-android-releases)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### What Needs to be Built

| Component | Language | Cross-Compile | Notes |
|-----------|----------|---------------|-------|
| `exo` Python package | Python | N/A | Portable, just push source |
| `exo_pyo3_bindings` | Rust | ⚠️ Medium | PyO3 + libp2p networking |
| `llama-cpp-python` | C++ | ⚠️ Hard | External dependency |

### Deployment Options

| Method | Complexity | Best For |
|--------|------------|----------|
| **ADB Push Source** | Easy | Development, testing |
| **Cross-compiled Wheels** | Medium | Pre-release testing |
| **GitHub Release** | Automated | Distribution |
| **Build on Device** | Slow | Guaranteed compatibility |

---

## Quick Start: ADB Push for Testing

The fastest way to test on Android is pushing the Python source directly.

### Prerequisites

1. **Android Device** with USB Debugging enabled
2. **Termux** installed from [F-Droid](https://f-droid.org/packages/com.termux/)
3. **ADB** (Android Debug Bridge) installed on Windows

### Step 1: Push Source to Device

```powershell
# From the exo directory
.\scripts\adb_push.ps1
```

Or manually:

```powershell
# Push source files
adb push src/exo /sdcard/exo/src/exo
adb push scripts /sdcard/exo/scripts
adb push pyproject.toml /sdcard/exo/
```

### Step 2: Install in Termux

In Termux on your Android device:

```bash
# Copy from shared storage to Termux home
cp -r /sdcard/exo ~/exo
cd ~/exo

# Run the full setup (includes llama-cpp-python build)
chmod +x scripts/termux_setup.sh
./scripts/termux_setup.sh
```

### Step 3: Test

```bash
# Test the import
python3 -c "from exo.shared.platform import is_android; print(f'Android: {is_android()}')"

# Run exo
python3 -m exo
```

---

## Cross-Compilation Setup

Cross-compiling allows building on your PC and pushing pre-built binaries.

### Requirements

1. **Rust** with nightly toolchain
2. **Android NDK** (r26 or later)
3. **maturin** for building Python wheels

### Step 1: Install Prerequisites

```powershell
# Add Rust Android target
rustup target add aarch64-linux-android

# Install maturin
pip install maturin
```

### Step 2: Install Android NDK

**Option A: Via Android Studio**
1. Open Android Studio → SDK Manager
2. SDK Tools → NDK (Side by side)
3. Install the latest version

**Option B: Direct Download**
Download from: https://developer.android.com/ndk/downloads

Set environment variable:
```powershell
$env:ANDROID_NDK_HOME = "C:\Users\<user>\AppData\Local\Android\Sdk\ndk\<version>"
```

### Step 3: Configure Cargo

Create `.cargo/config.toml` in the exo root:

```toml
[target.aarch64-linux-android]
linker = "C:\\Users\\<user>\\AppData\\Local\\Android\\Sdk\\ndk\\<version>\\toolchains\\llvm\\prebuilt\\windows-x86_64\\bin\\aarch64-linux-android24-clang.cmd"

[env]
CC_aarch64_linux_android = "..."  # Same path with -clang.cmd
CXX_aarch64_linux_android = "..." # Same path with -clang++.cmd
```

Or use the setup script:
```powershell
.\scripts\cross_compile_android.ps1 -Setup
```

---

## Building Pre-compiled Wheels

### Build Rust Bindings

```powershell
.\scripts\cross_compile_android.ps1 -Build
```

Or manually:

```powershell
cd rust\exo_pyo3_bindings
maturin build --release --target aarch64-linux-android
```

The wheel will be in `target/wheels/`.

### Push and Install Wheel

```powershell
# Push to device
adb push target\wheels\exo_pyo3_bindings-*.whl /sdcard/

# In Termux
pip install /sdcard/exo_pyo3_bindings-*.whl
```

### Building llama-cpp-python

Cross-compiling llama-cpp-python is more complex. Options:

**Option 1: Build on Device (Recommended)**
```bash
# In Termux - takes 10-20 minutes
pip install llama-cpp-python
```

**Option 2: Pre-built Wheel from CI**
Check GitHub releases for pre-built wheels.

**Option 3: Cross-compile with Docker**
```bash
# Use a Linux container with Android NDK
docker run --rm -v $(pwd):/work -w /work \
  ghcr.io/pyo3/maturin:latest \
  maturin build --target aarch64-linux-android
```

---

## CI/CD for Android Releases

The `.github/workflows/android-build.yml` workflow automates Android builds.

### Triggering a Build

**Manual trigger:**
1. Go to Actions → android-build
2. Click "Run workflow"

**Tag trigger:**
```bash
git tag android-v0.1.0
git push origin android-v0.1.0
```

### Release Artifacts

The workflow produces:
- `exo_pyo3_bindings-*.whl` - Pre-built Rust bindings
- `exo-android-*.tar.gz` - Complete Android package

### Installing from Release

```bash
# In Termux
wget https://github.com/exo-explore/exo/releases/download/android-v0.1.0/exo-android-*.tar.gz
tar -xzf exo-android-*.tar.gz
./install_android.sh
```

---

## Troubleshooting

### ADB Connection Issues

```powershell
# Check connected devices
adb devices

# If device shows "unauthorized"
# → Check phone for USB debugging prompt

# For wireless debugging
adb tcpip 5555
adb connect <device-ip>:5555
```

### Cross-Compilation Errors

**"linker not found"**
- Verify `ANDROID_NDK_HOME` is set
- Check path in `.cargo/config.toml`

**"libstdc++ not found"**
- Use NDK's libc++ instead
- Add to CMAKE_ARGS: `-DCMAKE_CXX_FLAGS="-stdlib=libc++"`

### Termux Issues

**"pip install fails"**
```bash
# Don't upgrade pip in Termux!
# Use the system pip directly
pip install package_name
```

**"llama-cpp-python build fails"**
```bash
# Check build dependencies
pkg install cmake clang make

# Try with minimal features
export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=OFF"
pip install llama-cpp-python --no-cache-dir
```

### Permission Denied

```bash
# In Termux, ensure scripts are executable
chmod +x scripts/*.sh

# For /sdcard access
termux-setup-storage
```

---

## Architecture Reference

### Supported Targets

| Target | Description | Common Devices |
|--------|-------------|----------------|
| `aarch64-linux-android` | 64-bit ARM | Most 2015+ phones |
| `armv7-linux-androideabi` | 32-bit ARM | Older devices |
| `x86_64-linux-android` | 64-bit x86 | Emulators |

### API Level Considerations

| API Level | Android Version | Notes |
|-----------|-----------------|-------|
| 24 (Recommended) | Android 7.0+ | Good compatibility |
| 28 | Android 9.0+ | Better crypto support |
| 33 | Android 13+ | Latest features |

---

## See Also

- [Termux Setup Guide](./TERMUX_DISTRIBUTED_AI_GUIDE.md)
- [ARM Optimization Guide](./ARM_CORTEX_OPTIMIZATION_GUIDE.md)
- [Model Download Guide](./MODEL_DOWNLOAD_INTEGRATION_GUIDE.md)

---

*This guide is part of the exo project documentation.*

