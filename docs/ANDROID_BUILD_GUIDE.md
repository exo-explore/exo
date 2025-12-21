# Android/Termux Build & Deployment Guide

> **Building and deploying exo to Android devices via Termux**

This guide covers building exo natively on Android and cross-compiling for Android deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Native Termux Build (Recommended)](#native-termux-build-recommended)
3. [Cross-Compilation from PC](#cross-compilation-from-pc)
4. [ADB Push for Development](#adb-push-for-development)
5. [CI/CD for Android Releases](#cicd-for-android-releases)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### What Needs to be Built

| Component | Language | Build Location | Notes |
|-----------|----------|----------------|-------|
| `exo` Python package | Python | Any | Portable, just install |
| `exo_pyo3_bindings` | Rust | Termux (recommended) | Requires Rust nightly |
| `llama-cpp-python` | C++ | Termux (recommended) | Uses pre-built llama.cpp |
| `llama.cpp` | C++ | Termux | Shared libraries for inference |
| Dashboard | TypeScript | Any (npm) | SvelteKit web app |

### Build Options

| Method | Complexity | Best For |
|--------|------------|----------|
| **Native Termux Build** | Medium | Full functionality, recommended |
| **ADB Push Source** | Easy | Quick testing |
| **Cross-compiled Wheels** | Hard | Pre-built distribution |
| **GitHub Release** | Automated | End-user distribution |

---

## Native Termux Build (Recommended)

Building directly on Android ensures full compatibility with the device.

### Prerequisites

- Android device with ARM64 processor
- [Termux](https://f-droid.org/packages/com.termux/) from F-Droid
- At least 5GB free storage
- Stable internet connection

### Step 1: Install Base Packages

```bash
# Update packages
pkg update && pkg upgrade -y

# Install essential tools
pkg install git python python-pip cmake ninja nodejs

# Install TUR repository (has Rust nightly)
pkg install tur-repo

# Install Rust nightly (required for pyo3 async features)
pkg install rustc-nightly rust-nightly-std-aarch64-linux-android

# Activate nightly Rust
source $PREFIX/etc/profile.d/rust-nightly.sh

# Make it permanent
echo 'source $PREFIX/etc/profile.d/rust-nightly.sh' >> ~/.bashrc
```

### Step 2: Build llama.cpp

Build llama.cpp with shared libraries:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure with shared libraries
cmake -B build -DBUILD_SHARED_LIBS=ON

# Build (uses multiple cores)
cmake --build build --config Release -j4

# Verify
ls -la build/bin/*.so
./build/bin/llama-cli --help
```

### Step 3: Configure Environment

```bash
# Set library paths for llama-cpp-python
echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so' >> ~/.bashrc
source ~/.bashrc

# Install llama-cpp-python (uses pre-built library)
pip install llama-cpp-python

# Verify
python -c "from llama_cpp import Llama; print('OK')"
```

### Step 4: Clone exo Repository

```bash
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo
```

### Step 5: Build Rust Bindings

```bash
# Install maturin
pip install maturin

# Navigate to bindings directory
cd ~/exo/rust/exo_pyo3_bindings

# Edit pyproject.toml to allow Python 3.12
# Change: requires-python = ">=3.13" 
# To:     requires-python = ">=3.12"
nano pyproject.toml

# Build the wheel (takes ~10 minutes)
maturin build --release

# Install the wheel
pip install ~/exo/target/wheels/exo_pyo3_bindings-*.whl
```

### Step 6: Install exo

```bash
cd ~/exo
pip install -e .
```

### Step 7: Build Dashboard

```bash
cd ~/exo/dashboard
npm install
npm run build
```

### Step 8: Verify and Run

```bash
# Verify components
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
python -c "import exo_pyo3_bindings; print('exo_pyo3_bindings OK')"

# Run exo
cd ~/exo
python -m exo
```

### Build Times Reference

| Component | Approximate Time |
|-----------|------------------|
| llama.cpp | 5-10 minutes |
| exo_pyo3_bindings | 10-15 minutes |
| Dashboard | 2-5 minutes |
| Python dependencies | 5-10 minutes |
| **Total** | **25-40 minutes** |

---

## Cross-Compilation from PC

Cross-compiling allows building on your PC and pushing pre-built binaries to Android.

### Requirements

1. **Rust** with nightly toolchain
2. **Android NDK** (r26 or later)
3. **maturin** for building Python wheels

### Step 1: Install Prerequisites (Windows)

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

### Step 4: Build Rust Bindings

```powershell
cd rust\exo_pyo3_bindings
maturin build --release --target aarch64-linux-android
```

The wheel will be in `target/wheels/`.

### Step 5: Push and Install

```powershell
# Push to device
adb push target\wheels\exo_pyo3_bindings-*.whl /sdcard/

# In Termux
pip install /sdcard/exo_pyo3_bindings-*.whl
```

> **Note**: Cross-compiling llama-cpp-python is complex. Building on-device is recommended.

---

## ADB Push for Development

For quick development iteration, push source files directly.

### Prerequisites

1. **Android Device** with USB Debugging enabled
2. **Termux** installed from F-Droid
3. **ADB** installed on your PC

### Push Source Files

```powershell
# From the exo directory
adb push src/exo /sdcard/exo/src/exo
adb push scripts /sdcard/exo/scripts
adb push pyproject.toml /sdcard/exo/
adb push dashboard/dist /sdcard/exo/dashboard/dist
```

### Install in Termux

```bash
# Copy from shared storage to Termux home
cp -r /sdcard/exo ~/exo
cd ~/exo

# If dependencies already installed:
pip install -e .

# Run
python -m exo
```

### Sync Script

Create a sync script for repeated pushes:

```powershell
# sync_android.ps1
$device = "your-device-serial"  # From 'adb devices'

adb -s $device push src/exo /sdcard/exo/src/exo
adb -s $device shell "run-as com.termux cp -r /sdcard/exo/src/exo /data/data/com.termux/files/home/exo/src/"

Write-Host "Synced to device"
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

### Rust Nightly Issues

**"cannot find type `Option` in this scope"**

This error means Rust stable is being used instead of nightly:

```bash
# Check current rustc
rustc --version

# Should show "nightly", not "stable"
# If wrong, source the nightly profile:
source $PREFIX/etc/profile.d/rust-nightly.sh

# Verify
rustc --version
# Expected: rustc 1.94.0-nightly (...)

# If conflicts exist, remove stable:
pkg remove rust rust-std-aarch64-linux-android

# Clean and rebuild
rm -rf ~/.cargo/registry
rm -rf ~/exo/target
cd ~/exo/rust/exo_pyo3_bindings
maturin build --release
```

### llama.cpp Build Issues

**"CMake not found"**
```bash
pkg install cmake
```

**Build fails with memory errors**
```bash
# Use fewer parallel jobs
cmake --build build --config Release -j2
```

### llama-cpp-python Issues

**Import fails**
```bash
# Check library paths
echo $LD_LIBRARY_PATH
echo $LLAMA_CPP_LIB

# Libraries should exist
ls -la ~/llama.cpp/build/bin/*.so

# Reinstall
pip uninstall llama-cpp-python
pip install llama-cpp-python
```

### maturin Build Issues

**"requires Python >= 3.13"**

Edit `rust/exo_pyo3_bindings/pyproject.toml`:
```toml
# Change from:
requires-python = ">=3.13"
# To:
requires-python = ">=3.12"
```

**Build takes too long**

Building on Android is slow. Expect 10-15 minutes for the Rust bindings. Consider:
- Closing other apps to free memory
- Keeping device plugged in (prevents CPU throttling)
- Using `--release` flag (slower build but faster runtime)

### Dashboard Issues

**404 errors on dashboard**
```bash
# Rebuild dashboard
cd ~/exo/dashboard
rm -rf node_modules dist
npm install
npm run build
```

**npm not found**
```bash
pkg install nodejs
```

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

### Build Artifacts Location

| Artifact | Location |
|----------|----------|
| llama.cpp libraries | `~/llama.cpp/build/bin/*.so` |
| Rust bindings wheel | `~/exo/target/wheels/*.whl` |
| Dashboard build | `~/exo/dashboard/dist/` |

---

## See Also

- [Android Installation Guide](./ANDROID_INSTALLATION_GUIDE.md) - End-user installation
- [ARM Optimization Guide](./ARM_CORTEX_OPTIMIZATION_GUIDE.md) - CPU optimization
- [Model Download Guide](./MODEL_DOWNLOAD_INTEGRATION_GUIDE.md) - Model management

---

*This guide is part of the exo project documentation.*
*Last updated: December 2024*
