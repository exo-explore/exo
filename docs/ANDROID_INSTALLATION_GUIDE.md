# exo on Android - Complete Installation Guide

> **Successfully tested on Android devices using Termux**

This guide documents the complete process to run exo (Distributed AI Inference Cluster) on Android devices.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Method 1: Native Termux (Recommended)](#installation-method-1-native-termux-recommended)
4. [Installation Method 2: proot-distro Ubuntu](#installation-method-2-proot-distro-ubuntu)
5. [Downloading Models](#downloading-models)
6. [Running exo](#running-exo)
7. [Troubleshooting](#troubleshooting)
8. [Architecture Notes](#architecture-notes)

---

## Overview

### What Works ✅

| Feature | Status |
|---------|--------|
| exo Node | ✅ Working |
| Web Dashboard | ✅ Working |
| API Endpoint | ✅ Working |
| Worker/Master Election | ✅ Working |
| Single-Node Inference | ✅ Working |
| CPU-based Inference | ✅ Working |
| llama.cpp Backend | ✅ Working |
| Multi-Node Networking | ✅ Working (with Rust nightly) |

### Installation Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Native Termux** | Most users | Simpler, faster, full networking | Requires Rust nightly from TUR |
| **proot-distro Ubuntu** | Full glibc compatibility | Standard Linux environment | ~5-10% overhead, CMake issues |

---

## Prerequisites

- **Android Device**: ARM64 (aarch64) processor
- **RAM**: Minimum 4GB recommended (see [model recommendations](#downloading-models))
- **Storage**: At least 5GB free space (more for models)
- **Network**: WiFi connection for installation
- **Termux**: From [F-Droid](https://f-droid.org/packages/com.termux/) (NOT Google Play - that version is outdated)

---

## Installation Method 1: Native Termux (Recommended)

This is the recommended approach using native Termux with full networking support.

### Step 1: Install Termux and Setup Storage

1. Download Termux from [F-Droid](https://f-droid.org/packages/com.termux/)
2. Open Termux and grant storage permission:
   ```bash
   termux-setup-storage
   ```

### Step 2: Install Base Packages

```bash
pkg update && pkg upgrade -y
pkg install git python python-pip cmake ninja nodejs
```

### Step 3: Install Rust Nightly

exo's networking bindings require Rust nightly for `pyo3` async features.

```bash
# Install TUR repository (has nightly Rust)
pkg install tur-repo

# Install Rust nightly
pkg install rustc-nightly rust-nightly-std-aarch64-linux-android

# Activate nightly Rust
source $PREFIX/etc/profile.d/rust-nightly.sh

# Verify nightly is active
rustc --version
# Should show: rustc 1.94.0-nightly or similar
```

**Important**: You must run `source $PREFIX/etc/profile.d/rust-nightly.sh` in every new terminal session, or add it to your `~/.bashrc`:

```bash
echo 'source $PREFIX/etc/profile.d/rust-nightly.sh' >> ~/.bashrc
```

### Step 4: Build llama.cpp

Build llama.cpp with shared libraries for llama-cpp-python:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with shared libraries
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4

# Verify build
./build/bin/llama-cli --help
```

### Step 5: Configure llama-cpp-python

Set environment variables to use the pre-built llama.cpp:

```bash
# Add to ~/.bashrc for persistence
echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so' >> ~/.bashrc
source ~/.bashrc

# Install llama-cpp-python (uses pre-built library)
pip install llama-cpp-python

# Verify
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
```

### Step 6: Clone and Build exo

```bash
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo
```

### Step 7: Build the Rust Networking Bindings

```bash
# Install maturin
pip install maturin

# Build the pyo3 bindings
cd ~/exo/rust/exo_pyo3_bindings

# Edit pyproject.toml to allow Python 3.12 (if needed)
# Change: requires-python = ">=3.13" to requires-python = ">=3.12"
nano pyproject.toml

# Build the wheel
maturin build --release

# Install the wheel
pip install ~/exo/target/wheels/exo_pyo3_bindings-*.whl
```

### Step 8: Install exo

```bash
cd ~/exo
pip install -e .
```

### Step 9: Build the Dashboard

```bash
cd ~/exo/dashboard
npm install
npm run build
```

### Step 10: Verify Installation

```bash
# Test imports
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
python -c "import exo_pyo3_bindings; print('exo_pyo3_bindings OK')"
python -c "from exo.shared.platform import is_android; print(f'Android: {is_android()}')"
```

### Step 11: Run exo

```bash
cd ~/exo
python -m exo
```

The dashboard will be available at `http://<your-android-ip>:52415`

---

## Installation Method 2: proot-distro Ubuntu

Alternative method using proot for full glibc compatibility.

> **Note**: This method has more complexity due to CMake detection issues in proot.

### Step 1: Install Termux and proot-distro

```bash
# In Termux
pkg update && pkg upgrade -y
pkg install proot-distro -y
proot-distro install ubuntu
proot-distro login ubuntu
```

**From now on, all commands are run INSIDE the Ubuntu environment.**

### Step 2: Install Dependencies in Ubuntu

```bash
# Update Ubuntu packages
apt update && apt upgrade -y

# Install Python 3.13 and essential tools
apt install -y \
    python3.13 \
    python3.13-venv \
    python3-pip \
    python3-dev \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev

# Verify Python version
python3.13 --version
```

### Step 3: Install Rust Nightly

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Select option 1 (default installation)

source ~/.cargo/env
rustup default nightly

# Verify
rustc --version
```

### Step 4: Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

### Step 5: Clone and Install exo

```bash
git clone https://github.com/exo-explore/exo.git
cd exo
uv sync
```

This may take 10-15 minutes as it compiles the Rust bindings.

### Step 6: Run exo

```bash
EXO_LOCAL_MODE=1 uv run exo
```

> **Note**: Use `EXO_LOCAL_MODE=1` in proot to bypass libp2p permission issues.

---

## Downloading Models

### Using the Download Script

```bash
# List all available models
./scripts/download_model.sh list

# Get personalized recommendation based on RAM
./scripts/download_model.sh recommend

# Download a specific model
./scripts/download_model.sh <model_name>
```

### Model Recommendations by RAM

| RAM | Recommended Models | Size |
|-----|-------------------|------|
| **≤4GB** | `qwen-0.5b`, `tinyllama`, `llama-1b` | 400-750MB |
| **5-6GB** | `qwen-1.5b`, `llama-3b`, `qwen-3b`, `phi-3` | 1-2.3GB |
| **7-8GB** | `llama-3b`, `phi-3`, `gemma-2b` | 1.5-2.3GB |
| **8GB+** | `llama-8b`, `qwen-7b`, `mistral-7b` | 4-4.5GB |

### Specialized Models

| Model | Size | Use Case |
|-------|------|----------|
| `qwen-coder-1.5b` | ~1GB | Code generation |
| `qwen-coder-3b` | ~2GB | Better code |
| `qwen-coder-7b` | ~4GB | Best code (8GB+ RAM) |
| `deepseek-r1-1.5b` | ~1GB | Reasoning/Chain-of-Thought |
| `deepseek-r1-7b` | ~4GB | Advanced reasoning |

### Model Location

Models are downloaded to `~/.exo/models/`

---

## Running exo

### Basic Usage

```bash
# Native Termux (full networking)
python -m exo

# Native Termux (local mode, no networking)
EXO_LOCAL_MODE=1 python -m exo

# proot-distro Ubuntu (local mode recommended)
EXO_LOCAL_MODE=1 uv run exo
```

### Accessing the Dashboard

The dashboard is accessible at `http://localhost:52415` by default.

To access from another device on the same network:
```bash
# Get your device's IP
hostname -I
# or
ip addr show wlan0

# Then open http://<android-ip>:52415 in a browser
```

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `EXO_LOCAL_MODE` | `1`, `true`, `yes` | Use LocalRouter, bypass libp2p |
| `EXO_DISABLE_MDNS` | `1` (any value) | Disable mDNS discovery |
| `DASHBOARD_DIR` | Path | Custom dashboard location |

---

## Troubleshooting

### Native Termux Issues

#### Rust nightly not found
```bash
# Reinstall and source
pkg install rustc-nightly rust-nightly-std-aarch64-linux-android
source $PREFIX/etc/profile.d/rust-nightly.sh
```

#### "cannot find type `Option` in this scope" during Rust build
This means Rust stable is being used instead of nightly:
```bash
# Remove stable Rust
pkg remove rust rust-std-aarch64-linux-android

# Ensure nightly is sourced
source $PREFIX/etc/profile.d/rust-nightly.sh

# Clean and rebuild
rm -rf ~/.cargo/registry
cd ~/exo/rust/exo_pyo3_bindings
rm -rf target
maturin build --release
```

#### llama-cpp-python import fails
```bash
# Check library path
echo $LD_LIBRARY_PATH
echo $LLAMA_CPP_LIB

# Should point to your llama.cpp build
ls -la ~/llama.cpp/build/bin/*.so

# Re-add to bashrc if missing
echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so' >> ~/.bashrc
source ~/.bashrc
```

#### Dashboard shows 404 errors
```bash
# Rebuild the dashboard
cd ~/exo/dashboard
npm install
npm run build
```

### proot-distro Issues

#### "Permission denied (os error 13)" on startup
**Cause**: libp2p networking trying to bind sockets in proot

**Solution**: Use `EXO_LOCAL_MODE=1`:
```bash
EXO_LOCAL_MODE=1 uv run exo
```

#### CMake "api-level.h not found" error
**Cause**: proot uses Termux's CMake which looks for Android headers

**Solution**: Install CMake inside proot:
```bash
apt install cmake
```

### General Issues

#### Low memory during inference
1. Close other apps
2. Use a smaller model (`qwen-0.5b` or `tinyllama`)
3. Run `./scripts/download_model.sh recommend` for suggestions

#### Network unreachable during installation
```bash
ping -c 3 github.com

# If DNS fails:
echo "nameserver 8.8.8.8" > $PREFIX/etc/resolv.conf
```

---

## Architecture Notes

### Why Native Termux is Recommended

**Native Termux**:
- Uses Termux's native Python and package ecosystem
- Rust nightly available via TUR repository
- llama-cpp-python works well with ARM NEON optimizations
- Full libp2p networking works (multi-node support)
- No syscall emulation overhead

**proot-distro Ubuntu**:
- Provides standard glibc (Termux uses Bionic libc)
- Python 3.13 available in repos
- CMake detection issues (thinks it's Android)
- libp2p has permission issues in proot
- ~5-10% performance penalty

### Performance Considerations

- **CPU Only**: No GPU acceleration in Termux
- **Memory**: Monitor with `free -h`, use appropriate model size
- **Inference Speed**: Depends on device CPU (Snapdragon 8xx recommended)
- **ARM NEON**: llama.cpp uses NEON for optimized ARM inference

### Multi-Node Networking

Native Termux supports full libp2p networking:
- Peer discovery via mDNS
- Distributed inference across multiple devices
- All devices must be on the same network

---

## Quick Start Summary

### Native Termux (Recommended)

```bash
# 1. Install packages
pkg update && pkg upgrade -y
pkg install git python python-pip cmake ninja nodejs tur-repo
pkg install rustc-nightly rust-nightly-std-aarch64-linux-android
source $PREFIX/etc/profile.d/rust-nightly.sh

# 2. Build llama.cpp
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4

# 3. Configure environment
echo 'source $PREFIX/etc/profile.d/rust-nightly.sh' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so' >> ~/.bashrc
source ~/.bashrc

# 4. Install llama-cpp-python
pip install llama-cpp-python maturin

# 5. Clone and build exo
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo/rust/exo_pyo3_bindings
# Edit pyproject.toml: change requires-python to ">=3.12"
maturin build --release
pip install ~/exo/target/wheels/exo_pyo3_bindings-*.whl

# 6. Install exo
cd ~/exo
pip install -e .

# 7. Build dashboard
cd dashboard
npm install
npm run build

# 8. Run
cd ~/exo
python -m exo
```

---

## Success Indicators

When exo is running correctly, you'll see:

```
[ INFO ] Starting EXO
[ INFO ] Starting node 12D3KooW...
[ INFO ] Starting Worker
[ INFO ] Starting Election
[ INFO ] Starting Master
[ INFO ] Node elected Master
[ INFO ] Starting API
[ INFO ] Running on http://0.0.0.0:52415
[ INFO ] Dashboard & API Ready
```

---

*Last updated: December 2024*
*Tested on: Android 13+ with Snapdragon processors, Native Termux with Rust Nightly*
