# Android/Termux Setup Guide

> **Complete guide to running exo on Android devices**

This guide covers installing and running exo (Distributed AI Inference) on Android devices using Termux.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Downloading Models](#downloading-models)
5. [Running exo](#running-exo)
6. [Cross-Compilation (Optional)](#cross-compilation-optional)
7. [Troubleshooting](#troubleshooting)
8. [Reference](#reference)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Processor | ARM64 (aarch64) | Snapdragon 8 Gen 1+ |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB free | 50+ GB free |
| Android | 7.0+ | 10.0+ |

### Software Requirements

- **Termux** from [F-Droid](https://f-droid.org/packages/com.termux/) (NOT Google Play)
- Stable WiFi connection

---

## Quick Start

For experienced users who want the fastest path to running exo:

```bash
# 1. Install packages
pkg update && pkg upgrade -y
pkg install git python python-pip python-numpy cmake ninja nodejs tur-repo
pkg install rustc-nightly rust-nightly-std-aarch64-linux-android
source $PREFIX/etc/profile.d/rust-nightly.sh

# 2. Build llama.cpp (includes llama-server for EXO)
cd ~
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4

# Verify llama-server exists (REQUIRED)
ls ~/llama.cpp/build/bin/llama-server

# 3. Configure environment
echo 'source $PREFIX/etc/profile.d/rust-nightly.sh' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so' >> ~/.bashrc
source ~/.bashrc

# 4. Install Python packages
pip install llama-cpp-python maturin requests

# 5. Clone and build exo
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo/rust/exo_pyo3_bindings
# Edit pyproject.toml: change requires-python to ">=3.12"
maturin build --release
pip install ~/exo/target/wheels/exo_pyo3_bindings-*.whl

# 6. Install exo and build dashboard
cd ~/exo
pip install -e .
cd dashboard && npm install && npm run build

# 7. Run exo
cd ~/exo && python -m exo
```

**Total time:** ~30-45 minutes

---

## Detailed Setup

### Step 1: Install Termux

1. Download Termux from [F-Droid](https://f-droid.org/packages/com.termux/)
2. Open Termux and grant storage permission:

```bash
termux-setup-storage
```

### Step 2: Install Base Packages

```bash
pkg update && pkg upgrade -y
pkg install git python python-pip python-numpy cmake ninja nodejs
```

> **Important:** Install `python-numpy` from pkg (pre-built) instead of pip. Building numpy from source fails on Termux due to pointer-size detection issues with Android's Bionic libc.

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

# Make it permanent
echo 'source $PREFIX/etc/profile.d/rust-nightly.sh' >> ~/.bashrc
```

### Step 4: Build llama.cpp and llama-server

Build llama.cpp with shared libraries. **Important:** EXO uses `llama-server` on Android for reliable inference via HTTP API.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with shared libraries (includes llama-server)
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4

# Verify llama-cli works
./build/bin/llama-cli --help

# Verify llama-server exists (REQUIRED for EXO on Android)
ls -la ./build/bin/llama-server
```

> **Why llama-server?** Direct subprocess calls have TTY/stdin issues on Android. EXO automatically starts llama-server in the background and communicates via HTTP. This is transparent to users - just select a model and chat!

### Step 5: Configure Environment

Set environment variables for llama-cpp-python:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so' >> ~/.bashrc
source ~/.bashrc

# Install Python packages for llama.cpp integration
pip install llama-cpp-python requests

# Verify
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
python -c "import requests; print('requests OK')"
```

### Step 6: Clone and Build exo

```bash
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo
```

### Step 7: Build Rust Networking Bindings

```bash
# Install maturin (via pip, NOT pkg)
pip install maturin

cd ~/exo/rust/exo_pyo3_bindings

# Edit pyproject.toml to allow Python 3.12 (if needed)
# Change: requires-python = ">=3.13" to requires-python = ">=3.12"
nano pyproject.toml

# Build the wheel (~10-15 minutes)
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
# Check Python packages
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
python -c "import exo_pyo3_bindings; print('exo_pyo3_bindings OK')"
python -c "import requests; print('requests OK')"
python -c "from exo.shared.platform import is_android; print(f'Android: {is_android()}')"

# Check llama-server exists (REQUIRED for EXO on Android)
ls ~/llama.cpp/build/bin/llama-server && echo "llama-server OK"
```

If all checks pass, you're ready to run EXO!

---

## Downloading Models

### Using the Download Script

```bash
cd ~/exo

# List available models
./scripts/download_model.sh list

# Get recommendation based on your RAM
./scripts/download_model.sh recommend

# Download a model
./scripts/download_model.sh tinyllama    # 700MB, needs 2GB RAM
./scripts/download_model.sh qwen-0.5b    # 400MB, needs 1.5GB RAM
./scripts/download_model.sh llama-3b     # 2GB, needs 4GB RAM
```

### Model Recommendations by RAM

| RAM | Recommended Models |
|-----|-------------------|
| ≤4GB | `qwen-0.5b`, `tinyllama`, `llama-1b` |
| 5-6GB | `qwen-1.5b`, `llama-3b`, `phi-3` |
| 7-8GB | `llama-3b`, `phi-3`, `gemma-2b` |
| 8GB+ | `llama-8b`, `qwen-7b` |

For detailed model lists, see [MODELS.md](./MODELS.md).

---

## Running exo

### Basic Usage

```bash
cd ~/exo
python -m exo
```

### Run Modes

```bash
# Full networking (recommended for clusters)
python -m exo

# Local mode (no networking, single device)
EXO_LOCAL_MODE=1 python -m exo
```

### Accessing the Dashboard

The dashboard is available at `http://localhost:52415`.

To access from another device on the same network:

```bash
# Get your device's IP
ip addr show wlan0

# Then open http://<android-ip>:52415 in a browser
```

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `EXO_LOCAL_MODE` | `1`, `true`, `yes` | Use LocalRouter, bypass libp2p |
| `EXO_DISABLE_MDNS` | `1` | Disable mDNS discovery |
| `DASHBOARD_DIR` | Path | Custom dashboard location |
| `EXO_LLAMA_SERVER` | `1`, `true`, `yes` | Force llama-server mode (default on Android) |
| `EXO_LLAMA_SERVER` | `0`, `false`, `no` | Disable llama-server, use subprocess |

### Success Indicators

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

## How EXO Works on Android

### Architecture

On Android, EXO uses **llama-server** for reliable inference:

```
┌─────────────────────────────────────┐
│         EXO Dashboard (Browser)     │
│         http://<ip>:52415           │
└─────────────────┬───────────────────┘
                  │ HTTP/SSE
                  ▼
┌─────────────────────────────────────┐
│         EXO Master/API              │
│    (coordinates, routes requests)   │
└─────────────────┬───────────────────┘
                  │ Events
                  ▼
┌─────────────────────────────────────┐
│         EXO Worker (Android)        │
│    LlamaServerManager               │
│         │                           │
│         ▼                           │
│    llama-server (localhost:8080)    │
│         │                           │
│         ▼                           │
│    GGUF Model (in memory)           │
└─────────────────────────────────────┘
```

### What Happens When You Chat

1. **Select a model** → EXO downloads GGUF file to `~/.exo/models/`
2. **Instance starts** → EXO automatically starts `llama-server` with the model
3. **Send a message** → Worker calls `http://localhost:8080/v1/chat/completions`
4. **Streaming response** → Tokens stream back to dashboard in real-time

### Why llama-server?

Direct subprocess calls (like `llama-cli`) have issues on Android:
- No proper TTY
- stdin/stdout buffering problems
- Interactive mode conflicts

`llama-server` solves all of this:
- ✅ HTTP API - no TTY issues
- ✅ Model stays loaded - faster subsequent requests
- ✅ OpenAI-compatible - proper streaming support
- ✅ Automatic management - EXO starts/stops it for you

### Server Logs

When the server starts, you'll see:

```
Using llama-server HTTP mode (most reliable on Android)
Starting llama-server: llama-server -m model.gguf --port 8080...
llama-server started successfully on port 8080
```

### Switching Models

When you select a different model in the dashboard:
1. EXO stops the current llama-server
2. Downloads the new model (if needed)
3. Starts llama-server with the new model

This happens automatically - just select a model and wait for "READY"!

---

## Cross-Compilation (Optional)

For building on your PC and pushing to Android. See [Cross-Compilation](#cross-compilation-from-pc) for details.

### Requirements

- Rust with nightly toolchain
- Android NDK (r26+)
- maturin

### Quick Steps

```powershell
# Add Android target
rustup target add aarch64-linux-android

# Build
cd rust\exo_pyo3_bindings
maturin build --release --target aarch64-linux-android

# Push to device
adb push target\wheels\exo_pyo3_bindings-*.whl /sdcard/

# In Termux
pip install /sdcard/exo_pyo3_bindings-*.whl
```

### ADB Push for Development

For quick iteration during development:

```powershell
# Push source files
adb push src/exo /sdcard/exo/src/exo
adb push pyproject.toml /sdcard/exo/

# In Termux
cp -r /sdcard/exo ~/exo
cd ~/exo && pip install -e .
```

---

## Troubleshooting

### Rust Nightly Issues

**"cannot find type `Option` in this scope"**

This means Rust stable is being used instead of nightly:

```bash
# Verify current rustc
rustc --version  # Should show "nightly"

# If wrong, source nightly:
source $PREFIX/etc/profile.d/rust-nightly.sh

# If conflicts, remove stable:
pkg remove rust rust-std-aarch64-linux-android

# Clean and rebuild
rm -rf ~/.cargo/registry ~/exo/target
cd ~/exo/rust/exo_pyo3_bindings
maturin build --release
```

### llama-cpp-python Import Fails

```bash
# Check library paths
echo $LD_LIBRARY_PATH
echo $LLAMA_CPP_LIB

# Verify libraries exist
ls -la ~/llama.cpp/build/bin/*.so

# Reinstall if needed
pip uninstall llama-cpp-python
pip install llama-cpp-python
```

### Dashboard 404 Errors

```bash
cd ~/exo/dashboard
rm -rf node_modules dist
npm install
npm run build
```

### Numpy Build Failure

**"Did not find correct pointer sized integer"**

Don't build numpy from source on Termux. Use the pre-built package:

```bash
# Remove any failed numpy install attempts
pip uninstall numpy

# Install pre-built numpy from Termux repos
pkg install python-numpy

# Verify
python -c "import numpy; print(numpy.__version__)"
```

### maturin Not Found

Maturin is a Python package, not a Termux package:

```bash
# Wrong:
pkg install maturin  # This won't work

# Correct:
pip install maturin
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

### Low Memory During Inference

1. Close other apps
2. Use a smaller model (`qwen-0.5b` or `tinyllama`)
3. Run `./scripts/download_model.sh recommend`

### Network Unreachable

```bash
ping -c 3 github.com

# If DNS fails:
echo "nameserver 8.8.8.8" > $PREFIX/etc/resolv.conf
```

### llama-server Not Found

If you see "llama-server not found", build it:

```bash
cd ~/llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4

# Verify
ls ~/llama.cpp/build/bin/llama-server
```

### llama-server Port Already in Use

If you get port conflicts:

```bash
# Kill any existing llama-server processes
pkill -9 -f llama-server

# Or kill by port
pkill -9 -f 'port 8080'
```

### Model Loading Timeout

If the model takes too long to load:
- Use a smaller model (`qwen2.5-0.5b-gguf`)
- Close other apps to free RAM
- Wait longer - first load can take 60+ seconds on slower devices

### Server Not Responding

```bash
# Check if server is running
curl http://localhost:8080/health

# Check server logs
ps aux | grep llama-server
```

### ADB Connection Issues

```powershell
adb devices

# If "unauthorized": check phone for USB debugging prompt

# For wireless debugging:
adb tcpip 5555
adb connect <device-ip>:5555
```

---

## Reference

### Build Times

| Component | Time |
|-----------|------|
| llama.cpp | 5-10 min |
| exo_pyo3_bindings | 10-15 min |
| Dashboard | 2-5 min |
| Python deps | 5-10 min |
| **Total** | **25-40 min** |

### Architecture Support

| Target | Description |
|--------|-------------|
| `aarch64-linux-android` | 64-bit ARM (most 2015+ phones) |
| `armv7-linux-androideabi` | 32-bit ARM (older devices) |

### Build Artifacts

| Artifact | Location |
|----------|----------|
| llama.cpp libraries | `~/llama.cpp/build/bin/*.so` |
| llama-server binary | `~/llama.cpp/build/bin/llama-server` |
| llama-cli binary | `~/llama.cpp/build/bin/llama-cli` |
| Rust bindings wheel | `~/exo/target/wheels/*.whl` |
| Dashboard build | `~/exo/dashboard/dist/` |
| Models | `~/.exo/models/` |

---

## See Also

- [Models Guide](./MODELS.md) - Model selection and download
- [ARM Optimization](./ARM_OPTIMIZATION.md) - CPU-specific optimizations
- [Termux Advanced](./TERMUX_ADVANCED.md) - Advanced Termux topics

---

*Last updated: December 2024*

