# exo on Android - Complete Installation Guide

> **Successfully tested on Android devices using Termux + proot-distro Ubuntu**

This guide documents the complete process to run exo (Distributed AI Inference Cluster) on Android devices.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Install Termux](#step-1-install-termux)
4. [Step 2: Install proot-distro Ubuntu](#step-2-install-proot-distro-ubuntu)
5. [Step 3: Install Dependencies in Ubuntu](#step-3-install-dependencies-in-ubuntu)
6. [Step 4: Install Rust Nightly](#step-4-install-rust-nightly)
7. [Step 5: Install uv (Python Package Manager)](#step-5-install-uv-python-package-manager)
8. [Step 6: Clone and Install exo](#step-6-clone-and-install-exo)
9. [Step 7: Run exo](#step-7-run-exo)
10. [Troubleshooting](#troubleshooting)
11. [Code Changes Made](#code-changes-made)
12. [Architecture Notes](#architecture-notes)

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

### What Requires Workarounds ⚠️

| Feature | Issue | Solution |
|---------|-------|----------|
| libp2p Networking | Permission denied in proot | Use `EXO_LOCAL_MODE=1` |
| mDNS Discovery | UDP multicast blocked | Use `EXO_DISABLE_MDNS=1` |
| npm in proot | Cache filesystem issues | Build dashboard on host, push via git |

### Architecture

```
Android Device
└── Termux (Terminal Emulator)
    └── proot-distro (Ubuntu 24.04)
        ├── Python 3.13
        ├── Rust Nightly
        ├── uv (Package Manager)
        └── exo
            ├── LocalRouter (bypasses libp2p)
            ├── Worker
            ├── Master
            └── Dashboard (Web UI)
```

---

## Prerequisites

- **Android Device**: ARM64 (aarch64) processor
- **RAM**: Minimum 4GB recommended
- **Storage**: At least 5GB free space
- **Network**: WiFi connection for installation

---

## Step 1: Install Termux

1. **Download Termux** from [F-Droid](https://f-droid.org/packages/com.termux/) (NOT Google Play - that version is outdated)

2. **Open Termux** and grant storage permission:
   ```bash
   termux-setup-storage
   ```

3. **Update packages**:
   ```bash
   pkg update && pkg upgrade -y
   ```

---

## Step 2: Install proot-distro Ubuntu

proot-distro is essential because native Termux lacks glibc and has many compatibility issues.

```bash
# Install proot-distro
pkg install proot-distro -y

# Install Ubuntu
proot-distro install ubuntu

# Login to Ubuntu
proot-distro login ubuntu
```

**From now on, all commands are run INSIDE the Ubuntu environment.**

---

## Step 3: Install Dependencies in Ubuntu

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
    pkg-config \
    libssl-dev \
    libffi-dev

# Verify Python version
python3.13 --version
# Should output: Python 3.13.x
```

---

## Step 4: Install Rust Nightly

exo's networking bindings require Rust nightly for `pyo3` async features.

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# When prompted, select option 1 (default installation)

# Source the environment
source ~/.cargo/env

# Switch to nightly toolchain
rustup default nightly

# Verify installation
rustc --version
# Should output: rustc 1.9x.0-nightly
```

---

## Step 5: Install uv (Python Package Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
source $HOME/.local/bin/env

# Verify installation
uv --version
```

---

## Step 6: Clone and Install exo

```bash
# Clone the repository (use your fork with Android support)
git clone https://github.com/lukewrightmain/exo.git
cd exo

# Install with uv (this builds the Rust bindings)
uv sync

# This may take 5-10 minutes on first run as it:
# - Creates a virtual environment
# - Compiles the Rust exo_pyo3_bindings
# - Installs all Python dependencies
```

### If Rust bindings need rebuilding:

```bash
# Install maturin
pip install maturin

# Navigate to bindings directory
cd rust/exo_pyo3_bindings

# Build and install
maturin develop --release

# Return to main directory
cd ../..
```

---

## Step 7: Run exo

### Run in Local Mode (Recommended for Android)

```bash
EXO_LOCAL_MODE=1 uv run exo
```

This runs exo with:
- **LocalRouter**: Bypasses libp2p networking (which has permission issues in proot)
- **Single-node operation**: The node elects itself as master
- **Full dashboard**: Web UI accessible at `http://localhost:52415`

### Access the Dashboard

1. Find your device's IP address:
   ```bash
   hostname -I
   ```

2. Open a browser on any device on the same network:
   ```
   http://<android-ip>:52415
   ```

### Run Without Dashboard (Minimal Mode)

```bash
EXO_LOCAL_MODE=1 uv run exo --no-api
```

---

## Troubleshooting

### Issue: "Permission denied (os error 13)" on startup

**Cause**: libp2p networking trying to bind sockets in proot

**Solution**: Use `EXO_LOCAL_MODE=1`:
```bash
EXO_LOCAL_MODE=1 uv run exo
```

### Issue: "failed to build behaviour" error

**Cause**: mDNS discovery requires UDP multicast, blocked in proot

**Solution**: Already handled by `EXO_LOCAL_MODE=1`, or manually:
```bash
EXO_DISABLE_MDNS=1 uv run exo
```

### Issue: Dashboard not found

**Cause**: Dashboard needs to be built with npm/Node.js

**Solution**: Build on a host machine and push via git:
```bash
# On Windows/Mac/Linux host:
cd exo/dashboard
npm install
npm run build

# Commit and push the build folder
git add -f dashboard/build
git commit -m "Add pre-built dashboard"
git push

# On Android, pull the changes:
git pull
```

### Issue: npm fails in proot with ENOENT errors

**Cause**: proot filesystem limitations with npm cache

**Solution**: Build dashboard on host machine (see above)

### Issue: "No interpreter found for Python 3.13"

**Solution**: Ensure Python 3.13 is installed:
```bash
apt install python3.13 python3.13-venv
```

### Issue: Rust compilation fails

**Solution**: Ensure nightly toolchain is active:
```bash
rustup default nightly
rustc --version  # Should show "nightly"
```

### Issue: Network unreachable during installation

**Solution**: Check DNS and try again:
```bash
# Test connectivity
ping -c 3 github.com

# If DNS fails, try:
echo "nameserver 8.8.8.8" > /etc/resolv.conf
```

---

## Code Changes Made

The following modifications were made to enable Android/proot support:

### 1. LocalRouter (`src/exo/routing/local_router.py`)

A new router implementation that works without libp2p networking:

```python
# Key features:
- Bypasses NetworkingHandle (Rust libp2p bindings)
- Routes messages locally only
- Enables single-node operation
- Activated via EXO_LOCAL_MODE=1 environment variable
```

### 2. Main Entry Point (`src/exo/main.py`)

Modified to support LocalRouter:

```python
# Added imports
from exo.routing.local_router import LocalRouter

# Added environment variable check
EXO_LOCAL_MODE = os.environ.get("EXO_LOCAL_MODE", "").lower() in ("1", "true", "yes")

# Router selection
if EXO_LOCAL_MODE:
    router = LocalRouter.create()
else:
    router = Router.create(keypair)
```

### 3. mDNS Toggle (`rust/networking/src/discovery.rs`)

Made mDNS discovery optional to handle proot permission issues:

```rust
// Added Toggle wrapper for mDNS
use libp2p::swarm::behaviour::toggle::Toggle;

// Environment variable to disable mDNS
pub const EXO_DISABLE_MDNS_ENV_VAR: &str = "EXO_DISABLE_MDNS";

// Graceful fallback when mDNS fails
let mdns = if mdns_disabled {
    Toggle::from(None)
} else {
    match mdns_behaviour(keypair) {
        Ok(behaviour) => Toggle::from(Some(behaviour)),
        Err(_) => Toggle::from(None)  // Graceful fallback
    }
};
```

---

## Architecture Notes

### Why proot-distro?

Native Termux has limitations:
- Uses Bionic libc (Android's C library) instead of glibc
- Missing `sem_clockwait` and other POSIX functions
- Python 3.13 not available in repos
- Rust nightly not available
- `uv` doesn't recognize Android as a platform

proot-distro provides:
- Full Ubuntu userspace
- Standard glibc
- Access to Ubuntu package repos
- Standard toolchain (gcc, g++, etc.)

### Why LocalRouter?

libp2p in proot has issues:
- TCP socket binding sometimes fails
- UDP multicast for mDNS is blocked
- proot's syscall emulation doesn't fully support all network operations

LocalRouter provides:
- Pure Python implementation
- No native dependencies
- Same interface as the networked Router
- Enables single-node operation

### Performance Considerations

- **CPU Only**: No GPU acceleration in proot
- **Memory**: Monitor with `free -h`, swap if needed
- **Inference Speed**: Depends on device CPU (Snapdragon 8xx recommended)
- **proot Overhead**: ~5-10% performance penalty

---

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `EXO_LOCAL_MODE` | `1`, `true`, `yes` | Use LocalRouter, bypass libp2p |
| `EXO_DISABLE_MDNS` | `1` (any value) | Disable mDNS discovery |
| `DASHBOARD_DIR` | Path | Custom dashboard location |

---

## Quick Start Summary

```bash
# In Termux
pkg install proot-distro
proot-distro install ubuntu
proot-distro login ubuntu

# In Ubuntu (proot)
apt update && apt install -y python3.13 python3.13-venv git curl build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup default nightly
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/lukewrightmain/exo.git
cd exo
uv sync

# Run exo
EXO_LOCAL_MODE=1 uv run exo

# Access dashboard at http://<device-ip>:52415
```

---

## Success Indicators

When exo is running correctly, you'll see:

```
[ INFO ] Starting EXO
[ INFO ] Running in LOCAL MODE - distributed networking disabled
[ WARNING ] Running in LOCAL MODE - no distributed networking
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

## Credits

This guide was developed through extensive testing and debugging to get exo running on Android devices. Key contributions:

- **LocalRouter implementation**: Bypasses libp2p for proot compatibility
- **mDNS Toggle**: Graceful handling of network permission issues
- **proot-distro approach**: Full Linux environment on Android

---

*Last updated: December 2024*
*Tested on: Android 13+ with Snapdragon processors*

