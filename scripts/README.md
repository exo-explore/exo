# exo Termux/Android Scripts

These scripts help set up and run exo with llama.cpp on Android devices using Termux.

## Prerequisites

1. **Install Termux from F-Droid** (NOT Play Store - that version is outdated):
   - https://f-droid.org/packages/com.termux/

2. **Clone this repository** in Termux:
   ```bash
   pkg install git
   git clone https://github.com/YOUR_USERNAME/exo.git
   cd exo
   ```

## Quick Start

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run the setup script
./scripts/termux_setup.sh

# Download a model
./scripts/download_model.sh tinyllama

# Verify everything works
./scripts/termux_verify.sh
```

## Available Scripts

### `termux_setup.sh` - Full Setup (Recommended)

Complete setup script with verification:

```bash
./scripts/termux_setup.sh
```

**What it does:**
- Updates Termux packages
- Installs Python, build tools, and libraries
- Configures the build environment for ARM
- Builds and installs llama-cpp-python
- Installs exo and dependencies
- Verifies the installation

**Time:** 15-30 minutes (llama-cpp-python compilation takes most of the time)

### `termux_quick_setup.sh` - Minimal Setup

Faster setup with less output:

```bash
./scripts/termux_quick_setup.sh
```

### `termux_verify.sh` - Verify Installation

Check if everything is working:

```bash
./scripts/termux_verify.sh
```

### `termux_troubleshoot.sh` - Fix Common Issues

Diagnose and fix problems:

```bash
./scripts/termux_troubleshoot.sh
```

### `download_model.sh` - Download Models

Download GGUF models for inference:

```bash
# List available models
./scripts/download_model.sh list

# Download a specific model
./scripts/download_model.sh tinyllama   # ~700MB
./scripts/download_model.sh qwen-0.5b   # ~400MB (smallest)
./scripts/download_model.sh llama-3b    # ~2GB (more capable)
```

## Model Recommendations

| Device RAM | Recommended Model | Size | Command |
|------------|------------------|------|---------|
| 2-3GB | qwen-0.5b | ~400MB | `./scripts/download_model.sh qwen-0.5b` |
| 3-4GB | tinyllama | ~700MB | `./scripts/download_model.sh tinyllama` |
| 4-6GB | llama-1b | ~750MB | `./scripts/download_model.sh llama-1b` |
| 6GB+ | llama-3b or phi-3 | ~2GB | `./scripts/download_model.sh llama-3b` |

## Troubleshooting

### "pip install" shows errors

Termux manages pip through its package system. **Never run `pip install --upgrade pip`** in Termux - this breaks things!

Just use pip normally:
```bash
pip install package_name
```

### llama-cpp-python fails to build

Try with minimal build flags:
```bash
export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=OFF"
pip install llama-cpp-python --no-cache-dir
```

If it still fails:
1. Make sure you have enough disk space (2GB+ free)
2. Close other apps to free memory
3. Check build logs: `cat /tmp/llama_build.log`

### Package update errors / mirror issues

Run this to select a working mirror:
```bash
termux-change-repo
```
Select a mirror close to your location.

### "No space left on device"

```bash
# Check space
df -h

# Clear pip cache
pip cache purge

# Clear package cache
pkg clean
```

### Out of memory during build

Close all other apps and try:
```bash
# Set lower parallelism
export CMAKE_BUILD_PARALLEL_LEVEL=1
pip install llama-cpp-python --no-cache-dir
```

## Environment Variables

You can customize behavior with these environment variables:

```bash
# llama.cpp settings (add to ~/.bashrc for persistence)
export LLAMA_N_THREADS=4        # CPU threads (default: auto)
export LLAMA_N_CTX=2048         # Context size (default: 4096)
export LLAMA_N_BATCH=512        # Batch size (default: 512)
export LLAMA_N_GPU_LAYERS=0     # GPU layers (0 = CPU only)

# Build settings
export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON"
export CC=clang
export CXX=clang++

# exo settings
export EXO_HOME=~/.exo
export EXO_MODELS_DIR=~/.exo/models
```

## Multi-Device Cluster

To set up a cluster of Android devices:

1. Run the setup script on each device
2. Ensure all devices are on the same WiFi network
3. Each device needs the same model downloaded
4. Start exo on each device - they will auto-discover each other

**Note:** The Rust networking bindings may need to be compiled for Android ARM. This is a work in progress.

## File Locations

| Path | Purpose |
|------|---------|
| `~/.exo/models/` | Downloaded GGUF models |
| `~/.exo/logs/` | Log files |
| `~/.exo/node_id.keypair` | Node identity (auto-generated) |
| `/tmp/llama_build.log` | Build log for troubleshooting |

## Getting Help

If you're stuck:
1. Run `./scripts/termux_troubleshoot.sh`
2. Check the build log: `cat /tmp/llama_build.log`
3. Make sure you installed Termux from F-Droid
4. Try a fresh Termux installation
