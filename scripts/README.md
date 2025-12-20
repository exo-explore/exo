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

# Get personalized recommendation based on your device
./scripts/download_model.sh recommend

# Download a specific model
./scripts/download_model.sh tinyllama   # ~700MB
./scripts/download_model.sh qwen-0.5b   # ~400MB (smallest)
./scripts/download_model.sh llama-3b    # ~2GB (more capable)
```

### `arm_detect.sh` - ARM CPU Detection & Optimization

Detect ARM CPU cores, features, and get optimal compiler flags:

```bash
# Full detection report
./scripts/arm_detect.sh

# Get only compiler flags
./scripts/arm_detect.sh --flags

# Output as JSON (for scripting)
./scripts/arm_detect.sh --json

# Export as environment variables
source ./scripts/arm_detect.sh --env
```

**What it detects:**
- CPU cores (Cortex-X4, A720, A55, etc.)
- Architecture features (NEON, dotprod, FP16, SVE2, I8MM)
- Optimal `-mcpu` and `-march` compiler flags
- Device tier based on RAM

### `thermal_monitor.sh` - Temperature Management

Monitor device temperature and throttle inference when too hot:

```bash
# Run in foreground (press Ctrl+C to stop)
./scripts/thermal_monitor.sh

# Run as background daemon
./scripts/thermal_monitor.sh --daemon

# Check current status
./scripts/thermal_monitor.sh --status

# Stop background daemon
./scripts/thermal_monitor.sh --stop
```

**What it does:**
- Monitors battery/thermal zone temperature
- Pauses exo/llama processes when device exceeds 42°C
- Resumes when cooled below 38°C
- Sends Android notifications on state changes

### `termux_boot_cluster.sh` - Auto-Start on Boot

Start exo cluster node automatically when device boots:

```bash
# Install boot script
mkdir -p ~/.termux/boot
cp scripts/termux_boot_cluster.sh ~/.termux/boot/01-exo-cluster.sh
chmod +x ~/.termux/boot/01-exo-cluster.sh
```

**Requirements:**
- Install **Termux:Boot** from F-Droid
- Open Termux:Boot once to register with Android
- Disable battery optimization for Termux and Termux:Boot

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

## ARM Optimization

exo automatically detects your ARM CPU and optimizes compilation. For best performance:

### Automatic Optimization

The setup script automatically:
1. Detects your CPU cores (Cortex-X4, A720, A55, etc.)
2. Identifies available extensions (dotprod, FP16, SVE2, I8MM)
3. Compiles llama-cpp-python with optimal flags

### Manual Optimization

If you want to manually optimize:

```bash
# Detect your device's optimal flags
./scripts/arm_detect.sh --flags

# Example output: -O3 -mcpu=cortex-x4 -march=armv9.2-a+sve2+i8mm+bf16 -flto

# Set and rebuild
source ./scripts/arm_detect.sh --env
pip uninstall llama-cpp-python
pip install llama-cpp-python --no-cache-dir
```

### Key ARM Features for LLM Inference

| Feature | Benefit | Check |
|---------|---------|-------|
| **dotprod** | 2-4x speedup for quantized models | `grep asimddp /proc/cpuinfo` |
| **fp16** | 2x throughput for FP16 | `grep fphp /proc/cpuinfo` |
| **i8mm** | 2-4x speedup for INT8 | `grep i8mm /proc/cpuinfo` |
| **sve2** | Better vectorization | `grep sve2 /proc/cpuinfo` |

### Performance Tips

1. **Use Q4_K_M quantization** - best quality/speed balance for mobile
2. **Set thread count to big core count** (usually 4 on modern phones)
3. **Keep device plugged in** for sustained performance
4. **Use thermal monitoring** to prevent throttling
5. **Close other apps** before running inference

## Environment Variables

You can customize behavior with these environment variables:

```bash
# llama.cpp settings (add to ~/.bashrc for persistence)
export LLAMA_N_THREADS=4        # CPU threads (default: auto)
export LLAMA_N_CTX=2048         # Context size (default: 4096)
export LLAMA_N_BATCH=512        # Batch size (default: 512)
export LLAMA_N_GPU_LAYERS=0     # GPU layers (0 = CPU only)

# Build settings (auto-detected by arm_detect.sh)
export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON"
export CC=clang
export CXX=clang++

# ARM optimization flags (auto-detected)
# source ./scripts/arm_detect.sh --env  # Sets these automatically
export EXO_ARM_MCPU="cortex-x4"
export EXO_ARM_MARCH="armv9.2-a+sve2+i8mm"
export CFLAGS="-O3 -mcpu=$EXO_ARM_MCPU -march=$EXO_ARM_MARCH -flto"

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
