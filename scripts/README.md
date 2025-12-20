# exo Termux/Android Scripts

These scripts help set up and run exo on Android devices using Termux.

## Prerequisites

1. Install Termux from **F-Droid** (NOT Play Store - that version is outdated):
   - https://f-droid.org/packages/com.termux/

2. Clone this repository:
   ```bash
   pkg install git
   git clone https://github.com/YOUR_USERNAME/exo.git
   cd exo
   ```

## Scripts

### `termux_setup.sh` - Full Setup (Recommended)

Complete setup script with all options and verification:

```bash
chmod +x scripts/termux_setup.sh
./scripts/termux_setup.sh
```

Features:
- Updates Termux packages
- Installs all dependencies
- Builds llama-cpp-python for ARM
- Installs exo
- Optional: Downloads test model
- Verifies installation

### `termux_quick_setup.sh` - Minimal Setup

Faster, minimal setup without prompts:

```bash
chmod +x scripts/termux_quick_setup.sh
./scripts/termux_quick_setup.sh
```

### `termux_verify.sh` - Verify Installation

Check if everything is working:

```bash
chmod +x scripts/termux_verify.sh
./scripts/termux_verify.sh
```

### `download_model.sh` - Download Models

Download GGUF models for inference:

```bash
chmod +x scripts/download_model.sh

# List available models
./scripts/download_model.sh list

# Download a specific model
./scripts/download_model.sh tinyllama   # ~700MB, good for testing
./scripts/download_model.sh qwen-0.5b   # ~400MB, ultra-light
./scripts/download_model.sh llama-1b    # ~750MB, good balance
```

## Available Models

| Name | Size | RAM Needed | Notes |
|------|------|------------|-------|
| tinyllama | ~700MB | 1-2GB | Best for low-memory devices |
| qwen-0.5b | ~400MB | <1GB | Ultra-lightweight |
| qwen-1.5b | ~1GB | 2GB | Light but capable |
| llama-1b | ~750MB | 1-2GB | Good balance |
| llama-3b | ~2GB | 4GB | More capable |
| phi-3 | ~2.3GB | 4-6GB | Strong reasoning |

## Troubleshooting

### llama-cpp-python fails to build

Try with minimal features:
```bash
export CMAKE_ARGS="-DGGML_BLAS=OFF"
pip install llama-cpp-python --no-cache-dir
```

### Out of memory during build

Close other apps and try:
```bash
pkg install -y proot
termux-chroot
# Then run the setup script again
```

### Network issues

If HuggingFace downloads fail:
```bash
# Set a mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com
```

## Multi-Device Cluster

To set up a cluster of Android devices:

1. Run the setup script on each device
2. Ensure all devices are on the same WiFi network
3. Start exo on each device - they will auto-discover each other

## Environment Variables

You can customize behavior with these environment variables:

```bash
# llama.cpp settings
export LLAMA_N_THREADS=4      # Number of CPU threads
export LLAMA_N_CTX=2048       # Context size
export LLAMA_N_BATCH=512      # Batch size

# exo settings
export EXO_HOME=~/.exo        # exo data directory
export EXO_MODELS_DIR=~/.exo/models  # Model storage
```

