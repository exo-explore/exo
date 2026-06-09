# RKLLM Docker Tools

Docker tools for Rockchip RK3588/RK3576 NPU:
- **Converter** (x86_64): Convert HuggingFace models to `.rkllm` format
- **Runtime** (ARM64): Run RKLLM inference on Rockchip devices

## Architecture Overview

| Component | Architecture | Purpose |
|-----------|--------------|---------|
| `Dockerfile` | x86_64 | Model conversion (RKLLM-Toolkit) |
| `Dockerfile.arm64-runtime` | ARM64 | Inference runtime (rkllama server) |

**RKLLM-Toolkit only runs on x86_64 Linux.**

> ⚠️ **Note**: QEMU emulation on ARM64 may be unstable for complex operations. For reliable conversion, run on a native x86_64 machine or use a remote build server.

## Deployment on x86_64 Systems (Recommended)

For native x86_64 Linux systems (Intel/AMD processors), deployment is straightforward:

### Prerequisites

- Docker Engine 20.10+ installed
- At least 16GB RAM (32GB for 7B+ models)
- 50GB+ free disk space for models and cache

### Step 1: Clone and Build

```bash
# Clone the repository
git clone https://github.com/jfreed-dev/exo-rkllama.git
cd exo-rkllama/rkllm-converter

# Build the converter image
docker build -t rkllm-converter .

# Or using docker compose
docker compose build convert
```

### Step 2: Convert Models

```bash
# Using docker run directly
docker run -it --rm \
    -v $(pwd)/output:/workspace/output \
    -v $(pwd)/cache:/workspace/cache \
    rkllm-converter \
    python /workspace/scripts/convert.py \
    -m Qwen/Qwen2.5-1.5B-Instruct \
    -o /workspace/output/qwen2.5-1.5b.rkllm

# Or using docker compose (simpler)
docker compose run convert -m Qwen/Qwen2.5-1.5B-Instruct -o qwen2.5-1.5b.rkllm
```

### Step 3: Deploy to RK3588 Device

```bash
# Copy converted model to your Rockchip device
scp output/qwen2.5-1.5b.rkllm user@rk3588-host:~/RKLLAMA/models/
```

## Quick Reference (docker compose)

```bash
# Build
docker compose build convert

# Convert single model
docker compose run convert -m Qwen/Qwen2.5-1.5B-Instruct -o qwen2.5-1.5b.rkllm

# Show help
docker compose run convert --help

# Batch convert (edit models.yaml first)
docker compose run batch

# Interactive shell
docker compose run shell
```

## Deployment on ARM64 Systems (DGX Spark, Apple Silicon)

ARM64 systems require QEMU emulation to run the x86_64 toolkit:

### Step 1: Enable QEMU Emulation

```bash
# Install QEMU binfmt handlers
docker run --privileged --rm tonistiigi/binfmt --install amd64

# Create buildx builder with multi-platform support
docker buildx create --use --name rkllm-builder
docker buildx inspect --bootstrap
```

### Step 2: Build with Platform Flag

```bash
cd rkllm-converter

# Build for x86_64 using buildx
docker buildx build --platform linux/amd64 -t rkllm-converter --load .
```

### Step 3: Run with Platform Flag

```bash
docker run --platform linux/amd64 -it --rm \
    -v $(pwd)/output:/workspace/output \
    -v $(pwd)/cache:/workspace/cache \
    rkllm-converter \
    python /workspace/scripts/convert.py \
    -m Qwen/Qwen2.5-1.5B-Instruct \
    -o /workspace/output/qwen2.5-1.5b.rkllm
```

> ⚠️ QEMU emulation is significantly slower and may be unstable. For production use, run conversion on a native x86_64 system.

## Directory Structure

```
rkllm-converter/
├── Dockerfile           # Container definition (x86_64)
├── docker-compose.yml   # Service definitions
├── models.yaml          # Batch conversion config
├── scripts/
│   ├── convert.py       # Single model converter
│   └── batch_convert.py # Batch converter
├── models/              # Local model files (optional)
├── output/              # Converted .rkllm files
└── cache/               # HuggingFace download cache
```

## Usage Examples

### Convert from HuggingFace

```bash
# Default settings (w8a8 quantization, rk3588)
docker compose run convert -m Qwen/Qwen2.5-1.5B-Instruct -o qwen2.5-1.5b.rkllm

# With specific quantization
docker compose run convert -m Qwen/Qwen2.5-1.5B-Instruct -o model.rkllm --quant w4a16

# For RK3576 platform
docker compose run convert -m Qwen/Qwen2.5-1.5B-Instruct -o model.rkllm --platform rk3576
```

### Convert Local Model

Place model files in `models/` directory:

```bash
docker compose run convert -m /workspace/models/my-model -o my-model.rkllm
```

### Convert Gated Models

For models requiring authentication (e.g., Llama):

```bash
HF_TOKEN=your_token_here docker compose run convert -m meta-llama/Llama-3.2-1B-Instruct -o llama3.2-1b.rkllm
```

## Quantization Options

| Type | Description | Size | Quality |
|------|-------------|------|---------|
| `w4a16` | 4-bit weights, 16-bit activations | Smallest | Lower |
| `w4a16_g128` | Group-wise 4-bit (group size 128) | Small | Better |
| `w8a8` | 8-bit weights and activations | Medium | **Recommended** |
| `w8a8_g128` | Group-wise 8-bit (group size 128) | Larger | Highest |

## Conversion Options

```
--model, -m       HuggingFace model ID or local path (required)
--output, -o      Output .rkllm file path (required)
--platform, -p    Target platform: rk3588, rk3576 (default: rk3588)
--quant, -q       Quantization type (default: w8a8)
--opt-level       Optimization level 0-2 (default: 1)
--npu-cores       Number of NPU cores 1-3 (default: 3)
--max-context     Maximum context length (default: 4096)
--max-new-tokens  Maximum new tokens (default: 2048)
--dataset         Custom calibration dataset path
--verbose, -v     Enable verbose output
```

## Batch Configuration

Edit `models.yaml`:

```yaml
models:
  - name: qwen2.5-1.5b
    model: Qwen/Qwen2.5-1.5B-Instruct
    output: qwen2.5-1.5b.rkllm
    platform: rk3588
    quant: w8a8
    max_context: 4096
    max_new_tokens: 2048

  - name: deepseek-r1
    model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    output: deepseek-r1-1.5b.rkllm
    platform: rk3588
    quant: w8a8
    max_new_tokens: 8192  # Needs more for chain-of-thought
```

Then run:

```bash
docker compose run batch
```

## Deploying Converted Models

After conversion, copy `.rkllm` files to your RK3588/RK3576 device:

```bash
# From conversion machine
scp output/*.rkllm user@rk3588-device:~/RKLLAMA/models/

# On RK3588 device
ls ~/RKLLAMA/models/
# qwen2.5-1.5b.rkllm

# Start RKLLAMA server
cd /opt/rkllama
python server.py --target_platform rk3588 --port 8080
```

## Troubleshooting

### "exec format error" on ARM64

Ensure QEMU is installed and Docker is configured for multi-platform:

```bash
# Install QEMU (Ubuntu/Debian)
sudo apt-get install qemu-user-static

# Verify platform support
docker run --rm --platform linux/amd64 alpine uname -m
# Should output: x86_64
```

### Download Errors

1. Check internet connectivity
2. For gated models, set `HF_TOKEN` environment variable
3. Increase cache directory space

### Memory Issues

Large models (7B+) require significant RAM during conversion:
- 7B model: ~32GB RAM recommended
- 3B model: ~16GB RAM recommended
- 1.5B model: ~8GB RAM recommended

### Conversion Fails

1. Enable verbose output: `--verbose`
2. Check RKLLM-Toolkit compatibility with your model architecture
3. Verify model uses supported architecture (Qwen, Llama, etc.)

## Supported Models

RKLLM-Toolkit 1.2.3 supports:

- Qwen2.5 series
- DeepSeek-R1 series
- Llama 3.x series
- Phi-3 series
- And more (see [RKNN-LLM releases](https://github.com/airockchip/rknn-llm/releases))

## ARM64 Runtime (for RK3588/RK3576 Devices)

The runtime container runs RKLLM inference directly on ARM64 Rockchip devices.

### Prerequisites

- Rockchip device with NPU (RK3588, RK3576)
- NPU kernel drivers v0.9.8+ installed on host
- Docker Engine on ARM64 Linux

### Build and Run

```bash
# On your RK3588/RK3576 device:
docker compose build runtime
docker compose up runtime

# Or manually:
docker build -f Dockerfile.arm64-runtime -t rkllm-runtime .
docker run --privileged -it --rm \
    -v /dev:/dev \
    -v $(pwd)/output:/opt/rkllama/models \
    -p 8080:8080 \
    rkllm-runtime
```

### Community ARM64 Images

Pre-built ARM64 images are available:

- `hajajmaor/rkllm-docker-image:latest` - ezrknpu-based with RKLLM toolchain
- `ghcr.io/notpunchnox/rkllama:main` - OpenAI-compatible API
- `alefris/rkllm-gradio-adv` - Gradio web interface

```bash
# Example using community image
docker run --privileged -it hajajmaor/rkllm-docker-image:latest
```

## Resources

- [RKNN-LLM Repository](https://github.com/airockchip/rknn-llm)
- [RKLLM-Toolkit Releases](https://github.com/airockchip/rknn-llm/releases)
- [exo-rkllama Documentation](../docs/)
- [RKLLAMA Server](https://github.com/jfreed-dev/rkllama)
- [Collabora RK3588 Notes](https://gitlab.collabora.com/hardware-enablement/rockchip-3588/notes-for-rockchip-3588)
