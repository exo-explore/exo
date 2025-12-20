# Model Download & Integration Guide for exo

> **A complete guide to selecting, downloading, and running LLM models based on your device capabilities**

This guide helps you choose the right model for your Android device or cluster, understand download options, and integrate models into exo for distributed AI inference.

---

## Table of Contents

1. [Understanding Your Device Limits](#1-understanding-your-device-limits)
2. [Model Selection Guide](#2-model-selection-guide)
3. [Downloading Models](#3-downloading-models)
4. [Single Device vs Cluster](#4-single-device-vs-cluster)
5. [Integration with exo](#5-integration-with-exo)
6. [Quick Reference Tables](#6-quick-reference-tables)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Understanding Your Device Limits

Before selecting a model, you need to know your device's capabilities.

### Check Your Android Device Specs

Run these commands in Termux to check your resources:

```bash
# Check total RAM
free -h

# Check available storage
df -h

# Check CPU info
cat /proc/cpuinfo | grep -E "processor|model name|Hardware"

# Get a summary
echo "=== Device Summary ===" && \
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')" && \
echo "Free Storage: $(df -h ~ | awk 'NR==2 {print $4}')" && \
echo "CPU Cores: $(nproc)"
```

### Memory Requirements Formula

For GGUF models with Q4_K_M quantization:

```
Required RAM ≈ Model Size (GB) × 1.2 + 0.5 GB (overhead)
Required Storage ≈ Model Size (GB) × 1.1
```

### Device Tiers

| Tier | RAM | Storage | Max Model (Single) | Example Devices |
|------|-----|---------|-------------------|-----------------|
| **Low** | 4GB | 32GB | 1-2B | Budget phones, older flagships |
| **Medium** | 6-8GB | 64GB | 3-4B | Mid-range phones, most tablets |
| **High** | 8-12GB | 128GB+ | 7-8B | Flagship phones (2022+) |
| **Ultra** | 12-16GB | 256GB+ | 8-9B (tight) | Gaming phones, high-end tablets |

---

## 2. Model Selection Guide

### Step 1: Determine Your RAM Tier

```bash
# Quick RAM check
RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
echo "Your device has ${RAM_GB}GB RAM"

if [ "$RAM_GB" -le 4 ]; then
    echo "Tier: LOW - Recommended: 0.5B - 1.5B models"
elif [ "$RAM_GB" -le 8 ]; then
    echo "Tier: MEDIUM - Recommended: 1.5B - 4B models"
elif [ "$RAM_GB" -le 12 ]; then
    echo "Tier: HIGH - Recommended: 3B - 8B models"
else
    echo "Tier: ULTRA - Can run up to 9B models (tight fit)"
fi
```

### Step 2: Choose Your Model

#### 📱 For LOW RAM Devices (4GB or less)

| Model | Size | RAM Needed | Quality | Best For |
|-------|------|------------|---------|----------|
| **Qwen 2.5 0.5B** | ~400MB | ~1.5GB | Basic | Simple chat, testing |
| **TinyLlama 1.1B** | ~700MB | ~2GB | Good | General chat |
| **Llama 3.2 1B** | ~750MB | ~2GB | Good | General purpose |
| **Qwen 2.5 1.5B** | ~1GB | ~2.5GB | Better | Chat + basic code |

**Recommended:** Start with `qwen-0.5b` or `tinyllama`

#### 📱 For MEDIUM RAM Devices (6-8GB)

| Model | Size | RAM Needed | Quality | Best For |
|-------|------|------------|---------|----------|
| **Llama 3.2 3B** | ~2GB | ~4GB | Great | General purpose |
| **Qwen 2.5 3B** | ~2GB | ~4GB | Great | Chat + code + math |
| **Phi 3.5 Mini** | ~2.3GB | ~4.5GB | Excellent | Reasoning tasks |

**Recommended:** `llama-3b` or `phi-3` for best balance

#### 📱 For HIGH RAM Devices (8-12GB)

| Model | Size | RAM Needed | Quality | Best For |
|-------|------|------------|---------|----------|
| **Llama 3.1 8B** | ~4.5GB | ~6.5GB | Excellent | High-quality chat |
| **Qwen 2.5 7B** | ~4GB | ~6GB | Excellent | General + code |
| **Mistral 7B** | ~4GB | ~6GB | Excellent | Efficient inference |

**Recommended:** `llama-3.1-8b` for best overall quality

### Step 3: Consider Storage

```bash
# Check free storage
FREE_GB=$(df -BG ~ | awk 'NR==2 {print $4}' | tr -d 'G')
echo "Free storage: ${FREE_GB}GB"

# Model storage requirements
echo ""
echo "Model storage needs:"
echo "  qwen-0.5b:  ~0.5GB"
echo "  tinyllama:  ~0.7GB"
echo "  llama-1b:   ~0.8GB"
echo "  qwen-1.5b:  ~1GB"
echo "  llama-3b:   ~2GB"
echo "  phi-3:      ~2.3GB"
echo "  llama-8b:   ~4.5GB"
```

---

## 3. Downloading Models

### Method 1: Using the Download Script (Recommended)

exo includes a convenient download script:

```bash
# Navigate to exo directory
cd ~/exo

# Make script executable
chmod +x scripts/download_model.sh

# List available models
./scripts/download_model.sh list

# Download a model
./scripts/download_model.sh tinyllama    # ~700MB
./scripts/download_model.sh qwen-0.5b    # ~400MB
./scripts/download_model.sh qwen-1.5b    # ~1GB
./scripts/download_model.sh llama-1b     # ~750MB
./scripts/download_model.sh llama-3b     # ~2GB
./scripts/download_model.sh phi-3        # ~2.3GB
```

### Method 2: Using Python + Hugging Face Hub

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download using Python
python3 << 'EOF'
from huggingface_hub import hf_hub_download
from pathlib import Path

# Choose your model
MODELS = {
    "qwen-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "qwen-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "qwen-3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "llama-1b": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "llama-3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "llama-8b": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    "phi-3": ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf"),
    "tinyllama": ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
}

# Select model
model_key = "tinyllama"  # Change this to your chosen model

repo_id, filename = MODELS[model_key]
model_dir = Path.home() / ".exo" / "models" / repo_id.replace("/", "--")
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {model_key}...")
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=str(model_dir),
    local_dir_use_symlinks=False
)
print(f"Downloaded to: {path}")
EOF
```

### Method 3: Direct wget/curl Download

```bash
# Create models directory
mkdir -p ~/.exo/models

# Example: Download TinyLlama directly
cd ~/.exo/models
wget -c https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Example: Download Qwen 2.5 0.5B
wget -c https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Method 4: Interactive Model Selector Script

Create this script to interactively select and download models:

```bash
cat > ~/select_model.sh << 'SCRIPT'
#!/bin/bash

echo "╔══════════════════════════════════════════════╗"
echo "║       exo Model Selector & Downloader        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check RAM
RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
FREE_STORAGE=$(df -BG ~ | awk 'NR==2 {print $4}' | tr -d 'G')

echo "📱 Device Info:"
echo "   RAM: ${RAM_GB}GB"
echo "   Free Storage: ${FREE_STORAGE}GB"
echo ""

# Recommend tier
if [ "$RAM_GB" -le 4 ]; then
    TIER="LOW"
    echo "📊 Your Tier: LOW (4GB or less RAM)"
    echo "   Recommended models: qwen-0.5b, tinyllama, llama-1b"
elif [ "$RAM_GB" -le 8 ]; then
    TIER="MEDIUM"
    echo "📊 Your Tier: MEDIUM (6-8GB RAM)"
    echo "   Recommended models: llama-3b, qwen-3b, phi-3"
else
    TIER="HIGH"
    echo "📊 Your Tier: HIGH (8GB+ RAM)"
    echo "   Recommended models: llama-8b, qwen-7b"
fi

echo ""
echo "Available Models:"
echo "  [1] qwen-0.5b   - 400MB  (RAM: 1.5GB) - Ultra-light"
echo "  [2] tinyllama   - 700MB  (RAM: 2GB)   - Lightweight classic"
echo "  [3] llama-1b    - 750MB  (RAM: 2GB)   - Good balance"
echo "  [4] qwen-1.5b   - 1GB    (RAM: 2.5GB) - Better quality"
echo "  [5] llama-3b    - 2GB    (RAM: 4GB)   - Great quality"
echo "  [6] qwen-3b     - 2GB    (RAM: 4GB)   - Great + code"
echo "  [7] phi-3       - 2.3GB  (RAM: 4.5GB) - Strong reasoning"
echo "  [8] llama-8b    - 4.5GB  (RAM: 6.5GB) - Excellent (needs 8GB+ RAM)"
echo ""

read -p "Select model [1-8]: " choice

case $choice in
    1) MODEL="qwen-0.5b" ;;
    2) MODEL="tinyllama" ;;
    3) MODEL="llama-1b" ;;
    4) MODEL="qwen-1.5b" ;;
    5) MODEL="llama-3b" ;;
    6) MODEL="qwen-3b" ;;
    7) MODEL="phi-3" ;;
    8) MODEL="llama-8b" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo ""
echo "Downloading $MODEL..."
cd ~/exo 2>/dev/null || cd ~
./scripts/download_model.sh "$MODEL" 2>/dev/null || {
    # Fallback to Python download
    python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path

MODELS = {
    'qwen-0.5b': ('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q4_k_m.gguf'),
    'tinyllama': ('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'),
    'llama-1b': ('bartowski/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'),
    'qwen-1.5b': ('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf'),
    'llama-3b': ('bartowski/Llama-3.2-3B-Instruct-GGUF', 'Llama-3.2-3B-Instruct-Q4_K_M.gguf'),
    'qwen-3b': ('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf'),
    'phi-3': ('bartowski/Phi-3.5-mini-instruct-GGUF', 'Phi-3.5-mini-instruct-Q4_K_M.gguf'),
    'llama-8b': ('bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'),
}

model = '$MODEL'
repo_id, filename = MODELS[model]
model_dir = Path.home() / '.exo' / 'models' / repo_id.replace('/', '--')
model_dir.mkdir(parents=True, exist_ok=True)

print(f'Downloading from {repo_id}...')
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(model_dir))
print(f'✓ Downloaded to: {path}')
"
}

echo ""
echo "✓ Download complete!"
SCRIPT

chmod +x ~/select_model.sh
```

Run with: `~/select_model.sh`

---

## 4. Single Device vs Cluster

### When You Need a Cluster

| Model Size | Parameters | Single Device Needs | Cluster Alternative |
|------------|------------|---------------------|---------------------|
| 0.5-3B | 0.5-3B | 4-6GB RAM | ✗ Not needed |
| 7-8B | 7-8B | 8-12GB RAM | 2 devices × 4GB |
| 13-14B | 13-14B | 14-18GB RAM | 2-3 devices × 6GB |
| 30-32B | 30-32B | 35-40GB RAM | 4-5 devices × 8GB |
| 70B | 70B | 80-90GB RAM | 8-10 devices × 8GB |

### Decision Tree

```
Is your model ≤ 3B parameters?
    └── YES → Run on single device
    └── NO → Continue...

Do you have 8GB+ RAM?
    └── YES → Can run 7-8B on single device
    └── NO → Need cluster for 7B+ models

For 13B+ models:
    └── Always need multiple devices

For 70B+ models:
    └── Need 8+ devices (phones/tablets)
    └── OR 2-4 high-memory Macs/PCs
```

### Cluster Model Requirements

#### 2-Device Cluster (Example: 2 phones with 6GB each)

Can run:
- Llama 3.1 8B (split across 2 devices)
- Qwen 2.5 7B (split across 2 devices)
- Any model up to ~12B total

```bash
# Device 1: Runs layers 0-15
# Device 2: Runs layers 16-31
```

#### 4-Device Cluster (Example: 4 phones with 6GB each)

Can run:
- Models up to ~25B parameters
- Qwen 2.5 14B with room to spare
- Gemma 2 27B (tight)

#### 8-Device Cluster

Can run:
- Llama 3.3 70B (4-bit quantized)
- Qwen 2.5 72B
- Most open-source models

### Cluster Setup Quick Start

```bash
# On EACH device in your cluster:

# 1. Install Termux (from F-Droid)

# 2. Run setup
pkg update && pkg upgrade -y
pkg install python git openssh

# 3. Clone exo
git clone https://github.com/exo-explore/exo
cd exo

# 4. Run setup script
chmod +x scripts/termux_setup.sh
./scripts/termux_setup.sh

# 5. Start exo (it auto-discovers other nodes)
python3 -m exo

# Devices on the same WiFi network will automatically
# find each other and form a cluster!
```

---

## 5. Integration with exo

### Model Path Configuration

exo stores models in `~/.exo/models/`. You can customize this:

```bash
# Default location
~/.exo/models/

# Custom location (set environment variable)
export EXO_MODELS_DIR=/path/to/your/models

# Or on Android with SD card
export EXO_MODELS_DIR=~/storage/external-1/exo_models
```

### Available Model Cards in exo

exo has pre-configured model cards for easy selection:

#### GGUF Models (for Android/Termux)

| Short ID | Model | Size | Use Case |
|----------|-------|------|----------|
| `llama-3.2-1b-gguf` | Llama 3.2 1B | ~750MB | Mobile, edge |
| `llama-3.2-3b-gguf` | Llama 3.2 3B | ~2GB | Mobile |
| `qwen2.5-0.5b-gguf` | Qwen 2.5 0.5B | ~400MB | Ultra-light |
| `qwen2.5-1.5b-gguf` | Qwen 2.5 1.5B | ~1GB | Light |
| `qwen2.5-3b-gguf` | Qwen 2.5 3B | ~2GB | Balanced |
| `phi-3-mini-gguf` | Phi 3.5 Mini | ~2.3GB | Reasoning |
| `tinyllama-1.1b-gguf` | TinyLlama 1.1B | ~700MB | Edge/tiny |

#### MLX Models (for Apple Silicon)

| Short ID | Model | Size | Use Case |
|----------|-------|------|----------|
| `llama-3.1-8b` | Llama 3.1 8B | ~4.4GB | General purpose |
| `llama-3.1-70b` | Llama 3.1 70B | ~38GB | High quality |
| `llama-3.3-70b` | Llama 3.3 70B | ~38GB | Latest Llama |
| `qwen3-30b` | Qwen3 30B | ~17GB | Great reasoning |
| `qwen3-235b-a22b-4bit` | Qwen3 235B | ~132GB | Cluster required |
| `deepseek-v3.1-4bit` | DeepSeek V3.1 | ~378GB | Large cluster |

### Running with a Specific Model

```bash
# Start exo with a specific model
cd ~/exo
python3 -m exo --model llama-3.2-3b-gguf

# For MLX models on Mac
python3 -m exo --model llama-3.1-8b

# For cluster (large model)
python3 -m exo --model llama-3.3-70b
```

### API Usage After Starting exo

Once exo is running, you can use the OpenAI-compatible API:

```bash
# Check available models
curl http://localhost:52415/v1/models

# Chat completion
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 6. Quick Reference Tables

### Model Selection by RAM

| Your RAM | Best Models | Avoid |
|----------|-------------|-------|
| 4GB | qwen-0.5b, tinyllama, llama-1b | Anything > 2B |
| 6GB | llama-3b, qwen-3b, phi-3 | Anything > 4B |
| 8GB | llama-3b, qwen-3b, phi-3, llama-8b (tight) | 13B+ without cluster |
| 12GB | All up to 8B comfortably | 13B+ without cluster |

### Download Commands Quick Reference

```bash
# Ultra-light (< 1GB download)
./scripts/download_model.sh qwen-0.5b    # 400MB
./scripts/download_model.sh tinyllama    # 700MB
./scripts/download_model.sh llama-1b     # 750MB

# Light (1-2GB download)  
./scripts/download_model.sh qwen-1.5b    # 1GB
./scripts/download_model.sh llama-3b     # 2GB
./scripts/download_model.sh qwen-3b      # 2GB

# Medium (2-5GB download)
./scripts/download_model.sh phi-3        # 2.3GB
# For 8B models, use Python download method
```

### Cluster Size by Model

| Model | Min Devices (6GB each) | Min Devices (8GB each) |
|-------|------------------------|------------------------|
| 7-8B | 2 | 1 |
| 13-14B | 3 | 2 |
| 30B | 5 | 4 |
| 70B | 10 | 8 |
| 235B+ | 25+ | 20+ |

---

## 7. Troubleshooting

### Download Issues

```bash
# Clear partial downloads
rm -rf ~/.exo/models/*/*.partial

# Check disk space
df -h ~

# Retry with resume
wget -c <url>  # -c flag resumes partial downloads
```

### Memory Issues During Inference

```bash
# Check current memory usage
free -h

# Kill other processes
pkill -f "heavy_app"

# Use smaller quantization
# Instead of Q8, use Q4_K_M or Q4_K_S
```

### Model Not Found

```bash
# List downloaded models
ls -la ~/.exo/models/

# Check if GGUF file exists
find ~/.exo/models -name "*.gguf"

# Verify model card short_id matches
python3 -c "from exo.shared.models.model_cards import ALL_MODEL_CARDS; print(list(ALL_MODEL_CARDS.keys()))"
```

### Network Issues in Cluster

```bash
# Check if devices can see each other
ping <other_device_ip>

# Check if exo port is open
netstat -tlnp | grep 52415

# Ensure same WiFi network
ip addr show wlan0
```

### Slow Inference

```bash
# Check CPU usage
htop

# Check thermal throttling (if device is hot)
termux-battery-status | jq '.temperature'

# Use smaller/more quantized model
# Switch from Q8 to Q4_K_M for 2x speed improvement
```

---

## Summary: Quick Start

### For Single Phone (4-6GB RAM)

```bash
# 1. Install in Termux
pkg update && pkg install python git
pip install huggingface_hub

# 2. Download small model
cd ~/exo
./scripts/download_model.sh tinyllama

# 3. Run exo
python3 -m exo --model tinyllama-1.1b-gguf
```

### For Single Phone (8GB+ RAM)

```bash
# 1. Setup
cd ~/exo
chmod +x scripts/termux_setup.sh
./scripts/termux_setup.sh

# 2. Download capable model
./scripts/download_model.sh llama-3b

# 3. Run
python3 -m exo --model llama-3.2-3b-gguf
```

### For Phone Cluster (2+ devices)

```bash
# On EACH device:

# 1. Run setup
./scripts/termux_setup.sh

# 2. Ensure same WiFi network

# 3. Start exo (auto-discovers peers)
python3 -m exo

# The cluster will automatically form and can run
# larger models like 7B, 14B, or even 70B depending
# on how many devices you have!
```

---

## Further Reading

- [Mobile & Cluster Models Guide](./MOBILE_AND_CLUSTER_MODELS.md) - Full list of models with links
- [Termux Distributed AI Guide](./TERMUX_DISTRIBUTED_AI_GUIDE.md) - Detailed Termux setup
- [exo GitHub Repository](https://github.com/exo-explore/exo) - Main project

---

*This guide is part of the exo project documentation. Models and recommendations are based on Q4_K_M quantization which provides the best balance of quality and size for mobile devices.*

