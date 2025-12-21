# LLM Models Guide

> **Model selection, recommendations, and download instructions for exo**

This guide helps you choose the right model for your device or cluster and explains how to download and run models.

---

## Table of Contents

1. [Quick Recommendations](#quick-recommendations)
2. [Understanding Model Requirements](#understanding-model-requirements)
3. [Mobile/Phone Models](#mobilephonemodelsl)
4. [Cluster Models](#cluster-models)
5. [Downloading Models](#downloading-models)
6. [Quantization Guide](#quantization-guide)

---

## Quick Recommendations

### By Device RAM

| RAM | Best Models | Download Command |
|-----|-------------|------------------|
| ≤4GB | `qwen-0.5b`, `tinyllama` | `./scripts/download_model.sh qwen-0.5b` |
| 6GB | `llama-3b`, `qwen-3b` | `./scripts/download_model.sh llama-3b` |
| 8GB | `llama-3b`, `phi-3`, `llama-8b` (tight) | `./scripts/download_model.sh phi-3` |
| 12GB+ | Up to 8B comfortably | `./scripts/download_model.sh llama-8b` |

### By Use Case

| Use Case | Recommended |
|----------|-------------|
| Simple chat, testing | `qwen-0.5b`, `tinyllama` |
| General purpose | `llama-3b`, `qwen-3b` |
| Code generation | `qwen-coder-1.5b`, `qwen-coder-3b` |
| Reasoning | `phi-3`, `deepseek-r1-1.5b` |
| High-end single device | `llama-8b`, `qwen-7b` |
| Small cluster (2-4 devices) | `qwen-14b`, `mistral-24b` |
| Large cluster (8+ devices) | `llama-70b`, `deepseek-v3` |

---

## Understanding Model Requirements

### Check Your Device

```bash
# In Termux
RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
FREE_GB=$(df -BG ~ | awk 'NR==2 {print $4}' | tr -d 'G')
echo "RAM: ${RAM_GB}GB, Free Storage: ${FREE_GB}GB"
```

### Memory Formula

For GGUF models with Q4_K_M quantization:

```
Required RAM ≈ Model Size (GB) × 1.2 + 0.5 GB overhead
Required Storage ≈ Model Size (GB) × 1.1
```

### Memory Requirements (Q4_K_M)

| Model Size | Storage | RAM Required |
|------------|---------|--------------|
| 0.5B | ~400MB | ~1.5GB |
| 1B | ~750MB | ~2GB |
| 1.5B | ~1GB | ~2.5GB |
| 3B | ~2GB | ~4GB |
| 7B | ~4GB | ~6GB |
| 8B | ~4.5GB | ~6.5GB |
| 13-14B | ~8GB | ~10GB |
| 32B | ~20GB | ~25GB |
| 70B | ~42GB | ~50GB |

---

## Mobile/Phone Models

Models optimized for running on single phones or tablets. Focus on GGUF Q4_K_M quantization.

### Ultra-Tiny (< 1B parameters)

| Model | Params | Size | Use Case | Link |
|-------|--------|------|----------|------|
| **Qwen 2.5 0.5B** | 0.5B | ~400MB | Chat, code | [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) |
| **SmolLM2 135M** | 135M | ~100MB | Ultra-light | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| **SmolLM2 360M** | 360M | ~250MB | Edge devices | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| **Danube 3 500M** | 500M | ~350MB | General | [h2oai/h2o-danube3-500m-chat](https://huggingface.co/h2oai/h2o-danube3-500m-chat) |

### Small (1B - 2B)

| Model | Params | Size | Use Case | Link |
|-------|--------|------|----------|------|
| **Llama 3.2 1B** | 1B | ~750MB | General chat | [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |
| **TinyLlama 1.1B** | 1.1B | ~700MB | Edge devices | [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
| **Qwen 2.5 1.5B** | 1.5B | ~1GB | Chat + code | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) |
| **DeepSeek R1 Distill 1.5B** | 1.5B | ~1GB | Reasoning | [bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) |
| **Gemma 2 2B** | 2B | ~1.5GB | General chat | [bartowski/gemma-2-2b-it-GGUF](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF) |

### Medium-Small (3B - 4B)

| Model | Params | Size | Use Case | Link |
|-------|--------|------|----------|------|
| **Llama 3.2 3B** | 3B | ~2GB | Good quality | [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| **Qwen 2.5 3B** | 3B | ~2GB | Chat + code + math | [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) |
| **Phi-3.5 Mini** | 3.8B | ~2.3GB | Reasoning | [bartowski/Phi-3.5-mini-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) |

### Capable Mobile (7B - 9B) — 8GB+ RAM

| Model | Params | Size | Use Case | Link |
|-------|--------|------|----------|------|
| **Llama 3.1 8B** | 8B | ~4.5GB | High-quality | [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| **Qwen 2.5 7B** | 7B | ~4GB | General + code | [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |
| **DeepSeek R1 Distill 8B** | 8B | ~4.5GB | Reasoning | [bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF) |
| **Gemma 2 9B** | 9B | ~5GB | High quality | [bartowski/gemma-2-9b-it-GGUF](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF) |
| **Mistral 7B v0.3** | 7B | ~4GB | Efficient | [bartowski/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF) |

### Specialized: Code Models

| Model | Params | Size | Link |
|-------|--------|------|------|
| **Qwen 2.5 Coder 1.5B** | 1.5B | ~1GB | [Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) |
| **Qwen 2.5 Coder 3B** | 3B | ~2GB | [Qwen/Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF) |
| **Qwen 2.5 Coder 7B** | 7B | ~4GB | [Qwen/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) |
| **CodeGemma 2B** | 2B | ~1.5GB | [bartowski/codegemma-2b-GGUF](https://huggingface.co/bartowski/codegemma-2b-GGUF) |
| **CodeGemma 7B** | 7B | ~4GB | [bartowski/codegemma-7b-it-GGUF](https://huggingface.co/bartowski/codegemma-7b-it-GGUF) |

---

## Cluster Models

Models that benefit from distributed inference across multiple devices.

### Medium-Large (14B - 32B) — 2-4 devices

| Model | Params | Size | Link |
|-------|--------|------|------|
| **Qwen 2.5 14B** | 14B | ~8GB | [Qwen/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF) |
| **DeepSeek R1 Distill 14B** | 14B | ~8GB | [bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF) |
| **Qwen 2.5 32B** | 32B | ~20GB | [Qwen/Qwen2.5-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF) |
| **DeepSeek R1 Distill 32B** | 32B | ~20GB | [bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF) |
| **Gemma 2 27B** | 27B | ~16GB | [bartowski/gemma-2-27b-it-GGUF](https://huggingface.co/bartowski/gemma-2-27b-it-GGUF) |
| **Mistral Small 24B** | 24B | ~14GB | [bartowski/Mistral-Small-24B-Instruct-2501-GGUF](https://huggingface.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF) |

### Large (70B - 72B) — 4-8 devices

| Model | Params | Size | Link |
|-------|--------|------|------|
| **Llama 3.1 70B** | 70B | ~42GB | [bartowski/Meta-Llama-3.1-70B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF) |
| **Llama 3.3 70B** | 70B | ~42GB | [bartowski/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF) |
| **Qwen 2.5 72B** | 72B | ~43GB | [Qwen/Qwen2.5-72B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF) |
| **DeepSeek R1 Distill 70B** | 70B | ~42GB | [bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF) |
| **Qwen 2.5 Coder 32B** | 32B | ~20GB | [Qwen/Qwen2.5-Coder-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF) |

### Mixture-of-Experts (MoE)

| Model | Total/Active | Size | Link |
|-------|--------------|------|------|
| **Mixtral 8x7B** | 47B/12B | ~26GB | [bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF) |
| **Mixtral 8x22B** | 176B/39B | ~90GB | [bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF](https://huggingface.co/bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF) |
| **DeepSeek V2 Lite** | 16B/2.4B | ~10GB | [bartowski/DeepSeek-V2-Lite-Chat-GGUF](https://huggingface.co/bartowski/DeepSeek-V2-Lite-Chat-GGUF) |
| **DeepSeek V2** | 236B | ~140GB | [bartowski/DeepSeek-V2-Chat-GGUF](https://huggingface.co/bartowski/DeepSeek-V2-Chat-GGUF) |

### Frontier-Class (100B+) — 8+ devices

| Model | Params | Size | Link |
|-------|--------|------|------|
| **Llama 3.1 405B** | 405B | ~250GB | [bartowski/Meta-Llama-3.1-405B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-405B-Instruct-GGUF) |
| **DeepSeek V3** | 671B (37B active) | ~378GB | [mlx-community/DeepSeek-V3.1-4bit](https://huggingface.co/mlx-community/DeepSeek-V3.1-4bit) |
| **DeepSeek R1** | 671B | ~378GB | [mlx-community/DeepSeek-R1-4bit](https://huggingface.co/mlx-community/DeepSeek-R1-4bit) |
| **Command R+** | 104B | ~60GB | [bartowski/c4ai-command-r-plus-GGUF](https://huggingface.co/bartowski/c4ai-command-r-plus-GGUF) |

### Vision-Language (Multimodal)

| Model | Params | Use Case | Link |
|-------|--------|----------|------|
| **LLaVA 1.6 Mistral 7B** | 7B | Image + text | [bartowski/llava-v1.6-mistral-7b-hf-GGUF](https://huggingface.co/bartowski/llava-v1.6-mistral-7b-hf-GGUF) |
| **Qwen2-VL 7B** | 7B | Vision-language | [Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4) |
| **Qwen2-VL 72B** | 72B | High-quality VL | [Qwen/Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) |
| **Pixtral 12B** | 12B | Vision | [bartowski/Pixtral-12B-2409-GGUF](https://huggingface.co/bartowski/Pixtral-12B-2409-GGUF) |

---

## Downloading Models

### Method 1: Download Script (Recommended)

```bash
cd ~/exo

# List all models
./scripts/download_model.sh list

# Get personalized recommendation
./scripts/download_model.sh recommend

# Download specific model
./scripts/download_model.sh qwen-0.5b
./scripts/download_model.sh tinyllama
./scripts/download_model.sh llama-3b
./scripts/download_model.sh phi-3
```

### Method 2: Hugging Face Hub (Python)

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from pathlib import Path

repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

model_dir = Path.home() / ".exo" / "models" / repo_id.replace("/", "--")
model_dir.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(model_dir))
print(f"Downloaded to: {path}")
```

### Method 3: Direct wget

```bash
mkdir -p ~/.exo/models && cd ~/.exo/models

# TinyLlama
wget -c https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Qwen 0.5B
wget -c https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Model Storage Location

Models are stored in `~/.exo/models/`.

```bash
# List downloaded models
ls -la ~/.exo/models/

# Find GGUF files
find ~/.exo/models -name "*.gguf"
```

---

## Quantization Guide

### Choosing Quantization Level

| Quant | Size | Quality | Speed | Best For |
|-------|------|---------|-------|----------|
| **Q8_0** | Largest | Best | Slow | Desktop with RAM |
| **Q6_K** | Large | Excellent | Medium | Good RAM devices |
| **Q5_K_M** | Medium | Great | Medium | Balanced |
| **Q4_K_M** | Small | Good | Fast | **Mobile recommended** |
| **Q4_K_S** | Smaller | Acceptable | Faster | Low RAM |
| **Q3_K_M** | Tiny | Fair | Fastest | Very constrained |
| **Q2_K** | Smallest | Poor | Fastest | Emergency only |

**Recommendation:** Use **Q4_K_M** for mobile devices. It provides the best balance of quality and size.

### Cluster Size Requirements

| Model Size | Devices (6GB each) | Devices (8GB each) |
|------------|--------------------|--------------------|
| 7-8B | 2 | 1 |
| 13-14B | 3 | 2 |
| 30B | 5 | 4 |
| 70B | 10 | 8 |
| 235B+ | 25+ | 20+ |

---

## Resources

- **Hugging Face GGUF Models**: [huggingface.co/models?library=gguf](https://huggingface.co/models?library=gguf)
- **bartowski's Collection**: [huggingface.co/bartowski](https://huggingface.co/bartowski)
- **TheBloke's Models**: [huggingface.co/TheBloke](https://huggingface.co/TheBloke)
- **llama.cpp**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

---

*Model recommendations based on Q4_K_M quantization. Last updated: December 2024*

