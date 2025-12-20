#!/data/data/com.termux/files/usr/bin/bash
#
# exo Model Download Script
# ==========================
# Download GGUF models for llama.cpp backend.
#
# Usage:
#   chmod +x scripts/download_model.sh
#   ./scripts/download_model.sh [model_name]
#
# Available models:
#   tinyllama   - TinyLlama 1.1B (~700MB) - Best for low-memory devices
#   qwen-0.5b   - Qwen 2.5 0.5B (~400MB) - Ultra-light
#   qwen-1.5b   - Qwen 2.5 1.5B (~1GB) - Light
#   llama-1b    - Llama 3.2 1B (~750MB) - Good balance
#   llama-3b    - Llama 3.2 3B (~2GB) - More capable
#   phi-3       - Phi 3.5 Mini (~2.3GB) - Strong reasoning
#

MODEL=${1:-"tinyllama"}

echo "=== exo Model Downloader ==="
echo ""

download_model() {
    local repo_id=$1
    local filename=$2
    local display_name=$3
    
    echo "Downloading: $display_name"
    echo "Repository: $repo_id"
    echo "File: $filename"
    echo ""
    
    python3 << EOF
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    
    repo_id = "$repo_id"
    filename = "$filename"
    
    # Create model directory
    safe_name = repo_id.replace("/", "--")
    model_dir = Path.home() / ".exo" / "models" / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading to: {model_dir}")
    print("This may take a while depending on your connection...")
    print("")
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False
    )
    
    print("")
    print(f"✓ Download complete!")
    print(f"  Location: {path}")
    
except Exception as e:
    print(f"✗ Download failed: {e}")
    exit(1)
EOF
}

case $MODEL in
    "tinyllama")
        download_model \
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
            "TinyLlama 1.1B (Q4_K_M) - ~700MB"
        ;;
    "qwen-0.5b")
        download_model \
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF" \
            "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 0.5B (Q4_K_M) - ~400MB"
        ;;
    "qwen-1.5b")
        download_model \
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF" \
            "qwen2.5-1.5b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 1.5B (Q4_K_M) - ~1GB"
        ;;
    "llama-1b")
        download_model \
            "bartowski/Llama-3.2-1B-Instruct-GGUF" \
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
            "Llama 3.2 1B (Q4_K_M) - ~750MB"
        ;;
    "llama-3b")
        download_model \
            "bartowski/Llama-3.2-3B-Instruct-GGUF" \
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
            "Llama 3.2 3B (Q4_K_M) - ~2GB"
        ;;
    "phi-3")
        download_model \
            "bartowski/Phi-3.5-mini-instruct-GGUF" \
            "Phi-3.5-mini-instruct-Q4_K_M.gguf" \
            "Phi 3.5 Mini (Q4_K_M) - ~2.3GB"
        ;;
    "list")
        echo "Available models:"
        echo ""
        echo "  tinyllama   - TinyLlama 1.1B (~700MB) - Best for low-memory"
        echo "  qwen-0.5b   - Qwen 2.5 0.5B (~400MB) - Ultra-light"
        echo "  qwen-1.5b   - Qwen 2.5 1.5B (~1GB) - Light"
        echo "  llama-1b    - Llama 3.2 1B (~750MB) - Good balance"
        echo "  llama-3b    - Llama 3.2 3B (~2GB) - More capable"
        echo "  phi-3       - Phi 3.5 Mini (~2.3GB) - Strong reasoning"
        echo ""
        echo "Usage: ./download_model.sh <model_name>"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo ""
        echo "Run './download_model.sh list' to see available models."
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="

