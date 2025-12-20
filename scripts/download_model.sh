#!/data/data/com.termux/files/usr/bin/bash
#
# exo Model Download Script
# ==========================
# Download GGUF models for llama.cpp backend on Android/Termux.
#
# Usage:
#   chmod +x scripts/download_model.sh
#   ./scripts/download_model.sh [model_name]
#   ./scripts/download_model.sh list           # Show all available models
#   ./scripts/download_model.sh recommend      # Get recommendation based on device
#
# Models are organized by RAM requirements:
#   LOW (4GB):     qwen-0.5b, tinyllama, llama-1b
#   MEDIUM (6GB):  qwen-1.5b, qwen-3b, llama-3b, phi-3
#   HIGH (8GB+):   llama-8b, qwen-7b, mistral-7b, gemma-2b, gemma-9b
#   CLUSTER:       Models requiring multiple devices
#

set -e

MODEL=${1:-"list"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         exo Model Downloader v2.0            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

download_model() {
    local repo_id=$1
    local filename=$2
    local display_name=$3
    
    echo -e "${BLUE}Downloading:${NC} $display_name"
    echo -e "${BLUE}Repository:${NC} $repo_id"
    echo -e "${BLUE}File:${NC} $filename"
    echo ""
    
    python3 << EOF
import os
import sys
import time
from pathlib import Path

MAX_RETRIES = 3
RETRY_DELAY = 5

def download_with_retry():
    from huggingface_hub import hf_hub_download
    
    repo_id = "$repo_id"
    filename = "$filename"
    
    safe_name = repo_id.replace("/", "--")
    model_dir = Path.home() / ".exo" / "models" / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Downloading to: {model_dir}")
    print("⏳ This may take a while depending on your connection...")
    print("")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,  # Resume if interrupted
            )
            
            print("")
            print(f"✅ Download complete!")
            print(f"📍 Location: {path}")
            return True
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Network errors - retry
            if any(x in error_str for x in ["network", "unreachable", "connection", "timeout", "closed"]):
                if attempt < MAX_RETRIES:
                    print(f"⚠️  Network error (attempt {attempt}/{MAX_RETRIES}): {e}")
                    print(f"   Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    continue
            
            # Fatal error or last attempt
            print(f"❌ Download failed: {e}")
            return False
    
    return False

try:
    from huggingface_hub import hf_hub_download
    
    if not download_with_retry():
        print("")
        print("💡 Tips:")
        print("   - Check your internet connection")
        print("   - Try running the command again (downloads resume)")
        print("   - If using mobile data, try WiFi instead")
        sys.exit(1)
    
except ImportError:
    print("❌ huggingface_hub not installed.")
    print("   Run: pip install huggingface_hub")
    sys.exit(1)
EOF
}

show_recommendation() {
    echo -e "${YELLOW}📱 Checking your device...${NC}"
    echo ""
    
    RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
    RAM_GB=$((RAM_KB / 1024 / 1024))
    
    if [ "$RAM_GB" -eq 0 ]; then
        RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "4")
    fi
    
    FREE_STORAGE=$(df -BG ~ 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G' || echo "10")
    
    echo -e "   RAM: ${GREEN}${RAM_GB}GB${NC}"
    echo -e "   Free Storage: ${GREEN}${FREE_STORAGE}GB${NC}"
    echo ""
    
    if [ "$RAM_GB" -le 4 ]; then
        echo -e "${YELLOW}📊 Your Tier: LOW (≤4GB RAM)${NC}"
        echo ""
        echo "Recommended models:"
        echo -e "  ${GREEN}qwen-0.5b${NC}   - 400MB  - Ultra-light, fast"
        echo -e "  ${GREEN}tinyllama${NC}   - 700MB  - Good quality for size"
        echo -e "  ${GREEN}llama-1b${NC}    - 750MB  - Latest Llama, good balance"
        echo ""
        echo -e "Best choice: ${CYAN}./download_model.sh qwen-0.5b${NC}"
    elif [ "$RAM_GB" -le 6 ]; then
        echo -e "${YELLOW}📊 Your Tier: MEDIUM (5-6GB RAM)${NC}"
        echo ""
        echo "Recommended models:"
        echo -e "  ${GREEN}qwen-1.5b${NC}   - 1GB    - Good quality"
        echo -e "  ${GREEN}llama-3b${NC}    - 2GB    - Great balance"
        echo -e "  ${GREEN}qwen-3b${NC}     - 2GB    - Strong reasoning"
        echo ""
        echo -e "Best choice: ${CYAN}./download_model.sh llama-3b${NC}"
    elif [ "$RAM_GB" -le 8 ]; then
        echo -e "${YELLOW}📊 Your Tier: HIGH (7-8GB RAM)${NC}"
        echo ""
        echo "Recommended models:"
        echo -e "  ${GREEN}llama-3b${NC}    - 2GB    - Runs great"
        echo -e "  ${GREEN}phi-3${NC}       - 2.3GB  - Strong reasoning"
        echo -e "  ${GREEN}gemma-2b${NC}    - 1.5GB  - Google quality"
        echo ""
        echo -e "Best choice: ${CYAN}./download_model.sh phi-3${NC}"
    else
        echo -e "${YELLOW}📊 Your Tier: ULTRA (>8GB RAM)${NC}"
        echo ""
        echo "Recommended models:"
        echo -e "  ${GREEN}llama-8b${NC}    - 4.5GB  - Excellent quality"
        echo -e "  ${GREEN}qwen-7b${NC}     - 4GB    - Great + code"
        echo -e "  ${GREEN}mistral-7b${NC}  - 4GB    - Efficient"
        echo ""
        echo -e "Best choice: ${CYAN}./download_model.sh llama-8b${NC}"
    fi
    echo ""
}

show_list() {
    echo -e "${YELLOW}Available Models:${NC}"
    echo ""
    echo -e "${GREEN}═══ LOW RAM (4GB) ═══${NC}"
    echo "  qwen-0.5b     Qwen 2.5 0.5B     ~400MB   Ultra-light"
    echo "  tinyllama     TinyLlama 1.1B    ~700MB   Classic lightweight"
    echo "  llama-1b      Llama 3.2 1B      ~750MB   Latest Llama, small"
    echo ""
    echo -e "${GREEN}═══ MEDIUM RAM (6GB) ═══${NC}"
    echo "  qwen-1.5b     Qwen 2.5 1.5B     ~1GB     Good quality"
    echo "  gemma-2b      Gemma 2 2B        ~1.5GB   Google model"
    echo "  llama-3b      Llama 3.2 3B      ~2GB     Great balance"
    echo "  qwen-3b       Qwen 2.5 3B       ~2GB     Strong reasoning"
    echo "  phi-3         Phi 3.5 Mini      ~2.3GB   Microsoft, reasoning"
    echo ""
    echo -e "${GREEN}═══ HIGH RAM (8GB+) ═══${NC}"
    echo "  qwen-7b       Qwen 2.5 7B       ~4GB     Excellent quality"
    echo "  mistral-7b    Mistral 7B v0.3   ~4GB     Fast, efficient"
    echo "  llama-8b      Llama 3.1 8B      ~4.5GB   Top quality"
    echo "  gemma-9b      Gemma 2 9B        ~5.5GB   Google, high quality"
    echo ""
    echo -e "${GREEN}═══ SPECIALIZED ═══${NC}"
    echo "  qwen-coder-1.5b  Qwen Coder 1.5B   ~1GB   Code generation"
    echo "  qwen-coder-3b    Qwen Coder 3B     ~2GB   Better code"
    echo "  qwen-coder-7b    Qwen Coder 7B     ~4GB   Best code (8GB+ RAM)"
    echo "  deepseek-r1-1.5b DeepSeek R1 1.5B  ~1GB   Reasoning/CoT"
    echo "  deepseek-r1-7b   DeepSeek R1 7B    ~4GB   Advanced reasoning"
    echo ""
    echo -e "${YELLOW}Usage:${NC} ./download_model.sh <model_name>"
    echo -e "${YELLOW}Help:${NC}  ./download_model.sh recommend   # Get personalized recommendation"
    echo ""
}

case $MODEL in
    # === LOW RAM MODELS ===
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
    "llama-1b")
        download_model \
            "bartowski/Llama-3.2-1B-Instruct-GGUF" \
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
            "Llama 3.2 1B (Q4_K_M) - ~750MB"
        ;;
    
    # === MEDIUM RAM MODELS ===
    "qwen-1.5b")
        download_model \
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF" \
            "qwen2.5-1.5b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 1.5B (Q4_K_M) - ~1GB"
        ;;
    "gemma-2b")
        download_model \
            "bartowski/gemma-2-2b-it-GGUF" \
            "gemma-2-2b-it-Q4_K_M.gguf" \
            "Gemma 2 2B (Q4_K_M) - ~1.5GB"
        ;;
    "llama-3b")
        download_model \
            "bartowski/Llama-3.2-3B-Instruct-GGUF" \
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
            "Llama 3.2 3B (Q4_K_M) - ~2GB"
        ;;
    "qwen-3b")
        download_model \
            "Qwen/Qwen2.5-3B-Instruct-GGUF" \
            "qwen2.5-3b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 3B (Q4_K_M) - ~2GB"
        ;;
    "phi-3")
        download_model \
            "bartowski/Phi-3.5-mini-instruct-GGUF" \
            "Phi-3.5-mini-instruct-Q4_K_M.gguf" \
            "Phi 3.5 Mini (Q4_K_M) - ~2.3GB"
        ;;
    
    # === HIGH RAM MODELS ===
    "qwen-7b")
        download_model \
            "Qwen/Qwen2.5-7B-Instruct-GGUF" \
            "qwen2.5-7b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 7B (Q4_K_M) - ~4GB"
        ;;
    "mistral-7b")
        download_model \
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF" \
            "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
            "Mistral 7B v0.3 (Q4_K_M) - ~4GB"
        ;;
    "llama-8b")
        download_model \
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" \
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
            "Llama 3.1 8B (Q4_K_M) - ~4.5GB"
        ;;
    "gemma-9b")
        download_model \
            "bartowski/gemma-2-9b-it-GGUF" \
            "gemma-2-9b-it-Q4_K_M.gguf" \
            "Gemma 2 9B (Q4_K_M) - ~5.5GB"
        ;;
    
    # === SPECIALIZED: CODE MODELS ===
    "qwen-coder-1.5b")
        download_model \
            "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF" \
            "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 Coder 1.5B (Q4_K_M) - ~1GB"
        ;;
    "qwen-coder-3b")
        download_model \
            "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF" \
            "qwen2.5-coder-3b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 Coder 3B (Q4_K_M) - ~2GB"
        ;;
    "qwen-coder-7b")
        download_model \
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF" \
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf" \
            "Qwen 2.5 Coder 7B (Q4_K_M) - ~4GB"
        ;;
    
    # === SPECIALIZED: REASONING MODELS ===
    "deepseek-r1-1.5b")
        download_model \
            "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF" \
            "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf" \
            "DeepSeek R1 Distill 1.5B (Q4_K_M) - ~1GB"
        ;;
    "deepseek-r1-7b")
        download_model \
            "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF" \
            "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf" \
            "DeepSeek R1 Distill 7B (Q4_K_M) - ~4GB"
        ;;
    "deepseek-r1-8b")
        download_model \
            "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF" \
            "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf" \
            "DeepSeek R1 Distill 8B (Q4_K_M) - ~4.5GB"
        ;;
    
    # === COMMANDS ===
    "list")
        show_list
        ;;
    "recommend")
        show_recommendation
        ;;
    "help"|"-h"|"--help")
        echo "Usage: ./download_model.sh <model_name>"
        echo ""
        echo "Commands:"
        echo "  list       Show all available models"
        echo "  recommend  Get personalized recommendation based on device"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./download_model.sh tinyllama    # Download TinyLlama 1.1B"
        echo "  ./download_model.sh recommend    # Get recommendation"
        echo ""
        ;;
    *)
        echo -e "${RED}Unknown model: $MODEL${NC}"
        echo ""
        echo "Run './download_model.sh list' to see available models."
        echo "Run './download_model.sh recommend' for personalized suggestions."
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== Done ===${NC}"

