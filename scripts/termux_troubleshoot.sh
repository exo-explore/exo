#!/data/data/com.termux/files/usr/bin/bash
#
# exo Termux Troubleshooting Script
# ===================================
# Diagnose and fix common issues.
#
# Usage:
#   chmod +x scripts/termux_troubleshoot.sh
#   ./scripts/termux_troubleshoot.sh
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     exo Termux Troubleshooting            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Function to fix common issues
fix_issue() {
    local issue=$1
    echo ""
    echo -e "${YELLOW}Attempting to fix: $issue${NC}"
}

# Check 1: Termux packages
echo -e "${BLUE}[1/6] Checking Termux packages...${NC}"
MISSING_PKGS=""

check_pkg() {
    if ! command -v $1 &> /dev/null; then
        MISSING_PKGS="$MISSING_PKGS $2"
        return 1
    fi
    return 0
}

check_pkg python3 python || true
check_pkg pip python-pip || true
check_pkg clang clang || true
check_pkg cmake cmake || true
check_pkg make make || true
check_pkg git git || true

if [ -n "$MISSING_PKGS" ]; then
    echo -e "${YELLOW}Missing packages:$MISSING_PKGS${NC}"
    echo "Installing..."
    pkg install -y $MISSING_PKGS
    echo -e "${GREEN}✓ Packages installed${NC}"
else
    echo -e "${GREEN}✓ All required packages installed${NC}"
fi

# Check 2: Build environment
echo ""
echo -e "${BLUE}[2/6] Checking build environment...${NC}"

if [ -z "$CC" ] || [ -z "$CXX" ]; then
    echo "Setting up compiler environment..."
    export CC=clang
    export CXX=clang++
    echo -e "${GREEN}✓ Compiler environment set${NC}"
else
    echo -e "${GREEN}✓ Compiler environment OK${NC}"
fi

# Check 3: llama-cpp-python
echo ""
echo -e "${BLUE}[3/6] Checking llama-cpp-python...${NC}"

if python3 -c "import llama_cpp" 2>/dev/null; then
    VERSION=$(python3 -c "import llama_cpp; print(llama_cpp.__version__)")
    echo -e "${GREEN}✓ llama-cpp-python $VERSION installed${NC}"
else
    echo -e "${YELLOW}llama-cpp-python not installed${NC}"
    echo ""
    echo "Would you like to install it now? (This takes 10-20 minutes)"
    read -p "[y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing llama-cpp-python..."
        export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF"
        export FORCE_CMAKE=1
        
        pip install llama-cpp-python --no-cache-dir 2>&1 | tee /tmp/llama_build.log
        
        if python3 -c "import llama_cpp" 2>/dev/null; then
            echo -e "${GREEN}✓ llama-cpp-python installed successfully${NC}"
        else
            echo -e "${RED}✗ Installation failed${NC}"
            echo "Build log saved to /tmp/llama_build.log"
            echo ""
            echo "Common fixes:"
            echo "  1. Try: export CMAKE_ARGS='-DGGML_BLAS=OFF -DGGML_NATIVE=OFF'"
            echo "  2. Try installing from source (see README)"
            echo "  3. Check if you have enough storage space"
        fi
    fi
fi

# Check 4: exo installation
echo ""
echo -e "${BLUE}[4/6] Checking exo installation...${NC}"

if python3 -c "import exo" 2>/dev/null; then
    echo -e "${GREEN}✓ exo module importable${NC}"
else
    echo -e "${YELLOW}exo not fully installed${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXO_DIR="$(dirname "$SCRIPT_DIR")"
    
    if [ -f "$EXO_DIR/pyproject.toml" ]; then
        echo "Attempting to reinstall exo..."
        cd "$EXO_DIR"
        pip install -e . 2>&1 || {
            echo "Installing core dependencies only..."
            pip install aiofiles aiohttp pydantic fastapi huggingface-hub psutil loguru anyio bidict
            pip install -e . --no-deps
        }
    else
        echo -e "${RED}Cannot find exo source directory${NC}"
        echo "Please run this script from within the exo repository"
    fi
fi

# Check 5: Disk space
echo ""
echo -e "${BLUE}[5/6] Checking disk space...${NC}"

AVAILABLE_MB=$(df -m ~ 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "$AVAILABLE_MB" ]; then
    if [ "$AVAILABLE_MB" -lt 1000 ]; then
        echo -e "${YELLOW}⚠ Low disk space: ${AVAILABLE_MB}MB available${NC}"
        echo "  Models typically need 500MB-3GB each"
        echo "  Consider freeing up space before downloading models"
    else
        echo -e "${GREEN}✓ Disk space OK: ${AVAILABLE_MB}MB available${NC}"
    fi
else
    echo "Could not check disk space"
fi

# Check 6: Memory
echo ""
echo -e "${BLUE}[6/6] Checking memory...${NC}"

if [ -f /proc/meminfo ]; then
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
    FREE_MEM=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')
    
    echo "Total RAM: ${TOTAL_MEM}MB"
    echo "Available: ${FREE_MEM}MB"
    
    if [ "$FREE_MEM" -lt 500 ]; then
        echo -e "${YELLOW}⚠ Low memory! Close other apps for better performance${NC}"
    else
        echo -e "${GREEN}✓ Memory OK${NC}"
    fi
    
    echo ""
    echo "Recommended models based on your RAM:"
    if [ "$TOTAL_MEM" -lt 2000 ]; then
        echo "  • qwen-0.5b (~400MB) - Ultra-light"
    elif [ "$TOTAL_MEM" -lt 4000 ]; then
        echo "  • tinyllama (~700MB) - Best for your device"
        echo "  • qwen-0.5b (~400MB) - Alternative"
    elif [ "$TOTAL_MEM" -lt 6000 ]; then
        echo "  • llama-3b (~2GB) - Good balance"
        echo "  • tinyllama (~700MB) - Faster option"
    else
        echo "  • phi-3 (~2.3GB) - Strong reasoning"
        echo "  • llama-3b (~2GB) - Good balance"
    fi
else
    echo "Could not check memory"
fi

echo ""
echo -e "${BLUE}=== Troubleshooting Complete ===${NC}"
echo ""
echo "If you're still having issues:"
echo "  1. Check the build log: cat /tmp/llama_build.log"
echo "  2. Try a clean reinstall: pip uninstall llama-cpp-python && pip install llama-cpp-python"
echo "  3. Make sure Termux is from F-Droid (not Play Store)"
echo "  4. Restart Termux and try again"
echo ""

