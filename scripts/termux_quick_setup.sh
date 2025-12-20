#!/data/data/com.termux/files/usr/bin/bash
#
# exo Termux Quick Setup (Minimal)
# =================================
# Minimal setup script - just the essentials.
#
# Usage:
#   chmod +x scripts/termux_quick_setup.sh
#   ./scripts/termux_quick_setup.sh
#

set -e

echo "=== exo Quick Setup for Termux ==="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXO_DIR="$(dirname "$SCRIPT_DIR")"

# Update packages
echo "[1/6] Updating packages..."
pkg update -y || echo "Warning: Some updates may have failed"
pkg upgrade -y || echo "Warning: Some upgrades may have failed"

# Install dependencies
echo "[2/6] Installing dependencies..."
pkg install -y python python-pip git clang cmake make binutils libffi openssl libc++

# Create directories
echo "[3/6] Creating directories..."
mkdir -p ~/.exo/models ~/.exo/logs

# Set up build environment
echo "[4/6] Setting up build environment..."
export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF"
export FORCE_CMAKE=1
export CC=clang
export CXX=clang++

# Install llama-cpp-python
echo "[5/6] Installing llama-cpp-python (this takes 10-20 minutes)..."
pip install wheel setuptools || true
pip install llama-cpp-python --no-cache-dir || {
    echo "Warning: llama-cpp-python may not have installed correctly"
    echo "Trying minimal build..."
    export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=OFF"
    pip install llama-cpp-python --no-cache-dir || echo "Build failed - you may need to install manually"
}

# Install exo
echo "[6/6] Installing exo..."
cd "$EXO_DIR"
pip install -e . || {
    echo "Warning: Some exo dependencies may have failed"
    echo "Installing core dependencies individually..."
    pip install aiofiles aiohttp pydantic fastapi huggingface-hub psutil loguru anyio bidict || true
    pip install -e . --no-deps || true
}

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Test with: python3 -c \"from llama_cpp import Llama; print('OK')\""
echo ""
echo "Download a model:"
echo "  ./scripts/download_model.sh tinyllama"
echo ""
