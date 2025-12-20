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

# Update packages
echo "[1/5] Updating packages..."
pkg update -y && pkg upgrade -y

# Install dependencies
echo "[2/5] Installing dependencies..."
pkg install -y python git clang cmake make binutils libffi openssl

# Create directories
echo "[3/5] Creating directories..."
mkdir -p ~/.exo/models ~/.exo/logs

# Install llama-cpp-python
echo "[4/5] Installing llama-cpp-python (this takes a while)..."
export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON"
pip install --upgrade pip
pip install llama-cpp-python --no-cache-dir

# Install exo
echo "[5/5] Installing exo..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
pip install -e .

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Test with: python3 -c \"from llama_cpp import Llama; print('OK')\""
echo ""

