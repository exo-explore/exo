#!/bin/bash
# Setup script for cross-compiling RKLLM converter on ARM64 systems
#
# RKLLM-Toolkit only runs on x86_64, so on ARM64 systems (DGX Spark, etc.)
# we need QEMU emulation to build and run the container.

set -e

echo "=== RKLLM Converter Cross-Compile Setup ==="
echo ""

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    echo "✓ Running on x86_64 - no cross-compilation needed"
    echo ""
    echo "Build normally with:"
    echo "  docker build -t rkllm-converter ."
    exit 0
fi

echo "Detected architecture: $ARCH"
echo "Setting up x86_64 emulation via QEMU..."
echo ""

# Step 1: Install QEMU binfmt support
echo "[1/3] Installing QEMU binfmt handlers..."
if docker run --privileged --rm tonistiigi/binfmt --install amd64 2>/dev/null; then
    echo "✓ QEMU binfmt installed"
else
    echo "✗ Failed to install QEMU binfmt"
    echo "  You may need to run: sudo docker run --privileged --rm tonistiigi/binfmt --install amd64"
    exit 1
fi
echo ""

# Step 2: Create buildx builder
echo "[2/3] Creating Docker buildx builder..."
if docker buildx inspect rkllm-builder >/dev/null 2>&1; then
    echo "Builder 'rkllm-builder' already exists"
    docker buildx use rkllm-builder
else
    docker buildx create --name rkllm-builder --use
    echo "✓ Created and activated 'rkllm-builder'"
fi
echo ""

# Step 3: Bootstrap the builder
echo "[3/3] Bootstrapping builder..."
docker buildx inspect --bootstrap
echo ""

# Verify
echo "=== Verification ==="
echo "Available platforms:"
docker buildx inspect | grep -A 5 "Platforms:"
echo ""

# Check if linux/amd64 is available
if docker buildx inspect 2>/dev/null | grep -q "linux/amd64"; then
    echo "✓ linux/amd64 platform available"
else
    echo "✗ linux/amd64 platform NOT available"
    echo "  QEMU installation may have failed"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Build the container with:"
echo "  docker buildx build --platform linux/amd64 -t rkllm-converter --load ."
echo ""
echo "Run with:"
echo "  docker run --platform linux/amd64 -it --rm \\"
echo "             -v \$(pwd)/output:/workspace/output \\"
echo "             -v \$(pwd)/cache:/workspace/cache \\"
echo "             rkllm-converter --help"
echo ""
