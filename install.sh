#!/usr/bin/env bash

echo "=== exo Installation Script ==="
echo ""

# Check Python version
if command -v python3.12 &>/dev/null; then
    echo "✅ Python 3.12 is installed, proceeding with python3.12..."
    PYTHON_CMD=python3.12
else
    echo "⚠️  The recommended version of Python to run exo with is Python 3.12, but $(python3 --version) is installed. Proceeding with $(python3 --version)"
    PYTHON_CMD=python3
fi

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv-linux
source venv-linux/bin/activate

# Detect GPU for optimized installation
NVIDIA_GPU_DETECTED=false
CUDA_AVAILABLE=false

echo ""
echo "🔍 Detecting hardware acceleration support..."

# Check for NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi -L 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$GPU_INFO" ]; then
        NVIDIA_GPU_DETECTED=true
        echo "✅ NVIDIA GPU detected:"
        echo "$GPU_INFO"
        
        # Check for CUDA toolkit
        if command -v nvcc &>/dev/null; then
            CUDA_AVAILABLE=true
            echo "✅ CUDA toolkit detected: $(nvcc --version | grep release | head -1)"
        else
            echo "⚠️  CUDA toolkit not found - GPU acceleration will be limited"
            echo "   Install CUDA toolkit for optimal performance"
        fi
    fi
else
    echo "ℹ️  No NVIDIA GPU detected, installing CPU-only version"
fi

echo ""

# Install exo with automatic GPU detection
echo "🚀 Installing exo with automatic GPU detection..."
echo "   setup.py will auto-detect hardware and compile with appropriate support"

if [ "$NVIDIA_GPU_DETECTED" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
    echo "   NVIDIA GPU and CUDA detected - will compile with GPU acceleration"
    echo "   This may take several minutes..."
elif [ "$NVIDIA_GPU_DETECTED" = true ]; then
    echo "   NVIDIA GPU detected but CUDA missing - installing CPU version"
    echo "   Run ./fix_llamacpp_gpu.sh later for GPU support"
else
    echo "   No GPU detected - installing CPU-only version"
fi

echo ""
echo "Installing base dependencies..."
pip install --upgrade pip setuptools wheel

# Install llama-cpp-python with CUDA support if available
if [ "$NVIDIA_GPU_DETECTED" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
    echo "Installing llama-cpp-python with CUDA support..."
    export CMAKE_ARGS="-DGGML_CUDA=on"
    export FORCE_CMAKE=1
    
    # Optionally add architecture-specific flags here if needed
    if echo "$GPU_INFO" | grep -q "5070"; then
        echo "[RTX 5070 Ti] Applying Blackwell architecture optimizations..."
        export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_ARCHITECTURES=90 -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CUDA_DMMV_F16=ON"
        echo "CMAKE_ARGS: $CMAKE_ARGS"
    elif echo "$GPU_INFO" | grep -q "3060"; then
        echo "[RTX 3060] Applying Ampere architecture optimizations..."
        export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_ARCHITECTURES=86"
        echo "CMAKE_ARGS: $CMAKE_ARGS"
    fi
    
    # First attempt: Install from PyPI with CUDA compilation
    echo "Attempting PyPI install with CUDA compilation..."
    if pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python; then
        echo "[OK] llama-cpp-python with CUDA support installed from PyPI"
        
        # Verify CUDA support was actually compiled
        echo "Verifying CUDA support..."
        if $PYTHON_CMD -c "from llama_cpp import llama_cpp; print('CUDA support:', llama_cpp.llama_supports_gpu_offload())" 2>/dev/null | grep -q "True"; then
            echo "[OK] CUDA support verified successfully"
        else
            echo "[WARNING] PyPI install succeeded but CUDA support not detected. Trying source compilation..."
            pip uninstall -y llama-cpp-python
            
            # Second attempt: Compile from source
            echo "Compiling llama-cpp-python from source with CUDA..."
            if command -v git &>/dev/null; then
                git clone --recursive https://github.com/abetlen/llama-cpp-python.git /tmp/llama-cpp-python
                cd /tmp/llama-cpp-python
                pip install -e . --verbose
                cd - > /dev/null
                rm -rf /tmp/llama-cpp-python
                
                # Verify source compilation
                if $PYTHON_CMD -c "from llama_cpp import llama_cpp; print('CUDA support:', llama_cpp.llama_supports_gpu_offload())" 2>/dev/null | grep -q "True"; then
                    echo "[OK] CUDA support compiled successfully from source"
                else
                    echo "[ERROR] Source compilation failed. Installing CPU version..."
                    pip uninstall -y llama-cpp-python
                    unset CMAKE_ARGS
                    pip install --upgrade llama-cpp-python
                fi
            else
                echo "[ERROR] Git not available for source compilation. Installing CPU version..."
                unset CMAKE_ARGS
                pip install --upgrade llama-cpp-python
            fi
        fi
    else
        echo "[ERROR] PyPI install failed. Trying source compilation..."
        
        # Second attempt: Compile from source
        if command -v git &>/dev/null; then
            echo "Compiling llama-cpp-python from source with CUDA..."
            git clone --recursive https://github.com/abetlen/llama-cpp-python.git /tmp/llama-cpp-python
            cd /tmp/llama-cpp-python
            if pip install -e . --verbose; then
                cd - > /dev/null
                rm -rf /tmp/llama-cpp-python
                
                # Verify source compilation
                if $PYTHON_CMD -c "from llama_cpp import llama_cpp; print('CUDA support:', llama_cpp.llama_supports_gpu_offload())" 2>/dev/null | grep -q "True"; then
                    echo "[OK] CUDA support compiled successfully from source"
                else
                    echo "[ERROR] Source compilation failed. Installing CPU version..."
                    pip uninstall -y llama-cpp-python
                    unset CMAKE_ARGS
                    pip install --upgrade llama-cpp-python
                fi
            else
                echo "[ERROR] Source compilation failed. Installing CPU version..."
                cd - > /dev/null
                rm -rf /tmp/llama-cpp-python
                unset CMAKE_ARGS
                pip install --upgrade llama-cpp-python
            fi
        else
            echo "[ERROR] Git not available. Installing CPU version..."
            unset CMAKE_ARGS
            pip install --upgrade llama-cpp-python
        fi
    fi
else
    echo "Installing llama-cpp-python (CPU version)..."
    pip install --upgrade llama-cpp-python
fi

echo "Installing exo in development mode..."
pip install -e . --use-pep517

if [ "$NVIDIA_GPU_DETECTED" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
    echo ""
    echo "🧪 Testing GPU support..."
    $PYTHON_CMD -c "import sys; from llama_cpp import llama_cpp; gpu_support = llama_cpp.llama_supports_gpu_offload() if hasattr(llama_cpp, 'llama_supports_gpu_offload') else False; print('GPU offload support:', gpu_support); print('✅ CUDA support successfully enabled!' if gpu_support else '❌ CUDA support not detected')"
fi

echo ""
echo "✅ Installation complete!"

if [ "$NVIDIA_GPU_DETECTED" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
    echo "🎯 GPU acceleration is enabled and ready to use!"
elif [ "$NVIDIA_GPU_DETECTED" = true ]; then
    echo "💡 To enable GPU acceleration:"
    echo "   1. Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
    echo "   2. Run: ./fix_llamacpp_gpu.sh"
fi

echo ""
echo "🚀 You can now run exo with: python -m exo"
