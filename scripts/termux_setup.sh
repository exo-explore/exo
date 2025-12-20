#!/data/data/com.termux/files/usr/bin/bash
#
# exo Termux Setup Script
# ========================
# This script sets up exo with llama.cpp backend on Android/Termux.
# Run this after cloning the exo repository.
#
# Usage:
#   chmod +x scripts/termux_setup.sh
#   ./scripts/termux_setup.sh
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running in Termux
check_termux() {
    print_header "Checking Environment"
    
    if [ -z "$TERMUX_VERSION" ] && [ ! -d "/data/data/com.termux" ]; then
        print_error "This script must be run in Termux on Android!"
        print_info "Install Termux from F-Droid: https://f-droid.org/packages/com.termux/"
        exit 1
    fi
    
    print_success "Running in Termux"
    print_info "Termux version: ${TERMUX_VERSION:-unknown}"
}

# Update and upgrade packages
update_packages() {
    print_header "Updating Termux Packages"
    
    pkg update -y
    pkg upgrade -y
    
    print_success "Packages updated"
}

# Install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    # Core build tools
    pkg install -y python git clang cmake make binutils
    
    # For Rust bindings (if needed later)
    pkg install -y rust
    
    # Networking and utilities
    pkg install -y openssh curl wget
    
    # Libraries needed for compilation
    pkg install -y libffi openssl
    
    print_success "System dependencies installed"
}

# Set up storage access
setup_storage() {
    print_header "Setting Up Storage Access"
    
    if [ ! -d "$HOME/storage" ]; then
        print_info "Requesting storage permission..."
        termux-setup-storage
        sleep 2
    else
        print_success "Storage already accessible"
    fi
}

# Create exo directories
create_directories() {
    print_header "Creating exo Directories"
    
    mkdir -p ~/.exo/models
    mkdir -p ~/.exo/logs
    
    print_success "Created ~/.exo/models"
    print_success "Created ~/.exo/logs"
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    # Upgrade pip first
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Set build flags for llama-cpp-python on ARM
    export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF"
    export FORCE_CMAKE=1
    
    # Install llama-cpp-python separately first (it can be tricky)
    print_info "Installing llama-cpp-python (this may take a while)..."
    pip install llama-cpp-python --no-cache-dir
    
    if [ $? -eq 0 ]; then
        print_success "llama-cpp-python installed"
    else
        print_error "Failed to install llama-cpp-python"
        print_warning "Trying alternative method..."
        
        # Try with minimal features
        CMAKE_ARGS="-DGGML_BLAS=OFF" pip install llama-cpp-python --no-cache-dir
    fi
}

# Install exo package
install_exo() {
    print_header "Installing exo"
    
    # Get the directory where the script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXO_DIR="$(dirname "$SCRIPT_DIR")"
    
    cd "$EXO_DIR"
    
    print_info "Installing exo from: $EXO_DIR"
    
    # Install exo in editable mode
    pip install -e .
    
    if [ $? -eq 0 ]; then
        print_success "exo installed successfully"
    else
        print_error "Failed to install exo"
        exit 1
    fi
}

# Download a small test model
download_test_model() {
    print_header "Downloading Test Model"
    
    print_info "Downloading TinyLlama 1.1B (Q4_K_M) - ~700MB"
    print_info "This is a small model suitable for testing on Android"
    
    python3 << 'EOF'
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    
    model_dir = Path.home() / ".exo" / "models" / "TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading model...")
    path = hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to: {path}")
    
except Exception as e:
    print(f"Warning: Could not download model: {e}")
    print("You can download it manually later.")
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Test model downloaded"
    else
        print_warning "Could not download test model (you can do this later)"
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_info "Python: $PYTHON_VERSION"
    
    # Check llama-cpp-python
    python3 -c "import llama_cpp; print('llama-cpp-python version:', llama_cpp.__version__)" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "llama-cpp-python is working"
    else
        print_error "llama-cpp-python import failed"
    fi
    
    # Check exo platform detection
    python3 << 'EOF'
try:
    from exo.shared.platform import get_platform_info, get_recommended_backend, is_android
    
    info = get_platform_info()
    print(f"Platform: {info['system']} ({info['machine']})")
    print(f"Is Android: {is_android()}")
    print(f"Recommended backend: {get_recommended_backend()}")
    print("✓ exo platform detection working")
except Exception as e:
    print(f"✗ Platform detection failed: {e}")
EOF
    
    # Check available memory
    print_info "System resources:"
    free -h
}

# Print final instructions
print_instructions() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}exo is now installed with llama.cpp backend!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Test the installation:"
    echo "   python3 -c \"from llama_cpp import Llama; print('OK')\""
    echo ""
    echo "2. Run exo (when ready):"
    echo "   cd ~/exo"
    echo "   python3 -m exo"
    echo ""
    echo "3. For multi-device cluster:"
    echo "   - Run this script on each Android device"
    echo "   - Ensure devices are on the same network"
    echo "   - exo will auto-discover other nodes"
    echo ""
    echo "Model location: ~/.exo/models/"
    echo "Logs location: ~/.exo/logs/"
    echo ""
    echo -e "${YELLOW}Note: The Rust networking bindings may need additional setup.${NC}"
    echo -e "${YELLOW}Check the README for more information.${NC}"
}

# Main execution
main() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     exo Termux Setup Script               ║${NC}"
    echo -e "${GREEN}║     llama.cpp backend for Android         ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
    echo ""
    
    check_termux
    update_packages
    install_system_deps
    setup_storage
    create_directories
    install_python_deps
    install_exo
    
    # Optional: Download test model
    read -p "Download test model (~700MB)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_test_model
    fi
    
    verify_installation
    print_instructions
}

# Run main function
main "$@"

