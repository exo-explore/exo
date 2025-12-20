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
# IMPORTANT: Install Termux from F-Droid, NOT Play Store!
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

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXO_DIR="$(dirname "$SCRIPT_DIR")"

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
    print_info "Script directory: $SCRIPT_DIR"
    print_info "exo directory: $EXO_DIR"
}

# Update and upgrade packages
update_packages() {
    print_header "Updating Termux Packages"
    
    # Try to update, but don't fail if mirrors have issues
    pkg update -y || print_warning "Some package updates may have failed"
    pkg upgrade -y || print_warning "Some package upgrades may have failed"
    
    print_success "Packages updated"
}

# Install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    # Core Python and build tools
    print_info "Installing Python and build tools..."
    pkg install -y python python-pip || {
        print_error "Failed to install python/pip"
        exit 1
    }
    
    # Build essentials
    print_info "Installing build essentials..."
    pkg install -y git clang cmake make binutils ninja || {
        print_warning "Some build tools may not have installed"
    }
    
    # Libraries needed for compilation
    print_info "Installing development libraries..."
    pkg install -y libffi openssl libc++ || {
        print_warning "Some libraries may not have installed"
    }
    
    # Networking and utilities
    print_info "Installing utilities..."
    pkg install -y curl wget || true
    
    print_success "System dependencies installed"
}

# Set up storage access
setup_storage() {
    print_header "Setting Up Storage Access"
    
    if [ ! -d "$HOME/storage" ]; then
        print_info "Requesting storage permission..."
        print_warning "Please grant storage permission when prompted!"
        termux-setup-storage || true
        sleep 3
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

# Set up build environment for llama.cpp
setup_build_env() {
    print_header "Setting Up Build Environment"
    
    # Set environment variables for Android/ARM builds
    export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF"
    export FORCE_CMAKE=1
    
    # Set compiler flags for Termux
    export CFLAGS="-Wno-error"
    export CXXFLAGS="-Wno-error"
    
    # Ensure we use the right compilers
    export CC=clang
    export CXX=clang++
    
    print_success "Build environment configured"
    print_info "CMAKE_ARGS: $CMAKE_ARGS"
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    # NOTE: Do NOT upgrade pip in Termux - it breaks the python-pip package!
    # Just use pip directly
    
    print_info "Installing wheel and setuptools..."
    pip install wheel setuptools --quiet || {
        print_warning "wheel/setuptools install had issues, continuing..."
    }
    
    print_info "Installing llama-cpp-python..."
    print_info "This may take 10-20 minutes on first build..."
    print_info "Please be patient!"
    echo ""
    
    # Try to install llama-cpp-python with our build settings
    if pip install llama-cpp-python --no-cache-dir --verbose 2>&1 | tee /tmp/llama_install.log; then
        print_success "llama-cpp-python installed successfully!"
    else
        print_error "llama-cpp-python installation failed"
        print_info "Trying alternative build method..."
        
        # Try with even more minimal settings
        export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF"
        
        if pip install llama-cpp-python --no-cache-dir; then
            print_success "llama-cpp-python installed with minimal features"
        else
            print_error "Could not install llama-cpp-python"
            print_info "Check /tmp/llama_install.log for details"
            print_info "You may need to build from source manually"
            
            # Don't exit - let user decide what to do
            print_warning "Continuing without llama-cpp-python..."
        fi
    fi
}

# Install exo package
install_exo() {
    print_header "Installing exo"
    
    cd "$EXO_DIR"
    
    print_info "Installing exo from: $EXO_DIR"
    
    # Install core dependencies first (without the optional backends)
    print_info "Installing exo core dependencies..."
    
    # Install exo in editable mode
    if pip install -e . --no-deps 2>&1; then
        print_info "Installed exo package"
    fi
    
    # Now install dependencies one by one to handle failures gracefully
    print_info "Installing Python dependencies (this may take a while)..."
    
    # Core dependencies that should work on Termux
    CORE_DEPS=(
        "aiofiles"
        "aiohttp"
        "pydantic"
        "fastapi"
        "filelock"
        "huggingface-hub"
        "psutil"
        "loguru"
        "anyio"
        "bidict"
        "hypercorn"
        "tiktoken"
        "rich"
        "networkx"
    )
    
    for dep in "${CORE_DEPS[@]}"; do
        print_info "  Installing $dep..."
        pip install "$dep" --quiet || print_warning "  Failed to install $dep"
    done
    
    # Try to install the full package now
    print_info "Finalizing exo installation..."
    pip install -e . || {
        print_warning "Some exo dependencies may not be installed"
        print_info "Core functionality should still work"
    }
    
    print_success "exo installation complete"
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_info "Python: $PYTHON_VERSION"
    
    # Check llama-cpp-python
    echo ""
    print_info "Checking llama-cpp-python..."
    python3 -c "
try:
    import llama_cpp
    print('  Version:', llama_cpp.__version__)
    print('  Status: OK')
except ImportError as e:
    print('  Status: NOT INSTALLED')
    print('  Error:', str(e))
" 2>/dev/null || print_warning "llama-cpp-python check failed"
    
    # Check exo platform detection
    echo ""
    print_info "Checking exo..."
    python3 -c "
try:
    from exo.shared.platform import get_platform_info, get_recommended_backend, is_android
    info = get_platform_info()
    print('  System:', info['system'])
    print('  Machine:', info['machine'])
    print('  Is Android:', is_android())
    print('  Backend:', get_recommended_backend())
    print('  Status: OK')
except Exception as e:
    print('  Status: PARTIAL')
    print('  Note: Some exo modules may need Rust bindings')
" 2>/dev/null || print_warning "exo check had issues"
    
    # Check available memory
    echo ""
    print_info "System resources:"
    free -h 2>/dev/null || print_warning "Could not check memory"
}

# Print final instructions
print_instructions() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}exo setup is complete!${NC}"
    echo ""
    echo "What was installed:"
    echo "  • Python packages for exo"
    echo "  • llama-cpp-python (if build succeeded)"
    echo "  • Build tools for future compilations"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Download a model:"
    echo "   chmod +x scripts/download_model.sh"
    echo "   ./scripts/download_model.sh list"
    echo "   ./scripts/download_model.sh tinyllama"
    echo ""
    echo "2. Verify installation:"
    echo "   ./scripts/termux_verify.sh"
    echo ""
    echo "3. Test llama.cpp:"
    echo "   python3 -c \"from llama_cpp import Llama; print('OK')\""
    echo ""
    echo "Model location: ~/.exo/models/"
    echo "Logs location: ~/.exo/logs/"
    echo ""
    
    if ! python3 -c "import llama_cpp" 2>/dev/null; then
        echo -e "${YELLOW}NOTE: llama-cpp-python may not have installed correctly.${NC}"
        echo -e "${YELLOW}You may need to build it manually or check build logs.${NC}"
        echo ""
    fi
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
    setup_build_env
    install_python_deps
    install_exo
    verify_installation
    print_instructions
}

# Run main function
main "$@"
