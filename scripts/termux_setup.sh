#!/data/data/com.termux/files/usr/bin/bash
#
# exo Termux Setup Script v2
# ==========================
# This script sets up exo with llama.cpp backend on Android/Termux.
#
# Usage:
#   chmod +x scripts/termux_setup.sh
#   ./scripts/termux_setup.sh
#
# IMPORTANT: Install Termux from F-Droid, NOT Play Store!
#

# Don't exit immediately on error - we handle errors ourselves
set +e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================${NC}"
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
    print_info "exo directory: $EXO_DIR"
}

# Update packages with retry
update_packages() {
    print_header "Updating Termux Packages"
    
    print_info "Running termux-change-repo to ensure mirrors are set..."
    print_info "If prompted, select a mirror close to your location"
    
    # This might prompt user for mirror selection
    termux-change-repo 2>/dev/null || true
    
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if pkg update -y 2>/dev/null && pkg upgrade -y 2>/dev/null; then
            print_success "Packages updated"
            return 0
        fi
        retry=$((retry + 1))
        print_warning "Update attempt $retry failed, retrying..."
        sleep 2
    done
    
    print_warning "Package updates may have failed - continuing anyway"
}

# Install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    # Core packages - install all at once
    print_info "Installing Python, Rust, and build tools..."
    
    pkg install -y \
        python python-pip \
        rust \
        git clang cmake make binutils ninja \
        libffi openssl libc++ \
        curl wget \
        2>/dev/null || {
            print_warning "Some packages may not have installed, trying individually..."
            pkg install -y python python-pip || true
            pkg install -y rust || true
            pkg install -y git clang cmake make || true
            pkg install -y libffi openssl || true
        }
    
    # Verify critical packages
    if ! command -v python3 &>/dev/null; then
        print_error "Python not installed!"
        exit 1
    fi
    
    if ! command -v rustc &>/dev/null; then
        print_error "Rust not installed! Required for exo_pyo3_bindings"
        print_info "Try: pkg install rust"
        exit 1
    fi
    
    print_success "Python: $(python3 --version)"
    print_success "Rust: $(rustc --version)"
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
    export CC=clang
    export CXX=clang++
    export CFLAGS="-O2 -Wno-error"
    export CXXFLAGS="-O2 -Wno-error"
    
    print_success "Build environment configured"
    print_info "CMAKE_ARGS: $CMAKE_ARGS"
}

# Install maturin for building Rust Python bindings
install_maturin() {
    print_header "Installing Maturin (Rust Python Builder)"
    
    print_info "Installing maturin..."
    
    if pip install maturin --quiet 2>/dev/null; then
        print_success "maturin installed"
    else
        print_warning "pip install failed, trying with --user..."
        pip install maturin --user --quiet 2>/dev/null || {
            print_error "Could not install maturin"
            print_info "Try manually: pip install maturin"
            return 1
        }
    fi
    
    return 0
}

# Build exo_pyo3_bindings from source
build_pyo3_bindings() {
    print_header "Building exo_pyo3_bindings (Rust Networking)"
    
    local bindings_dir="$EXO_DIR/rust/exo_pyo3_bindings"
    
    if [ ! -d "$bindings_dir" ]; then
        print_error "exo_pyo3_bindings directory not found at $bindings_dir"
        return 1
    fi
    
    cd "$bindings_dir"
    
    print_info "Building Rust bindings (this may take 5-15 minutes)..."
    print_info "Directory: $bindings_dir"
    
    # Try to build and install using maturin
    if maturin develop --release 2>&1 | tee /tmp/maturin_build.log; then
        print_success "exo_pyo3_bindings built and installed!"
        cd "$EXO_DIR"
        return 0
    else
        print_error "maturin develop failed"
        print_info "Trying alternative: maturin build..."
        
        if maturin build --release 2>&1; then
            # Find and install the wheel
            local wheel=$(find target/wheels -name "*.whl" | head -1)
            if [ -n "$wheel" ]; then
                pip install "$wheel" && {
                    print_success "exo_pyo3_bindings installed from wheel"
                    cd "$EXO_DIR"
                    return 0
                }
            fi
        fi
        
        print_error "Could not build exo_pyo3_bindings"
        print_info "Check /tmp/maturin_build.log for details"
        print_warning "exo will run with limited networking functionality"
        cd "$EXO_DIR"
        return 1
    fi
}

# Install llama-cpp-python
install_llama_cpp() {
    print_header "Installing llama-cpp-python"
    
    print_info "This may take 10-20 minutes on first build..."
    print_info "Building with CPU-only backend for ARM..."
    
    # Set optimal build flags
    export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF"
    export FORCE_CMAKE=1
    
    if pip install llama-cpp-python --no-cache-dir 2>&1 | tee /tmp/llama_install.log | tail -20; then
        print_success "llama-cpp-python installed!"
        return 0
    else
        print_warning "First attempt failed, trying minimal build..."
        
        export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF"
        
        if pip install llama-cpp-python --no-cache-dir 2>&1; then
            print_success "llama-cpp-python installed with minimal features"
            return 0
        fi
        
        print_error "Could not install llama-cpp-python"
        print_info "Check /tmp/llama_install.log for details"
        return 1
    fi
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    cd "$EXO_DIR"
    
    # Install wheel and setuptools first
    print_info "Installing build tools..."
    pip install wheel setuptools --quiet 2>/dev/null || true
    
    # Check for requirements-termux.txt
    if [ -f "requirements-termux.txt" ]; then
        print_info "Installing dependencies from requirements-termux.txt..."
        print_info "This may take a few minutes..."
        
        if pip install -r requirements-termux.txt 2>&1 | tail -10; then
            print_success "Core dependencies installed"
        else
            print_warning "Some dependencies failed, trying individually..."
            
            # Install critical deps one by one
            for pkg in pydantic fastapi aiohttp loguru psutil huggingface-hub; do
                print_info "  Installing $pkg..."
                pip install "$pkg" --quiet 2>/dev/null || print_warning "  $pkg failed"
            done
        fi
    else
        print_warning "requirements-termux.txt not found, installing manually..."
        
        # Core deps that should work on Termux
        pip install \
            pydantic fastapi aiohttp loguru psutil \
            huggingface-hub aiofiles filelock \
            anyio bidict rich networkx \
            2>/dev/null || print_warning "Some deps failed"
    fi
}

# Install exo package
install_exo() {
    print_header "Installing exo Package"
    
    cd "$EXO_DIR"
    
    print_info "Installing exo from: $EXO_DIR"
    
    # First try without deps (deps already installed)
    if pip install -e . --no-deps 2>&1; then
        print_success "exo package installed"
    else
        print_warning "Editable install failed, trying regular install..."
        pip install . --no-deps 2>/dev/null || {
            print_error "Could not install exo package"
            return 1
        }
    fi
    
    # Now try to install any remaining deps
    print_info "Checking for missing dependencies..."
    pip install -e . 2>&1 | grep -E "(Requirement|Successfully)" | tail -5 || true
    
    print_success "exo installation complete"
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    local all_ok=true
    
    # Check Python
    print_info "Python: $(python3 --version)"
    
    # Check llama-cpp-python
    echo ""
    print_info "Checking llama-cpp-python..."
    if python3 -c "import llama_cpp; print('  Version:', llama_cpp.__version__)" 2>/dev/null; then
        print_success "llama-cpp-python: OK"
    else
        print_warning "llama-cpp-python: NOT AVAILABLE"
        all_ok=false
    fi
    
    # Check exo_pyo3_bindings
    echo ""
    print_info "Checking exo_pyo3_bindings..."
    if python3 -c "import exo_pyo3_bindings; print('  Imported successfully')" 2>/dev/null; then
        print_success "exo_pyo3_bindings: OK"
    else
        print_warning "exo_pyo3_bindings: NOT AVAILABLE"
        all_ok=false
    fi
    
    # Check exo
    echo ""
    print_info "Checking exo core..."
    if python3 -c "import exo; print('  Imported successfully')" 2>/dev/null; then
        print_success "exo: OK"
    else
        print_warning "exo: NOT FULLY AVAILABLE"
        all_ok=false
    fi
    
    # Show memory
    echo ""
    print_info "System resources:"
    free -h 2>/dev/null | head -2 || true
    
    echo ""
    if [ "$all_ok" = true ]; then
        print_success "All components verified!"
    else
        print_warning "Some components need attention (see above)"
    fi
}

# Print final instructions
print_instructions() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}exo Termux setup finished!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Download a model:"
    echo "   ${CYAN}./scripts/download_model.sh list${NC}"
    echo "   ${CYAN}./scripts/download_model.sh qwen-0.5b${NC}  # ~400MB, for 4GB RAM"
    echo "   ${CYAN}./scripts/download_model.sh tinyllama${NC}  # ~700MB"
    echo ""
    echo "2. Test llama.cpp:"
    echo "   ${CYAN}python3 -c \"from llama_cpp import Llama; print('OK')\"${NC}"
    echo ""
    echo "3. Test inference (after downloading a model):"
    echo "   ${CYAN}python3 scripts/test_inference.py${NC}"
    echo ""
    echo "Model location: ~/.exo/models/"
    echo ""
    
    # Show warnings if needed
    if ! python3 -c "import llama_cpp" 2>/dev/null; then
        echo -e "${YELLOW}NOTE: llama-cpp-python not installed.${NC}"
        echo "Try manually: pip install llama-cpp-python --no-cache-dir"
        echo ""
    fi
    
    if ! python3 -c "import exo_pyo3_bindings" 2>/dev/null; then
        echo -e "${YELLOW}NOTE: exo_pyo3_bindings not installed.${NC}"
        echo "Networking features will be limited."
        echo "To build manually:"
        echo "  cd rust/exo_pyo3_bindings"
        echo "  maturin develop --release"
        echo ""
    fi
}

# Main execution
main() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     exo Termux Setup Script v2                ║${NC}"
    echo -e "${GREEN}║     llama.cpp backend for Android             ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
    echo ""
    
    check_termux
    update_packages
    install_system_deps
    setup_storage
    create_directories
    setup_build_env
    install_maturin
    install_python_deps
    install_llama_cpp
    build_pyo3_bindings
    install_exo
    verify_installation
    print_instructions
}

# Run main function
main "$@"
