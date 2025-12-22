#!/bin/bash
#
# EXO Termux/Android Installation Script
# 
# This script installs everything needed to run EXO on Android/Termux:
# - System packages (Python, Rust, CMake, Node.js, etc.)
# - llama.cpp with llama-server (for AI inference)
# - Rust networking bindings
# - EXO and dashboard
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/lukewrightmain/exo/main/scripts/install_exo-termux.sh | bash
#
# Or after cloning:
#   cd exo && ./scripts/install_exo-termux.sh
#
# Repository: https://github.com/lukewrightmain/exo
#

set -e  # Exit on error

# Capture script location at startup (before any cd commands)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/../pyproject.toml" ]]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_DIR="$HOME/exo-termux"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Check if running on Termux
check_termux() {
    if [[ ! -d "/data/data/com.termux" ]]; then
        log_error "This script must be run on Termux (Android)"
        log_info "Download Termux from F-Droid: https://f-droid.org/packages/com.termux/"
        exit 1
    fi
    log_success "Running on Termux"
}

# Print banner
print_banner() {
    echo ""
    echo -e "${YELLOW}"
    echo "  ███████╗██╗  ██╗ ██████╗ "
    echo "  ██╔════╝╚██╗██╔╝██╔═══██╗"
    echo "  █████╗   ╚███╔╝ ██║   ██║"
    echo "  ██╔══╝   ██╔██╗ ██║   ██║"
    echo "  ███████╗██╔╝ ██╗╚██████╔╝"
    echo "  ╚══════╝╚═╝  ╚═╝ ╚═════╝ "
    echo -e "${NC}"
    echo "  Distributed AI Inference for Android"
    echo "  https://github.com/lukewrightmain/exo"
    echo ""
}

# Step 1: Update and install base packages
install_base_packages() {
    log_step "Step 1/8: Installing Base Packages"
    
    log_info "Updating package lists..."
    pkg update -y
    pkg upgrade -y
    
    log_info "Installing required packages..."
    pkg install -y \
        git \
        python \
        python-pip \
        python-numpy \
        cmake \
        ninja \
        nodejs \
        tur-repo \
        curl \
        wget
    
    log_success "Base packages installed"
}

# Step 2: Install Rust Nightly
install_rust_nightly() {
    log_step "Step 2/8: Installing Rust Nightly"
    
    # Check if already installed
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
        if [[ "$RUST_VERSION" == *"nightly"* ]]; then
            log_success "Rust nightly already installed: $RUST_VERSION"
            return 0
        else
            log_warning "Rust stable found, need nightly. Installing..."
        fi
    fi
    
    log_info "Installing Rust nightly from TUR..."
    pkg install -y rustc-nightly rust-nightly-std-aarch64-linux-android
    
    # Source the nightly profile
    if [[ -f "$PREFIX/etc/profile.d/rust-nightly.sh" ]]; then
        source "$PREFIX/etc/profile.d/rust-nightly.sh"
    fi
    
    # Add to bashrc if not already there
    if ! grep -q "rust-nightly.sh" ~/.bashrc 2>/dev/null; then
        echo 'source $PREFIX/etc/profile.d/rust-nightly.sh' >> ~/.bashrc
    fi
    
    log_success "Rust nightly installed: $(rustc --version)"
}

# Step 3: Build llama.cpp
build_llama_cpp() {
    log_step "Step 3/8: Building llama.cpp"
    
    LLAMA_DIR="$HOME/llama.cpp"
    
    if [[ -f "$LLAMA_DIR/build/bin/llama-server" ]] && [[ -f "$LLAMA_DIR/build/bin/llama-cli" ]] && [[ -f "$LLAMA_DIR/build/bin/rpc-server" ]]; then
        log_success "llama.cpp already built (llama-server, llama-cli, and rpc-server exist)"
        return 0
    fi
    
    # Clean up incomplete or outdated builds (missing rpc-server means no RPC support)
    if [[ -d "$LLAMA_DIR/build" ]]; then
        if [[ ! -f "$LLAMA_DIR/build/bin/llama-cli" ]] || [[ ! -f "$LLAMA_DIR/build/bin/rpc-server" ]]; then
            log_warning "Found incomplete or outdated build (missing rpc-server), rebuilding..."
            rm -rf "$LLAMA_DIR/build"
        fi
    fi
    
    log_info "Cloning llama.cpp..."
    if [[ -d "$LLAMA_DIR" ]]; then
        cd "$LLAMA_DIR"
        git pull
    else
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
        cd "$LLAMA_DIR"
    fi
    
    log_info "Building llama.cpp (this may take 5-10 minutes)..."
    log_warning "Tip: Close other apps to free memory during build"
    
    # Enable RPC for distributed inference across multiple devices
    if ! cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_RPC=ON; then
        log_error "cmake configuration failed!"
        exit 1
    fi
    
    if ! cmake --build build --config Release -j4; then
        log_error "cmake build failed! Try with fewer jobs: cmake --build build -j2"
        rm -rf build
        exit 1
    fi
    
    # Build rpc-server for worker nodes
    log_info "Building rpc-server for distributed inference..."
    cmake --build build --target rpc-server -j4 || log_warning "rpc-server build failed (optional for distributed mode)"
    
    # Verify builds
    if [[ ! -f "$LLAMA_DIR/build/bin/llama-server" ]]; then
        log_error "llama-server not built! Build failed."
        exit 1
    fi
    
    if [[ ! -f "$LLAMA_DIR/build/bin/llama-cli" ]]; then
        log_error "llama-cli not built! Build failed."
        exit 1
    fi
    
    log_success "llama.cpp built successfully"
    log_success "  - llama-server: $LLAMA_DIR/build/bin/llama-server"
    log_success "  - llama-cli: $LLAMA_DIR/build/bin/llama-cli"
    if [[ -f "$LLAMA_DIR/build/bin/rpc-server" ]]; then
        log_success "  - rpc-server: $LLAMA_DIR/build/bin/rpc-server"
    fi
}

# Step 4: Configure environment
configure_environment() {
    log_step "Step 4/8: Configuring Environment"
    
    BASHRC="$HOME/.bashrc"
    
    # LD_LIBRARY_PATH
    if ! grep -q "llama.cpp/build/bin" "$BASHRC" 2>/dev/null; then
        echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> "$BASHRC"
        log_info "Added LD_LIBRARY_PATH to ~/.bashrc"
    fi
    
    # LLAMA_CPP_LIB
    if ! grep -q "LLAMA_CPP_LIB" "$BASHRC" 2>/dev/null; then
        echo 'export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so' >> "$BASHRC"
        log_info "Added LLAMA_CPP_LIB to ~/.bashrc"
    fi
    
    # Source for current session
    export LD_LIBRARY_PATH="$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH"
    export LLAMA_CPP_LIB="$HOME/llama.cpp/build/bin/libllama.so"
    
    log_success "Environment configured"
}

# Step 5: Install Python packages
install_python_packages() {
    log_step "Step 5/8: Installing Python Packages"
    
    log_info "Installing llama-cpp-python..."
    pip install llama-cpp-python
    
    log_info "Installing maturin..."
    pip install maturin
    
    log_info "Installing requests..."
    pip install requests
    
    log_success "Python packages installed"
}

# Step 6: Clone/update EXO repo
setup_exo_repo() {
    log_step "Step 6/8: Setting Up EXO Repository"
    
    # Check if we're running from inside the repo (already detected at startup)
    if [[ -f "$REPO_DIR/pyproject.toml" ]]; then
        log_info "Running from inside repo: $REPO_DIR"
    elif [[ -d "$REPO_DIR" ]]; then
        log_info "Updating existing EXO installation..."
        cd "$REPO_DIR"
        git pull
    else
        log_info "Cloning EXO repository..."
        git clone https://github.com/lukewrightmain/exo.git "$REPO_DIR"
    fi
    
    cd "$REPO_DIR"
    log_success "EXO repository ready at: $REPO_DIR"
}

# Step 7: Build Rust bindings
build_rust_bindings() {
    log_step "Step 7/8: Building Rust Networking Bindings"
    
    # Source rust nightly
    if [[ -f "$PREFIX/etc/profile.d/rust-nightly.sh" ]]; then
        source "$PREFIX/etc/profile.d/rust-nightly.sh"
    fi
    
    cd "$REPO_DIR/rust/exo_pyo3_bindings"
    
    # Check if already built
    WHEEL_FILE=$(ls "$REPO_DIR/target/wheels/exo_pyo3_bindings"*.whl 2>/dev/null | head -1)
    if [[ -n "$WHEEL_FILE" ]]; then
        log_info "Existing wheel found, checking if installed..."
        if python -c "import exo_pyo3_bindings" 2>/dev/null; then
            log_success "Rust bindings already installed"
            return 0
        fi
    fi
    
    # Fix Python version requirement if needed
    if grep -q 'requires-python = ">=3.13"' pyproject.toml; then
        log_info "Updating Python version requirement..."
        sed -i 's/requires-python = ">=3.13"/requires-python = ">=3.12"/' pyproject.toml
    fi
    
    log_info "Building Rust bindings (this may take 10-15 minutes)..."
    maturin build --release
    
    # Find and install the wheel
    WHEEL_FILE=$(ls "$REPO_DIR/target/wheels/exo_pyo3_bindings"*.whl 2>/dev/null | head -1)
    if [[ -z "$WHEEL_FILE" ]]; then
        log_error "Wheel file not found after build!"
        exit 1
    fi
    
    log_info "Installing wheel: $WHEEL_FILE"
    pip install "$WHEEL_FILE" --force-reinstall
    
    log_success "Rust bindings built and installed"
}

# Step 8: Install EXO and build dashboard
install_exo() {
    log_step "Step 8/8: Installing EXO and Building Dashboard"
    
    cd "$REPO_DIR"
    
    log_info "Installing EXO..."
    pip install -e .
    
    log_info "Building dashboard..."
    cd "$REPO_DIR/dashboard"
    npm install
    npm run build
    
    log_success "EXO installed and dashboard built"
}

# Verify installation
verify_installation() {
    log_step "Verifying Installation"
    
    ERRORS=0
    
    # Check llama-server
    if [[ -f "$HOME/llama.cpp/build/bin/llama-server" ]]; then
        log_success "llama-server: OK"
    else
        log_error "llama-server: NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check llama-cli
    if [[ -f "$HOME/llama.cpp/build/bin/llama-cli" ]]; then
        log_success "llama-cli: OK"
    else
        log_error "llama-cli: NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check rpc-server (optional, for distributed inference)
    if [[ -f "$HOME/llama.cpp/build/bin/rpc-server" ]]; then
        log_success "rpc-server: OK (distributed inference ready)"
    else
        log_warning "rpc-server: NOT FOUND (distributed inference won't work)"
    fi
    
    # Check llama-cpp-python
    if python -c "from llama_cpp import Llama" 2>/dev/null; then
        log_success "llama-cpp-python: OK"
    else
        log_warning "llama-cpp-python: Not working (not critical, using server mode)"
    fi
    
    # Check requests
    if python -c "import requests" 2>/dev/null; then
        log_success "requests: OK"
    else
        log_error "requests: NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check exo_pyo3_bindings
    if python -c "import exo_pyo3_bindings" 2>/dev/null; then
        log_success "exo_pyo3_bindings: OK"
    else
        log_error "exo_pyo3_bindings: NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check EXO import
    if python -c "from exo.main import main" 2>/dev/null; then
        log_success "exo: OK"
    else
        log_error "exo: NOT FOUND"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check dashboard
    if [[ -d "$REPO_DIR/dashboard/dist" ]]; then
        log_success "dashboard: OK"
    else
        log_error "dashboard: NOT BUILT"
        ERRORS=$((ERRORS + 1))
    fi
    
    return $ERRORS
}

# Print completion message
print_completion() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ EXO Installation Complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${CYAN}To run EXO:${NC}"
    echo -e "    cd $REPO_DIR"
    echo -e "    python -m exo"
    echo ""
    echo -e "  ${CYAN}Dashboard:${NC}"
    echo -e "    http://localhost:52415"
    echo ""
    echo -e "  ${CYAN}From other devices:${NC}"
    echo -e "    http://$(ip route get 1 2>/dev/null | awk '{print $7}' | head -1):52415"
    echo ""
    echo -e "  ${CYAN}Documentation:${NC}"
    echo -e "    https://github.com/lukewrightmain/exo/blob/main/docs/ANDROID_SETUP.md"
    echo ""
    echo -e "${YELLOW}  Note: Run 'source ~/.bashrc' or restart Termux to apply environment changes${NC}"
    echo ""
}

# Main installation flow
main() {
    print_banner
    check_termux
    
    log_info "Starting EXO installation for Termux/Android..."
    log_info "This will take approximately 30-45 minutes."
    echo ""
    
    install_base_packages
    install_rust_nightly
    build_llama_cpp
    configure_environment
    install_python_packages
    setup_exo_repo
    build_rust_bindings
    install_exo
    
    if verify_installation; then
        print_completion
    else
        echo ""
        log_error "Installation completed with errors. Please check the output above."
        log_info "You can try running the script again or follow the manual setup guide:"
        log_info "https://github.com/lukewrightmain/exo/blob/main/docs/ANDROID_SETUP.md"
        exit 1
    fi
}

# Run main
main "$@"

