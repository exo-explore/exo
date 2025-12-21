#!/data/data/com.termux/files/usr/bin/bash
#
# Install Python 3.13 on Termux by building from source
# ======================================================
# This takes 15-30 minutes depending on your device
#
# Usage:
#   chmod +x scripts/termux_install_python313.sh
#   ./scripts/termux_install_python313.sh
#

set -e

PYTHON_VERSION="3.13.1"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Python ${PYTHON_VERSION} Installer for Termux        ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Check if running in Termux
if [ -z "$PREFIX" ]; then
    echo -e "${RED}This script must be run in Termux!${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Installing build dependencies...${NC}"
pkg update -y
pkg upgrade -y

# Install all required build dependencies
pkg install -y \
    build-essential \
    clang \
    make \
    pkg-config \
    libffi \
    openssl \
    zlib \
    bzip2 \
    liblzma \
    libsqlite \
    ncurses \
    readline \
    gdbm \
    2>/dev/null || {
        echo -e "${YELLOW}Some packages might have different names, trying alternatives...${NC}"
        pkg install -y python build-essential libffi openssl zlib || true
    }

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create build directory
BUILD_DIR="$HOME/python-build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo -e "${BLUE}Step 2: Downloading Python ${PYTHON_VERSION}...${NC}"

if [ -f "Python-${PYTHON_VERSION}.tgz" ]; then
    echo -e "${YELLOW}Found existing download, using cached file${NC}"
else
    curl -L -o "Python-${PYTHON_VERSION}.tgz" "$PYTHON_URL" || {
        wget -O "Python-${PYTHON_VERSION}.tgz" "$PYTHON_URL"
    }
fi

echo -e "${GREEN}✓ Downloaded${NC}"

echo ""
echo -e "${BLUE}Step 3: Extracting...${NC}"
tar xzf "Python-${PYTHON_VERSION}.tgz"
cd "Python-${PYTHON_VERSION}"
echo -e "${GREEN}✓ Extracted${NC}"

echo ""
echo -e "${BLUE}Step 4: Configuring build...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

# Configure with Termux-specific settings
./configure \
    --prefix="$PREFIX" \
    --enable-shared \
    --with-system-ffi \
    --with-openssl="$PREFIX" \
    --enable-optimizations \
    ac_cv_file__dev_ptmx=yes \
    ac_cv_file__dev_ptc=no \
    2>&1 | tail -5

echo -e "${GREEN}✓ Configured${NC}"

echo ""
echo -e "${BLUE}Step 5: Building Python (this takes 15-30 minutes)...${NC}"
echo -e "${YELLOW}Please be patient - your phone is compiling Python!${NC}"
echo ""

# Build using all available cores
CORES=$(nproc)
echo -e "${BLUE}Using $CORES CPU cores for compilation...${NC}"

make -j"$CORES" 2>&1 | tail -10

echo -e "${GREEN}✓ Build complete${NC}"

echo ""
echo -e "${BLUE}Step 6: Installing...${NC}"

# Install (may override existing python3)
make install 2>&1 | tail -5

echo -e "${GREEN}✓ Installed${NC}"

# Verify installation
echo ""
echo -e "${BLUE}Step 7: Verifying installation...${NC}"

INSTALLED_VERSION=$("$PREFIX/bin/python3.13" --version 2>&1 || echo "not found")
echo -e "   Python version: ${GREEN}${INSTALLED_VERSION}${NC}"

# Update .python-version if in exo directory
if [ -f "$HOME/exo/.python-version" ]; then
    echo "3.13" > "$HOME/exo/.python-version"
    echo -e "${GREEN}✓ Updated exo/.python-version${NC}"
fi

# Cleanup
echo ""
echo -e "${BLUE}Cleaning up build files...${NC}"
cd "$HOME"
rm -rf "$BUILD_DIR"
echo -e "${GREEN}✓ Cleaned up${NC}"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Python ${PYTHON_VERSION} Installation Complete!     ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
echo ""
echo "Python 3.13 is now installed. Verify with:"
echo "  python3.13 --version"
echo ""
echo "To use with uv in exo:"
echo "  cd ~/exo"
echo "  uv run exo"
echo ""

