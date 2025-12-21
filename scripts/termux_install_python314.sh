#!/data/data/com.termux/files/usr/bin/bash
#
# Install Python 3.14 on Termux using pre-built packages
# =======================================================
# Uses community packages from yubrajbhoi/termux-python
# This is faster than building from source!
#
# Usage:
#   chmod +x scripts/termux_install_python314.sh
#   ./scripts/termux_install_python314.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Python 3.14 Quick Installer for Termux       ║${NC}"
echo -e "${CYAN}║  (Using pre-built community packages)         ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Check if running in Termux
if [ -z "$PREFIX" ]; then
    echo -e "${RED}This script must be run in Termux!${NC}"
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${RED}This script only supports ARM64 (aarch64) devices.${NC}"
    echo -e "Your architecture: $ARCH"
    exit 1
fi

echo -e "${BLUE}Architecture: $ARCH ✓${NC}"
echo ""

# Option 1: Try yubrajbhoi's packages
echo -e "${BLUE}Downloading Python 3.14 pre-built packages...${NC}"
echo -e "${YELLOW}Source: github.com/yubrajbhoi/termux-python${NC}"
echo ""

BUILD_DIR="$HOME/python314-install"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download the packages
REPO_URL="https://github.com/yubrajbhoi/termux-python/releases/download/v3.14.0a6"

echo -e "${BLUE}Downloading libpython3.14...${NC}"
curl -L -O "${REPO_URL}/libpython3.14_3.14.0a6_aarch64.deb" || {
    echo -e "${RED}Download failed. Trying alternative source...${NC}"
    
    # Try alternative: Fibogacci's packages
    echo ""
    echo -e "${BLUE}Trying Fibogacci's Python 3.14 packages...${NC}"
    
    FIBO_URL="https://github.com/fibogacci/python314t-for-termux/releases/download/v3.14.0"
    curl -L -O "${FIBO_URL}/termux-python314t-3.14.0.tar.gz" || {
        echo -e "${RED}Both sources failed. Check your internet connection.${NC}"
        exit 1
    }
    
    echo -e "${BLUE}Extracting...${NC}"
    tar xzf termux-python314t-3.14.0.tar.gz
    cd termux-python314t-3.14.0
    
    echo -e "${BLUE}Installing...${NC}"
    bash install.sh
    
    echo -e "${GREEN}✓ Python 3.14 installed via Fibogacci's package${NC}"
    
    # Update .python-version
    if [ -d "$HOME/exo" ]; then
        echo "3.14" > "$HOME/exo/.python-version"
        echo -e "${GREEN}✓ Updated exo/.python-version to 3.14${NC}"
    fi
    
    cd "$HOME"
    rm -rf "$BUILD_DIR"
    
    echo ""
    echo -e "${GREEN}Done! Verify with: python3.14 --version${NC}"
    exit 0
}

echo -e "${BLUE}Downloading python3.14...${NC}"
curl -L -O "${REPO_URL}/python3.14_3.14.0a6_aarch64.deb"

echo ""
echo -e "${BLUE}Installing packages...${NC}"

# Install the deb packages
dpkg -i libpython3.14_3.14.0a6_aarch64.deb 2>/dev/null || apt-get install -f -y
dpkg -i python3.14_3.14.0a6_aarch64.deb 2>/dev/null || apt-get install -f -y

echo -e "${GREEN}✓ Python 3.14 installed${NC}"

# Verify
echo ""
echo -e "${BLUE}Verifying installation...${NC}"
python3.14 --version 2>/dev/null || {
    echo -e "${YELLOW}python3.14 not found in PATH, trying alternatives...${NC}"
    "$PREFIX/bin/python3.14" --version 2>/dev/null || {
        echo -e "${RED}Installation may have failed. Try the source build script instead.${NC}"
        exit 1
    }
}

# Update .python-version for exo
if [ -d "$HOME/exo" ]; then
    echo "3.14" > "$HOME/exo/.python-version"
    echo -e "${GREEN}✓ Updated exo/.python-version to 3.14${NC}"
fi

# Cleanup
cd "$HOME"
rm -rf "$BUILD_DIR"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Python 3.14 Installation Complete!           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
echo ""
echo "Verify with:"
echo "  python3.14 --version"
echo ""
echo "To use with uv in exo:"
echo "  cd ~/exo"
echo "  uv run exo"
echo ""
echo -e "${YELLOW}Note: Python 3.14 is newer than 3.13 and should work with exo.${NC}"
echo ""

