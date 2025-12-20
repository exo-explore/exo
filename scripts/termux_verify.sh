#!/data/data/com.termux/files/usr/bin/bash
#
# exo Termux Verification Script
# ================================
# Verify that exo is properly installed.
#
# Usage:
#   chmod +x scripts/termux_verify.sh
#   ./scripts/termux_verify.sh
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}=== exo Installation Verification ===${NC}"
echo ""

# System info
echo -e "${BLUE}--- System Info ---${NC}"
echo "Platform: $(uname -s) $(uname -m)"
echo "Termux: ${TERMUX_VERSION:-not detected}"
if [ -f /proc/cpuinfo ]; then
    CPU_CORES=$(grep -c processor /proc/cpuinfo 2>/dev/null || echo "unknown")
    echo "CPU Cores: $CPU_CORES"
fi
echo ""

# Python info
echo -e "${BLUE}--- Python ---${NC}"
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Python: $(python3 --version)"
    echo "  Path: $(which python3)"
else
    echo -e "${RED}✗${NC} Python not found!"
fi
echo ""

# Memory info
echo -e "${BLUE}--- Memory ---${NC}"
if command -v free &> /dev/null; then
    free -h 2>/dev/null || echo "Could not read memory info"
else
    echo "free command not available"
fi
echo ""

# Storage info
echo -e "${BLUE}--- Storage ---${NC}"
if [ -d "$HOME/.exo" ]; then
    echo -e "${GREEN}✓${NC} exo directory: ~/.exo"
    du -sh ~/.exo 2>/dev/null || echo "  Size: unknown"
else
    echo -e "${YELLOW}⚠${NC} exo directory not found (will be created on first run)"
fi
echo ""

# Check llama-cpp-python
echo -e "${BLUE}--- llama-cpp-python ---${NC}"
python3 << 'EOF'
try:
    import llama_cpp
    print(f"\033[0;32m✓\033[0m Version: {llama_cpp.__version__}")
    print(f"  Status: Installed and working")
except ImportError as e:
    print(f"\033[0;31m✗\033[0m Not installed or import error")
    print(f"  Error: {e}")
except Exception as e:
    print(f"\033[1;33m⚠\033[0m Unexpected error: {e}")
EOF
echo ""

# Check exo modules
echo -e "${BLUE}--- exo Modules ---${NC}"
python3 << 'EOF'
modules_to_check = [
    ("exo.shared.platform", "Platform detection"),
    ("exo.shared.types.worker.instances", "Instance types"),
    ("exo.worker.engines.llamacpp.generate", "llama.cpp engine"),
]

for module, description in modules_to_check:
    try:
        __import__(module)
        print(f"\033[0;32m✓\033[0m {description}")
    except ImportError as e:
        print(f"\033[0;31m✗\033[0m {description}: {e}")
    except Exception as e:
        print(f"\033[1;33m⚠\033[0m {description}: {e}")
EOF
echo ""

# Check platform detection
echo -e "${BLUE}--- Platform Detection ---${NC}"
python3 << 'EOF'
try:
    from exo.shared.platform import get_platform_info, get_recommended_backend, is_android
    info = get_platform_info()
    print(f"  System: {info['system']}")
    print(f"  Machine: {info['machine']}")
    print(f"  Is Android: {is_android()}")
    print(f"  Recommended backend: {get_recommended_backend()}")
except Exception as e:
    print(f"  Could not detect: {e}")
EOF
echo ""

# Check models directory
echo -e "${BLUE}--- Downloaded Models ---${NC}"
if [ -d "$HOME/.exo/models" ]; then
    MODEL_COUNT=$(find ~/.exo/models -name "*.gguf" 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓${NC} Found $MODEL_COUNT GGUF model(s):"
        find ~/.exo/models -name "*.gguf" -exec basename {} \; 2>/dev/null | while read model; do
            echo "    • $model"
        done
    else
        echo -e "${YELLOW}⚠${NC} No GGUF models found"
        echo "  Run: ./scripts/download_model.sh tinyllama"
    fi
else
    echo -e "${YELLOW}⚠${NC} Models directory not found"
fi
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
python3 << 'EOF'
import sys

issues = []
warnings = []

# Check llama-cpp-python
try:
    import llama_cpp
except ImportError:
    issues.append("llama-cpp-python not installed")

# Check exo
try:
    from exo.shared.platform import get_recommended_backend
    backend = get_recommended_backend()
    if backend != "llamacpp":
        warnings.append(f"Recommended backend is '{backend}', expected 'llamacpp'")
except ImportError:
    issues.append("exo platform module not working")

# Check engine
try:
    from exo.worker.engines.llamacpp.generate import llamacpp_generate
except ImportError as e:
    if "llama_cpp" in str(e):
        # This is expected if llama-cpp-python is not installed
        pass
    else:
        issues.append(f"llama.cpp engine issue: {e}")

if issues:
    print(f"\033[0;31m✗ {len(issues)} issue(s) found:\033[0m")
    for issue in issues:
        print(f"  • {issue}")
elif warnings:
    print(f"\033[1;33m⚠ Installation OK with {len(warnings)} warning(s)\033[0m")
    for warning in warnings:
        print(f"  • {warning}")
else:
    print("\033[0;32m✓ All checks passed!\033[0m")
EOF
echo ""
