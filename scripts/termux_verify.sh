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

echo "=== exo Installation Verification ==="
echo ""

# System info
echo "--- System Info ---"
echo "Platform: $(uname -a)"
echo "Python: $(python3 --version)"
echo ""

# Memory info
echo "--- Memory ---"
free -h
echo ""

# Check llama-cpp-python
echo "--- llama-cpp-python ---"
python3 -c "
try:
    import llama_cpp
    print(f'Version: {llama_cpp.__version__}')
    print('Status: OK')
except ImportError as e:
    print(f'Status: FAILED - {e}')
"
echo ""

# Check exo
echo "--- exo ---"
python3 -c "
try:
    from exo.shared.platform import get_platform_info, get_recommended_backend, is_android
    info = get_platform_info()
    print(f'System: {info[\"system\"]}')
    print(f'Machine: {info[\"machine\"]}')
    print(f'Is Android: {is_android()}')
    print(f'Backend: {get_recommended_backend()}')
    print('Status: OK')
except Exception as e:
    print(f'Status: FAILED - {e}')
"
echo ""

# Check models directory
echo "--- Models ---"
if [ -d "$HOME/.exo/models" ]; then
    MODEL_COUNT=$(find ~/.exo/models -name "*.gguf" 2>/dev/null | wc -l)
    echo "Directory: ~/.exo/models"
    echo "GGUF files found: $MODEL_COUNT"
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "Models:"
        find ~/.exo/models -name "*.gguf" -exec basename {} \;
    fi
else
    echo "Directory not found!"
fi
echo ""

echo "=== Verification Complete ==="

