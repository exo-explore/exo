#!/bin/bash

# Launch script for testing MoE model with distributed inference

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="test_moe_distributed.py"
HOSTS_FILE="$SCRIPT_DIR/hosts.json"
LOG_FILE="$SCRIPT_DIR/moe_test.log"

echo -e "${GREEN}MoE Distributed Inference Test${NC}"
echo "================================="

# Check if hosts.json exists
if [ ! -f "$HOSTS_FILE" ]; then
    echo -e "${YELLOW}Creating hosts.json...${NC}"
    cat > "$HOSTS_FILE" << 'EOF'
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "192.168.5.2", "ips": ["192.168.5.2"]}
]
EOF
fi

# Function to check if mini2 is reachable
check_mini2() {
    echo -n "Checking connection to mini2..."
    if ping -c 1 -W 1 192.168.5.2 > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        return 0
    else
        echo -e " ${RED}✗${NC}"
        return 1
    fi
}

# Function to sync code to mini2
sync_code() {
    echo "Syncing code to mini2..."
    
    # Create directory on mini2
    ssh mini2@192.168.5.2 "mkdir -p /Users/mini2/Movies/exo/exo/inference/mlx/models" 2>/dev/null || true
    ssh mini2@192.168.5.2 "mkdir -p /Users/mini2/Movies/exo/exo/inference" 2>/dev/null || true
    
    # Copy necessary files
    scp -q "$SCRIPT_DIR/exo/inference/shard.py" mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/ || true
    scp -q "$SCRIPT_DIR/exo/inference/mlx/models/base.py" mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/mlx/models/ || true
    scp -q "$SCRIPT_DIR/exo/inference/mlx/models/qwen_moe_mini.py" mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/mlx/models/
    scp -q "$SCRIPT_DIR/test_moe_distributed.py" mini2@192.168.5.2:/Users/mini2/Movies/exo/
    scp -q "$SCRIPT_DIR/moe_launcher.py" mini2@192.168.5.2:/Users/mini2/Movies/exo/
    
    echo -e "Code synced ${GREEN}✓${NC}"
}

# Function to run distributed test
run_test() {
    echo -e "\n${YELLOW}Starting distributed MoE test...${NC}"
    
    # Check if we should run single-device or multi-device
    if check_mini2; then
        echo "Running in distributed mode (2 devices)"
        sync_code
        
        # Use mlx.launch with uv for distributed execution
        echo "Launching with mlx.launch via uv..."
        
        # Get the Python path from uv
        PYTHON_PATH=$(uv run which python)
        echo "Using Python: $PYTHON_PATH"
        
        # Run mlx.launch with explicit Python path
        MLX_PYTHON="$PYTHON_PATH" uv run mlx.launch \
            --hostfile "$HOSTS_FILE" \
            "$PYTHON_PATH" "$SCRIPT_DIR/moe_launcher.py" --mock-weights \
            2>&1 | tee "$LOG_FILE"
    else
        echo -e "${YELLOW}mini2 not available, running single-device test${NC}"
        python "$SCRIPT_DIR/$TEST_SCRIPT" --mock-weights 2>&1 | tee "$LOG_FILE"
    fi
    
    # Check if test succeeded
    if grep -q "SUCCESS" "$LOG_FILE"; then
        echo -e "\n${GREEN}✅ Test PASSED!${NC}"
        echo "Both GPUs are actively processing with the MoE model!"
        
        # Show memory usage
        echo -e "\n${YELLOW}Memory Usage Summary:${NC}"
        grep "Memory" "$LOG_FILE" | tail -4
        
        # Show layer distribution
        echo -e "\n${YELLOW}Layer Distribution:${NC}"
        grep "layers" "$LOG_FILE" | tail -2
        
        return 0
    else
        echo -e "\n${RED}❌ Test failed${NC}"
        echo "Check $LOG_FILE for details"
        return 1
    fi
}

# Function to test single device
test_single() {
    echo -e "\n${YELLOW}Testing single-device MoE...${NC}"
    python "$SCRIPT_DIR/$TEST_SCRIPT" --mock-weights 2>&1 | tee "${LOG_FILE}.single"
}

# Main command handling
case "${1:-test}" in
    test)
        run_test
        ;;
    single)
        test_single
        ;;
    sync)
        if check_mini2; then
            sync_code
        else
            echo -e "${RED}mini2 not reachable${NC}"
            exit 1
        fi
        ;;
    clean)
        echo "Cleaning up..."
        rm -f "$LOG_FILE" "${LOG_FILE}.single"
        echo -e "${GREEN}Cleaned${NC}"
        ;;
    *)
        echo "Usage: $0 {test|single|sync|clean}"
        echo "  test   - Run distributed MoE test (default)"
        echo "  single - Test on single device only"
        echo "  sync   - Just sync code to mini2"
        echo "  clean  - Clean up log files"
        exit 1
        ;;
esac