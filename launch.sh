#!/bin/bash

# Exo Cluster Launch Script
# Starts both mini1 and mini2 nodes and FastAPI server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MINI2_USER="${MINI2_USER:-mini2}"
MINI2_HOST="192.168.5.2"
EXO_DIR="${EXO_DIR:-~/Movies/exo}"
LOG_DIR="./logs"
PID_FILE=".exo_pids"

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${GREEN}Starting Exo Cluster...${NC}"

# Function to check if mini2 is reachable
check_mini2() {
    echo -e "${YELLOW}Checking connection to mini2...${NC}"
    if ping -c 1 -W 1 "$MINI2_HOST" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ mini2 is reachable${NC}"
        return 0
    else
        echo -e "${RED}✗ Cannot reach mini2 at $MINI2_HOST${NC}"
        return 1
    fi
}

# Function to sync files to mini2
sync_to_mini2() {
    echo -e "${YELLOW}Syncing files to mini2...${NC}"
    rsync -av --exclude='.venv' --exclude='logs' --exclude='.exo_pids' \
        ./ "${MINI2_USER}@${MINI2_HOST}:${EXO_DIR}/" || {
        echo -e "${RED}Failed to sync files to mini2${NC}"
        return 1
    }
    echo -e "${GREEN}✓ Files synced to mini2${NC}"
}

# Function to setup mini2 environment
setup_mini2() {
    echo -e "${YELLOW}Setting up mini2 environment...${NC}"
    ssh "${MINI2_USER}@${MINI2_HOST}" "cd ${EXO_DIR} && bash -c '
        # Add pipx and homebrew to PATH
        export PATH=\"/Users/mini2/.local/bin:/opt/homebrew/bin:\$PATH\"
        
        # Check if uv is installed
        if ! command -v uv &> /dev/null; then
            echo \"Error: uv not found in PATH\"
            exit 1
        fi
        
        if [ ! -d .venv ]; then
            echo \"Creating virtual environment on mini2...\"
            uv venv .venv --python 3.12
            source .venv/bin/activate
            uv pip install -e .
        else
            echo \"Virtual environment already exists on mini2\"
        fi
    '" || {
        echo -e "${RED}Failed to setup mini2 environment${NC}"
        return 1
    }
    echo -e "${GREEN}✓ mini2 environment ready${NC}"
}

# Function to start mini2 node
start_mini2() {
    echo -e "${YELLOW}Starting exo on mini2...${NC}"
    
    # Create logs directory first
    ssh "${MINI2_USER}@${MINI2_HOST}" "mkdir -p ${EXO_DIR}/logs"
    
    # Start the exo process
    ssh "${MINI2_USER}@${MINI2_HOST}" "cd ${EXO_DIR} && bash -c '
        # Add pipx and homebrew to PATH
        export PATH=\"/Users/mini2/.local/bin:/opt/homebrew/bin:\$PATH\"
        source .venv/bin/activate
        
        # Start exo in background
        nohup exo \
            --node-id mini2 \
            --node-host 192.168.5.2 \
            --node-port 50051 \
            --discovery-module manual \
            --discovery-config-path discovery_config.json \
            --inference-engine mlx \
            --chatgpt-api-port 8003 \
            > logs/mini2.log 2>&1 &
        
        # Save PID
        echo \$! > .mini2_pid
        
        # Wait a moment and check if it started
        sleep 2
        if kill -0 \$(cat .mini2_pid) 2>/dev/null; then
            echo \"mini2 exo process started with PID \$(cat .mini2_pid)\"
            exit 0
        else
            echo \"Failed to start mini2 exo process\"
            cat logs/mini2.log 2>/dev/null || true
            exit 1
        fi
    '"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ mini2 node started${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to start mini2 node${NC}"
        echo -e "${YELLOW}Checking mini2 logs...${NC}"
        ssh "${MINI2_USER}@${MINI2_HOST}" "tail -20 ${EXO_DIR}/logs/mini2.log 2>/dev/null || echo 'No logs available'"
        return 1
    fi
}

# Function to start mini1 node
start_mini1() {
    echo -e "${YELLOW}Starting exo on mini1...${NC}"
    source .venv/bin/activate
    
    exo \
        --node-id mini1 \
        --node-host 192.168.2.13 \
        --node-port 50051 \
        --discovery-module manual \
        --discovery-config-path discovery_config.json \
        --inference-engine mlx \
        --chatgpt-api-port 8000 \
        > "$LOG_DIR/mini1.log" 2>&1 &
    
    local MINI1_PID=$!
    echo "$MINI1_PID" > "$PID_FILE"
    
    sleep 3
    
    if kill -0 "$MINI1_PID" 2>/dev/null; then
        echo -e "${GREEN}✓ mini1 node started (PID: $MINI1_PID)${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to start mini1 node${NC}"
        return 1
    fi
}

# Function to start FastAPI server
start_fastapi() {
    echo -e "${YELLOW}Starting FastAPI server...${NC}"
    source .venv/bin/activate
    
    python fastapi_server.py > "$LOG_DIR/fastapi.log" 2>&1 &
    local FASTAPI_PID=$!
    echo "$FASTAPI_PID" >> "$PID_FILE"
    
    sleep 2
    
    if kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo -e "${GREEN}✓ FastAPI server started (PID: $FASTAPI_PID)${NC}"
        echo -e "${GREEN}✓ API available at: http://localhost:8800${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to start FastAPI server${NC}"
        return 1
    fi
}

# Function to monitor cluster health
monitor_cluster() {
    echo -e "\n${GREEN}=== Cluster Status ===${NC}"
    echo -e "mini1 node: ${GREEN}Running${NC}"
    
    if ssh "${MINI2_USER}@${MINI2_HOST}" "cd ${EXO_DIR} && [ -f .mini2_pid ] && kill -0 \$(cat .mini2_pid) 2>/dev/null"; then
        echo -e "mini2 node: ${GREEN}Running${NC}"
    else
        echo -e "mini2 node: ${RED}Stopped${NC}"
    fi
    
    echo -e "\n${YELLOW}Logs:${NC}"
    echo "  mini1: tail -f $LOG_DIR/mini1.log"
    echo "  mini2: ssh ${MINI2_USER}@${MINI2_HOST} 'tail -f ${EXO_DIR}/logs/mini2.log'"
    echo "  FastAPI: tail -f $LOG_DIR/fastapi.log"
    
    echo -e "\n${YELLOW}API Endpoints:${NC}"
    echo "  ChatGPT API (mini1): http://192.168.2.13:8000"
    echo "  ChatGPT API (mini2): http://192.168.5.2:8003"
    echo "  FastAPI: http://localhost:8800"
    
    echo -e "\n${YELLOW}To stop the cluster:${NC} ./stop.sh"
}

# Main execution
main() {
    # Check prerequisites
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error: uv is not installed${NC}"
        echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    # Check and setup mini2
    if ! check_mini2; then
        echo -e "${RED}Cannot proceed without mini2 connection${NC}"
        exit 1
    fi
    
    # Sync files and setup mini2
    sync_to_mini2
    setup_mini2
    
    # Start nodes
    if ! start_mini2; then
        echo -e "${RED}Failed to start mini2, aborting...${NC}"
        exit 1
    fi
    
    if ! start_mini1; then
        echo -e "${RED}Failed to start mini1, stopping mini2...${NC}"
        ssh "${MINI2_USER}@${MINI2_HOST}" "cd ${EXO_DIR} && [ -f .mini2_pid ] && kill \$(cat .mini2_pid) 2>/dev/null; rm -f .mini2_pid"
        exit 1
    fi
    
    # Start FastAPI server if the script exists
    if [ -f "fastapi_server.py" ]; then
        start_fastapi
    else
        echo -e "${YELLOW}FastAPI server script not found, skipping...${NC}"
    fi
    
    # Show cluster status
    monitor_cluster
    
    echo -e "\n${GREEN}✓ Exo cluster successfully started!${NC}"
}

# Run main function
main