#!/bin/bash

# Exo Cluster Stop Script
# Stops all running exo nodes and FastAPI server

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
PID_FILE=".exo_pids"

echo -e "${YELLOW}Stopping Exo Cluster...${NC}"

# Function to stop mini1 processes
stop_mini1() {
    echo -e "${YELLOW}Stopping mini1 processes...${NC}"
    
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
                echo -e "${GREEN}✓ Stopped process $pid${NC}"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    else
        echo -e "${YELLOW}No PID file found for mini1${NC}"
    fi
    
    # Also try to stop any exo processes
    pkill -f "exo.*mini1" 2>/dev/null || true
    pkill -f "fastapi_server.py" 2>/dev/null || true
}

# Function to stop mini2 processes
stop_mini2() {
    echo -e "${YELLOW}Stopping mini2 processes...${NC}"
    
    # Check if mini2 is reachable
    if ! ping -c 1 -W 1 "$MINI2_HOST" > /dev/null 2>&1; then
        echo -e "${RED}Cannot reach mini2, it may already be stopped${NC}"
        return
    fi
    
    # Stop mini2 exo process
    ssh "${MINI2_USER}@${MINI2_HOST}" "
        cd ${EXO_DIR}
        if [ -f .mini2_pid ]; then
            pid=\$(cat .mini2_pid)
            if kill -0 \$pid 2>/dev/null; then
                kill \$pid 2>/dev/null
                echo 'Stopped mini2 process'
            fi
            rm -f .mini2_pid
        fi
        # Also try pkill as backup
        pkill -f 'exo.*mini2' 2>/dev/null || true
    " 2>/dev/null || echo -e "${YELLOW}Could not stop mini2 processes${NC}"
    
    echo -e "${GREEN}✓ mini2 processes stopped${NC}"
}

# Function to clean up logs
clean_logs() {
    read -p "Do you want to clean up log files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cleaning up logs...${NC}"
        rm -rf logs/*.log 2>/dev/null || true
        
        # Clean mini2 logs
        ssh "${MINI2_USER}@${MINI2_HOST}" "
            cd ${EXO_DIR}
            rm -rf logs/*.log 2>/dev/null || true
        " 2>/dev/null || true
        
        echo -e "${GREEN}✓ Logs cleaned${NC}"
    fi
}

# Function to show final status
show_status() {
    echo -e "\n${GREEN}=== Cluster Status ===${NC}"
    
    # Check if any exo processes are still running
    if pgrep -f "exo" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Some exo processes may still be running${NC}"
        echo "Running processes:"
        ps aux | grep -E "exo|fastapi_server" | grep -v grep || true
    else
        echo -e "${GREEN}✓ All local exo processes stopped${NC}"
    fi
    
    # Check mini2
    if ping -c 1 -W 1 "$MINI2_HOST" > /dev/null 2>&1; then
        if ssh "${MINI2_USER}@${MINI2_HOST}" "pgrep -f 'exo' > /dev/null 2>&1"; then
            echo -e "${YELLOW}Warning: Some processes may still be running on mini2${NC}"
        else
            echo -e "${GREEN}✓ All mini2 processes stopped${NC}"
        fi
    fi
}

# Main execution
main() {
    # Stop mini2 first (remote)
    stop_mini2
    
    # Stop mini1 (local)
    stop_mini1
    
    # Optional: clean logs
    clean_logs
    
    # Show final status
    show_status
    
    echo -e "\n${GREEN}✓ Exo cluster stopped${NC}"
}

# Trap to ensure cleanup on script exit
trap 'echo -e "\n${RED}Interrupted${NC}"; exit 1' INT TERM

# Run main function
main