#!/data/data/com.termux/files/usr/bin/bash
#
# exo Cluster Node Boot Script for Termux:Boot
# =============================================
# This script is designed to be placed in ~/.termux/boot/ to automatically
# start an exo cluster node when the Android device boots.
#
# Installation:
#   mkdir -p ~/.termux/boot
#   cp scripts/termux_boot_cluster.sh ~/.termux/boot/01-exo-cluster.sh
#   chmod +x ~/.termux/boot/01-exo-cluster.sh
#
# Requirements:
#   - Termux:Boot app installed from F-Droid
#   - Termux:Boot must be opened once to register
#   - Battery optimization disabled for Termux and Termux:Boot
#
# What this script does:
#   1. Acquires wake lock to prevent device sleep
#   2. Waits for network connectivity
#   3. Starts SSH server for remote management
#   4. Starts thermal monitoring (background)
#   5. Starts exo cluster node
#   6. Sends notification on success/failure
#

# Configuration
EXO_DIR="${HOME}/exo"
LOG_DIR="${HOME}/.exo/logs"
LOG_FILE="${LOG_DIR}/boot.log"
WAIT_FOR_NETWORK=30  # Seconds to wait for network
ENABLE_SSH=true
ENABLE_THERMAL_MONITOR=true

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

notify() {
    local title="$1"
    local content="$2"
    
    if command -v termux-notification &> /dev/null; then
        termux-notification \
            --title "$title" \
            --content "$content" \
            --id "exo-boot" \
            2>/dev/null || true
    fi
}

# ============================================================================
# Boot Sequence
# ============================================================================

log "=========================================="
log "exo cluster boot script started"
log "=========================================="

# Step 1: Acquire wake lock
log "Acquiring wake lock..."
if command -v termux-wake-lock &> /dev/null; then
    termux-wake-lock
    log "Wake lock acquired"
else
    log "WARNING: termux-wake-lock not available"
fi

# Step 2: Wait for network
log "Waiting for network connectivity..."
NETWORK_READY=false
for i in $(seq 1 $WAIT_FOR_NETWORK); do
    if ping -c 1 -W 1 8.8.8.8 &>/dev/null || ping -c 1 -W 1 1.1.1.1 &>/dev/null; then
        NETWORK_READY=true
        log "Network ready after ${i}s"
        break
    fi
    sleep 1
done

if [ "$NETWORK_READY" != "true" ]; then
    log "WARNING: Network not ready after ${WAIT_FOR_NETWORK}s, continuing anyway"
fi

# Get IP address for logging
IP_ADDR=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || echo "unknown")
log "Device IP: $IP_ADDR"

# Step 3: Start SSH server
if [ "$ENABLE_SSH" = "true" ]; then
    log "Starting SSH server..."
    if command -v sshd &> /dev/null; then
        # Kill any existing sshd
        pkill -9 sshd 2>/dev/null || true
        sleep 1
        
        # Start sshd
        sshd
        
        if pgrep -x sshd &>/dev/null; then
            log "SSH server started on port 8022"
        else
            log "WARNING: SSH server failed to start"
        fi
    else
        log "WARNING: sshd not installed (run: pkg install openssh)"
    fi
fi

# Step 4: Start thermal monitoring
if [ "$ENABLE_THERMAL_MONITOR" = "true" ]; then
    log "Starting thermal monitor..."
    THERMAL_SCRIPT="$EXO_DIR/scripts/thermal_monitor.sh"
    
    if [ -x "$THERMAL_SCRIPT" ]; then
        "$THERMAL_SCRIPT" --daemon
        log "Thermal monitor started"
    else
        log "WARNING: Thermal monitor script not found at $THERMAL_SCRIPT"
    fi
fi

# Step 5: Start exo cluster node
log "Starting exo cluster node..."

if [ -d "$EXO_DIR" ]; then
    cd "$EXO_DIR"
    
    # Activate virtual environment if it exists
    if [ -f "$EXO_DIR/venv/bin/activate" ]; then
        source "$EXO_DIR/venv/bin/activate"
        log "Activated virtual environment"
    fi
    
    # Start exo in background
    nohup python3 -m exo >> "$LOG_DIR/exo.log" 2>&1 &
    EXO_PID=$!
    
    # Give it a moment to start
    sleep 5
    
    if kill -0 $EXO_PID 2>/dev/null; then
        log "exo started successfully (PID: $EXO_PID)"
        echo $EXO_PID > "$HOME/.exo/exo.pid"
        
        notify "✓ exo Cluster Node" "Node started on $IP_ADDR"
    else
        log "ERROR: exo failed to start"
        notify "✗ exo Failed" "Check logs: $LOG_DIR/exo.log"
    fi
else
    log "ERROR: exo directory not found at $EXO_DIR"
    notify "✗ exo Not Found" "Clone exo to $EXO_DIR first"
fi

# Step 6: Log completion
log "Boot sequence completed"
log "=========================================="
log ""

# Keep script running briefly to ensure all background processes are stable
sleep 10

