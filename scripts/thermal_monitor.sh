#!/data/data/com.termux/files/usr/bin/bash
#
# Thermal Monitoring Script for exo on Android/Termux
# =====================================================
# Monitors device temperature and throttles inference workloads to prevent
# overheating and maintain sustained performance.
#
# Based on research from docs/ARM_CORTEX_OPTIMIZATION_GUIDE.md
#
# Usage:
#   chmod +x scripts/thermal_monitor.sh
#   ./scripts/thermal_monitor.sh                    # Run in foreground
#   ./scripts/thermal_monitor.sh --daemon           # Run in background
#   ./scripts/thermal_monitor.sh --status           # Check current status
#   ./scripts/thermal_monitor.sh --stop             # Stop background daemon
#
# Temperature Thresholds:
#   - PAUSE:  42°C - Pause compute workloads temporarily
#   - RESUME: 38°C - Resume when cooled down
#   - WARN:   45°C - Show warning notification
#

set -e

# Configuration
TEMP_PAUSE_THRESHOLD=42      # Pause at this temperature (°C)
TEMP_RESUME_THRESHOLD=38     # Resume at this temperature (°C)
TEMP_WARN_THRESHOLD=45       # Show warning at this temperature (°C)
CHECK_INTERVAL=10            # Check every N seconds
PID_FILE="$HOME/.exo/thermal_monitor.pid"
LOG_FILE="$HOME/.exo/logs/thermal.log"
PROCESS_PATTERNS="python|llama|exo"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# ============================================================================
# Temperature Reading Functions
# ============================================================================

get_battery_temp() {
    # Try termux-api first (most accurate on Android)
    if command -v termux-battery-status &> /dev/null; then
        local temp=$(termux-battery-status 2>/dev/null | grep -o '"temperature":[0-9.]*' | cut -d: -f2)
        if [[ -n "$temp" ]]; then
            # Battery temp is in tenths of °C
            echo "$temp"
            return 0
        fi
    fi
    return 1
}

get_thermal_zone_temp() {
    # Try reading from thermal zones (works on most Android devices)
    local max_temp=0
    
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        if [[ -f "$zone" ]]; then
            local temp=$(cat "$zone" 2>/dev/null || echo "0")
            # Thermal zones report in millidegrees
            temp=$((temp / 1000))
            if [[ $temp -gt $max_temp ]]; then
                max_temp=$temp
            fi
        fi
    done
    
    if [[ $max_temp -gt 0 ]]; then
        echo "$max_temp"
        return 0
    fi
    return 1
}

get_temperature() {
    # Try battery temp first, then thermal zones
    local temp
    
    temp=$(get_battery_temp)
    if [[ $? -eq 0 ]] && [[ -n "$temp" ]]; then
        echo "$temp"
        return 0
    fi
    
    temp=$(get_thermal_zone_temp)
    if [[ $? -eq 0 ]] && [[ -n "$temp" ]]; then
        echo "$temp"
        return 0
    fi
    
    # Fallback - return safe value
    echo "35"
    return 1
}

# ============================================================================
# Process Control Functions
# ============================================================================

get_target_pids() {
    pgrep -f "$PROCESS_PATTERNS" 2>/dev/null || true
}

pause_processes() {
    local pids=$(get_target_pids)
    local count=0
    
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -STOP "$pid" 2>/dev/null && ((count++))
        fi
    done
    
    echo "$count"
}

resume_processes() {
    local pids=$(get_target_pids)
    local count=0
    
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -CONT "$pid" 2>/dev/null && ((count++))
        fi
    done
    
    echo "$count"
}

# ============================================================================
# Notification Functions
# ============================================================================

notify() {
    local title="$1"
    local content="$2"
    
    if command -v termux-notification &> /dev/null; then
        termux-notification \
            --title "$title" \
            --content "$content" \
            --id "exo-thermal" \
            2>/dev/null || true
    fi
}

notify_vibrate() {
    if command -v termux-vibrate &> /dev/null; then
        termux-vibrate -d 200 2>/dev/null || true
    fi
}

# ============================================================================
# Logging
# ============================================================================

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    if [[ "$DAEMON_MODE" != "true" ]]; then
        case "$level" in
            INFO)  echo -e "${BLUE}[$timestamp]${NC} $message" ;;
            WARN)  echo -e "${YELLOW}[$timestamp]${NC} ⚠ $message" ;;
            ERROR) echo -e "${RED}[$timestamp]${NC} ✗ $message" ;;
            OK)    echo -e "${GREEN}[$timestamp]${NC} ✓ $message" ;;
        esac
    fi
}

# ============================================================================
# Monitor Loop
# ============================================================================

PAUSED=false
LAST_WARN=0

monitor_loop() {
    log "INFO" "Thermal monitor started (pause: ${TEMP_PAUSE_THRESHOLD}°C, resume: ${TEMP_RESUME_THRESHOLD}°C)"
    
    while true; do
        local temp=$(get_temperature)
        local current_time=$(date +%s)
        
        # High temperature warning (max once per 5 minutes)
        if [[ $temp -ge $TEMP_WARN_THRESHOLD ]]; then
            if [[ $((current_time - LAST_WARN)) -gt 300 ]]; then
                log "WARN" "High temperature: ${temp}°C"
                notify "⚠ High Temperature" "Device at ${temp}°C. Consider cooling."
                notify_vibrate
                LAST_WARN=$current_time
            fi
        fi
        
        # Pause logic
        if [[ $temp -ge $TEMP_PAUSE_THRESHOLD ]] && [[ "$PAUSED" != "true" ]]; then
            local count=$(pause_processes)
            PAUSED=true
            log "WARN" "Temperature ${temp}°C >= ${TEMP_PAUSE_THRESHOLD}°C. Paused $count processes."
            notify "🌡 Thermal Throttle" "Paused $count exo processes (${temp}°C)"
        fi
        
        # Resume logic
        if [[ $temp -le $TEMP_RESUME_THRESHOLD ]] && [[ "$PAUSED" == "true" ]]; then
            local count=$(resume_processes)
            PAUSED=false
            log "OK" "Temperature ${temp}°C <= ${TEMP_RESUME_THRESHOLD}°C. Resumed $count processes."
            notify "✓ Thermal OK" "Resumed $count exo processes (${temp}°C)"
        fi
        
        # Periodic log (every 6 checks = ~1 minute)
        if [[ $((SECONDS % 60)) -lt $CHECK_INTERVAL ]]; then
            log "INFO" "Current temperature: ${temp}°C (paused: $PAUSED)"
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# ============================================================================
# Daemon Management
# ============================================================================

start_daemon() {
    if [[ -f "$PID_FILE" ]]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo -e "${YELLOW}Thermal monitor already running (PID: $old_pid)${NC}"
            return 1
        fi
        rm -f "$PID_FILE"
    fi
    
    DAEMON_MODE=true
    nohup "$0" --monitor >> "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    echo -e "${GREEN}Thermal monitor started in background (PID: $pid)${NC}"
    echo "Log file: $LOG_FILE"
    echo "Stop with: $0 --stop"
}

stop_daemon() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            rm -f "$PID_FILE"
            
            # Resume any paused processes
            resume_processes > /dev/null
            
            echo -e "${GREEN}Thermal monitor stopped (PID: $pid)${NC}"
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    echo -e "${YELLOW}Thermal monitor not running${NC}"
    return 1
}

show_status() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       exo Thermal Monitor Status             ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Current temperature
    local temp=$(get_temperature)
    echo -e "${BLUE}Current Temperature:${NC} ${temp}°C"
    
    if [[ $temp -ge $TEMP_PAUSE_THRESHOLD ]]; then
        echo -e "Status: ${RED}HOT - Throttling active${NC}"
    elif [[ $temp -ge $TEMP_RESUME_THRESHOLD ]]; then
        echo -e "Status: ${YELLOW}WARM${NC}"
    else
        echo -e "Status: ${GREEN}COOL${NC}"
    fi
    echo ""
    
    # Daemon status
    echo -e "${BLUE}Monitor Daemon:${NC}"
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${GREEN}Running${NC} (PID: $pid)"
        else
            echo -e "  ${YELLOW}Stale PID file${NC}"
        fi
    else
        echo -e "  ${YELLOW}Not running${NC}"
    fi
    echo ""
    
    # Thresholds
    echo -e "${BLUE}Thresholds:${NC}"
    echo "  Pause at:  ${TEMP_PAUSE_THRESHOLD}°C"
    echo "  Resume at: ${TEMP_RESUME_THRESHOLD}°C"
    echo "  Warn at:   ${TEMP_WARN_THRESHOLD}°C"
    echo ""
    
    # Monitored processes
    echo -e "${BLUE}Monitored Processes:${NC}"
    local pids=$(get_target_pids)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            local cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | cut -c1-60)
            echo "  PID $pid: $cmdline..."
        done
    else
        echo "  (none running)"
    fi
    echo ""
    
    # Recent log entries
    if [[ -f "$LOG_FILE" ]]; then
        echo -e "${BLUE}Recent Log:${NC}"
        tail -5 "$LOG_FILE" 2>/dev/null | while read line; do
            echo "  $line"
        done
    fi
    echo ""
}

# ============================================================================
# Main
# ============================================================================

case "${1:-}" in
    "--daemon"|"-d")
        start_daemon
        ;;
    "--stop")
        stop_daemon
        ;;
    "--status"|"-s")
        show_status
        ;;
    "--monitor")
        # Internal: called by daemon mode
        DAEMON_MODE=true
        monitor_loop
        ;;
    "--help"|"-h")
        echo "exo Thermal Monitor"
        echo ""
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  (none)         Run monitor in foreground"
        echo "  --daemon, -d   Run monitor in background"
        echo "  --stop         Stop background daemon"
        echo "  --status, -s   Show current status"
        echo "  --help, -h     Show this help"
        echo ""
        echo "Configuration (edit script to change):"
        echo "  TEMP_PAUSE_THRESHOLD  = ${TEMP_PAUSE_THRESHOLD}°C"
        echo "  TEMP_RESUME_THRESHOLD = ${TEMP_RESUME_THRESHOLD}°C"
        echo "  CHECK_INTERVAL        = ${CHECK_INTERVAL}s"
        echo ""
        echo "Log file: $LOG_FILE"
        echo ""
        ;;
    *)
        echo ""
        echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║       exo Thermal Monitor                    ║${NC}"
        echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BLUE}Thresholds:${NC} Pause at ${TEMP_PAUSE_THRESHOLD}°C, Resume at ${TEMP_RESUME_THRESHOLD}°C"
        echo -e "${BLUE}Monitoring:${NC} $PROCESS_PATTERNS"
        echo ""
        echo "Press Ctrl+C to stop"
        echo ""
        monitor_loop
        ;;
esac

