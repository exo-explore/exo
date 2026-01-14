#!/usr/bin/env bash
#
# EXO Uninstaller Script
#
# This script removes all EXO system components that persist after deleting the app.
# Run with: sudo ./uninstall-exo.sh
#
# Components removed:
# - LaunchDaemon: /Library/LaunchDaemons/io.exo.networksetup.plist
# - Network script: /Library/Application Support/EXO/
# - Log files: /var/log/io.exo.networksetup.*
# - Network location: "exo"
# - Launch at login registration
#

set -euo pipefail

LABEL="io.exo.networksetup"
SCRIPT_DEST="/Library/Application Support/EXO/disable_bridge_enable_dhcp.sh"
PLIST_DEST="/Library/LaunchDaemons/io.exo.networksetup.plist"
LOG_OUT="/var/log/${LABEL}.log"
LOG_ERR="/var/log/${LABEL}.err.log"
APP_BUNDLE_ID="io.exo.EXO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo_error "This script must be run as root (use sudo)"
    exit 1
fi

echo ""
echo "========================================"
echo "        EXO Uninstaller"
echo "========================================"
echo ""

# Unload the LaunchDaemon if running
echo_info "Stopping network setup daemon..."
if launchctl list | grep -q "$LABEL"; then
    launchctl bootout system/"$LABEL" 2>/dev/null || true
    echo_info "Daemon stopped"
else
    echo_warn "Daemon was not running"
fi

# Remove LaunchDaemon plist
if [[ -f "$PLIST_DEST" ]]; then
    rm -f "$PLIST_DEST"
    echo_info "Removed LaunchDaemon plist"
else
    echo_warn "LaunchDaemon plist not found (already removed?)"
fi

# Remove the script and parent directory
if [[ -f "$SCRIPT_DEST" ]]; then
    rm -f "$SCRIPT_DEST"
    echo_info "Removed network setup script"
else
    echo_warn "Network setup script not found (already removed?)"
fi

# Remove EXO directory if empty
if [[ -d "/Library/Application Support/EXO" ]]; then
    rmdir "/Library/Application Support/EXO" 2>/dev/null && \
        echo_info "Removed EXO support directory" || \
        echo_warn "EXO support directory not empty, leaving in place"
fi

# Remove log files
if [[ -f "$LOG_OUT" ]] || [[ -f "$LOG_ERR" ]]; then
    rm -f "$LOG_OUT" "$LOG_ERR"
    echo_info "Removed log files"
else
    echo_warn "Log files not found (already removed?)"
fi

# Switch back to Automatic network location
echo_info "Restoring network configuration..."
if networksetup -listlocations | grep -q "^Automatic$"; then
    networksetup -switchtolocation Automatic 2>/dev/null || true
    echo_info "Switched to Automatic network location"
else
    echo_warn "Automatic network location not found"
fi

# Delete the exo network location if it exists
if networksetup -listlocations | grep -q "^exo$"; then
    networksetup -deletelocation exo 2>/dev/null || true
    echo_info "Deleted 'exo' network location"
else
    echo_warn "'exo' network location not found (already removed?)"
fi

# Re-enable Thunderbolt Bridge if it exists
if networksetup -listnetworkservices 2>/dev/null | grep -q "Thunderbolt Bridge"; then
    networksetup -setnetworkserviceenabled "Thunderbolt Bridge" on 2>/dev/null || true
    echo_info "Re-enabled Thunderbolt Bridge"
fi

# Note about launch at login registration
# SMAppService-based login items cannot be removed from a shell script.
# They can only be unregistered from within the app itself or manually via System Settings.
echo_warn "Launch at login must be removed manually:"
echo_warn "  System Settings → General → Login Items → Remove EXO"

# Check if EXO.app exists in common locations
APP_FOUND=false
for app_path in "/Applications/EXO.app" "$HOME/Applications/EXO.app"; do
    if [[ -d "$app_path" ]]; then
        if [[ "$APP_FOUND" == false ]]; then
            echo ""
            APP_FOUND=true
        fi
        echo_warn "EXO.app found at: $app_path"
        echo_warn "You may want to move it to Trash manually."
    fi
done

echo ""
echo "========================================"
echo_info "EXO uninstall complete!"
echo "========================================"
echo ""
echo "The following have been removed:"
echo "  • Network setup LaunchDaemon"
echo "  • Network configuration script"
echo "  • Log files"
echo "  • 'exo' network location"
echo ""
echo "Your network has been restored to use the 'Automatic' location."
echo "Thunderbolt Bridge has been re-enabled (if present)."
echo ""
echo "Manual step required:"
echo "  Remove EXO from Login Items in System Settings → General → Login Items"
echo ""

