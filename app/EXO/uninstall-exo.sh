#!/usr/bin/env bash
#
# EXO Uninstaller Script
#
# This script removes all EXO system components that persist after deleting the app.
# Run with: sudo ./uninstall-exo.sh [--keep-models]
#
# Options:
#   --keep-models   Preserve ~/.exo/models when removing the EXO data directory.
#
# Components removed:
# - LaunchDaemon: /Library/LaunchDaemons/io.exo.networksetup.plist
# - Network script: /Library/Application Support/EXO/
# - Log files: /var/log/io.exo.networksetup.*
# - Network location: "exo"
# - EXO data directory: ~/.exo (or all of ~/.exo except models/ when --keep-models is set)
# - Launch at login registration
#

set -euo pipefail

KEEP_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --keep-models)
      KEEP_MODELS=1
      ;;
    -h | --help)
      echo "Usage: sudo ./uninstall-exo.sh [--keep-models]"
      echo "  --keep-models   Preserve ~/.exo/models when removing the EXO data directory."
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: sudo ./uninstall-exo.sh [--keep-models]" >&2
      exit 2
      ;;
  esac
done

LABEL="io.exo.networksetup"
# Current script path. Older installs used a different filename; keep the
# legacy path here so a fresh uninstall still cleans up upgraded machines.
CURRENT_SCRIPT_DEST="/Library/Application Support/EXO/disable_bridge.sh"
LEGACY_SCRIPT_DEST="/Library/Application Support/EXO/disable_bridge_enable_dhcp.sh"
PLIST_DEST="/Library/LaunchDaemons/io.exo.networksetup.plist"
LOG_OUT="/var/log/${LABEL}.log"
LOG_ERR="/var/log/${LABEL}.err.log"
APP_BUNDLE_ID="io.exo.EXO"

# Resolve the invoking user's home, even when run via sudo.
USER_HOME="$(eval echo "~${SUDO_USER:-$USER}")"
EXO_DIR="$USER_HOME/.exo"

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
if [[ -f $PLIST_DEST ]]; then
  rm -f "$PLIST_DEST"
  echo_info "Removed LaunchDaemon plist"
else
  echo_warn "LaunchDaemon plist not found (already removed?)"
fi

# Remove the script (current and legacy filenames) — backwards-compatible:
# tolerate either, both, or neither being present.
removed_any_script=0
for script in "$CURRENT_SCRIPT_DEST" "$LEGACY_SCRIPT_DEST"; do
  if [[ -f $script ]]; then
    rm -f "$script"
    echo_info "Removed network setup script: $script"
    removed_any_script=1
  fi
done
if [[ $removed_any_script -eq 0 ]]; then
  echo_warn "Network setup script not found (already removed?)"
fi

# Remove EXO directory if empty
if [[ -d "/Library/Application Support/EXO" ]]; then
  rmdir "/Library/Application Support/EXO" 2>/dev/null &&
    echo_info "Removed EXO support directory" ||
    echo_warn "EXO support directory not empty, leaving in place"
fi

# Remove log files
if [[ -f $LOG_OUT ]] || [[ -f $LOG_ERR ]]; then
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

# Remove EXO data directory (~/.exo)
EXO_DIR_REMOVED=""
if [[ -d $EXO_DIR ]]; then
  if [[ $KEEP_MODELS == "1" && -d "$EXO_DIR/models" ]]; then
    find "$EXO_DIR" -mindepth 1 -maxdepth 1 ! -name models -exec rm -rf {} +
    EXO_DIR_REMOVED="kept_models"
    echo_info "Removed ~/.exo (preserved models/)"
  else
    rm -rf "$EXO_DIR"
    EXO_DIR_REMOVED="full"
    echo_info "Removed ~/.exo"
  fi
else
  echo_warn "~/.exo not found (already removed?)"
fi

# Note about launch at login registration
# SMAppService-based login items cannot be removed from a shell script.
# They can only be unregistered from within the app itself or manually via System Settings.
echo_warn "Launch at login must be removed manually:"
echo_warn "  System Settings → General → Login Items → Remove EXO"

# Check if EXO.app exists in common locations
APP_FOUND=false
for app_path in "/Applications/EXO.app" "$HOME/Applications/EXO.app"; do
  if [[ -d $app_path ]]; then
    if [[ $APP_FOUND == false ]]; then
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
case "$EXO_DIR_REMOVED" in
  full) echo "  • EXO data directory (~/.exo)" ;;
  kept_models) echo "  • EXO data directory (~/.exo, models preserved)" ;;
esac
echo ""
echo "Your network has been restored to use the 'Automatic' location."
echo "Thunderbolt Bridge has been re-enabled (if present)."
echo ""
echo "Manual step required:"
echo "  Remove EXO from Login Items in System Settings → General → Login Items"
echo ""
