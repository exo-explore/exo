#!/usr/bin/env bash
#
# EXO Uninstaller Script
#
# Removes what EXO adds to a Mac, like Uninstall EXO in the app's Settings.
# Use it if you have already deleted the app.
# Run with: sudo ./uninstall-exo.sh [--delete-models]
#
# Options:
#   --delete-models  Also delete downloaded models (~/.exo/models). They are
#                    kept by default, as in the app.
#   --keep-models    Accepted so older commands keep working. Models are kept
#                    by default now, so it changes nothing.
#
# Components removed:
# - LaunchDaemon: /Library/LaunchDaemons/io.exo.networksetup.plist
# - Network script: /Library/Application Support/EXO/
# - Log files: /var/log/io.exo.networksetup.*
# - Network location: "exo" (the Mac is switched back to "Automatic" first)
# - EXO data directory: ~/.exo, except downloaded models in ~/.exo/models
#   unless --delete-models is given
#
# Restored:
# - The "Automatic" network location
# - Thunderbolt Bridge: its interface is recreated if EXO's network setup
#   deleted it, and the service is enabled again
#
# Not removed (the script says how):
# - EXO in Login Items
# - EXO.app itself
#

set -euo pipefail

usage() {
  echo "Usage: sudo ./uninstall-exo.sh [--delete-models]"
  echo "  --delete-models  Also delete downloaded models (~/.exo/models). They are kept by default."
}

DELETE_MODELS=0
KEEP_MODELS_GIVEN=0
for arg in "$@"; do
  case "$arg" in
  --delete-models)
    DELETE_MODELS=1
    ;;
  --keep-models)
    KEEP_MODELS_GIVEN=1
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $arg" >&2
    usage >&2
    exit 2
    ;;
  esac
done
if [[ $DELETE_MODELS == "1" && $KEEP_MODELS_GIVEN == "1" ]]; then
  echo "--delete-models and --keep-models can't be used together" >&2
  usage >&2
  exit 2
fi

LABEL="io.exo.networksetup"
SUPPORT_DIR="/Library/Application Support/EXO"
# Current script path. Older installs used a different filename; keep the
# legacy path here so a fresh uninstall still cleans up upgraded machines.
CURRENT_SCRIPT_DEST="$SUPPORT_DIR/disable_bridge.sh"
LEGACY_SCRIPT_DEST="$SUPPORT_DIR/disable_bridge_enable_dhcp.sh"
PLIST_DEST="/Library/LaunchDaemons/io.exo.networksetup.plist"
LOG_OUT="/var/log/${LABEL}.log"
LOG_ERR="/var/log/${LABEL}.err.log"
EXO_LOCATION="exo"
DEFAULT_LOCATION="Automatic"

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

# Unload the LaunchDaemon first, so it can't switch the network back to the
# exo location while it is being restored below.
echo_info "Stopping network setup daemon..."
if launchctl print system/"$LABEL" >/dev/null 2>&1; then
  launchctl bootout system/"$LABEL" 2>/dev/null || true
  echo_info "Daemon stopped"
else
  echo_warn "Daemon was not loaded"
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
if [[ -d $SUPPORT_DIR ]]; then
  rmdir "$SUPPORT_DIR" 2>/dev/null &&
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

# Leave the exo network location for Automatic, as the app's uninstall does.
# If the Mac is on another location the user chose, it stays there.
echo_info "Restoring network configuration..."
# grep reads all of its input (no -q), so pipefail can't turn a match into a
# failure while networksetup is still writing.
has_location() {
  networksetup -listlocations 2>/dev/null | grep -x "$1" >/dev/null
}
if [[ "$(networksetup -getcurrentlocation 2>/dev/null)" == "$EXO_LOCATION" ]]; then
  if ! has_location "$DEFAULT_LOCATION"; then
    # Without another location to switch to, the exo location couldn't be
    # deleted, so recreate Automatic with the default services.
    networksetup -createlocation "$DEFAULT_LOCATION" populate >/dev/null 2>&1 || true
    echo_info "Created the '$DEFAULT_LOCATION' network location"
  fi
  if networksetup -switchtolocation "$DEFAULT_LOCATION" >/dev/null 2>&1; then
    echo_info "Switched to the '$DEFAULT_LOCATION' network location"
  else
    echo_warn "Couldn't switch to the '$DEFAULT_LOCATION' network location"
  fi
else
  echo_info "Network location is not '$EXO_LOCATION'; leaving it as it is"
fi

# Delete the exo network location if it exists
if has_location "$EXO_LOCATION"; then
  networksetup -deletelocation "$EXO_LOCATION" >/dev/null 2>&1 || true
  if has_location "$EXO_LOCATION"; then
    echo_warn "Couldn't delete the '$EXO_LOCATION' network location"
  else
    echo_info "Deleted the '$EXO_LOCATION' network location"
  fi
else
  echo_warn "'$EXO_LOCATION' network location not found (already removed?)"
fi

# Thunderbolt Bridge. macOS bridges the Mac's Thunderbolt ports as bridge0.
# EXO's network setup deletes that interface (not just the service), so
# switching back to Automatic leaves a "Thunderbolt Bridge" service with no
# interface behind it. Recreate the interface the way macOS defines it, then
# make sure the service is enabled.
BRIDGE_DEVICE="bridge0"
NETWORK_PREFS="/Library/Preferences/SystemConfiguration/preferences.plist"

# The service in the current location that uses bridge0, found by device so a
# renamed or localised service is found too. Its interface may be missing.
bridge_service() {
  networksetup -listnetworkserviceorder 2>/dev/null | awk -v device="Device: $BRIDGE_DEVICE)" '
    /^\([0-9*]+\) / { service = $0; sub(/^\([0-9*]+\) /, "", service) }
    index($0, device) { print service; exit }
  '
}

# The Mac's Thunderbolt ports (en2, en3, …), not counting bridges.
thunderbolt_ports() {
  networksetup -listallhardwareports 2>/dev/null | awk '
    /^Hardware Port:/ { port = tolower(substr($0, 16)) }
    /^Device:/ { device = substr($0, 9); if (port ~ /thunderbolt/ && device !~ /^bridge/) print device }
  '
}

TB_BRIDGE_SERVICE="$(bridge_service)"
if [[ -n $TB_BRIDGE_SERVICE ]] &&
  ! /usr/libexec/PlistBuddy -c "Print :VirtualNetworkInterfaces:Bridge:$BRIDGE_DEVICE" "$NETWORK_PREFS" >/dev/null 2>&1; then
  TB_PORTS="$(thunderbolt_ports | tr '\n' ' ')"
  if [[ -n ${TB_PORTS// /} ]]; then
    # Leaving the exo location can leave the Thunderbolt ports down, and a
    # port that is down can't join a bridge.
    for port in $TB_PORTS; do
      ifconfig "$port" up 2>/dev/null || true
    done
    scutil --prefs >/dev/null <<EOF
d.init
d.add Interfaces * $TB_PORTS
set /VirtualNetworkInterfaces/Bridge/$BRIDGE_DEVICE
d.init
d.add __AUTO__ thunderbolt-bridge
set /VirtualNetworkInterfaces/Bridge/$BRIDGE_DEVICE/Options
commit
apply
quit
EOF
    echo_info "Recreated the Thunderbolt Bridge interface ($BRIDGE_DEVICE: ${TB_PORTS% })"
  fi
fi
# Disabled services are listed with a leading "*".
if [[ -z $TB_BRIDGE_SERVICE ]] &&
  networksetup -listallnetworkservices 2>/dev/null | sed 's/^\*//' | grep -x "Thunderbolt Bridge" >/dev/null; then
  TB_BRIDGE_SERVICE="Thunderbolt Bridge"
fi
if [[ -n $TB_BRIDGE_SERVICE ]]; then
  networksetup -setnetworkserviceenabled "$TB_BRIDGE_SERVICE" on 2>/dev/null || true
  echo_info "Re-enabled $TB_BRIDGE_SERVICE"
fi

# Remove EXO data directory (~/.exo). Downloaded models are kept unless
# --delete-models is given, like the app's "Keep downloaded models" checkbox.
EXO_DIR_REMOVED=""
if [[ -d $EXO_DIR ]]; then
  if [[ $DELETE_MODELS == "1" ]]; then
    rm -rf "$EXO_DIR"
    EXO_DIR_REMOVED="full"
    echo_info "Removed ~/.exo, including downloaded models"
  else
    find "$EXO_DIR" -mindepth 1 -maxdepth 1 ! -name models -exec rm -rf {} +
    if [[ -e "$EXO_DIR/models" ]]; then
      EXO_DIR_REMOVED="kept_models"
      echo_info "Removed ~/.exo, except downloaded models in ~/.exo/models"
    else
      rmdir "$EXO_DIR" 2>/dev/null || true
      EXO_DIR_REMOVED="full"
      echo_info "Removed ~/.exo"
    fi
  fi
else
  echo_warn "~/.exo not found (already removed?)"
fi

# Note about launch at login registration
# SMAppService-based login items cannot be removed from a shell script.
# They can only be unregistered from within the app itself or manually via System Settings.
echo_warn "Launch at login must be removed manually:"
echo_warn "  System Settings → General → Login Items & Extensions → Remove EXO"

# Check if EXO.app exists in common locations
APP_FOUND=false
for app_path in "/Applications/EXO.app" "$USER_HOME/Applications/EXO.app"; do
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
kept_models) echo "  • EXO data directory (~/.exo), except downloaded models in ~/.exo/models" ;;
esac
if [[ $EXO_DIR_REMOVED == "kept_models" ]]; then
  echo ""
  echo "To delete the downloaded models too, run this script again with --delete-models."
fi
echo ""
echo "Your network uses the 'Automatic' location again, unless you had chosen another one."
echo "Thunderbolt Bridge has been re-enabled (if present)."
echo ""
echo "Manual step required:"
echo "  Remove EXO from Login Items in System Settings → General → Login Items & Extensions"
echo ""
