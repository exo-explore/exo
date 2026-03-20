#!/bin/bash
# setup_universal_control.sh — Enable Universal Control for MacBook ↔ Mac Mini ↔ iPad
# Run on BOTH machines (MacBook and Mac Mini) for seamless mouse/keyboard sharing.
#
# Usage:
#   ./setup_universal_control.sh enable   — enable UC + restart daemon
#   ./setup_universal_control.sh disable  — disable UC
#   ./setup_universal_control.sh status   — show current state
#
# Requirements: macOS 12.3+, same Wi-Fi subnet, Bluetooth ON on both machines.
# Apple ID does NOT need to match between machines.
#
# iPad as UC target (optional):
#   Settings → General → AirPlay & Handoff → Cursor and Keyboard → ON
#   iPad will appear in Displays layout automatically after that.
#
# iPhone keyboard:
#   Open iPhone Mirroring.app (macOS 15/26) — typing flows from MacBook keyboard.
set -euo pipefail

ACTION="${1:-status}"

_UC_DOMAIN="com.apple.universalcontrol"

case "$ACTION" in
  enable)
    echo "[UC] Enabling Universal Control..."

    # 1. Start sharingd (Handoff / AirDrop / UC discovery backbone)
    UID_NUM=$(id -u)
    echo "[UC] Bootstrapping sharingd..."
    launchctl bootstrap gui/$UID_NUM /System/Library/LaunchAgents/com.apple.sharingd.plist 2>/dev/null || true
    launchctl kickstart -k gui/$UID_NUM/com.apple.sharingd 2>/dev/null || true

    # 2. Enable Handoff (required for UC peer discovery)
    defaults write com.apple.coreservices.useractivityd ActivityAdvertisingAllowed -bool true
    defaults write com.apple.coreservices.useractivityd ActivityReceivingAllowed -bool true

    # 3. UC prefs (idempotent)
    defaults write "$_UC_DOMAIN" Enabled -bool true
    defaults write "$_UC_DOMAIN" allowsCursorMovement -bool true
    defaults write "$_UC_DOMAIN" allowsKeyboardMouse -bool true

    # 4. Restart UniversalControl.app (correct process name on macOS 12.3+)
    echo "[UC] Restarting UniversalControl.app..."
    killall UniversalControl 2>/dev/null || true
    sleep 1
    open /System/Library/CoreServices/UniversalControl.app 2>/dev/null || true

    # 5. Bounce ScreenSharingAgent so display arrangement is re-read
    killall ScreenSharingAgent 2>/dev/null || true

    # 6. Open iPhone Mirroring if available (macOS 15/26)
    if [[ -d "/System/Applications/iPhone Mirroring.app" ]]; then
      echo "[UC] Opening iPhone Mirroring.app..."
      open "/System/Applications/iPhone Mirroring.app" 2>/dev/null || true
    fi

    echo ""
    echo "[UC] Universal Control ENABLED."

    # Auto-arrange displays: Mini to the right of MacBook after UC discovery (~30s)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/arrange_displays.sh" ]] && command -v displayplacer &>/dev/null; then
      echo "[UC] Auto-arranging displays in 35s (background)..."
      (sleep 35 && bash "$SCRIPT_DIR/arrange_displays.sh") &
    fi

    echo ""
    echo "[UC] Next steps:"
    echo "      1. Run on Mac Mini:  ssh macmini '~/exo/exo/setup_universal_control.sh enable'"
    echo "      2. Wait ~35s — displays auto-arranged (Mini to the right)"
    echo "      3. Mouse slides seamlessly — keyboard follows cursor"
    echo ""
    echo "[UC] iPad (one-time tap):"
    echo "      Settings → General → AirPlay & Handoff → Cursor and Keyboard → ON"
    ;;

  disable)
    defaults write "$_UC_DOMAIN" Enabled -bool false
    killall UniversalControl 2>/dev/null || echo "[UC] UniversalControl not running"
    echo "[UC] Universal Control DISABLED."
    ;;

  status)
    enabled=$(defaults read "$_UC_DOMAIN" Enabled 2>/dev/null || echo "not set")
    cursor=$(defaults read "$_UC_DOMAIN" allowsCursorMovement 2>/dev/null || echo "not set")
    keyboard=$(defaults read "$_UC_DOMAIN" allowsKeyboardMouse 2>/dev/null || echo "not set")
    handoff_adv=$(defaults read com.apple.coreservices.useractivityd ActivityAdvertisingAllowed 2>/dev/null || echo "not set")
    handoff_rcv=$(defaults read com.apple.coreservices.useractivityd ActivityReceivingAllowed 2>/dev/null || echo "not set")
    echo "[UC] Enabled                     = $enabled"
    echo "[UC] CursorMovement              = $cursor"
    echo "[UC] KeyboardMouse               = $keyboard"
    echo "[UC] Handoff Advertising         = $handoff_adv"
    echo "[UC] Handoff Receiving           = $handoff_rcv"
    if pgrep -x UniversalControl >/dev/null 2>&1; then
      echo "[UC] UniversalControl.app        = running (PID $(pgrep -x UniversalControl))"
    else
      echo "[UC] UniversalControl.app        = NOT running"
    fi
    UID_NUM=$(id -u)
    sharingd_status=$(launchctl list com.apple.sharingd 2>&1 || true)
    if echo "$sharingd_status" | grep -q '"PID"'; then
      echo "[UC] sharingd                    = running"
    else
      echo "[UC] sharingd                    = NOT loaded (run 'enable' to fix)"
    fi
    echo ""
    echo "[UC] Run on Mac Mini: ssh macmini '~/exo/exo/setup_universal_control.sh status'"
    ;;

  *)
    echo "Usage: $0 {enable|disable|status}"
    exit 1
    ;;
esac
