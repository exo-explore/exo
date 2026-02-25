#!/usr/bin/env bash
# create-dmg.sh — Build a polished macOS DMG installer for EXO
#
# Usage:
#   ./packaging/dmg/create-dmg.sh <app-path> <output-dmg> [volume-name]
#
# Example:
#   ./packaging/dmg/create-dmg.sh output/EXO.app EXO-1.0.0.dmg "EXO"
#
# Creates a DMG with:
#   - Custom background image with drag-to-Applications arrow
#   - App icon on left, Applications alias on right
#   - Proper window size and icon positioning
set -euo pipefail

APP_PATH="${1:?Usage: create-dmg.sh <app-path> <output-dmg> [volume-name]}"
OUTPUT_DMG="${2:?Usage: create-dmg.sh <app-path> <output-dmg> [volume-name]}"
VOLUME_NAME="${3:-EXO}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKGROUND_SCRIPT="${SCRIPT_DIR}/generate-background.py"
TEMP_DIR="$(mktemp -d)"
DMG_STAGING="${TEMP_DIR}/dmg-root"
TEMP_DMG="${TEMP_DIR}/temp.dmg"
BACKGROUND_PNG="${TEMP_DIR}/background.png"

cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

echo "==> Creating DMG installer for ${VOLUME_NAME}"

# ── Step 1: Generate background image ────────────────────────────────────────
if command -v python3 &>/dev/null; then
  python3 "$BACKGROUND_SCRIPT" "$BACKGROUND_PNG"
  echo "    Background image generated"
else
  echo "    Warning: python3 not found, skipping custom background"
  BACKGROUND_PNG=""
fi

# ── Step 2: Prepare staging directory ─────────────────────────────────────────
mkdir -p "$DMG_STAGING"
cp -R "$APP_PATH" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

# ── Step 3: Create writable DMG ──────────────────────────────────────────────
# Calculate required size (app size + 20MB headroom)
APP_SIZE_KB=$(du -sk "$APP_PATH" | cut -f1)
DMG_SIZE_KB=$((APP_SIZE_KB + 20480))

hdiutil create \
  -volname "$VOLUME_NAME" \
  -size "${DMG_SIZE_KB}k" \
  -fs HFS+ \
  -layout SPUD \
  "$TEMP_DMG"

# ── Step 4: Mount and configure ──────────────────────────────────────────────
MOUNT_DIR=$(hdiutil attach "$TEMP_DMG" -readwrite -noverify | awk -F'\t' '/Apple_HFS/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $NF); print $NF}')
echo "    Mounted at: $MOUNT_DIR"

# Copy contents
cp -R "$DMG_STAGING/"* "$MOUNT_DIR/"

# Add background image
if [[ -n $BACKGROUND_PNG && -f $BACKGROUND_PNG ]]; then
  mkdir -p "$MOUNT_DIR/.background"
  cp "$BACKGROUND_PNG" "$MOUNT_DIR/.background/background.png"
fi

# ── Step 5: Configure window appearance via AppleScript ──────────────────────
# Window: 800×400, app icon on left, Applications on right (matches Ollama layout)
# Background image is 1600×740 (2× retina for 800×400 logical window).
APP_NAME="$(basename "$APP_PATH")"

osascript <<APPLESCRIPT
tell application "Finder"
    tell disk "$VOLUME_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {200, 120, 1000, 520}
        set opts to icon view options of container window
        set icon size of opts to 128
        set text size of opts to 12
        set arrangement of opts to not arranged
        if exists file ".background:background.png" then
            set background picture of opts to file ".background:background.png"
        end if
        set position of item "$APP_NAME" of container window to {200, 190}
        set position of item "Applications" of container window to {600, 190}
        close
        open
        update without registering applications
        delay 1
        close
    end tell
end tell
APPLESCRIPT

echo "    Window layout configured"

# Ensure Finder updates are flushed
sync

# ── Step 6: Finalise ─────────────────────────────────────────────────────────
hdiutil detach "$MOUNT_DIR" -quiet
hdiutil convert "$TEMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$OUTPUT_DMG"

echo "==> DMG created: $OUTPUT_DMG"
echo "    Size: $(du -h "$OUTPUT_DMG" | cut -f1)"
