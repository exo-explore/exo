#!/usr/bin/env bash
# create-dmg.sh: build the EXO disk image
#
# Usage:
#   ./packaging/dmg/create-dmg.sh <app-path> <output-dmg> [volume-name]
#
# Example:
#   ./packaging/dmg/create-dmg.sh output/EXO.app EXO-1.0.0.dmg EXO
#
# The disk opens as a drag-to-install window: EXO.app on the left, a link to
# Applications on the right, and a branded background (background.png and
# background@2x.png, drawn by generate-background.py) saying "Drag EXO to
# Applications".
#
# The window layout is written by dmgbuild, which creates the .DS_Store file
# itself, so this needs neither Finder nor AppleScript and works on headless
# CI runners. dmgbuild and its dependencies are installed from
# requirements.txt, pinned with hashes, into a temporary environment.
set -euo pipefail

USAGE="Usage: create-dmg.sh <app-path> <output-dmg> [volume-name]"
APP_PATH="${1:?$USAGE}"
OUTPUT_DMG="${2:?$USAGE}"
VOLUME_NAME="${3:-EXO}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d $APP_PATH ]]; then
  echo "App not found: $APP_PATH" >&2
  exit 1
fi
for file in background.png background@2x.png dmgbuild-settings.py requirements.txt; do
  if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
    echo "Missing $SCRIPT_DIR/$file" >&2
    exit 1
  fi
done

TEMP_DIR="$(mktemp -d)"
MOUNT_DIR="$TEMP_DIR/mount"
cleanup() {
  if [[ -d $MOUNT_DIR ]]; then
    hdiutil detach "$MOUNT_DIR" -quiet 2>/dev/null ||
      hdiutil detach "$MOUNT_DIR" -force -quiet 2>/dev/null || true
  fi
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "==> Installing dmgbuild"
uv venv --quiet --no-config --python 3.13 "$TEMP_DIR/venv"
uv pip install --quiet --no-config --python "$TEMP_DIR/venv/bin/python" \
  --require-hashes --only-binary :all: -r "$SCRIPT_DIR/requirements.txt"

echo "==> Creating ${OUTPUT_DMG}"
"$TEMP_DIR/venv/bin/dmgbuild" \
  -s "$SCRIPT_DIR/dmgbuild-settings.py" \
  -D app="$APP_PATH" \
  -D background="$SCRIPT_DIR/background.png" \
  "$VOLUME_NAME" "$OUTPUT_DMG"

hdiutil verify -quiet "$OUTPUT_DMG"

# dmgbuild copies the app with ditto, which keeps code signatures, but it
# doesn't stop if the copy fails. Check that the copy is complete, and that it
# is still validly signed if the original is.
echo "==> Checking the app in the image"
APP_NAME="$(basename "$APP_PATH")"
mkdir -p "$MOUNT_DIR"
hdiutil attach "$OUTPUT_DMG" -readonly -nobrowse -noautoopen -quiet \
  -mountpoint "$MOUNT_DIR"
SOURCE_ENTRIES="$(find "$APP_PATH" | wc -l | tr -d ' ')"
IMAGE_ENTRIES="$(find "$MOUNT_DIR/$APP_NAME" | wc -l | tr -d ' ')"
if [[ $SOURCE_ENTRIES != "$IMAGE_ENTRIES" ]]; then
  echo "The image has ${IMAGE_ENTRIES} of the app's ${SOURCE_ENTRIES} files" >&2
  exit 1
fi
if codesign --verify --deep --strict "$APP_PATH" 2>/dev/null; then
  codesign --verify --deep --strict "$MOUNT_DIR/$APP_NAME"
fi
hdiutil detach "$MOUNT_DIR" -quiet
rmdir "$MOUNT_DIR"

echo "==> Created ${OUTPUT_DMG} ($(du -h "$OUTPUT_DMG" | cut -f1))"
