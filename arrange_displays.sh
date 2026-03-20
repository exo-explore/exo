#!/bin/bash
# arrange_displays.sh — Auto-position Mac Mini display to the RIGHT of MacBook
# Run once after UC connects Mini (~30s after enable). Safe to re-run.
set -euo pipefail

echo "[displays] Detecting displays..."

if ! command -v displayplacer &>/dev/null; then
    echo "[displays] displayplacer not found — install with: brew install jakehilborn/jakehilborn/displayplacer"
    exit 1
fi

DISPLAYS=$(displayplacer list 2>/dev/null)

# MacBook built-in resolution
MB_RES=$(echo "$DISPLAYS" | grep -A10 "Type: built in" | grep "Resolution:" | head -1 | grep -oE '[0-9]+ x [0-9]+' | head -1 | sed 's/ x /x/')
MB_W=$(echo "$MB_RES" | cut -dx -f1)

if [[ -z "$MB_W" ]]; then
    echo "[displays] Could not detect MacBook built-in resolution"
    exit 1
fi

# Find external UC-connected display (not built-in, not AirPlay)
EXTERNAL_ID=$(echo "$DISPLAYS" | grep -B5 "Type: " | grep -v "built in\|AirPlay\|Type:" | grep "^id:" | head -1 | grep -oE '"[^"]+"' | head -1 | tr -d '"')

# Fallback: grep for id lines directly
if [[ -z "$EXTERNAL_ID" ]]; then
    EXTERNAL_ID=$(displayplacer list 2>/dev/null | grep "^id:" | head -1 | awk '{print $2}' | tr -d '"')
fi

if [[ -z "$EXTERNAL_ID" ]]; then
    echo "[displays] No external UC display found yet — waiting 10s..."
    sleep 10
    EXTERNAL_ID=$(displayplacer list 2>/dev/null | grep "^id:" | head -1 | awk '{print $2}' | tr -d '"')
fi

if [[ -z "$EXTERNAL_ID" ]]; then
    echo "[displays] Still no external display. UC not connected yet."
    exit 0
fi

echo "[displays] External display ID: $EXTERNAL_ID"
echo "[displays] MacBook res: ${MB_RES}, width: ${MB_W}"

# Save current arrangement for reference
mkdir -p ~/.exo
displayplacer list 2>/dev/null > ~/.exo/display_layout.txt

# Place MacBook at origin, Mini to the right
displayplacer \
    "id:main res:${MB_RES} origin:(0,0) degree:0" \
    "id:${EXTERNAL_ID} origin:(${MB_W},0) degree:0" 2>/dev/null \
    && echo "[displays] Mini display placed to the RIGHT of MacBook ✓" \
    || echo "[displays] displayplacer failed — may need manual drag once"
