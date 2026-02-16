#!/usr/bin/env python3
"""Copy the static DMG background image to the specified output path.

The background is a 1600×740 retina PNG (2× for 800×400 logical window) with a
hand-drawn arrow and yellow bookmark accents on a white background.  Based on
Ollama's DMG background (MIT-licensed).

Usage:
    python3 generate-background.py output.png
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

BACKGROUND_PNG = Path(__file__).parent / "background.png"


def generate_background(output_path: str) -> None:
    """Copy the static background image to the output path."""
    if not BACKGROUND_PNG.exists():
        print(f"Error: {BACKGROUND_PNG} not found", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(BACKGROUND_PNG, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output.png>", file=sys.stderr)
        sys.exit(1)
    generate_background(sys.argv[1])
    print(f"Background image written to {sys.argv[1]}")
