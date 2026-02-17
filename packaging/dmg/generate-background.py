#!/usr/bin/env python3
"""Generate the DMG background image with a centered drag-to-Applications arrow.

The output is a 1600×740 retina PNG (2× for 800×400 logical window).
Icons are positioned at (200, 190) and (600, 190) in logical coordinates;
the arrow is drawn centered between them.

Usage:
    python3 generate-background.py [output.png]

If no output path is given, overwrites the bundled background.png in-place.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Retina dimensions (2× logical 800×400)
WIDTH = 1600
HEIGHT = 740

# Icon positions in logical coords → retina coords
# App icon at (200, 190), Applications at (600, 190)
APP_X = 200 * 2  # 400
APPS_X = 600 * 2  # 1200
ICON_Y = 190 * 2  # 380

# Arrow drawn between icons, slightly above icon center
ARROW_START_X = APP_X + 160  # past the icon
ARROW_END_X = APPS_X - 160  # before the Applications icon
ARROW_Y = ICON_Y  # same height as icons
ARROW_RISE = 120  # upward arc height


def draw_arrow(draw: ImageDraw.ImageDraw) -> None:
    """Draw a hand-drawn-style curved arrow from app icon toward Applications."""
    color = (30, 30, 30)
    line_width = 8

    # Compute bezier curve points for a gentle upward arc
    points: list[tuple[float, float]] = []
    steps = 80
    for i in range(steps + 1):
        t = i / steps
        # Quadratic bezier: start → control → end
        cx = (ARROW_START_X + ARROW_END_X) / 2
        cy = ARROW_Y - ARROW_RISE
        x = (1 - t) ** 2 * ARROW_START_X + 2 * (1 - t) * t * cx + t**2 * ARROW_END_X
        y = (1 - t) ** 2 * ARROW_Y + 2 * (1 - t) * t * cy + t**2 * ARROW_Y
        points.append((x, y))

    # Draw the curve as connected line segments
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=line_width)

    # Arrowhead at the end
    end_x, end_y = points[-1]
    # Direction from second-to-last to last point
    prev_x, prev_y = points[-3]
    angle = math.atan2(end_y - prev_y, end_x - prev_x)
    head_len = 36
    head_angle = math.radians(25)

    left_x = end_x - head_len * math.cos(angle - head_angle)
    left_y = end_y - head_len * math.sin(angle - head_angle)
    right_x = end_x - head_len * math.cos(angle + head_angle)
    right_y = end_y - head_len * math.sin(angle + head_angle)

    draw.polygon(
        [(end_x, end_y), (left_x, left_y), (right_x, right_y)],
        fill=color,
    )


def generate_background(output_path: str) -> None:
    """Generate a white DMG background with a centered arrow."""
    img = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_arrow(draw)
    img.save(output_path, "PNG")


if __name__ == "__main__":
    default_output = str(Path(__file__).parent / "background.png")
    out = sys.argv[1] if len(sys.argv) >= 2 else default_output
    generate_background(out)
    print(f"Background image written to {out}")
