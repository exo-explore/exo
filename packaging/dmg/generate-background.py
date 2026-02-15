#!/usr/bin/env python3
"""Generate a DMG background image for the EXO installer.

Creates a 660x400 PNG with:
- Clean dark gradient background (no grid)
- Minimal right-pointing arrow between app and Applications
- White "Drag to install" instruction text
- Premium style inspired by Slack/Discord/VSCode DMGs

Usage:
    python3 generate-background.py output.png
"""

from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk (type + data + CRC)."""
    raw = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + raw
        + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
    )


def _create_png(
    width: int, height: int, pixels: list[list[tuple[int, int, int, int]]]
) -> bytes:
    """Create a minimal RGBA PNG from pixel data."""
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    ihdr = _png_chunk(b"IHDR", ihdr_data)

    # IDAT — build raw scanlines then deflate
    raw_lines = bytearray()
    for row in pixels:
        raw_lines.append(0)  # filter: None
        for r, g, b, a in row:
            raw_lines.extend((r, g, b, a))
    idat = _png_chunk(b"IDAT", zlib.compress(bytes(raw_lines), 9))

    # IEND
    iend = _png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


def _lerp(a: int, b: int, t: float) -> int:
    return max(0, min(255, int(a + (b - a) * t)))


def _blend(
    bg: tuple[int, int, int, int], fg: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Alpha-blend fg over bg."""
    fa = fg[3] / 255.0
    ba = bg[3] / 255.0
    oa = fa + ba * (1 - fa)
    if oa == 0:
        return (0, 0, 0, 0)
    r = int((fg[0] * fa + bg[0] * ba * (1 - fa)) / oa)
    g = int((fg[1] * fa + bg[1] * ba * (1 - fa)) / oa)
    b = int((fg[2] * fa + bg[2] * ba * (1 - fa)) / oa)
    return (r, g, b, int(oa * 255))


def _draw_smooth_arrow(
    pixels: list[list[tuple[int, int, int, int]]],
    cx: int,
    cy: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a clean, minimal right-pointing arrow with anti-aliased edges."""
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0

    # Slim shaft
    shaft_half_len = 32
    shaft_half_thickness = 1.5

    for x in range(cx - shaft_half_len, cx + shaft_half_len + 1):
        for y_offset_10 in range(-30, 31):  # sub-pixel sampling
            y_f = cy + y_offset_10 / 10.0
            yi = int(y_f)
            dist = abs(y_f - cy)
            if dist <= shaft_half_thickness and 0 <= yi < height and 0 <= x < width:
                # Smooth edge falloff
                edge_dist = shaft_half_thickness - dist
                alpha = min(1.0, edge_dist * 2.0)
                a = int(alpha * 200)
                fg = (color[0], color[1], color[2], a)
                pixels[yi][x] = _blend(pixels[yi][x], fg)

    # Chevron arrowhead (> shape) — clean and modern
    head_x = cx + shaft_half_len - 2
    head_size = 14
    stroke_width = 2.0

    for i_10 in range(head_size * 10):
        t = i_10 / 10.0
        # Top arm of chevron
        px_f = head_x + t
        py_top_f = cy - t
        # Bottom arm of chevron
        py_bot_f = cy + t

        for dy_10 in range(int(-stroke_width * 10), int(stroke_width * 10) + 1):
            for arm_py in [py_top_f, py_bot_f]:
                py = int(arm_py + dy_10 / 10.0)
                px = int(px_f)
                if 0 <= py < height and 0 <= px < width:
                    dist = abs(dy_10 / 10.0)
                    alpha = max(0.0, min(1.0, (stroke_width - dist) * 1.5))
                    a = int(alpha * 200)
                    fg = (color[0], color[1], color[2], a)
                    pixels[py][px] = _blend(pixels[py][px], fg)


def _draw_text_pixel(
    pixels: list[list[tuple[int, int, int, int]]],
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int, int],
    scale: int = 1,
) -> None:
    """Draw simple pixel text. Limited to the phrase 'Drag to install'."""
    # 5x7 pixel font for the letters we need
    glyphs: dict[str, list[str]] = {
        "D": ["1110 ", "1  01", "1  01", "1  01", "1  01", "1  01", "1110 "],
        "r": ["     ", "     ", " 110 ", "1    ", "1    ", "1    ", "1    "],
        "a": ["     ", "     ", " 111 ", "   01", " 1111", "1  01", " 1111"],
        "g": ["     ", "     ", " 1111", "1  01", " 1111", "   01", " 110 "],
        "t": ["     ", " 1   ", "1111 ", " 1   ", " 1   ", " 1   ", "  11 "],
        "o": ["     ", "     ", " 110 ", "1  01", "1  01", "1  01", " 110 "],
        "i": ["     ", "  1  ", "     ", "  1  ", "  1  ", "  1  ", "  1  "],
        "n": ["     ", "     ", "1 10 ", "11 01", "1  01", "1  01", "1  01"],
        "s": ["     ", "     ", " 111 ", "1    ", " 11  ", "   01", "111  "],
        "l": ["     ", " 1   ", " 1   ", " 1   ", " 1   ", " 1   ", "  1  "],
        " ": ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
    }

    cursor_x = x
    for ch in text:
        glyph = glyphs.get(ch)
        if glyph is None:
            cursor_x += 6 * scale
            continue
        for row_idx, row_str in enumerate(glyph):
            for col_idx, pixel_ch in enumerate(row_str):
                if pixel_ch == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            py = y + row_idx * scale + sy
                            px = cursor_x + col_idx * scale + sx
                            if 0 <= py < len(pixels) and 0 <= px < len(pixels[0]):
                                pixels[py][px] = _blend(pixels[py][px], color)
        cursor_x += (len(glyph[0]) + 1) * scale


def generate_background(output_path: str) -> None:
    """Generate the DMG background image."""
    width, height = 660, 400

    # Clean dark gradient — no grid, no noise
    top_color = (28, 28, 30)  # macOS dark mode surface
    bottom_color = (16, 16, 18)  # slightly darker at bottom

    pixels: list[list[tuple[int, int, int, int]]] = []
    for y in range(height):
        t = y / (height - 1)
        row: list[tuple[int, int, int, int]] = []
        for x in range(width):
            # Radial vignette: slightly brighter in center for depth
            dx = (x - width / 2) / (width / 2)
            dy = (y - height * 0.45) / (height / 2)
            dist = math.sqrt(dx * dx + dy * dy)
            vignette = max(0.0, 1.0 - dist * 0.3)

            r = _lerp(top_color[0], bottom_color[0], t) + int(vignette * 6)
            g = _lerp(top_color[1], bottom_color[1], t) + int(vignette * 6)
            b = _lerp(top_color[2], bottom_color[2], t) + int(vignette * 6)
            row.append((min(255, r), min(255, g), min(255, b), 255))
        pixels.append(row)

    # Draw a clean, minimal arrow between app (x=155) and Applications (x=505)
    arrow_color = (255, 255, 255)  # white — clean and visible
    _draw_smooth_arrow(pixels, width // 2, 200, arrow_color)

    # Draw instruction text below the arrow — white for readability
    text_color = (255, 255, 255, 140)  # white, semi-transparent
    _draw_text_pixel(pixels, 268, 310, "Drag to install", text_color, scale=2)

    # Write PNG
    png_data = _create_png(width, height, pixels)
    Path(output_path).write_bytes(png_data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output.png>", file=sys.stderr)
        sys.exit(1)
    generate_background(sys.argv[1])
    print(f"Background image written to {sys.argv[1]}")
