#!/usr/bin/env python3
"""Generate a DMG background image for the EXO installer.

Creates a 660x400 PNG with:
- Dark gradient background matching the EXO brand
- Right-pointing arrow between app and Applications
- "Drag to install" instruction text

Usage:
    python3 generate-background.py output.png
"""

from __future__ import annotations

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


def _draw_arrow(
    pixels: list[list[tuple[int, int, int, int]]],
    cx: int,
    cy: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a simple right-pointing arrow at (cx, cy)."""
    # Shaft: horizontal line
    shaft_len = 60
    shaft_thickness = 3
    for dx in range(-shaft_len, shaft_len + 1):
        for dy in range(-shaft_thickness, shaft_thickness + 1):
            y = cy + dy
            x = cx + dx
            if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
                pixels[y][x] = color

    # Arrowhead: triangle pointing right
    head_size = 20
    for i in range(head_size):
        spread = int(i * 1.2)
        x = cx + shaft_len + i
        for dy in range(-spread, spread + 1):
            y = cy + dy
            if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
                pixels[y][x] = color


def _draw_text_pixel(
    pixels: list[list[tuple[int, int, int, int]]],
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int, int],
    scale: int = 1,
) -> None:
    """Draw simple pixel text. Limited to the phrase 'Drag to install'."""
    # 5x7 pixel font for uppercase + lowercase letters we need
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
                                pixels[py][px] = color
        cursor_x += (len(glyph[0]) + 1) * scale


def generate_background(output_path: str) -> None:
    """Generate the DMG background image."""
    width, height = 660, 400

    # Build gradient background: dark gray to slightly darker
    top_color = (30, 30, 30)  # #1e1e1e — matches exo-dark-gray
    bottom_color = (18, 18, 18)  # #121212 — matches exo-black

    pixels: list[list[tuple[int, int, int, int]]] = []
    for y in range(height):
        t = y / (height - 1)
        r = _lerp(top_color[0], bottom_color[0], t)
        g = _lerp(top_color[1], bottom_color[1], t)
        b = _lerp(top_color[2], bottom_color[2], t)
        pixels.append([(r, g, b, 255)] * width)

    # Draw subtle grid lines (matches the exo dashboard grid)
    grid_color = (40, 40, 40, 255)
    for y in range(0, height, 40):
        for x in range(width):
            pixels[y][x] = grid_color
    for x in range(0, width, 40):
        for y in range(height):
            pixels[y][x] = grid_color

    # Draw the arrow in the center (between app icon at x=155 and Applications at x=505)
    arrow_color = (200, 180, 50, 255)  # EXO yellow
    _draw_arrow(pixels, width // 2, 200, arrow_color)

    # Draw instruction text below the arrow
    text_color = (150, 150, 150, 200)
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
