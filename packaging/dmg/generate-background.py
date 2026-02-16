#!/usr/bin/env python3
"""Generate a DMG background image for the EXO installer.

Creates a 660x440 PNG with:
- Clean solid dark background
- Bold right-pointing arrow (thick shaft + filled triangle head)
- White "Drag to install" instruction text
- Style inspired by Slack/Discord/VSCode DMGs

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


def _set_pixel(
    pixels: list[list[tuple[int, int, int, int]]],
    x: int,
    y: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Set a pixel with alpha blending."""
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0
    if not (0 <= x < width and 0 <= y < height):
        return
    a = max(0, min(255, int(alpha * 255)))
    if a == 0:
        return
    bg = pixels[y][x]
    if a == 255:
        pixels[y][x] = (color[0], color[1], color[2], 255)
        return
    fa = a / 255.0
    ba = bg[3] / 255.0
    oa = fa + ba * (1 - fa)
    if oa == 0:
        return
    r = int((color[0] * fa + bg[0] * ba * (1 - fa)) / oa)
    g = int((color[1] * fa + bg[1] * ba * (1 - fa)) / oa)
    b = int((color[2] * fa + bg[2] * ba * (1 - fa)) / oa)
    pixels[y][x] = (r, g, b, int(oa * 255))


def _draw_arrow(
    pixels: list[list[tuple[int, int, int, int]]],
    color: tuple[int, int, int],
) -> None:
    """Draw a bold right-pointing arrow: thick shaft + solid filled triangle head.

    Arrow is positioned between the app icon (x=155) and Applications (x=505),
    centered vertically at y=160 (icons moved up to reduce top space).
    """
    # Arrow geometry
    cy = 160  # vertical center — aligned with icon row

    # Shaft: solid rectangle
    shaft_x1 = 250
    shaft_x2 = 380
    shaft_half_h = 3  # 6px thick shaft

    for y in range(cy - shaft_half_h, cy + shaft_half_h + 1):
        for x in range(shaft_x1, shaft_x2 + 1):
            # Anti-alias top and bottom edges
            dist_from_edge = shaft_half_h - abs(y - cy)
            if dist_from_edge >= 1:
                _set_pixel(pixels, x, y, color, 1.0)
            elif dist_from_edge > 0:
                _set_pixel(pixels, x, y, color, dist_from_edge)

    # Arrowhead: filled triangle pointing right
    # Vertices: left-top (375, 180), left-bottom (375, 220), tip (420, 200)
    head_left = 375
    head_right = 420
    head_half_h = 20  # 40px tall triangle

    for x in range(head_left, head_right + 1):
        # At this x, how tall is the triangle?
        t = (x - head_left) / (head_right - head_left)  # 0 at left, 1 at tip
        half_height = head_half_h * (1.0 - t)

        y_top = cy - half_height
        y_bot = cy + half_height

        for y in range(int(y_top) - 1, int(y_bot) + 2):
            dist_top = y - y_top  # positive = inside from top
            dist_bot = y_bot - y  # positive = inside from bottom

            if dist_top >= 1.0 and dist_bot >= 1.0:
                _set_pixel(pixels, x, y, color, 1.0)
            elif dist_top > 0 and dist_bot > 0:
                alpha = min(dist_top, dist_bot)
                _set_pixel(pixels, x, y, color, min(1.0, alpha))


def _draw_text(
    pixels: list[list[tuple[int, int, int, int]]],
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    scale: int = 1,
) -> None:
    """Draw pixel text using a built-in 5x7 bitmap font."""
    glyphs: dict[str, list[str]] = {
        "A": [" 111 ", "1   1", "1   1", "11111", "1   1", "1   1", "1   1"],
        "D": ["1110 ", "1  01", "1  01", "1  01", "1  01", "1  01", "1110 "],
        "E": ["11111", "1    ", "1    ", "1111 ", "1    ", "1    ", "11111"],
        "O": [" 111 ", "1   1", "1   1", "1   1", "1   1", "1   1", " 111 "],
        "X": ["1   1", " 1 1 ", "  1  ", " 1 1 ", "1   1", "1   1", "     "],
        "a": ["     ", "     ", " 111 ", "   01", " 1111", "1  01", " 1111"],
        "c": ["     ", "     ", " 111 ", "1    ", "1    ", "1    ", " 111 "],
        "g": ["     ", "     ", " 1111", "1  01", " 1111", "   01", " 110 "],
        "i": ["     ", "  1  ", "     ", "  1  ", "  1  ", "  1  ", "  1  "],
        "l": ["     ", " 1   ", " 1   ", " 1   ", " 1   ", " 1   ", "  1  "],
        "n": ["     ", "     ", "1 10 ", "11 01", "1  01", "1  01", "1  01"],
        "o": ["     ", "     ", " 110 ", "1  01", "1  01", "1  01", " 110 "],
        "p": ["     ", "     ", "1110 ", "1  01", "1110 ", "1    ", "1    "],
        "r": ["     ", "     ", " 110 ", "1    ", "1    ", "1    ", "1    "],
        "s": ["     ", "     ", " 111 ", "1    ", " 11  ", "   01", "111  "],
        "t": ["     ", " 1   ", "1111 ", " 1   ", " 1   ", " 1   ", "  11 "],
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
                            _set_pixel(pixels, px, py, color, 1.0)
        cursor_x += (len(glyph[0]) + 1) * scale


def generate_background(output_path: str) -> None:
    """Generate the DMG background image."""
    width, height = 660, 440

    # Solid dark background — clean, no gradients or vignettes
    bg_color = (22, 22, 24, 255)  # macOS dark mode surface

    pixels: list[list[tuple[int, int, int, int]]] = [
        [bg_color] * width for _ in range(height)
    ]

    # Draw bold white right-pointing arrow between icon positions
    # Arrow is vertically centered at icon row (y=160)
    _draw_arrow(pixels, (255, 255, 255))

    # Draw white icon labels (Finder's labels are hidden via text size 1)
    # Finder positions icons by center point: x=155 and x=505
    # Icon bottom edge is at y=160+64=224, labels start below at y=232
    label_color = (255, 255, 255)
    label_scale = 2
    char_width = 6 * label_scale  # 5px glyph + 1px spacing, scaled
    # "EXO" — 3 chars, centered under icon at x=155
    exo_width = 3 * char_width
    _draw_text(pixels, 155 - exo_width // 2, 232, "EXO", label_color, scale=label_scale)
    # "Applications" — 12 chars, centered under icon at x=505
    apps_width = 12 * char_width
    _draw_text(
        pixels,
        505 - apps_width // 2,
        232,
        "Applications",
        label_color,
        scale=label_scale,
    )

    # Draw "Drag to install" instruction — bright white, scale=3 for readability
    # Text is ~15 chars × 18px/char (at scale=3) = ~270px wide
    text_x = (width - 270) // 2
    _draw_text(pixels, text_x, 340, "Drag to install", (255, 255, 255), scale=3)

    # Write PNG
    png_data = _create_png(width, height, pixels)
    Path(output_path).write_bytes(png_data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output.png>", file=sys.stderr)
        sys.exit(1)
    generate_background(sys.argv[1])
    print(f"Background image written to {sys.argv[1]}")
