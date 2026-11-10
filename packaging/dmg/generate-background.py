# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow==11.3.0"]
# ///
"""Draw the background of the EXO disk image window.

Writes background.png (1x) and background@2x.png (2x) next to this script.
dmgbuild combines the pair into one multi-resolution TIFF, so Finder shows
the sharp one on Retina displays.

The window is dark, with the yellow pixel "exo" wordmark at the top, the app
and Applications on light tiles, a yellow pixel arrow between them, and
"Drag EXO to Applications" below. The tiles are there because Finder treats
a window with a background picture as light, and always draws the names
under the icons in black, in Dark Mode too.

The wordmark comes from dashboard/static/exo-logo.png. Text is set in the
macOS system font, so run this on a Mac:

    uv run packaging/dmg/generate-background.py

The window size and icon positions must match dmgbuild-settings.py.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WORDMARK_SOURCE = HERE.parent.parent / "dashboard" / "static" / "exo-logo.png"
SYSTEM_FONT = Path("/System/Library/Fonts/SFNS.ttf")

# Window size in points. Keep in sync with dmgbuild-settings.py.
#
# Finder treats the size as the whole window, title bar included, and draws
# the background from the top of the area below the title bar. Making the
# image as tall as the window means it always fills that area, whatever the
# title bar's height on a given macOS version; only the bottom edge of the
# image, which is plain background, is cut off. Everything drawn stays
# between 36 and 404 points from the top so it is visible either way.
WIDTH = 640
HEIGHT = 440
# Icon centres in points. Keep in sync with dmgbuild-settings.py.
APP_CENTRE = (160, 200)
APPLICATIONS_CENTRE = (480, 200)

BACKGROUND = (22, 22, 22)
EXO_YELLOW = (0xFF, 0xDE, 0x00)
TEXT = (235, 235, 235)

# A light tile behind each 128 pt icon and its name, relative to the icon
# centre: the icon, a gap, a line of 13 pt text, and some padding.
TILE = (176, 192)
TILE_TOP_OFFSET = -84
TILE_RADIUS = 20
TILE_COLOUR = (236, 236, 236)

# The wordmark is drawn on a 20 x 5 grid. Each grid cell is this many points.
WORDMARK_CELL = 7
WORDMARK_TOP = 40

# A pixel arrow in the same style as the wordmark: X is a filled cell.
ARROW = (
    ".........X.....",
    ".........XX....",
    ".........XXX...",
    "XXXXXXXXXXXXX..",
    "XXXXXXXXXXXXXX.",
    "XXXXXXXXXXXXX..",
    ".........XXX...",
    ".........XX....",
    ".........X.....",
)
ARROW_CELL = 5

INSTRUCTION = "Drag EXO to Applications"
INSTRUCTION_SIZE = 17
INSTRUCTION_CENTRE_Y = 356


def wordmark_cells() -> list[list[bool]]:
    """Read the wordmark's pixel grid from the dashboard logo.

    The logo is a 1-bit image whose cells are 100 px squares inside a
    2000 x 500 px bounding box, so sampling each cell's centre recovers
    the grid exactly.
    """
    logo = Image.open(WORDMARK_SOURCE).convert("RGBA")
    alpha = logo.getchannel("A")
    box = alpha.getbbox()
    if box is None:
        raise SystemExit(f"{WORDMARK_SOURCE} is empty")
    left, top, right, bottom = box
    columns, rows = 20, 5
    cell_width = (right - left) / columns
    cell_height = (bottom - top) / rows
    return [
        [
            alpha.getpixel(
                (
                    int(left + (column + 0.5) * cell_width),
                    int(top + (row + 0.5) * cell_height),
                )
            )
            != 0
            for column in range(columns)
        ]
        for row in range(rows)
    ]


def draw_cells(
    draw: ImageDraw.ImageDraw,
    cells: list[list[bool]],
    origin: tuple[int, int],
    cell: int,
    colour: tuple[int, int, int],
) -> None:
    origin_x, origin_y = origin
    for row_index, row in enumerate(cells):
        for column_index, filled in enumerate(row):
            if filled:
                x = origin_x + column_index * cell
                y = origin_y + row_index * cell
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=colour)


def draw_tiles(image: Image.Image, scale: int) -> None:
    """Draw the rounded tiles, smoothing their corners by supersampling."""
    supersample = 4
    factor = scale * supersample
    mask = Image.new("L", (WIDTH * factor, HEIGHT * factor), 0)
    mask_draw = ImageDraw.Draw(mask)
    for centre_x, centre_y in (APP_CENTRE, APPLICATIONS_CENTRE):
        left = centre_x - TILE[0] // 2
        top = centre_y + TILE_TOP_OFFSET
        mask_draw.rounded_rectangle(
            (
                left * factor,
                top * factor,
                (left + TILE[0]) * factor - 1,
                (top + TILE[1]) * factor - 1,
            ),
            radius=TILE_RADIUS * factor,
            fill=255,
        )
    mask = mask.reduce(supersample)
    image.paste(TILE_COLOUR, (0, 0, *image.size), mask)


def render(scale: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), BACKGROUND)
    draw_tiles(image, scale)
    draw = ImageDraw.Draw(image)

    wordmark = wordmark_cells()
    wordmark_width = len(wordmark[0]) * WORDMARK_CELL * scale
    draw_cells(
        draw,
        wordmark,
        ((WIDTH * scale - wordmark_width) // 2, WORDMARK_TOP * scale),
        WORDMARK_CELL * scale,
        EXO_YELLOW,
    )

    arrow = [[character == "X" for character in row] for row in ARROW]
    arrow_width = len(arrow[0]) * ARROW_CELL * scale
    arrow_height = len(arrow) * ARROW_CELL * scale
    arrow_centre_x = (APP_CENTRE[0] + APPLICATIONS_CENTRE[0]) * scale // 2
    draw_cells(
        draw,
        arrow,
        (
            arrow_centre_x - arrow_width // 2,
            APP_CENTRE[1] * scale - arrow_height // 2,
        ),
        ARROW_CELL * scale,
        EXO_YELLOW,
    )

    font = ImageFont.truetype(str(SYSTEM_FONT), INSTRUCTION_SIZE * scale)
    # Axes: width, optical size, grade, weight (medium).
    font.set_variation_by_axes([100, INSTRUCTION_SIZE, 400, 500])
    draw.text(
        (WIDTH * scale // 2, INSTRUCTION_CENTRE_Y * scale),
        INSTRUCTION,
        font=font,
        fill=TEXT,
        anchor="mm",
    )
    return image


def main() -> None:
    for scale, name in ((1, "background.png"), (2, "background@2x.png")):
        path = HERE / name
        image = render(scale)
        # 72 and 144 dpi tell tiffutil which image is which resolution.
        image.save(path, "PNG", dpi=(72 * scale, 72 * scale), optimize=True)
        print(f"Wrote {path.relative_to(HERE.parent.parent)} {image.size}")


if __name__ == "__main__":
    main()
