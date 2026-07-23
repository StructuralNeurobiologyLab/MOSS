#!/usr/bin/env python3
"""
Pixel-art animal avatars + per-user border colors for the Hub.

Each connected user is assigned (deterministically, by join order) one cute
pixel-art animal and one high-contrast border color. Avatars are authored as
small character grids and rendered to self-contained SVG — no external assets,
so they render anywhere QtSvg is available.

Add a new animal by appending a (name, palette, rows) entry to ANIMALS.
Rows may be ragged; short rows are padded with transparent cells.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtGui import QPainter


# High-contrast border colors, cycled by join order. Chosen to stay distinct
# against the dark hub background and against each other.
USER_COLORS = [
    "#e6584d",  # red
    "#4d90e6",  # blue
    "#5fbf6a",  # green
    "#e6a33c",  # amber
    "#a86fe0",  # purple
    "#3cc9c0",  # teal
    "#e86fb0",  # pink
    "#c9c23c",  # chartreuse
    "#e6773c",  # orange
    "#6f8de0",  # periwinkle
]


# Each animal: char -> hex color. Space / '.' = transparent.
# Grids are ~12 wide; the renderer pads ragged rows.
_ANIMAL_DEFS = {
    "cat": (
        {"B": "#e8863b", "W": "#ffffff", "K": "#2b2b2b", "P": "#f2a0a0"},
        [
            "  B      B  ",
            "  BB    BB  ",
            "  BBBBBBBB  ",
            " BBBBBBBBBB ",
            " BBBBBBBBBB ",
            " BWKBBBBKWB ",
            " BBBBBBBBBB ",
            " BBBBPPBBBB ",
            " BBBKKKKBBB ",
            "  BBBBBBBB  ",
            "   B B B B  ",
        ],
    ),
    "fox": (
        {"O": "#e5622d", "W": "#ffffff", "K": "#222222"},
        [
            "  O      O  ",
            "  OO    OO  ",
            "  OWO  OWO  ",
            " OOOOOOOOOO ",
            " OWWOOOOWWO ",
            " OWKOOOOKWO ",
            " OOOOOOOOOO ",
            " WOOOKKOOOW ",
            " WWOOOOOOWW ",
            "  WWOOOOWW  ",
            "   WWWWWW   ",
        ],
    ),
    "owl": (
        {"N": "#8a5a2b", "Y": "#f4c542", "K": "#2b2b2b", "B": "#5c3a1a"},
        [
            "  N      N  ",
            " NNNNNNNNNN ",
            " NNNNNNNNNN ",
            " NYYNNNNYYN ",
            " NYKYNNYKYN ",
            " NYYNBBNYYN ",
            " NNNNBBNNNN ",
            " NNNNNNNNNN ",
            "  NNNNNNNN  ",
            "   N NN N   ",
            "   B    B   ",
        ],
    ),
    "frog": (
        {"G": "#5fbf5a", "D": "#3f9f3a", "W": "#ffffff", "K": "#1a1a1a"},
        [
            "  G      G  ",
            " GWG    GWG ",
            " GWKG  GKWG ",
            " GGGGGGGGGG ",
            "GGGGGGGGGGGG",
            "GGGGGGGGGGGG",
            "GDDGGGGGGDDG",
            " GGGGGGGGGG ",
            " GGKKKKKKGG ",
            "  GGGGGGGG  ",
            " GG      GG ",
        ],
    ),
    "panda": (
        {"W": "#f2f2f2", "K": "#2b2b2b", "P": "#f2a0a0"},
        [
            " K        K ",
            " KK      KK ",
            " WWWWWWWWWW ",
            "WWWWWWWWWWWW",
            "WWKKWWWWKKWW",
            "WWKKWWWWKKWW",
            "WWWWWPPWWWWW",
            "WWWWKKKKWWWW",
            " WWWWWWWWWW ",
            "  WWWWWWWW  ",
            "            ",
        ],
    ),
    "rabbit": (
        {"W": "#f4f4f4", "G": "#d8d8d8", "K": "#2b2b2b", "P": "#f2a0a0"},
        [
            "  W      W  ",
            "  WG    GW  ",
            "  WG    GW  ",
            "  WG    GW  ",
            "  WWW  WWW  ",
            " WWWWWWWWWW ",
            " WWKWWWWKWW ",
            " WWWWPPWWWW ",
            " WWWKKKKWWW ",
            "  WWWWWWWW  ",
            "   WWWWWW   ",
        ],
    ),
    "bear": (
        {"B": "#a3733f", "D": "#7d5426", "K": "#2b2b2b", "P": "#c98d5a"},
        [
            " BB      BB ",
            "BDDB    BDDB",
            "BBBBBBBBBBBB",
            "BBBBBBBBBBBB",
            "BBKBBBBBBKBB",
            "BBBBBBBBBBBB",
            "BBBBPPPPBBBB",
            "BBBBPKKPBBBB",
            "BBBBPPPPBBBB",
            " BBBBBBBBBB ",
            "  BBBBBBBB  ",
        ],
    ),
    "penguin": (
        {"K": "#2b2b2b", "W": "#f4f4f4", "O": "#f0a93c", "E": "#1a1a1a"},
        [
            "   KKKKKK   ",
            "  KKKKKKKK  ",
            " KKKKKKKKKK ",
            " KKEKKKKEKK ",
            " KKWWOOWWKK ",
            " KWWWOOWWWK ",
            " KWWWWWWWWK ",
            " KWWWWWWWWK ",
            " KWWWWWWWWK ",
            "  KWWWWWWK  ",
            "  O      O  ",
        ],
    ),
}

ANIMAL_NAMES = list(_ANIMAL_DEFS.keys())


def _grid_to_svg(palette: dict, rows: list[str], cell: int = 8) -> str:
    """Render a character grid to an SVG string of <rect> pixels."""
    width = max(len(r) for r in rows)
    height = len(rows)
    w_px = width * cell
    h_px = height * cell
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_px}" height="{h_px}" '
        f'viewBox="0 0 {w_px} {h_px}" shape-rendering="crispEdges">'
    ]
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            color = palette.get(ch)
            if color is None:
                continue
            parts.append(
                f'<rect x="{x * cell}" y="{y * cell}" '
                f'width="{cell}" height="{cell}" fill="{color}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def animal_for_index(index: int) -> str:
    """Deterministic animal name for a given join index."""
    return ANIMAL_NAMES[index % len(ANIMAL_NAMES)]


def color_for_index(index: int) -> str:
    """Deterministic border color for a given join index."""
    return USER_COLORS[index % len(USER_COLORS)]


def animal_svg(name: str, cell: int = 8) -> str:
    """Return the SVG markup for a named animal (falls back to first animal)."""
    palette, rows = _ANIMAL_DEFS.get(name, next(iter(_ANIMAL_DEFS.values())))
    return _grid_to_svg(palette, rows, cell=cell)


def animal_pixmap(name: str, size: int = 96) -> QPixmap:
    """Render a named animal to a crisp QPixmap of the given square size."""
    svg = animal_svg(name)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    # Keep pixels crisp — no smoothing on upscale.
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    renderer.render(painter)
    painter.end()
    return pm
