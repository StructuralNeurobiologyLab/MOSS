#!/usr/bin/env python3
"""
Minimalist single-color animal brand-marks for the Hub.

Thin, minimal, one-color LOGO-style marks (boutique/company-logo register):
elegant side-profile silhouettes, NOT cartoon faces. Each user is assigned
one mark + one color by join order; the mark is tinted to the user color.
SVGs use currentColor and are recolored on render (Qt) or via CSS (web).

The 13 marks were hand-picked from candidate sets all drawn to match the
anchor CAT (thin 1.1 stroke, fill none, minimal internal detail).
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QPixmap, QPainter
from PyQt6.QtSvg import QSvgRenderer

# Border/tint colors: a curated GENTLE BOTANICAL palette (clay/rose, sage/fern,
# teal + soft blues, plum/berry) tuned to harmonize with the light "Botanical"
# console theme and to read on BOTH the light ivory background and the dark Qt
# cards. Mark and color COMBINE into the user identity: 12 colors is coprime with
# the 13 marks, so consecutive joins differ in BOTH mark and color, any one mark
# cycles all 12 colors, and no (mark, color) pair repeats until join 157
# (13 x 12 = 156 distinct identities). Order interleaves warm/green/blue so
# consecutive joiners look distinct.
USER_COLORS = [
    "#b4583c",  # clay
    "#388a8a",  # teal
    "#a77f35",  # goldenrod
    "#4676b4",  # dusty blue
    "#579348",  # sage
    "#b74e87",  # berry
    "#3f85ab",  # sky blue
    "#768f3d",  # olive
    "#975bae",  # plum
    "#3c8b63",  # fern
    "#5c67bc",  # periwinkle
    "#b84756",  # rose
]

# name -> minimalist thin SVG logo-mark (viewBox 0 0 24 24, currentColor).
_MARKS = {
    'cat': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M6.2 9.4 C6.2 8.1 6.6 7.3 7.2 6.7 L7.6 4.5 L8.9 6.4 C9.1 6.35 9.3 6.35 9.5 6.4 L10.6 4.6 L11.2 6.8 C11.9 7.7 13.0 9.3 14.3 11.6 C15.4 13.5 16.0 15.0 16.1 16.3 C16.2 17.3 16.0 18.0 15.8 18.2 L7.7 18.2 C7.0 18.2 6.8 17.0 6.8 15.0 C6.8 12.6 6.2 10.8 6.2 9.4 Z"/><path d="M16.0 16.2 C16.4 14.2 17.6 13.0 18.8 13.4 C20.0 13.8 20.0 15.7 18.9 16.4 C18.1 16.9 17.1 16.6 16.8 15.8"/></svg>'),
    'fox': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M 4.52 12.4 C 5.54 11.63 6.13 11.12 6.65 10.45 L 6.9 6.03 L 8 8.32 L 9.37 5.94 L 10.13 8.49 C 11.23 9.25 12.59 10.19 13.62 11.46 C 14.46 12.57 14.89 13.93 14.8 15.38 C 14.72 16.48 14.46 17.25 13.96 17.5 L 9.11 17.5 C 8.94 15.88 8.94 14.53 8.86 13.68 C 8.77 12.74 8.26 12.48 7.41 12.74 C 6.48 12.99 5.54 12.83 4.52 12.4 Z"/><path d="M 14.46 14.78 C 16.25 15.46 16.76 17.33 15.91 18.52 C 14.89 19.88 11.66 20.05 8.94 19.62 C 7.41 19.37 6.39 18.52 6.22 17.5 C 6.9 18.35 8.43 18.77 10.47 18.77 C 12.85 18.77 14.55 18.35 15.06 17.5 C 15.57 16.65 15.06 15.63 14.21 15.29 C 14.21 15.12 14.29 14.95 14.46 14.78 Z"/></svg>'),
    'dog': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M4.2 8.0 C4.9 7.6 5.7 7.4 6.4 7.2 C6.8 6.3 7.3 5.9 8.0 6.1 C8.7 6.3 9.0 6.9 9.0 7.6 C9.1 8.9 8.7 10.0 8.1 10.4 C8.9 10.1 9.4 9.3 9.7 8.5 C11.2 8.0 13.4 8.0 15.2 8.9 C16.3 9.5 16.9 11.1 16.9 13.1 C16.9 15.4 16.7 17.1 16.4 17.9 L13.8 17.9 C12.6 17.9 11.4 17.4 10.2 17.2 C9.4 17.1 8.6 17.1 8.0 17.3 L7.6 17.9 L6.9 17.9 C6.9 15.5 6.9 12.6 7.0 11.0 C7.1 10.2 6.9 9.7 6.6 9.4 C5.8 9.7 4.9 9.4 4.3 8.7 Z"/><path d="M16.9 13.0 C18.3 12.3 19.1 10.8 18.6 9.6 C18.3 8.9 17.7 8.7 17.2 9.1"/></svg>'),
    'owl': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M9.9 5.4 L9.1 3.1 L8.5 5.7 C6.7 7.2 5.7 9.4 5.7 12 C5.7 16.1 8.1 19 12 19 C15.9 19 18.3 16.1 18.3 12 C18.3 9.4 17.3 7.2 15.5 5.7 L14.9 3.1 L14.1 5.4 C13.3 6.2 12.8 6.4 12 6.4 C11.2 6.4 10.7 6.2 9.9 5.4 Z"/><path d="M8.3 9.3 C7.7 12 7.9 15 9 17.6"/><path d="M15.7 9.3 C16.3 12 16.1 15 15 17.6"/><path d="M12 8.6 L12 9.4"/><path d="M10.3 19 L9.7 20.5 M10.3 19 L10.9 20.5"/><path d="M13.7 19 L13.1 20.5 M13.7 19 L14.3 20.5"/></svg>'),
    'bird': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M4.8 9.15 L6.6 8.75 C7.1 8.55 7.2 8.0 7.7 7.7 C8.9 7.0 10.6 6.9 12.2 7.9 C13.4 8.7 14.0 9.6 14.3 10.6 C15.0 9.8 16.0 8.6 16.9 7.6 C16.6 9.0 16.2 10.3 15.4 11.4 C14.4 13.0 12.2 14.0 10.0 13.8 C7.8 13.6 6.2 12.6 5.9 11.0 C5.8 10.4 6.0 9.85 6.6 9.65 L4.8 9.15 Z"/><path d="M8.9 13.8 L8.7 15.6"/><path d="M10.7 13.9 L10.9 15.6"/><path d="M6.8 15.4 C9.7 15.8 12.4 15.9 13.9 15.1"/></svg>'),
    'rabbit': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M5.2 12.8 C4.9 11.0 5.6 10.2 6.4 9.6 C6.3 8.0 6.4 5.6 6.9 4.1 C7.0 3.5 7.6 3.5 7.8 4.1 C8.2 5.7 8.1 7.2 7.9 8.4 C8.3 7.2 8.7 5.3 9.2 4.0 C9.4 3.4 10.0 3.5 10.1 4.2 C10.5 5.9 10.3 7.9 10.2 9.3 C11.6 8.6 13.6 8.2 15.3 9.5 C16.9 10.6 17.6 12.7 17.5 14.9 C17.4 16.9 16.8 18.6 15.6 19.1 L8.2 19.1 C7.0 19.1 6.4 17.5 6.4 15.8 C6.4 14.6 6.3 13.6 5.2 12.8 Z"/><path d="M17.2 15.1 C18.3 14.5 19.3 15.2 19.1 16.3 C18.9 17.3 17.6 17.4 17.1 16.4"/></svg>'),
    'deer': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><g transform="translate(0.9 2.08)"><path d="M4.9 7.7 C5.1 6.9 5.7 6.4 6.5 6.4 C7.1 6.4 7.3 6.9 7.4 7.6 C7.55 8.7 7.95 9.8 8.55 10.8 C9.1 11.7 9.9 12.4 11.2 12.7 C13 13.1 15 12.9 16.3 13.6 C16.9 13.95 17.1 14.7 16.95 15.5 C16.8 16.1 16.3 16.5 15.5 16.5 L8.7 16.5 C8 16.5 7.7 16 7.75 15.2 C7.85 13.9 7.9 12.3 7.7 11.1 C7.5 10 7.1 9.1 6.3 8.6 C5.9 8.4 5.3 8.1 4.9 7.7 Z"/><path d="M8.9 16.3 C10.6 15.4 12.6 15.3 14.2 15.9"/><path d="M16.9 14.9 C17.6 15.05 17.85 15.65 17.5 16.2"/><path d="M6.5 6.2 L6.5 5.72"/><path d="M6.5 5.72 C6.02 5.32 5.62 5.08 5.38 4.68"/><path d="M5.38 4.68 C4.98 4.28 4.74 3.96 4.58 3.56"/><path d="M5.38 4.68 C5.38 4.2 5.46 3.8 5.54 3.4"/><path d="M6.5 5.72 C6.98 5.32 7.38 5.08 7.62 4.68"/><path d="M7.62 4.68 C8.02 4.28 8.26 3.96 8.42 3.56"/><path d="M7.62 4.68 C7.62 4.2 7.54 3.8 7.46 3.4"/></g></svg>'),
    'bear': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M5.5 8.9 C5.3 8.1 5.8 7.6 6.2 7.4 C6.0 6.6 6.2 5.8 6.9 5.6 C7.6 5.4 8.0 6.1 8.0 6.8 C8.5 6.4 9.1 6.4 9.5 6.7 C9.6 6.0 10.2 5.5 10.8 5.7 C11.5 5.9 11.6 6.9 11.1 7.4 C11.6 7.9 12.4 8.0 13.0 7.8 C14.6 7.7 16.0 8.7 16.6 10.2 C17.4 12.2 17.4 14.8 16.6 16.8 C16.2 17.9 15.4 18.5 14.4 18.4 C12.6 18.7 10.4 18.7 8.8 18.4 C8.0 18.3 7.5 17.9 7.4 17.1 C7.2 15.2 7.2 12.8 7.0 11.6 C6.8 10.4 6.2 9.9 5.8 9.5 C5.6 9.3 5.5 9.1 5.5 8.9 Z"/></svg>'),
    'raccoon': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M19.8 9.7 C19.5 8.4 18.6 7.5 17.3 6.9 L17.2 5.0 L16.0 6.4 C15.75 6.2 15.45 6.2 15.2 6.35 L14.5 5.0 L13.3 6.6 C11.2 7.1 8.4 8.1 7.2 10.8 C6.6 12.2 6.5 14.6 7.2 16.6 C7.6 17.8 8.2 18.2 9.0 18.2 L14.7 18.2 C15.5 18.2 16.0 15.5 16.4 13.2 C16.8 11.3 17.6 10.7 18.4 10.5 C19.0 10.35 19.5 10.15 19.8 9.7 Z"/><path d="M7.6 12.9 C5.6 11.7 3.7 9.7 4.3 7.5 C4.6 6.3 6.1 5.8 7.4 6.6 C8.8 7.5 9.2 9.7 8.6 11.6 C8.35 12.4 8.0 12.7 7.6 12.9 Z"/><path d="M5.2 9.5 C6.0 9.1 7.0 9.1 7.9 9.6"/><path d="M4.6 7.8 C5.3 7.3 6.3 7.2 7.2 7.7"/></svg>'),
    'penguin': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M13.0 4.1 C11.2 3.7 10.4 4.7 10.5 5.9 L8.6 6.0 L10.5 6.9 C10.1 7.7 9.2 9.6 9.0 12.7 C8.9 15.4 9.8 17.6 11.4 18.5 C12.0 18.7 12.8 18.7 13.4 18.5 C15.0 17.6 15.7 15.2 15.6 12.4 C15.5 10.2 15.3 8.2 14.6 7.3 C15.1 6.6 15.0 5.1 13.0 4.1 Z"/><path d="M11.4 8.5 C10.3 10.7 10.4 13.4 11.6 15.0"/><path d="M11.1 18.6 L10.6 19.7 M13.0 18.6 L13.5 19.7"/></svg>'),
    'octopus': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M8 11 C7.4 5.2, 16.6 5.2, 16 11"/><path d="M8.4 11 C8 13.2, 8.8 15.8, 10.4 17.4 C11.1 18.1, 11.9 17.8, 11.7 16.9"/><path d="M10 11.5 C9.7 13.6, 10.6 15.9, 12.3 17.2 C13 17.7, 13.7 17.3, 13.4 16.5"/><path d="M11.7 11.7 C11.7 13.8, 12.8 15.8, 14.5 16.7"/><path d="M13.3 11.6 C13.6 13.4, 14.8 15, 16.3 15.8"/><path d="M14.9 11.2 C15.4 12.8, 16.6 13.9, 18 14.4"/></svg>'),
    'frog': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 14.7 C3.5 13.2 4.5 12.4 5.9 12.3 C6.7 12.25 7.1 12.5 7.3 13.0 C7.9 12.8 8.7 12.8 9.4 12.5 C10.8 12.0 11.7 9.7 13.6 9.0 C16.3 8.5 18.8 10.2 18.6 13.7 C18.4 15.5 17.8 17.0 16.5 17.9 C13 18.2 8.5 18.2 5.0 18.0 C4.1 17.1 3.5 16.3 3.5 14.7 Z"/><path d="M16.8 13.4 C15.3 14.9 12.6 17.4 8.6 18.0"/></svg>'),
    'panda': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5 C13.4 5 14.7 5.5 15.4 8.3 C17 8.8 17.6 10.6 17.4 12.2 C17.6 14 17.5 16.2 17 17.8 C16.6 19.2 15.6 19.8 14.4 19.8 C13.6 19.8 13 19.4 12.7 18.9 C13.5 18.5 14 17.7 13.9 16.9 C13.8 15.9 13 15.3 12 15.3 C11 15.3 10.2 15.9 10.1 16.9 C10 17.7 10.5 18.5 11.3 18.9 C11 19.4 10.4 19.8 9.6 19.8 C8.4 19.8 7.4 19.2 7 17.8 C6.5 16.2 6.4 14 6.6 12.2 C6.4 10.6 7 8.8 8.6 8.3 C9.3 5.5 10.6 5 12 5 Z"/><path d="M9.6 8.3 C9.5 6.4 8.1 5.5 6.9 5.9 C5.7 6.3 5.7 8.1 6.9 8.9"/><path d="M14.4 8.3 C14.5 6.4 15.9 5.5 17.1 5.9 C18.3 6.3 18.3 8.1 17.1 8.9"/></svg>'),
}

ANIMAL_NAMES = list(_MARKS.keys())


def animal_for_index(index: int) -> str:
    return ANIMAL_NAMES[index % len(ANIMAL_NAMES)]


def color_for_index(index: int) -> str:
    """Curated gentle-botanical tint. Combined with animal_for_index (13 marks)
    so consecutive joins differ in BOTH mark and color; a mark cycles all 12
    colors before any (mark, color) pair repeats (13 x 12 = 156 identities)."""
    return USER_COLORS[index % len(USER_COLORS)]


def animal_svg(name: str, color: str = "currentColor") -> str:
    """SVG markup for a named mark, optionally recolored (else currentColor)."""
    svg = _MARKS.get(name, next(iter(_MARKS.values())))
    if color and color != "currentColor":
        svg = svg.replace("currentColor", color)
    return svg


def animal_pixmap(name: str, size: int = 96, color: str = "#c9d1d9") -> QPixmap:
    """Render a named mark to a crisp QPixmap tinted to color."""
    renderer = QSvgRenderer(QByteArray(animal_svg(name, color).encode("utf-8")))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    return pm
