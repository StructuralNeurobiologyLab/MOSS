#!/usr/bin/env python3
"""
Minimalist single-color animal brand-marks for the Hub.

Thin, minimal, one-color LOGO-style marks (boutique/company-logo register):
elegant side-profile silhouettes, NOT cartoon faces. Each user is assigned
one mark + one color by join order; the mark is tinted to the user color.
SVGs use currentColor and are recolored on render (Qt) or via CSS (web).
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QPixmap, QPainter
from PyQt6.QtSvg import QSvgRenderer

# High-contrast border/tint colors, cycled by join order.
USER_COLORS = [
    "#e6584d",
    "#4d90e6",
    "#5fbf6a",
    "#e6a33c",
    "#a86fe0",
    "#3cc9c0",
    "#e86fb0",
    "#c9c23c",
    "#e6773c",
    "#6f8de0",
]

# name -> minimalist thin SVG logo-mark (viewBox 0 0 24 24, currentColor).
_MARKS = {
    'cat': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M6.2 9.4 C6.2 8.1 6.6 7.3 7.2 6.7 L7.6 4.5 L8.9 6.4 C9.1 6.35 9.3 6.35 9.5 6.4 L10.6 4.6 L11.2 6.8 C11.9 7.7 13.0 9.3 14.3 11.6 C15.4 13.5 16.0 15.0 16.1 16.3 C16.2 17.3 16.0 18.0 15.8 18.2 L7.7 18.2 C7.0 18.2 6.8 17.0 6.8 15.0 C6.8 12.6 6.2 10.8 6.2 9.4 Z"/><path d="M16.0 16.2 C16.4 14.2 17.6 13.0 18.8 13.4 C20.0 13.8 20.0 15.7 18.9 16.4 C18.1 16.9 17.1 16.6 16.8 15.8"/></svg>'),
    'fox': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M4.3 12.4 C5.6 11.8 6.5 11.1 7.1 10.0 L6.6 5.0 L9.4 8.3 C9.9 8.1 10.4 8.1 10.9 8.3 L13.5 5.1 L13.1 10.0 C13.6 11.4 13.7 12.6 13.5 14.0"/><path d="M4.3 12.4 C5.4 12.8 6.5 13.0 7.5 12.8"/><path d="M13.5 14.0 C13.3 15.7 13.4 17.2 13.2 18.3 L9.0 18.3 C8.2 18.3 7.8 17.4 8.0 16.4"/><path d="M13.6 15.6 C15.6 15.2 18.0 13.7 18.9 10.9 C19.1 10.1 18.7 9.6 18.0 9.9 C16.5 10.6 14.4 13.6 13.2 18.0"/></svg>'),
    'owl': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M8.1 6.5 L9.1 4.3 L10.1 6.4"/><path d="M13.9 6.4 L14.9 4.3 L15.9 6.5"/><path d="M9.0 6.3 C7.3 7.6 6.5 9.6 6.5 11.9 C6.5 15.4 8.6 18.0 12.0 18.0 C15.4 18.0 17.5 15.4 17.5 11.9 C17.5 9.6 16.7 7.6 15.0 6.3"/><path d="M11.3 8.5 C11.6 9.3 12.4 9.3 12.7 8.5"/><path d="M9.1 9.2 C8.2 11.1 8.2 14.0 9.3 16.0"/><path d="M14.9 9.2 C15.8 11.1 15.8 14.0 14.7 16.0"/><path d="M9.4 18.0 L8.5 19.3 M14.6 18.0 L15.5 19.3"/><path d="M4.8 19.5 L19.2 19.5"/></svg>'),
    'frog': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M4.9 14.7 C4.6 12.4 6.1 10.3 8.5 9.6 C10.7 8.95 12.9 9.3 14.4 10.6 C15.6 11.5 16.4 12.5 16.5 13.9 C16.55 14.5 16.0 15.0 15.2 15.0 L7.0 15.0 C5.6 15.0 4.95 15.7 4.9 14.7 Z"/><path d="M16.4 14.1 C15.3 14.5 14.1 14.5 13.1 14.3"/><path d="M6.2 11.2 C7.7 12.9 7.9 14.4 6.9 15.0"/><path d="M11.8 14.9 C11.9 15.9 12.2 16.7 12.7 17.1"/><circle cx="13.8" cy="9.2" r="1.1"/></svg>'),
    'panda': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><circle cx="8.0" cy="6.7" r="1.9"/><circle cx="16.0" cy="6.7" r="1.9"/><path d="M6.4 8.1 C4.6 9.5 4.0 11.7 4.0 13.7 C4.0 16.9 6.6 19.0 12.0 19.0 C17.4 19.0 20.0 16.9 20.0 13.7 C20.0 11.7 19.4 9.5 17.6 8.1"/><path d="M8.9 17.4 L15.4 4.3"/><path d="M15.4 4.3 C14.4 4.5 13.6 5.1 13.4 6.1 C14.4 6.1 15.2 5.5 15.4 4.3 Z"/><path d="M13.8 6.9 C12.9 7.1 12.2 7.7 12.0 8.7 C12.9 8.7 13.6 8.1 13.8 6.9 Z"/><path d="M9.0 12.4 C10.2 11.8 11.5 11.9 12.4 12.7"/></svg>'),
    'rabbit': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M6.2 12.8 C6.0 11.4 6.4 10.4 7.2 9.8 C6.4 8.4 6.6 5.4 7.2 4.2 C7.6 3.6 8.4 3.8 8.6 4.6 C8.9 5.8 8.8 8.2 8.4 9.4"/><path d="M8.4 9.4 C8.8 8.2 9.6 6.2 10.4 5.6 C11.0 5.2 11.6 5.6 11.6 6.4 C11.6 7.8 10.8 9.4 9.8 10.2 C11.6 10.6 13.6 11.8 15.2 13.6 C16.8 15.4 17.2 17.0 17.0 18.0"/><path d="M6.2 12.8 C6.2 14.8 6.2 16.8 7.0 18.0 L17.0 18.0"/><path d="M6.2 12.8 C6.6 13.0 7.0 13.0 7.4 12.8"/><circle cx="6.4" cy="11.8" r="0.32"/></svg>'),
    'bear': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M6.5 12.7 C6.2 11.7 6.7 10.7 7.7 10.3 C8.3 9.3 9.1 8.5 10.2 8.3 C10.2 7.1 11.0 6.5 11.8 6.9 C12.4 7.2 12.4 8.0 11.8 8.4 C12.4 8.5 13.2 8.7 14.0 9.3 C15.4 10.1 16.2 11.5 16.4 13.3 C16.5 14.7 16.2 16.1 15.3 17.0 L8.9 17.0 C8.0 17.0 7.4 16.0 7.2 14.6 C7.0 13.6 6.7 13.1 6.5 12.7 Z"/><path d="M6.7 12.9 C7.8 13.3 8.6 13.3 9.4 13.0"/><circle cx="7.2" cy="11.9" r="0.3"/></svg>'),
    'penguin': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"><path d="M9.4 5.2 C10.0 4.2 11.4 4.0 12.4 4.8 C13.2 5.4 13.6 6.6 13.4 7.8 C14.6 9.6 15.2 12.0 15.2 14.4 C15.2 16.6 14.6 18.0 13.6 18.6"/><path d="M9.4 5.2 L6.6 6.2 L9.2 7.2"/><path d="M9.2 7.2 C8.4 9.2 8.0 11.6 8.0 14.0 C8.0 16.4 8.6 18.0 9.6 18.6"/><path d="M13.6 18.6 L9.6 18.6"/><path d="M13.6 18.6 L15.4 19.6 M9.6 18.6 L7.8 19.6"/><path d="M12.6 8.6 C13.4 11.0 13.6 14.0 12.8 16.4"/><circle cx="11.2" cy="6.6" r="0.35"/></svg>'),
}

ANIMAL_NAMES = list(_MARKS.keys())


def animal_for_index(index: int) -> str:
    return ANIMAL_NAMES[index % len(ANIMAL_NAMES)]


def color_for_index(index: int) -> str:
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
