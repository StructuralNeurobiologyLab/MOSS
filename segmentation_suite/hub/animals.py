#!/usr/bin/env python3
"""
Minimalist single-color animal brand-marks for the Hub.

Line-icon style (Lucide/Feather/Swiss/WWF register): flat, geometric,
single-color vector marks — NOT illustrations. Each user is assigned one
mark + one color by join order; the mark is tinted to the user color.
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

# name -> minimalist SVG mark (viewBox 0 0 24 24, currentColor).
_MARKS = {
    'cat': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9.5 L7.2 4 L10.6 7 Q12 7.6 13.4 7 L16.8 4 L18 9.5 Q18 15 15 17.5 Q12 19.6 9 17.5 Q6 15 6 9.5 Z"/><path d="M12 12.6 L11.2 13.6 L12.8 13.6 Z"/><path d="M11.2 13.6 Q12 14.4 12.8 13.6"/><path d="M7 12.2 L3.6 11.4 M7 14 L3.6 14.2"/><path d="M17 12.2 L20.4 11.4 M17 14 L20.4 14.2"/></svg>'),
    'fox': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19.5 L5.5 11.5 L6.6 4.2 L9.8 8 L14.2 8 L17.4 4.2 L18.5 11.5 Z"/><path d="M9 11.2 L10.6 11.8 M15 11.2 L13.4 11.8"/><path d="M12 15.4 L11.2 16.4 L12.8 16.4 Z"/></svg>'),
    'owl': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 8.5 L7 4.5 L9.2 7.2 Q12 5.8 14.8 7.2 L17 4.5 L19 8.5 Q19.6 13 16 16.2 Q12 19.2 8 16.2 Q4.4 13 5 8.5 Z"/><circle cx="9.4" cy="11" r="2.2"/><circle cx="14.6" cy="11" r="2.2"/><path d="M11 13.6 L12 15.2 L13 13.6"/></svg>'),
    'frog': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 11.5 Q4 18 12 18.2 Q20 18 20 11.5 Z"/><circle cx="8" cy="10.2" r="2.3"/><circle cx="16" cy="10.2" r="2.3"/><path d="M7.5 14.4 Q12 16 16.5 14.4"/></svg>'),
    'panda': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.4" cy="7.6" r="2.1"/><circle cx="16.6" cy="7.6" r="2.1"/><circle cx="12" cy="13" r="6"/><ellipse cx="9.6" cy="12" rx="1.5" ry="2.1" transform="rotate(-18 9.6 12)"/><ellipse cx="14.4" cy="12" rx="1.5" ry="2.1" transform="rotate(18 14.4 12)"/><path d="M12 15 L11.2 16 L12.8 16 Z"/></svg>'),
    'rabbit': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M11.4 12 C9.9 9 9.4 5.6 10.1 4.1 C10.5 3.3 11.3 3.5 11.5 4.6 C12 7.5 12.1 9.6 12 12 Z"/><path d="M12.6 12 C14.1 9 14.6 5.6 13.9 4.1 C13.5 3.3 12.7 3.5 12.5 4.6 C12 7.5 11.9 9.6 12 12 Z"/><ellipse cx="12" cy="15.2" rx="4.7" ry="4"/><circle cx="10.2" cy="14.6" r="0.35"/><circle cx="13.8" cy="14.6" r="0.35"/><path d="M12 15.6 L11.3 16.5 L12.7 16.5 Z"/></svg>'),
    'bear': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.6" cy="8" r="2.2"/><circle cx="16.4" cy="8" r="2.2"/><circle cx="12" cy="13.2" r="5.6"/><ellipse cx="12" cy="15.2" rx="2.4" ry="2"/><path d="M12 14 L11.3 14.9 L12.7 14.9 Z"/><circle cx="9.6" cy="11.6" r="0.35"/><circle cx="14.4" cy="11.6" r="0.35"/></svg>'),
    'penguin': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.8 C8.9 3.8 8 6.4 8.2 8.6 C6.6 10 6.2 13 6.2 15.4 C6.2 18.4 8.6 20.4 12 20.4 C15.4 20.4 17.8 18.4 17.8 15.4 C17.8 13 17.4 10 15.8 8.6 C16 6.4 15.1 3.8 12 3.8 Z"/><path d="M9.4 9.4 Q8.6 15 12 19 Q15.4 15 14.6 9.4"/><path d="M11 8 L12 9.8 L13 8 Z"/><circle cx="10.4" cy="7" r="0.35"/><circle cx="13.6" cy="7" r="0.35"/></svg>'),
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
