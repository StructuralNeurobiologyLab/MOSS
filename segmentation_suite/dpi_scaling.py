#!/usr/bin/env python3
"""
DPI-aware scaling utility for consistent UI across different monitors.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont


# Reference DPI (standard 96 DPI monitor)
REFERENCE_DPI = 96.0

# Cached scale factor
_scale_factor = None


def get_scale_factor() -> float:
    """Get the scale factor based on screen DPI.

    Returns a multiplier relative to a standard 96 DPI display.
    E.g., a 144 DPI display returns 1.5
    """
    global _scale_factor

    if _scale_factor is not None:
        return _scale_factor

    app = QApplication.instance()
    if app is None:
        return 1.0

    screen = app.primaryScreen()
    if screen is None:
        return 1.0

    # Get logical DPI (accounts for system scaling)
    dpi = screen.logicalDotsPerInch()
    _scale_factor = dpi / REFERENCE_DPI

    return _scale_factor


def scaled(value: int) -> int:
    """Scale a pixel value based on screen DPI.

    Args:
        value: Pixel value designed for 96 DPI

    Returns:
        Scaled pixel value for current screen
    """
    return int(value * get_scale_factor())


def scaled_font_size(base_size: int) -> int:
    """Scale a font size based on screen DPI.

    Args:
        base_size: Font point size for 96 DPI

    Returns:
        Scaled font size
    """
    # Font scaling is typically less aggressive than pixel scaling
    factor = get_scale_factor()
    # Use square root for gentler font scaling
    font_factor = (factor + 1) / 2  # Average with 1.0 for less extreme scaling
    return int(base_size * font_factor)


def scaled_font(base_size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Create a scaled font.

    Args:
        base_size: Base font point size for 96 DPI
        weight: Font weight

    Returns:
        QFont with scaled size
    """
    font = QFont()
    font.setPointSize(scaled_font_size(base_size))
    font.setWeight(weight)
    return font


def get_screen_size() -> tuple:
    """Get the primary screen size in pixels.

    Returns:
        (width, height) tuple
    """
    app = QApplication.instance()
    if app is None:
        return (1920, 1080)  # Default fallback

    screen = app.primaryScreen()
    if screen is None:
        return (1920, 1080)

    geometry = screen.geometry()
    return (geometry.width(), geometry.height())


def scaled_window_size(base_width: int, base_height: int, max_screen_fraction: float = 0.9) -> tuple:
    """Calculate scaled window size, capped to screen size.

    Args:
        base_width: Base window width for 96 DPI
        base_height: Base window height for 96 DPI
        max_screen_fraction: Maximum fraction of screen to use (0.0-1.0)

    Returns:
        (width, height) tuple
    """
    screen_w, screen_h = get_screen_size()

    # Scale the base sizes
    width = scaled(base_width)
    height = scaled(base_height)

    # Cap to screen fraction
    max_width = int(screen_w * max_screen_fraction)
    max_height = int(screen_h * max_screen_fraction)

    width = min(width, max_width)
    height = min(height, max_height)

    return (width, height)


def center_on_screen(window) -> None:
    """Center a window on the primary screen.

    Args:
        window: QWidget or QMainWindow to center
    """
    app = QApplication.instance()
    if app is None:
        return

    screen = app.primaryScreen()
    if screen is None:
        return

    screen_geo = screen.geometry()
    window_geo = window.geometry()

    x = (screen_geo.width() - window_geo.width()) // 2
    y = (screen_geo.height() - window_geo.height()) // 2

    window.move(x, y)
