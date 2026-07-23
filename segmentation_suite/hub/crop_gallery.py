#!/usr/bin/env python3
"""
CropGallery — a live thumbnail grid of one user's incoming crops.

Opened from a UserTile click. New crops append in real time. Each thumbnail
shows the image with a small mask-overlay swatch; hovering shows metadata.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget, QGridLayout, QFrame
)


class _Thumb(QFrame):
    """A single crop thumbnail cell."""

    def __init__(self, pixmap: QPixmap, caption: str, color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 138)
        self.setStyleSheet(
            f"QFrame {{ background:#242427; border:2px solid {color}; border-radius:8px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(3)
        img = QLabel()
        img.setPixmap(pixmap.scaled(106, 106, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation))
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(img)
        cap = QLabel(caption)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("color:#9a9a9a; font-size:10px; border:none;")
        lay.addWidget(cap)


class CropGallery(QDialog):
    """Live grid of a single user's incoming crops."""

    COLUMNS = 4

    def __init__(self, user_id: str, display_name: str, color: str, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.color = color
        self._count = 0

        self.setWindowTitle(f"Crops — {display_name}")
        self.resize(560, 620)
        self.setStyleSheet("QDialog { background:#1a1a1d; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        self.header = QLabel(f"{display_name} — 0 crops")
        self.header.setStyleSheet(
            f"color:{color}; font-size:16px; font-weight:bold; border:none;"
        )
        root.addWidget(self.header)
        self.display_name = display_name

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; }")
        scroll.viewport().setStyleSheet("background:transparent;")
        self._container = QWidget()
        self._container.setStyleSheet("background:transparent;")
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(8)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._container)
        root.addWidget(scroll)

    def add_crop(self, image: QImage, caption: str):
        """Append a new crop thumbnail to the grid."""
        pm = QPixmap.fromImage(image)
        row, col = divmod(self._count, self.COLUMNS)
        self._grid.addWidget(_Thumb(pm, caption, self.color), row, col)
        self._count += 1
        self.header.setText(f"{self.display_name} — {self._count} crops")
