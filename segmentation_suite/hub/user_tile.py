#!/usr/bin/env python3
"""
UserTile — one connected user's box in the Hub's central grid.

Shows a pixel-art animal avatar, display name, live crop count, an owner
marker, and an include/exclude toggle that controls whether this user's crops
feed the cluster training set. Clicking the tile body opens the user's crop
gallery.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSizePolicy
)

from .animals import animal_pixmap


class UserTile(QFrame):
    """A clickable card representing one connected user."""

    clicked = pyqtSignal(str)              # user_id
    toggled = pyqtSignal(str, bool)        # user_id, included

    def __init__(self, user_id: str, display_name: str, animal: str,
                 color: str, is_owner: bool = False, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.display_name = display_name
        self.animal = animal
        self.color = color
        self.is_owner = is_owner
        self._included = True
        self._crop_count = 0

        self.setFixedSize(150, 172)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("userTile")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header row: owner crown (left) + include toggle (right)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.owner_label = QLabel("♛ owner" if is_owner else "")
        self.owner_label.setStyleSheet("color: #f4c542; font-size: 10px; font-weight: bold;")
        header.addWidget(self.owner_label)
        header.addStretch()
        self.include_check = QCheckBox()
        self.include_check.setChecked(True)
        self.include_check.setToolTip("Include this user's crops in cluster training")
        self.include_check.toggled.connect(self._on_toggle)
        header.addWidget(self.include_check)
        layout.addLayout(header)

        # Avatar
        self.avatar = QLabel()
        self.avatar.setPixmap(animal_pixmap(animal, size=80))
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avatar.setFixedHeight(84)
        layout.addWidget(self.avatar)

        # Name
        self.name_label = QLabel(display_name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("color: #f0f0f0; font-size: 13px; font-weight: 600;")
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)

        # Crop count
        self.count_label = QLabel("0 crops")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        layout.addWidget(self.count_label)

        layout.addStretch()
        self._apply_style()

    # ------------------------------------------------------------------ state
    def set_crop_count(self, n: int):
        self._crop_count = n
        self.count_label.setText(f"{n} crop{'s' if n != 1 else ''}")

    def increment_crops(self, by: int = 1):
        self.set_crop_count(self._crop_count + by)

    @property
    def included(self) -> bool:
        return self._included

    def set_included(self, included: bool):
        """Set inclusion without re-emitting the toggled signal."""
        self.include_check.blockSignals(True)
        self.include_check.setChecked(included)
        self.include_check.blockSignals(False)
        self._included = included
        self._apply_style()

    # ------------------------------------------------------------------ events
    def _on_toggle(self, checked: bool):
        self._included = checked
        self._apply_style()
        self.toggled.emit(self.user_id, checked)

    def mousePressEvent(self, event):
        # Clicks on the checkbox are handled by the checkbox itself.
        child = self.childAt(event.position().toPoint())
        if child is not self.include_check:
            self.clicked.emit(self.user_id)
        super().mousePressEvent(event)

    # ------------------------------------------------------------------ style
    def _apply_style(self):
        if self._included:
            border = self.color
            bg = "#2a2a2e"
            avatar_op = ""
        else:
            border = "#555555"
            bg = "#232326"
            avatar_op = ""  # dimming handled via opacity effect if desired
        self.setStyleSheet(f"""
            QFrame#userTile {{
                background-color: {bg};
                border: 3px solid {border};
                border-radius: 12px;
            }}
        """)
        self.avatar.setEnabled(self._included)
        self.name_label.setStyleSheet(
            "color: %s; font-size: 13px; font-weight: 600;"
            % ("#f0f0f0" if self._included else "#888888")
        )
