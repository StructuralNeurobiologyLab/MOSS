#!/usr/bin/env python3
"""
Subproject panel widget for the sidebar.
Shows subprojects as large clickable buttons with an add (+) button.
"""

import shutil
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QInputDialog, QMessageBox, QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt

from ..dpi_scaling import scaled
from ..project_config import (
    list_subprojects, get_active_subproject, set_active_subproject,
    create_subproject, migrate_to_subprojects, has_subprojects,
    get_subproject_dir, DEFAULT_SUBPROJECT,
)


class SubprojectPanel(QWidget):
    """Panel for managing subprojects (segmentation targets) within a project."""

    # Emitted when user switches to a different subproject
    subproject_changed = pyqtSignal(str)  # subproject_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_dir = None
        self._active_subproject = None
        self._buttons = {}  # name -> QPushButton
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Subprojects")
        header.setStyleSheet(f"""
            QLabel {{
                color: #aaaaaa;
                font-size: {scaled(11)}px;
                padding: {scaled(10)}px {scaled(15)}px {scaled(5)}px {scaled(15)}px;
            }}
        """)
        layout.addWidget(header)

        # Scrollable area for buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(scaled(180))
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QWidget { background: transparent; }
        """)

        self._btn_container = QWidget()
        self._btn_layout = QVBoxLayout(self._btn_container)
        self._btn_layout.setContentsMargins(scaled(10), scaled(2), scaled(10), scaled(2))
        self._btn_layout.setSpacing(scaled(4))
        scroll.setWidget(self._btn_container)
        layout.addWidget(scroll)

        # Start hidden until project loaded
        self.setVisible(False)

    def _make_sp_button(self, name: str, active: bool) -> QPushButton:
        """Create a styled subproject button."""
        btn = QPushButton(name)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.setFixedHeight(scaled(32))
        self._style_sp_button(btn, active)
        btn.clicked.connect(lambda checked, n=name: self._on_sp_clicked(n))

        # Right-click context menu for rename/delete
        btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        btn.customContextMenuRequested.connect(lambda pos, n=name: self._show_context_menu(n, pos))
        return btn

    def _style_sp_button(self, btn: QPushButton, active: bool):
        """Apply active or inactive style to a subproject button."""
        r = scaled(6)
        pad_v = scaled(6)
        pad_h = scaled(12)
        margin_h = scaled(6)
        if active:
            btn.setStyleSheet(f"""
                QPushButton {{
                    color: white;
                    background-color: #3a5a3a;
                    border: none;
                    border-left: 3px solid #4CAF50;
                    border-radius: {r}px;
                    padding: {pad_v}px {pad_h}px;
                    margin: 2px {margin_h}px;
                    text-align: left;
                    font-size: {scaled(12)}px;
                }}
                QPushButton:hover {{
                    background-color: #456b45;
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    color: #bbbbbb;
                    background-color: #353535;
                    border: none;
                    border-radius: {r}px;
                    padding: {pad_v}px {pad_h}px;
                    margin: 2px {margin_h}px;
                    text-align: left;
                    font-size: {scaled(12)}px;
                }}
                QPushButton:hover {{
                    background-color: #424242;
                    color: white;
                }}
            """)

    def _make_add_button(self) -> QPushButton:
        """Create the + (add) button."""
        btn = QPushButton("+")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.setFixedHeight(scaled(32))
        btn.setToolTip("Add new subproject")
        r = scaled(6)
        margin_h = scaled(6)
        btn.setStyleSheet(f"""
            QPushButton {{
                color: #777777;
                background-color: #2a2a2a;
                border: none;
                border-radius: {r}px;
                padding: {scaled(4)}px;
                margin: 2px {margin_h}px;
                font-size: {scaled(16)}px;
            }}
            QPushButton:hover {{
                color: white;
                background-color: #383838;
            }}
        """)
        btn.clicked.connect(self._on_add)
        return btn

    def set_project(self, project_dir: str):
        """Set the project directory and refresh the list."""
        self._project_dir = project_dir
        if not project_dir:
            self.setVisible(False)
            return
        self.refresh()
        self.setVisible(True)

    def refresh(self):
        """Refresh the subproject buttons from disk."""
        if not self._project_dir:
            return

        # Clear existing buttons
        self._buttons.clear()
        while self._btn_layout.count():
            item = self._btn_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        subprojects = list_subprojects(self._project_dir)

        if subprojects:
            active = get_active_subproject(self._project_dir)
            if active not in subprojects:
                active = subprojects[0]
                set_active_subproject(self._project_dir, active)
            self._active_subproject = active

            for name in subprojects:
                btn = self._make_sp_button(name, name == active)
                self._buttons[name] = btn
                self._btn_layout.addWidget(btn)
        else:
            self._active_subproject = None

        # Always show the + button at bottom
        self._btn_layout.addWidget(self._make_add_button())

    def _on_sp_clicked(self, name: str):
        """Handle clicking a subproject button."""
        if name == self._active_subproject:
            return

        self._active_subproject = name
        set_active_subproject(self._project_dir, name)

        # Update button styles
        for btn_name, btn in self._buttons.items():
            self._style_sp_button(btn, btn_name == name)

        self.subproject_changed.emit(name)

    def _show_context_menu(self, name: str, pos):
        """Show right-click context menu for a subproject button."""
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #4d4d4d;
                padding: {scaled(4)}px;
            }}
            QMenu::item {{
                padding: {scaled(6)}px {scaled(20)}px;
            }}
            QMenu::item:selected {{
                background-color: #3d3d3d;
            }}
        """)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")

        action = menu.exec(self._buttons[name].mapToGlobal(pos))
        if action == rename_action:
            self._on_rename(name)
        elif action == delete_action:
            self._on_delete(name)

    def _on_add(self):
        """Add a new subproject."""
        if not self._project_dir:
            return

        # If first subproject, prompt to name the existing data first
        if not has_subprojects(self._project_dir):
            existing_name, ok = QInputDialog.getText(
                self, "Name Existing Data",
                "Your current training data will become the first subproject.\n"
                "Enter a name for it (or leave as default):",
                text=DEFAULT_SUBPROJECT
            )
            if not ok:
                return
            existing_name = existing_name.strip() if existing_name.strip() else DEFAULT_SUBPROJECT
            existing_safe = "".join(
                c if c.isalnum() or c in ('_', '-') else '_' for c in existing_name
            )
            if not existing_safe:
                existing_safe = DEFAULT_SUBPROJECT

            migrate_to_subprojects(self._project_dir, existing_safe)

        # Now prompt for the new subproject name
        name, ok = QInputDialog.getText(
            self, "New Subproject",
            "Enter a name for the new segmentation target\n"
            "(e.g., muscles, nuclei, mitochondria):"
        )
        if not ok or not name.strip():
            # Still refresh to show the migrated one
            self.refresh()
            if self._active_subproject:
                self.subproject_changed.emit(self._active_subproject)
            return

        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name.strip())
        if not safe_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid name.")
            return

        try:
            create_subproject(self._project_dir, safe_name)
            set_active_subproject(self._project_dir, safe_name)
            self.refresh()
            self.subproject_changed.emit(safe_name)
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_rename(self, name: str):
        """Rename a subproject."""
        if not self._project_dir:
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename Subproject",
            f"Rename '{name}' to:",
            text=name
        )
        if not ok or not new_name.strip() or new_name.strip() == name:
            return

        safe_name = "".join(
            c if c.isalnum() or c in ('_', '-') else '_' for c in new_name.strip()
        )

        old_dir = get_subproject_dir(self._project_dir, name)
        new_dir = get_subproject_dir(self._project_dir, safe_name)

        if new_dir.exists():
            QMessageBox.warning(self, "Error", f"'{safe_name}' already exists.")
            return

        try:
            old_dir.rename(new_dir)
            if self._active_subproject == name:
                set_active_subproject(self._project_dir, safe_name)
            self.refresh()
            if self._active_subproject == safe_name:
                self.subproject_changed.emit(safe_name)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to rename: {e}")

    def _on_delete(self, name: str):
        """Delete a subproject."""
        if not self._project_dir:
            return

        subprojects = list_subprojects(self._project_dir)
        if len(subprojects) <= 1:
            QMessageBox.warning(self, "Error", "Cannot delete the only subproject.")
            return

        reply = QMessageBox.question(
            self, "Delete Subproject",
            f"Delete '{name}' and ALL its training data?\n\n"
            "This includes masks, training crops, and checkpoints.\n"
            "Raw images will NOT be affected.\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        sp_dir = get_subproject_dir(self._project_dir, name)
        try:
            shutil.rmtree(sp_dir)
            remaining = list_subprojects(self._project_dir)
            if remaining:
                new_active = remaining[0]
                set_active_subproject(self._project_dir, new_active)
            self.refresh()
            if remaining:
                self.subproject_changed.emit(remaining[0])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete: {e}")
