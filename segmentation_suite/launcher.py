#!/usr/bin/env python3
"""
Launcher Window for MOSS - Segmentation Suite
Single-window application with welcome page and training wizard.
"""

import sys
import os
from pathlib import Path

# Fix Qt plugin path on macOS before importing PyQt6
def _setup_qt_plugin_path():
    """Set up Qt plugin path to fix 'cocoa plugin not found' error on macOS."""
    if sys.platform == 'darwin':  # macOS
        try:
            import PyQt6
            pyqt6_path = Path(PyQt6.__path__[0])
            plugin_path = pyqt6_path / 'Qt6' / 'plugins'
            if plugin_path.exists():
                os.environ['QT_PLUGIN_PATH'] = str(plugin_path)
        except Exception:
            pass  # Silently continue if this fails

_setup_qt_plugin_path()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QStackedWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon, QPixmap

from .dpi_scaling import scaled, scaled_font, scaled_window_size, center_on_screen

# Get resources directory
RESOURCES_DIR = Path(__file__).parent / "resources"


class WelcomePage(QWidget):
    """Welcome/intro page with 'Let's get started' button."""

    def __init__(self, on_start_callback):
        super().__init__()
        self.on_start_callback = on_start_callback
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(scaled(30))
        margin = scaled(50)
        main_layout.setContentsMargins(margin, margin, margin, margin)

        # Main title - MOSS
        self.title_label = QLabel("MOSS")
        self.title_label.setFont(scaled_font(48, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: white; background: transparent; border: none;")
        main_layout.addWidget(self.title_label)

        # Acronym definition
        self.acronym_label = QLabel("Microscopy Oriented Segmentation with Supervision")
        self.acronym_label.setFont(scaled_font(14))
        self.acronym_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.acronym_label.setStyleSheet("color: #888888; background: transparent; border: none;")
        main_layout.addWidget(self.acronym_label)

        main_layout.addStretch()

        # Content row: Button on left, Description on right
        content_layout = QHBoxLayout()
        content_layout.setSpacing(scaled(60))

        # Left side - Big "Let's get started" button
        left_container = QFrame()
        left_container.setMinimumSize(scaled(320), scaled(350))
        self.left_container = left_container  # Store reference for resizing
        radius = scaled(15)
        left_container.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: {radius}px;
            }}
            QFrame:hover {{
                background-color: #3d3d3d;
                border: 2px solid #5d5d5d;
            }}
        """)
        left_container.setCursor(Qt.CursorShape.PointingHandCursor)

        left_layout = QVBoxLayout(left_container)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.setSpacing(scaled(20))

        # Icon (50% larger than before: 120 -> 180)
        self.training_icon_path = RESOURCES_DIR / "training_icon.png"
        self.icon_label = QLabel()
        self.icon_label.setStyleSheet("background: transparent; border: none;")
        if self.training_icon_path.exists():
            pixmap = QPixmap(str(self.training_icon_path))
            icon_size = scaled(180)
            pixmap = pixmap.scaled(
                icon_size, icon_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.icon_label.setPixmap(pixmap)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.icon_label)

        # Button text
        self.btn_label = QLabel("Let's get started")
        self.btn_label.setFont(scaled_font(18, QFont.Weight.Bold))
        self.btn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_label.setStyleSheet("color: white; background: transparent; border: none;")
        left_layout.addWidget(self.btn_label)

        # Make the frame clickable
        left_container.mousePressEvent = lambda e: self.on_start_callback()

        content_layout.addWidget(left_container)

        # Right side - Description
        right_container = QWidget()
        right_container.setMinimumWidth(scaled(350))
        self.right_container = right_container  # Store reference for resizing
        right_layout = QVBoxLayout(right_container)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.setSpacing(scaled(15))

        # Description title
        self.desc_title = QLabel("Interactive Segmentation Training")
        self.desc_title.setFont(scaled_font(16, QFont.Weight.Bold))
        self.desc_title.setStyleSheet("color: white; background: transparent; border: none;")
        self.desc_title.setWordWrap(True)
        right_layout.addWidget(self.desc_title)

        # Description text
        self.desc_text = QLabel(
            "Train deep learning models for image segmentation with an "
            "interactive painting interface. Paint annotations, train in "
            "real-time, and watch predictions improve as you work.\n\n"
            "Features:\n"
            "  - Paint-based annotation tools\n"
            "  - Real-time model training\n"
            "  - Live prediction overlay\n"
            "  - Multi-user collaborative training\n"
            "  - Support for 2D and 2.5D models"
        )
        self.desc_text.setFont(scaled_font(11))
        self.desc_text.setStyleSheet("color: #aaaaaa; background: transparent; border: none;")
        self.desc_text.setWordWrap(True)
        self.desc_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right_layout.addWidget(self.desc_text)

        content_layout.addWidget(right_container)

        main_layout.addLayout(content_layout)
        main_layout.addStretch()

        # Footer
        self.footer_label = QLabel("v1.0.0")
        self.footer_label.setFont(scaled_font(9))
        self.footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.footer_label.setStyleSheet("color: #555555; background: transparent; border: none;")
        main_layout.addWidget(self.footer_label)

        # Store base sizes for scaling
        self._base_width = scaled(900)
        self._base_height = scaled(650)

    def resizeEvent(self, event):
        """Scale UI elements when window is resized."""
        super().resizeEvent(event)

        # Calculate scale factor based on window size
        width = self.width()
        height = self.height()
        scale_w = width / self._base_width
        scale_h = height / self._base_height
        scale = min(scale_w, scale_h)  # Use smaller to maintain proportions
        scale = max(0.8, min(scale, 2.0))  # Clamp between 0.8x and 2x

        # Scale fonts
        self.title_label.setFont(scaled_font(int(48 * scale), QFont.Weight.Bold))
        self.acronym_label.setFont(scaled_font(int(14 * scale)))
        self.btn_label.setFont(scaled_font(int(18 * scale), QFont.Weight.Bold))
        self.desc_title.setFont(scaled_font(int(16 * scale), QFont.Weight.Bold))
        self.desc_text.setFont(scaled_font(int(11 * scale)))
        self.footer_label.setFont(scaled_font(int(9 * scale)))

        # Scale icon
        if self.training_icon_path.exists():
            icon_size = int(scaled(180) * scale)
            pixmap = QPixmap(str(self.training_icon_path))
            pixmap = pixmap.scaled(
                icon_size, icon_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.icon_label.setPixmap(pixmap)


class MainWindow(QMainWindow):
    """Main application window with stacked pages."""

    def __init__(self):
        super().__init__()
        self.training_wizard = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MOSS - Segmentation Suite")

        # Use scaled window size
        win_w, win_h = scaled_window_size(900, 650)
        self.setMinimumSize(win_w, win_h)
        self.resize(win_w, win_h)

        # Center on screen
        center_on_screen(self)

        # Set application icon
        app_icon_path = RESOURCES_DIR / "app_icon.png"
        if app_icon_path.exists():
            self.setWindowIcon(QIcon(str(app_icon_path)))

        # Stacked widget for page navigation
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Page 0: Welcome page
        self.welcome_page = WelcomePage(self.start_training)
        self.stack.addWidget(self.welcome_page)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
            }
        """)

    def start_training(self):
        """Transition to the training wizard."""
        try:
            from .training_wizard import TrainingWizard

            # Create training wizard and add to stack
            self.training_wizard = TrainingWizard()

            # Connect wizard_closed signal to return to welcome page
            self.training_wizard.wizard_closed.connect(self.show_welcome)

            self.stack.addWidget(self.training_wizard)
            self.stack.setCurrentWidget(self.training_wizard)

            # Resize window to fit training wizard
            win_w, win_h = scaled_window_size(1400, 900)
            self.resize(win_w, win_h)
            center_on_screen(self)

        except ImportError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Error",
                f"Failed to load training wizard:\n{e}"
            )

    def show_welcome(self):
        """Return to welcome page."""
        self.stack.setCurrentWidget(self.welcome_page)
        win_w, win_h = scaled_window_size(900, 650)
        self.resize(win_w, win_h)
        center_on_screen(self)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set application icon
    app_icon_path = RESOURCES_DIR / "app_icon.png"
    if app_icon_path.exists():
        app.setWindowIcon(QIcon(str(app_icon_path)))

    # Set dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
