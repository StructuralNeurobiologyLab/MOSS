#!/usr/bin/env python3
"""
Loading dialog with animated tardigrade and progress bar.
"""

from pathlib import Path
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QMovie

from .dpi_scaling import scaled, scaled_font

# Resources directory
RESOURCES_DIR = Path(__file__).parent / "resources"


class LoadingDialog(QDialog):
    """Dialog showing animated tardigrade and loading progress - floating style."""

    def __init__(self, parent=None, message="Loading"):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Scaled fixed size
        dialog_w = scaled(200)
        dialog_h = scaled(100)
        self.setFixedSize(dialog_w, dialog_h)

        # Center on parent
        if parent:
            parent_geo = parent.geometry()
            x = parent_geo.x() + (parent_geo.width() - dialog_w) // 2
            y = parent_geo.y() + (parent_geo.height() - dialog_h) // 2
            self.move(x, y)

        # Layout with scaled margins
        layout = QVBoxLayout()
        margin = scaled(10)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(scaled(6))

        # Animated tardigrade GIF - scaled
        self.movie_label = QLabel()
        self.movie_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        gif_path = RESOURCES_DIR / "loading.gif"
        if gif_path.exists():
            self.movie = QMovie(str(gif_path))
            self.movie.setScaledSize(QSize(scaled(50), scaled(22)))
            self.movie_label.setMovie(self.movie)
            self.movie.start()
        else:
            self.movie_label.setText("🦠")
            font_size = scaled(24)
            self.movie_label.setStyleSheet(f"font-size: {font_size}px; color: white;")
            self.movie = None

        layout.addWidget(self.movie_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Message label with dots animation
        self.base_message = message
        self.dots = 0
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setFont(scaled_font(12))
        self.message_label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self.message_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(scaled(16))
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        # Transparent background, styled progress bar only
        radius = scaled(4)
        self.setStyleSheet(f"""
            QDialog {{
                background: transparent;
            }}
            QLabel {{
                color: white;
                background: transparent;
            }}
            QProgressBar {{
                border: 1px solid #555;
                border-radius: {radius}px;
                text-align: center;
                background-color: rgba(30, 30, 30, 180);
                color: white;
            }}
            QProgressBar::chunk {{
                background-color: #2196F3;
                border-radius: {radius - 1}px;
            }}
        """)

        # Timer for dots animation
        self.dots_timer = QTimer()
        self.dots_timer.timeout.connect(self._update_dots)
        self.dots_timer.start(400)

    def _update_dots(self):
        """Animate the dots (. .. ...)."""
        self.dots = (self.dots + 1) % 4
        dots_str = "." * self.dots if self.dots > 0 else ""
        self.message_label.setText(f"{self.base_message}{dots_str}")

    def set_message(self, message):
        """Update the loading message."""
        self.base_message = message
        self.dots = 0
        self._process_events()

    def set_progress(self, value, maximum=100):
        """Update progress bar."""
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)
        self._process_events()

    def _process_events(self):
        """Force event processing to keep animation running."""
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    def showEvent(self, event):
        """Ensure movie starts when dialog is shown."""
        super().showEvent(event)
        if self.movie:
            self.movie.start()
        self._process_events()

    def closeEvent(self, event):
        """Stop the timer and movie when closing."""
        if hasattr(self, 'dots_timer'):
            self.dots_timer.stop()
        if self.movie:
            self.movie.stop()
        super().closeEvent(event)
