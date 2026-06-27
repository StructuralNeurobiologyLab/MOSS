#!/usr/bin/env python3
"""
Home/Dashboard page for MOSS - shows project overview and pipeline status.
Consolidates project loading/creation with dashboard functionality.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QGroupBox, QScrollArea, QProgressBar,
    QLineEdit, QFileDialog, QMessageBox, QStackedWidget, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QThread, QTimer
from PyQt6.QtGui import QFont

from ..dpi_scaling import scaled
from ..project_config import load_project_config, save_project_config, project_exists
from ..widgets.loading_overlay import LoadingOverlay


class ZarrConversionWorker(QThread):
    """Background worker for TIFF to Zarr conversion with pyramid generation."""
    progress = pyqtSignal(int, int, str)  # completed, total, message
    finished = pyqtSignal(bool, str)  # success, message/path
    log = pyqtSignal(str)

    def __init__(self, source_dir: str, dest_path: str, num_levels: int = 4, parent=None):
        super().__init__(parent)
        self.source_dir = source_dir
        self.dest_path = dest_path
        self.num_levels = num_levels
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from segmentation_suite.em_pipeline.data.pyramid import generate_ome_zarr_pyramid
            from segmentation_suite.em_pipeline.data.convert import tiff_to_zarr
            import os
            import psutil
            from pathlib import Path

            source_path = Path(self.source_dir)
            dest_path = Path(self.dest_path)

            # Get list of TIFF files
            tiff_files = sorted(
                list(source_path.glob("*.tif")) + list(source_path.glob("*.tiff"))
            )

            if not tiff_files:
                self.finished.emit(False, "No TIFF files found")
                return

            self.log.emit(f"Converting {len(tiff_files)} TIFF files to Zarr with {self.num_levels}-level pyramid...")
            self.log.emit(f"Using optimized chunk-aligned writes for maximum speed!")

            def progress_callback(completed, total, msg):
                if not self._stop:
                    self.progress.emit(completed, total, msg)

            # Check available memory and image size to adjust workers
            available_gb = psutil.virtual_memory().available / (1024**3)

            # Estimate per-slice UNCOMPRESSED memory usage from first TIFF
            # Note: TIFF files are often compressed, so file size != memory size
            import tifffile
            with tifffile.TiffFile(str(tiff_files[0])) as tif:
                page = tif.pages[0]
                height, width = page.shape[0], page.shape[1]
                bytes_per_pixel = page.dtype.itemsize
                # Uncompressed size in memory
                slice_mb = (height * width * bytes_per_pixel) / (1024**2)
                # Account for processing overhead: read buffer + write buffer + zarr cache
                # Typically 3-4x the raw data size during conversion
                effective_slice_mb = slice_mb * 4

            self.log.emit(f"Image: {height}x{width}, {slice_mb:.1f} MB/slice (uncompressed)")
            self.log.emit(f"Available RAM: {available_gb:.1f} GB")

            # Adaptive worker settings based on available memory
            # Use more workers if we have plenty of RAM
            if available_gb > 60:
                # Plenty of RAM - use 4 workers for significant speedup
                num_convert_workers = 4
                num_pyramid_workers = 4
            elif available_gb > 30:
                # Moderate RAM - use 2 workers
                num_convert_workers = 2
                num_pyramid_workers = 2
            else:
                # Low RAM - use sequential processing (safest)
                num_convert_workers = 1
                num_pyramid_workers = 1

            # Calculate safe chunk_z: target using at most 40% of available memory
            # (can be more aggressive with parallel workers since they share data)
            max_memory_mb = available_gb * 1024 * 0.4
            safe_chunk_z = max(1, int(max_memory_mb / effective_slice_mb))
            # Clamp to reasonable range
            chunk_z = min(64, max(1, safe_chunk_z))

            # For very large images, be conservative but not too much
            # With plenty of RAM (60+ GB), we can handle larger chunks
            if slice_mb > 100:
                chunk_z = min(chunk_z, 8 if available_gb > 60 else 4)
                self.log.emit(f"Very large images ({slice_mb:.0f} MB) - using chunk_z={chunk_z}")
            elif slice_mb > 50:
                chunk_z = min(chunk_z, 16 if available_gb > 60 else 8)
            elif slice_mb > 25:
                chunk_z = min(chunk_z, 32 if available_gb > 60 else 16)
            # Don't cap for smaller images - use calculated safe_chunk_z

            self.log.emit(f"Using {num_convert_workers} workers for conversion, chunk_z={chunk_z}")
            if num_convert_workers > 1:
                self.log.emit(f"Parallel processing enabled - conversion will be {num_convert_workers}x faster!")

            # First convert TIFFs to base Zarr (level 0)
            base_zarr = dest_path / "0"
            self.log.emit("Step 1/2: Converting TIFFs to base Zarr volume...")

            tiff_to_zarr(
                str(source_path),
                str(base_zarr),
                chunk_size=(chunk_z, 256, 256),
                progress_callback=lambda c, t, m: progress_callback(c, t * 2, f"Converting: {m}"),
                num_workers=num_convert_workers
            )

            if self._stop:
                self.finished.emit(False, "Cancelled")
                return

            # Generate pyramid levels
            self.log.emit(f"Step 2/2: Generating {self.num_levels - 1} pyramid levels...")

            from segmentation_suite.em_pipeline.data.pyramid import generate_pyramid
            generate_pyramid(
                str(base_zarr),
                str(dest_path),
                num_levels=self.num_levels,
                factors=(2, 2, 2),
                method='mean',
                progress_callback=lambda c, t, m: progress_callback(c + t, t * 2, f"Pyramid: {m}"),
                num_workers=num_pyramid_workers
            )

            # Create OME-Zarr metadata
            import json
            multiscales = [{
                'version': '0.4',
                'name': dest_path.stem,
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
                ],
                'datasets': [
                    {'path': str(level), 'coordinateTransformations': [{'type': 'scale', 'scale': [1.0, 2**level, 2**level, 2**level]}]}
                    for level in range(self.num_levels)
                ],
                'type': 'mean',
            }]

            with open(dest_path / '.zattrs', 'w') as f:
                json.dump({'multiscales': multiscales}, f, indent=2)
            with open(dest_path / '.zgroup', 'w') as f:
                json.dump({'zarr_format': 2}, f)

            self.finished.emit(True, str(dest_path))

        except Exception as e:
            import traceback
            self.log.emit(f"Error: {traceback.format_exc()}")
            self.finished.emit(False, str(e))


class DataStatusCard(QFrame):
    """Card showing status of a data type with optional action button and progress bar."""

    # Signal emitted when action button is clicked
    action_clicked = pyqtSignal()
    secondary_action_clicked = pyqtSignal()

    def __init__(self, title: str, action_text: str = None, secondary_action_text: str = None, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            DataStatusCard {
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(scaled(6))

        # Title row
        title_row = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #ffffff;")
        title_row.addWidget(self.title_label)
        title_row.addStretch()

        # Secondary action button (optional) - shown first (left)
        self.secondary_btn = None
        if secondary_action_text:
            self.secondary_btn = QPushButton(secondary_action_text)
            self.secondary_btn.setStyleSheet("""
                QPushButton {
                    background-color: #424242;
                    color: white;
                    border: none;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-size: 10px;
                }
                QPushButton:hover { background-color: #616161; }
            """)
            self.secondary_btn.clicked.connect(self.secondary_action_clicked.emit)
            self.secondary_btn.setVisible(False)  # Hidden by default
            title_row.addWidget(self.secondary_btn)

        # Action button (optional) - primary action (right)
        self.action_btn = None
        if action_text:
            self.action_btn = QPushButton(action_text)
            self.action_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1565C0;
                    color: white;
                    border: none;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-size: 10px;
                }
                QPushButton:hover { background-color: #1976D2; }
            """)
            self.action_btn.clicked.connect(self.action_clicked.emit)
            self.action_btn.setVisible(False)  # Hidden by default
            title_row.addWidget(self.action_btn)

        layout.addLayout(title_row)

        # Status indicator
        self.status_label = QLabel("Not found")
        self.status_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 3px;
                text-align: center;
                background-color: #1d1d1d;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Details
        self.details_label = QLabel("")
        self.details_label.setStyleSheet("color: #666666; font-size: 10px;")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

    def set_status(self, found: bool, details: str = "", show_action: bool = False, show_secondary: bool = False,
                   action_override: str = None, secondary_override: str = None):
        """Set card status.

        Args:
            found: Whether data is available
            details: Status details text
            show_action: Show primary action button
            show_secondary: Show secondary action button
            action_override: Override primary button text (e.g., "Re-convert" instead of "Convert")
            secondary_override: Override secondary button text
        """
        if found:
            self.status_label.setText("Available")
            self.status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_label.setText("Not found")
            self.status_label.setStyleSheet("color: #f44336;")
        self.details_label.setText(details)
        self.progress_bar.setVisible(False)

        # Show action button - can be shown even when found (e.g., for re-convert)
        if self.action_btn:
            self.action_btn.setVisible(show_action)
            if action_override:
                self.action_btn.setText(action_override)
        # Show secondary button (e.g., "Load" option alongside "Convert")
        if self.secondary_btn:
            self.secondary_btn.setVisible(show_secondary)
            if secondary_override:
                self.secondary_btn.setText(secondary_override)

    def set_progress(self, percent: int, status_text: str = ""):
        """Show progress bar with given percentage."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(percent)
        if status_text:
            self.status_label.setText(status_text)
            self.status_label.setStyleSheet("color: #2196F3;")
        if self.action_btn:
            self.action_btn.setVisible(False)


class PipelineStepWidget(QFrame):
    """Widget showing a pipeline step with status - clickable row."""

    clicked = pyqtSignal(str)  # Emits step name when clicked

    def __init__(self, step_name: str, step_number: int, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            PipelineStepWidget {
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px;
            }
            PipelineStepWidget:hover {
                background-color: #4d4d4d;
                border: 1px solid #2196F3;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(scaled(10), scaled(8), scaled(10), scaled(8))

        # Step number circle
        self.number_label = QLabel(str(step_number))
        self.number_label.setFixedSize(scaled(28), scaled(28))
        self.number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.number_label.setStyleSheet("""
            background-color: #555555;
            border-radius: 14px;
            color: white;
            font-weight: bold;
        """)
        layout.addWidget(self.number_label)

        # Step name
        self.name_label = QLabel(step_name)
        self.name_label.setFont(QFont("Arial", 10))
        self.name_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.name_label)

        layout.addStretch()

        # Status
        self.status_label = QLabel("Pending")
        self.status_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_label)

        # Go arrow indicator
        self.go_label = QLabel("  →")
        self.go_label.setStyleSheet("color: #555555; font-size: 14px;")
        layout.addWidget(self.go_label)

    def mousePressEvent(self, event):
        self.clicked.emit(self.step_name)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        """Highlight arrow on hover."""
        self.go_label.setStyleSheet("color: #2196F3; font-size: 14px;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Reset arrow on leave."""
        self.go_label.setStyleSheet("color: #555555; font-size: 14px;")
        super().leaveEvent(event)

    def set_status(self, status: str):
        """Set step status: 'pending', 'in_progress', 'completed', 'available'."""
        self.status_label.setText(status.replace("_", " ").title())

        colors = {
            'pending': '#888888',
            'in_progress': '#2196F3',
            'completed': '#4CAF50',
            'available': '#FF9800',
            'error': '#f44336'
        }
        color = colors.get(status, '#888888')
        self.status_label.setStyleSheet(f"color: {color};")

        if status == 'completed':
            self.number_label.setStyleSheet(f"""
                background-color: #4CAF50;
                border-radius: 14px;
                color: white;
                font-weight: bold;
            """)
        elif status == 'in_progress':
            self.number_label.setStyleSheet(f"""
                background-color: #2196F3;
                border-radius: 14px;
                color: white;
                font-weight: bold;
            """)


class HomePage(QWidget):
    """Home/Dashboard page showing project overview and pipeline status."""

    # Signals
    start_training = pyqtSignal()  # Start 2D MOSS training
    start_3d_segmentation = pyqtSignal()  # Start 3D segmentation pipeline
    open_data_manager = pyqtSignal()  # Open data management
    project_loaded = pyqtSignal()  # Emitted when a project is loaded/created

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_dir = None
        self._config = {}
        self._loading_overlay = None
        self.init_ui()

        # Create loading overlay (parented to self)
        self._loading_overlay = LoadingOverlay(self, "Loading project...")
        self._loading_overlay.hide()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget to switch between "no project" and "dashboard" views
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # View 0: No project loaded - show project selection
        self.no_project_view = self._create_no_project_view()
        self.stack.addWidget(self.no_project_view)

        # View 1: Project loaded - show dashboard
        self.dashboard_view = self._create_dashboard_view()
        self.stack.addWidget(self.dashboard_view)

        # Start with no project view
        self.stack.setCurrentIndex(0)

    def resizeEvent(self, event):
        """Resize loading overlay to fill widget."""
        super().resizeEvent(event)
        if self._loading_overlay:
            self._loading_overlay.setGeometry(self.rect())

    def _create_no_project_view(self) -> QWidget:
        """Create the view shown when no project is loaded."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(scaled(30))
        layout.setContentsMargins(scaled(40), scaled(40), scaled(40), scaled(40))

        # Header
        header = QLabel("MOSS - EM Segmentation Pipeline")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #ffffff;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel("Microscopy Oriented Segmentation with Supervision")
        subtitle.setStyleSheet("color: #888888; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addStretch()

        # Project selection buttons
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setSpacing(scaled(20))

        # Load existing project button
        self.load_project_btn = QPushButton("Load Existing Project")
        self.load_project_btn.setMinimumSize(scaled(200), scaled(80))
        self.load_project_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
        """)
        self.load_project_btn.clicked.connect(self._on_load_project)
        btn_layout.addWidget(self.load_project_btn)

        # Create new project button
        self.new_project_btn = QPushButton("Create New Project")
        self.new_project_btn.setMinimumSize(scaled(200), scaled(80))
        self.new_project_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:pressed { background-color: #1B5E20; }
        """)
        self.new_project_btn.clicked.connect(self._on_new_project)
        btn_layout.addWidget(self.new_project_btn)

        layout.addWidget(btn_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Resume last project (if available)
        settings = QSettings("MOSS", "SegmentationSuite")
        last_project = settings.value("last_project_dir", "")
        if last_project and os.path.isdir(last_project):
            resume_container = QWidget()
            resume_layout = QVBoxLayout(resume_container)
            resume_layout.setSpacing(scaled(8))

            separator = QLabel("— or —")
            separator.setStyleSheet("color: #666666;")
            separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
            resume_layout.addWidget(separator)

            self.resume_btn = QPushButton(f"Resume: {os.path.basename(last_project)}")
            self.resume_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    color: #cccccc;
                    border: 1px solid #555555;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 12px;
                }
                QPushButton:hover { background-color: #4d4d4d; }
            """)
            self.resume_btn.clicked.connect(self._on_resume_last)
            resume_layout.addWidget(self.resume_btn, alignment=Qt.AlignmentFlag.AlignCenter)

            layout.addWidget(resume_container)

        layout.addStretch()

        # New project form (hidden by default)
        self.new_project_form = self._create_new_project_form()
        self.new_project_form.setVisible(False)
        layout.addWidget(self.new_project_form)

        layout.addStretch()

        return widget

    def _create_new_project_form(self) -> QWidget:
        """Create the form for new project creation."""
        form = QGroupBox("New Project")
        form.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 8px;
                margin-top: 10px;
                padding: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout(form)
        layout.setSpacing(scaled(15))

        # Project name
        name_row = QHBoxLayout()
        name_label = QLabel("Project Name:")
        name_label.setMinimumWidth(scaled(120))
        name_label.setStyleSheet("color: #cccccc;")
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("Enter project name")
        self.project_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        name_row.addWidget(name_label)
        name_row.addWidget(self.project_name_input)
        layout.addLayout(name_row)

        # Output directory
        dir_row = QHBoxLayout()
        dir_label = QLabel("Location:")
        dir_label.setMinimumWidth(scaled(120))
        dir_label.setStyleSheet("color: #cccccc;")
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Select where to save the project")
        self.output_dir_input.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #666666; }
        """)
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(dir_label)
        dir_row.addWidget(self.output_dir_input)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        # Raw data directory (optional)
        data_row = QHBoxLayout()
        data_label = QLabel("Raw Data:")
        data_label.setMinimumWidth(scaled(120))
        data_label.setStyleSheet("color: #cccccc;")
        self.raw_data_input = QLineEdit()
        self.raw_data_input.setPlaceholderText("(Optional) TIFF folder or Zarr volume")
        self.raw_data_input.setStyleSheet("""
            QLineEdit {
                background-color: #3d3d3d;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        data_browse_btn = QPushButton("Browse...")
        data_browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #666666; }
        """)
        data_browse_btn.clicked.connect(self._browse_raw_data)
        data_row.addWidget(data_label)
        data_row.addWidget(self.raw_data_input)
        data_row.addWidget(data_browse_btn)
        layout.addLayout(data_row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #666666; }
        """)
        cancel_btn.clicked.connect(self._cancel_new_project)
        btn_row.addWidget(cancel_btn)

        create_btn = QPushButton("Create Project")
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #388E3C; }
        """)
        create_btn.clicked.connect(self._create_project)
        btn_row.addWidget(create_btn)

        layout.addLayout(btn_row)

        return form

    def _create_dashboard_view(self) -> QWidget:
        """Create the dashboard view shown when a project is loaded."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(scaled(20))
        layout.setContentsMargins(scaled(20), scaled(20), scaled(20), scaled(20))

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(scaled(20))

        # Header with project info
        header_row = QHBoxLayout()

        header = QLabel("MOSS - EM Segmentation Pipeline")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #ffffff;")
        header_row.addWidget(header)

        header_row.addStretch()

        # Change project button
        change_project_btn = QPushButton("Change Project")
        change_project_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #666666; }
        """)
        change_project_btn.clicked.connect(self._on_change_project)
        header_row.addWidget(change_project_btn)

        content_layout.addLayout(header_row)

        # Project info section
        self.project_group = QGroupBox("Current Project")
        self.project_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        project_layout = QVBoxLayout(self.project_group)

        self.project_path_label = QLabel("No project loaded")
        self.project_path_label.setStyleSheet("color: #888888;")
        self.project_path_label.setWordWrap(True)
        project_layout.addWidget(self.project_path_label)

        content_layout.addWidget(self.project_group)

        # Data status section
        data_group = QGroupBox("Data Status")
        data_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        data_layout = QGridLayout(data_group)
        data_layout.setSpacing(scaled(10))

        # Data status cards with action buttons
        self.raw_data_card = DataStatusCard("Raw EM Data", "Import")
        self.training_data_card = DataStatusCard("Training Labels", "Create")
        self.model_card = DataStatusCard("Trained Model", "Train")
        self.zarr_card = DataStatusCard("Zarr Volume", "Convert", secondary_action_text="Load")

        # Connect card actions
        self.raw_data_card.action_clicked.connect(self._on_import_data)
        self.training_data_card.action_clicked.connect(self._on_start_training)
        self.model_card.action_clicked.connect(self._on_start_training)
        self.zarr_card.action_clicked.connect(self._on_convert_to_zarr)
        self.zarr_card.secondary_action_clicked.connect(self._on_load_existing_zarr)

        self.segmentation_card = DataStatusCard("Segmentation", "Run")

        self.segmentation_card.action_clicked.connect(lambda: self.start_3d_segmentation.emit())

        data_layout.addWidget(self.raw_data_card, 0, 0)
        data_layout.addWidget(self.training_data_card, 0, 1)
        data_layout.addWidget(self.model_card, 0, 2)
        data_layout.addWidget(self.zarr_card, 1, 0)
        data_layout.addWidget(self.segmentation_card, 1, 1)

        content_layout.addWidget(data_group)

        # Pipeline section
        pipeline_group = QGroupBox("Pipeline Steps")
        pipeline_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        pipeline_layout = QVBoxLayout(pipeline_group)
        pipeline_layout.setSpacing(scaled(6))

        # Pipeline steps (matches actual wizard workflow)
        steps = [
            ("1. Data Import", "Import raw EM images (TIFF/Zarr)"),
            ("2. Ground Truth", "Create training labels & train segmentation model"),
            ("3. Segmentation", "Run MOSS or LSD 2D segmentation"),
            ("4. Export", "Export final segmentation"),
        ]

        self.pipeline_steps = []
        for i, (name, tooltip) in enumerate(steps, 1):
            step = PipelineStepWidget(name, i)
            step.setToolTip(tooltip)
            step.clicked.connect(self._on_step_clicked)
            pipeline_layout.addWidget(step)
            self.pipeline_steps.append(step)

        content_layout.addWidget(pipeline_group)

        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

        return widget

    # === Project Loading/Creation Methods ===

    def _on_load_project(self):
        """Handle Load Existing Project button click."""
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Existing Project Directory"
        )
        if not project_dir:
            return

        # Check if this looks like a valid project
        if not project_exists(project_dir):
            QMessageBox.warning(
                self, "Invalid Project",
                "This doesn't appear to be a valid project directory.\n"
                "A valid project should have TIFF/Zarr data or training masks."
            )
            return

        # Show loading overlay
        self._loading_overlay.set_message("Loading project...")
        self._loading_overlay.start()

        # Load project after short delay to show overlay
        self._pending_project_dir = project_dir
        QTimer.singleShot(50, self._load_pending_project)

    def _on_resume_last(self):
        """Handle Resume Last Project button click."""
        settings = QSettings("MOSS", "SegmentationSuite")
        project_dir = settings.value("last_project_dir", "")

        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "No Recent Project", "No recent project found.")
            return

        if not project_exists(project_dir):
            QMessageBox.warning(
                self, "Invalid Project",
                f"The last project no longer exists or is invalid:\n{project_dir}"
            )
            settings.remove("last_project_dir")
            return

        # Show loading overlay
        self._loading_overlay.set_message("Loading project...")
        self._loading_overlay.start()

        # Load project after short delay to show overlay
        self._pending_project_dir = project_dir
        QTimer.singleShot(50, self._load_pending_project)

    def _on_new_project(self):
        """Handle Create New Project button click."""
        # Show the new project form, hide buttons
        self.load_project_btn.setVisible(False)
        self.new_project_btn.setVisible(False)
        if hasattr(self, 'resume_btn'):
            self.resume_btn.parent().setVisible(False)
        self.new_project_form.setVisible(True)

        # Set default output directory
        default_dir = os.path.expanduser("~/segmentation_projects")
        self.output_dir_input.setText(default_dir)

    def _cancel_new_project(self):
        """Cancel new project creation."""
        self.new_project_form.setVisible(False)
        self.load_project_btn.setVisible(True)
        self.new_project_btn.setVisible(True)
        if hasattr(self, 'resume_btn'):
            self.resume_btn.parent().setVisible(True)

    def _browse_output_dir(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir_input.setText(folder)

    def _browse_raw_data(self):
        """Browse for raw data directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select TIFF Folder or Zarr Volume")
        if folder:
            self.raw_data_input.setText(folder)

    def _create_project(self):
        """Create a new project."""
        project_name = self.project_name_input.text().strip()
        output_dir = self.output_dir_input.text().strip()
        raw_data = self.raw_data_input.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a project name.")
            return

        if not output_dir:
            QMessageBox.warning(self, "Missing Location", "Please select an output directory.")
            return

        # Create project directory
        project_dir = os.path.join(output_dir, project_name)
        if os.path.exists(project_dir):
            reply = QMessageBox.question(
                self, "Project Exists",
                f"A folder named '{project_name}' already exists.\nDo you want to use it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        else:
            os.makedirs(project_dir, exist_ok=True)

        # Create train_images directory (for annotation crops, not raw data)
        train_images_dir = Path(project_dir) / "train_images"
        train_images_dir.mkdir(exist_ok=True)

        # Save basic project config
        config = {
            "project_name": project_name,
            "raw_images_dir": raw_data if (raw_data and os.path.isdir(raw_data)) else None,
        }
        save_project_config(project_dir, config)

        self._load_project(project_dir)

    def _load_pending_project(self):
        """Load the pending project (called after loading overlay is shown)."""
        if hasattr(self, '_pending_project_dir') and self._pending_project_dir:
            self._load_project(self._pending_project_dir)
            self._pending_project_dir = None

    def _load_project(self, project_dir: str):
        """Load a project and switch to dashboard view."""
        self.project_dir = Path(project_dir)

        # Save as last project
        settings = QSettings("MOSS", "SegmentationSuite")
        settings.setValue("last_project_dir", project_dir)

        # Load project config if available
        config = load_project_config(project_dir) or {}
        self._config = config
        self._config['project_dir'] = project_dir
        self._config['project_name'] = config.get('project_name', self.project_dir.name)

        # Check for large TIFF files that need Zarr conversion
        self._check_large_tiffs()

        # Update dashboard and switch view
        self._scan_project()
        self.stack.setCurrentIndex(1)  # Show dashboard

        # Hide loading overlay
        if self._loading_overlay:
            self._loading_overlay.stop()

        print(f"[HomePage] Loaded project: {project_dir}")
        self.project_loaded.emit()

    def _check_large_tiffs(self):
        """Check for TIFF files and automatically convert to Zarr (required for pipeline)."""
        if not self.project_dir:
            return

        # Look for TIFF files
        tiff_files = list(self.project_dir.glob("**/*.tif")) + list(self.project_dir.glob("**/*.tiff"))
        if not tiff_files:
            return

        # Check if Zarr already exists (v2 uses .zarray, v3 uses zarr.json)
        zarr_dirs = list(self.project_dir.glob("**/*.zarr"))
        valid_zarr = [z for z in zarr_dirs if (z / '.zarray').exists() or any(z.glob('*/.zarray')) or any(z.glob('*/zarr.json'))]
        if valid_zarr:
            # Zarr exists, no conversion needed
            return

        # TIFF files found but no Zarr - conversion is required
        try:
            sample_files = tiff_files[:3]
            total_size_mb = sum(f.stat().st_size for f in sample_files) / 1e6
            avg_size_mb = total_size_mb / len(sample_files)
            estimated_total_mb = avg_size_mb * len(tiff_files)
            size_str = f"{estimated_total_mb / 1000:.1f} GB" if estimated_total_mb > 1000 else f"{estimated_total_mb:.0f} MB"

            # Inform user that conversion will start
            QMessageBox.information(
                self, "Converting to Zarr",
                f"This project contains {len(tiff_files)} TIFF files (~{size_str}).\n\n"
                f"MOSS requires Zarr format for optimal performance. Converting now...\n\n"
                f"This enables:\n"
                f"• Chunked loading (only load what's needed)\n"
                f"• Multi-resolution mipmaps (fast zooming)\n"
                f"• Efficient 3D segmentation"
            )

            # Start conversion with inline progress
            tiff_source = tiff_files[0].parent
            self._start_zarr_conversion_inline(str(tiff_source))

        except Exception as e:
            print(f"[HomePage] Error checking TIFF sizes: {e}")

    def _start_zarr_conversion_inline(self, tiff_source: str, auto_navigate_after=False):
        """Start Zarr conversion with progress shown in the Zarr card."""
        source_path = Path(tiff_source)
        dest_path = self.project_dir / "raw_data.zarr"

        # Show progress in the Zarr card
        self.zarr_card.set_progress(0, "Converting to Zarr...")

        # Track if we should navigate after conversion
        self._navigate_after_conversion = auto_navigate_after

        # Start worker
        self._conversion_worker = ZarrConversionWorker(str(tiff_source), str(dest_path), num_levels=4)
        self._conversion_worker.progress.connect(self._on_inline_conversion_progress)
        self._conversion_worker.finished.connect(self._on_inline_conversion_finished)
        self._conversion_worker.log.connect(lambda msg: print(f"[ZarrConversion] {msg}"))
        self._conversion_worker.start()

    def _on_inline_conversion_progress(self, completed: int, total: int, msg: str):
        """Handle inline conversion progress update."""
        if total > 0:
            percent = int(100 * completed / total)
            self.zarr_card.set_progress(percent, msg[:40] + "..." if len(msg) > 40 else msg)

    def _on_inline_conversion_finished(self, success: bool, message: str):
        """Handle inline conversion completion."""
        if success:
            # Refresh project view to show new Zarr
            self._scan_project()

            # If this conversion was triggered by starting training, navigate now
            if hasattr(self, '_navigate_after_conversion') and self._navigate_after_conversion:
                self._navigate_after_conversion = False
                self.start_training.emit()
        else:
            self.zarr_card.set_status(False, f"Conversion failed: {message}", show_action=True)
            QMessageBox.warning(
                self, "Conversion Failed",
                f"Failed to convert to Zarr:\n{message}"
            )

    def _start_zarr_conversion(self, tiff_source: str):
        """Start Zarr conversion with pyramid generation (dialog version)."""
        source_path = Path(tiff_source)
        dest_path = self.project_dir / "raw_data.zarr"

        # Create progress dialog
        self._conversion_dialog = QDialog(self)
        self._conversion_dialog.setWindowTitle("Converting to Zarr with Pyramids")
        self._conversion_dialog.setMinimumSize(400, 150)
        self._conversion_dialog.setModal(True)

        layout = QVBoxLayout(self._conversion_dialog)

        self._conversion_status_label = QLabel("Initializing conversion...")
        layout.addWidget(self._conversion_status_label)

        self._conversion_progress = QProgressBar()
        self._conversion_progress.setMinimum(0)
        self._conversion_progress.setMaximum(100)
        layout.addWidget(self._conversion_progress)

        self._conversion_log_label = QLabel("")
        self._conversion_log_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(self._conversion_log_label)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel_conversion)
        layout.addWidget(cancel_btn)

        # Start worker
        self._conversion_worker = ZarrConversionWorker(str(tiff_source), str(dest_path), num_levels=4)
        self._conversion_worker.progress.connect(self._on_conversion_progress)
        self._conversion_worker.finished.connect(self._on_conversion_finished)
        self._conversion_worker.log.connect(self._on_conversion_log)
        self._conversion_worker.start()

        self._conversion_dialog.exec()

    def _on_conversion_progress(self, completed: int, total: int, msg: str):
        """Handle conversion progress update (dialog version)."""
        if total > 0:
            percent = int(100 * completed / total)
            self._conversion_progress.setValue(percent)
        self._conversion_status_label.setText(msg)

    def _on_conversion_log(self, msg: str):
        """Handle conversion log message."""
        self._conversion_log_label.setText(msg)
        print(f"[ZarrConversion] {msg}")

    def _on_conversion_finished(self, success: bool, message: str):
        """Handle conversion completion (dialog version)."""
        self._conversion_dialog.accept()

        if success:
            QMessageBox.information(
                self, "Conversion Complete",
                f"Successfully converted to Zarr with multi-resolution pyramids:\n{message}"
            )
            # Refresh project view
            self._scan_project()
        else:
            QMessageBox.warning(
                self, "Conversion Failed",
                f"Failed to convert to Zarr:\n{message}"
            )

    def _cancel_conversion(self):
        """Cancel ongoing conversion."""
        if hasattr(self, '_conversion_worker') and self._conversion_worker:
            self._conversion_worker.stop()
            self._conversion_worker.wait(2000)
        self._conversion_dialog.reject()

    def _on_change_project(self):
        """Switch back to project selection view."""
        self.stack.setCurrentIndex(0)  # Show project selection

    # === Dashboard Methods ===

    def set_project(self, project_dir: str):
        """Set the current project directory and scan for data."""
        if project_dir:
            self._load_project(project_dir)
        else:
            self.project_dir = None
            self.stack.setCurrentIndex(0)

    def get_config(self) -> dict:
        """Get the current project configuration."""
        if not self.project_dir:
            return {}

        # Build config from project directory
        project_dir = str(self.project_dir)
        config = {
            'project_dir': project_dir,
            'project_name': self._config.get('project_name', self.project_dir.name),
            'raw_images_dir': str(self.project_dir / 'train_images'),
            'train_masks_dir': str(self.project_dir / 'train_masks'),
            'checkpoint_path': str(self.project_dir / 'checkpoint.pth'),
            'interactive': True,
            'num_epochs': 50000,
            'batch_size': 2,
            'learning_rate': 0.0001,
            'tile_size': 512,
        }

        # Merge with loaded config
        config.update(self._config)
        config['project_dir'] = project_dir  # Ensure this is always set

        return config

    def _scan_project(self):
        """Scan project directory for existing data."""
        if not self.project_dir or not self.project_dir.exists():
            self.project_path_label.setText("No project loaded")
            self._reset_status()
            return

        self.project_path_label.setText(str(self.project_dir))
        self.project_path_label.setStyleSheet("color: #4CAF50;")

        # Check for raw data (TIFF files or Zarr)
        # First check the project directory for TIFF files (for old projects)
        tiff_files = list(self.project_dir.glob("**/*.tif")) + list(self.project_dir.glob("**/*.tiff"))

        # Also check the raw_images_dir from config (for new projects with external raw data)
        raw_images_dir = self._config.get('raw_images_dir')
        has_external_raw = False
        external_tiff_count = 0
        external_tiff_dir = None
        if raw_images_dir:
            raw_path = Path(raw_images_dir)
            if not raw_path.is_absolute() and self.project_dir:
                raw_path = self.project_dir / raw_path

            if raw_path.exists():
                # Check if it's a zarr volume (v2 uses .zarray, v3 uses zarr.json)
                if str(raw_path).endswith('.zarr') and (
                    (raw_path / '.zarray').exists() or
                    any(raw_path.glob('*/.zarray')) or
                    any(raw_path.glob('*/zarr.json'))
                ):
                    has_external_raw = True
                    total_size = sum(f.stat().st_size for f in raw_path.rglob('*') if f.is_file())
                    size_str = f"{total_size / 1e9:.1f} GB" if total_size > 1e9 else f"{total_size / 1e6:.1f} MB"
                    self.raw_data_card.set_status(True, f"Zarr volume ({size_str})", show_action=False)
                    self.pipeline_steps[0].set_status('completed')
                # Check if it's a directory with TIFF files
                elif raw_path.is_dir():
                    external_tiffs = list(raw_path.glob("*.tif")) + list(raw_path.glob("*.tiff"))
                    if external_tiffs:
                        has_external_raw = True
                        external_tiff_count = len(external_tiffs)
                        external_tiff_dir = str(raw_path)
                        self.raw_data_card.set_status(True, f"{len(external_tiffs)} TIFF files", show_action=False)
                        self.pipeline_steps[0].set_status('completed')

        # If no external raw data found, check project directory
        if not has_external_raw:
            if tiff_files:
                self.raw_data_card.set_status(True, f"{len(tiff_files)} TIFF files found", show_action=False)
                self.pipeline_steps[0].set_status('completed')
            else:
                self.raw_data_card.set_status(False, "Import TIFF or Zarr data", show_action=True)
                self.pipeline_steps[0].set_status('available')

        # Check for training labels
        labels_dir = self.project_dir / "labels"
        masks_dir = self.project_dir / "masks"
        train_masks_dir = self.project_dir / "train_masks"
        has_labels = (
            (labels_dir.exists() and any(labels_dir.iterdir())) or
            (masks_dir.exists() and any(masks_dir.iterdir())) or
            (train_masks_dir.exists() and any(train_masks_dir.iterdir()))
        )
        if has_labels:
            self.training_data_card.set_status(True, "Training labels available", show_action=False)
        else:
            self.training_data_card.set_status(False, "Create labels in Ground Truth step", show_action=True)

        # Check for trained model
        model_files = list(self.project_dir.glob("**/*.pt")) + list(self.project_dir.glob("**/*.pth"))
        if model_files:
            self.model_card.set_status(True, f"{len(model_files)} model(s) found", show_action=False)
        else:
            self.model_card.set_status(False, "Train a model first", show_action=has_labels)

        # Pipeline step 1 (Ground Truth) = labels + model combined
        if model_files:
            self.pipeline_steps[1].set_status('completed')
        elif has_labels:
            self.pipeline_steps[1].set_status('available')
        else:
            self.pipeline_steps[1].set_status('pending' if not tiff_files else 'available')

        # Check for Zarr data
        zarr_dirs = list(self.project_dir.glob("**/*.zarr"))
        has_zarr = False
        # Check if we have TIFFs available for conversion (either in project dir or external)
        has_tiffs_for_conversion = bool(tiff_files) or bool(external_tiff_count)

        if zarr_dirs:
            valid_zarr = [z for z in zarr_dirs if (z / '.zarray').exists() or any(z.glob('*/.zarray')) or any(z.glob('*/zarr.json'))]
            if valid_zarr:
                has_zarr = True
                total_size = sum(sum(f.stat().st_size for f in z.rglob('*') if f.is_file()) for z in valid_zarr)
                size_str = f"{total_size / 1e9:.1f} GB" if total_size > 1e9 else f"{total_size / 1e6:.1f} MB"

                # Check if pyramids are missing (takes priority over re-convert)
                needs_pyramids = self._check_zarr_missing_pyramids(valid_zarr[0])

                if needs_pyramids:
                    self.zarr_card.set_status(
                        True, f"{len(valid_zarr)} Zarr volume(s), {size_str} (no pyramids)",
                        show_action=True, action_override="Generate Pyramids"
                    )
                elif has_tiffs_for_conversion and total_size < 100 * 1e6:  # Less than 100MB suggests incomplete
                    self.zarr_card.set_status(
                        True, f"{size_str} (may be incomplete)",
                        show_action=True, show_secondary=True,
                        action_override="Re-convert", secondary_override="Load"
                    )
                elif has_tiffs_for_conversion:
                    # Zarr exists and looks complete, but still allow re-convert
                    self.zarr_card.set_status(
                        True, f"{len(valid_zarr)} Zarr volume(s), {size_str}",
                        show_action=True, show_secondary=False,
                        action_override="Re-convert"
                    )
                else:
                    self.zarr_card.set_status(True, f"{len(valid_zarr)} Zarr volume(s), {size_str}", show_action=False)
            else:
                # Show both Convert (if TIFFs exist) and Load buttons
                self.zarr_card.set_status(False, "No valid Zarr volumes", show_action=has_tiffs_for_conversion, show_secondary=True)
        else:
            # No Zarr yet - show Convert button if TIFFs exist (project dir or external)
            if has_tiffs_for_conversion:
                tiff_source = external_tiff_dir if external_tiff_dir else str(self.project_dir)
                self.zarr_card.set_status(False, "Convert TIFF to Zarr or load existing", show_action=True, show_secondary=True)
            else:
                self.zarr_card.set_status(False, "Load existing Zarr or convert TIFFs", show_action=False, show_secondary=True)

        # Check for segmentation results
        seg_patterns = ["**/segmentation*.zarr", "**/seg*.zarr", "**/output*.zarr"]
        seg_files = []
        for pattern in seg_patterns:
            seg_files.extend(self.project_dir.glob(pattern))
        has_segmentation = bool(seg_files)
        if has_segmentation:
            self.segmentation_card.set_status(True, f"{len(seg_files)} segmentation(s)", show_action=False)
            self.pipeline_steps[2].set_status('completed')
            self.pipeline_steps[3].set_status('available')  # Export available
        else:
            # Can run segmentation if we have zarr OR model
            can_segment = has_zarr or bool(model_files)
            self.segmentation_card.set_status(False, "Run segmentation", show_action=can_segment)
            self.pipeline_steps[2].set_status('available' if can_segment else 'pending')
            self.pipeline_steps[3].set_status('pending')  # Export not ready

    def _reset_status(self):
        """Reset all status indicators."""
        self.raw_data_card.set_status(False)
        self.training_data_card.set_status(False)
        self.model_card.set_status(False)
        self.zarr_card.set_status(False)
        self.segmentation_card.set_status(False)

        for step in self.pipeline_steps:
            step.set_status('pending')

    def _on_step_clicked(self, step_name: str):
        """Handle pipeline step click."""
        step_num = int(step_name.split('.')[0])
        if step_num == 1:  # Data Import
            self._on_import_data()
        elif step_num == 2:  # Ground Truth (training labels + model training)
            self._on_start_training()  # Check for Zarr conversion first
        elif step_num == 3:  # Segmentation
            self.start_3d_segmentation.emit()
        elif step_num == 4:  # Export
            pass  # TODO: navigate to export page

    def _on_start_training(self):
        """Handle start training - check if Zarr conversion is needed first."""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "Please load or create a project first.")
            return

        # Check if Zarr already exists (v2 uses .zarray, v3 uses zarr.json)
        zarr_dirs = list(self.project_dir.glob("**/*.zarr"))
        valid_zarr = [z for z in zarr_dirs if (z / '.zarray').exists() or any(z.glob('*/.zarray')) or any(z.glob('*/zarr.json'))]

        if valid_zarr:
            # Zarr exists, proceed directly to training
            self.start_training.emit()
            return

        # No Zarr - check if we have TIFFs to convert
        raw_images_dir = self._config.get('raw_images_dir')
        tiff_source = None
        tiff_count = 0

        # Check external raw data location first
        if raw_images_dir:
            raw_path = Path(raw_images_dir)
            if not raw_path.is_absolute() and self.project_dir:
                raw_path = self.project_dir / raw_path

            if raw_path.exists() and raw_path.is_dir() and not str(raw_path).endswith('.zarr'):
                external_tiffs = list(raw_path.glob("*.tif")) + list(raw_path.glob("*.tiff"))
                if external_tiffs:
                    tiff_source = str(raw_path)
                    tiff_count = len(external_tiffs)

        # Fallback to project directory TIFFs
        if not tiff_source:
            project_tiffs = list(self.project_dir.glob("**/*.tif")) + list(self.project_dir.glob("**/*.tiff"))
            if project_tiffs:
                tiff_source = str(project_tiffs[0].parent)
                tiff_count = len(project_tiffs)

        if not tiff_source:
            # No TIFFs or Zarr - show error
            QMessageBox.warning(
                self, "No Data",
                "No raw data found. Please import TIFF files or load existing Zarr first."
            )
            return

        # TIFFs found but no Zarr - offer to convert
        total_size_mb = sum(f.stat().st_size for f in list(Path(tiff_source).glob("*.tif"))[:3]) / 1e6
        avg_size_mb = total_size_mb / min(3, tiff_count)
        estimated_total_mb = avg_size_mb * tiff_count
        size_str = f"{estimated_total_mb / 1000:.1f} GB" if estimated_total_mb > 1000 else f"{estimated_total_mb:.0f} MB"

        reply = QMessageBox.question(
            self, "Convert to Zarr?",
            f"This project contains {tiff_count} TIFF files (~{size_str}).\n\n"
            f"MOSS requires Zarr format for optimal performance with large datasets.\n\n"
            f"Convert now? (Recommended)\n\n"
            f"This enables:\n"
            f"• Chunked loading (only load what's needed)\n"
            f"• Multi-resolution pyramids (fast zooming)\n"
            f"• Efficient 3D segmentation",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Start conversion, then navigate to training when done
            QMessageBox.information(
                self, "Converting to Zarr",
                "Zarr conversion starting...\n\n"
                "Watch the 'Zarr Volume' card on the Home page for progress.\n\n"
                "You'll be automatically taken to Ground Truth when conversion completes."
            )
            self._start_zarr_conversion_inline(tiff_source, auto_navigate_after=True)
        else:
            # User declined, proceed anyway (might be slow)
            self.start_training.emit()

    def _on_import_data(self):
        """Handle Import Data button click - opens file dialog for TIFFs or Zarr."""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "Please load or create a project first.")
            return

        path = QFileDialog.getExistingDirectory(
            self, "Select Data Folder (TIFF directory or Zarr volume)",
            str(self.project_dir)
        )
        if not path:
            return

        path = Path(path)

        # Check if selected path is a Zarr volume
        is_zarr = (path.suffix == '.zarr' or (path / '.zarray').exists()
                   or (path / '.zgroup').exists() or (path / 'zarr.json').exists())

        if is_zarr:
            # Link/copy the Zarr volume into the project
            self._link_zarr_to_project(path)
            return

        # Check for TIFFs
        tiff_files = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
        if tiff_files:
            # Store as raw_images_dir in config (reference, don't copy)
            self._config['raw_images_dir'] = str(path)
            from ..project_config import save_project_config
            save_project_config(str(self.project_dir), self._config)
            self._scan_project()
            QMessageBox.information(self, "Data Linked",
                f"Linked {len(tiff_files)} TIFF files from:\n{path}\n\n"
                "To use with MOSS, convert to Zarr using the Zarr Volume card below.")
        else:
            QMessageBox.warning(self, "No Data Found",
                "No TIFF files or Zarr volume found in selected directory.\n\n"
                "Select a folder containing .tif/.tiff files, or a .zarr directory.")

    def _on_convert_to_zarr(self):
        """Handle Convert to Zarr / Generate Pyramids button click."""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "Please load or create a project first.")
            return

        # Check if this is a "Generate Pyramids" action
        if self.zarr_card.action_btn and self.zarr_card.action_btn.text() == "Generate Pyramids":
            self._generate_pyramids()
            return

        # Look for TIFF directories - check raw_images_dir from config
        # DO NOT check train_images/ - that's for crops, not raw data!
        tiff_dirs = []
        raw_images_dir = self._config.get('raw_images_dir')

        if raw_images_dir:
            raw_path = Path(raw_images_dir)
            # Handle relative paths
            if not raw_path.is_absolute() and self.project_dir:
                raw_path = self.project_dir / raw_path

            # Check if it's a valid TIFF directory (not a .zarr)
            if raw_path.exists() and raw_path.is_dir() and not str(raw_path).endswith('.zarr'):
                tiff_files = list(raw_path.glob("*.tif")) + list(raw_path.glob("*.tiff"))
                if tiff_files:
                    tiff_dirs.append(str(raw_path))

        if not tiff_dirs:
            # Ask user to select a directory
            path = QFileDialog.getExistingDirectory(
                self, "Select TIFF Directory or Zarr Volume",
                str(self.project_dir)
            )
            if path:
                path_obj = Path(path)

                # Check if this is already a Zarr directory — link it instead
                is_zarr = (path_obj.suffix == '.zarr' or (path_obj / '.zgroup').exists()
                           or (path_obj / 'zarr.json').exists() or (path_obj / '.zarray').exists())
                if is_zarr:
                    self._link_zarr_to_project(path_obj)
                    return

                tiff_files = list(path_obj.glob("*.tif")) + list(path_obj.glob("*.tiff"))
                if tiff_files:
                    tiff_dirs.append(path)
                else:
                    QMessageBox.warning(self, "No Data Found",
                        "No TIFF files or Zarr volume found in selected directory.\n\n"
                        "Select a folder containing .tif/.tiff files or a .zarr directory.")
                    return
            else:
                return

        # Use first found directory (or let user select if multiple)
        tiff_source = tiff_dirs[0]
        tiff_count = len(list(Path(tiff_source).glob("*.tif")) + list(Path(tiff_source).glob("*.tiff")))

        # Check if Zarr already exists (re-convert case)
        existing_zarr = list(self.project_dir.glob("**/*.zarr"))
        is_reconvert = bool(existing_zarr)

        if is_reconvert:
            reply = QMessageBox.question(
                self, "Re-convert to Zarr",
                f"Existing Zarr volume(s) found:\n"
                + "\n".join(f"  • {z.name}" for z in existing_zarr[:3])
                + (f"\n  ... and {len(existing_zarr) - 3} more" if len(existing_zarr) > 3 else "")
                + f"\n\nRe-convert {tiff_count} TIFF files from:\n{tiff_source}\n\n"
                f"⚠️ This will DELETE existing Zarr volumes and create new ones.\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
        else:
            reply = QMessageBox.question(
                self, "Convert to Zarr",
                f"Convert {tiff_count} TIFF files from:\n{tiff_source}\n\n"
                f"This will create a Zarr volume with multi-resolution pyramids for:\n"
                f"• Efficient chunked loading\n"
                f"• Fast zooming at multiple scales\n"
                f"• Better 3D segmentation performance\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

        if reply == QMessageBox.StandardButton.Yes:
            # Delete existing Zarr volumes if re-converting
            if is_reconvert:
                import shutil
                for zarr_dir in existing_zarr:
                    try:
                        if zarr_dir.is_symlink():
                            zarr_dir.unlink()
                        else:
                            shutil.rmtree(zarr_dir)
                        print(f"Deleted existing Zarr: {zarr_dir}")
                    except Exception as e:
                        print(f"Warning: Could not delete {zarr_dir}: {e}")

            self._start_zarr_conversion(tiff_source)

    def _check_zarr_missing_pyramids(self, zarr_path) -> bool:
        """Check if a zarr volume is missing pyramid/downsampled levels."""
        try:
            import zarr
            group = zarr.open_group(str(zarr_path), mode='r')
            attrs = dict(group.attrs) if hasattr(group, 'attrs') else {}

            # Has multiscales metadata = has pyramids
            if 'multiscales' in attrs:
                return False

            # Check for standard pyramid paths
            standard_paths = ['0', 's1', 's2', 's3']
            found_levels = sum(1 for p in standard_paths if p in group)
            if found_levels >= 2:
                return False

            # Only a single array (like 'volume') = missing pyramids
            return True
        except Exception:
            return False

    def _generate_pyramids(self):
        """Generate 2x, 4x, 8x downsampled pyramid levels for the zarr volume."""
        zarr_dirs = list(self.project_dir.glob("**/*.zarr"))
        valid_zarr = [z for z in zarr_dirs if (z / '.zarray').exists() or any(z.glob('*/.zarray')) or any(z.glob('*/zarr.json'))]

        if not valid_zarr:
            QMessageBox.warning(self, "No Zarr", "No valid Zarr volume found.")
            return

        zarr_path = valid_zarr[0]

        reply = QMessageBox.question(
            self, "Generate Pyramids",
            f"Generate 2x, 4x, and 8x downsampled pyramid levels for:\n"
            f"{zarr_path.name}\n\n"
            "This may take a while for large volumes but will greatly\n"
            "improve navigation and rendering performance.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Run pyramid generation in a thread
        self.zarr_card.set_status(True, "Generating pyramids...", show_action=False)

        from PyQt6.QtCore import QThread, pyqtSignal

        class PyramidWorker(QThread):
            finished = pyqtSignal(bool, str)
            progress = pyqtSignal(str)
            progress_percent = pyqtSignal(int, str)  # percent, status text

            def __init__(self, zarr_path):
                super().__init__()
                self.zarr_path = zarr_path

            def run(self):
                try:
                    import zarr
                    import numpy as np
                    from scipy.ndimage import zoom

                    self.progress.emit("Opening zarr volume...")
                    group = zarr.open_group(str(self.zarr_path), mode='r+')

                    # Find the source array
                    source_key = None
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, zarr.Array) and len(item.shape) == 3:
                            source_key = key
                            break

                    if source_key is None:
                        self.finished.emit(False, "No 3D array found in zarr")
                        return

                    source = group[source_key]
                    z_size, y_size, x_size = source.shape
                    self.progress.emit(f"Source: {z_size}x{y_size}x{x_size}")

                    # Generate each level
                    # Total work: sum of slices across all levels for progress
                    levels_to_do = [(f, f"s{f}") for f in [2, 4, 8] if f"s{f}" not in group]
                    total_slices_all = sum(z_size // f for f, _ in levels_to_do)
                    slices_done = 0

                    for factor, level_name in levels_to_do:
                        new_z = z_size // factor
                        new_y = y_size // factor
                        new_x = x_size // factor
                        self.progress.emit(f"Generating {level_name} ({new_z}x{new_y}x{new_x})...")
                        self.progress_percent.emit(
                            int(100 * slices_done / max(total_slices_all, 1)),
                            f"Generating {level_name}..."
                        )

                        # Create output array
                        out = group.create_dataset(
                            level_name,
                            shape=(new_z, new_y, new_x),
                            chunks=(1, min(1024, new_y), min(1024, new_x)),
                            dtype=source.dtype,
                        )

                        # Process slice by slice (memory efficient)
                        for z_out in range(new_z):
                            z_in = z_out * factor
                            # Average 'factor' slices together
                            z_end = min(z_in + factor, z_size)
                            chunk = np.array(source[z_in:z_end], dtype=np.float32)
                            avg_slice = chunk.mean(axis=0)
                            # Downsample XY
                            downsampled = zoom(avg_slice, 1.0 / factor, order=1)
                            # Handle rounding
                            out[z_out] = downsampled[:new_y, :new_x].astype(source.dtype)

                            slices_done += 1
                            if (z_out + 1) % 50 == 0:
                                pct = int(100 * slices_done / max(total_slices_all, 1))
                                self.progress_percent.emit(pct, f"{level_name}: {z_out+1}/{new_z}")
                                self.progress.emit(f"  {level_name}: {z_out+1}/{new_z} slices")

                    self.finished.emit(True, "Pyramids generated successfully!")
                except Exception as e:
                    self.finished.emit(False, str(e))

        self._pyramid_worker = PyramidWorker(zarr_path)
        self._pyramid_worker.progress.connect(lambda msg: print(f"[Pyramids] {msg}"))
        self._pyramid_worker.progress_percent.connect(
            lambda pct, msg: self.zarr_card.set_progress(pct, msg)
        )
        self._pyramid_worker.finished.connect(self._on_pyramid_generation_finished)
        self.zarr_card.set_progress(0, "Starting pyramid generation...")
        self._pyramid_worker.start()

    def _on_pyramid_generation_finished(self, success: bool, message: str):
        """Handle pyramid generation completion."""
        if success:
            QMessageBox.information(self, "Pyramids Generated", message)
        else:
            QMessageBox.warning(self, "Pyramid Generation Failed", f"Error: {message}")
        self._scan_project()

    def _on_load_existing_zarr(self):
        """Handle Load Existing Zarr button click."""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "Please load or create a project first.")
            return

        # Ask user to select an existing Zarr directory
        path = QFileDialog.getExistingDirectory(
            self, "Select Existing Zarr Volume",
            str(Path.home())
        )

        if not path:
            return

        self._link_zarr_to_project(Path(path))

    def _link_zarr_to_project(self, zarr_path: Path):
        """Validate and link/copy a Zarr volume into the project."""
        # Validate it's a Zarr volume (v2 uses .zarray, v3 uses zarr.json)
        is_valid_zarr = ((zarr_path / '.zarray').exists()
                         or any(zarr_path.glob('*/.zarray'))
                         or any(zarr_path.glob('*/zarr.json'))
                         or (zarr_path / 'zarr.json').exists())
        if not is_valid_zarr:
            QMessageBox.warning(
                self, "Invalid Zarr",
                f"'{zarr_path.name}' does not appear to be a valid Zarr volume.\n\n"
                f"A valid Zarr should contain a .zarray (v2) or zarr.json (v3) file."
            )
            return

        # Use 'raw_data.zarr' as the canonical name in the project
        dest_name = 'raw_data.zarr' if zarr_path.name != 'raw_data.zarr' else zarr_path.name
        dest_path = self.project_dir / dest_name

        # Skip if already pointing to the same location
        if dest_path.exists() and dest_path.is_symlink() and dest_path.resolve() == zarr_path.resolve():
            QMessageBox.information(self, "Already Linked",
                f"This Zarr volume is already linked to the project.")
            return

        if dest_path.exists():
            reply = QMessageBox.question(
                self, "Zarr Exists",
                f"'{dest_name}' already exists in project.\n\nReplace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            # Remove existing
            import shutil
            if dest_path.is_symlink():
                dest_path.unlink()
            else:
                shutil.rmtree(dest_path)

        # Create symlink to avoid copying large data
        try:
            dest_path.symlink_to(zarr_path)
            QMessageBox.information(
                self, "Zarr Loaded",
                f"Linked Zarr volume as {dest_name}\n\n"
                f"Original location: {zarr_path}"
            )
        except OSError:
            # Symlink failed (e.g., cross-device), try copying
            reply = QMessageBox.question(
                self, "Copy Zarr?",
                f"Cannot create link. Copy the Zarr volume instead?\n\n"
                f"This may take a while for large volumes.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                import shutil
                shutil.copytree(zarr_path, dest_path)
                QMessageBox.information(self, "Zarr Copied", f"Copied Zarr volume: {dest_name}")
            else:
                return

        # Update config to point to the zarr
        self._config['raw_images_dir'] = str(dest_path)
        from ..project_config import save_project_config
        save_project_config(str(self.project_dir), self._config)

        # Refresh to show the new Zarr
        self._scan_project()

    def refresh(self):
        """Refresh the data status."""
        self._scan_project()
