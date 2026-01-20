#!/usr/bin/env python3
"""
Cleanup page - threshold heatmap and interactively clean up binary mask.
Based on interactive_training_page but simplified for mask cleanup.
"""

import os
import multiprocessing
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QToolBar, QSlider, QFileDialog, QSizePolicy, QMessageBox,
    QProgressBar, QCheckBox, QSpinBox, QGroupBox, QFormLayout,
    QLineEdit
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import QApplication

from ..dpi_scaling import scaled, scaled_font


def _load_single_heatmap(args):
    """Load a single heatmap image (for parallel loading)."""
    idx, heatmap_path = args
    try:
        img = np.array(Image.open(heatmap_path))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.uint8)
        return idx, img, None
    except Exception as e:
        return idx, None, str(e)


def otsu_threshold(image):
    """Calculate Otsu's threshold for an image."""
    # Compute histogram
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Total number of pixels
    total = image.size

    # Initialize
    sum_total = np.sum(np.arange(256) * hist)
    sum_bg = 0
    weight_bg = 0
    max_variance = 0
    threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    return threshold


# Import OptimizedCanvas
try:
    from ..widgets.optimized_canvas import OptimizedCanvas
except ImportError:
    OptimizedCanvas = None


class CleanupPage(QWidget):
    """Cleanup page for thresholding and editing binary masks from heatmaps."""

    # Signals
    cleanup_complete = pyqtSignal(str)  # output_dir
    busy_changed = pyqtSignal(bool)

    # Sliding window parameters
    WINDOW_SIZE = 200
    LOAD_THRESHOLD = 30
    BATCH_SIZE = 100

    def __init__(self):
        super().__init__()

        # Application state
        self.current_slice_index = 0
        self.current_tool = 'brush'

        # Image file list
        self.heatmap_files = []  # List of Path objects for heatmaps

        # Sliding window of loaded images
        self.heatmaps = {}  # {index: numpy array}
        self.masks = {}     # {index: numpy array} - binary masks after threshold
        self.window_start = 0
        self.window_end = 0
        self._loading_in_progress = False

        # Config from wizard
        self.config = {}
        self.heatmap_dir = None
        self.output_dir = None

        # Threshold
        self.threshold_value = 128  # Default threshold
        self.use_otsu = True

        # Undo history
        self.undo_stack = []
        self.max_undo = 50

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        main_layout = QVBoxLayout(self)
        margin = scaled(5)
        main_layout.setContentsMargins(margin, margin, margin, margin)
        main_layout.setSpacing(scaled(5))

        if OptimizedCanvas is None:
            error_label = QLabel(
                "Error: OptimizedCanvas not found.\n\n"
                "This component is required for the cleanup page."
            )
            error_label.setFont(scaled_font(14))
            error_label.setStyleSheet("color: red;")
            main_layout.addWidget(error_label)
            return

        # Load heatmaps controls
        load_group = QGroupBox("Heatmap Source")
        load_layout = QHBoxLayout(load_group)

        self.heatmap_dir_label = QLineEdit()
        self.heatmap_dir_label.setPlaceholderText("Select heatmap directory...")
        self.heatmap_dir_label.setReadOnly(True)
        load_layout.addWidget(self.heatmap_dir_label)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_heatmap_dir)
        load_layout.addWidget(self.browse_btn)

        self.load_btn = QPushButton("Load Heatmaps")
        self.load_btn.clicked.connect(self._load_selected_heatmaps)
        load_layout.addWidget(self.load_btn)

        main_layout.addWidget(load_group)

        # Threshold controls at top
        threshold_group = QGroupBox("Threshold Settings")
        threshold_layout = QHBoxLayout(threshold_group)

        self.otsu_checkbox = QCheckBox("Use Otsu (auto)")
        self.otsu_checkbox.setChecked(True)
        self.otsu_checkbox.stateChanged.connect(self._on_otsu_changed)
        threshold_layout.addWidget(self.otsu_checkbox)

        threshold_layout.addWidget(QLabel("Manual:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_slider.setEnabled(False)  # Disabled when Otsu is checked
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("128")
        self.threshold_label.setMinimumWidth(scaled(40))
        threshold_layout.addWidget(self.threshold_label)

        self.apply_threshold_btn = QPushButton("Apply to All")
        self.apply_threshold_btn.clicked.connect(self._apply_threshold_to_all)
        threshold_layout.addWidget(self.apply_threshold_btn)

        main_layout.addWidget(threshold_group)

        # Create canvas
        self.canvas = OptimizedCanvas()
        self.canvas.edit_made.connect(self.on_edit_made)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(scaled(600), scaled(400))

        # Create toolbar
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)

        # Add canvas
        main_layout.addWidget(self.canvas)

        # Create bottom controls
        controls = self.create_controls()
        main_layout.addWidget(controls)

        # Status label
        self.status_label = QLabel("Load heatmaps to begin.")
        main_layout.addWidget(self.status_label)

        # Connect brush size changes
        self.canvas.brush_size_changed.connect(self._on_canvas_brush_size_changed)

    def create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)

        # Brush tool
        self.brush_btn = QPushButton("Brush (B)")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.brush_btn.clicked.connect(lambda: self.select_tool('brush'))
        toolbar.addWidget(self.brush_btn)

        # Eraser tool
        self.eraser_btn = QPushButton("Eraser (E)")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self.select_tool('eraser'))
        toolbar.addWidget(self.eraser_btn)

        toolbar.addSeparator()

        # Brush size
        toolbar.addWidget(QLabel("Size:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 100)
        self.brush_slider.setValue(20)
        self.brush_slider.setMaximumWidth(scaled(150))
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        toolbar.addWidget(self.brush_slider)

        self.brush_size_label = QLabel("20")
        self.brush_size_label.setMinimumWidth(scaled(30))
        toolbar.addWidget(self.brush_size_label)

        toolbar.addSeparator()

        # Undo
        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)
        toolbar.addWidget(self.undo_btn)

        return toolbar

    def create_controls(self):
        """Create bottom controls."""
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)

        # Navigation slider
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        layout.addWidget(self.slice_slider)

        # Slice label
        self.slice_label = QLabel("0 / 0")
        self.slice_label.setMinimumWidth(scaled(100))
        layout.addWidget(self.slice_label)

        # Save button
        self.save_btn = QPushButton("Save Masks")
        self.save_btn.clicked.connect(self.save_all_masks)
        layout.addWidget(self.save_btn)

        return controls

    def select_tool(self, tool):
        """Select a tool."""
        self.current_tool = tool
        self.brush_btn.setChecked(tool == 'brush')
        self.eraser_btn.setChecked(tool == 'eraser')
        self.canvas.set_tool(tool)

    def on_brush_size_changed(self, value):
        """Handle brush size slider change."""
        self.brush_size_label.setText(str(value))
        self.canvas.set_brush_size(value)

    def _on_canvas_brush_size_changed(self, size):
        """Update slider when brush size changed via scroll."""
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(size)
        self.brush_slider.blockSignals(False)
        self.brush_size_label.setText(str(size))

    def _on_otsu_changed(self, state):
        """Handle Otsu checkbox change."""
        self.use_otsu = (state == 2)
        self.threshold_slider.setEnabled(not self.use_otsu)

        if self.use_otsu and self.current_slice_index in self.heatmaps:
            # Calculate Otsu threshold for current image
            heatmap = self.heatmaps[self.current_slice_index]
            self.threshold_value = otsu_threshold(heatmap)
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(self.threshold_value)
            self.threshold_slider.blockSignals(False)
            self.threshold_label.setText(str(self.threshold_value))
            self._apply_threshold_to_current()

    def _on_threshold_changed(self, value):
        """Handle threshold slider change."""
        self.threshold_value = value
        self.threshold_label.setText(str(value))
        self._apply_threshold_to_current()

    def _apply_threshold_to_current(self):
        """Apply threshold to current slice."""
        if self.current_slice_index not in self.heatmaps:
            return

        heatmap = self.heatmaps[self.current_slice_index]
        mask = (heatmap >= self.threshold_value).astype(np.uint8) * 255
        self.masks[self.current_slice_index] = mask
        self._update_display()

    def _apply_threshold_to_all(self):
        """Apply threshold to all loaded slices."""
        self.busy_changed.emit(True)
        self.status_label.setText("Applying threshold to all slices...")
        QApplication.processEvents()

        count = 0
        for idx, heatmap in self.heatmaps.items():
            if self.use_otsu:
                threshold = otsu_threshold(heatmap)
            else:
                threshold = self.threshold_value

            mask = (heatmap >= threshold).astype(np.uint8) * 255
            self.masks[idx] = mask
            count += 1

        self._update_display()
        self.status_label.setText(f"Applied threshold to {count} slices.")
        self.busy_changed.emit(False)

    def _browse_heatmap_dir(self):
        """Browse for heatmap directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Heatmap Directory",
            str(self.config.get('project_dir', ''))
        )
        if folder:
            self.heatmap_dir_label.setText(folder)

    def _load_selected_heatmaps(self):
        """Load heatmaps from the selected directory."""
        folder = self.heatmap_dir_label.text()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid heatmap directory.")
            return

        self.heatmap_dir = Path(folder)
        self.config['heatmap_dir'] = folder

        # Clear existing data
        self.heatmaps.clear()
        self.masks.clear()
        self.heatmap_files.clear()

        self._load_heatmap_files()

    def set_config(self, config: dict):
        """Set configuration from wizard."""
        self.config = config

        # Get heatmap directory and display it
        heatmap_dir = config.get('heatmap_dir', '')
        if heatmap_dir and os.path.isdir(heatmap_dir):
            self.heatmap_dir = Path(heatmap_dir)
            if hasattr(self, 'heatmap_dir_label'):
                self.heatmap_dir_label.setText(heatmap_dir)
            self._load_heatmap_files()
        elif hasattr(self, 'heatmap_dir_label'):
            # Auto-detect heatmap_consensus directory
            project_dir = config.get('project_dir', '')
            if project_dir:
                consensus_dir = os.path.join(project_dir, 'heatmap_consensus')
                if os.path.isdir(consensus_dir):
                    self.heatmap_dir_label.setText(consensus_dir)

        # Set output directory
        project_dir = config.get('project_dir', '')
        if project_dir:
            self.output_dir = Path(project_dir) / 'binary_masks'

    def _load_heatmap_files(self):
        """Load list of heatmap files."""
        if not self.heatmap_dir:
            return

        # Find all heatmap files
        self.heatmap_files = sorted(
            list(self.heatmap_dir.glob("heatmap_z*.tif")) +
            list(self.heatmap_dir.glob("*.tif"))
        )
        # Deduplicate
        self.heatmap_files = sorted(set(self.heatmap_files))

        if self.heatmap_files:
            self.slice_slider.setRange(0, len(self.heatmap_files) - 1)
            self.slice_label.setText(f"0 / {len(self.heatmap_files) - 1}")
            self.status_label.setText(f"Found {len(self.heatmap_files)} heatmap files.")
            self._load_initial_window()
        else:
            self.status_label.setText("No heatmap files found.")

    def _load_initial_window(self):
        """Load the initial window of images using multiple CPU cores."""
        if not self.heatmap_files:
            return

        self._loading_in_progress = True

        # Use ProcessPoolExecutor for true parallelism across CPU cores
        num_workers = min(multiprocessing.cpu_count(), 16)
        print(f"[Cleanup] Loading heatmaps using {num_workers} CPU cores...")

        self.status_label.setText(f"Loading heatmaps using {num_workers} cores...")
        QApplication.processEvents()

        # Load first WINDOW_SIZE images
        end_idx = min(self.WINDOW_SIZE, len(self.heatmap_files))

        load_args = [
            (i, self.heatmap_files[i])
            for i in range(end_idx)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_load_single_heatmap, load_args))

        # Just store heatmaps - DON'T calculate thresholds yet
        for idx, heatmap, error in results:
            if heatmap is not None:
                self.heatmaps[idx] = heatmap

        self.window_start = 0
        self.window_end = end_idx
        self._loading_in_progress = False

        print(f"[Cleanup] Loaded {len(self.heatmaps)} heatmaps")

        # Display first slice (threshold calculated on-demand)
        self.current_slice_index = 0
        self._update_display()
        self.status_label.setText(f"Loaded {len(self.heatmaps)} heatmaps using {num_workers} cores. Use 'Apply to All' to threshold.")

    def _update_display(self):
        """Update canvas with current slice."""
        if self.current_slice_index not in self.heatmaps:
            return

        heatmap = self.heatmaps[self.current_slice_index]
        mask = self.masks.get(self.current_slice_index)

        if mask is None:
            if self.use_otsu:
                threshold = otsu_threshold(heatmap)
            else:
                threshold = self.threshold_value
            mask = (heatmap >= threshold).astype(np.uint8) * 255
            self.masks[self.current_slice_index] = mask

        # Display heatmap as image, mask as overlay
        self.canvas.set_image(heatmap.astype(np.float32))
        self.canvas.set_mask(mask)
        self.slice_label.setText(f"{self.current_slice_index} / {len(self.heatmap_files) - 1}")

    def on_slice_changed(self, value):
        """Handle slice slider change."""
        if value == self.current_slice_index:
            return

        # Save current mask
        if hasattr(self.canvas, 'mask') and self.canvas.mask is not None:
            self.masks[self.current_slice_index] = self.canvas.mask.copy()

        self.current_slice_index = value

        # Check if we need to load more
        self._maybe_load_more()
        self._update_display()

    def _maybe_load_more(self):
        """Load more images if approaching edge of window."""
        if self._loading_in_progress:
            return

        idx = self.current_slice_index

        # Check if we need to load more towards the end
        if idx >= self.window_end - self.LOAD_THRESHOLD:
            self._load_forward()

        # Check if we need to load more towards the start
        if idx <= self.window_start + self.LOAD_THRESHOLD:
            self._load_backward()

    def _load_forward(self):
        """Load more images forward using multiple CPU cores."""
        if self._loading_in_progress:
            return
        if self.window_end >= len(self.heatmap_files):
            return

        self._loading_in_progress = True

        start = self.window_end
        end = min(start + self.BATCH_SIZE, len(self.heatmap_files))

        load_args = [
            (i, self.heatmap_files[i])
            for i in range(start, end)
        ]

        num_workers = min(multiprocessing.cpu_count(), 16)
        print(f"[Cleanup] Loading forward {start}-{end} using {num_workers} cores...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_load_single_heatmap, load_args))

        # Just store heatmaps - threshold on-demand
        for idx, heatmap, error in results:
            if heatmap is not None:
                self.heatmaps[idx] = heatmap

        self.window_end = end

        # Unload old images if window too large
        if self.window_end - self.window_start > self.WINDOW_SIZE * 2:
            for i in range(self.window_start, self.window_start + self.BATCH_SIZE):
                self.heatmaps.pop(i, None)
                # Don't remove masks - keep edits
            self.window_start += self.BATCH_SIZE

        self._loading_in_progress = False
        print(f"[Cleanup] Forward load complete")

    def _load_backward(self):
        """Load more images backward using multiple CPU cores."""
        if self._loading_in_progress:
            return
        if self.window_start <= 0:
            return

        self._loading_in_progress = True

        end = self.window_start
        start = max(0, end - self.BATCH_SIZE)

        load_args = [
            (i, self.heatmap_files[i])
            for i in range(start, end)
        ]

        num_workers = min(multiprocessing.cpu_count(), 16)
        print(f"[Cleanup] Loading backward {start}-{end} using {num_workers} cores...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_load_single_heatmap, load_args))

        # Just store heatmaps - threshold on-demand
        for idx, heatmap, error in results:
            if heatmap is not None:
                self.heatmaps[idx] = heatmap

        self.window_start = start

        # Unload old images if window too large
        if self.window_end - self.window_start > self.WINDOW_SIZE * 2:
            for i in range(self.window_end - self.BATCH_SIZE, self.window_end):
                self.heatmaps.pop(i, None)
            self.window_end -= self.BATCH_SIZE

        self._loading_in_progress = False
        print(f"[Cleanup] Backward load complete")

    def on_edit_made(self, mask):
        """Handle edit made to mask."""
        # Save to undo stack
        if self.current_slice_index in self.masks:
            old_mask = self.masks[self.current_slice_index].copy()
            self.undo_stack.append((self.current_slice_index, old_mask))
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)

        # Update mask
        self.masks[self.current_slice_index] = mask.copy()

    def undo(self):
        """Undo last edit."""
        if not self.undo_stack:
            return

        idx, old_mask = self.undo_stack.pop()
        self.masks[idx] = old_mask

        if idx == self.current_slice_index:
            self._update_display()

    def save_all_masks(self):
        """Save all masks to output directory."""
        if not self.output_dir:
            self.output_dir = Path(QFileDialog.getExistingDirectory(
                self, "Select Output Directory"
            ))
            if not self.output_dir:
                return

        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.busy_changed.emit(True)
        self.status_label.setText("Saving masks...")
        QApplication.processEvents()

        # First, save current mask from canvas
        if hasattr(self.canvas, 'mask') and self.canvas.mask is not None:
            self.masks[self.current_slice_index] = self.canvas.mask.copy()

        saved = 0
        for idx, mask in self.masks.items():
            out_path = self.output_dir / f"mask_z{idx:05d}.tif"
            try:
                Image.fromarray(mask).save(out_path, compression='tiff_lzw')
                saved += 1
            except Exception as e:
                print(f"Failed to save mask {idx}: {e}")

        self.status_label.setText(f"Saved {saved} masks to {self.output_dir}")
        self.busy_changed.emit(False)
        self.cleanup_complete.emit(str(self.output_dir))

    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_B:
            self.select_tool('brush')
        elif event.key() == Qt.Key.Key_E:
            self.select_tool('eraser')
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key.Key_Left:
            if self.current_slice_index > 0:
                self.slice_slider.setValue(self.current_slice_index - 1)
        elif event.key() == Qt.Key.Key_Right:
            if self.current_slice_index < len(self.heatmap_files) - 1:
                self.slice_slider.setValue(self.current_slice_index + 1)
        else:
            super().keyPressEvent(event)
