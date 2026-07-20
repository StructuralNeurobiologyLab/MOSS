#!/usr/bin/env python3
"""
Training Data Reviewer - Fast tool for reviewing and discarding training crops.

Keyboard shortcuts:
    A or Left Arrow:  Previous image
    D or Right Arrow: Next image
    Space:            Discard current image (moves to discarded folder)
    Escape:           Close reviewer
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QWidget, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent

import numpy as np
from PIL import Image

from .paint_canvas import PaintCanvas
# Disable PIL decompression bomb warning for large EM images
Image.MAX_IMAGE_PIXELS = None


class TrainingDataReviewer(QDialog):
    """
    Fast popup for reviewing training data crops and discarding bad ones.

    Displays images nearly fullscreen with minimal controls.
    Use A/D or arrow keys to navigate, Space to discard.
    """

    # Signal emitted when data was modified (images discarded)
    data_modified = pyqtSignal()

    def __init__(self, train_images_dir: Path, train_masks_dir: Path,
                 project_dir: Path, parent=None):
        super().__init__(parent)

        self.train_images_dir = Path(train_images_dir)
        self.train_masks_dir = Path(train_masks_dir)
        self.project_dir = Path(project_dir)

        # Create discarded folders
        self.discarded_dir = self.project_dir / "discarded"
        self.discarded_images_dir = self.discarded_dir / "train_images"
        self.discarded_masks_dir = self.discarded_dir / "train_masks"

        # 2.5D training directories
        self.train_images_25d_dir = self.project_dir / "train_images_25d"
        self.train_masks_25d_dir = self.project_dir / "train_masks_25d"
        self.discarded_images_25d_dir = self.discarded_dir / "train_images_25d"
        self.discarded_masks_25d_dir = self.discarded_dir / "train_masks_25d"

        # Dwarf 2.5D training directories
        self.train_images_dwarf25d_dir = self.project_dir / "train_images_dwarf25d"
        self.train_masks_dwarf25d_dir = self.project_dir / "train_masks_dwarf25d"
        self.discarded_images_dwarf25d_dir = self.discarded_dir / "train_images_dwarf25d"
        self.discarded_masks_dwarf25d_dir = self.discarded_dir / "train_masks_dwarf25d"

        # Load image list
        self.image_files: List[Path] = []
        self.current_index = 0
        self.discard_count = 0
        self._modified = False

        # Paint-mode state: correct the current crop's mask in place with the brush.
        self._paint_mode = False
        self._canvas_dirty = False

        self._load_image_list()
        self._init_ui()

    def _load_image_list(self):
        """Load list of training images."""
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        # Sort by modification time (newest first) instead of name
        self.image_files = sorted(
            [f for f in self.train_images_dir.iterdir()
             if f.suffix.lower() in extensions],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

    def _init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Training Data Reviewer")
        self.setModal(True)

        # Make window large but not fullscreen
        self.resize(900, 750)

        # Dark background for better image viewing
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
            QPushButton#discardBtn {
                background-color: #8b0000;
                border-color: #aa0000;
            }
            QPushButton#discardBtn:hover {
                background-color: #aa0000;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Top toolbar: brush toggle to correct this crop's mask in place.
        top_bar = QHBoxLayout()
        self.paint_toggle = QPushButton("\U0001F58C  Paint")
        self.paint_toggle.setCheckable(True)
        self.paint_toggle.setToolTip("Toggle painting to correct this crop's mask "
                                     "(same brush/erase as the Ground Truth tab)")
        self.paint_toggle.toggled.connect(self._toggle_paint)
        top_bar.addWidget(self.paint_toggle)

        self.eraser_toggle = QPushButton("Eraser")
        self.eraser_toggle.setCheckable(True)
        self.eraser_toggle.toggled.connect(self._toggle_eraser)
        top_bar.addWidget(self.eraser_toggle)

        brush_label = QLabel("Brush size:")
        top_bar.addWidget(brush_label)
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 200)
        self.brush_size_spin.setValue(10)
        self.brush_size_spin.setStyleSheet(
            "QSpinBox { background:#333; color:#fff; border:1px solid #555; padding:4px; }")
        self.brush_size_spin.valueChanged.connect(self._on_brush_size)
        top_bar.addWidget(self.brush_size_spin)

        # These only apply while painting; hidden until the brush is toggled on.
        self._paint_controls = [self.eraser_toggle, brush_label, self.brush_size_spin]
        for wdg in self._paint_controls:
            wdg.setVisible(False)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Image display (takes most of the space)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(400, 400)
        layout.addWidget(self.image_label, stretch=1)

        # Editable canvas (shown only in paint mode). Reuses the Ground Truth
        # painting widget so brush/erase behave identically.
        self.paint_canvas = PaintCanvas()
        self.paint_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.paint_canvas.setMinimumSize(400, 400)
        self.paint_canvas.setMouseTracking(True)
        self.paint_canvas.edit_made.connect(self._on_canvas_edit)
        self.paint_canvas.setVisible(False)
        layout.addWidget(self.paint_canvas, stretch=1)

        # Bottom controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Navigation info
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 14px; color: #aaaaaa;")
        controls_layout.addWidget(self.info_label)

        controls_layout.addStretch()

        # Keyboard hints
        hints = QLabel("A/← Prev  |  D/→ Next  |  Space: Discard  |  Esc: Close")
        hints.setStyleSheet("font-size: 12px; color: #666666;")
        controls_layout.addWidget(hints)

        controls_layout.addStretch()

        # Discard button (also triggered by Space)
        self.discard_btn = QPushButton("Discard (Space)")
        self.discard_btn.setObjectName("discardBtn")
        self.discard_btn.clicked.connect(self._discard_current)
        controls_layout.addWidget(self.discard_btn)

        # Close button
        close_btn = QPushButton("Done (Esc)")
        close_btn.clicked.connect(self.close)
        controls_layout.addWidget(close_btn)

        layout.addWidget(controls)

        # Show first image
        self._update_display()

    def _update_display(self):
        """Update the displayed image and info."""
        if not self.image_files:
            self.image_label.setText("No training images found")
            self.info_label.setText("0 / 0")
            self.discard_btn.setEnabled(False)
            return

        self.discard_btn.setEnabled(True)

        # Load and display image (also updates info label)
        image_path = self.image_files[self.current_index]
        self._display_image(image_path)

    def _display_image(self, image_path: Path):
        """Load and display an image with mask overlay, scaled to fit."""
        try:
            # Load image
            img = Image.open(image_path)

            # Convert to grayscale array
            if img.mode == 'L':
                img_array = np.array(img)
            elif img.mode in ('RGB', 'RGBA'):
                img_array = np.array(img.convert('L'))
            else:
                img_array = np.array(img.convert('L'))

            h, w = img_array.shape

            # Load corresponding mask
            mask_path = self.train_masks_dir / image_path.name
            mask_array = None
            if mask_path.exists():
                try:
                    mask_img = Image.open(mask_path)
                    mask_array = np.array(mask_img.convert('L'))
                    # Ensure mask is same size as image
                    if mask_array.shape != img_array.shape:
                        mask_array = None
                except Exception:
                    mask_array = None

            # Create RGB composite with mask overlay
            # Convert grayscale to RGB
            rgb = np.stack([img_array, img_array, img_array], axis=-1).astype(np.uint8)

            # Overlay mask in green with 25% opacity
            if mask_array is not None:
                mask_bool = mask_array > 127
                alpha = 0.25
                # Green overlay where mask is present
                rgb[mask_bool, 0] = (rgb[mask_bool, 0] * (1 - alpha)).astype(np.uint8)  # R
                rgb[mask_bool, 1] = (rgb[mask_bool, 1] * (1 - alpha) + 255 * alpha).astype(np.uint8)  # G
                rgb[mask_bool, 2] = (rgb[mask_bool, 2] * (1 - alpha)).astype(np.uint8)  # B

            # Create QImage from RGB (keep reference to prevent garbage collection)
            self._display_buffer = np.ascontiguousarray(rgb)
            bytes_per_line = 3 * w
            qimg = QImage(self._display_buffer.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Scale to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

            # Update info with mask status
            has_mask = mask_array is not None
            mask_status = "mask" if has_mask else "NO MASK"
            filename = image_path.name
            self.info_label.setText(
                f"{self.current_index + 1} / {len(self.image_files)}  |  "
                f"{filename}  |  {mask_status}  |  Discarded: {self.discard_count}"
            )

        except Exception as e:
            self.image_label.setText(f"Error loading image:\n{e}")

    # ------------------------------------------------------------------
    # In-place mask correction (paint mode)
    # ------------------------------------------------------------------
    def _toggle_paint(self, on: bool):
        """Enter/leave paint mode from the top-left brush toggle."""
        if on and not self.image_files:
            self.paint_toggle.setChecked(False)
            return
        self._paint_mode = on
        for wdg in self._paint_controls:
            wdg.setVisible(on)
        if on:
            self.paint_toggle.setText("\U0001F58C  Painting")
            self._load_into_canvas()
            self.image_label.setVisible(False)
            self.paint_canvas.setVisible(True)
            self.paint_canvas.set_tool('brush')
            self.paint_canvas.set_brush_size(self.brush_size_spin.value())
            QTimer.singleShot(0, self._fit_canvas)  # fit once the widget has its size
        else:
            self.paint_toggle.setText("\U0001F58C  Paint")
            self.eraser_toggle.setChecked(False)
            self._save_canvas_mask()
            self.paint_canvas.setVisible(False)
            self.image_label.setVisible(True)
            if self.image_files:
                self._display_image(self.image_files[self.current_index])

    def _load_into_canvas(self):
        """Load the current crop (image + mask) into the editable canvas."""
        image_path = self.image_files[self.current_index]
        img = Image.open(image_path)
        img_array = np.array(img) if img.mode == 'L' else np.array(img.convert('L'))

        mask_path = self.train_masks_dir / image_path.name
        if mask_path.exists():
            m = np.array(Image.open(mask_path).convert('L'))
            if m.shape != img_array.shape:
                m = np.zeros(img_array.shape, dtype=np.uint8)
        else:
            m = np.zeros(img_array.shape, dtype=np.uint8)

        self.paint_canvas.set_image(img_array)
        self.paint_canvas.set_mask(((m > 127).astype(np.uint8) * 255))
        self._canvas_dirty = False
        self._fit_canvas()

    def _fit_canvas(self):
        """Scale the canvas so the crop fills the view."""
        img = getattr(self.paint_canvas, 'raw_image', None)
        if img is None:
            return
        h, w = img.shape
        cw = max(1, self.paint_canvas.width())
        ch = max(1, self.paint_canvas.height())
        self.paint_canvas.zoom_level = min(cw / w, ch / h) * 0.98
        self.paint_canvas.offset = QPoint(0, 0)
        self.paint_canvas.update()

    def _on_canvas_edit(self, *args):
        self._canvas_dirty = True

    def _toggle_eraser(self, on: bool):
        self.paint_canvas.set_tool('eraser' if on else 'brush')

    def _on_brush_size(self, value: int):
        self.paint_canvas.set_brush_size(value)

    def _save_canvas_mask(self):
        """Write the edited mask back to the crop's mask file(s) (auto-save)."""
        if not self._canvas_dirty or self.paint_canvas.mask is None or not self.image_files:
            return
        name = self.image_files[self.current_index].name
        mask = (self.paint_canvas.mask > 127).astype(np.uint8) * 255

        # Always update the primary mask; update the 2.5D/dwarf copies only where a
        # mask already exists for this crop (they mirror the same label).
        targets = [self.train_masks_dir / name]
        for d in (self.train_masks_25d_dir, self.train_masks_dwarf25d_dir):
            if (d / name).exists():
                targets.append(d / name)

        saved = False
        for p in targets:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                # Match MOSS's mask format: LZW-compressed TIFF (falls back to plain
                # save for any non-TIFF extension).
                if p.suffix.lower() in ('.tif', '.tiff'):
                    Image.fromarray(mask).save(p, compression='tiff_lzw')
                else:
                    Image.fromarray(mask).save(p)
                saved = True
            except Exception as e:
                print(f"[Reviewer] Error saving mask {p}: {e}")
        if saved:
            self._modified = True
            self._canvas_dirty = False
            print(f"[Reviewer] Saved corrected mask: {name}")

    def _discard_current(self):
        """Discard the current image and its mask."""
        if not self.image_files:
            return
        # Don't auto-save a crop we're about to discard.
        self._canvas_dirty = False

        image_path = self.image_files[self.current_index]

        # Find corresponding mask
        mask_path = self.train_masks_dir / image_path.name

        # Create discarded directories if needed
        self.discarded_images_dir.mkdir(parents=True, exist_ok=True)
        self.discarded_masks_dir.mkdir(parents=True, exist_ok=True)

        # Move image
        dest_image = self.discarded_images_dir / image_path.name
        try:
            shutil.move(str(image_path), str(dest_image))
            print(f"[Reviewer] Discarded image: {image_path.name}")
        except Exception as e:
            print(f"[Reviewer] Error moving image: {e}")
            return

        # Move mask if it exists
        if mask_path.exists():
            dest_mask = self.discarded_masks_dir / mask_path.name
            try:
                shutil.move(str(mask_path), str(dest_mask))
                print(f"[Reviewer] Discarded mask: {mask_path.name}")
            except Exception as e:
                print(f"[Reviewer] Error moving mask: {e}")

        # Also check for SAM2 features
        sam2_dir = self.project_dir / "sam2_features"
        sam2_path = sam2_dir / (image_path.stem + ".npy")
        if sam2_path.exists():
            discarded_sam2_dir = self.discarded_dir / "sam2_features"
            discarded_sam2_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(sam2_path), str(discarded_sam2_dir / sam2_path.name))
                print(f"[Reviewer] Discarded SAM2 features: {sam2_path.name}")
            except Exception as e:
                print(f"[Reviewer] Error moving SAM2 features: {e}")

        # Also check for 2.5D training data
        if self.train_images_25d_dir.exists():
            image_25d_path = self.train_images_25d_dir / image_path.name
            if image_25d_path.exists():
                self.discarded_images_25d_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(image_25d_path), str(self.discarded_images_25d_dir / image_path.name))
                    print(f"[Reviewer] Discarded 2.5D image: {image_path.name}")
                except Exception as e:
                    print(f"[Reviewer] Error moving 2.5D image: {e}")

        if self.train_masks_25d_dir.exists():
            mask_25d_path = self.train_masks_25d_dir / image_path.name
            if mask_25d_path.exists():
                self.discarded_masks_25d_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(mask_25d_path), str(self.discarded_masks_25d_dir / image_path.name))
                    print(f"[Reviewer] Discarded 2.5D mask: {image_path.name}")
                except Exception as e:
                    print(f"[Reviewer] Error moving 2.5D mask: {e}")

        # Also check for dwarf 2.5D training data
        if self.train_images_dwarf25d_dir.exists():
            image_dwarf25d_path = self.train_images_dwarf25d_dir / image_path.name
            if image_dwarf25d_path.exists():
                self.discarded_images_dwarf25d_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(image_dwarf25d_path), str(self.discarded_images_dwarf25d_dir / image_path.name))
                    print(f"[Reviewer] Discarded dwarf 2.5D image: {image_path.name}")
                except Exception as e:
                    print(f"[Reviewer] Error moving dwarf 2.5D image: {e}")

        if self.train_masks_dwarf25d_dir.exists():
            mask_dwarf25d_path = self.train_masks_dwarf25d_dir / image_path.name
            if mask_dwarf25d_path.exists():
                self.discarded_masks_dwarf25d_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(mask_dwarf25d_path), str(self.discarded_masks_dwarf25d_dir / image_path.name))
                    print(f"[Reviewer] Discarded dwarf 2.5D mask: {image_path.name}")
                except Exception as e:
                    print(f"[Reviewer] Error moving dwarf 2.5D mask: {e}")

        # Update state
        self.discard_count += 1
        self._modified = True

        # Remove from list and update display
        del self.image_files[self.current_index]

        # Adjust index if needed
        if self.current_index >= len(self.image_files):
            self.current_index = max(0, len(self.image_files) - 1)

        if self._paint_mode and self.image_files:
            self._load_into_canvas()
        else:
            if self._paint_mode:  # nothing left to paint
                self.paint_toggle.setChecked(False)
            self._update_display()

    def _go_previous(self):
        """Go to previous image (auto-saving any paint edits first)."""
        if self.image_files and self.current_index > 0:
            if self._paint_mode:
                self._save_canvas_mask()
            self.current_index -= 1
            if self._paint_mode:
                self._load_into_canvas()
            else:
                self._update_display()

    def _go_next(self):
        """Go to next image (auto-saving any paint edits first)."""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            if self._paint_mode:
                self._save_canvas_mask()
            self.current_index += 1
            if self._paint_mode:
                self._load_into_canvas()
            else:
                self._update_display()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        key = event.key()

        if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self._go_previous()
        elif key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self._go_next()
        elif key == Qt.Key.Key_Space:
            self._discard_current()
        elif key == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """Handle resize to re-scale image."""
        super().resizeEvent(event)
        if self._paint_mode:
            self._fit_canvas()
        elif self.image_files:
            self._display_image(self.image_files[self.current_index])

    def closeEvent(self, event):
        """Auto-save any pending paint edit, then emit if data was modified."""
        if self._paint_mode:
            self._save_canvas_mask()
        if self._modified:
            self.data_modified.emit()
        super().closeEvent(event)

    def was_modified(self) -> bool:
        """Check if any data was discarded."""
        return self._modified
