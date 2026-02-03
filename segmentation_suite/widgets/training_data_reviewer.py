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
    QSizePolicy, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent

import numpy as np
from PIL import Image


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

        # Load image list
        self.image_files: List[Path] = []
        self.current_index = 0
        self.discard_count = 0
        self._modified = False

        self._load_image_list()
        self._init_ui()

    def _load_image_list(self):
        """Load list of training images, sorted by modification time (newest first)."""
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.image_files = sorted(
            [f for f in self.train_images_dir.iterdir() if f.suffix.lower() in extensions],
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

        # Image display (takes most of the space)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(400, 400)
        layout.addWidget(self.image_label, stretch=1)

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

    def _discard_current(self):
        """Discard the current image and its mask."""
        if not self.image_files:
            return

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

        # Update state
        self.discard_count += 1
        self._modified = True

        # Remove from list and update display
        del self.image_files[self.current_index]

        # Adjust index if needed
        if self.current_index >= len(self.image_files):
            self.current_index = max(0, len(self.image_files) - 1)

        self._update_display()

    def _go_previous(self):
        """Go to previous image."""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self._update_display()

    def _go_next(self):
        """Go to next image."""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
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
        if self.image_files:
            self._display_image(self.image_files[self.current_index])

    def closeEvent(self, event):
        """Emit signal if data was modified."""
        if self._modified:
            self.data_modified.emit()
        super().closeEvent(event)

    def was_modified(self) -> bool:
        """Check if any data was discarded."""
        return self._modified
