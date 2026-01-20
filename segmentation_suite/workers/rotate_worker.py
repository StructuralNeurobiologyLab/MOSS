#!/usr/bin/env python3
"""
Rotation worker for undoing diagonal rotations on prediction volumes.

Adapted from prerotate_diagonal_predictions.py
"""

import gc
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import rotate
from PyQt6.QtCore import QThread, pyqtSignal


def load_mask_slice(filepath):
    """Load a single mask image and return as array (0-255)."""
    try:
        img = Image.open(filepath)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.uint8)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


class RotateWorker(QThread):
    """Background worker for rotating diagonal predictions back to XY orientation."""

    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(str, int, int)  # diagonal_name, current, total
    finished = pyqtSignal(bool, dict)  # success, output_dirs dict
    log = pyqtSignal(str)

    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary with:
                - xy_dir: Reference XY predictions directory (for dimensions)
                - diagonals: List of dicts with:
                    - input_dir: Input prediction directory
                    - output_dir: Output directory for rotated predictions
                    - angle: Rotation angle (will be negated to undo)
                    - axes: Rotation axes tuple, e.g., (0, 2)
                    - name: Diagonal name
        """
        super().__init__()
        self.config = config
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def run(self):
        try:
            self.started.emit()

            xy_dir = Path(self.config['xy_dir'])
            diagonals = self.config['diagonals']

            # Get reference dimensions from XY
            xy_files = sorted(list(xy_dir.glob("*_pred.tif")) +
                             list(xy_dir.glob("*_pred.png")) +
                             list(xy_dir.glob("*.tif")))

            if not xy_files:
                self.finished.emit(False, {"error": "No XY reference files found"})
                return

            first_xy = load_mask_slice(xy_files[0])
            y_size, x_size = first_xy.shape
            total_z = len(xy_files)

            self.log.emit(f"Reference dimensions: Z={total_z}, Y={y_size}, X={x_size}")

            output_dirs = {}

            for diag_config in diagonals:
                if self.should_stop:
                    break

                name = diag_config['name']
                input_dir = Path(diag_config['input_dir'])
                output_dir = Path(diag_config['output_dir'])
                angle = diag_config['angle']
                axes = tuple(diag_config['axes'])

                output_dir.mkdir(exist_ok=True, parents=True)
                output_dirs[name] = str(output_dir)

                self.log.emit(f"Processing {name}: rotating by {-angle} degrees (undoing {angle})")
                self._rotate_diagonal(input_dir, output_dir, angle, axes, y_size, x_size, name)

            self.finished.emit(True, output_dirs)

        except Exception as e:
            self.log.emit(f"Rotation error: {e}")
            self.finished.emit(False, {"error": str(e)})

    def _rotate_diagonal(self, input_dir, output_dir, angle, axes, y_size, x_size, name):
        """Rotate a diagonal prediction volume back to XY orientation."""
        # Find prediction files
        pred_files = sorted(list(input_dir.glob("*_pred.tif")) +
                           list(input_dir.glob("*_pred.png")))

        if not pred_files:
            self.log.emit(f"No prediction files found in {input_dir}")
            return

        self.log.emit(f"Found {len(pred_files)} {name} slices")

        # Load full volume
        self.log.emit(f"Loading {name} volume...")
        images = []
        for i, filepath in enumerate(pred_files):
            if self.should_stop:
                return
            img = load_mask_slice(filepath)
            if img is not None:
                images.append(img[:y_size, :x_size])
            if (i + 1) % 100 == 0:
                self.progress.emit(name + " (loading)", i + 1, len(pred_files))

        volume = np.stack(images, axis=0)
        self.log.emit(f"Volume shape: {volume.shape}")

        # Rotate back to XY orientation (use negative angle)
        self.log.emit(f"Rotating {name} back to XY (angle={-angle})...")
        rotated = rotate(volume, -angle, axes=axes, reshape=False, order=1)
        rotated_uint8 = rotated.astype(np.uint8)

        # Save
        self.log.emit(f"Saving {name} rotated slices...")
        for z in range(rotated_uint8.shape[0]):
            if self.should_stop:
                break
            slice_2d = rotated_uint8[z]
            out_path = output_dir / f"rotated_z{z:05d}.tif"
            Image.fromarray(slice_2d).save(out_path, compression='tiff_deflate')
            if (z + 1) % 100 == 0:
                self.progress.emit(name + " (saving)", z + 1, rotated_uint8.shape[0])

        # Stats
        nonzero_orig = np.sum(volume > 0)
        nonzero_rot = np.sum(rotated_uint8 > 0)
        self.log.emit(f"{name} - Original non-zero: {nonzero_orig:,}, Rotated: {nonzero_rot:,}")

        # Clean up
        del volume, rotated, rotated_uint8
        gc.collect()
