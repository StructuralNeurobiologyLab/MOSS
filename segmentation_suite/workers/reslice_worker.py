#!/usr/bin/env python3
"""
Reslicing worker for creating XZ, YZ, and diagonal views of a z-stack.

Adapted from reslice_stack.py

Uses ProcessPoolExecutor for true parallelism (bypasses Python GIL).
"""

import os
import gc
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.ndimage import rotate
from PyQt6.QtCore import QThread, pyqtSignal


def load_image(image_path):
    """Load a single image and return as numpy array."""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def save_image_normalized(image_array, output_path):
    """Save a 2D array as a normalized TIFF image with LZW compression."""
    if image_array.dtype != np.uint8:
        img_min = image_array.min()
        img_max = image_array.max()
        if img_max > img_min:
            normalized = ((image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image_array, dtype=np.uint8)
    else:
        normalized = image_array

    img = Image.fromarray(normalized)
    img.save(output_path, compression='tiff_lzw')


def _save_xz_slice_worker(args):
    """Save a single XZ slice (runs in separate process)."""
    y_idx, xz_batch_slice, output_dir, batch_idx = args
    slice_path = Path(output_dir) / f"xz_slice_{y_idx:05d}.tif"

    if batch_idx == 0:
        save_image_normalized(xz_batch_slice, slice_path)
    else:
        existing = np.array(Image.open(slice_path))
        combined = np.vstack([existing, xz_batch_slice])
        save_image_normalized(combined, slice_path)
    return y_idx


def _save_yz_slice_worker(args):
    """Save a single YZ slice (runs in separate process)."""
    x_idx, yz_batch_slice, output_dir, batch_idx = args
    slice_path = Path(output_dir) / f"yz_slice_{x_idx:05d}.tif"

    if batch_idx == 0:
        save_image_normalized(yz_batch_slice, slice_path)
    else:
        existing = np.array(Image.open(slice_path))
        combined = np.vstack([existing, yz_batch_slice])
        save_image_normalized(combined, slice_path)
    return x_idx


class ResliceWorker(QThread):
    """Background worker for creating resliced views of a z-stack."""

    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(str, int, int)  # plane_name, current, total
    finished = pyqtSignal(bool, dict)  # success, output_dirs dict
    log = pyqtSignal(str)

    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary with:
                - input_dir: Directory containing z-stack images
                - output_dir: Base output directory
                - create_xz: bool - create XZ reslice
                - create_yz: bool - create YZ reslice
                - diagonals: list of dicts with {'angle': float, 'axes': tuple, 'name': str}
                - batch_size: int (default 200)
                - max_workers: int (default 8)
        """
        super().__init__()
        self.config = config
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def run(self):
        try:
            self.started.emit()

            input_dir = Path(self.config['input_dir'])
            output_base = Path(self.config['output_dir'])
            create_xz = self.config.get('create_xz', True)
            create_yz = self.config.get('create_yz', True)
            diagonals = self.config.get('diagonals', [])
            batch_size = self.config.get('batch_size', 200)
            max_workers = self.config.get('max_workers', 8)

            output_dirs = {}

            # Find all images
            image_paths = sorted(input_dir.glob("*.tif"))
            if not image_paths:
                image_paths = sorted(input_dir.glob("*.png"))
            if not image_paths:
                self.finished.emit(False, {"error": "No images found"})
                return

            self.log.emit(f"Found {len(image_paths)} images")
            total_z = len(image_paths)

            # Get dimensions from first image
            first_img = load_image(image_paths[0])
            y_size, x_size = first_img.shape[:2]
            self.log.emit(f"Image dimensions: {y_size} x {x_size}")

            # Process in batches
            num_batches = (total_z + batch_size - 1) // batch_size

            # XZ reslice
            if create_xz and not self.should_stop:
                xz_dir = output_base / "xz_reslice"
                xz_dir.mkdir(exist_ok=True, parents=True)
                output_dirs['xz'] = str(xz_dir)
                self._create_xz_reslice(image_paths, xz_dir, y_size, x_size, batch_size, max_workers)

            # YZ reslice
            if create_yz and not self.should_stop:
                yz_dir = output_base / "yz_reslice"
                yz_dir.mkdir(exist_ok=True, parents=True)
                output_dirs['yz'] = str(yz_dir)
                self._create_yz_reslice(image_paths, yz_dir, y_size, x_size, batch_size, max_workers)

            # Diagonal reslices
            for diag_config in diagonals:
                if self.should_stop:
                    break
                name = diag_config.get('name', f"diag_{diag_config['angle']}")
                diag_dir = output_base / name
                diag_dir.mkdir(exist_ok=True, parents=True)
                output_dirs[name] = str(diag_dir)
                self._create_diagonal_reslice(
                    image_paths, diag_dir,
                    diag_config['angle'], diag_config['axes'],
                    batch_size, max_workers, name
                )

            # Check if we were stopped vs completed
            if self.should_stop:
                self.log.emit("Reslicing stopped by user")
                self.finished.emit(False, {"error": "Stopped by user"})
            else:
                self.finished.emit(True, output_dirs)

        except Exception as e:
            self.log.emit(f"Reslice error: {e}")
            self.finished.emit(False, {"error": str(e)})

    def _load_batch(self, image_paths, max_workers):
        """Load a batch of images."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            images = list(executor.map(load_image, image_paths))
        images = [img for img in images if img is not None]
        if not images:
            return None
        return np.stack(images, axis=0)

    def _slice_generator_xz(self, batch_stack, output_dir, batch_idx, y_size):
        """Generator that yields slices one at a time (no pre-copying)."""
        output_dir_str = str(output_dir)
        for y_idx in range(y_size):
            # Copy happens here, one at a time
            yield (y_idx, batch_stack[:, y_idx, :].copy(), output_dir_str, batch_idx)

    def _slice_generator_yz(self, batch_stack, output_dir, batch_idx, x_size):
        """Generator that yields slices one at a time (no pre-copying)."""
        output_dir_str = str(output_dir)
        for x_idx in range(x_size):
            # Copy happens here, one at a time
            yield (x_idx, batch_stack[:, :, x_idx].copy(), output_dir_str, batch_idx)

    def _create_xz_reslice(self, image_paths, output_dir, y_size, x_size, batch_size, max_workers):
        """Create XZ reslices (views along Y axis) using multiprocessing."""
        self.log.emit(f"Creating XZ reslices (parallel with {max_workers} processes)...")
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        total_slices = y_size * num_batches

        slices_done = 0
        for batch_idx in range(num_batches):
            if self.should_stop:
                break

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]

            self.progress.emit("XZ (loading)", batch_idx + 1, num_batches)
            batch_stack = self._load_batch(batch_paths, max_workers)
            if batch_stack is None:
                continue

            # Use ProcessPoolExecutor with submit() for progress tracking
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for args in self._slice_generator_xz(batch_stack, output_dir, batch_idx, y_size):
                    if self.should_stop:
                        break
                    futures.append(executor.submit(_save_xz_slice_worker, args))

                # Track progress as futures complete
                for future in as_completed(futures):
                    if self.should_stop:
                        break
                    slices_done += 1
                    # Update progress every 100 slices or so
                    if slices_done % 100 == 0:
                        self.progress.emit("XZ (saving)", slices_done, total_slices)

            if self.should_stop:
                break

            self.progress.emit("XZ", batch_idx + 1, num_batches)
            del batch_stack
            gc.collect()

    def _create_yz_reslice(self, image_paths, output_dir, y_size, x_size, batch_size, max_workers):
        """Create YZ reslices (views along X axis) using multiprocessing."""
        self.log.emit(f"Creating YZ reslices (parallel with {max_workers} processes)...")
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        total_slices = x_size * num_batches

        slices_done = 0
        for batch_idx in range(num_batches):
            if self.should_stop:
                break

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]

            self.progress.emit("YZ (loading)", batch_idx + 1, num_batches)
            batch_stack = self._load_batch(batch_paths, max_workers)
            if batch_stack is None:
                continue

            # Use ProcessPoolExecutor with submit() for progress tracking
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for args in self._slice_generator_yz(batch_stack, output_dir, batch_idx, x_size):
                    if self.should_stop:
                        break
                    futures.append(executor.submit(_save_yz_slice_worker, args))

                # Track progress as futures complete
                for future in as_completed(futures):
                    if self.should_stop:
                        break
                    slices_done += 1
                    # Update progress every 100 slices or so
                    if slices_done % 100 == 0:
                        self.progress.emit("YZ (saving)", slices_done, total_slices)

            if self.should_stop:
                break

            self.progress.emit("YZ", batch_idx + 1, num_batches)
            del batch_stack
            gc.collect()

    def _create_diagonal_reslice(self, image_paths, output_dir, angle, axes, batch_size, max_workers, name):
        """Create diagonal reslice by rotating the volume."""
        self.log.emit(f"Creating {name} reslice (angle={angle}, axes={axes})...")

        # For diagonal reslicing, we need to load the entire volume and rotate
        # This is memory intensive, so we do it in chunks if needed
        total_z = len(image_paths)

        # Load full volume (this may need chunking for very large volumes)
        self.log.emit(f"Loading full volume for {name}...")
        all_images = []
        for i, path in enumerate(image_paths):
            if self.should_stop:
                return
            img = load_image(path)
            if img is not None:
                all_images.append(img)
            if (i + 1) % 100 == 0:
                self.progress.emit(name + " (loading)", i + 1, total_z)

        volume = np.stack(all_images, axis=0)
        self.log.emit(f"Volume shape: {volume.shape}")

        # Rotate
        self.log.emit(f"Rotating volume by {angle} degrees...")
        rotated = rotate(volume, angle, axes=axes, reshape=False, order=1)

        # Save slices
        self.log.emit(f"Saving {name} slices...")
        for z in range(rotated.shape[0]):
            if self.should_stop:
                break
            slice_2d = rotated[z]
            out_path = output_dir / f"{name}_slice_{z:05d}.tif"
            save_image_normalized(slice_2d, out_path)
            if (z + 1) % 100 == 0:
                self.progress.emit(name + " (saving)", z + 1, rotated.shape[0])

        del volume, rotated
        gc.collect()
