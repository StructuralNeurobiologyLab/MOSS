#!/usr/bin/env python3
"""
Voting worker for combining multi-view predictions into a consensus heatmap.

Adapted from voting_heatmap_chunked.py
"""

import os
import gc
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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


class VotingWorker(QThread):
    """Background worker for multi-view voting/averaging."""

    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(int, int)  # current_chunk, total_chunks
    finished = pyqtSignal(bool, str)  # success, output_dir or error
    log = pyqtSignal(str)

    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary with:
                - xy_dir: Directory with XY predictions (required)
                - yz_dir: Directory with YZ predictions (optional)
                - xz_dir: Directory with XZ predictions (optional)
                - diag_dirs: List of directories with pre-rotated diagonal predictions
                - output_dir: Output directory for heatmap
                - chunk_size: int (default 200)
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

            xy_dir = Path(self.config['xy_dir'])
            yz_dir = Path(self.config['yz_dir']) if self.config.get('yz_dir') else None
            xz_dir = Path(self.config['xz_dir']) if self.config.get('xz_dir') else None
            diag_dirs = [Path(d) for d in self.config.get('diag_dirs', [])]
            output_dir = Path(self.config['output_dir'])
            chunk_size = self.config.get('chunk_size', 200)
            max_workers = self.config.get('max_workers', 8)

            output_dir.mkdir(exist_ok=True, parents=True)

            # Get XY files (use set to avoid duplicates from overlapping patterns)
            xy_files = sorted(set(
                list(xy_dir.glob("*_pred.tif")) +
                list(xy_dir.glob("*_pred.png")) +
                list(xy_dir.glob("*.tif"))
            ))
            total_z = len(xy_files)
            self.log.emit(f"Found {total_z} XY slices")

            if total_z == 0:
                self.finished.emit(False, "No XY files found")
                return

            # Get dimensions
            first_xy = load_mask_slice(xy_files[0])
            y_size, x_size = first_xy.shape
            self.log.emit(f"Dimensions: {y_size} x {x_size}")

            # Get YZ files
            yz_files = None
            if yz_dir:
                yz_files = sorted(list(yz_dir.glob("yz_slice_*_pred.tif")) +
                                 list(yz_dir.glob("yz_slice_*_pred.png")))
                if yz_files:
                    self.log.emit(f"Found {len(yz_files)} YZ slices")
                    x_size = min(x_size, len(yz_files))

            # Get XZ files
            xz_files = None
            if xz_dir:
                xz_files = sorted(list(xz_dir.glob("xz_slice_*_pred.tif")) +
                                 list(xz_dir.glob("xz_slice_*_pred.png")))
                if xz_files:
                    self.log.emit(f"Found {len(xz_files)} XZ slices")
                    y_size = min(y_size, len(xz_files))

            # Get diagonal files
            diagonal_files_list = []
            for diag_dir in diag_dirs:
                diag_files = sorted(list(diag_dir.glob("rotated_z*.tif")) +
                                   list(diag_dir.glob("*_pred.tif")))
                if diag_files:
                    self.log.emit(f"Found {len(diag_files)} diagonal slices from {diag_dir.name}")
                    diagonal_files_list.append(diag_files)

            # Process in chunks
            num_chunks = (total_z + chunk_size - 1) // chunk_size
            self.log.emit(f"Processing {num_chunks} chunks...")

            for chunk_idx in range(num_chunks):
                if self.should_stop:
                    break

                start_z = chunk_idx * chunk_size
                end_z = min((chunk_idx + 1) * chunk_size, total_z)
                chunk_len = end_z - start_z

                self.log.emit(f"Processing chunk {chunk_idx + 1}/{num_chunks} (Z[{start_z}:{end_z}])")

                # Load all volumes for this chunk
                all_volumes = []

                # XY
                xy_volume = self._load_xy_chunk(xy_files, start_z, end_z, y_size, x_size, max_workers)
                all_volumes.append(xy_volume)

                # Diagonals (pre-rotated)
                for diag_files in diagonal_files_list:
                    diag_volume = self._load_xy_chunk(diag_files, start_z, end_z, y_size, x_size, max_workers)
                    all_volumes.append(diag_volume)

                # YZ
                if yz_files:
                    yz_volume = self._load_yz_chunk(yz_files, start_z, end_z, y_size, x_size, max_workers)
                    all_volumes.append(yz_volume)

                # XZ
                if xz_files:
                    xz_volume = self._load_xz_chunk(xz_files, start_z, end_z, y_size, x_size, max_workers)
                    all_volumes.append(xz_volume)

                # Compute average heatmap
                heatmap = np.zeros((chunk_len, y_size, x_size), dtype=np.float32)
                for vol in all_volumes:
                    heatmap += vol.astype(np.float32)
                heatmap = heatmap / len(all_volumes)
                heatmap_uint8 = np.round(heatmap).astype(np.uint8)

                # Save slices
                for i in range(chunk_len):
                    z_idx = start_z + i
                    slice_2d = heatmap_uint8[i]
                    img = Image.fromarray(slice_2d)
                    out_path = output_dir / f"heatmap_z{z_idx:05d}.tif"
                    img.save(out_path, compression='tiff_deflate')

                self.progress.emit(chunk_idx + 1, num_chunks)

                # Clean up
                del all_volumes, heatmap, heatmap_uint8
                gc.collect()

            self.log.emit(f"Saved {total_z} heatmap slices to {output_dir}")
            self.finished.emit(True, str(output_dir))

        except Exception as e:
            self.log.emit(f"Voting error: {e}")
            self.finished.emit(False, str(e))

    def _load_xy_chunk(self, files, start_z, end_z, y_size, x_size, max_workers):
        """Load XY-oriented slices for a chunk."""
        chunk_len = end_z - start_z
        volume = np.zeros((chunk_len, y_size, x_size), dtype=np.uint8)

        def load_slice(i):
            img = load_mask_slice(files[start_z + i])
            if img is not None:
                return i, img[:y_size, :x_size]
            return i, np.zeros((y_size, x_size), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, img in executor.map(load_slice, range(chunk_len)):
                volume[i] = img

        return volume

    def _load_yz_chunk(self, files, start_z, end_z, y_size, x_size, max_workers):
        """Load YZ slices and rearrange for chunk."""
        chunk_len = end_z - start_z
        volume = np.zeros((chunk_len, y_size, x_size), dtype=np.uint8)

        # Check first file dimensions
        first_img = load_mask_slice(files[0])
        if first_img is not None:
            self.log.emit(f"  YZ slice shape: {first_img.shape} (expected Z={end_z} or more, Y={y_size})")

        def load_slice(x_idx):
            img = load_mask_slice(files[x_idx])
            if img is None:
                return x_idx, np.zeros((chunk_len, y_size), dtype=np.uint8)

            # Handle dimension mismatch - YZ slice should be (Z, Y)
            z_dim = img.shape[0]
            y_dim = img.shape[1] if img.ndim > 1 else 1

            # Extract the chunk, handling boundary conditions
            z_start = min(start_z, z_dim)
            z_end = min(end_z, z_dim)
            actual_chunk = z_end - z_start

            result = np.zeros((chunk_len, y_size), dtype=np.uint8)
            if actual_chunk > 0 and z_start < z_dim:
                src = img[z_start:z_end, :min(y_dim, y_size)]
                result[:src.shape[0], :src.shape[1]] = src

            return x_idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for x_idx, slice_data in executor.map(load_slice, range(min(x_size, len(files)))):
                volume[:, :, x_idx] = slice_data

        return volume

    def _load_xz_chunk(self, files, start_z, end_z, y_size, x_size, max_workers):
        """Load XZ slices and rearrange for chunk."""
        chunk_len = end_z - start_z
        volume = np.zeros((chunk_len, y_size, x_size), dtype=np.uint8)

        # Check first file dimensions
        first_img = load_mask_slice(files[0])
        if first_img is not None:
            self.log.emit(f"  XZ slice shape: {first_img.shape} (expected Z={end_z} or more, X={x_size})")

        def load_slice(y_idx):
            img = load_mask_slice(files[y_idx])
            if img is None:
                return y_idx, np.zeros((chunk_len, x_size), dtype=np.uint8)

            # Handle dimension mismatch - XZ slice should be (Z, X)
            z_dim = img.shape[0]
            x_dim = img.shape[1] if img.ndim > 1 else 1

            # Extract the chunk, handling boundary conditions
            z_start = min(start_z, z_dim)
            z_end = min(end_z, z_dim)
            actual_chunk = z_end - z_start

            result = np.zeros((chunk_len, x_size), dtype=np.uint8)
            if actual_chunk > 0 and z_start < z_dim:
                src = img[z_start:z_end, :min(x_dim, x_size)]
                result[:src.shape[0], :src.shape[1]] = src

            return y_idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for y_idx, slice_data in executor.map(load_slice, range(min(y_size, len(files)))):
                volume[:, y_idx, :] = slice_data

        return volume
