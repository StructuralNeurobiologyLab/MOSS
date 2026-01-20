#!/usr/bin/env python3
"""
Background image loader worker with parallel loading and LRU cache.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import os


def _load_single_image(args):
    """Load a single image (function for thread pool)."""
    idx, image_path, mask_path = args
    try:
        img = np.array(Image.open(image_path))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32)

        if mask_path and Path(mask_path).exists():
            mask = np.array(Image.open(mask_path))
            mask = mask.astype(np.uint8)
        else:
            mask = np.zeros(img.shape, dtype=np.uint8)

        return idx, img, mask, None
    except Exception as e:
        return idx, None, None, str(e)


class ImageLoaderWorker(QThread):
    """Background worker for loading images with parallel loading and LRU cache."""

    # Signals
    image_loaded = pyqtSignal(int, np.ndarray, np.ndarray)  # idx, image, mask
    loading_started = pyqtSignal(int)  # idx
    loading_failed = pyqtSignal(int, str)  # idx, error message
    batch_progress = pyqtSignal(int, int)  # loaded_count, total_count
    batch_complete = pyqtSignal()  # emitted when initial batch is done

    def __init__(self, max_workers=None, cache_size=100):
        super().__init__()
        self.running = True
        self.mutex = QMutex()
        self.pending_requests = []  # List of (idx, image_path, mask_path)
        self.priority_idx = None  # High priority index (current slice)
        self.batch_mode = False  # True during initial batch load
        self.batch_total = 0
        self.batch_loaded = 0

        # Parallel loading
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = None

        # LRU cache for images
        self.cache_size = cache_size
        self.image_cache = OrderedDict()  # {idx: (image, mask)}

    def get_cached(self, idx: int):
        """Get image from cache if available."""
        self.mutex.lock()
        if idx in self.image_cache:
            # Move to end (most recently used)
            self.image_cache.move_to_end(idx)
            img, mask = self.image_cache[idx]
            self.mutex.unlock()
            return img, mask
        self.mutex.unlock()
        return None, None

    def _add_to_cache(self, idx: int, img: np.ndarray, mask: np.ndarray):
        """Add image to cache, evicting old entries if needed."""
        self.mutex.lock()
        self.image_cache[idx] = (img, mask)
        self.image_cache.move_to_end(idx)

        # Evict oldest entries if cache too large
        while len(self.image_cache) > self.cache_size:
            self.image_cache.popitem(last=False)
        self.mutex.unlock()

    def request_load(self, idx: int, image_path: Path, mask_path: Path = None, priority: bool = False):
        """Queue an image load request."""
        # Check cache first
        img, mask = self.get_cached(idx)
        if img is not None:
            self.image_loaded.emit(idx, img, mask)
            return

        self.mutex.lock()
        request = (idx, image_path, mask_path)
        if priority:
            self.priority_idx = idx
            # Insert at front
            self.pending_requests.insert(0, request)
        else:
            # Avoid duplicates
            if request not in self.pending_requests:
                self.pending_requests.append(request)
        self.mutex.unlock()

    def clear_pending(self):
        """Clear all pending requests except priority."""
        self.mutex.lock()
        if self.priority_idx is not None:
            self.pending_requests = [r for r in self.pending_requests if r[0] == self.priority_idx]
        else:
            self.pending_requests = []
        self.mutex.unlock()

    def start_batch_load(self, requests: list, batch_size: int = 100):
        """Start loading a batch of images with parallel loading.

        Args:
            requests: List of (idx, image_path, mask_path) tuples
            batch_size: Number of images to preload initially (default 100)
        """
        self.mutex.lock()
        self.batch_mode = True
        # Only preload first batch_size images
        self.batch_total = min(batch_size, len(requests))
        self.batch_loaded = 0
        # Queue only first batch_size for initial load
        self.pending_requests = requests[:batch_size]
        # Store all requests for later preloading
        self.all_requests = requests
        self.mutex.unlock()

    def stop(self):
        """Stop the worker."""
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=False)

    def run(self):
        """Main worker loop with parallel loading."""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        while self.running:
            # Get batch of requests to process in parallel
            self.mutex.lock()
            if self.pending_requests:
                # Take up to max_workers requests at once
                batch = self.pending_requests[:self.max_workers]
                self.pending_requests = self.pending_requests[self.max_workers:]
            else:
                batch = []
            self.mutex.unlock()

            if not batch:
                self.msleep(50)
                continue

            # Submit batch to thread pool
            futures = list(self.executor.map(_load_single_image, batch))

            # Process results
            for idx, img, mask, error in futures:
                if error:
                    self.loading_failed.emit(idx, error)
                else:
                    # Add to cache
                    self._add_to_cache(idx, img, mask)
                    self.image_loaded.emit(idx, img, mask)

                    # Update batch progress
                    if self.batch_mode:
                        self.mutex.lock()
                        self.batch_loaded += 1
                        loaded = self.batch_loaded
                        total = self.batch_total
                        is_complete = (loaded >= total)
                        self.mutex.unlock()

                        self.batch_progress.emit(loaded, total)

                        if is_complete:
                            self.batch_mode = False
                            self.batch_complete.emit()

        if self.executor:
            self.executor.shutdown(wait=False)
