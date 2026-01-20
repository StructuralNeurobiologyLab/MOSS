#!/usr/bin/env python3
"""
Background file saver worker to prevent UI freezing during saves.
"""

import numpy as np
from pathlib import Path
import tifffile
from PyQt6.QtCore import QThread, QMutex


class FileSaverWorker(QThread):
    """Background worker for saving files without blocking UI."""

    def __init__(self):
        super().__init__()
        self.running = True
        self.mutex = QMutex()
        self.pending_saves = []  # List of (path, array) tuples

    def queue_save(self, path: Path, array: np.ndarray):
        """Queue a file save operation."""
        self.mutex.lock()
        # Make a copy to avoid issues with array being modified
        self.pending_saves.append((path, array.copy()))
        self.mutex.unlock()

    def stop(self):
        """Stop the worker."""
        self.running = False

    def run(self):
        """Main worker loop."""
        while self.running or self.pending_saves:
            # Get next save request
            self.mutex.lock()
            if self.pending_saves:
                path, array = self.pending_saves.pop(0)
            else:
                path, array = None, None
            self.mutex.unlock()

            if path is None:
                if not self.running:
                    break
                self.msleep(50)
                continue

            try:
                tifffile.imwrite(str(path), array, compression='lzw')
            except Exception as e:
                print(f"Failed to save {path}: {e}")
