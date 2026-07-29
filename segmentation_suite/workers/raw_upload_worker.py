#!/usr/bin/env python3
"""
RawUploadWorker — uploads the owner's local raw volume TO the hub on a background
thread (for the case where the hub can't see the owner's data), so the GUI stays
responsive with a progress bar. Wraps network.raw_transfer.upload_raw (resumable,
stdlib-only) and re-emits progress as throttled Qt signals.

    worker = RawUploadWorker(base_url, src_dir)
    worker.progress.connect(bar.update)     # (done_bytes, total_bytes, done_files, total_files)
    worker.finished_ok.connect(on_done)     # ()  -> tell the hub upload is complete
    worker.failed.connect(on_error)         # (message)  ("cancelled" if stopped)
    worker.start()
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from ..network import raw_transfer


class RawUploadWorker(QThread):
    progress = pyqtSignal(int, int, int, int)   # done_bytes, total_bytes, done_files, total_files
    finished_ok = pyqtSignal()                  # upload complete
    failed = pyqtSignal(str)                     # error message ("cancelled" if stopped)

    def __init__(self, base_url: str, src_dir, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.src_dir = str(src_dir)
        self._stop = False

    def stop(self):
        """Request an abort. Leaves a resumable partial upload on the hub."""
        self._stop = True

    def run(self):
        last = {"bytes": -1, "files": -1}

        def cb(done_bytes, total_bytes, done_files, total_files):
            step = max(1, total_bytes // 300) if total_bytes else 1
            if (done_bytes - last["bytes"] >= step or done_files != last["files"]
                    or (total_bytes and done_bytes >= total_bytes) or last["bytes"] < 0):
                last["bytes"], last["files"] = done_bytes, done_files
                self.progress.emit(int(done_bytes), int(total_bytes),
                                   int(done_files), int(total_files))

        try:
            ok = raw_transfer.upload_raw(
                self.base_url, self.src_dir,
                progress_cb=cb, should_stop=lambda: self._stop)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        if ok:
            self.finished_ok.emit()
        else:
            self.failed.emit("cancelled")
