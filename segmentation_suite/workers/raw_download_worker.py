#!/usr/bin/env python3
"""
RawDownloadWorker — downloads a hub's raw volume on a background thread so the GUI
stays responsive and can show a progress bar ("Downloading data from hub").

Wraps network.raw_transfer.download_raw (resumable, stdlib-only) and re-emits its
progress as Qt signals. Progress is throttled so a multi-GB download doesn't flood
the event loop with a signal per 1 MB block.

    worker = RawDownloadWorker(base_url, dest_dir)
    worker.progress.connect(bar.update)      # (done_bytes, total_bytes, done_files, total_files)
    worker.finished_ok.connect(on_done)      # (dest_dir)  -> re-init the viewer's raw source
    worker.failed.connect(on_error)          # (message)
    worker.start()
    # worker.stop() leaves a resumable partial; starting again resumes it.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from ..network import raw_transfer


class RawDownloadWorker(QThread):
    progress = pyqtSignal(int, int, int, int)   # done_bytes, total_bytes, done_files, total_files
    finished_ok = pyqtSignal(str)               # destination path (download complete)
    failed = pyqtSignal(str)                     # error message ("cancelled" if stopped)

    def __init__(self, base_url: str, dest_dir, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.dest_dir = str(dest_dir)
        self._stop = False

    def stop(self):
        """Request an abort. Leaves a resumable partial download on disk."""
        self._stop = True

    def run(self):
        # Throttle: emit on file completion, on the first/last tick, or once the
        # byte counter advances ~0.3% — enough to animate a bar without flooding.
        last = {"bytes": -1, "files": -1}

        def cb(done_bytes, total_bytes, done_files, total_files):
            step = max(1, total_bytes // 300) if total_bytes else 1
            advanced = done_bytes - last["bytes"] >= step
            file_done = done_files != last["files"]
            complete = total_bytes and done_bytes >= total_bytes
            if advanced or file_done or complete or last["bytes"] < 0:
                last["bytes"], last["files"] = done_bytes, done_files
                self.progress.emit(int(done_bytes), int(total_bytes),
                                   int(done_files), int(total_files))

        try:
            ok = raw_transfer.download_raw(
                self.base_url, self.dest_dir,
                progress_cb=cb, should_stop=lambda: self._stop)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        if ok:
            self.finished_ok.emit(self.dest_dir)
        else:
            self.failed.emit("cancelled")
