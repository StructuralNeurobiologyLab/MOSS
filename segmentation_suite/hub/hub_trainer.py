#!/usr/bin/env python3
"""
HubTrainer — runs MOSS's TrainWorker headless on the hub's merged crop pool.

Owned by HubServer (a main-thread QObject). TrainWorker is itself a QThread, so
this class only builds config, wires the worker's signals up to the hub, and
manages start/stop/restart. No MOSS wizard/GUI is involved.

Signal flow (all cross-thread signals are auto-queued to the main thread):
    TrainWorker.loss_updated(loss, batch)      -> hub._on_loss       -> loss plot
    TrainWorker.progress(ep, tot, tr, val)     -> hub._on_train_progress -> status
    TrainWorker.weights_exported(w, ep, loss)  -> hub._on_weights_exported -> broadcast
    TrainWorker.finished(ok, msg)              -> hub._on_train_finished
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtCore import QObject

from ..workers.train_worker import TrainWorker
from ..models.unet import get_checkpoint_filename


class HubTrainer(QObject):
    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.hub = hub                       # HubServer (config + relay slots)
        self.worker: TrainWorker | None = None

    # ------------------------------------------------------------------ config
    def _arch(self) -> str:
        return self.hub.prediction_model or self.hub.architecture or "unet"

    def _model_dir(self) -> Path:
        d = self.hub.data_dir / "model"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def checkpoint_path(self) -> str:
        return str(self._model_dir() / get_checkpoint_filename(self._arch()))

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # --------------------------------------------------------------- lifecycle
    def start(self, resume: bool = True):
        if self.is_running():
            return
        if self.hub.force_cpu:
            os.environ["FORCE_CPU"] = "1"    # honored by models.unet.get_device()
        ckpt = self.checkpoint_path()
        pool = self.hub._pool_root()
        cfg = {
            # Pass ALL SIX training-dir keys, exactly like local MOSS. TrainWorker
            # derives n_channels from the architecture string and reads the matching
            # folder (train_images / _25d / _dwarf25d). NO n_channels key — the worker
            # owns that. Only the session's variant folder actually holds crops.
            "train_images": str(pool / "train_images"),
            "train_masks": str(pool / "train_masks"),
            "train_images_25d": str(pool / "train_images_25d"),
            "train_masks_25d": str(pool / "train_masks_25d"),
            "train_images_dwarf25d": str(pool / "train_images_dwarf25d"),
            "train_masks_dwarf25d": str(pool / "train_masks_dwarf25d"),
            "checkpoint_path": ckpt,
            "architecture": self._arch(),
            "tile_size": self.hub.crop_size or 256,
            "num_epochs": self.hub.train_epochs,
            "batch_size": self.hub.train_batch_size,
            "learning_rate": self.hub.train_lr,
            "resume_checkpoint": ckpt if (resume and os.path.exists(ckpt)) else None,
            "weights_export_interval": self.hub.broadcast_interval,
        }
        w = TrainWorker(cfg)
        w.loss_updated.connect(self.hub._on_loss)
        w.progress.connect(self.hub._on_train_progress)
        w.weights_exported.connect(self.hub._on_weights_exported)
        w.finished.connect(self.hub._on_train_finished)
        self.worker = w
        print(f"[HubTrainer] starting: arch={cfg['architecture']} tile={cfg['tile_size']} "
              f"resume={bool(cfg['resume_checkpoint'])} pool={cfg['train_images']}")
        w.start()

    def request_reload(self):
        """Additive: worker re-scans the pool at the end of the current epoch."""
        if self.is_running():
            self.worker.request_dataset_reload()

    def stop(self, timeout_ms: int = 30000):
        w = self.worker
        self.worker = None
        if w is not None:
            w.stop()
            w.wait(timeout_ms)   # never terminate() a CUDA/MPS thread
            time.sleep(0.3)      # let a shutdown-save flush

    def restart_resume(self):
        """Destructive pool change: stop cleanly, then resume from checkpoint."""
        self.stop()
        self.start(resume=True)
