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

    def _expected_model_channels(self) -> int:
        """Input channels the current architecture's model expects (1 / 3 / 11 / 12)."""
        from ..models.architectures import (
            get_n_context_slices, uses_z_coord, is_3d_architecture)
        arch = self._arch()
        if is_3d_architecture(arch):
            return 1
        return get_n_context_slices(arch) + (1 if uses_z_coord(arch) else 0)

    def _checkpoint_compatible(self, ckpt_path: str) -> bool:
        """True if the checkpoint's first-conv input-channel count matches what the
        current architecture needs. A mismatch = a stale checkpoint (e.g. a dwarf25d
        checkpoint trained back when only 1-channel 2D crops were sent) that would
        crash load_state_dict on resume."""
        try:
            import torch
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # TrainWorker saves under "model_state"; older/foreign checkpoints may use
            # "model_state_dict". Checking only the latter made this guard inert, so a
            # channel-mismatched checkpoint was never archived and load_state_dict
            # crashed on resume instead.
            if isinstance(data, dict):
                sd = data.get("model_state") or data.get("model_state_dict") or data
            else:
                sd = data
            w = sd.get("inc.double_conv.0.weight")
            if w is None or w.dim() < 2:
                return True   # can't determine — let the worker try
            return int(w.shape[1]) == self._expected_model_channels()
        except Exception:
            return True       # if unsure, don't block

    def _archive_checkpoint(self, ckpt_path: str, tag: str):
        """Move an unusable checkpoint aside so training can start fresh."""
        p = Path(ckpt_path)
        if p.exists():
            dst = p.with_name(f"{p.stem}_{tag}.pth")
            try:
                os.replace(p, dst)
                print(f"[HubTrainer] {p.name} channel-mismatched -> archived as "
                      f"{dst.name}; training fresh")
            except OSError as e:
                print(f"[HubTrainer] could not archive {p.name}: {e}")

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
        # Resume only from a COMPATIBLE checkpoint. If the saved checkpoint's input
        # channels don't match the architecture (a stale checkpoint from an earlier
        # run with a different crop variant), archive it and start fresh instead of
        # crashing load_state_dict with a size mismatch.
        resume_ckpt = ckpt if (resume and os.path.exists(ckpt)) else None
        if resume_ckpt and not self._checkpoint_compatible(ckpt):
            self._archive_checkpoint(ckpt, "incompatible")
            resume_ckpt = None
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
            "train_images_slab": str(pool / "train_images_slab"),
            "train_masks_slab": str(pool / "train_masks_slab"),
            "checkpoint_path": ckpt,
            "architecture": self._arch(),
            "tile_size": self.hub.crop_size or 256,
            "num_epochs": self.hub.train_epochs,
            "batch_size": self.hub.train_batch_size,
            "learning_rate": self.hub.train_lr,
            "resume_checkpoint": resume_ckpt,
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
