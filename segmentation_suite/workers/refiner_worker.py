#!/usr/bin/env python3
"""
Refiner Worker - background training and prediction for the refiner model.

Learns from user edits in real-time and provides refined predictions.
Loads training data from refiner folders and continuously trains.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from pathlib import Path
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import time


class RefinerWorker(QThread):
    """
    Background worker that trains a refiner model on user edits
    and provides predictions.

    Loads triplets from refiner folders: (raw_image, mask_before, mask_after)
    and continuously trains/predicts.
    """

    # Signals
    training_started = pyqtSignal()
    training_progress = pyqtSignal(int, int, float)  # epoch, total_epochs, loss
    training_complete = pyqtSignal()
    prediction_ready = pyqtSignal(np.ndarray, tuple)  # prediction, bounds

    # Configuration
    EPOCHS_PER_CYCLE = 5000   # Epochs per training cycle (matches main training)
    TRAIN_INTERVAL = 5.0      # Seconds between training cycles
    PATCH_SIZE = 512          # Patch size for training
    LEARNING_RATE = 1e-4      # Standard learning rate

    def __init__(self, project_dir: Optional[Path] = None):
        super().__init__()
        self.project_dir = project_dir

        # Folder paths
        self.refiner_images_dir = None
        self.refiner_masks_before_dir = None
        self.refiner_masks_after_dir = None
        if project_dir:
            self.refiner_images_dir = project_dir / 'refiner_images'
            self.refiner_masks_before_dir = project_dir / 'refiner_masks_before'
            self.refiner_masks_after_dir = project_dir / 'refiner_masks_after'

        # Model and training state
        self.model = None  # Training model (on GPU)
        self.model_cpu = None  # Prediction model (on CPU, copy of weights)
        self.optimizer = None
        self.device = None  # Training device (GPU if available)
        self.cpu_device = torch.device("cpu")  # Prediction device (always CPU)
        self._model_initialized = False
        self._weights_updated = False  # Flag to sync CPU model

        # Training state
        self._last_train_time = 0
        self._training_in_progress = False
        self._last_file_count = 0  # Track if new files added

        # Prediction state
        self._prediction_request = None  # (raw_image, mask, bounds)
        self._prediction_pending = False

        # Thread control
        self._running = False
        self._mutex = QMutex()
        self._condition = QWaitCondition()

    def _init_model(self):
        """Initialize the refiner model."""
        if self._model_initialized:
            return

        from ..models.refiner import RefinerUNet, get_device

        self.device = get_device()
        print(f"[Refiner] Training device: {self.device}")

        # Training model on GPU
        self.model = RefinerUNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

        # Prediction model on CPU (for parallel inference during training)
        self.model_cpu = RefinerUNet().to(self.cpu_device)
        self._model_initialized = True

        # Try to load existing checkpoint
        if self.project_dir:
            checkpoint_path = self.project_dir / 'refiner_checkpoint.pth'
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state'])
                    if 'optimizer_state' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print(f"[Refiner] Loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    print(f"[Refiner] Failed to load checkpoint: {e}")

        # Sync CPU model with initial weights
        self._sync_cpu_model()

    def _sync_cpu_model(self):
        """Copy weights from GPU model to CPU model for prediction."""
        if self.model is None or self.model_cpu is None:
            return
        # Copy state dict from GPU model to CPU model
        state_dict = self.model.state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        self.model_cpu.load_state_dict(cpu_state_dict)
        self._weights_updated = False

    def add_edit(self, raw_image: np.ndarray, mask_before: np.ndarray, mask_after: np.ndarray):
        """Add an edit - now just triggers a wake to check for new files."""
        self._mutex.lock()
        self._condition.wakeAll()
        self._mutex.unlock()

    def request_prediction(self, raw_image: np.ndarray, mask: np.ndarray, bounds: tuple):
        """
        Request a prediction for the given viewport.

        Args:
            raw_image: The raw grayscale image
            mask: Current mask state
            bounds: Viewport bounds (x_min, y_min, x_max, y_max)
        """
        self._mutex.lock()
        self._prediction_request = (raw_image.copy(), mask.copy(), bounds)
        self._prediction_pending = True
        self._condition.wakeAll()
        self._mutex.unlock()

    def _load_training_data_from_folders(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load training triplets from refiner folders."""
        if not self.refiner_images_dir or not self.refiner_images_dir.exists():
            return []

        triplets = []
        image_files = sorted(self.refiner_images_dir.glob("*.tif"))

        for img_path in image_files:
            try:
                before_path = self.refiner_masks_before_dir / img_path.name
                after_path = self.refiner_masks_after_dir / img_path.name

                if not before_path.exists() or not after_path.exists():
                    print(f"[Refiner] Missing mask for {img_path.name}")
                    continue

                raw_image = np.array(Image.open(img_path)).astype(np.float32)
                mask_before = np.array(Image.open(before_path))
                mask_after = np.array(Image.open(after_path))

                # Debug: print what we loaded
                print(f"[Refiner] Loading {img_path.name}:")
                print(f"  before_path: {before_path}")
                print(f"  after_path: {after_path}")
                print(f"  before shape={mask_before.shape}, dtype={mask_before.dtype}, unique={np.unique(mask_before)[:5]}")
                print(f"  after shape={mask_after.shape}, dtype={mask_after.dtype}, unique={np.unique(mask_after)[:5]}")

                # Handle multi-channel images
                if raw_image.ndim == 3:
                    raw_image = raw_image.mean(axis=-1)
                if mask_before.ndim == 3:
                    mask_before = mask_before[:, :, 0]
                if mask_after.ndim == 3:
                    mask_after = mask_after[:, :, 0]

                mask_before = mask_before.astype(np.uint8)
                mask_after = mask_after.astype(np.uint8)

                # Check if before and after are actually different
                diff = np.sum(mask_before != mask_after)
                before_sum = np.sum(mask_before > 127)
                after_sum = np.sum(mask_after > 127)
                print(f"[Refiner] {img_path.name}: before={before_sum}px, after={after_sum}px, diff={diff}px")

                # SKIP samples with no meaningful difference!
                # These teach the model to predict "no changes" = copy input
                if diff < 100:
                    print(f"[Refiner] SKIPPING {img_path.name} - diff too small ({diff}px)")
                    continue

                triplets.append((raw_image, mask_before, mask_after))

            except Exception as e:
                print(f"[Refiner] Failed to load {img_path.name}: {e}")

        return triplets

    def get_edit_count(self) -> int:
        """Get the number of training samples in folders."""
        if not self.refiner_images_dir or not self.refiner_images_dir.exists():
            return 0
        return len(list(self.refiner_images_dir.glob("*.tif")))

    def run(self):
        """Main worker loop - continuously trains and predicts."""
        self._running = True
        self._init_model()

        print("[Refiner] Worker started - continuous training mode")

        while self._running:
            self._mutex.lock()

            # Handle prediction request first (higher priority)
            if self._prediction_pending:
                request = self._prediction_request
                self._prediction_request = None
                self._prediction_pending = False
                self._mutex.unlock()

                if request is not None:
                    self._do_prediction(*request)
                continue

            # Check if we should train
            time_since_train = time.time() - self._last_train_time
            should_train = time_since_train > self.TRAIN_INTERVAL and not self._training_in_progress

            if should_train:
                self._training_in_progress = True
                self._mutex.unlock()

                # Load data from folders and train
                triplets = self._load_training_data_from_folders()
                if triplets:
                    print(f"[Refiner] Training on {len(triplets)} samples from folders")
                    self._do_training(triplets)
                    self._last_train_time = time.time()

                self._mutex.lock()
                self._training_in_progress = False
                self._mutex.unlock()
                continue

            # Wait for work or timeout
            self._condition.wait(self._mutex, int(self.TRAIN_INTERVAL * 1000))
            self._mutex.unlock()

    def _do_training(self, edit_pairs: list):
        """Run a training cycle on the edit pairs."""
        if not edit_pairs:
            return

        self.training_started.emit()

        from ..models.refiner import EditPairDataset

        # Create dataset and dataloader
        dataset = EditPairDataset(edit_pairs, patch_size=self.PATCH_SIZE)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=0
        )

        self.model.train()

        for epoch in range(self.EPOCHS_PER_CYCLE):
            if not self._running:
                break

            # Check for pending prediction every epoch - handle it mid-training
            self._mutex.lock()
            if self._prediction_pending:
                request = self._prediction_request
                self._prediction_request = None
                self._prediction_pending = False
                self._mutex.unlock()
                if request is not None:
                    self._do_prediction(*request)
                self.model.train()  # Back to training mode
            else:
                self._mutex.unlock()

            epoch_loss = 0.0
            num_batches = 0

            for inputs, targets, change_mask in dataloader:
                if not self._running:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                change_mask = change_mask.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                # Masked loss - ONLY train on pixels that changed
                # This forces the model to learn what to do with changes, not just predict zeros
                loss_per_pixel = nn.functional.binary_cross_entropy_with_logits(
                    outputs, targets, reduction='none'
                )

                # Expand change_mask to match outputs shape [B, 2, H, W]
                change_mask_expanded = change_mask.expand_as(loss_per_pixel)

                # Only count loss where changes occurred
                masked_loss = loss_per_pixel * change_mask_expanded
                num_changed = change_mask_expanded.sum() + 1e-8  # avoid div by zero
                loss = masked_loss.sum() / num_changed

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            self.training_progress.emit(epoch + 1, self.EPOCHS_PER_CYCLE, avg_loss)

            # Sync CPU model every 10 epochs for updated predictions
            if (epoch + 1) % 10 == 0:
                self._sync_cpu_model()

        # Final sync after training
        self._sync_cpu_model()

        # Save checkpoint
        if self.project_dir:
            self._save_checkpoint()

        self.training_complete.emit()

    def _do_prediction(self, raw_image: np.ndarray, mask: np.ndarray, bounds: tuple):
        """Generate a prediction for the viewport."""
        if not self._model_initialized or self.model is None:
            return

        from ..models.refiner import predict_refinement

        x_min, y_min, x_max, y_max = bounds
        h, w = raw_image.shape

        # Clamp bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Extract viewport region (with some padding)
        pad = 32
        vx_min = max(0, x_min - pad)
        vy_min = max(0, y_min - pad)
        vx_max = min(w, x_max + pad)
        vy_max = min(h, y_max + pad)

        viewport_raw = raw_image[vy_min:vy_max, vx_min:vx_max]
        viewport_mask = mask[vy_min:vy_max, vx_min:vx_max]

        # Predict using CPU model (parallel to GPU training)
        try:
            print(f"[Refiner] Making prediction on viewport {vx_min},{vy_min} to {vx_max},{vy_max}")
            prediction = predict_refinement(
                self.model_cpu, viewport_raw, viewport_mask,
                self.cpu_device, patch_size=self.PATCH_SIZE, overlap=32
            )

            # Place back in full image coordinates
            full_prediction = np.zeros((h, w), dtype=np.uint8)
            full_prediction[vy_min:vy_max, vx_min:vx_max] = prediction

            pred_pixels = np.sum(full_prediction > 127)
            print(f"[Refiner] Prediction ready: {pred_pixels} foreground pixels")
            self.prediction_ready.emit(full_prediction, bounds)
        except Exception as e:
            import traceback
            print(f"[Refiner] Prediction error: {e}")
            traceback.print_exc()

    def _save_checkpoint(self):
        """Save the model checkpoint."""
        if not self.project_dir or not self._model_initialized:
            return

        checkpoint_path = self.project_dir / 'refiner_checkpoint.pth'
        try:
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"[Refiner] Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"[Refiner] Failed to save checkpoint: {e}")

    def stop(self):
        """Stop the worker thread."""
        self._mutex.lock()
        self._running = False
        self._condition.wakeAll()
        self._mutex.unlock()
        self.wait()

    def is_training(self) -> bool:
        """Check if training is in progress."""
        self._mutex.lock()
        result = self._training_in_progress
        self._mutex.unlock()
        return result

    def clear_edits(self):
        """Clear refiner training data folders."""
        if self.refiner_images_dir and self.refiner_images_dir.exists():
            for f in self.refiner_images_dir.glob("*.tif"):
                try:
                    f.unlink()
                except:
                    pass
        if self.refiner_masks_before_dir and self.refiner_masks_before_dir.exists():
            for f in self.refiner_masks_before_dir.glob("*.tif"):
                try:
                    f.unlink()
                except:
                    pass
        if self.refiner_masks_after_dir and self.refiner_masks_after_dir.exists():
            for f in self.refiner_masks_after_dir.glob("*.tif"):
                try:
                    f.unlink()
                except:
                    pass
        print("[Refiner] Cleared training data folders")

    def reset_model(self):
        """Reset the model to untrained state and delete checkpoint."""
        self._mutex.lock()

        # Delete checkpoint file
        if self.project_dir:
            checkpoint_path = self.project_dir / 'refiner_checkpoint.pth'
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    print(f"[Refiner] Deleted checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"[Refiner] Failed to delete checkpoint: {e}")

        # Reinitialize model with fresh weights
        if self._model_initialized:
            from ..models.refiner import RefinerUNet
            self.model = RefinerUNet().to(self.device)
            self.model_cpu = RefinerUNet().to(self.cpu_device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
            self._sync_cpu_model()
            print("[Refiner] Model reset to fresh weights")

        # Reset training timer so it starts fresh
        self._last_train_time = 0

        self._mutex.unlock()
