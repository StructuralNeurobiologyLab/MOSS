#!/usr/bin/env python3
"""
Viewport prediction worker for fast predictions on visible region only.

Uses a cached model that only reloads periodically to avoid conflicts
with training writing to the checkpoint file.

Runs on CPU to avoid competing with training for GPU memory.
"""

import os
import time
import shutil
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal, QMutex


class ViewportPredictWorker(QThread):
    """Background worker for fast viewport-only predictions.

    Caches the model in memory and only reloads every RELOAD_INTERVAL seconds
    to avoid conflicts with training writing to the checkpoint.

    Uses CPU for inference to avoid GPU memory conflicts with training.
    """

    # Signals
    prediction_ready = pyqtSignal(np.ndarray, tuple)  # prediction, viewport_bounds

    # Only reload model every N seconds (avoids conflicts with training)
    # Reduced to 5 seconds for more responsive live predictions during training
    RELOAD_INTERVAL = 5.0

    # Minimum file size to consider checkpoint valid (avoid partial writes)
    MIN_CHECKPOINT_SIZE = 1000  # bytes

    def __init__(self):
        super().__init__()
        self.model = None
        self._training_active = True  # Assume training is active by default
        self._current_device_is_gpu = False
        self.device = torch.device('cpu')  # Start with CPU
        self.running = True
        self.pending_request = None
        self.mutex = QMutex()
        self.checkpoint_path = None
        self.architecture = 'unet'
        self._last_reload_time = 0
        self._last_checkpoint_mtime = 0
        self._last_checkpoint_size = 0
        self._cached_checkpoint_path = None
        self._force_reload = False
        self._consecutive_failures = 0
        self._is_25d = False
        self._is_sam2 = False
        self._n_channels = 1

        # SAM2 predictor (cached, initialized on first use)
        self._sam2_predictor = None
        self._sam2_initialized = False

        # Desired thread count for CPU inference (set in run())
        self._desired_num_threads = max(4, (os.cpu_count() or 4) // 2)

        # Check if GPU is available
        self._gpu_available = torch.cuda.is_available()

    def set_checkpoint(self, checkpoint_path: str):
        """Set the model checkpoint to use for predictions.

        Forces a reload if the path changes (not just the same file being modified).
        """
        self.mutex.lock()
        if checkpoint_path != self.checkpoint_path:
            self.checkpoint_path = checkpoint_path
            # Force reload when PATH changes - this is different from a file modification
            # which is already handled by the periodic mtime check
            self._force_reload = True
            # Reset mtime tracking since this is a new file
            self._last_checkpoint_mtime = 0
            self._last_checkpoint_size = 0
        self.mutex.unlock()

    def set_architecture(self, architecture: str):
        """Set the model architecture. Forces a model reload."""
        self.mutex.lock()
        if architecture != self.architecture:
            self.architecture = architecture
            self._is_25d = '25d' in architecture.lower()
            self._is_sam2 = 'sam2' in architecture.lower()
            # 2.5D uses 3 channels (z-3, z, z+3)
            self._n_channels = 3 if self._is_25d else 1
            self._force_reload = True
            self.model = None  # Clear cached model
            # Reset SAM2 predictor if needed (will reinitialize on next prediction)
            if self._is_sam2 and not self._sam2_initialized:
                self._sam2_predictor = None
        self.mutex.unlock()

    def set_training_active(self, active: bool):
        """Set whether training is currently active.

        When training is inactive, predictions can use GPU for better performance.
        When training is active, predictions use CPU to avoid GPU memory conflicts.
        """
        self.mutex.lock()
        if self._training_active != active:
            self._training_active = active
            # Determine target device
            if not active and self._gpu_available:
                # Training stopped - switch to GPU if available
                target_device = torch.device('cuda')
            else:
                # Training active or no GPU - use CPU
                target_device = torch.device('cpu')

            # If device needs to change, force reload
            if target_device != self.device:
                old_device = self.device
                self.device = target_device
                self._current_device_is_gpu = (target_device.type == 'cuda')
                self._force_reload = True
                self.model = None  # Clear model to force reload on new device
                print(f"Predictor: Switching from {old_device} to {target_device} (training_active={active})")
        self.mutex.unlock()

    def _should_reload_model(self) -> bool:
        """Check if we should reload the model (time-based + file changed + stable)."""
        # Force reload (e.g., architecture changed) - but still respect minimum interval
        if self._force_reload:
            now = time.time()
            # Even for force reload, wait at least 1 second between attempts
            if now - self._last_reload_time < 1.0:
                return False
            return True

        if self.checkpoint_path is None:
            return False

        if not os.path.exists(self.checkpoint_path):
            return False

        # Check if enough time has passed since last reload
        now = time.time()
        if now - self._last_reload_time < self.RELOAD_INTERVAL:
            return False

        # Check if checkpoint file has been modified and is stable
        try:
            mtime = os.path.getmtime(self.checkpoint_path)
            size = os.path.getsize(self.checkpoint_path)

            # Skip if file is too small (likely being written)
            if size < self.MIN_CHECKPOINT_SIZE:
                return False

            # Check if file has changed
            if mtime != self._last_checkpoint_mtime or size != self._last_checkpoint_size:
                # Wait a bit to ensure file write is complete
                time.sleep(0.5)
                new_size = os.path.getsize(self.checkpoint_path)
                if new_size != size:
                    # File is still being written, skip
                    return False
                return True
        except OSError:
            return False

        # First load - but back off if we've had consecutive failures
        if self.model is None:
            # Exponential backoff: wait longer with each failure
            if self._consecutive_failures > 0:
                backoff_seconds = min(30, 2 ** self._consecutive_failures)  # Max 30 seconds
                if now - self._last_reload_time < backoff_seconds:
                    return False
            return True

        return False

    def _safe_load_model(self):
        """Safely load model by copying checkpoint first to avoid read/write conflicts."""
        if self.checkpoint_path is None:
            return False

        # We'll restore thread count after loading (model loading often resets it)
        desired_threads = self._desired_num_threads

        try:
            from ..models.unet import load_model

            # Check file size before copying
            try:
                orig_size = os.path.getsize(self.checkpoint_path)
                if orig_size < self.MIN_CHECKPOINT_SIZE:
                    return False
            except OSError:
                return False

            # Copy checkpoint to temp file to avoid conflicts with training
            temp_path = self.checkpoint_path + ".predict_cache"
            try:
                shutil.copy2(self.checkpoint_path, temp_path)
                # Verify copy is complete
                copy_size = os.path.getsize(temp_path)
                if copy_size != orig_size:
                    os.remove(temp_path)
                    return False
            except (OSError, IOError) as e:
                # File might be locked by training, skip this reload
                return False

            # Load from the copy with correct architecture (always on CPU)
            self.model = load_model(temp_path, n_channels=self._n_channels, device=self.device,
                                   architecture=self.architecture)
            self._last_reload_time = time.time()
            self._last_checkpoint_mtime = os.path.getmtime(self.checkpoint_path)
            self._last_checkpoint_size = orig_size
            self._force_reload = False
            self._consecutive_failures = 0

            # Clean up temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass

            # Restore thread count (model loading may have reset it)
            if torch.get_num_threads() != desired_threads:
                torch.set_num_threads(desired_threads)

            device_info = f"{self.device}"
            if self.device.type == 'cpu':
                device_info += f" using {torch.get_num_threads()} threads"
            # Show which checkpoint file was loaded (helps debug snapshot vs main checkpoint)
            checkpoint_name = os.path.basename(self.checkpoint_path)
            print(f"Predictor: Loaded {checkpoint_name} ({self.architecture}) on {device_info}")
            return True

        except Exception as e:
            self._consecutive_failures += 1
            self._last_reload_time = time.time()  # Update for backoff calculation
            # Only print error occasionally to avoid spam
            if self._consecutive_failures <= 1:
                if "size mismatch" in str(e):
                    print(f"Predictor: Checkpoint incompatible with {self.architecture} (different input channels). "
                          f"Train with new architecture first.")
                else:
                    print(f"Failed to load model: {e}")
            self._force_reload = False
            # Clean up temp file on failure
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            return False

    def request_prediction(self, image: np.ndarray, viewport_bounds: tuple):
        """
        Queue a prediction request. Replaces any pending request.

        Args:
            image: Full image array (2D grayscale or 3D with 3 channels for 2.5D)
            viewport_bounds: (x_min, y_min, x_max, y_max) in image coordinates
        """
        self.mutex.lock()
        self.pending_request = (image, viewport_bounds)
        self.mutex.unlock()

    def is_25d(self) -> bool:
        """Check if currently using 2.5D mode."""
        return self._is_25d

    def is_sam2(self) -> bool:
        """Check if currently using SAM2 mode."""
        return self._is_sam2

    def _init_sam2_predictor(self):
        """Initialize SAM2 predictor for on-the-fly feature extraction.

        Uses CPU for viewport predictions to avoid GPU conflicts with training.

        Returns:
            SAM2ImagePredictor instance, or None if SAM2 not available
        """
        if self._sam2_initialized:
            return self._sam2_predictor

        try:
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # We'll restore thread count after loading (SAM2 loading may reset it)
            desired_threads = self._desired_num_threads

            # Use MOSS directory for SAM2 model cache (shared across all projects)
            import os
            from pathlib import Path
            cache_dir = str(Path(__file__).parent.parent.parent / "sam2_models")
            os.makedirs(cache_dir, exist_ok=True)

            # Check if model already cached
            cached_model = os.path.join(cache_dir, "MedSAM2_latest.pt")
            if os.path.exists(cached_model):
                ckpt_path = cached_model
            else:
                ckpt_path = hf_hub_download(
                    repo_id="wanglab/MedSAM2",
                    filename="MedSAM2_latest.pt",
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False,
                )

            # Build model with MedSAM2 config
            # Use same device as UNet model (CPU during training, GPU when idle)
            model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
            device_str = str(self.device)
            model = build_sam2(model_cfg, ckpt_path, device=device_str)
            self._sam2_predictor = SAM2ImagePredictor(model)
            self._sam2_initialized = True

            # Restore thread count if SAM2 reset it
            if torch.get_num_threads() != desired_threads:
                torch.set_num_threads(desired_threads)
                print(f"Restored thread count to {desired_threads} after SAM2 init")

            print(f"Viewport SAM2 predictor initialized on {device_str}")
            return self._sam2_predictor

        except ImportError as e:
            print(f"SAM2 not available for viewport predictions: {e}")
            self._sam2_initialized = True  # Mark as attempted
            return None
        except Exception as e:
            print(f"Failed to initialize viewport SAM2: {e}")
            self._sam2_initialized = True  # Mark as attempted
            return None

    def _extract_sam2_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract SAM2 features for a viewport crop on-the-fly.

        Args:
            image: Grayscale image as float32 array (H, W), normalized 0-1

        Returns:
            SAM2 features as tensor (1, 256, H/16, W/16), or None on failure
        """
        if self._sam2_predictor is None:
            return None

        try:
            # Convert normalized float to uint8 for SAM2
            img_uint8 = (image * 255).astype(np.uint8)

            # SAM expects RGB input
            rgb = np.repeat(img_uint8[..., None], 3, axis=-1)

            # Extract features
            device_str = str(self.device)
            amp_dtype = torch.bfloat16 if device_str == "cuda" else torch.float32

            with torch.inference_mode():
                with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=(device_str == "cuda")):
                    self._sam2_predictor.set_image(rgb)
                    embedding = self._sam2_predictor.get_image_embedding()  # (1, 256, Hf, Wf)

            return embedding.to(self.device)

        except Exception as e:
            # Silently fail - viewport predictions should not crash
            return None

    def stop(self):
        """Stop the worker."""
        self.running = False

    def run(self):
        """Main worker loop."""
        # Set up multi-threaded CPU inference (must be done in worker thread)
        # Wrap in try-except as these can only be called once before parallel work starts
        try:
            num_cores = os.cpu_count() or 4
            num_threads = max(4, num_cores // 2)
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass  # Already set, ignore

        while self.running:
            # Check for pending request
            self.mutex.lock()
            request = self.pending_request
            self.pending_request = None
            self.mutex.unlock()

            if request is None:
                # No request, but check if we should reload model
                if self._should_reload_model():
                    self._safe_load_model()
                self.msleep(50)
                continue

            # Check if model needs reloading (time-based, safe)
            if self._should_reload_model():
                self._safe_load_model()

            if self.model is None:
                # Try to load if we don't have a model yet
                self._safe_load_model()
                if self.model is None:
                    continue

            image, bounds = request
            x_min, y_min, x_max, y_max = bounds

            try:
                # Add padding for context
                padding = 64
                # Handle both 2D (H, W) and 3D (H, W, C) images
                h, w = image.shape[:2]
                x_min_pad = max(0, x_min - padding)
                y_min_pad = max(0, y_min - padding)
                x_max_pad = min(w, x_max + padding)
                y_max_pad = min(h, y_max + padding)

                # Crop to viewport + padding (works for both 2D and 3D)
                if image.ndim == 3:
                    cropped = image[y_min_pad:y_max_pad, x_min_pad:x_max_pad, :].copy()
                else:
                    cropped = image[y_min_pad:y_max_pad, x_min_pad:x_max_pad].copy()

                # Predict
                prediction = self._predict(cropped)

                if prediction is None:
                    continue

                # Trim padding from prediction to match original viewport
                trim_left = x_min - x_min_pad
                trim_top = y_min - y_min_pad
                trim_right = prediction.shape[1] - (x_max_pad - x_max)
                trim_bottom = prediction.shape[0] - (y_max_pad - y_max)

                prediction = prediction[trim_top:trim_bottom, trim_left:trim_right]

                # Emit result
                self.prediction_ready.emit(prediction, bounds)

            except Exception as e:
                # Silently skip failed predictions
                pass

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run prediction on a single image crop.

        Args:
            image: 2D grayscale image (H, W) or 3-channel image (H, W, 3) for 2.5D

        Returns:
            Prediction mask (uint8, 0-255) or None on failure
        """
        if self.model is None:
            return None

        try:
            img = image.astype(np.float32)
            is_multichannel = img.ndim == 3

            # Normalize
            if is_multichannel:
                # Normalize each channel independently (works for any number of channels)
                n_ch = img.shape[2]
                for c in range(n_ch):
                    ch = img[..., c]
                    ch_min, ch_max = ch.min(), ch.max()
                    if ch_max > ch_min:
                        img[..., c] = (ch - ch_min) / (ch_max - ch_min)
                    else:
                        img[..., c] = 0.0
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)

            # Pad to multiple of 64 for UNet (needed for deeper architectures with 6 levels)
            h, w = img.shape[:2]
            pad_h = (64 - h % 64) % 64
            pad_w = (64 - w % 64) % 64
            if pad_h > 0 or pad_w > 0:
                if is_multichannel:
                    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                else:
                    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

            # To tensor (on CPU)
            if is_multichannel:
                # (H, W, C) -> (N, C, H, W)
                tensor = torch.tensor(np.transpose(img, (2, 0, 1))[None, ...], dtype=torch.float32).to(self.device)
            else:
                # (H, W) -> (N, 1, H, W)
                tensor = torch.tensor(img[None, None, ...], dtype=torch.float32).to(self.device)

            # Extract SAM2 features on-the-fly if needed (only for 2D mode)
            sam2_feats = None
            if self._is_sam2 and not is_multichannel:
                # Initialize SAM2 predictor if not already done
                if not self._sam2_initialized:
                    self._init_sam2_predictor()

                # Extract features from the padded image
                sam2_feats = self._extract_sam2_features(img)

            # Predict (inference_mode is faster than no_grad)
            with torch.inference_mode():
                if sam2_feats is not None:
                    output = self.model(tensor, sam2_features=sam2_feats)
                else:
                    output = self.model(tensor)
                pred = torch.sigmoid(output)[0, 0].cpu().numpy()

            # Remove padding
            if pad_h > 0:
                pred = pred[:-pad_h, :]
            if pad_w > 0:
                pred = pred[:, :-pad_w]

            # Convert to uint8
            pred_uint8 = (pred * 255).astype(np.uint8)

            return pred_uint8

        except Exception as e:
            # Return None on any prediction error
            return None
