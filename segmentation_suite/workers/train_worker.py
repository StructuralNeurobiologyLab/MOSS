#!/usr/bin/env python3
"""
Training worker for running UNet training from scratch in a background thread.

Adapted from train_unet.py
"""

import os
import random
import numpy as np
from tqdm import tqdm
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PyQt6.QtCore import QThread, pyqtSignal


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape (N, 1, H, W)
            targets: Ground truth masks, shape (N, 1, H, W)
        """
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def get_loss_function(loss_type: str) -> nn.Module:
    """
    Get the loss function by name.

    Args:
        loss_type: 'bce', 'dice', or 'bce_dice'

    Returns:
        Loss function module
    """
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce_dice':
        return BCEDiceLoss()
    else:  # Default to BCE
        return nn.BCEWithLogitsLoss()


class NucleiPatchDataset(Dataset):
    """Dataset for training with balanced patch sampling."""

    def __init__(self, img_dir: str, mask_dir: str, tile: int = 512, fg_ratio: float = 0.5,
                 n_channels: int = 1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.tile = tile
        self.fg_ratio = fg_ratio
        self.n_channels = n_channels  # 1 for 2D, 3 for 2.5D

        # Find valid pairs
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
        ])

        self._img_cache = {}
        self._mask_cache = {}
        self._positive_pixels = {}
        self._load_all()

    def _load_all(self):
        """Load and cache all images."""
        print(f"Loading {len(self.images)} image-mask pairs (n_channels={self.n_channels})...")
        for fname in self.images:
            ip = os.path.join(self.img_dir, fname)
            mp = os.path.join(self.mask_dir, fname)

            if not os.path.exists(ip) or not os.path.exists(mp):
                continue

            try:
                img = io.imread(ip).astype(np.float32)
                mask = io.imread(mp).astype(np.float32)
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
                continue

            # Handle multi-channel images (2.5D)
            if self.n_channels > 1:
                # Files may be saved as (C, H, W) by tifffile, need (H, W, C)
                if img.ndim == 2:
                    # Single channel - replicate to n_channels
                    img = np.stack([img] * self.n_channels, axis=-1)
                elif img.ndim == 3:
                    # Check if shape is (C, H, W) and transpose to (H, W, C)
                    if img.shape[0] == self.n_channels and img.shape[0] < img.shape[1]:
                        # Likely (C, H, W) format - transpose
                        img = np.transpose(img, (1, 2, 0))
                    elif img.shape[-1] != self.n_channels:
                        # Wrong number of channels - skip or adapt
                        if img.shape[-1] > self.n_channels:
                            img = img[..., :self.n_channels]
                        else:
                            # Pad with repeated last channel
                            pad = self.n_channels - img.shape[-1]
                            img = np.concatenate([img] + [img[..., -1:]] * pad, axis=-1)
            else:
                # Single channel (2D)
                if img.ndim == 3:
                    img = img.mean(axis=-1)

            mask = (mask > 0.5).astype(np.float32)

            self._img_cache[ip] = img
            self._mask_cache[mp] = mask

            pos = np.argwhere(mask > 0)
            if len(pos) > 0:
                self._positive_pixels[fname] = pos

        print(f"Cached {len(self._img_cache)} pairs successfully.")

    def __len__(self):
        # Linux: 50x for thorough training, macOS: 10x for faster interactive feedback
        import platform
        multiplier = 50 if platform.system() == 'Linux' else 10
        return len(self.images) * multiplier

    def _random_crop(self, img, mask, y, x):
        t = self.tile
        # Handle both 2D (H, W) and 3D (H, W, C) images
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            h, w = img.shape
            c = None

        y1, x1 = min(y + t, h), min(x + t, w)
        patch_img = img[y:y1, x:x1]
        patch_msk = mask[y:y1, x:x1]

        # Check if padding is needed
        if patch_img.shape[0] != t or patch_img.shape[1] != t:
            pad_y, pad_x = t - patch_img.shape[0], t - patch_img.shape[1]
            if c is not None:
                # 3D: pad only H and W dimensions
                patch_img = np.pad(patch_img, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
            else:
                patch_img = np.pad(patch_img, ((0, pad_y), (0, pad_x)), mode="reflect")
            patch_msk = np.pad(patch_msk, ((0, pad_y), (0, pad_x)), mode="reflect")
        return patch_img, patch_msk

    def _augment(self, img, mask):
        # These operations work on both 2D (H, W) and 3D (H, W, C) arrays
        # flipud/fliplr operate on axis 0/1, rot90 operates on axes (0, 1) by default
        if random.random() < 0.5:
            img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
        if random.random() < 0.5:
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        k = random.randint(0, 3)
        if k:
            # Explicitly specify axes for 3D compatibility
            img = np.rot90(img, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        if random.random() < 0.3:
            img = np.clip(img * random.uniform(0.8, 1.2) +
                          random.uniform(-0.1, 0.1), 0, 1).copy()
        return img, mask

    def __getitem__(self, _):
        fname = random.choice(self.images)
        ip = os.path.join(self.img_dir, fname)
        mp = os.path.join(self.mask_dir, fname)

        if ip not in self._img_cache:
            # Return correct shape based on n_channels
            return (torch.zeros((self.n_channels, self.tile, self.tile)),
                    torch.zeros((1, self.tile, self.tile)))

        img = self._img_cache[ip]
        mask = self._mask_cache[mp]

        # Get spatial dimensions (works for both 2D and 3D)
        h, w = img.shape[:2]
        t = self.tile

        if fname in self._positive_pixels and random.random() < self.fg_ratio:
            y, x = random.choice(self._positive_pixels[fname])
            y = max(0, min(y - t // 2, h - t))
            x = max(0, min(x - t // 2, w - t))
        else:
            y = random.randint(0, max(0, h - t))
            x = random.randint(0, max(0, w - t))

        patch_img, patch_msk = self._random_crop(img, mask, y, x)
        m, M = patch_img.min(), patch_img.max()
        patch_img = (patch_img - m) / (M - m + 1e-8)
        patch_img, patch_msk = self._augment(patch_img, patch_msk)

        # Convert to tensor with correct shape (C, H, W)
        if patch_img.ndim == 3:
            # 3D: (H, W, C) -> (C, H, W)
            patch_img = torch.tensor(np.transpose(patch_img.copy(), (2, 0, 1)), dtype=torch.float32)
        else:
            # 2D: (H, W) -> (1, H, W)
            patch_img = torch.tensor(patch_img.copy()[None, ...], dtype=torch.float32)

        patch_msk = torch.tensor(patch_msk.copy()[None, ...], dtype=torch.float32)
        return patch_img, patch_msk


class NucleiPatchDatasetSAM2(Dataset):
    """Dataset for training with SAM2 features."""

    def __init__(self, img_dir: str, mask_dir: str, sam2_dir: str,
                 tile: int = 256, fg_ratio: float = 0.5, n_channels: int = 1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.sam2_dir = sam2_dir
        self.tile = tile
        self.fg_ratio = fg_ratio
        self.n_channels = n_channels

        # Find valid pairs
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
        ])

        self._img_cache = {}
        self._mask_cache = {}
        self._sam2_cache = {}
        self._positive_pixels = {}

        # Track augmentation state for syncing SAM2 features
        self._last_flipud = False
        self._last_fliplr = False
        self._last_k = 0

        self._load_all()

    def _load_all(self):
        """Load and cache all images and SAM2 features."""
        print(f"Loading {len(self.images)} image-mask pairs with SAM2 features...")
        loaded_sam2 = 0

        for fname in self.images:
            ip = os.path.join(self.img_dir, fname)
            mp = os.path.join(self.mask_dir, fname)
            stem = os.path.splitext(fname)[0]
            sp = os.path.join(self.sam2_dir, f"{stem}.npy")

            if not os.path.exists(ip) or not os.path.exists(mp):
                continue

            try:
                img = io.imread(ip).astype(np.float32)
                mask = io.imread(mp).astype(np.float32)
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
                continue

            # Handle image channels
            if self.n_channels > 1:
                if img.ndim == 2:
                    img = np.stack([img] * self.n_channels, axis=-1)
                elif img.ndim == 3:
                    if img.shape[0] == self.n_channels and img.shape[0] < img.shape[1]:
                        img = np.transpose(img, (1, 2, 0))
                    elif img.shape[-1] != self.n_channels:
                        if img.shape[-1] > self.n_channels:
                            img = img[..., :self.n_channels]
                        else:
                            pad = self.n_channels - img.shape[-1]
                            img = np.concatenate([img] + [img[..., -1:]] * pad, axis=-1)
            else:
                if img.ndim == 3:
                    img = img.mean(axis=-1)

            mask = (mask > 0.5).astype(np.float32)

            self._img_cache[ip] = img
            self._mask_cache[mp] = mask

            # Load SAM2 features if available
            if os.path.exists(sp):
                self._sam2_cache[fname] = np.load(sp).astype(np.float32)  # (256, 16, 16)
                loaded_sam2 += 1

            pos = np.argwhere(mask > 0)
            if len(pos) > 0:
                self._positive_pixels[fname] = pos

        print(f"Cached {len(self._img_cache)} pairs, {loaded_sam2} SAM2 features")

    def __len__(self):
        # Linux: 50x for thorough training, macOS: 10x for faster interactive feedback
        import platform
        multiplier = 50 if platform.system() == 'Linux' else 10
        return len(self.images) * multiplier

    def _augment(self, img, mask):
        """Apply geometric augmentations and track state for SAM2 sync."""
        self._last_flipud = random.random() < 0.5
        self._last_fliplr = random.random() < 0.5
        self._last_k = random.randint(0, 3)

        if self._last_flipud:
            img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
        if self._last_fliplr:
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        if self._last_k:
            img = np.rot90(img, self._last_k, axes=(0, 1)).copy()
            mask = np.rot90(mask, self._last_k, axes=(0, 1)).copy()

        # Intensity augmentation (only for image)
        if random.random() < 0.3:
            img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1).copy()

        return img, mask

    def _augment_sam2(self, sam2_feat):
        """Apply same geometric augmentations to SAM2 features."""
        # SAM2 features are (C, H, W), apply on spatial dims (1, 2)
        if self._last_flipud:
            sam2_feat = np.flip(sam2_feat, axis=1).copy()
        if self._last_fliplr:
            sam2_feat = np.flip(sam2_feat, axis=2).copy()
        if self._last_k:
            sam2_feat = np.rot90(sam2_feat, self._last_k, axes=(1, 2)).copy()
        return sam2_feat

    def __getitem__(self, _):
        fname = random.choice(self.images)
        ip = os.path.join(self.img_dir, fname)
        mp = os.path.join(self.mask_dir, fname)

        if ip not in self._img_cache:
            # Return zeros with correct shapes
            zeros_img = torch.zeros((self.n_channels, self.tile, self.tile))
            zeros_mask = torch.zeros((1, self.tile, self.tile))
            zeros_sam2 = torch.zeros((256, self.tile // 16, self.tile // 16))
            return zeros_img, zeros_mask, zeros_sam2

        img = self._img_cache[ip]
        mask = self._mask_cache[mp]

        # Training tiles are already the right size (256x256)
        # Just normalize and augment
        m, M = img.min(), img.max()
        img = (img - m) / (M - m + 1e-8)

        img, mask = self._augment(img, mask)

        # Convert image to tensor (C, H, W)
        if img.ndim == 3:
            patch_img = torch.tensor(np.transpose(img.copy(), (2, 0, 1)), dtype=torch.float32)
        else:
            patch_img = torch.tensor(img.copy()[None, ...], dtype=torch.float32)

        patch_msk = torch.tensor(mask.copy()[None, ...], dtype=torch.float32)

        # Get SAM2 features (apply same augmentations)
        if fname in self._sam2_cache:
            sam2_feat = self._sam2_cache[fname].copy()
            sam2_feat = self._augment_sam2(sam2_feat)
            sam2_feat = torch.tensor(sam2_feat, dtype=torch.float32)
        else:
            # No SAM2 features - return zeros
            sam2_feat = torch.zeros((256, self.tile // 16, self.tile // 16))

        return patch_img, patch_msk, sam2_feat


class TrainWorker(QThread):
    """Background worker for UNet training."""

    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(int, int, float, float)  # epoch, total_epochs, train_loss, val_loss
    finished = pyqtSignal(bool, str)  # success, checkpoint_path or error message
    log = pyqtSignal(str)  # log message
    model_updated = pyqtSignal(str)  # checkpoint_path - emitted after each epoch
    sam2_extraction_progress = pyqtSignal(int, int, str)  # current, total, message
    weights_exported = pyqtSignal(dict, int, float)  # weights, epoch, loss - for multi-user sync

    # New signals for multi-user redesign
    snapshot_created = pyqtSignal(dict, int)  # weights, snapshot_id - time-based snapshot for predictions
    loss_updated = pyqtSignal(float, int)  # loss_value, batch_number - for live plot

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.should_stop = False
        self.is_paused = False
        self._reload_requested = False
        self._train_ds = None

        # Multi-user sync settings
        self.weights_export_interval = config.get('weights_export_interval', 5)  # Export every N epochs
        self._last_export_epoch = -1

        # Time-based snapshot settings (for multi-user redesign)
        self.snapshot_interval = config.get('snapshot_interval', 30.0)  # seconds
        self._last_snapshot_time = 0.0
        self._snapshot_id = 0
        self._snapshot_checkpoint_path = None  # Path for snapshot checkpoint

        # Hot-reload support for training data from clients
        self._pending_training_data = []  # List of (image_bytes, mask_bytes, metadata) tuples
        self._pending_data_lock = None  # Will be initialized in run()

    def stop(self):
        """Request the worker to stop."""
        self.should_stop = True

    def pause(self):
        """Pause training."""
        self.is_paused = True

    def resume(self):
        """Resume training."""
        self.is_paused = False

    def request_dataset_reload(self):
        """Request the dataset to be reloaded (e.g., when new files are added)."""
        self._reload_requested = True

    def set_snapshot_interval(self, interval: float):
        """
        Set the interval for creating model snapshots (for multi-user predictions).

        Args:
            interval: Create snapshot every N seconds. Set to 0 to disable.
        """
        self.snapshot_interval = interval

    def add_training_data(self, image_bytes: bytes, mask_bytes: bytes, metadata: dict):
        """
        Add training data received from a client (host only).

        This data will be hot-loaded into the dataset during training.

        Args:
            image_bytes: PNG/TIFF encoded image data
            mask_bytes: PNG/TIFF encoded mask data
            metadata: Dict with sender_id, crop_size, slice_index, timestamp
        """
        if self._pending_data_lock:
            import threading
            with self._pending_data_lock:
                self._pending_training_data.append((image_bytes, mask_bytes, metadata))
        else:
            self._pending_training_data.append((image_bytes, mask_bytes, metadata))

    def set_weights_export_interval(self, interval: int):
        """
        Set the interval for exporting weights (for multi-user sync).

        Args:
            interval: Export weights every N epochs. Set to 0 to disable.
        """
        self.weights_export_interval = interval

    def _extract_sam2_features(self, train_images_dir: str, sam2_dir: str):
        """
        Extract SAM2 features for all training images.

        This is called automatically when training SAM2 architecture
        for the first time (or when features are missing).

        Uses parallel image loading and GPU inference for speed.
        """
        import time
        from pathlib import Path
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue
        import threading

        train_path = Path(train_images_dir)
        sam2_path = Path(sam2_dir)
        sam2_path.mkdir(exist_ok=True)

        # Find all training images
        image_files = sorted([
            f for f in train_path.iterdir()
            if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')
        ])

        # Check which need extraction
        to_extract = [
            f for f in image_files
            if not (sam2_path / f"{f.stem}.npy").exists()
        ]

        total_images = len(image_files)
        total_to_extract = len(to_extract)
        already_done = total_images - total_to_extract

        self.log.emit(f"SAM2 features: {already_done}/{total_images} already extracted")

        if not to_extract:
            self.log.emit("All SAM2 features ready - starting training")
            # Don't emit any progress signal - just continue to training
            return True

        self.log.emit(f"Extracting SAM2 features for {total_to_extract} new images...")
        self.sam2_extraction_progress.emit(0, total_to_extract, "Initializing SAM2...")

        try:
            # Import SAM2
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

            # Use MOSS directory for SAM2 model cache (shared across all projects)
            import os
            from pathlib import Path
            cache_dir = str(Path(__file__).parent.parent.parent / "sam2_models")
            os.makedirs(cache_dir, exist_ok=True)

            # Check if model already cached
            cached_model = os.path.join(cache_dir, "MedSAM2_latest.pt")
            if os.path.exists(cached_model):
                self.sam2_extraction_progress.emit(0, total_to_extract, f"Loading cached SAM2 on {device_name}...")
                self.log.emit(f"Using cached SAM2 model: {cached_model}")
                ckpt_path = cached_model
            else:
                self.sam2_extraction_progress.emit(0, total_to_extract, f"Downloading SAM2 model (one-time, ~300MB)...")
                self.log.emit("Downloading MedSAM2 model (first time only)...")
                ckpt_path = hf_hub_download(
                    repo_id="wanglab/MedSAM2",
                    filename="MedSAM2_latest.pt",
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False,
                )
                self.log.emit(f"Model cached at: {ckpt_path}")

            self.sam2_extraction_progress.emit(0, total_to_extract, f"Building SAM2 model on {device_name}...")

            model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
            model = build_sam2(model_cfg, ckpt_path, device=device)
            predictor = SAM2ImagePredictor(model)

            self.sam2_extraction_progress.emit(0, total_to_extract, f"SAM2 ready on {device_name}! Starting extraction...")
            self.log.emit(f"SAM2 loaded on {device_name}")

            # Parallel image loading with a prefetch queue
            def load_image(path):
                """Load and prepare image for SAM2."""
                try:
                    img = Image.open(path)
                    gray = np.array(img).astype(np.uint8)
                    if gray.ndim == 3:
                        gray = gray.mean(axis=-1).astype(np.uint8)
                    rgb = np.repeat(gray[..., None], 3, axis=-1)
                    return path, rgb, None
                except Exception as e:
                    return path, None, str(e)

            # Prefetch queue - load images in parallel while GPU works
            prefetch_queue = queue.Queue(maxsize=8)
            stop_prefetch = threading.Event()

            def prefetch_worker():
                """Background thread to prefetch images."""
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for path in to_extract:
                        if stop_prefetch.is_set():
                            break
                        future = executor.submit(load_image, path)
                        prefetch_queue.put(future.result())

            # Start prefetch thread
            prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
            prefetch_thread.start()

            # Extract features with timing
            amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            start_time = time.time()
            processed = 0

            for i in range(total_to_extract):
                if self.should_stop:
                    stop_prefetch.set()
                    self.log.emit("SAM2 extraction stopped by user")
                    return False

                # Get preloaded image from queue
                try:
                    image_path, rgb, error = prefetch_queue.get(timeout=30)
                except queue.Empty:
                    self.log.emit("Image loading timeout")
                    continue

                if error:
                    self.log.emit(f"Error loading {image_path.name}: {error}")
                    continue

                # Calculate ETA
                elapsed = time.time() - start_time
                if processed > 0:
                    avg_time = elapsed / processed
                    remaining = (total_to_extract - processed) * avg_time
                    if remaining >= 60:
                        eta_str = f"~{remaining / 60:.1f} min remaining"
                    else:
                        eta_str = f"~{remaining:.0f}s remaining"
                else:
                    eta_str = "calculating..."

                self.sam2_extraction_progress.emit(
                    i + 1, total_to_extract,
                    f"[{device_name}] {image_path.name} | {eta_str}"
                )

                try:
                    # Extract features on GPU
                    with torch.inference_mode():
                        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                            predictor.set_image(rgb)
                            embedding = predictor.get_image_embedding()

                    # Save as float16
                    features = embedding.squeeze(0).to(torch.float16).cpu().numpy()
                    np.save(sam2_path / f"{image_path.stem}.npy", features)
                    processed += 1

                except Exception as e:
                    self.log.emit(f"Error extracting features for {image_path.name}: {e}")

            # Cleanup
            stop_prefetch.set()

            total_time = time.time() - start_time
            self.log.emit(f"SAM2 features extracted in {total_time:.1f}s ({processed} images)")

            # Emit final progress to signal completion
            self.sam2_extraction_progress.emit(total_to_extract, total_to_extract, f"Done! Extracted {processed} features in {total_time:.1f}s")

            return True

        except ImportError as e:
            self.log.emit(f"SAM2 not installed: {e}")
            self.log.emit("Install with: pip install git+https://github.com/facebookresearch/sam2.git")
            return False
        except Exception as e:
            self.log.emit(f"SAM2 extraction failed: {e}")
            return False

    def run(self):
        """Main training loop."""
        import time
        import threading
        from ..models.unet import get_device, get_model_class

        try:
            self.started.emit()

            # Initialize threading lock for pending data
            self._pending_data_lock = threading.Lock()

            # Initialize snapshot timing (0 so first snapshot happens on epoch 1)
            self._last_snapshot_time = 0
            self._global_batch_count = 0  # Track total batches for loss plot

            # Extract config
            train_images = self.config.get('train_images')
            train_masks = self.config.get('train_masks')
            val_images = self.config.get('val_images')
            val_masks = self.config.get('val_masks')
            checkpoint_path = self.config.get('checkpoint_path', 'checkpoint.pth')
            num_epochs = self.config.get('num_epochs', 5000)
            batch_size = self.config.get('batch_size', 2)
            tile_size = self.config.get('tile_size', 512)
            learning_rate = self.config.get('learning_rate', 1e-4)
            resume_checkpoint = self.config.get('resume_checkpoint', None)
            architecture = self.config.get('architecture', 'unet')

            # Detect architecture variants
            # 2.5D uses 3 channels (z-3, z, z+3)
            is_25d = '25d' in architecture.lower()
            is_sam2 = 'sam2' in architecture.lower()
            n_channels = 3 if is_25d else 1

            if is_25d:
                # Switch to 2.5D training folders if they exist
                train_images_25d = train_images.replace('train_images', 'train_images_25d')
                train_masks_25d = train_masks.replace('train_masks', 'train_masks_25d')

                if os.path.isdir(train_images_25d) and os.path.isdir(train_masks_25d):
                    train_images = train_images_25d
                    train_masks = train_masks_25d
                    self.log.emit(f"Using 2.5D training data from: {train_images}")
                else:
                    self.log.emit(f"WARNING: 2.5D folders not found, using regular folders")
                    self.log.emit(f"  Expected: {train_images_25d}")
                    # Fall back to 1 channel if 2.5D data not available
                    n_channels = 1

            # SAM2 feature extraction (if needed)
            sam2_dir = None
            if is_sam2:
                sam2_dir = os.path.join(os.path.dirname(train_images), 'sam2_features')
                self.log.emit("SAM2 architecture detected - checking features...")

                # Extract SAM2 features if they don't exist
                if not self._extract_sam2_features(train_images, sam2_dir):
                    if self.should_stop:
                        self.finished.emit(False, "SAM2 extraction cancelled")
                        return
                    # Continue anyway but warn user
                    self.log.emit("WARNING: SAM2 features may be incomplete")

            device = get_device()
            self.log.emit(f"Using device: {device}")

            # Setup CUDA optimizations
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")

            # Create datasets
            self.log.emit(f"Loading training data (n_channels={n_channels})...")
            if is_sam2 and sam2_dir:
                train_ds = NucleiPatchDatasetSAM2(train_images, train_masks, sam2_dir,
                                                  tile=tile_size, fg_ratio=0.5, n_channels=n_channels)
            else:
                train_ds = NucleiPatchDataset(train_images, train_masks, tile=tile_size, fg_ratio=0.5,
                                              n_channels=n_channels)

            val_ds = None
            if val_images and val_masks and os.path.exists(val_images):
                self.log.emit("Loading validation data...")
                # Validation doesn't need SAM2 features for now (uses same dataset type)
                val_ds = NucleiPatchDataset(val_images, val_masks, tile=tile_size, fg_ratio=0.0,
                                           n_channels=n_channels)

            # Platform-specific DataLoader settings
            import platform
            is_linux = platform.system() == 'Linux'

            if is_linux:
                train_workers, val_workers = 8, 4
                use_persistent = True
                use_pin_memory = True
            else:
                # macOS: use workers but no persistent_workers
                # (MPS sync every 10 batches prevents the hang)
                train_workers, val_workers = 4, 2
                use_persistent = False
                use_pin_memory = False

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=train_workers, pin_memory=use_pin_memory,
                persistent_workers=use_persistent if train_workers > 0 else False
            )

            val_loader = None
            if val_ds:
                val_loader = DataLoader(
                    val_ds, batch_size=1, shuffle=False,
                    num_workers=val_workers, pin_memory=use_pin_memory,
                    persistent_workers=use_persistent if val_workers > 0 else False
                )

            # ============================================================
            # MODEL INITIALIZATION
            # ============================================================
            print("\n" + "="*60)
            print("TRAINING CONFIGURATION")
            print("="*60)

            # Architecture
            print(f"  Architecture:    {architecture}")
            try:
                ModelClass = get_model_class(architecture)
                print(f"  Model class:     {ModelClass.__name__}")
            except Exception as e:
                self.log.emit(f"ERROR: Failed to load architecture '{architecture}': {e}")
                self.finished.emit(False, f"Architecture not found: {architecture}")
                return

            # Loss function
            from ..models.architectures import get_preferred_loss
            loss_type = get_preferred_loss(architecture)
            criterion = get_loss_function(loss_type)
            print(f"  Loss function:   {loss_type} ({criterion.__class__.__name__})")

            # Training params
            print(f"  Learning rate:   {learning_rate}")
            print(f"  Batch size:      {batch_size}")
            print(f"  Tile size:       {tile_size}")
            print(f"  Max epochs:      {num_epochs}")
            print(f"  Device:          {device}")
            print(f"  Input channels:  {n_channels} ({'2.5D mode' if is_25d else '2D mode'})")

            # Checkpoint
            print(f"  Checkpoint:      {checkpoint_path}")

            # Create model
            model = ModelClass(n_channels=n_channels, n_classes=1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters:      {num_params:,}")

            start_epoch = 0

            # Resume from checkpoint if provided
            if resume_checkpoint and os.path.exists(resume_checkpoint):
                ckpt = torch.load(resume_checkpoint, map_location=device)
                # Handle different checkpoint formats
                if "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"])
                elif "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    # Assume the checkpoint IS the state dict
                    model.load_state_dict(ckpt)
                # Load optimizer if available
                if "optimizer_state" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                elif "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1 if isinstance(ckpt, dict) and "epoch" in ckpt else 0
                print(f"  Resuming from:   epoch {start_epoch}")
            else:
                print(f"  Starting:        from scratch")

            print("="*60 + "\n")

            self.log.emit(f"Training {architecture} with {loss_type} loss")

            # Training loop
            for epoch in range(start_epoch, num_epochs):
                if self.should_stop:
                    self.log.emit("Training stopped by user")
                    break

                # Handle pause
                while self.is_paused and not self.should_stop:
                    self.msleep(100)

                model.train()

                # Freeze BatchNorm running stats for stable live predictions
                # Only when resuming from a checkpoint (stats are already good).
                # When training from scratch, let stats adapt normally.
                if resume_checkpoint:
                    for module in model.modules():
                        if isinstance(module, nn.BatchNorm2d):
                            module.eval()  # Freeze running_mean/running_var updates

                train_loss = 0.0
                batch_count = 0

                for batch in train_loader:
                    batch_count += 1
                    if self.should_stop:
                        break

                    # Handle both 2-tuple (img, mask) and 3-tuple (img, mask, sam2_feat)
                    if is_sam2 and len(batch) == 3:
                        imgs, masks, sam2_feats = batch
                        imgs = imgs.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        sam2_feats = sam2_feats.to(device, non_blocking=True)
                    else:
                        imgs, masks = batch[:2]
                        imgs = imgs.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        sam2_feats = None

                    optimizer.zero_grad(set_to_none=True)

                    if scaler:
                        with torch.amp.autocast('cuda'):
                            if sam2_feats is not None:
                                outputs = model(imgs, sam2_features=sam2_feats)
                            else:
                                outputs = model(imgs)
                            loss = criterion(outputs, masks)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if sam2_feats is not None:
                            outputs = model(imgs, sam2_features=sam2_feats)
                        else:
                            outputs = model(imgs)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()

                    batch_loss = loss.item()
                    train_loss += batch_loss

                    # Emit loss for live plot (every batch)
                    self._global_batch_count += 1
                    self.loss_updated.emit(batch_loss, self._global_batch_count)

                    # MPS requires sync after every batch to prevent queue buildup/hangs
                    if device.type == 'mps':
                        torch.mps.synchronize()

                    # Time-based snapshot for live predictions (every N seconds)
                    if self.snapshot_interval > 0:
                        current_time = time.time()
                        if current_time - self._last_snapshot_time >= self.snapshot_interval:
                            self._create_snapshot(model, checkpoint_path)
                            self._last_snapshot_time = current_time

                avg_train = train_loss / len(train_loader)

                # Validation
                val_loss = 0.0
                if val_loader:
                    model.eval()
                    with torch.no_grad():
                        for imgs, masks in val_loader:
                            imgs = imgs.to(device)
                            masks = masks.to(device)
                            if scaler:
                                with torch.amp.autocast('cuda'):
                                    outputs = model(imgs)
                                    loss = criterion(outputs, masks)
                            else:
                                outputs = model(imgs)
                                loss = criterion(outputs, masks)
                            val_loss += loss.item()
                    val_loss /= max(1, len(val_loader))

                # Emit progress to UI
                self.progress.emit(epoch + 1, num_epochs, avg_train, val_loss)

                # Save checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, checkpoint_path)

                # Notify that model was updated
                self.model_updated.emit(checkpoint_path)

                # Export weights for multi-user sync at configured intervals
                if self.weights_export_interval > 0:
                    if (epoch + 1) % self.weights_export_interval == 0 and epoch != self._last_export_epoch:
                        self._last_export_epoch = epoch
                        # Deep copy weights to avoid issues with ongoing training
                        weights_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        self.weights_exported.emit(weights_copy, epoch + 1, avg_train)

                # Check if dataset reload was requested
                if self._reload_requested:
                    self._reload_requested = False
                    self.log.emit("Reloading dataset with new files...")
                    train_ds = NucleiPatchDataset(train_images, train_masks, tile=tile_size, fg_ratio=0.5,
                                                  n_channels=n_channels)
                    train_loader = DataLoader(
                        train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False
                    )

            # Final save
            final_path = checkpoint_path.replace('.pth', '_final.pth')
            torch.save(model.state_dict(), final_path)
            self.log.emit(f"Training complete. Model saved to {final_path}")
            self.finished.emit(True, checkpoint_path)

        except Exception as e:
            self.log.emit(f"Training error: {e}")
            self.finished.emit(False, str(e))

    def _create_snapshot(self, model, checkpoint_path: str):
        """
        Create a time-based snapshot for predictions.

        Snapshots are saved separately from the main checkpoint and are used
        for predictions while training continues on the main model.

        Args:
            model: The current model
            checkpoint_path: Base checkpoint path (snapshot uses _snapshot suffix)
        """
        import torch

        try:
            # Create snapshot path
            if checkpoint_path.endswith('.pth'):
                snapshot_path = checkpoint_path.replace('.pth', '_snapshot.pth')
            else:
                snapshot_path = checkpoint_path + '_snapshot'

            self._snapshot_checkpoint_path = snapshot_path
            self._snapshot_id += 1

            # Deep copy weights
            weights_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Save snapshot checkpoint
            torch.save({
                "snapshot_id": self._snapshot_id,
                "model_state": weights_copy,
            }, snapshot_path)

            # Emit signal with weights for multi-user broadcast
            self.snapshot_created.emit(weights_copy, self._snapshot_id)

            self.log.emit(f"Created snapshot #{self._snapshot_id}")

        except Exception as e:
            self.log.emit(f"Error creating snapshot: {e}")

    def _process_pending_training_data(self, train_images_dir: str, train_masks_dir: str):
        """
        Process pending training data received from clients.

        Saves the data to disk and requests dataset reload.

        Args:
            train_images_dir: Directory for training images
            train_masks_dir: Directory for training masks
        """
        import io
        from PIL import Image

        if not self._pending_training_data:
            return 0

        processed = 0
        with self._pending_data_lock:
            pending = self._pending_training_data[:]
            self._pending_training_data.clear()

        for image_bytes, mask_bytes, metadata in pending:
            try:
                # Decode images
                img = Image.open(io.BytesIO(image_bytes))
                mask = Image.open(io.BytesIO(mask_bytes))

                # Generate filename
                sender_id = metadata.get('sender_id', 'unknown')[:8]
                timestamp = metadata.get('timestamp', 0)
                crop_id = f"remote_{sender_id}_{timestamp}"

                # Save to training directories
                img_path = os.path.join(train_images_dir, f"{crop_id}.tif")
                mask_path = os.path.join(train_masks_dir, f"{crop_id}.tif")

                img.save(img_path, compression='tiff_lzw')
                mask.save(mask_path, compression='tiff_lzw')

                processed += 1
                self.log.emit(f"Received training data from {sender_id}")

            except Exception as e:
                self.log.emit(f"Error processing remote training data: {e}")

        if processed > 0:
            self._reload_requested = True

        return processed
