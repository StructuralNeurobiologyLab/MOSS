#!/usr/bin/env python3
"""
Prediction worker for running UNet inference on image folders.

Adapted from predict_unet.py
"""

import os
import gc
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from PyQt6.QtCore import QThread, pyqtSignal


class PredictWorker(QThread):
    """Background worker for UNet prediction on image folders."""

    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(str, int, int)  # view_name, current, total
    finished = pyqtSignal(bool, dict)  # success, output_dirs dict
    log = pyqtSignal(str)

    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary with:
                - checkpoint_path: Path to model checkpoint
                - views: List of dicts with {'name': str, 'input_dir': str, 'output_dir': str}
                - patch_size: int (default 512)
                - overlap: int (default 64)
        """
        super().__init__()
        self.config = config
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def _init_sam2_predictor(self, device):
        """Initialize SAM2 predictor for on-the-fly feature extraction.

        Returns:
            SAM2ImagePredictor instance, or None if SAM2 not available
        """
        try:
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

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
            model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
            device_str = str(device) if hasattr(device, '__str__') else device
            model = build_sam2(model_cfg, ckpt_path, device=device_str)
            predictor = SAM2ImagePredictor(model)

            self.log.emit("SAM2 predictor initialized successfully")
            return predictor

        except ImportError as e:
            self.log.emit(f"SAM2 not available: {e}")
            return None
        except Exception as e:
            self.log.emit(f"Failed to initialize SAM2: {e}")
            return None

    def _extract_sam2_patch_features(self, predictor, patch: np.ndarray, device) -> torch.Tensor:
        """Extract SAM2 features for a single patch on-the-fly.

        Args:
            predictor: SAM2ImagePredictor instance
            patch: Grayscale patch as float32 array (H, W), normalized 0-1
            device: torch device

        Returns:
            SAM2 features as tensor (1, 256, H/16, W/16)
        """
        # Convert normalized float to uint8 for SAM2
        patch_uint8 = (patch * 255).astype(np.uint8)

        # SAM expects RGB input
        rgb = np.repeat(patch_uint8[..., None], 3, axis=-1)

        # Extract features
        device_str = str(device) if hasattr(device, '__str__') else device
        amp_dtype = torch.bfloat16 if device_str == "cuda" else torch.float32

        with torch.inference_mode():
            with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=(device_str == "cuda")):
                predictor.set_image(rgb)
                embedding = predictor.get_image_embedding()  # (1, 256, Hf, Wf)

        return embedding.to(device)

    def run(self):
        from ..models.unet import load_model, get_device

        try:
            self.started.emit()

            checkpoint_path = self.config['checkpoint_path']
            architecture = self.config.get('architecture', 'unet')
            views = self.config['views']
            patch_size = self.config.get('patch_size', 512)
            overlap = self.config.get('overlap', 64)

            # Detect architecture variants
            from ..models.architectures import (get_n_context_slices, uses_z_coord,
                                                 is_3d_architecture, get_3d_patch_depth, get_3d_patch_size)
            is_3d = is_3d_architecture(architecture)
            is_25d = '25d' in architecture.lower() and not is_3d
            is_sam2 = 'sam2' in architecture.lower()
            n_channels = 1 if is_3d else get_n_context_slices(architecture)
            add_z_coord = False if is_3d else uses_z_coord(architecture)
            model_n_channels = n_channels + (1 if add_z_coord else 0)

            device = get_device()
            self.log.emit(f"Using device: {device}")

            # Load model
            self.log.emit(f"Loading model ({architecture}) from {checkpoint_path}...")
            if is_3d:
                self.log.emit(f"  Mode: 3D volumetric (n_channels={model_n_channels})")
            elif add_z_coord:
                self.log.emit(f"  Mode: {'2.5D' if is_25d else '2D'} (n_channels={model_n_channels}: {n_channels} image + z-coord)")
            else:
                self.log.emit(f"  Mode: {'2.5D' if is_25d else '2D'} (n_channels={n_channels})")
            model = load_model(checkpoint_path, n_channels=model_n_channels, device=device, architecture=architecture)

            # Initialize SAM2 predictor if needed (for on-the-fly feature extraction)
            sam2_predictor = None
            if is_sam2:
                self.log.emit("Initializing SAM2 for on-the-fly feature extraction...")
                sam2_predictor = self._init_sam2_predictor(device)
                if sam2_predictor is None:
                    self.log.emit("WARNING: SAM2 not available, predictions may be suboptimal")

            output_dirs = {}

            # Process each view
            for view_config in views:
                if self.should_stop:
                    break

                name = view_config['name']
                input_dir = Path(view_config['input_dir'])
                output_dir = Path(view_config['output_dir'])
                output_dir.mkdir(exist_ok=True, parents=True)
                output_dirs[name] = str(output_dir)

                self.log.emit(f"Processing {name}...")
                if is_3d:
                    patch_depth = get_3d_patch_depth(architecture)
                    patch_size_3d = get_3d_patch_size(architecture)
                    self._predict_folder_3d(model, input_dir, output_dir, patch_size_3d,
                                            patch_depth, overlap, name, device)
                else:
                    self._predict_folder(model, input_dir, output_dir, patch_size, overlap, name, device,
                                        is_25d=is_25d, is_sam2=is_sam2, sam2_predictor=sam2_predictor,
                                        architecture=architecture, add_z_coord=add_z_coord)

            # Check if we were stopped vs completed
            if self.should_stop:
                self.log.emit("Prediction stopped by user")
                self.finished.emit(False, {"error": "Stopped by user"})
            else:
                self.finished.emit(True, output_dirs)

        except Exception as e:
            self.log.emit(f"Prediction error: {e}")
            self.finished.emit(False, {"error": str(e)})

    def _predict_folder(self, model, input_dir, output_dir, patch_size, overlap, name, device,
                        is_25d=False, is_sam2=False, sam2_predictor=None, architecture='unet',
                        add_z_coord=False):
        """Predict on all images in a folder."""
        # Find all images
        image_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')
        ])

        total = len(image_files)
        self.log.emit(f"Found {total} images in {name}")

        stride = patch_size - overlap

        # Image loading with LRU cache for 2.5D (adjacent slices overlap heavily)
        _slice_cache = {}

        def load_image(path, idx=None):
            """Load image as native dtype (uint8/uint16), with caching."""
            if idx is not None and idx in _slice_cache:
                return _slice_cache[idx]
            img = Image.open(path)
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[..., 0]
            if idx is not None:
                _slice_cache[idx] = arr
            return arr

        # Pre-compute 2.5D parameters
        if is_25d:
            from ..models.architectures import get_n_context_slices, get_slice_spacing
            n_ctx = get_n_context_slices(architecture)
            slice_spacing = get_slice_spacing(architecture)
            n_flanking = (n_ctx - 1) // 2
            # Max cache size: keep only slices that could be needed
            max_cache_idx_range = n_flanking * slice_spacing + 1

        skipped = 0
        for i, image_path in enumerate(image_files):
            if self.should_stop:
                break

            # Load center image
            image = load_image(image_path, idx=i)
            h, w = image.shape

            # For 2.5D, load adjacent slices (cached — most are reused)
            if is_25d:
                slices = []
                for k in range(-n_flanking, n_flanking + 1):
                    idx = i + k * slice_spacing
                    idx = max(0, min(total - 1, idx))
                    adj = load_image(image_files[idx], idx=idx)
                    if adj.shape != (h, w):
                        adj = np.resize(adj, (h, w))
                    slices.append(adj)

                # Stack as multi-channel image (H, W, C)
                image_stack = np.stack(slices, axis=-1)

                # Evict old cache entries no longer needed
                min_needed = max(0, i - n_flanking * slice_spacing)
                for cached_idx in list(_slice_cache.keys()):
                    if cached_idx < min_needed:
                        del _slice_cache[cached_idx]
            else:
                image_stack = None

            # Skip fully black images (save empty mask instead)
            img_min, img_max = int(image.min()), int(image.max())
            if img_max == img_min:
                output_path = output_dir / f"{image_path.stem}_pred.tif"
                Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(
                    output_path, compression='tiff_lzw'
                )
                skipped += 1
                self.progress.emit(name, i + 1, total)
                continue

            # Normalization is deferred to patch level to avoid full-image float32 conversion

            # Patch-based prediction with batching
            pred_full = np.zeros((h, w), dtype=np.float32)
            count = np.zeros((h, w), dtype=np.float32)
            batch_limit = 28  # All patches in one GPU batch

            with torch.no_grad():
                # Collect all valid patches first
                patch_batch = []  # (tensor, y, x, ph, pw)

                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        if is_25d:
                            patch = image_stack[y:y+patch_size, x:x+patch_size, :]
                            ph, pw = patch.shape[:2]

                            if patch.max() == 0:
                                continue

                            pad_bottom = patch_size - ph if ph < patch_size else 0
                            pad_right = patch_size - pw if pw < patch_size else 0
                            if pad_bottom or pad_right:
                                patch = np.pad(patch, ((0, pad_bottom), (0, pad_right), (0, 0)))

                            patch = patch.astype(np.float32)
                            for c in range(patch.shape[-1]):
                                ch = patch[..., c]
                                ch_min, ch_max = ch.min(), ch.max()
                                if ch_max > ch_min:
                                    patch[..., c] = (ch - ch_min) / (ch_max - ch_min)
                                else:
                                    patch[..., c] = 0.0

                            tensor = torch.tensor(
                                np.transpose(patch, (2, 0, 1))[None, ...],
                                dtype=torch.float32
                            )
                        else:
                            patch = image[y:y+patch_size, x:x+patch_size]
                            ph, pw = patch.shape

                            if patch.max() == 0:
                                continue

                            pad_bottom = patch_size - ph if ph < patch_size else 0
                            pad_right = patch_size - pw if pw < patch_size else 0
                            if pad_bottom or pad_right:
                                patch = np.pad(patch, ((0, pad_bottom), (0, pad_right)))

                            patch = patch.astype(np.float32)
                            p_min, p_max = patch.min(), patch.max()
                            if p_max > p_min:
                                patch = (patch - p_min) / (p_max - p_min)

                            tensor = torch.tensor(
                                patch[None, None, ...],
                                dtype=torch.float32
                            )

                        # Append z-coordinate channel if needed
                        if add_z_coord:
                            z_val = i / max(total - 1, 1)
                            z_ch = torch.full((1, 1, patch_size, patch_size), z_val, dtype=torch.float32)
                            tensor = torch.cat([tensor, z_ch], dim=1)

                        patch_batch.append((tensor, y, x, ph, pw))

                        # Run batch when full
                        if len(patch_batch) >= batch_limit:
                            self._run_patch_batch(model, patch_batch, pred_full, count, device,
                                                  is_sam2, sam2_predictor, is_25d)
                            patch_batch = []

                # Run remaining patches
                if patch_batch:
                    self._run_patch_batch(model, patch_batch, pred_full, count, device,
                                          is_sam2, sam2_predictor, is_25d)

            # Normalize and binarize
            pred_full /= np.maximum(count, 1e-8)
            mask_bin = ((pred_full > 0.5) * 255).astype(np.uint8)

            # Save with LZW compression
            output_path = output_dir / f"{image_path.stem}_pred.tif"
            Image.fromarray(mask_bin).save(output_path, compression='tiff_lzw')

            # Progress
            self.progress.emit(name, i + 1, total)

            # Clear GPU cache periodically
            if device.type == 'cuda' and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        if skipped > 0:
            self.log.emit(f"Skipped {skipped} blank images in {name}")

    def _run_patch_batch(self, model, patch_batch, pred_full, count, device,
                         is_sam2=False, sam2_predictor=None, is_25d=False):
        """Run a batch of patches through the model in one forward pass."""
        if not patch_batch:
            return

        # SAM2 doesn't support batching — fall back to one-at-a-time
        if is_sam2 and sam2_predictor is not None and not is_25d:
            for tensor, y, x, ph, pw in patch_batch:
                tensor = tensor.to(device)
                sam2_feats = self._extract_sam2_patch_features(
                    sam2_predictor, tensor[0, 0].cpu().numpy(), device
                )
                if sam2_feats is not None:
                    pred = torch.sigmoid(model(tensor, sam2_features=sam2_feats))[0, 0].cpu().numpy()
                else:
                    pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                pred = pred[:ph, :pw]
                pred_full[y:y+ph, x:x+pw] += pred
                count[y:y+ph, x:x+pw] += 1
            return

        # Stack all patches into one batch tensor
        batch_tensor = torch.cat([t for t, _, _, _, _ in patch_batch], dim=0).to(device)

        # Single forward pass for entire batch
        preds = torch.sigmoid(model(batch_tensor)).cpu().numpy()

        # Place results
        for idx, (_, y, x, ph, pw) in enumerate(patch_batch):
            pred = preds[idx, 0, :ph, :pw]
            pred_full[y:y+ph, x:x+pw] += pred
            count[y:y+ph, x:x+pw] += 1

    def _predict_folder_3d(self, model, input_dir, output_dir, patch_size, patch_depth,
                           overlap, name, device):
        """Predict on a folder of images using 3D sliding window."""
        # Find all images
        image_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')
        ])

        total_z = len(image_files)
        if total_z == 0:
            self.log.emit(f"No images found in {name}")
            return

        # Load first image to get dimensions
        first_img = np.array(Image.open(image_files[0]))
        if first_img.ndim == 3:
            first_img = first_img[..., 0]
        h, w = first_img.shape

        self.log.emit(f"3D prediction: {total_z} slices, {h}x{w}, "
                      f"patch={patch_depth}x{patch_size}x{patch_size}")

        # Allocate output volume and count arrays
        pred_volume = np.zeros((total_z, h, w), dtype=np.float32)
        count_volume = np.zeros((total_z, h, w), dtype=np.float32)

        # Sliding window parameters
        stride_z = max(1, patch_depth - overlap)
        stride_xy = max(1, patch_size - overlap)

        # Preload all slices (needed for 3D blocks)
        self.log.emit(f"Loading volume into memory...")
        volume = np.zeros((total_z, h, w), dtype=np.float32)
        for i, f in enumerate(image_files):
            img = np.array(Image.open(f))
            if img.ndim == 3:
                img = img[..., 0]
            volume[i] = img.astype(np.float32)

        # 3D sliding window
        with torch.no_grad():
            total_blocks = 0
            for z in range(0, total_z, stride_z):
                z_end = min(z + patch_depth, total_z)
                z_start = max(0, z_end - patch_depth)  # Ensure full depth patch
                actual_d = z_end - z_start

                for y in range(0, h, stride_xy):
                    for x in range(0, w, stride_xy):
                        if self.should_stop:
                            return

                        # Extract 3D patch
                        y_end = min(y + patch_size, h)
                        x_end = min(x + patch_size, w)
                        patch = volume[z_start:z_end, y:y_end, x:x_end]

                        ph, pw = patch.shape[1], patch.shape[2]

                        if patch.max() == 0:
                            continue

                        # Pad if needed
                        pad_d = patch_depth - patch.shape[0]
                        pad_h = patch_size - ph
                        pad_w = patch_size - pw
                        if pad_d > 0 or pad_h > 0 or pad_w > 0:
                            patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)))

                        # Normalize
                        p_min, p_max = patch.min(), patch.max()
                        if p_max > p_min:
                            patch = (patch - p_min) / (p_max - p_min)

                        # (D, H, W) -> (1, 1, D, H, W)
                        tensor = torch.tensor(
                            patch[None, None, ...], dtype=torch.float32
                        ).to(device)

                        pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

                        # Place result (only the valid region)
                        pred_volume[z_start:z_end, y:y_end, x:x_end] += pred[:actual_d, :ph, :pw]
                        count_volume[z_start:z_end, y:y_end, x:x_end] += 1

                        total_blocks += 1

                self.progress.emit(name, min(z + stride_z, total_z), total_z)

        self.log.emit(f"Processed {total_blocks} 3D blocks")

        # Normalize and save per-slice
        pred_volume /= np.maximum(count_volume, 1e-8)

        for i, image_path in enumerate(image_files):
            mask_bin = ((pred_volume[i] > 0.5) * 255).astype(np.uint8)
            output_path = output_dir / f"{image_path.stem}_pred.tif"
            Image.fromarray(mask_bin).save(output_path, compression='tiff_lzw')

        del volume, pred_volume, count_volume
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
