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
            is_25d = '25d' in architecture.lower()
            is_sam2 = 'sam2' in architecture.lower()
            n_channels = 3 if is_25d else 1

            device = get_device()
            self.log.emit(f"Using device: {device}")

            # Load model
            self.log.emit(f"Loading model ({architecture}) from {checkpoint_path}...")
            self.log.emit(f"  Mode: {'2.5D' if is_25d else '2D'} (n_channels={n_channels})")
            model = load_model(checkpoint_path, n_channels=n_channels, device=device, architecture=architecture)

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
                self._predict_folder(model, input_dir, output_dir, patch_size, overlap, name, device,
                                    is_25d=is_25d, is_sam2=is_sam2, sam2_predictor=sam2_predictor)

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
                        is_25d=False, is_sam2=False, sam2_predictor=None):
        """Predict on all images in a folder."""
        # Find all images
        image_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')
        ])

        total = len(image_files)
        self.log.emit(f"Found {total} images in {name}")

        stride = patch_size - overlap

        # For 2.5D, we need a helper to load adjacent slices
        def load_image(path):
            """Load and convert image to grayscale float32."""
            img = Image.open(path)
            arr = np.array(img).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr

        skipped = 0
        for i, image_path in enumerate(image_files):
            if self.should_stop:
                break

            # Load center image
            image = load_image(image_path)
            h, w = image.shape

            # For 2.5D, load slices with spacing of 3 (z-3, z, z+3)
            if is_25d:
                slice_spacing = 3
                slices = []
                for offset in [-slice_spacing, 0, slice_spacing]:
                    idx = i + offset
                    # Clamp at boundaries
                    idx = max(0, min(total - 1, idx))

                    if offset == 0:
                        slices.append(image)
                    else:
                        adj_image = load_image(image_files[idx])
                        # Ensure same shape
                        if adj_image.shape != (h, w):
                            adj_image = np.resize(adj_image, (h, w))
                        slices.append(adj_image)

                # Stack as 11-channel image (H, W, C)
                image_stack = np.stack(slices, axis=-1)
            else:
                image_stack = None

            # Skip fully black images (save empty mask instead)
            img_min, img_max = image.min(), image.max()
            if img_max == img_min:
                # Image is uniform (likely all black) - save empty mask
                output_path = output_dir / f"{image_path.stem}_pred.tif"
                Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(
                    output_path, compression='tiff_lzw'
                )
                skipped += 1
                self.progress.emit(name, i + 1, total)
                continue

            # Normalize (center slice for 2D, per-channel for 2.5D)
            if is_25d:
                # Normalize each channel independently
                for c in range(image_stack.shape[-1]):
                    ch = image_stack[..., c]
                    ch_min, ch_max = ch.min(), ch.max()
                    if ch_max > ch_min:
                        image_stack[..., c] = (ch - ch_min) / (ch_max - ch_min)
                    else:
                        image_stack[..., c] = 0.0
            else:
                image = (image - img_min) / (img_max - img_min)

            # Patch-based prediction
            pred_full = np.zeros((h, w), dtype=np.float32)
            count = np.zeros((h, w), dtype=np.float32)

            with torch.no_grad():
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        if is_25d:
                            patch = image_stack[y:y+patch_size, x:x+patch_size, :]
                            ph, pw = patch.shape[:2]

                            # Pad if needed
                            pad_bottom = patch_size - ph if ph < patch_size else 0
                            pad_right = patch_size - pw if pw < patch_size else 0
                            if pad_bottom or pad_right:
                                patch = np.pad(patch, ((0, pad_bottom), (0, pad_right), (0, 0)))

                            # Convert to (N, C, H, W) tensor
                            patch_tensor = torch.tensor(
                                np.transpose(patch.copy(), (2, 0, 1))[None, ...],
                                dtype=torch.float32
                            ).to(device)
                        else:
                            patch = image[y:y+patch_size, x:x+patch_size]
                            ph, pw = patch.shape

                            # Pad if needed
                            pad_bottom = patch_size - ph if ph < patch_size else 0
                            pad_right = patch_size - pw if pw < patch_size else 0
                            if pad_bottom or pad_right:
                                patch = np.pad(patch, ((0, pad_bottom), (0, pad_right)))

                            # Predict
                            patch_tensor = torch.tensor(
                                patch.copy()[None, None, ...],
                                dtype=torch.float32
                            ).to(device)

                        # Extract SAM2 features on-the-fly if needed
                        sam2_feats = None
                        if is_sam2 and sam2_predictor is not None and not is_25d:
                            # For SAM2, extract features from the normalized patch
                            # Need to use the padded patch at full patch_size
                            if pad_bottom or pad_right:
                                sam2_patch = patch  # Already padded
                            else:
                                sam2_patch = patch.copy()
                            sam2_feats = self._extract_sam2_patch_features(
                                sam2_predictor, sam2_patch, device
                            )

                        # Model forward pass
                        if sam2_feats is not None:
                            pred = torch.sigmoid(model(patch_tensor, sam2_features=sam2_feats))[0, 0].cpu().numpy()
                        else:
                            pred = torch.sigmoid(model(patch_tensor))[0, 0].cpu().numpy()
                        pred = pred[:ph, :pw]

                        pred_full[y:y+ph, x:x+pw] += pred
                        count[y:y+ph, x:x+pw] += 1

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
