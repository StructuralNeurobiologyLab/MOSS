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


# A patch's border is where the convolutions' padding makes it least reliable, so the
# blend must not weight it like the centre. Never let the window reach zero, though:
# the outermost rows and columns of a volume are covered only by the patch whose
# window is zero there, and they would come out blank.
XY_BLEND_FLOOR = 0.05


def build_xy_blend_window(patch_size: int, xy_blend: str = 'hann') -> np.ndarray:
    """Separable XY blend weights of shape (patch_size, patch_size).

    Hann is constant-overlap-add at 50% overlap, which is the stride the 3D path runs
    at by default (patch 128, overlap 64), so interior weights sum flat and the blend
    adds no amplitude ripple of its own. 'tophat' is the old uniform weighting.
    """
    if xy_blend == 'tophat':
        return np.ones((patch_size, patch_size), dtype=np.float32)
    if xy_blend != 'hann':
        raise ValueError(f"unknown xy_blend {xy_blend!r}; expected 'hann' or 'tophat'")
    i = np.arange(patch_size, dtype=np.float32)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * i / max(patch_size - 1, 1))
    hann = np.maximum(hann, XY_BLEND_FLOOR)
    return np.outer(hann, hann).astype(np.float32)


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
                                                 is_3d_architecture, get_3d_patch_depth, get_3d_patch_size,
                                                 is_slab_architecture)
            is_3d = is_3d_architecture(architecture)
            is_slab = is_slab_architecture(architecture)
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

            # Slab geometry comes from the checkpoint: a model must be run at the depth
            # and jitter it was trained with, whatever the architecture now declares.
            patch_depth, z_jitter = 0, 0
            if is_3d:
                if is_slab:
                    from ..models.slab_inference import read_slab_geometry
                    patch_depth, z_jitter = read_slab_geometry(checkpoint_path, architecture)
                    self.log.emit(f"  Slab geometry: depth {patch_depth}, Z-jitter {z_jitter}")
                else:
                    patch_depth = get_3d_patch_depth(architecture)

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
                    patch_size_3d = get_3d_patch_size(architecture)
                    self._predict_folder_3d(model, input_dir, output_dir, patch_size_3d,
                                            patch_depth, overlap, name, device,
                                            z_jitter=z_jitter)
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
        """Predict on all slices of a view (a folder of TIFFs, or a Zarr volume).

        XY is the native orientation of the raw volume, so when the input is a Zarr
        we read its slices directly rather than requiring a duplicated folder of XY
        TIFFs. Resliced views (xz/yz/diagonals) are still passed in as image folders.
        """
        input_dir = Path(input_dir)
        is_zarr = (input_dir.suffix == '.zarr' or (input_dir / '.zarray').exists()
                   or (input_dir / '.zgroup').exists() or (input_dir / 'zarr.json').exists())

        zarr_source = None
        if is_zarr:
            from ..zarr_image_source import ZarrImageSource
            zarr_source = ZarrImageSource(str(input_dir))
            total = zarr_source.num_slices
            self.log.emit(f"Found {total} images in {name} (reading from Zarr volume)")

            def stem_for(idx):
                # Zero-padded so predictions sort by z, which the voting step relies on.
                return f"{name}_slice_{idx:05d}"
        else:
            image_files = sorted([
                f for f in input_dir.iterdir()
                if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')
            ])
            total = len(image_files)
            self.log.emit(f"Found {total} images in {name}")

            def stem_for(idx):
                return image_files[idx].stem

        stride = patch_size - overlap

        # Slice loading with a small cache for 2.5D (adjacent slices overlap heavily).
        _slice_cache = {}

        def load_slice(idx):
            """Load slice `idx` as a 2D native-dtype array (folder or Zarr), cached."""
            if idx in _slice_cache:
                return _slice_cache[idx]
            if zarr_source is not None:
                arr = zarr_source.get_slice(idx, pyramid_level=0)
            else:
                arr = np.array(Image.open(image_files[idx]))
            if arr is not None and arr.ndim == 3:
                arr = arr[..., 0]
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
        for i in range(total):
            if self.should_stop:
                break

            # Load center slice
            image = load_slice(i)
            h, w = image.shape

            # For 2.5D, load adjacent slices (cached — most are reused)
            if is_25d:
                slices = []
                for k in range(-n_flanking, n_flanking + 1):
                    idx = i + k * slice_spacing
                    idx = max(0, min(total - 1, idx))
                    adj = load_slice(idx)
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
                output_path = output_dir / f"{stem_for(i)}_pred.tif"
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
            output_path = output_dir / f"{stem_for(i)}_pred.tif"
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
                           overlap, name, device, z_jitter=0, batch_limit=8,
                           xy_blend='hann'):
        """Predict on a folder of images using a 3D sliding window.

        With z_jitter > 0 (a slab model) only the Z planes that training actually
        supervised are blended in, and Z advances by that trained width. With
        z_jitter = 0 every plane is weighted equally, which is the plain 3D behaviour.

        Z is streamed. Slabs are visited in ascending order and a plane is written out
        and freed as soon as no later slab can reach it, so peak memory scales with
        patch_depth instead of with the slice count. Holding the input volume plus the
        two accumulators outright wanted 356 GiB *each* on a 4607x3475x5963 dataset.

        xy_blend selects the XY blend window: 'hann' tapers each patch toward its
        border, 'tophat' is the old uniform weighting kept for reproducing earlier
        predictions. Z blending is unaffected -- its hard rectangle is deliberate,
        because the untrained margins must contribute exactly nothing.
        """
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

        from ..models.slab_inference import (slab_z_weights, slab_z_starts,
                                             slab_read_indices, trained_z_window)

        z_weights = slab_z_weights(patch_depth, z_jitter)
        # A slab model steps by its trained width -- stepping less would only re-average
        # the same trained planes. A plain 3D model has no untrained margin, so it keeps
        # honouring `overlap` as it always did: at patch_depth 32 with overlap 64 that is
        # a stride of 1, i.e. ~32 votes per plane. Dropping that would quietly change
        # every existing unet_3d prediction.
        z_stride = None if z_jitter else max(1, patch_depth - overlap)
        z_starts = slab_z_starts(total_z, patch_depth, z_jitter, stride=z_stride)
        # First plane of a slab carrying any weight. A slab starting at s cannot touch
        # any plane below s + w_lo, which is what makes retiring planes early safe.
        _nonzero = np.nonzero(z_weights)[0]
        w_lo = int(_nonzero[0]) if len(_nonzero) else 0
        if z_jitter:
            lo, hi = trained_z_window(patch_depth, z_jitter)
            self.log.emit(f"3D slab prediction: {total_z} slices, {h}x{w}, "
                          f"patch={patch_depth}x{patch_size}x{patch_size}")
            self.log.emit(f"  blending Z planes [{lo}..{hi}] of {patch_depth} "
                          f"(the rest were never trained), {len(z_starts)} slabs")
        else:
            self.log.emit(f"3D prediction: {total_z} slices, {h}x{w}, "
                          f"patch={patch_depth}x{patch_size}x{patch_size}")

        stride_xy = max(1, patch_size - overlap)
        xy_win = build_xy_blend_window(patch_size, xy_blend)

        # Slices are read on demand and kept only while some slab still needs them.
        # Cached in native dtype -- converting to float32 up front quadrupled a uint8
        # volume for no benefit, since normalization happens per patch anyway.
        slice_cache = {}

        def load_slice(z):
            arr = slice_cache.get(z)
            if arr is None:
                arr = np.array(Image.open(image_files[z]))
                if arr.ndim == 3:
                    arr = arr[..., 0]
                slice_cache[z] = arr
            return arr

        # Planes still in flight: z -> [weighted sum, accumulated weight].
        planes = {}

        def plane(z):
            entry = planes.get(z)
            if entry is None:
                entry = [np.zeros((h, w), dtype=np.float32),
                         np.zeros((h, w), dtype=np.float32)]
                planes[z] = entry
            return entry

        next_z = 0  # planes are finalized in ascending z, one output file each

        def flush_upto(limit):
            """Normalize, write and free every plane below `limit`."""
            nonlocal next_z
            limit = min(limit, total_z)
            for z in range(next_z, limit):
                entry = planes.pop(z, None)
                if entry is None:
                    # Every patch on this plane was blank, which the whole-volume
                    # version also emitted as an empty mask. Keep doing that: the
                    # voting step downstream expects one file per slice.
                    mask_bin = np.zeros((h, w), dtype=np.uint8)
                else:
                    total, weight = entry
                    total /= np.maximum(weight, 1e-8)
                    mask_bin = ((total > 0.5) * 255).astype(np.uint8)
                Image.fromarray(mask_bin).save(
                    output_dir / f"{image_files[z].stem}_pred.tif",
                    compression='tiff_lzw'
                )
            if limit > next_z:
                next_z = limit

        state = {'blocks': 0, 'batch': max(1, int(batch_limit))}

        def run_batch(patch_batch, z_start):
            """One forward pass for a whole batch of patches from the same slab."""
            batch_tensor = torch.cat([t for t, _, _, _, _ in patch_batch], dim=0).to(device)
            preds = torch.sigmoid(model(batch_tensor)).cpu().numpy()

            for idx, (_, y, x, ph, pw) in enumerate(patch_batch):
                pred = preds[idx, 0]
                # A partial patch keeps the leading corner of the window; the trailing
                # taper is dropped along with the padding it would have covered.
                win = xy_win[:ph, :pw]
                # Accumulate only the trained Z planes, each at its blend weight.
                for j in range(patch_depth):
                    wj = z_weights[j]
                    if wj <= 0:
                        continue
                    z = z_start + j
                    if not 0 <= z < total_z:
                        continue
                    wxy = win * wj
                    total, weight = plane(z)
                    total[y:y + ph, x:x + pw] += pred[j, :ph, :pw] * wxy
                    weight[y:y + ph, x:x + pw] += wxy
            state['blocks'] += len(patch_batch)

        def run_batch_safe(patch_batch, z_start):
            """Run a batch, halving it if the GPU cannot hold it.

            Nothing is accumulated until the forward pass has returned, so a failed
            batch leaves no partial contribution behind and can simply be redone.
            """
            if not patch_batch:
                return
            try:
                run_batch(patch_batch, z_start)
                return
            except RuntimeError as e:
                if len(patch_batch) == 1 or 'out of memory' not in str(e).lower():
                    raise
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            half = max(1, len(patch_batch) // 2)
            # A size that OOMs once will OOM again, so keep the smaller batch.
            state['batch'] = min(state['batch'], half)
            self.log.emit(f"  GPU out of memory; 3D batch reduced to {state['batch']}")
            for i in range(0, len(patch_batch), half):
                run_batch_safe(patch_batch[i:i + half], z_start)

        # 3D sliding window, one slab at a time
        with torch.no_grad():
            for slab_i, z_start in enumerate(z_starts):
                # Clamped read, so a slab may hang off either end of the volume and
                # still be full depth (matching how training slabs were captured).
                z_idx = slab_read_indices(z_start, patch_depth, total_z)
                slab = [load_slice(int(z)) for z in z_idx]

                patch_batch = []  # (tensor, y, x, ph, pw)

                for y in range(0, h, stride_xy):
                    for x in range(0, w, stride_xy):
                        if self.should_stop:
                            return

                        # Extract 3D patch (full depth via the clamped Z indices)
                        y_end = min(y + patch_size, h)
                        x_end = min(x + patch_size, w)
                        patch = np.stack([s[y:y_end, x:x_end] for s in slab])

                        ph, pw = patch.shape[1], patch.shape[2]

                        if patch.max() == 0:
                            continue

                        patch = patch.astype(np.float32)

                        # Pad XY if needed; Z is already exactly patch_depth
                        pad_h = patch_size - ph
                        pad_w = patch_size - pw
                        if pad_h > 0 or pad_w > 0:
                            patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)))

                        # Normalize
                        p_min, p_max = patch.min(), patch.max()
                        if p_max > p_min:
                            patch = (patch - p_min) / (p_max - p_min)

                        # (D, H, W) -> (1, 1, D, H, W)
                        tensor = torch.tensor(
                            patch[None, None, ...], dtype=torch.float32
                        )
                        patch_batch.append((tensor, y, x, ph, pw))

                        if len(patch_batch) >= state['batch']:
                            run_batch_safe(patch_batch, z_start)
                            patch_batch = []

                if patch_batch:
                    run_batch_safe(patch_batch, z_start)

                # Retire everything the remaining slabs can no longer reach, and drop
                # the slices none of them will read again.
                if slab_i + 1 < len(z_starts):
                    next_start = z_starts[slab_i + 1]
                    flush_upto(next_start + w_lo)
                    keep_from = max(0, next_start)
                    for z in [k for k in slice_cache if k < keep_from]:
                        del slice_cache[z]
                else:
                    flush_upto(total_z)

                self.progress.emit(name, min(slab_i + 1, len(z_starts)), len(z_starts))

                if device.type == 'cuda' and (slab_i + 1) % 20 == 0:
                    torch.cuda.empty_cache()

        self.log.emit(f"Processed {state['blocks']} 3D blocks")

        slice_cache.clear()
        planes.clear()
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
