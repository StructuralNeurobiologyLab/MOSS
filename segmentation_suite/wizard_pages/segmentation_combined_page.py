#!/usr/bin/env python3
"""
Combined Segmentation page - offers both MOSS (recommended) and LSD 2D options.
"""

import os
import psutil
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QProgressBar, QGroupBox, QFileDialog,
    QMessageBox, QTextEdit, QStackedWidget, QRadioButton, QButtonGroup,
    QCheckBox, QScrollArea, QGridLayout, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

from ..dpi_scaling import scaled
from ..project_config import (
    has_subprojects, list_subprojects, get_active_subproject,
    get_subproject_dir,
)


class ConversionWorker(QThread):
    """Worker thread for TIFF to Zarr conversion."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, source: str, dest: str, chunk_size: tuple, workers: int):
        super().__init__()
        self.source = source
        self.dest = dest
        self.chunk_size = chunk_size
        self.workers = workers
        self._stop = False

    def run(self):
        try:
            from segmentation_suite.em_pipeline.data.convert import convert as do_convert

            def progress_callback(completed, total, msg):
                if not self._stop:
                    self.progress.emit(completed, total, msg)

            self.log.emit(f"Converting {self.source} to Zarr...")
            do_convert(
                self.source,
                self.dest,
                chunk_size=self.chunk_size,
                progress_callback=progress_callback,
                num_workers=self.workers,
            )
            self.finished.emit(True, str(self.dest))
        except Exception as e:
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(e))

    def stop(self):
        self._stop = True


class SegmentationWorker(QThread):
    """Worker thread for 3D segmentation."""

    progress = pyqtSignal(str, int)
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str, strategy: str,
                 quality: str, device: str, model_path: Optional[str] = None,
                 lsd_params: Optional[dict] = None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.strategy = strategy
        self.quality = quality
        self.device = device
        self.model_path = model_path
        self.lsd_params = lsd_params or {}
        self._stop = False

    def run(self):
        try:
            from segmentation_suite.em_pipeline.pipeline import SegmentationPipeline, PipelineConfig

            self.log.emit(f"Starting {self.strategy} segmentation on {self.device}...")

            config = PipelineConfig(
                strategy=self.strategy,
                quality=self.quality,
                device=self.device,
                resume=True,
            )

            pipeline = SegmentationPipeline(config)

            # Apply LSD-specific parameters to strategy config if provided
            if self.lsd_params and self.strategy in ('lsd', 'lsd3d'):
                pipeline._load_strategy()
                for param_name, param_value in self.lsd_params.items():
                    if hasattr(pipeline._strategy.config, param_name):
                        setattr(pipeline._strategy.config, param_name, param_value)
                        self.log.emit(f"  {param_name}: {param_value}")

            result = pipeline.run(
                self.input_path,
                self.output_path,
                model_path=self.model_path,
            )

            self.log.emit(f"Complete! {result.num_segments} segments in {result.total_time:.1f}s")
            self.finished.emit(True, self.output_path)

        except Exception as e:
            import traceback
            self.log.emit(f"Error: {e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(e))

    def stop(self):
        self._stop = True


class SegmentationTrainWorker(QThread):
    """Worker thread for training segmentation models from 3D GT data.

    Supports multiple model types:
    - 'affinity': Simple 2D affinity model (fast, good baseline)
    - 'joint': Joint LSD+FFN model (more powerful, better accuracy)
    """

    progress = pyqtSignal(int, int, float)  # epoch, total_epochs, loss
    finished = pyqtSignal(bool, str)  # success, message (checkpoint path or error)
    log = pyqtSignal(str)

    def __init__(
        self,
        train_images_dir: str,
        train_masks_dir: str,
        output_dir: str,
        model_type: str = "affinity",  # 'affinity' or 'joint'
        device: str = "cuda",
        epochs: int = 100,
        batch_size: int = 4,
    ):
        super().__init__()
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self._stop = False

    def run(self):
        if self.model_type == 'joint':
            self._train_joint_model()
        else:
            self._train_affinity_model()

    def _get_versioned_checkpoint_path(self, output_dir: Path, base_name: str) -> Path:
        """
        Get next versioned checkpoint path (e.g., affinity_model_v1.pth, v2, v3...).

        Args:
            output_dir: Directory to save model
            base_name: Base name (e.g., "affinity_model")

        Returns:
            Path to next version (e.g., output_dir/affinity_model_v3.pth)
        """
        # Find existing versions
        existing = list(output_dir.glob(f"{base_name}_v*.pth"))
        if not existing:
            version = 1
        else:
            # Extract version numbers
            versions = []
            for path in existing:
                # Extract number from "base_name_v123.pth"
                stem = path.stem  # e.g., "affinity_model_v2"
                if "_v" in stem:
                    try:
                        v = int(stem.split("_v")[-1])
                        versions.append(v)
                    except ValueError:
                        pass
            version = max(versions) + 1 if versions else 1

        return output_dir / f"{base_name}_v{version}.pth"

    def _train_affinity_model(self):
        """Train simple 2D affinity model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            import numpy as np
            from pathlib import Path
            import tifffile

            self.log.emit("Training Affinity Model...")
            self.log.emit("Loading 3D training data...")

            images_dir = Path(self.train_images_dir)
            masks_dir = Path(self.train_masks_dir)
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(images_dir.glob("*.tif"))
            if not image_files:
                self.finished.emit(False, "No 3D training images found")
                return

            self.log.emit(f"Found {len(image_files)} 3D volumes")

            class VolumeDataset(Dataset):
                def __init__(self, image_files, masks_dir):
                    self.samples = []
                    for img_path in image_files:
                        mask_path = masks_dir / img_path.name
                        if mask_path.exists():
                            self.samples.append((img_path, mask_path))

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    img_path, mask_path = self.samples[idx]
                    image = tifffile.imread(str(img_path)).astype(np.float32)
                    mask = tifffile.imread(str(mask_path)).astype(np.int32)

                    if image.max() > 0:
                        image = image / image.max()

                    if mask.ndim == 3:
                        z_mid = mask.shape[0] // 2
                        image_2d = image[z_mid]
                        mask_2d = mask[z_mid]

                        affinities = np.zeros((2, mask_2d.shape[0], mask_2d.shape[1]), dtype=np.float32)
                        affinities[0, :-1, :] = (mask_2d[:-1, :] == mask_2d[1:, :]) & (mask_2d[:-1, :] > 0)
                        affinities[1, :, :-1] = (mask_2d[:, :-1] == mask_2d[:, 1:]) & (mask_2d[:, :-1] > 0)

                        return torch.from_numpy(image_2d[None]), torch.from_numpy(affinities)
                    else:
                        affinities = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
                        affinities[0, :-1, :] = (mask[:-1, :] == mask[1:, :]) & (mask[:-1, :] > 0)
                        affinities[1, :, :-1] = (mask[:, :-1] == mask[:, 1:]) & (mask[:, :-1] > 0)

                        return torch.from_numpy(image[None]), torch.from_numpy(affinities)

            dataset = VolumeDataset(image_files, masks_dir)
            if len(dataset) == 0:
                self.finished.emit(False, "No matching image/mask pairs found")
                return

            self.log.emit(f"Training on {len(dataset)} samples")
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            from moss_full.models.architectures.unet_affinities import UNetAffinities
            model = UNetAffinities(n_channels=1, n_classes=2)
            model = model.to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            best_loss = float('inf')
            checkpoint_path = self._get_versioned_checkpoint_path(output_dir, "affinity_model")
            self.log.emit(f"Will save to: {checkpoint_path.name}")

            for epoch in range(self.epochs):
                if self._stop:
                    self.log.emit("Training stopped by user")
                    break

                model.train()
                epoch_loss = 0.0
                num_batches = 0

                for images, targets in dataloader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)

                    if outputs.shape != targets.shape:
                        min_h = min(outputs.shape[2], targets.shape[2])
                        min_w = min(outputs.shape[3], targets.shape[3])
                        outputs = outputs[:, :, :min_h, :min_w]
                        targets = targets[:, :, :min_h, :min_w]

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                self.progress.emit(epoch + 1, self.epochs, avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        'model_type': 'affinity',
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss,
                    }, checkpoint_path)

                if (epoch + 1) % 10 == 0:
                    self.log.emit(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

            self.log.emit(f"Affinity training complete! Best loss: {best_loss:.4f}")
            self.finished.emit(True, str(checkpoint_path))

        except Exception as e:
            import traceback
            self.log.emit(f"Error: {e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(e))

    def _train_joint_model(self):
        """Train Joint LSD+FFN model for more accurate segmentation."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            import numpy as np
            from pathlib import Path
            import tifffile
            from scipy.ndimage import binary_dilation, binary_erosion

            self.log.emit("Training Joint LSD+FFN Model...")
            self.log.emit("This model outputs affinities, LSDs, and boundary predictions")
            self.log.emit("Loading 3D training data...")

            images_dir = Path(self.train_images_dir)
            masks_dir = Path(self.train_masks_dir)
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(images_dir.glob("*.tif"))
            if not image_files:
                self.finished.emit(False, "No 3D training images found")
                return

            self.log.emit(f"Found {len(image_files)} 3D volumes")

            class JointDataset(Dataset):
                """Dataset that provides targets for Joint LSD+FFN model."""

                def __init__(self, image_files, masks_dir):
                    self.samples = []
                    for img_path in image_files:
                        mask_path = masks_dir / img_path.name
                        if mask_path.exists():
                            self.samples.append((img_path, mask_path))

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    img_path, mask_path = self.samples[idx]
                    image = tifffile.imread(str(img_path)).astype(np.float32)
                    mask = tifffile.imread(str(mask_path)).astype(np.int32)

                    if image.max() > 0:
                        image = image / image.max()

                    # For 3D volumes, use all slices
                    if mask.ndim == 3:
                        d, h, w = mask.shape

                        # Compute 3D affinities (12 channels: 3 short + 9 long range)
                        affinities = self._compute_affinities_3d(mask)

                        # Compute LSDs (10 channels: local shape descriptors)
                        lsds = self._compute_lsds_3d(mask)

                        # Compute boundaries (3 channels: z, y, x boundaries)
                        boundaries = self._compute_boundaries_3d(mask)

                        return (
                            torch.from_numpy(image[None]),  # (1, D, H, W)
                            torch.from_numpy(affinities),   # (12, D, H, W)
                            torch.from_numpy(lsds),         # (10, D, H, W)
                            torch.from_numpy(boundaries),   # (3, D, H, W)
                        )
                    else:
                        # 2D fallback - expand to pseudo-3D
                        h, w = mask.shape
                        image_3d = image[None, :, :]  # (1, H, W)
                        mask_3d = mask[None, :, :]    # (1, H, W)

                        affinities = self._compute_affinities_3d(mask_3d)
                        lsds = self._compute_lsds_3d(mask_3d)
                        boundaries = self._compute_boundaries_3d(mask_3d)

                        return (
                            torch.from_numpy(image_3d[None]),
                            torch.from_numpy(affinities),
                            torch.from_numpy(lsds),
                            torch.from_numpy(boundaries),
                        )

                def _compute_affinities_3d(self, mask):
                    """Compute 3D affinities (12 channels)."""
                    d, h, w = mask.shape
                    # Short-range offsets: z, y, x at distance 1
                    # Long-range: 3, 9, 27 in each dimension
                    offsets = [
                        (1, 0, 0), (0, 1, 0), (0, 0, 1),    # short-range
                        (3, 0, 0), (0, 3, 0), (0, 0, 3),    # medium
                        (9, 0, 0), (0, 9, 0), (0, 0, 9),    # long
                        (27, 0, 0), (0, 27, 0), (0, 0, 27), # very long
                    ]

                    affinities = np.zeros((12, d, h, w), dtype=np.float32)

                    for i, (dz, dy, dx) in enumerate(offsets):
                        if dz > 0:
                            affinities[i, :-dz, :, :] = (
                                (mask[:-dz, :, :] == mask[dz:, :, :]) &
                                (mask[:-dz, :, :] > 0)
                            ).astype(np.float32)
                        elif dy > 0:
                            affinities[i, :, :-dy, :] = (
                                (mask[:, :-dy, :] == mask[:, dy:, :]) &
                                (mask[:, :-dy, :] > 0)
                            ).astype(np.float32)
                        elif dx > 0:
                            affinities[i, :, :, :-dx] = (
                                (mask[:, :, :-dx] == mask[:, :, dx:]) &
                                (mask[:, :, :-dx] > 0)
                            ).astype(np.float32)

                    return affinities

                def _compute_lsds_3d(self, mask):
                    """Compute local shape descriptors (10 channels)."""
                    from scipy.ndimage import gaussian_filter

                    d, h, w = mask.shape
                    lsds = np.zeros((10, d, h, w), dtype=np.float32)
                    sigma = (2.0, 4.0, 4.0)  # Anisotropic sigma

                    # Channels 0-2: Mean offset to segment center (normalized)
                    coords = np.indices(mask.shape, dtype=np.float32)
                    fg_mask = (mask > 0).astype(np.float64)

                    for dim in range(3):
                        coord_masked = coords[dim].copy()
                        coord_masked[mask == 0] = 0

                        coord_smooth = gaussian_filter(coord_masked.astype(np.float64), sigma=sigma)
                        mask_smooth = gaussian_filter(fg_mask, sigma=sigma)
                        mask_smooth = np.maximum(mask_smooth, 1e-6)

                        mean_coord = coord_smooth / mask_smooth
                        offset = coords[dim] - mean_coord
                        offset[mask == 0] = 0
                        lsds[dim] = np.tanh(offset / sigma[dim]).astype(np.float32)

                    # Channels 3-5: Variance in each dimension
                    for dim in range(3):
                        offset = coords[dim] - gaussian_filter(
                            (coords[dim] * fg_mask),
                            sigma=sigma
                        ) / np.maximum(gaussian_filter(fg_mask, sigma=sigma), 1e-6)

                        var_d = gaussian_filter(
                            (offset ** 2 * fg_mask),
                            sigma=sigma
                        ) / np.maximum(gaussian_filter(fg_mask, sigma=sigma), 1e-6)

                        lsds[3 + dim] = np.tanh(np.sqrt(var_d) / sigma[dim]).astype(np.float32)

                    # Channels 6-8: Covariance (simplified)
                    lsds[6:9] = 0  # Simplified: set to zero

                    # Channel 9: Local size
                    size_smooth = gaussian_filter(fg_mask, sigma=sigma)
                    lsds[9] = np.tanh(size_smooth).astype(np.float32)

                    # Zero out background
                    lsds[:, mask == 0] = 0

                    return lsds

                def _compute_boundaries_3d(self, mask):
                    """Compute boundary maps (3 channels for z, y, x)."""
                    d, h, w = mask.shape
                    boundaries = np.zeros((3, d, h, w), dtype=np.float32)

                    # Z boundaries
                    if d > 1:
                        boundaries[0, :-1, :, :] = (mask[:-1, :, :] != mask[1:, :, :]).astype(np.float32)

                    # Y boundaries
                    boundaries[1, :, :-1, :] = (mask[:, :-1, :] != mask[:, 1:, :]).astype(np.float32)

                    # X boundaries
                    boundaries[2, :, :, :-1] = (mask[:, :, :-1] != mask[:, :, 1:]).astype(np.float32)

                    return boundaries

            dataset = JointDataset(image_files, masks_dir)
            if len(dataset) == 0:
                self.finished.emit(False, "No matching image/mask pairs found")
                return

            self.log.emit(f"Training on {len(dataset)} samples")

            # Use smaller batch size for 3D data
            effective_batch_size = max(1, self.batch_size // 2)
            dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)

            # Create Joint LSD+FFN model
            from segmentation_suite.em_pipeline.models.joint_lsd_ffn import JointLSDFFN, JointModelConfig

            config = JointModelConfig(
                in_channels=1,
                base_features=32,
                depth=4,
                num_lsd_affinities=12,
                num_lsd_descriptors=10,
                num_ffn_outputs=3,
            )
            model = JointLSDFFN(config)
            model = model.to(self.device)

            self.log.emit(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # Loss functions for each output
            affinity_criterion = nn.BCELoss()
            lsd_criterion = nn.MSELoss()
            boundary_criterion = nn.BCELoss()

            best_loss = float('inf')
            checkpoint_path = self._get_versioned_checkpoint_path(output_dir, "joint_lsd_affinity_model")
            self.log.emit(f"Will save to: {checkpoint_path.name}")

            for epoch in range(self.epochs):
                if self._stop:
                    self.log.emit("Training stopped by user")
                    break

                model.train()
                epoch_loss = 0.0
                num_batches = 0

                for batch in dataloader:
                    images, target_affinities, target_lsds, target_boundaries = batch
                    images = images.to(self.device)
                    target_affinities = target_affinities.to(self.device)
                    target_lsds = target_lsds.to(self.device)
                    target_boundaries = target_boundaries.to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    output = model(images)

                    # Compute losses
                    aff_loss = affinity_criterion(output.lsd_affinities, target_affinities)
                    lsd_loss = lsd_criterion(output.lsds, target_lsds)
                    bnd_loss = boundary_criterion(output.ffn_boundaries, target_boundaries)

                    # Combined loss (weighted)
                    loss = aff_loss + 0.5 * lsd_loss + bnd_loss

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                self.progress.emit(epoch + 1, self.epochs, avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        'model_type': 'joint',
                        'config': {
                            'in_channels': config.in_channels,
                            'base_features': config.base_features,
                            'depth': config.depth,
                            'num_lsd_affinities': config.num_lsd_affinities,
                            'num_lsd_descriptors': config.num_lsd_descriptors,
                            'num_ffn_outputs': config.num_ffn_outputs,
                        },
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss,
                    }, checkpoint_path)

                if (epoch + 1) % 10 == 0:
                    self.log.emit(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

            self.log.emit(f"Joint model training complete! Best loss: {best_loss:.4f}")
            self.finished.emit(True, str(checkpoint_path))

        except Exception as e:
            import traceback
            self.log.emit(f"Error: {e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(e))

    def stop(self):
        self._stop = True


class SegmentationCombinedPage(QWidget):
    """Combined segmentation page with MOSS (recommended) and LSD 2D options."""

    segmentation_complete = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_dir = None
        self.conversion_worker = None
        self.segmentation_worker = None
        self.train_worker = None
        self.trained_model_path = None
        self._is_busy = False
        # MOSS workflow workers
        self.reslice_worker = None
        self.predict_worker = None
        self.rotation_worker = None
        self.voting_worker = None
        self._moss_workflow_config = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(scaled(15))

        # Title
        title = QLabel("Segmentation")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        # Method selection
        method_group = QGroupBox("Segmentation Method")
        method_group.setStyleSheet(self._group_style())
        method_layout = QVBoxLayout(method_group)

        self.method_group = QButtonGroup(self)

        # Option 1: MOSS (recommended)
        self.moss_radio = QRadioButton("MOSS (Recommended)")
        self.moss_radio.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.moss_radio.setChecked(True)
        self.method_group.addButton(self.moss_radio, 0)
        method_layout.addWidget(self.moss_radio)

        moss_desc = QLabel(
            "   Uses your trained MOSS model with multi-view consensus.\n"
            "   Best for: General-purpose segmentation, trained on your annotations."
        )
        moss_desc.setStyleSheet("color: #888888; margin-left: 20px;")
        method_layout.addWidget(moss_desc)

        # Option 2: LSD 2D
        self.lsd_radio = QRadioButton("LSD 2D")
        self.lsd_radio.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.method_group.addButton(self.lsd_radio, 1)
        method_layout.addWidget(self.lsd_radio)

        lsd_desc = QLabel(
            "   Uses Local Shape Descriptors for 3D neuron segmentation.\n"
            "   Best for: Neuron instance segmentation in large EM volumes."
        )
        lsd_desc.setStyleSheet("color: #888888; margin-left: 20px;")
        method_layout.addWidget(lsd_desc)

        self.method_group.buttonClicked.connect(self._on_method_changed)
        layout.addWidget(method_group)

        # Stacked options for each method - wrapped in scroll area
        self.options_stack = QStackedWidget()

        # MOSS Options (index 0 — matches moss_radio)
        moss_widget = self._create_moss_options()
        moss_scroll = QScrollArea()
        moss_scroll.setWidget(moss_widget)
        moss_scroll.setWidgetResizable(True)
        moss_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.options_stack.addWidget(moss_scroll)

        # LSD Options (index 1 — matches lsd_radio)
        lsd_widget = self._create_lsd_options()
        lsd_scroll = QScrollArea()
        lsd_scroll.setWidget(lsd_widget)
        lsd_scroll.setWidgetResizable(True)
        lsd_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.options_stack.addWidget(lsd_scroll)

        layout.addWidget(self.options_stack)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_group.setStyleSheet(self._group_style())
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        progress_layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(scaled(100))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        progress_layout.addWidget(self.log_text)

        layout.addWidget(progress_group)

        # Run button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.run_btn = QPushButton("Run Segmentation")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                border: none;
                padding: 12px 32px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #1B5E20; color: #666666; }
        """)
        self.run_btn.clicked.connect(self._run_segmentation)
        btn_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def _create_lsd_options(self) -> QWidget:
        """Create LSD 3D options panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, scaled(10), 0, 0)

        # Data input - simplified, uses Zarr from home tab
        data_group = QGroupBox("Input Data")
        data_group.setStyleSheet(self._group_style())
        data_layout = QVBoxLayout(data_group)

        # Zarr volume (primary - from home tab)
        zarr_row = QHBoxLayout()
        zarr_row.addWidget(QLabel("Zarr Volume:"))
        self.zarr_label = QLabel("Not configured - set up in Home tab")
        self.zarr_label.setStyleSheet("color: #888888;")
        zarr_row.addWidget(self.zarr_label, 1)
        self.browse_zarr_btn = QPushButton("Browse")
        self.browse_zarr_btn.clicked.connect(self._browse_zarr)
        self.browse_zarr_btn.setToolTip("Override with a different Zarr volume")
        zarr_row.addWidget(self.browse_zarr_btn)
        data_layout.addLayout(zarr_row)

        # Hidden TIFF conversion section (only shown if needed)
        self.tiff_section = QWidget()
        tiff_layout = QVBoxLayout(self.tiff_section)
        tiff_layout.setContentsMargins(0, 0, 0, 0)

        tiff_row = QHBoxLayout()
        tiff_row.addWidget(QLabel("TIFF Directory:"))
        self.tiff_label = QLabel("Not selected")
        self.tiff_label.setStyleSheet("color: #888888;")
        tiff_row.addWidget(self.tiff_label, 1)
        self.browse_tiff_btn = QPushButton("Browse")
        self.browse_tiff_btn.clicked.connect(self._browse_tiff)
        tiff_row.addWidget(self.browse_tiff_btn)
        tiff_layout.addLayout(tiff_row)

        self.convert_btn = QPushButton("Convert TIFF → Zarr")
        self.convert_btn.clicked.connect(self._start_conversion)
        self.convert_btn.setEnabled(False)
        tiff_layout.addWidget(self.convert_btn)

        self.tiff_section.setVisible(False)  # Hidden by default
        data_layout.addWidget(self.tiff_section)

        layout.addWidget(data_group)

        # Model Training section (uses 3D GT data from Ground Truth tab)
        train_group = QGroupBox("Model Training (Optional)")
        train_group.setStyleSheet(self._group_style())
        train_layout = QVBoxLayout(train_group)

        train_desc = QLabel(
            "Train a segmentation model from 3D ground truth data created in the Ground Truth tab.\n"
            "This produces a model customized for your specific data."
        )
        train_desc.setStyleSheet("color: #888888;")
        train_desc.setWordWrap(True)
        train_layout.addWidget(train_desc)

        # 3D GT data status
        gt_row = QHBoxLayout()
        gt_row.addWidget(QLabel("3D Training Data:"))
        self.gt_3d_label = QLabel("Not found")
        self.gt_3d_label.setStyleSheet("color: #888888;")
        gt_row.addWidget(self.gt_3d_label, 1)
        train_layout.addLayout(gt_row)

        # Model type selection
        model_type_row = QHBoxLayout()
        model_type_row.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("Affinity (Fast, 2D)", "affinity")
        self.model_type_combo.addItem("3D LSD+Affinities (Accurate, 3D)", "joint")
        self.model_type_combo.setCurrentIndex(1)  # Default to 3D LSD (better)
        self.model_type_combo.setToolTip(
            "Affinity: Fast 2D affinity model (2 channels), good baseline\n"
            "3D LSD+Affinities: Full 3D model (12 affinities + 10 LSDs + 3 boundaries)"
        )
        model_type_row.addWidget(self.model_type_combo)
        model_type_row.addStretch()
        train_layout.addLayout(model_type_row)

        # Trained model status
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Trained Model:"))
        self.trained_model_label = QLabel("Not trained")
        self.trained_model_label.setStyleSheet("color: #888888;")
        model_row.addWidget(self.trained_model_label, 1)

        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.setEnabled(False)
        self.train_model_btn.clicked.connect(self._start_model_training)
        self.train_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #424242; color: #666666; }
        """)
        model_row.addWidget(self.train_model_btn)
        train_layout.addLayout(model_row)

        # Training progress
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setVisible(False)
        train_layout.addWidget(self.train_progress_bar)

        self.train_status_label = QLabel("")
        self.train_status_label.setStyleSheet("color: #888888;")
        self.train_status_label.setVisible(False)
        train_layout.addWidget(self.train_status_label)

        layout.addWidget(train_group)

        # Strategy settings
        settings_group = QGroupBox("LSD Settings")
        settings_group.setStyleSheet(self._group_style())
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["lsd", "lsd3d", "ensemble", "joint"])
        self.strategy_combo.setToolTip(
            "lsd: Chunked processing (good for very large volumes)\n"
            "lsd3d: Full-volume 3D watershed (best for continuity, v10 approach)\n"
            "ensemble: Multiple models\n"
            "joint: Joint LSD+FFN"
        )
        self.strategy_combo.setMinimumWidth(scaled(120))
        settings_layout.addWidget(self.strategy_combo)

        settings_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["fast", "balanced", "accurate"])
        self.quality_combo.setCurrentText("balanced")
        self.quality_combo.setMinimumWidth(scaled(120))
        settings_layout.addWidget(self.quality_combo)

        settings_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self._populate_devices()
        self.device_combo.setMinimumWidth(scaled(150))
        settings_layout.addWidget(self.device_combo)

        # Test LSD button
        self.test_lsd_btn = QPushButton("Test LSD")
        self.test_lsd_btn.setToolTip("Preview LSD predictions with adjustable parameters")
        self.test_lsd_btn.clicked.connect(self._open_lsd_test_window)
        self.test_lsd_btn.setMinimumWidth(scaled(100))
        settings_layout.addWidget(self.test_lsd_btn)

        settings_layout.addStretch()
        layout.addWidget(settings_group)

        # Advanced LSD Parameters
        advanced_group = QGroupBox("Advanced LSD Parameters")
        advanced_group.setStyleSheet(self._group_style())
        advanced_layout = QGridLayout(advanced_group)

        # Row 0: min_distance_value, raw_avg_window
        advanced_layout.addWidget(QLabel("Min Distance Value:"), 0, 0)
        self.min_distance_spin = QSpinBox()
        self.min_distance_spin.setRange(1, 20)
        self.min_distance_spin.setValue(7)
        self.min_distance_spin.setToolTip("Seeds must be ≥ this distance from boundaries (v10 default: 7)")
        advanced_layout.addWidget(self.min_distance_spin, 0, 1)

        advanced_layout.addWidget(QLabel("Raw Avg Window:"), 0, 2)
        self.raw_avg_window_spin = QSpinBox()
        self.raw_avg_window_spin.setRange(1, 11)
        self.raw_avg_window_spin.setSingleStep(2)  # Force odd values
        self.raw_avg_window_spin.setValue(3)
        self.raw_avg_window_spin.setToolTip("Z-sliding-window averaging before prediction (must be odd, v10 default: 3)")
        advanced_layout.addWidget(self.raw_avg_window_spin, 0, 3)

        # Row 1: clahe_clip_limit, min_segment_size
        advanced_layout.addWidget(QLabel("CLAHE Clip Limit:"), 1, 0)
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.001, 0.1)
        self.clahe_clip_spin.setSingleStep(0.005)
        self.clahe_clip_spin.setDecimals(3)
        self.clahe_clip_spin.setValue(0.01)
        self.clahe_clip_spin.setToolTip("CLAHE contrast enhancement strength (lower = more aggressive, v10 default: 0.01)")
        advanced_layout.addWidget(self.clahe_clip_spin, 1, 1)

        advanced_layout.addWidget(QLabel("Min Segment Size:"), 1, 2)
        self.min_segment_size_spin = QSpinBox()
        self.min_segment_size_spin.setRange(10, 10000)
        self.min_segment_size_spin.setSingleStep(10)
        self.min_segment_size_spin.setValue(100)
        self.min_segment_size_spin.setToolTip("Remove segments smaller than this (voxels, v10 default: 100)")
        advanced_layout.addWidget(self.min_segment_size_spin, 1, 3)

        # Row 2: boundary_amplification, gamma
        advanced_layout.addWidget(QLabel("Boundary Amplification:"), 2, 0)
        self.boundary_amp_spin = QDoubleSpinBox()
        self.boundary_amp_spin.setRange(0.5, 5.0)
        self.boundary_amp_spin.setSingleStep(0.1)
        self.boundary_amp_spin.setDecimals(1)
        self.boundary_amp_spin.setValue(2.0)
        self.boundary_amp_spin.setToolTip("Multiply boundaries before gamma (v10 default: 2.0)")
        advanced_layout.addWidget(self.boundary_amp_spin, 2, 1)

        advanced_layout.addWidget(QLabel("Gamma:"), 2, 2)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(1.0, 10.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setDecimals(1)
        self.gamma_spin.setValue(4.0)
        self.gamma_spin.setToolTip("Boundary sharpening exponent (v10 default: 4.0)")
        advanced_layout.addWidget(self.gamma_spin, 2, 3)

        # Row 3: seed_threshold, min_seed_distance
        advanced_layout.addWidget(QLabel("Seed Threshold:"), 3, 0)
        self.seed_threshold_spin = QDoubleSpinBox()
        self.seed_threshold_spin.setRange(0.3, 0.9)
        self.seed_threshold_spin.setSingleStep(0.05)
        self.seed_threshold_spin.setDecimals(2)
        self.seed_threshold_spin.setValue(0.7)
        self.seed_threshold_spin.setToolTip("Interior threshold for seed detection (v10 default: 0.7)")
        advanced_layout.addWidget(self.seed_threshold_spin, 3, 1)

        advanced_layout.addWidget(QLabel("Min Seed Distance:"), 3, 2)
        self.min_seed_distance_spin = QSpinBox()
        self.min_seed_distance_spin.setRange(5, 30)
        self.min_seed_distance_spin.setValue(15)
        self.min_seed_distance_spin.setToolTip("Minimum distance between seeds (v10 default: 15)")
        advanced_layout.addWidget(self.min_seed_distance_spin, 3, 3)

        layout.addWidget(advanced_group)

        layout.addStretch()
        return widget

    def _create_moss_options(self) -> QWidget:
        """Create MOSS 2D options panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, scaled(10), 0, 0)

        # Subproject selector (hidden until project with subprojects is loaded)
        self._moss_sp_group = QGroupBox("Subproject")
        self._moss_sp_group.setStyleSheet(self._group_style())
        sp_layout = QHBoxLayout(self._moss_sp_group)
        sp_layout.addWidget(QLabel("Segment using:"))
        self._moss_sp_combo = QComboBox()
        self._moss_sp_combo.setMinimumWidth(scaled(150))
        self._moss_sp_combo.currentTextChanged.connect(self._on_moss_subproject_changed)
        sp_layout.addWidget(self._moss_sp_combo, 1)
        self._moss_sp_group.setVisible(False)
        layout.addWidget(self._moss_sp_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_group.setStyleSheet(self._group_style())
        model_layout = QVBoxLayout(model_group)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Trained Model:"))
        self.moss_model_label = QLabel("Not found")
        self.moss_model_label.setStyleSheet("color: #888888;")
        model_row.addWidget(self.moss_model_label, 1)
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self._browse_moss_model)
        model_row.addWidget(self.browse_model_btn)
        model_layout.addLayout(model_row)

        layout.addWidget(model_group)

        # Reslicing options
        reslice_group = QGroupBox("Reslicing Views")
        reslice_group.setStyleSheet(self._group_style())
        reslice_layout = QVBoxLayout(reslice_group)

        reslice_info = QLabel("Select which views to use for multi-view consensus:")
        reslice_info.setStyleSheet("color: #888888; font-size: 11px;")
        reslice_layout.addWidget(reslice_info)

        # Primary views
        self.moss_xy_check = QCheckBox("XY (original orientation)")
        self.moss_xy_check.setChecked(True)
        reslice_layout.addWidget(self.moss_xy_check)

        self.moss_xz_check = QCheckBox("XZ (side view along Y)")
        self.moss_xz_check.setChecked(True)
        reslice_layout.addWidget(self.moss_xz_check)

        self.moss_yz_check = QCheckBox("YZ (side view along X)")
        self.moss_yz_check.setChecked(True)
        reslice_layout.addWidget(self.moss_yz_check)

        # Diagonal views (collapsed by default)
        diag_label = QLabel("Diagonal views (optional, slower):")
        diag_label.setStyleSheet("color: #888888; margin-top: 8px;")
        reslice_layout.addWidget(diag_label)

        self.moss_diag_zx45_check = QCheckBox("Diagonal ZX 45°")
        reslice_layout.addWidget(self.moss_diag_zx45_check)

        self.moss_diag_zy45_check = QCheckBox("Diagonal ZY 45°")
        reslice_layout.addWidget(self.moss_diag_zy45_check)

        self.moss_diag_zx30_check = QCheckBox("Diagonal ZX 30°")
        reslice_layout.addWidget(self.moss_diag_zx30_check)

        layout.addWidget(reslice_group)

        # Input selection
        input_group = QGroupBox("Input Volume")
        input_group.setStyleSheet(self._group_style())
        input_layout = QVBoxLayout(input_group)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Raw Images:"))
        self.moss_input_label = QLabel("Not selected")
        self.moss_input_label.setStyleSheet("color: #888888;")
        input_row.addWidget(self.moss_input_label, 1)
        self.moss_input_btn = QPushButton("Browse")
        self.moss_input_btn.clicked.connect(self._browse_moss_input)
        input_row.addWidget(self.moss_input_btn)
        input_layout.addLayout(input_row)

        layout.addWidget(input_group)

        layout.addStretch()
        return widget

    def _group_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """

    def _populate_devices(self):
        """Populate device dropdown, preferring GPU."""
        self.device_combo.clear()

        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                    self.device_combo.addItem(f"cuda:{i} ({name}, {mem:.0f}GB)", f"cuda:{i}")
                # GPU is default - don't add CPU as first option
                self.device_combo.addItem("CPU (slower)", "cpu")
                return
        except ImportError:
            pass

        self.device_combo.addItem("CPU", "cpu")

    def _on_method_changed(self, button):
        """Handle method selection change."""
        if button == self.moss_radio:
            self.options_stack.setCurrentIndex(0)
        else:
            self.options_stack.setCurrentIndex(1)

    def set_config(self, config: dict):
        """Set configuration from project."""
        self.project_dir = Path(config.get('project_dir', '')) if config.get('project_dir') else None

        # Populate subproject combo if project uses subprojects
        self._refresh_subproject_combo()

        # Use Zarr path from home tab if available
        zarr_path = config.get('zarr_path') or config.get('train_images_zarr')
        if zarr_path and Path(zarr_path).exists():
            self.zarr_label.setText(str(zarr_path))
            self.zarr_label.setStyleSheet("color: #4CAF50;")
            self.run_btn.setEnabled(True)
            self._log(f"Using Zarr volume from project: {zarr_path}")
        else:
            # Fall back to scanning project directory
            self._scan_for_data()

        # Check for trained MOSS model — prefer subproject checkpoint
        self._update_moss_model_from_subproject()
        if self.moss_model_label.text() in ("Not found", ""):
            if config.get('checkpoint_path'):
                self.moss_model_label.setText(config['checkpoint_path'])
                self.moss_model_label.setStyleSheet("color: #4CAF50;")

    def _scan_for_data(self):
        """Scan project directory for data files."""
        if not self.project_dir or not self.project_dir.exists():
            return

        zarr_found = False

        # Look for Zarr data first (preferred)
        for zarr_dir in self.project_dir.glob("*.zarr"):
            if (zarr_dir / '.zarray').exists() or (zarr_dir / '.zgroup').exists():
                self.zarr_label.setText(str(zarr_dir))
                self.zarr_label.setStyleSheet("color: #4CAF50;")
                self.run_btn.setEnabled(True)
                self._log(f"Found Zarr volume: {zarr_dir}")
                # Also set MOSS input to this Zarr
                self.moss_input_label.setText(str(zarr_dir))
                self.moss_input_label.setStyleSheet("color: #4CAF50;")
                zarr_found = True
                break

        # Also check for raw_data.zarr and train_images.zarr specifically
        if not zarr_found:
            # Check raw_data.zarr first (new naming), then train_images.zarr (backwards compatibility)
            for zarr_name in ["raw_data.zarr", "train_images.zarr"]:
                zarr_candidate = self.project_dir / zarr_name
                if zarr_candidate.exists():
                    self.zarr_label.setText(str(zarr_candidate))
                    self.zarr_label.setStyleSheet("color: #4CAF50;")
                    self.run_btn.setEnabled(True)
                    self._log(f"Found Zarr volume: {zarr_candidate}")
                    # Also set MOSS input to this Zarr
                    self.moss_input_label.setText(str(zarr_candidate))
                    zarr_found = True
                    break
                self.moss_input_label.setStyleSheet("color: #4CAF50;")
                zarr_found = True

        # Only show TIFF section if no Zarr found
        if not zarr_found:
            self.tiff_section.setVisible(True)
            # Look for TIFF data
            for subdir in ['data', 'raw', 'images', 'tiff', '.']:
                check_dir = self.project_dir / subdir
                if check_dir.exists():
                    tiff_files = list(check_dir.glob("*.tif")) + list(check_dir.glob("*.tiff"))
                    if tiff_files:
                        self.tiff_label.setText(str(check_dir))
                        self.tiff_label.setStyleSheet("color: #4CAF50;")
                        self.convert_btn.setEnabled(True)
                        self._log(f"No Zarr found. Found {len(tiff_files)} TIFF files in {check_dir}")
                        break

        # Look for MOSS model
        for model_file in self.project_dir.glob("**/*.pt"):
            self.moss_model_label.setText(str(model_file))
            self.moss_model_label.setStyleSheet("color: #4CAF50;")
            break

        # Look for 3D GT training data
        train_3d_images = self.project_dir / "train_images_3d"
        train_3d_masks = self.project_dir / "train_masks_3d"
        if train_3d_images.exists() and train_3d_masks.exists():
            num_volumes = len(list(train_3d_images.glob("*.tif")))
            if num_volumes > 0:
                self.gt_3d_label.setText(f"{num_volumes} volumes")
                self.gt_3d_label.setStyleSheet("color: #4CAF50;")
                self.train_model_btn.setEnabled(True)
                self._log(f"Found {num_volumes} 3D training volumes")

        # Look for trained models (prefer joint over affinity, use latest version)
        models_dir = self.project_dir / "models"
        if models_dir.exists():
            # Find latest joint model
            joint_models = sorted(models_dir.glob("joint_lsd_affinity_model_v*.pth"))
            # Also check old naming for backwards compatibility
            old_joint = models_dir / "joint_lsd_ffn_model.pth"
            if old_joint.exists():
                joint_models.append(old_joint)

            # Find latest affinity model
            affinity_models = sorted(models_dir.glob("affinity_model_v*.pth"))
            old_affinity = models_dir / "affinity_model.pth"
            if old_affinity.exists():
                affinity_models.append(old_affinity)

            if joint_models:
                latest_joint = joint_models[-1]  # Last in sorted list = highest version
                self.trained_model_path = str(latest_joint)
                self.trained_model_label.setText(f"3D LSD+Affinity Ready ({latest_joint.name})")
                self.trained_model_label.setStyleSheet("color: #4CAF50;")
                self._log(f"Found trained model: {latest_joint.name}")
            elif affinity_models:
                latest_affinity = affinity_models[-1]
                self.trained_model_path = str(latest_affinity)
                self.trained_model_label.setText(f"Affinity Ready ({latest_affinity.name})")
                self.trained_model_label.setStyleSheet("color: #4CAF50;")
                self._log(f"Found trained model: {latest_affinity.name}")

    def _refresh_subproject_combo(self):
        """Populate subproject combo if project has subprojects."""
        if not self.project_dir or not has_subprojects(str(self.project_dir)):
            self._moss_sp_group.setVisible(False)
            return

        self._moss_sp_combo.blockSignals(True)
        self._moss_sp_combo.clear()
        subprojects = list_subprojects(str(self.project_dir))
        active = get_active_subproject(str(self.project_dir))

        for sp_name in subprojects:
            self._moss_sp_combo.addItem(sp_name)

        # Select active subproject
        idx = self._moss_sp_combo.findText(active)
        if idx >= 0:
            self._moss_sp_combo.setCurrentIndex(idx)

        self._moss_sp_combo.blockSignals(False)
        self._moss_sp_group.setVisible(len(subprojects) > 0)

    def _on_moss_subproject_changed(self, sp_name: str):
        """Handle subproject combo change — update model path."""
        if not sp_name or not self.project_dir:
            return
        self._update_moss_model_from_subproject()

    def _update_moss_model_from_subproject(self):
        """Find the best checkpoint in the selected subproject."""
        if not self.project_dir:
            return

        # Determine which directory to scan for checkpoints
        if has_subprojects(str(self.project_dir)) and self._moss_sp_combo.currentText():
            sp_name = self._moss_sp_combo.currentText()
            search_dir = get_subproject_dir(str(self.project_dir), sp_name)
        else:
            search_dir = self.project_dir

        # Find checkpoint_*.pth files (not final, not archived)
        checkpoints = sorted(
            search_dir.glob("checkpoint_*.pth"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        # Filter out archived ones
        checkpoints = [c for c in checkpoints if '_old_' not in c.name]

        if checkpoints:
            best = checkpoints[0]
            self.moss_model_label.setText(str(best))
            self.moss_model_label.setStyleSheet("color: #4CAF50;")
        else:
            self.moss_model_label.setText("Not found")
            self.moss_model_label.setStyleSheet("color: #888888;")

    def _browse_tiff(self):
        """Browse for TIFF directory."""
        # Show TIFF section when user wants to convert
        self.tiff_section.setVisible(True)
        path = QFileDialog.getExistingDirectory(
            self, "Select TIFF Directory",
            str(self.project_dir) if self.project_dir else ""
        )
        if path:
            tiff_files = list(Path(path).glob("*.tif")) + list(Path(path).glob("*.tiff"))
            if tiff_files:
                self.tiff_label.setText(path)
                self.tiff_label.setStyleSheet("color: #4CAF50;")
                self.convert_btn.setEnabled(True)
                self._log(f"Found {len(tiff_files)} TIFF files")
            else:
                QMessageBox.warning(self, "No TIFF Files", "No TIFF files found")

    def _browse_zarr(self):
        """Browse for Zarr volume directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Zarr Volume",
            str(self.project_dir) if self.project_dir else "",
        )
        if path:
            zarr_path = Path(path)
            # Check if it's a valid Zarr store
            if (zarr_path / '.zarray').exists() or (zarr_path / '.zgroup').exists():
                self.zarr_label.setText(str(zarr_path))
                self.zarr_label.setStyleSheet("color: #4CAF50;")
                self.run_btn.setEnabled(True)
                self._log(f"Selected Zarr volume: {zarr_path}")
            else:
                # Maybe they selected a directory containing zarr stores
                zarr_stores = list(zarr_path.glob("*.zarr"))
                if zarr_stores:
                    # Use the first one found
                    self.zarr_label.setText(str(zarr_stores[0]))
                    self.zarr_label.setStyleSheet("color: #4CAF50;")
                    self.run_btn.setEnabled(True)
                    self._log(f"Found Zarr volume: {zarr_stores[0]}")
                else:
                    # Show TIFF conversion option
                    self.tiff_section.setVisible(True)
                    QMessageBox.information(
                        self, "No Zarr Found",
                        "No Zarr volume found in this directory.\n\n"
                        "You can convert TIFF files to Zarr using the conversion option below."
                    )

    def _browse_moss_model(self):
        """Browse for MOSS model checkpoint."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model",
            str(self.project_dir) if self.project_dir else "",
            "Model Files (*.pt *.pth)"
        )
        if path:
            self.moss_model_label.setText(path)
            self.moss_model_label.setStyleSheet("color: #4CAF50;")

    def _browse_moss_input(self):
        """Browse for raw images folder for MOSS segmentation."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Raw Images Folder",
            str(self.project_dir) if self.project_dir else ""
        )
        if folder:
            self.moss_input_label.setText(folder)
            self.moss_input_label.setStyleSheet("color: #4CAF50;")

    def _open_lsd_test_window(self):
        """Open LSD preview window for testing parameters."""
        # Check if zarr volume is available
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "No project directory set")
            return

        # Check for raw_data.zarr first, then train_images.zarr for backwards compatibility
        zarr_path = self.project_dir / "raw_data.zarr"
        if not zarr_path.exists():
            zarr_path = self.project_dir / "train_images.zarr"

        if not zarr_path.exists():
            QMessageBox.warning(
                self, "No Data",
                "No zarr volume found. Set up data in the Home tab first."
            )
            return

        # Get current slice index - default to middle slice
        # (In future, could get from viewport if available)
        try:
            from ..zarr_image_source import ZarrImageSource
            zarr_source = ZarrImageSource(zarr_path)
            num_slices = zarr_source.num_slices
            current_slice = num_slices // 2  # Middle slice
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load zarr volume: {e}"
            )
            return

        # Model path
        model_path = Path('/home/nmedina/projects/em-pipeline/pretrained_models/lsd_mtlsd_checkpoint.pth')
        if not model_path.exists():
            QMessageBox.critical(
                self, "Model Not Found",
                f"LSD model not found at:\n{model_path}"
            )
            return

        # Open preview dialog
        try:
            from ..widgets.lsd_preview_dialog import LSDPreviewDialog
            dialog = LSDPreviewDialog(zarr_source, current_slice, model_path, parent=self)

            # Set initial parameter values from GUI
            dialog.amplification = self.boundary_amp_spin.value()
            dialog.amp_slider.setValue(int(dialog.amplification * 10))
            dialog.amp_value.setText(f"{dialog.amplification:.1f}")

            dialog.gamma = self.gamma_spin.value()
            dialog.gamma_slider.setValue(int(dialog.gamma * 10))
            dialog.gamma_value.setText(f"{dialog.gamma:.1f}")

            dialog.seed_threshold = self.seed_threshold_spin.value()
            dialog.thresh_slider.setValue(int(dialog.seed_threshold * 100))
            dialog.thresh_value.setText(f"{dialog.seed_threshold:.2f}")

            dialog.min_distance = self.min_seed_distance_spin.value()
            dialog.dist_slider.setValue(dialog.min_distance)
            dialog.dist_value.setText(f"{dialog.min_distance}")

            dialog.min_distance_value = self.min_distance_spin.value()
            dialog.min_dist_val_slider.setValue(dialog.min_distance_value)
            dialog.min_dist_val_value.setText(f"{dialog.min_distance_value}")

            dialog.raw_avg_window = self.raw_avg_window_spin.value()
            dialog.raw_avg_slider.setValue(dialog.raw_avg_window)
            dialog.raw_avg_value.setText(f"{dialog.raw_avg_window}")

            dialog.clahe_clip_limit = self.clahe_clip_spin.value()
            dialog.clahe_slider.setValue(int(dialog.clahe_clip_limit * 1000))
            dialog.clahe_value.setText(f"{dialog.clahe_clip_limit:.3f}")

            dialog.min_segment_size = self.min_segment_size_spin.value()
            dialog.min_seg_slider.setValue(dialog.min_segment_size)
            dialog.min_seg_value.setText(f"{dialog.min_segment_size}")

            # Execute dialog and save values back if user clicked OK
            result = dialog.exec()

            # Save parameter values back to GUI (whether OK or Cancel, keep the tested values)
            self.boundary_amp_spin.setValue(dialog.amplification)
            self.gamma_spin.setValue(dialog.gamma)
            self.seed_threshold_spin.setValue(dialog.seed_threshold)
            self.min_seed_distance_spin.setValue(dialog.min_distance)
            self.min_distance_spin.setValue(dialog.min_distance_value)
            self.raw_avg_window_spin.setValue(dialog.raw_avg_window)
            self.clahe_clip_spin.setValue(dialog.clahe_clip_limit)
            self.min_segment_size_spin.setValue(dialog.min_segment_size)
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to open preview: {e}\n\n{type(e).__name__}"
            )
            import traceback
            traceback.print_exc()

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _start_model_training(self):
        """Start training segmentation model from 3D GT data."""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "No project directory set")
            return

        train_images = self.project_dir / "train_images_3d"
        train_masks = self.project_dir / "train_masks_3d"
        output_dir = self.project_dir / "models"

        if not train_images.exists() or not train_masks.exists():
            QMessageBox.warning(
                self, "No Training Data",
                "No 3D training data found. Create 3D ground truth in the Training tab first."
            )
            return

        # Get device and model type from combos
        device = self.device_combo.currentData() or "cpu"
        model_type = self.model_type_combo.currentData() or "affinity"

        # Adjust epochs based on model type
        epochs = 50 if model_type == "affinity" else 100  # Joint needs more epochs

        self._set_busy(True)
        self.train_progress_bar.setVisible(True)
        self.train_status_label.setVisible(True)
        model_name = "Joint LSD+FFN" if model_type == "joint" else "Affinity"
        self.train_status_label.setText(f"Starting {model_name} training...")
        self.train_model_btn.setEnabled(False)

        self.train_worker = SegmentationTrainWorker(
            str(train_images),
            str(train_masks),
            str(output_dir),
            model_type=model_type,
            device=device,
            epochs=epochs,
            batch_size=4 if model_type == "affinity" else 2,  # Smaller batch for 3D
        )
        self.train_worker.progress.connect(self._on_train_progress)
        self.train_worker.finished.connect(self._on_train_finished)
        self.train_worker.log.connect(self._log)
        self.train_worker.start()

    def _on_train_progress(self, epoch: int, total: int, loss: float):
        pct = int(100 * epoch / total)
        self.train_progress_bar.setValue(pct)
        self.train_status_label.setText(f"Epoch {epoch}/{total}, Loss: {loss:.4f}")

    def _on_train_finished(self, success: bool, message: str):
        self._set_busy(False)
        self.train_model_btn.setEnabled(True)

        if success:
            self.trained_model_path = message
            # Determine model type from the saved path
            if "joint" in message.lower():
                self.trained_model_label.setText("Joint LSD+FFN Ready")
            else:
                self.trained_model_label.setText("Affinity Ready")
            self.trained_model_label.setStyleSheet("color: #4CAF50;")
            self.train_status_label.setText("Training complete!")
            self._log(f"Model saved to: {message}")
        else:
            self.train_status_label.setText(f"Training failed: {message}")
            QMessageBox.critical(self, "Training Failed", message)

    def _start_conversion(self):
        """Start TIFF to Zarr conversion."""
        tiff_path = self.tiff_label.text()
        if tiff_path == "Not selected":
            return

        tiff_dir = Path(tiff_path)
        zarr_path = tiff_dir.parent / f"{tiff_dir.name}.zarr"

        # Smart memory-aware settings
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9

        tiff_files = list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff"))
        if tiff_files:
            file_size_mb = tiff_files[0].stat().st_size / 1e6
            if file_size_mb > 100:
                workers, chunk_z = 1, 1
            elif file_size_mb > 50:
                workers, chunk_z = min(2, int(available_gb / 4)), 8
            else:
                workers, chunk_z = min(4, int(available_gb / 2)), 32
            self._log(f"Using {workers} workers, chunk_z={chunk_z} (files ~{file_size_mb:.0f}MB)")
        else:
            workers, chunk_z = 2, 32

        self._set_busy(True)
        self.status_label.setText("Converting TIFF to Zarr...")

        self.conversion_worker = ConversionWorker(
            str(tiff_dir), str(zarr_path), (chunk_z, 512, 512), workers
        )
        self.conversion_worker.progress.connect(self._on_conversion_progress)
        self.conversion_worker.finished.connect(self._on_conversion_finished)
        self.conversion_worker.log.connect(self._log)
        self.conversion_worker.start()

    def _on_conversion_progress(self, completed: int, total: int, msg: str):
        if total > 0:
            pct = int(100 * completed / total)
            self.progress_bar.setValue(pct)
            self.status_label.setText(f"Converting: {completed}/{total}")

    def _on_conversion_finished(self, success: bool, message: str):
        self._set_busy(False)
        if success:
            self._log(f"Conversion complete: {message}")
            self.zarr_label.setText(message)
            self.zarr_label.setStyleSheet("color: #4CAF50;")
            self.run_btn.setEnabled(True)
            self.progress_bar.setValue(100)
            self.status_label.setText("Ready to segment")
        else:
            self._log(f"Conversion failed: {message}")
            self.status_label.setText(f"Error: {message}")
            QMessageBox.critical(self, "Conversion Failed", message)

    def _run_segmentation(self):
        """Run the selected segmentation method."""
        if self.lsd_radio.isChecked():
            self._run_lsd_segmentation()
        else:
            self._run_moss_segmentation()

    def _run_lsd_segmentation(self):
        """Run LSD 3D segmentation."""
        zarr_path = self.zarr_label.text()
        if zarr_path in ("Not found", ""):
            QMessageBox.warning(self, "No Input", "Please convert TIFF to Zarr first")
            return

        input_path = Path(zarr_path)
        output_path = input_path.parent / f"segmentation_{self.strategy_combo.currentText()}.zarr"
        device = self.device_combo.currentData() or "cpu"

        self._set_busy(True)
        self.status_label.setText("Running segmentation...")
        self.progress_bar.setValue(0)

        # Check for trained model (prefer our trained model, then MOSS model)
        model_path = None

        # First check for our trained segmentation model (affinity or joint)
        if self.trained_model_path and Path(self.trained_model_path).exists():
            model_path = self.trained_model_path
            model_name = "Joint LSD+FFN" if "joint" in model_path.lower() else "Affinity"
            self._log(f'Using trained {model_name} model: {model_path}')
        else:
            # Fall back to MOSS model if available
            moss_model_text = self.moss_model_label.text()
            if moss_model_text not in ('Not found', ''):
                if 'affinity' in moss_model_text.lower() or 'affinities' in moss_model_text.lower():
                    model_path = moss_model_text
                    self._log(f'Using MOSS affinity model: {model_path}')
                else:
                    self._log('Note: MOSS model found but not affinity-trained. Using default LSD model.')

        # Collect LSD parameters from GUI
        lsd_params = {
            'min_distance_value': self.min_distance_spin.value(),
            'raw_avg_window': self.raw_avg_window_spin.value(),
            'clahe_clip_limit': self.clahe_clip_spin.value(),
            'min_segment_size': self.min_segment_size_spin.value(),
            'boundary_amplification': self.boundary_amp_spin.value(),
            'gamma': self.gamma_spin.value(),
            'seed_threshold': self.seed_threshold_spin.value(),
            'min_seed_distance': self.min_seed_distance_spin.value(),
        }

        self.segmentation_worker = SegmentationWorker(
            str(zarr_path),
            str(output_path),
            self.strategy_combo.currentText(),
            self.quality_combo.currentText(),
            device,
            model_path,
            lsd_params
        )
        self.segmentation_worker.progress.connect(self._on_seg_progress)
        self.segmentation_worker.finished.connect(self._on_seg_finished)
        self.segmentation_worker.log.connect(self._log)
        self.segmentation_worker.start()

    def _run_moss_segmentation(self):
        """Run MOSS 2D multi-view segmentation."""
        model_path = self.moss_model_label.text()
        if model_path in ("Not found", ""):
            QMessageBox.warning(self, "No Model", "Please select a trained model")
            return

        # Check for input
        input_dir = self.moss_input_label.text()
        if input_dir in ("Not selected", ""):
            # Try to use raw_images from project
            if self.project_dir:
                raw_dir = self.project_dir / "raw_images"
                if raw_dir.exists():
                    input_dir = str(raw_dir)
                    self.moss_input_label.setText(input_dir)
                    self.moss_input_label.setStyleSheet("color: #4CAF50;")
            if input_dir in ("Not selected", ""):
                QMessageBox.warning(self, "No Input", "Please select raw images folder")
                return

        if not Path(input_dir).exists():
            QMessageBox.warning(self, "Invalid Input", f"Folder not found: {input_dir}")
            return

        # Determine which views to use
        views_to_run = []
        if self.moss_xy_check.isChecked():
            views_to_run.append('xy')
        if self.moss_xz_check.isChecked():
            views_to_run.append('xz')
        if self.moss_yz_check.isChecked():
            views_to_run.append('yz')

        diagonals = []
        if self.moss_diag_zx45_check.isChecked():
            diagonals.append({'angle': 45, 'axes': (0, 2), 'name': 'diag_zx45'})
        if self.moss_diag_zy45_check.isChecked():
            diagonals.append({'angle': 45, 'axes': (0, 1), 'name': 'diag_zy45'})
        if self.moss_diag_zx30_check.isChecked():
            diagonals.append({'angle': 30, 'axes': (0, 2), 'name': 'diag_zx30'})

        # Set up output directories
        # Reslices are global (raw data transforms, shared across subprojects)
        project_base = self.project_dir if self.project_dir else Path(input_dir).parent
        reslice_dir = project_base / "reslices"
        # Predictions and heatmaps go to the active subproject
        output_base = project_base
        if (self.project_dir and has_subprojects(str(self.project_dir))
                and self._moss_sp_combo.currentText()):
            sp_name = self._moss_sp_combo.currentText()
            output_base = Path(get_subproject_dir(str(self.project_dir), sp_name))
        predict_dir = output_base / "predictions"
        heatmap_dir = output_base / "heatmap"

        self._set_busy(True)
        self.progress_bar.setValue(0)
        self._log(f"Starting MOSS 2D multi-view segmentation...")
        self._log(f"  Model: {model_path}")
        self._log(f"  Input: {input_dir}")
        self._log(f"  Views: {views_to_run}")
        if diagonals:
            self._log(f"  Diagonals: {[d['name'] for d in diagonals]}")

        # Store config for sequential workflow
        self._moss_workflow_config = {
            'model_path': model_path,
            'input_dir': input_dir,
            'views': views_to_run,
            'diagonals': diagonals,
            'reslice_dir': str(reslice_dir),
            'predict_dir': str(predict_dir),
            'heatmap_dir': str(heatmap_dir),
            'current_step': 'reslice',
        }

        # Check if reslicing is needed
        input_path = Path(input_dir)
        is_zarr = input_path.suffix == '.zarr' or (input_path / '.zarray').exists()
        needs_reslice = len(views_to_run) > 1 or len(diagonals) > 0  # More than just XY

        if is_zarr and not needs_reslice:
            # Zarr input with only XY view - skip reslicing, go straight to prediction
            self._log("XY-only on Zarr - skipping reslice step")
            self._run_moss_predict_step()
        else:
            # Reslicing needed (works with both Zarr and directories now)
            self._run_moss_reslice_step()

    def _check_reslice_complete(self, config):
        """Check if reslice outputs already exist and are complete.

        Returns True if all required reslices are already done.
        """
        from PIL import Image as PILImage
        import numpy as np

        input_dir = Path(config['input_dir'])
        reslice_dir = Path(config['reslice_dir'])

        # Count input images to get z_size
        is_zarr = (input_dir.suffix == '.zarr' or (input_dir / '.zarray').exists()
                   or (input_dir / '.zgroup').exists())

        if is_zarr:
            try:
                from ..zarr_image_source import ZarrImageSource
                zarr_src = ZarrImageSource(str(input_dir))
                z_size = zarr_src.num_slices
                y_size = zarr_src.height
                x_size = zarr_src.width
            except Exception:
                return False
        else:
            input_images = sorted(input_dir.glob("*.tif"))
            if not input_images:
                input_images = sorted(input_dir.glob("*.png"))
            if not input_images:
                return False

            z_size = len(input_images)

            # Get x/y dimensions from first image
            try:
                first_img = np.array(PILImage.open(input_images[0]))
                y_size, x_size = first_img.shape[:2]
            except Exception:
                return False

        # Check XZ reslice
        if 'xz' in config['views']:
            xz_dir = reslice_dir / "xz_reslice"
            if not xz_dir.exists():
                return False
            xz_count = len(list(xz_dir.glob("xz_slice_*.tif")))
            if xz_count < y_size:
                self._log(f"XZ reslice incomplete: {xz_count}/{y_size}")
                return False

        # Check YZ reslice
        if 'yz' in config['views']:
            yz_dir = reslice_dir / "yz_reslice"
            if not yz_dir.exists():
                return False
            yz_count = len(list(yz_dir.glob("yz_slice_*.tif")))
            if yz_count < x_size:
                self._log(f"YZ reslice incomplete: {yz_count}/{x_size}")
                return False

        # Check diagonal reslices
        for diag in config.get('diagonals', []):
            diag_name = diag['name']
            diag_dir = reslice_dir / diag_name
            if not diag_dir.exists():
                return False
            diag_count = len(list(diag_dir.glob(f"{diag_name}_slice_*.tif")))
            if diag_count < z_size:
                self._log(f"{diag_name} reslice incomplete: {diag_count}/{z_size}")
                return False

        return True

    def _run_moss_reslice_step(self):
        """Step 1: Reslice volume into different views."""
        config = self._moss_workflow_config

        # Check if reslicing is already complete
        if self._check_reslice_complete(config):
            self._log("Reslices already exist and are complete — skipping to prediction")
            self._run_moss_predict_step()
            return

        from ..workers.reslice_worker import ResliceWorker

        has_diagonals = len(config['diagonals']) > 0
        step_label = "Step 1/4: Reslicing..." if has_diagonals else "Step 1/3: Reslicing..."
        self.status_label.setText(step_label)

        reslice_config = {
            'input_dir': config['input_dir'],
            'output_dir': config['reslice_dir'],
            'create_xz': 'xz' in config['views'],
            'create_yz': 'yz' in config['views'],
            'diagonals': config['diagonals'],
            'batch_size': 200,
            'max_workers': 8,
        }

        self._log(f"Reslicing volume...")

        self.reslice_worker = ResliceWorker(reslice_config)
        self.reslice_worker.progress.connect(lambda name, cur, tot: self._on_moss_progress(f"Reslicing {name}", int(100 * cur / tot) if tot > 0 else 0))
        self.reslice_worker.log.connect(self._log)
        self.reslice_worker.finished.connect(self._on_moss_reslice_finished)
        self.reslice_worker.start()

    def _on_moss_reslice_finished(self, success: bool, result):
        """Handle reslice completion, start prediction."""
        if not success:
            self._set_busy(False)
            self._log(f"Reslice failed: {result}")
            self.status_label.setText("Reslice failed")
            return

        self._log("Reslicing complete!")
        self._run_moss_predict_step()

    def _run_moss_predict_step(self):
        """Step 2: Run predictions on each view."""
        from ..workers.predict_worker import PredictWorker

        config = self._moss_workflow_config
        has_diagonals = len(config['diagonals']) > 0
        step_label = "Step 2/4: Predicting..." if has_diagonals else "Step 2/3: Predicting..."
        self.status_label.setText(step_label)

        # Build views list for prediction
        views = []

        # XY view - predict on original input (only if selected)
        if 'xy' in config['views']:
            xy_pred_dir = Path(config['predict_dir']) / "xy"
            xy_pred_dir.mkdir(parents=True, exist_ok=True)
            views.append({
                'name': 'xy',
                'input_dir': config['input_dir'],
                'output_dir': str(xy_pred_dir),
            })

        # XZ view
        if 'xz' in config['views']:
            xz_input = Path(config['reslice_dir']) / "xz_reslice"
            xz_pred = Path(config['predict_dir']) / "xz"
            xz_pred.mkdir(parents=True, exist_ok=True)
            if xz_input.exists():
                views.append({
                    'name': 'xz',
                    'input_dir': str(xz_input),
                    'output_dir': str(xz_pred),
                })

        # YZ view
        if 'yz' in config['views']:
            yz_input = Path(config['reslice_dir']) / "yz_reslice"
            yz_pred = Path(config['predict_dir']) / "yz"
            yz_pred.mkdir(parents=True, exist_ok=True)
            if yz_input.exists():
                views.append({
                    'name': 'yz',
                    'input_dir': str(yz_input),
                    'output_dir': str(yz_pred),
                })

        # Diagonal views
        for diag in config['diagonals']:
            diag_name = diag['name']
            diag_input = Path(config['reslice_dir']) / diag_name
            diag_pred = Path(config['predict_dir']) / diag_name
            diag_pred.mkdir(parents=True, exist_ok=True)
            if diag_input.exists():
                views.append({
                    'name': diag_name,
                    'input_dir': str(diag_input),
                    'output_dir': str(diag_pred),
                })

        # Detect architecture from checkpoint name
        checkpoint_path = config['model_path']
        checkpoint_lower = checkpoint_path.lower()
        architecture = 'unet'

        # Check for specific architectures (order matters - most specific first)
        if 'unet_3d' in checkpoint_lower:
            architecture = 'unet_3d'
        elif 'unet_deep_dice_dwarf25d_zcoord' in checkpoint_lower or 'dwarf25d_zcoord' in checkpoint_lower:
            architecture = 'unet_deep_dice_dwarf25d_zcoord'
        elif 'unet_deep_dice_dwarf25d_v2' in checkpoint_lower:
            architecture = 'unet_deep_dice_dwarf25d_v2'
        elif 'unet_deep_dice_25d_v2' in checkpoint_lower:
            architecture = 'unet_deep_dice_25d_v2'
        elif 'unet_deep_dice_25d' in checkpoint_lower:
            architecture = 'unet_deep_dice_25d'
        elif 'dwarf25d' in checkpoint_lower:
            architecture = 'unet_deep_dice_dwarf25d_v2'
        elif 'unet_deep_dice_sam2' in checkpoint_lower:
            architecture = 'unet_deep_dice_sam2'
        elif 'unet_deep_dice' in checkpoint_lower:
            architecture = 'unet_deep_dice'
        elif 'unet_affinities' in checkpoint_lower:
            architecture = 'unet_affinities'
        elif 'unet_deep' in checkpoint_lower:
            architecture = 'unet_deep'
        elif 'lsd_boundary' in checkpoint_lower:
            architecture = 'lsd_boundary_2d'
        elif 'sam2' in checkpoint_lower:
            architecture = 'unet_deep_dice_sam2'
        elif '25d' in checkpoint_lower:
            architecture = 'unet_deep_dice_25d'
        elif 'dice' in checkpoint_lower:
            architecture = 'unet_deep_dice'

        # Larger patches = fewer forward passes, but a 1024 patch's activations
        # overflow small-VRAM GPUs (<8 GB), forcing the driver to page GPU memory
        # to system RAM (100% util but ~seconds/slice). Cap to 512 on such cards.
        # Big GPUs and CPU keep 1024 (unchanged).
        import torch as _torch
        try:
            _total_gb = (_torch.cuda.get_device_properties(0).total_memory / 1e9
                         if _torch.cuda.is_available() else 999)
        except Exception:
            _total_gb = 999
        _patch_size = 1024 if _total_gb >= 8 else 512

        predict_config = {
            'checkpoint_path': checkpoint_path,
            'architecture': architecture,
            'views': views,
            'patch_size': _patch_size,
            'overlap': 64,
        }

        self._log(f"Running predictions on {len(views)} views...")
        has_diagonals = len(config['diagonals']) > 0
        base_percent = 25 if has_diagonals else 33
        range_percent = 25 if has_diagonals else 33

        self.predict_worker = PredictWorker(predict_config)
        self.predict_worker.progress.connect(
            lambda name, cur, tot: self._on_moss_progress(f"Predicting {name}", base_percent + (cur * range_percent // max(tot, 1)))
        )
        self.predict_worker.log.connect(self._log)
        self.predict_worker.finished.connect(self._on_moss_predict_finished)
        self.predict_worker.start()

    def _on_moss_predict_finished(self, success: bool, result):
        """Handle prediction completion, start rotation (if diagonals) or voting."""
        if not success:
            self._set_busy(False)
            self._log(f"Prediction failed")
            self.status_label.setText("Prediction failed")
            return

        self._log("Predictions complete!")

        # Check if we have diagonal predictions that need rotation
        config = self._moss_workflow_config
        if config['diagonals']:
            self._log("Rotating diagonal predictions back to XY orientation...")
            self._run_moss_rotation_step()
        else:
            # No diagonals, go straight to voting
            self._run_moss_vote_step()

    def _run_moss_rotation_step(self):
        """Step 3: Rotate diagonal predictions back to XY orientation."""
        from ..workers.rotate_worker import RotateWorker

        config = self._moss_workflow_config
        has_diagonals = len(config['diagonals']) > 0
        step_label = "Step 3/4: Rotating..." if has_diagonals else "Step 3/3: Rotating..."
        self.status_label.setText(step_label)

        # Build diagonal configurations for rotation
        diagonal_configs = []
        for diag in config['diagonals']:
            diag_name = diag['name']
            angle = diag['angle']
            axes = diag['axes']

            # Input: predictions from predict step
            input_dir = Path(config['predict_dir']) / diag_name

            # Output: rotated predictions
            rotated_dir = Path(config['predict_dir']) / f"{diag_name}_rotated"
            rotated_dir.mkdir(parents=True, exist_ok=True)

            diagonal_configs.append({
                'name': diag_name,
                'angle': angle,
                'axes': axes,
                'input_dir': str(input_dir),
                'output_dir': str(rotated_dir),
            })

        # xy_dir for reference dimensions — use XY predictions if available,
        # otherwise fall back to the input image directory
        xy_pred_dir = Path(config['predict_dir']) / "xy"
        if xy_pred_dir.exists() and any(xy_pred_dir.glob("*.tif")):
            xy_ref = str(xy_pred_dir)
        else:
            xy_ref = config['input_dir']

        rotation_config = {
            'xy_dir': xy_ref,
            'diagonals': diagonal_configs,
            'max_workers': 8,
        }

        self._log(f"Rotating {len(diagonal_configs)} diagonal predictions...")
        self.rotation_worker = RotateWorker(rotation_config)
        self.rotation_worker.progress.connect(
            lambda name, cur, tot: self._on_moss_progress(f"Rotating {name}", 50 + (cur * 16 // max(tot, 1)))
        )
        self.rotation_worker.log.connect(self._log)
        self.rotation_worker.finished.connect(self._on_moss_rotation_finished)
        self.rotation_worker.start()

    def _on_moss_rotation_finished(self, success: bool, result):
        """Handle rotation completion, start voting."""
        if not success:
            self._set_busy(False)
            self._log(f"Rotation failed: {result.get('error', 'Unknown error')}")
            self.status_label.setText("Rotation failed")
            return

        self._log("Rotation complete!")
        self._log(f"Rotated predictions: {result}")

        # Store rotated directories for voting step
        self._moss_workflow_config['rotated_dirs'] = result
        self._run_moss_vote_step()

    def _run_moss_vote_step(self):
        """Step 4: Combine predictions via voting (or Step 3 if no rotation)."""
        from ..workers.voting_worker import VotingWorker

        config = self._moss_workflow_config
        has_diagonals = len(config['diagonals']) > 0
        step_label = "Step 4/4: Voting..." if has_diagonals else "Step 3/3: Voting..."
        self.status_label.setText(step_label)

        # Use rotated directories for diagonals if they exist, otherwise use original
        diag_dirs = []
        if 'rotated_dirs' in config:
            # Use rotated directories (after rotation step)
            diag_dirs = list(config['rotated_dirs'].values())
        else:
            # No rotation step (no diagonals)
            diag_dirs = []

        vote_config = {
            'xy_dir': str(Path(config['predict_dir']) / "xy"),
            'xz_dir': str(Path(config['predict_dir']) / "xz") if 'xz' in config['views'] else None,
            'yz_dir': str(Path(config['predict_dir']) / "yz") if 'yz' in config['views'] else None,
            'diag_dirs': diag_dirs,
            'output_dir': config['heatmap_dir'],
            'chunk_size': 200,
            'max_workers': 8,
        }

        self._log(f"Combining predictions via voting...")
        has_diagonals = len(config['diagonals']) > 0
        base_percent = 75 if has_diagonals else 66
        range_percent = 25 if has_diagonals else 34

        self.voting_worker = VotingWorker(vote_config)
        self.voting_worker.progress.connect(
            lambda cur, tot: self._on_moss_progress("Voting", base_percent + (cur * range_percent // max(tot, 1)))
        )
        self.voting_worker.log.connect(self._log)
        self.voting_worker.finished.connect(self._on_moss_vote_finished)
        self.voting_worker.start()

    def _on_moss_vote_finished(self, success: bool, result):
        """Handle voting completion."""
        self._set_busy(False)
        if success:
            self._log(f"MOSS 2D segmentation complete!")
            self._log(f"Output: {result}")
            self.status_label.setText("Complete!")
            self.progress_bar.setValue(100)
            self.segmentation_complete.emit(str(result))
        else:
            self._log(f"Voting failed: {result}")
            self.status_label.setText("Voting failed")

    def _on_moss_progress(self, stage: str, percent: int):
        """Update progress for MOSS workflow."""
        self.progress_bar.setValue(min(percent, 100))
        self.status_label.setText(f"{stage}: {percent}%")

    def _on_seg_progress(self, stage: str, percent: int):
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"{stage}: {percent}%")

    def _on_seg_finished(self, success: bool, message: str):
        self._set_busy(False)
        if success:
            self._log(f"Segmentation complete!")
            self.status_label.setText("Complete!")
            self.progress_bar.setValue(100)
            self.segmentation_complete.emit(message)
        else:
            self._log(f"Failed: {message}")
            self.status_label.setText(f"Error: {message}")
            QMessageBox.critical(self, "Segmentation Failed", message)

    def _cancel(self):
        """Cancel current operation."""
        if self.conversion_worker and self.conversion_worker.isRunning():
            self.conversion_worker.stop()
            self.conversion_worker.wait()
        if self.segmentation_worker and self.segmentation_worker.isRunning():
            self.segmentation_worker.stop()
            self.segmentation_worker.wait()
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.stop()
            self.train_worker.wait()
            self.train_progress_bar.setVisible(False)
            self.train_status_label.setText("Training cancelled")
        # MOSS workers
        if hasattr(self, 'reslice_worker') and self.reslice_worker and self.reslice_worker.isRunning():
            self.reslice_worker.stop()
            self.reslice_worker.wait()
        if hasattr(self, 'predict_worker') and self.predict_worker and self.predict_worker.isRunning():
            self.predict_worker.stop()
            self.predict_worker.wait()
        if hasattr(self, 'rotation_worker') and self.rotation_worker and self.rotation_worker.isRunning():
            self.rotation_worker.stop()
            self.rotation_worker.wait()
        if hasattr(self, 'voting_worker') and self.voting_worker and self.voting_worker.isRunning():
            self.voting_worker.stop()
            self.voting_worker.wait()
        self._set_busy(False)
        self.status_label.setText("Cancelled")

    def _set_busy(self, busy: bool):
        self._is_busy = busy
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.convert_btn.setEnabled(not busy and self.tiff_label.text() != "Not selected")
        self.lsd_radio.setEnabled(not busy)
        self.moss_radio.setEnabled(not busy)
        self.busy_changed.emit(busy)

    def is_busy(self) -> bool:
        return self._is_busy
