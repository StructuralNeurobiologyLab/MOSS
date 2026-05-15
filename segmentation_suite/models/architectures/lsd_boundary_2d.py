#!/usr/bin/env python3
"""
LSD Boundary 2D - Pretrained membrane/boundary prediction using Local Shape Descriptors.

This architecture wraps the pretrained MtLsdModel to provide membrane/boundary predictions
for use in MOSS 2D semantic segmentation workflows. The model outputs boundary probabilities
that can be used as suggestions for membrane annotation.

The pretrained model was trained on EM data and predicts:
- Affinities (2 channels: y, x neighbors)
- LSDs (6 channels: local shape descriptors)

The output is: boundary = 1 - mean(affinities), converted to logits for MOSS compatibility.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Architecture metadata for MOSS discovery
ARCHITECTURE_ID = 'lsd_boundary_2d'
ARCHITECTURE_NAME = 'LSD Boundary 2D (Pretrained)'
ARCHITECTURE_DESCRIPTION = (
    'Pretrained model for membrane/boundary detection using Local Shape Descriptors. '
    'Outputs boundary probabilities for cell membrane segmentation. '
    'No training needed - uses pretrained weights from EM data.'
)
PREFERRED_LOSS = 'bce'

HIDDEN = True

# Path to pretrained checkpoint (relative to this file's parent directory)
import os
from pathlib import Path
_current_dir = Path(__file__).parent.parent.parent.parent  # Go up to merged_moss root
PRETRAINED_CHECKPOINT = str(_current_dir / 'pretrained_models' / 'lsd_mtlsd_checkpoint.pth')


class ConvPass(nn.Module):
    """Two-layer conv block used in the mtlsd UNet."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_pass = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_pass(x)


class OutputHead(nn.Module):
    """Output head matching checkpoint structure (conv_pass.0)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_pass = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.conv_pass(x)


class MtLsdUNet(nn.Module):
    """
    2D UNet matching the pretrained mtlsd checkpoint architecture.

    Architecture:
    - 3 encoder levels with [12, 60, 300] channels
    - 2 decoder levels
    - 2D maxpool (2x2) between levels
    - Bilinear upsampling in decoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fmaps: int = 12,
        fmap_inc_factor: int = 5,
        downsample_factors: Tuple[int, ...] = (2, 2),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors

        # Encoder (left) path
        self.l_conv = nn.ModuleList()
        current_channels = in_channels
        for level in range(len(downsample_factors) + 1):
            out_channels = num_fmaps * (fmap_inc_factor ** level)
            self.l_conv.append(ConvPass(current_channels, out_channels))
            current_channels = out_channels

        # Decoder (right) path
        self.r_conv = nn.ModuleList([
            nn.ModuleList([
                ConvPass(72, 12),   # r_conv[0][0]: 60+12 -> 12 (shallow)
                ConvPass(360, 60),  # r_conv[0][1]: 300+60 -> 60 (deep)
            ])
        ])

    def forward(self, x):
        # Encoder path with skip connections
        skips = []
        for level, conv in enumerate(self.l_conv):
            x = conv(x)
            if level < len(self.downsample_factors):
                skips.append(x)
                x = F.max_pool2d(x, 2)

        # Decoder path
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([skips[1], x], dim=1)
        x = self.r_conv[0][1](x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([skips[0], x], dim=1)
        x = self.r_conv[0][0](x)

        return x


class MtLsdModelInternal(nn.Module):
    """
    Internal MtLsdModel matching the pretrained checkpoint.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fmaps: int = 12,
        fmap_inc_factor: int = 5,
        num_affinities: int = 2,
        num_lsds: int = 6,
    ):
        super().__init__()

        self.unet = MtLsdUNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
        )

        self.aff_head = OutputHead(num_fmaps, num_affinities)
        self.lsd_head = OutputHead(num_fmaps, num_lsds)

        self.num_affinities = num_affinities
        self.num_lsds = num_lsds

    def forward(self, x):
        features = self.unet(x)
        affinities = torch.sigmoid(self.aff_head(features))
        lsds = torch.tanh(self.lsd_head(features))
        return affinities, lsds


class LsdBoundary2D(nn.Module):
    """
    MOSS-compatible wrapper for pretrained LSD boundary prediction.

    This model loads the pretrained MtLsdModel and outputs boundary probabilities
    as logits (so MOSS's sigmoid produces the final boundary prediction).

    Usage:
        - The model automatically loads pretrained weights on initialization
        - Output is boundary logits: sigmoid(output) gives boundary probability
        - Boundaries represent membranes/cell edges in EM images
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1, **kwargs):
        """
        Initialize the LSD boundary model.

        Args:
            n_channels: Number of input channels (must be 1 for grayscale)
            n_classes: Number of output classes (ignored, always 1 for boundaries)
        """
        super().__init__()

        if n_channels != 1:
            print(f"[LsdBoundary2D] Warning: n_channels={n_channels} ignored, using 1")

        # Create the internal MtLsdModel
        self.mtlsd = MtLsdModelInternal(in_channels=1)

        # Track if pretrained weights are loaded
        self._pretrained_loaded = False

    def _ensure_pretrained(self, device=None):
        """Load pretrained weights if not already loaded."""
        if self._pretrained_loaded:
            return

        if os.path.exists(PRETRAINED_CHECKPOINT):
            try:
                if device is None:
                    device = next(self.parameters()).device
                checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=device, weights_only=False)

                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self.mtlsd.load_state_dict(state_dict)
                self._pretrained_loaded = True
                print(f"[LsdBoundary2D] Loaded pretrained weights from {PRETRAINED_CHECKPOINT}")
            except Exception as e:
                print(f"[LsdBoundary2D] Warning: Could not load pretrained weights: {e}")
        else:
            print(f"[LsdBoundary2D] Warning: Pretrained checkpoint not found at {PRETRAINED_CHECKPOINT}")

    def forward(self, x):
        """
        Forward pass returning boundary logits.

        Args:
            x: Input tensor (N, 1, H, W) - grayscale EM image normalized to [0, 1]

        Returns:
            Boundary logits (N, 1, H, W) - apply sigmoid for boundary probability
        """
        # Ensure pretrained weights are loaded
        self._ensure_pretrained(x.device)

        # Get affinities from MtLsdModel
        affinities, lsds = self.mtlsd(x)  # affinities: (N, 2, H, W) already sigmoided

        # Compute boundary as 1 - mean(affinities)
        # High affinity = same cell = low boundary
        # Low affinity = different cells = high boundary (membrane)
        boundary_prob = 1.0 - affinities.mean(dim=1, keepdim=True)  # (N, 1, H, W)

        # Convert to logits for MOSS compatibility
        # MOSS predict_worker applies sigmoid, so we need logits
        # logit(p) = log(p / (1-p))
        # Use larger epsilon to prevent inf values
        eps = 1e-3
        boundary_prob = boundary_prob.clamp(eps, 1 - eps)
        boundary_logits = torch.log(boundary_prob / (1 - boundary_prob))

        # Safety: clip extreme logits to prevent NaN after sigmoid
        boundary_logits = boundary_logits.clamp(-10, 10)

        return boundary_logits

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom state_dict loading to handle pretrained checkpoint format.

        If the state_dict matches MtLsdModel format, load it into mtlsd.
        Otherwise, try standard loading.
        """
        # Check if this looks like the MtLsdModel state dict (has 'unet.' prefix)
        if any(k.startswith('unet.') for k in state_dict.keys()):
            # This is the pretrained MtLsdModel format
            self.mtlsd.load_state_dict(state_dict, strict=strict)
            self._pretrained_loaded = True
        else:
            # Try standard loading (for MOSS-trained checkpoints)
            super().load_state_dict(state_dict, strict=strict)
            self._pretrained_loaded = True


# Export for MOSS architecture discovery
MODEL_CLASS = LsdBoundary2D
