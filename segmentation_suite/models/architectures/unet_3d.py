#!/usr/bin/env python3
"""
3D U-Net for volumetric segmentation.

Takes a 3D volume patch (1, D, H, W) and predicts a 3D segmentation mask.
Uses 3D convolutions throughout for true volumetric context.

Designed for ~16GB GPU memory with patch size (1, 32, 128, 128).
Uses 4 encoder levels with narrower channels than the 2D models.
"""

import torch
import torch.nn as nn


# Required metadata for discovery
ARCHITECTURE_ID = 'unet_3d'
ARCHITECTURE_NAME = 'UNet 3D'
ARCHITECTURE_DESCRIPTION = (
    "True 3D U-Net using volumetric convolutions. Takes 3D patches and "
    "learns spatial features in all three dimensions. Requires 3D training "
    "data (train_images_3d/, train_masks_3d/). No reslicing needed for prediction."
)

PREFERRED_LOSS = 'bce_dice'
TRAINING_V2 = True
IS_3D = True

HIDDEN = True

# 3D patch defaults
PATCH_DEPTH = 32    # Z slices per patch
PATCH_SIZE = 128    # XY size per patch


class DoubleConv3D(nn.Module):
    """Two consecutive 3D convolution blocks with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net with 4 encoder levels.

    Narrower than the 2D model (32->64->128->256) to fit in GPU memory
    with volumetric inputs.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super().__init__()

        # Encoder (4 levels: 32 -> 64 -> 128 -> 256)
        self.inc = DoubleConv3D(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(128, 256))

        # Decoder
        self.up1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv3D(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv3D(128, 64)

        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv3 = DoubleConv3D(64, 32)

        self.outc = nn.Conv3d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 32
        x2 = self.down1(x1)   # 64
        x3 = self.down2(x2)   # 128
        x4 = self.down3(x3)   # 256 (bottleneck)

        # Decoder with skip connections
        x = self.up1(x4)
        x = self.conv1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x1], dim=1))

        return self.outc(x)


# Required: export the model class
MODEL_CLASS = UNet3D
