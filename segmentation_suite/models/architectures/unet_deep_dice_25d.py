#!/usr/bin/env python3
"""
2.5D U-Net Deep with Dice loss.

Same architecture as unet_deep_dice but takes 3 slices as input channels
(z-3, z, z+3) to provide z-context while still using 2D convolutions.
Uses a spacing of 3 slices to capture broader z-context.
"""

import torch
import torch.nn as nn


# Required metadata for discovery
ARCHITECTURE_ID = 'unet_deep_dice_25d'
ARCHITECTURE_NAME = 'UNet Deep Dice 2.5D'
ARCHITECTURE_DESCRIPTION = (
    "2.5D variant that takes 3 z-slices (z-3, z, z+3) as input channels. "
    "Provides z-context with spacing of 3 slices using efficient 2D convolutions. "
    "Requires 2.5D training data (train_images_25d/, train_masks_25d/)."
)

# Use Dice loss like the parent architecture
PREFERRED_LOSS = 'dice'

# Number of input slices for 2.5D context (1 flanking on each side + center = 3)
N_CONTEXT_SLICES = 3
N_FLANKING_SLICES = 1  # Slices on each side of center
SLICE_SPACING = 3  # Distance between slices (z-3, z, z+3)


class DoubleConv(nn.Module):
    """Two consecutive convolution blocks with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetDeepDice25D(nn.Module):
    """
    2.5D U-Net with 3 input channels for z-context.

    Takes 3 slices (z-3, z, z+3) as input and predicts
    the segmentation for the center slice.
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super().__init__()

        # Note: n_channels=3 for 2.5D (z-3, z, z+3)
        # Encoder (6 levels: 32 -> 64 -> 128 -> 256 -> 512 -> 1024)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))  # bottleneck

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 32
        x2 = self.down1(x1)   # 64
        x3 = self.down2(x2)   # 128
        x4 = self.down3(x3)   # 256
        x5 = self.down4(x4)   # 512
        x6 = self.down5(x5)   # 1024 (bottleneck)

        # Decoder with skip connections
        x = self.up1(x6)
        x = self.conv1(torch.cat([x, x5], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x3], dim=1))

        x = self.up4(x)
        x = self.conv4(torch.cat([x, x2], dim=1))

        x = self.up5(x)
        x = self.conv5(torch.cat([x, x1], dim=1))

        return self.outc(x)


# Required: export the model class
MODEL_CLASS = UNetDeepDice25D
