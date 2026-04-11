#!/usr/bin/env python3
"""
Deep 2.5D UNet with z-coordinate channel — positional context for depth-varying data.

Same as dwarf25d_v2 (11 z-slices, spacing=2) but with one extra input channel
containing the normalized z-position (slice_index / total_slices). This lets the
model learn systematic differences that vary with depth in the stack.

Total input channels: 12 (11 image slices + 1 z-coordinate).
"""

import torch
import torch.nn as nn


# Required metadata for discovery
ARCHITECTURE_ID = 'unet_deep_dice_dwarf25d_zcoord'
ARCHITECTURE_NAME = 'UNet Deep 2.5D v2 (11-slice + Z-coord)'
ARCHITECTURE_DESCRIPTION = (
    "Deep 2.5D model with 11 image channels (z-10 to z+10, spacing 2) plus "
    "a z-coordinate channel encoding the slice's normalized position in the stack. "
    "Helps when appearance varies systematically with depth. "
    "Uses the same dwarf25d training data (train_images_dwarf25d/)."
)

PREFERRED_LOSS = 'bce_dice'
TRAINING_V2 = True
USES_Z_COORD = True

# 12 input channels: 11 image slices + 1 z-coordinate
N_CONTEXT_SLICES = 11  # Image slices (z-coord is added separately)
N_FLANKING_SLICES = 5
SLICE_SPACING = 2


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


class UNetDeepDiceDwarf25DZCoord(nn.Module):
    """
    Deep 2.5D U-Net with 12 input channels: 11 image slices + 1 z-coordinate.
    Same 6-level encoder/decoder as the dwarf model.
    """

    def __init__(self, n_channels: int = 12, n_classes: int = 1):
        super().__init__()

        # Encoder (6 levels: 32 -> 64 -> 128 -> 256 -> 512 -> 1024)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

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
MODEL_CLASS = UNetDeepDiceDwarf25DZCoord
