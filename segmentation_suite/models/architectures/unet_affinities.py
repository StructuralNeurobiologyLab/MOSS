#!/usr/bin/env python3
"""
UNet with affinity output for segmentation strategies.

This architecture outputs 12-channel affinities (3 short-range + 9 long-range)
that are compatible with the LSD and hybrid segmentation strategies.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


ARCHITECTURE_ID = 'unet_affinities'
ARCHITECTURE_NAME = 'UNet Affinities (Segmentation)'
ARCHITECTURE_DESCRIPTION = (
    'UNet that outputs 12-channel affinities (3 short-range + 9 long-range) '
    'for use with LSD/hybrid segmentation strategies.'
)
PREFERRED_LOSS = 'bce'

HIDDEN = True

DEFAULT_NEIGHBORHOOD: List[Tuple[int, int, int]] = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (3, 0, 0), (0, 3, 0), (0, 0, 3),
    (9, 0, 0), (0, 9, 0), (0, 0, 9),
    (27, 0, 0), (0, 27, 0), (0, 0, 27),
]


class DoubleConv(nn.Module):
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


class UNetAffinities(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 12, base_features: int = 32):
        super().__init__()
        bf = base_features
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 4, bf * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 8, bf * 16))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 16, bf * 32))

        self.up1 = nn.ConvTranspose2d(bf * 32, bf * 16, 2, stride=2)
        self.conv1 = DoubleConv(bf * 32, bf * 16)
        self.up2 = nn.ConvTranspose2d(bf * 16, bf * 8, 2, stride=2)
        self.conv2 = DoubleConv(bf * 16, bf * 8)
        self.up3 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.conv3 = DoubleConv(bf * 8, bf * 4)
        self.up4 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.conv4 = DoubleConv(bf * 4, bf * 2)
        self.up5 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.conv5 = DoubleConv(bf * 2, bf)
        self.outc = nn.Conv2d(bf, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

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

        # Raw logits for BCEWithLogitsLoss
        return self.outc(x)


MODEL_CLASS = UNetAffinities
