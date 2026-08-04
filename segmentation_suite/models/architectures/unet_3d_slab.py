#!/usr/bin/env python3
"""
3D slab U-Net trained from sparse 2D annotations.

Slab in, slab out. Unlike the 2.5D models (which take a stack of context slices
and emit a single centre slice), this one emits a prediction for every Z plane of
its input slab.

The point is that it can be trained from ordinary 2D annotations. Labels are not
complete in 3D -- the annotator painted one slice, and the slices above and below
are genuinely unknown, not background. So training stores a slab of depth
PATCH_DEPTH + Z_JITTER with the annotated plane at the stored centre, random-crops
a PATCH_DEPTH window out of it, and builds a label slab that is IGNORE_LABEL
everywhere except the one plane carrying the real annotation. The masked loss
supervises that plane only; the Z-jitter is what eventually teaches every output
plane, because across samples the annotated plane lands at every position in
[PATCH_DEPTH//2 - Z_JITTER//2, PATCH_DEPTH//2 + Z_JITTER//2].

Consequences worth knowing:
  - Only Z_JITTER + 1 output planes are ever supervised, so inference must weight
    the untrained margins zero and step in Z by at most Z_JITTER + 1.
  - Z is pooled at the first two levels only (XY at all four), so the depth axis
    shrinks 4x and PATCH_DEPTH must be a multiple of 4; XY must be a multiple of 16.
  - Z_JITTER must be < PATCH_DEPTH - 1, or the label index escapes the slab.

Ported from the Ais ezm-3d model (github.com/mgflast/Ais, core/se_model.py,
models/easymode_3d.py, models/losses.py).
"""

import torch
import torch.nn as nn


# Required metadata for discovery
ARCHITECTURE_ID = 'unet_3d_slab'
ARCHITECTURE_NAME = 'UNet 3D Slab (sparse 2D labels)'
ARCHITECTURE_DESCRIPTION = (
    "True 3D U-Net that predicts a whole Z-slab, trained from ordinary 2D "
    "annotations. Keep painting one slice at a time: select this architecture and "
    "each normal crop capture also writes a Z-slab to train_images_slab/, with your "
    "2D mask beside it. Training jitters the annotated plane through the slab and "
    "masks the loss on every unlabelled plane, so incomplete 3D labels are never "
    "mistaken for background. Do NOT use 3D GT mode: its dense mask volumes assert "
    "background on unpainted planes, which is what this model exists to avoid."
)

PREFERRED_LOSS = 'masked_bce_dice'
TRAINING_V2 = True
IS_3D = True

# Slab-specific contract, read by the trainer, the extractor and the predictors.
IS_SLAB = True
PATCH_DEPTH = 16    # D: model input/output depth. Multiple of 4, and > Z_JITTER + 1.
PATCH_SIZE = 128    # N: XY size per patch. Multiple of 16.
Z_JITTER = 8        # M: extra stored depth; supervises Z_JITTER + 1 output planes.

# Visible in the UI: a hidden architecture nobody can select is useless.
HIDDEN = False

# Per-level (D, H, W) pooling strides. Z is pooled at levels 0-1 only, XY at all
# four -- "a limited number of pooling layers in Z". Mirrors Ais easymode_3d,
# whose (2,2,1) Keras pools are XY-only in its (Y, X, Z, C) layout and therefore
# become (1, 2, 2) here in PyTorch's (N, C, D, H, W).
POOLS = ((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2))

Z_DOWNSAMPLE = 4    # product of the D strides above
XY_DOWNSAMPLE = 16  # product of the H/W strides above


class DoubleConv3D(nn.Module):
    """Two 3x3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3DSlab(nn.Module):
    """4-level 3D U-Net emitting logits for the full input slab.

    Input  (N, n_channels, D, H, W)
    Output (N, n_classes,  D, H, W)   -- logits, no sigmoid (MOSS convention)
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1,
                 base: int = 16, dropout: float = 0.25):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]   # 16 32 64 128 256

        self.enc0 = DoubleConv3D(n_channels, c[0])
        self.pool0 = nn.MaxPool3d(POOLS[0])
        self.enc1 = DoubleConv3D(c[0], c[1])
        self.pool1 = nn.MaxPool3d(POOLS[1])
        self.enc2 = DoubleConv3D(c[1], c[2])
        self.pool2 = nn.MaxPool3d(POOLS[2])
        self.enc3 = DoubleConv3D(c[2], c[3])
        self.pool3 = nn.MaxPool3d(POOLS[3])

        self.bottleneck = DoubleConv3D(c[3], c[4])
        self.dropout = nn.Dropout3d(dropout)

        self.up3 = nn.ConvTranspose3d(c[4], c[3], POOLS[3], stride=POOLS[3])
        self.dec3 = DoubleConv3D(c[3] * 2, c[3])
        self.up2 = nn.ConvTranspose3d(c[3], c[2], POOLS[2], stride=POOLS[2])
        self.dec2 = DoubleConv3D(c[2] * 2, c[2])
        self.up1 = nn.ConvTranspose3d(c[2], c[1], POOLS[1], stride=POOLS[1])
        self.dec1 = DoubleConv3D(c[1] * 2, c[1])
        self.up0 = nn.ConvTranspose3d(c[1], c[0], POOLS[0], stride=POOLS[0])
        self.dec0 = DoubleConv3D(c[0] * 2, c[0])

        self.outc = nn.Conv3d(c[0], n_classes, 1)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                f"{type(self).__name__} expects (N, C, D, H, W); got {tuple(x.shape)}")
        d, h, w = x.shape[-3:]
        if d % Z_DOWNSAMPLE or h % XY_DOWNSAMPLE or w % XY_DOWNSAMPLE:
            raise ValueError(
                f"slab shape D={d}, H={h}, W={w} is not compatible: D must be a "
                f"multiple of {Z_DOWNSAMPLE} and H/W a multiple of {XY_DOWNSAMPLE} "
                f"(skip connections would not align)")

        s0 = self.enc0(x)
        s1 = self.enc1(self.pool0(s0))
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        b = self.dropout(self.bottleneck(self.pool3(s3)))

        y = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        y = self.dec2(torch.cat([self.up2(y), s2], dim=1))
        y = self.dec1(torch.cat([self.up1(y), s1], dim=1))
        y = self.dec0(torch.cat([self.up0(y), s0], dim=1))
        return self.outc(y)


# Required: export the model class
MODEL_CLASS = UNet3DSlab
