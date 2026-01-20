#!/usr/bin/env python3
"""
UNet Deep with SAM2 feature integration.

This architecture fuses pre-computed SAM2 features with the UNet encoder
at the down4 level (16x downsampled from input).

Input:
    - image: (N, 1, H, W) - grayscale image
    - sam2_features: (N, 256, H/16, W/16) - pre-computed SAM2 features

SAM2 features are concatenated with down4 output before passing to down5.
For 256x256 input: down4 produces (512, 16, 16), SAM2 provides (256, 16, 16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Required metadata for discovery
ARCHITECTURE_ID = 'unet_deep_dice_sam2'
ARCHITECTURE_NAME = 'UNet Deep Dice + SAM2'
ARCHITECTURE_DESCRIPTION = (
    "UNet Deep with pre-computed SAM2 feature integration. "
    "SAM2 features (256ch) are concatenated at the down4 level (512ch) "
    "for enhanced semantic understanding. Requires sam2_features/ folder."
)

# Use Dice loss like the parent architecture
PREFERRED_LOSS = 'dice'

# Flag to indicate this model requires SAM2 features
REQUIRES_SAM2_FEATURES = True


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


class SAM2FusionBlock(nn.Module):
    """
    Fusion block that combines UNet features with SAM2 features.

    Concatenates UNet features (512ch) with SAM2 features (256ch) = 768ch,
    then projects back to the expected channel count for downstream layers.
    """

    def __init__(self, unet_channels: int = 512, sam2_channels: int = 256, output_channels: int = 512):
        super().__init__()
        combined = unet_channels + sam2_channels  # 768

        # Project fused features back to expected dimensions
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(combined, output_channels, 1),  # 1x1 conv to reduce channels
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, unet_features: torch.Tensor, sam2_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            unet_features: (N, 512, H, W) from down4
            sam2_features: (N, 256, H, W) pre-computed SAM2 features

        Returns:
            Fused features: (N, 512, H, W)
        """
        # Ensure spatial dimensions match (interpolate SAM2 if needed)
        if unet_features.shape[2:] != sam2_features.shape[2:]:
            sam2_features = F.interpolate(
                sam2_features,
                size=unet_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Concatenate along channel dimension
        fused = torch.cat([unet_features, sam2_features], dim=1)

        # Project to output channels
        return self.fusion_conv(fused)


class UNetDeepDiceSAM2(nn.Module):
    """
    Deeper U-Net with SAM2 feature integration.

    SAM2 features are fused at the down4 level before the bottleneck.
    This allows the model to leverage SAM2's semantic understanding
    while maintaining the UNet's localization precision.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1, sam2_channels: int = 256):
        super().__init__()
        self.sam2_channels = sam2_channels

        # Encoder (same as unet_deep_dice)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # SAM2 fusion at down4 level
        self.sam2_fusion = SAM2FusionBlock(
            unet_channels=512,
            sam2_channels=sam2_channels,
            output_channels=512  # Keep same for compatibility with down5
        )

        # Bottleneck (receives fused features)
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder (same as unet_deep_dice)
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

    def forward(self, x: torch.Tensor, sam2_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional SAM2 features.

        Args:
            x: Input image (N, C, H, W) where C is n_channels
            sam2_features: Pre-computed SAM2 features (N, 256, H/16, W/16)
                          If None, model works as standard UNet (fallback mode)

        Returns:
            Segmentation logits (N, 1, H, W)
        """
        # Encoder
        x1 = self.inc(x)      # 32, H
        x2 = self.down1(x1)   # 64, H/2
        x3 = self.down2(x2)   # 128, H/4
        x4 = self.down3(x3)   # 256, H/8
        x5 = self.down4(x4)   # 512, H/16 <- SAM2 fusion point

        # Fuse with SAM2 features if provided
        if sam2_features is not None:
            x5 = self.sam2_fusion(x5, sam2_features)

        x6 = self.down5(x5)   # 1024, H/32 (bottleneck)

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
MODEL_CLASS = UNetDeepDiceSAM2
