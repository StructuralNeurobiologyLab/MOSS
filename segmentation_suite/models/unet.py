#!/usr/bin/env python3
"""
UNet Model Definition for 2D Segmentation

Adapted from train_unet.py in cuticle_images_training
"""

import torch
import torch.nn as nn


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


class UNet(nn.Module):
    """
    U-Net architecture for 2D image segmentation (4 levels, smaller receptive field).

    Args:
        n_channels: Number of input channels (1 for grayscale, 2 for image+mask)
        n_classes: Number of output classes (1 for binary segmentation)
        base_features: Number of features in the first layer (default 32)
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1, base_features: int = 32):
        super().__init__()
        bf = base_features

        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 4, bf * 8))

        # Decoder (upsampling path)
        self.up1 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.conv1 = DoubleConv(bf * 8, bf * 4)
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.conv2 = DoubleConv(bf * 4, bf * 2)
        self.up3 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.conv3 = DoubleConv(bf * 2, bf)

        # Output
        self.outc = nn.Conv2d(bf, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with skip connections
        x = self.up1(x4)
        x = self.conv1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x1], dim=1))

        return self.outc(x)

    @classmethod
    def for_scratch_training(cls) -> 'UNet':
        """Factory for training from scratch (1-channel: raw image only)."""
        return cls(n_channels=1, n_classes=1)

    @classmethod
    def for_refinement_training(cls) -> 'UNet':
        """Factory for refinement training (2-channel: raw image + mask)."""
        return cls(n_channels=2, n_classes=1)


# Built-in architecture registry (standard UNet only)
# Additional architectures are loaded from the architectures/ folder
ARCHITECTURES = {
    'unet': UNet,
}

ARCHITECTURE_NAMES = {
    'unet': 'UNet (Standard)',
}


def get_model_class(architecture: str):
    """
    Get the model class for the given architecture name.

    First checks built-in architectures, then looks in architectures/ folder.
    """
    if architecture in ARCHITECTURES:
        return ARCHITECTURES[architecture]

    # Try loading from architectures folder
    from .architectures import get_model_class as arch_get_model_class
    return arch_get_model_class(architecture)


def get_checkpoint_filename(architecture: str) -> str:
    """Get the checkpoint filename for the given architecture."""
    if architecture == 'unet':
        return 'checkpoint.pth'
    else:
        return f'checkpoint_{architecture}.pth'


def get_available_architectures():
    """
    Get dict of all available architectures: {architecture_id: display_name}

    Includes built-in architectures plus any from the architectures/ folder.
    """
    from .architectures import get_available_architectures as arch_get_available
    return arch_get_available()


def get_device() -> torch.device:
    """Get the best available device for training."""
    import os
    # Allow forcing CPU via environment variable for debugging
    if os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
        print("[Device] Forcing CPU (FORCE_CPU env var set)")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path: str, n_channels: int = 1, device: torch.device = None,
                architecture: str = 'unet') -> nn.Module:
    """
    Load a UNet model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        n_channels: Number of input channels for the model
        device: Device to load the model onto
        architecture: Model architecture ('unet' or 'unet_deep')

    Returns:
        Loaded model
    """
    if device is None:
        device = get_device()

    ModelClass = get_model_class(architecture)
    model = ModelClass(n_channels=n_channels, n_classes=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model
