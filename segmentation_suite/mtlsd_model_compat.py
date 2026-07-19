"""
Compatible model class for the pretrained mtlsd checkpoint.

The checkpoint uses a 2D UNet architecture from the original lsd_pytorch codebase:
- 2D convolutions (3x3)
- l_conv: left/encoder path
- r_conv: right/decoder path
- lsd_head: 6-channel LSD output
- aff_head: 2-channel affinity output (y, x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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

    Checkpoint structure:
    - l_conv[0]: 1 -> 12
    - l_conv[1]: 12 -> 60
    - l_conv[2]: 60 -> 300 (bottleneck)
    - r_conv[0][1]: 360 (300+60) -> 60  (first decoder, deep)
    - r_conv[0][0]: 72 (60+12) -> 12    (second decoder, shallow)
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
        # Level 0: 1 -> 12
        # Level 1: 12 -> 60
        # Level 2: 60 -> 300
        self.l_conv = nn.ModuleList()
        current_channels = in_channels
        for level in range(len(downsample_factors) + 1):
            out_channels = num_fmaps * (fmap_inc_factor ** level)
            self.l_conv.append(ConvPass(current_channels, out_channels))
            current_channels = out_channels

        # Decoder (right) path - matching checkpoint structure r_conv[0][level]
        # r_conv[0][1]: 360 (300+60) -> 60  (deep decoder)
        # r_conv[0][0]: 72 (60+12) -> 12    (shallow decoder)
        # Stored as r_conv = [[shallow_decoder, deep_decoder]]
        self.r_conv = nn.ModuleList([
            nn.ModuleList([
                ConvPass(72, 12),   # r_conv[0][0]: 60+12 -> 12 (shallow)
                ConvPass(360, 60),  # r_conv[0][1]: 300+60 -> 60 (deep)
            ])
        ])

    def forward(self, x):
        # Encoder path with skip connections
        # skips[0] = output of l_conv[0] = 12 channels
        # skips[1] = output of l_conv[1] = 60 channels
        skips = []
        for level, conv in enumerate(self.l_conv):
            x = conv(x)
            if level < len(self.downsample_factors):
                skips.append(x)
                x = F.max_pool2d(x, 2)

        # After encoder: x has 300 channels (from l_conv[2])

        # Decoder path
        # First decoder: r_conv[0][1] - upsample 300, concat with skip[1] (60), get 360 -> 60
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([skips[1], x], dim=1)  # 60 + 300 = 360 (skip first!)
        x = self.r_conv[0][1](x)  # 360 -> 60

        # Second decoder: r_conv[0][0] - upsample 60, concat with skip[0] (12), get 72 -> 12
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([skips[0], x], dim=1)  # 12 + 60 = 72 (skip first!)
        x = self.r_conv[0][0](x)  # 72 -> 12

        return x


class MtLsdModel(nn.Module):
    """
    Complete mtlsd model matching the pretrained checkpoint.

    Outputs:
    - affinities: 2 channels (y, x neighbors)
    - lsds: 6 channels (local shape descriptors)
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

        # Output heads - matching checkpoint structure with conv_pass
        self.aff_head = OutputHead(num_fmaps, num_affinities)
        self.lsd_head = OutputHead(num_fmaps, num_lsds)

        # Store config
        self.num_affinities = num_affinities
        self.num_lsds = num_lsds

    def forward(self, x):
        features = self.unet(x)

        affinities = torch.sigmoid(self.aff_head(features))
        lsds = torch.tanh(self.lsd_head(features))

        return affinities, lsds

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'MtLsdModel':
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Create model
        model = cls()

        # Load weights
        model.load_state_dict(state_dict)

        return model.to(device).eval()


def predict_slice(
    model: MtLsdModel,
    slice_2d: torch.Tensor,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run prediction on a single 2D slice.

    Parameters
    ----------
    model : MtLsdModel
        The loaded model
    slice_2d : Tensor
        Input slice (H, W) or (1, H, W) or (1, 1, H, W)
    device : str
        Device to run on

    Returns
    -------
    affinities : Tensor (2, H, W)
    lsds : Tensor (6, H, W)
    """
    model.eval()

    # Ensure correct shape (N, C, H, W)
    if slice_2d.ndim == 2:
        slice_2d = slice_2d.unsqueeze(0).unsqueeze(0)
    elif slice_2d.ndim == 3:
        slice_2d = slice_2d.unsqueeze(0)

    # Normalize to [0, 1]
    if slice_2d.max() > 1.0:
        slice_2d = slice_2d / 255.0

    slice_2d = slice_2d.to(device)

    with torch.no_grad():
        affinities, lsds = model(slice_2d)

    return affinities[0], lsds[0]


def predict_volume_2d(
    model: MtLsdModel,
    volume: torch.Tensor,
    device: str = 'cuda',
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run 2D prediction slice-by-slice on a 3D volume.

    Parameters
    ----------
    model : MtLsdModel
        The loaded model
    volume : Tensor
        Input volume (D, H, W)
    device : str
        Device to run on
    show_progress : bool
        Show progress

    Returns
    -------
    affinities : Tensor (2, D, H, W) - y and x affinities
    lsds : Tensor (6, D, H, W)
    """
    model = model.to(device).eval()

    D, H, W = volume.shape
    affinities = torch.zeros(2, D, H, W)
    lsds = torch.zeros(6, D, H, W)

    # Normalize volume
    if volume.max() > 1.0:
        volume = volume.float() / 255.0

    with torch.no_grad():
        for z in range(D):
            slice_2d = volume[z:z+1].unsqueeze(0).to(device)  # (1, 1, H, W)
            aff, lsd = model(slice_2d)
            affinities[:, z] = aff[0].cpu()
            lsds[:, z] = lsd[0].cpu()

            if show_progress and (z + 1) % 10 == 0:
                print(f"  Predicted {z + 1}/{D} slices")

    return affinities, lsds


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import numpy as np

    # Default to the checkpoint bundled with the repo; no hardcoded personal paths.
    default_ckpt = Path(__file__).resolve().parent.parent / "pretrained_models" / "lsd_mtlsd_checkpoint.pth"
    ap = argparse.ArgumentParser(description="Smoke-test the MtLSD compatibility loader.")
    ap.add_argument("--model", default=str(default_ckpt),
                    help="MtLSD checkpoint path (default: the bundled pretrained model).")
    ap.add_argument("--volume", default=None,
                    help="Optional .npy EM volume to run a real prediction on.")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {args.model}")
    model = MtLsdModel.from_pretrained(args.model, device)
    print(f"Model loaded on {device}")

    # Test on dummy data
    dummy = torch.randn(1, 1, 256, 256).to(device)
    with torch.no_grad():
        aff, lsd = model(dummy)

    print(f"Affinity shape: {aff.shape}")  # Should be (1, 2, 256, 256)
    print(f"LSD shape: {lsd.shape}")        # Should be (1, 6, 256, 256)
    print(f"Affinity range: [{aff.min():.3f}, {aff.max():.3f}]")
    print(f"LSD range: [{lsd.min():.3f}, {lsd.max():.3f}]")

    # Optionally test on a real volume, if one was provided on the command line.
    if args.volume:
        volume = np.load(args.volume)
        print(f"\nTest volume shape: {volume.shape}")
        test_volume = torch.from_numpy(volume[:5].astype(np.float32))
        affinities, lsds = predict_volume_2d(model, test_volume, device, show_progress=True)
        print(f"\nOutput affinities: {affinities.shape}")
        print(f"Output LSDs: {lsds.shape}")

    print("Model loading and prediction successful!")
