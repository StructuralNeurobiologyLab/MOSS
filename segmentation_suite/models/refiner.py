#!/usr/bin/env python3
"""
Refiner Model - learns from user edits to predict corrections.

Takes 2-channel input: (raw_image, mask_before) and predicts mask_after.
Based on UNetDeep architecture for large receptive field.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
from PIL import Image


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


class RefinerUNet(nn.Module):
    """
    Refiner U-Net with 2-channel input (image + mask_before).

    Outputs 2 channels:
    - Channel 0: pixels to ADD (should be 1 but currently 0)
    - Channel 1: pixels to REMOVE (should be 0 but currently 1)

    This forces the model to learn actual corrections, not just copy the input.
    """

    def __init__(self):
        super().__init__()
        # 2-channel input: raw_image (normalized) + mask_before (binary)
        n_channels = 2
        n_classes = 2  # 2 outputs: add_mask, remove_mask

        # Encoder (4 levels: 32 -> 64 -> 128 -> 256)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, n_classes, 1)  # 2 output channels

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 32
        x2 = self.down1(x1)   # 64
        x3 = self.down2(x2)   # 128
        x4 = self.down3(x3)   # 256

        # Decoder with skip connections
        x = self.up1(x4)
        x = self.conv1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x1], dim=1))

        return self.outc(x)


class EditPairDataset(torch.utils.data.Dataset):
    """Dataset of edit pairs for training the refiner.

    Each sample contains:
    - raw_image: The grayscale image
    - mask_before: Mask state before user edit
    - mask_after: Mask state after user edit (ground truth)
    """

    def __init__(self, edit_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 patch_size: int = 256):
        """
        Args:
            edit_pairs: List of (raw_image, mask_before, mask_after) tuples
            patch_size: Size of training patches to extract
        """
        self.edit_pairs = edit_pairs
        self.patch_size = patch_size

    def __len__(self):
        return len(self.edit_pairs) * 5  # Generate 5 patches per edit (matches annotation tool)

    def __getitem__(self, idx):
        # Select a random edit triplet (matches annotation tool)
        pair_idx = idx % len(self.edit_pairs)
        raw_image, mask_before, mask_after = self.edit_pairs[pair_idx]

        # Ensure we have 2D arrays
        if raw_image.ndim > 2:
            raw_image = raw_image[:, :, 0] if raw_image.shape[2] <= 4 else raw_image[0]
        if mask_before.ndim > 2:
            mask_before = mask_before[:, :, 0] if mask_before.shape[2] <= 4 else mask_before[0]
        if mask_after.ndim > 2:
            mask_after = mask_after[:, :, 0] if mask_after.shape[2] <= 4 else mask_after[0]

        raw_image = raw_image.astype(np.float32)
        mask_before = mask_before.astype(np.float32)
        mask_after = mask_after.astype(np.float32)

        h, w = raw_image.shape
        t = self.patch_size

        # Find where changes occurred
        diff_mask = (mask_before > 127) != (mask_after > 127)
        change_coords = np.argwhere(diff_mask)

        # Sample patch centered on a changed region (80% of time) or random (20%)
        if len(change_coords) > 0 and np.random.random() < 0.8:
            # Pick a random changed pixel and center patch there
            idx = np.random.randint(len(change_coords))
            cy, cx = change_coords[idx]
            y = max(0, min(cy - t // 2, h - t))
            x = max(0, min(cx - t // 2, w - t))
        else:
            # Random crop
            if h > t and w > t:
                y = np.random.randint(0, h - t)
                x = np.random.randint(0, w - t)
            else:
                y, x = 0, 0

        patch_raw = raw_image[y:y+t, x:x+t]
        patch_before = mask_before[y:y+t, x:x+t]
        patch_after = mask_after[y:y+t, x:x+t]

        # Pad if needed
        if patch_raw.shape[0] < t or patch_raw.shape[1] < t:
            pad_h = max(0, t - patch_raw.shape[0])
            pad_w = max(0, t - patch_raw.shape[1])
            patch_raw = np.pad(patch_raw, ((0, pad_h), (0, pad_w)), mode='reflect')
            patch_before = np.pad(patch_before, ((0, pad_h), (0, pad_w)), mode='reflect')
            patch_after = np.pad(patch_after, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Normalize raw image
        img_min, img_max = patch_raw.min(), patch_raw.max()
        if img_max > img_min:
            patch_raw = (patch_raw - img_min) / (img_max - img_min)

        # Normalize masks to binary [0, 1]
        patch_before = (patch_before > 127).astype(np.float32)
        patch_after = (patch_after > 127).astype(np.float32)

        # Compute ADD and REMOVE targets
        # ADD: pixels that are 0 in before but 1 in after (need to add these)
        add_target = ((patch_before == 0) & (patch_after == 1)).astype(np.float32)
        # REMOVE: pixels that are 1 in before but 0 in after (need to remove these)
        remove_target = ((patch_before == 1) & (patch_after == 0)).astype(np.float32)

        # Create a mask for pixels that changed (used for masked loss)
        change_mask = (patch_before != patch_after).astype(np.float32)

        # Stack input: [raw_image, mask_before] -> [2, H, W]
        input_tensor = np.stack([patch_raw.astype(np.float32), patch_before], axis=0)

        # Stack targets: [add, remove] -> [2, H, W]
        target_tensor = np.stack([add_target, remove_target], axis=0)

        return (
            torch.tensor(input_tensor, dtype=torch.float32),
            torch.tensor(target_tensor, dtype=torch.float32),  # [2, H, W]
            torch.tensor(change_mask[None, ...], dtype=torch.float32)  # [1, H, W] mask for loss
        )


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def predict_refinement(model: RefinerUNet, raw_image: np.ndarray, mask: np.ndarray,
                       device: torch.device, patch_size: int = 256,
                       overlap: int = 32) -> np.ndarray:
    """
    Generate a refined mask prediction by predicting ADD/REMOVE deltas.

    Args:
        model: The refiner model (outputs 2 channels: add, remove)
        raw_image: Grayscale image [H, W]
        mask: Current mask [H, W]
        device: Torch device
        patch_size: Size of patches to process
        overlap: Overlap between patches for smooth blending

    Returns:
        Refined mask prediction [H, W] as uint8 (0 or 255)
    """
    model.eval()
    h, w = raw_image.shape

    # Normalize inputs
    if raw_image.max() > 1:
        raw_norm = raw_image / 255.0 if raw_image.max() <= 255 else raw_image / raw_image.max()
    else:
        raw_norm = raw_image.astype(np.float32)

    mask_norm = (mask > 127).astype(np.float32)

    # Output arrays for ADD and REMOVE channels
    add_sum = np.zeros((h, w), dtype=np.float32)
    remove_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    stride = patch_size - overlap

    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Get patch bounds
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                y_start = max(0, y_end - patch_size)
                x_start = max(0, x_end - patch_size)

                # Extract patches
                patch_raw = raw_norm[y_start:y_end, x_start:x_end]
                patch_mask = mask_norm[y_start:y_end, x_start:x_end]

                # Pad if needed
                ph, pw = patch_raw.shape
                if ph < patch_size or pw < patch_size:
                    pad_raw = np.zeros((patch_size, patch_size), dtype=np.float32)
                    pad_mask = np.zeros((patch_size, patch_size), dtype=np.float32)
                    pad_raw[:ph, :pw] = patch_raw
                    pad_mask[:ph, :pw] = patch_mask
                    patch_raw = pad_raw
                    patch_mask = pad_mask

                # Stack and predict
                input_tensor = np.stack([patch_raw, patch_mask], axis=0)
                input_batch = torch.tensor(input_tensor[None, ...], dtype=torch.float32).to(device)

                # Model outputs 2 channels: [add, remove]
                pred = torch.sigmoid(model(input_batch))[0].cpu().numpy()
                pred_add = pred[0, :ph, :pw]
                pred_remove = pred[1, :ph, :pw]

                # Accumulate
                add_sum[y_start:y_end, x_start:x_end] += pred_add
                remove_sum[y_start:y_end, x_start:x_end] += pred_remove
                count[y_start:y_end, x_start:x_end] += 1

    # Average overlapping regions
    add_avg = add_sum / np.maximum(count, 1e-8)
    remove_avg = remove_sum / np.maximum(count, 1e-8)

    # Debug: print what the model is predicting
    add_pixels = np.sum(add_avg > 0.5)
    remove_pixels = np.sum(remove_avg > 0.5)
    current_pixels = np.sum(mask_norm > 0.5)
    print(f"[Refiner] Prediction channels: ADD={add_pixels}px, REMOVE={remove_pixels}px, current_mask={current_pixels}px")

    # Apply delta to current mask:
    # result = current_mask + add - remove
    # Threshold add/remove at 0.5
    add_binary = (add_avg > 0.5).astype(np.float32)
    remove_binary = (remove_avg > 0.5).astype(np.float32)

    result = mask_norm + add_binary - remove_binary
    result = np.clip(result, 0, 1)

    final_pixels = np.sum(result > 0.5)
    print(f"[Refiner] Result: {final_pixels}px (change: {final_pixels - current_pixels:+d}px)")

    return ((result > 0.5) * 255).astype(np.uint8)
