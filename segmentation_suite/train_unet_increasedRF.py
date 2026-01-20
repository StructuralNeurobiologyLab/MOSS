#!/usr/bin/env python3
"""
Nuclei segmentation U-Net training (safe + diagnostic build)
 - Balanced patch sampling
 - Weighted loss
 - Mixed precision
 - Checkpoint resume
 - Debug logging for missing/misaligned files
"""

import os, random
import numpy as np
from tqdm import tqdm
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. U-Net
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
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
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()

        # -------- Encoder --------
        self.inc   = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))  # bottleneck

        # -------- Decoder --------
        self.up1   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.up5   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5 = DoubleConv(64, 32)

        self.outc  = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # 32
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 128
        x4 = self.down3(x3)  # 256
        x5 = self.down4(x4)  # 512
        x6 = self.down5(x5)  # 1024 (bottleneck)

        # Decoder
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


# ============================================================
# 2. Dataset (balanced, safe, diagnostic)
# ============================================================
class NucleiPatchDataset(Dataset):
    def __init__(self, img_dir, mask_dir, tile=512, fg_ratio=0.5):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.tile = tile
        self.fg_ratio = fg_ratio

        # only keep valid pairs
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
        ])

        self._img_cache, self._mask_cache, self._positive_pixels = {}, {}, {}
        self._load_all()

    def _load_all(self):
        print(f"🧩 Loading {len(self.images)} image–mask pairs from {self.img_dir}")
        for fname in self.images:
            ip = os.path.join(self.img_dir, fname)
            mp = os.path.join(self.mask_dir, fname)
            if not os.path.exists(ip):
                print(f"⚠️ Missing image: {ip}")
                continue
            if not os.path.exists(mp):
                print(f"⚠️ Missing mask:  {mp}")
                continue
            try:
                img = io.imread(ip).astype(np.float32)
                mask = io.imread(mp).astype(np.float32)
            except Exception as e:
                print(f"⚠️ Failed to read {fname}: {e}")
                continue
            if img.ndim == 3:
                img = img.mean(axis=-1)
            mask = (mask > 0.5).astype(np.float32)
            self._img_cache[ip] = img
            self._mask_cache[mp] = mask
            pos = np.argwhere(mask > 0)
            if len(pos) > 0:
                self._positive_pixels[fname] = pos
        print(f"✅ Cached {len(self._img_cache)} pairs successfully.")

    def __len__(self):
        return len(self.images) * 50

    def _random_crop(self, img, mask, y, x):
        t = self.tile
        h, w = img.shape
        y1, x1 = min(y + t, h), min(x + t, w)
        patch_img = img[y:y1, x:x1]
        patch_msk = mask[y:y1, x:x1]
        if patch_img.shape != (t, t):
            pad_y, pad_x = t - patch_img.shape[0], t - patch_img.shape[1]
            patch_img = np.pad(patch_img, ((0, pad_y), (0, pad_x)), mode="reflect")
            patch_msk = np.pad(patch_msk, ((0, pad_y), (0, pad_x)), mode="reflect")
        return patch_img, patch_msk

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
        k = random.randint(0, 4)
        if k:
            img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
        if random.random() < 0.3:
            img = np.clip(img * random.uniform(0.8, 1.2) +
                          random.uniform(-0.1, 0.1), 0, 1).copy()
        return img, mask

    def __getitem__(self, _):
        fname = random.choice(self.images)
        ip = os.path.join(self.img_dir, fname)
        mp = os.path.join(self.mask_dir, fname)

        # diagnostic logging if cache fails
        if ip not in self._img_cache:
            print(f"\n⚠️ Image not in cache: {ip}")
            print("Cached keys:", list(self._img_cache.keys())[:3])
        if mp not in self._mask_cache:
            print(f"⚠️ Mask not in cache: {mp}")
            print("Cached mask keys:", list(self._mask_cache.keys())[:3])

        img = self._img_cache[ip]
        mask = self._mask_cache[mp]

        h, w = img.shape
        t = self.tile
        if fname in self._positive_pixels and random.random() < self.fg_ratio:
            y, x = random.choice(self._positive_pixels[fname])
            y = max(0, min(y - t // 2, h - t))
            x = max(0, min(x - t // 2, w - t))
        else:
            y = random.randint(0, max(0, h - t))
            x = random.randint(0, max(0, w - t))

        patch_img, patch_msk = self._random_crop(img, mask, y, x)
        patch_img = patch_img.astype(np.float32)
        # below we map values to a range closer to 0 - 1, where gradients are more stable and learning is easier. so for the 8bit detector we just divide by 255, for the masks just to be sure we clip between 0 and 1.
        patch_img /= 255
	patch_msk = np.clip(patch_msk, 0.0, 1.0).astype(np.float32)
        
        patch_img, patch_msk = self._augment(patch_img, patch_msk)

        patch_img = torch.tensor(patch_img.copy()[None, ...], dtype=torch.float32)
        patch_msk = torch.tensor(patch_msk.copy()[None, ...], dtype=torch.float32)
        return patch_img, patch_msk


# ============================================================
# 3. Training loop with checkpoint resume
# ============================================================
def train():
    train_images = "train_images"
    train_masks = "train_masks"
    val_images = "val_images"
    val_masks = "val_masks"
    tile = 512
    batch_size = 2
    num_epochs = 4000
    lr = 1e-4
    checkpoint_path = "checkpoint_last.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_ds = NucleiPatchDataset(train_images, train_masks, tile=tile, fg_ratio=0.5)
    val_ds = NucleiPatchDataset(val_images, val_masks, tile=tile, fg_ratio=0.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    # pos_weight estimate
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
    else:
        print("No checkpoint found — starting fresh.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f"Validation Loss: {val_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

    torch.save(model.state_dict(), "unet_nuclei_balanced.pth")
    print("✅ Training complete — model saved.")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
    train()

