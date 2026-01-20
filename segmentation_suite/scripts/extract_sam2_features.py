#!/usr/bin/env python3
"""
Extract SAM2 features for all training tiles in a project.

Usage:
    python extract_sam2_features.py /path/to/project [--overwrite]

This will:
1. Read all images from train_images/
2. Extract SAM2 features for each (256x256 -> 16x16 features with 256 channels)
3. Save to sam2_features/ folder as .npy files (float16)

Requirements:
    - SAM2 installed: pip install git+https://github.com/facebookresearch/sam2.git
    - huggingface_hub: pip install huggingface_hub
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# Progress bar (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def setup_sam2_predictor(device: str = None):
    """Initialize SAM2 predictor with MedSAM2 weights.

    Args:
        device: 'cuda' or 'cpu'. Auto-detects if None.

    Returns:
        (predictor, device) tuple
    """
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Setting up SAM2 on {device}...")

    # Use MOSS directory for SAM2 model cache (shared across all projects)
    from pathlib import Path
    cache_dir = str(Path(__file__).parent.parent.parent / "sam2_models")
    os.makedirs(cache_dir, exist_ok=True)

    # Check if model already cached
    cached_model = os.path.join(cache_dir, "MedSAM2_latest.pt")
    if os.path.exists(cached_model):
        print(f"Using cached model: {cached_model}")
        ckpt_path = cached_model
    else:
        print("Downloading MedSAM2 model (first time only, ~300MB)...")
        ckpt_path = hf_hub_download(
            repo_id="wanglab/MedSAM2",
            filename="MedSAM2_latest.pt",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Model cached at: {ckpt_path}")

    # Build model with MedSAM2 config
    model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
    model = build_sam2(model_cfg, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(model)

    return predictor, device


def extract_features_for_image(predictor, image_path: Path, device: str) -> np.ndarray:
    """
    Extract SAM2 features for a single image.

    Args:
        predictor: SAM2ImagePredictor instance
        image_path: Path to grayscale image
        device: 'cuda' or 'cpu'

    Returns:
        Feature array of shape (256, H/16, W/16) as float16
        For 256x256 input: (256, 16, 16)
    """
    # Load grayscale image
    img = Image.open(image_path)
    gray = np.array(img).astype(np.uint8)

    # Handle multi-channel images (take first channel or mean)
    if gray.ndim == 3:
        gray = gray.mean(axis=-1).astype(np.uint8)

    # SAM expects RGB input - convert grayscale to RGB
    rgb = np.repeat(gray[..., None], 3, axis=-1)

    # Extract features
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            predictor.set_image(rgb)
            embedding = predictor.get_image_embedding()  # (1, 256, Hf, Wf)

    # Convert to numpy float16 for storage efficiency
    features = embedding.squeeze(0).to(torch.float16).cpu().numpy()

    return features


def extract_all_features(project_dir: str, overwrite: bool = False, device: str = None):
    """
    Extract SAM2 features for all training tiles in a project.

    Args:
        project_dir: Path to project folder containing train_images/
        overwrite: Whether to overwrite existing features
        device: 'cuda' or 'cpu'. Auto-detects if None.
    """
    project_path = Path(project_dir)
    train_images_dir = project_path / "train_images"
    features_dir = project_path / "sam2_features"

    if not train_images_dir.exists():
        raise ValueError(f"train_images/ not found in {project_dir}")

    # Create output directory
    features_dir.mkdir(exist_ok=True)

    # Find all training images
    image_files = sorted([
        f for f in train_images_dir.iterdir()
        if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')
    ])

    print(f"Found {len(image_files)} training images")

    if len(image_files) == 0:
        print("No images to process.")
        return

    # Check existing features
    existing = [f for f in image_files if (features_dir / f"{f.stem}.npy").exists()]
    if existing and not overwrite:
        print(f"  {len(existing)} already have features (use --overwrite to regenerate)")
        image_files = [f for f in image_files if f not in existing]
        if not image_files:
            print("All features already extracted!")
            return
        print(f"  {len(image_files)} remaining to process")

    # Setup SAM2
    print("\nLoading SAM2 model...")
    predictor, device = setup_sam2_predictor(device)
    print(f"Using device: {device}")

    # Process each image
    iterator = tqdm(image_files, desc="Extracting features") if HAS_TQDM else image_files
    errors = []

    for i, image_path in enumerate(iterator):
        if not HAS_TQDM and i % 50 == 0:
            print(f"  Processing {i+1}/{len(image_files)}...")

        output_path = features_dir / f"{image_path.stem}.npy"

        try:
            features = extract_features_for_image(predictor, image_path, device)
            np.save(output_path, features)
        except Exception as e:
            errors.append((image_path.name, str(e)))
            if len(errors) <= 3:
                print(f"Error processing {image_path.name}: {e}")

    # Summary
    print(f"\nFeatures saved to {features_dir}")
    print(f"  Total: {len(image_files) - len(errors)} successful")
    if errors:
        print(f"  Errors: {len(errors)}")
        if len(errors) > 3:
            print(f"  (showing first 3 errors)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SAM2 features for training tiles"
    )
    parser.add_argument(
        "project_dir",
        help="Path to project folder containing train_images/"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detected if not specified)"
    )

    args = parser.parse_args()

    extract_all_features(
        args.project_dir,
        overwrite=args.overwrite,
        device=args.device
    )


if __name__ == "__main__":
    main()
