#!/usr/bin/env python3
"""
Build a ready-to-open MOSS demo project from public FAFB EM data.

Downloads a small block of the Full Adult Fly Brain (FAFB v14) electron-microscopy
volume, converts it into the pyramidal OME-Zarr layout MOSS uses, and writes a
``project.json`` preconfigured to use the bundled pretrained LSD boundary/membrane
model. Optionally runs the model on one slice and saves a preview so you can verify
the whole pipeline end to end in a few minutes.

Data source (public, no credentials needed):
    precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe
    Zheng et al. 2018, "A Complete Electron Microscopy Volume of the Brain of
    Adult Drosophila melanogaster", Cell. CLAHE-normalized variant, 8x8x40 nm.

Usage
-----
    # ~64 MB block, fastest way to verify the pipeline:
    python build_demo.py --preset quick

    # ~0.5 GB block (matches the tutorial figures):
    python build_demo.py --preset demo

    python build_demo.py --help   # all options

Requires ``cloud-volume`` for the download step (see requirements.txt in this
folder); everything else uses MOSS's own dependencies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

# FAFB v14 CLAHE, mip 0 = 8 x 8 x 40 nm, 8-bit.
CLOUDPATH = "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe"

# (x, y, z) block sizes in voxels. quick ~= 64 MB, demo ~= 512 MB.
PRESETS = {
    "quick": (512, 512, 256),
    "demo": (1024, 1024, 512),
}


def download_block(sx: int, sy: int, sz: int, out_tif: Path) -> Path:
    """Cut a centered (sx, sy, sz) block out of FAFB and save it as a TIFF stack."""
    try:
        from cloudvolume import CloudVolume
    except ImportError:
        sys.exit(
            "cloud-volume is required for the download step.\n"
            "  pip install cloud-volume   (or: pip install -r requirements.txt)"
        )
    vol = CloudVolume(CLOUDPATH, mip=0, use_https=True, progress=True,
                      fill_missing=True, bounded=False)
    b = vol.bounds
    cx = (b.minpt[0] + b.maxpt[0]) // 2
    cy = (b.minpt[1] + b.maxpt[1]) // 2
    cz = (b.minpt[2] + b.maxpt[2]) // 2
    x0, y0, z0 = cx - sx // 2, cy - sy // 2, cz - sz // 2
    print(f"Downloading {sx}x{sy}x{sz} voxels (~{sx*sy*sz/1e6:.0f} MB) "
          f"at x[{x0}] y[{y0}] z[{z0}] ...")
    cutout = vol[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz]     # (x, y, z, 1)
    arr = np.asarray(cutout)[..., 0]
    arr = np.transpose(arr, (2, 1, 0))                    # -> (z, y, x)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_tif, arr, photometric="minisblack",
                     metadata={"axes": "ZYX"}, bigtiff=True)
    print(f"  wrote {out_tif} ({out_tif.stat().st_size/1e6:.0f} MB), shape {arr.shape}")
    return out_tif


def build_project(src_tif: Path, project_dir: Path, num_levels: int = 4) -> None:
    """Convert the TIFF stack into MOSS's raw_data.zarr pyramid + project.json."""
    # These are MOSS's own conversion routines — importable once MOSS is installed.
    from segmentation_suite.em_pipeline.data.convert import tiff_to_zarr
    from segmentation_suite.em_pipeline.data.pyramid import generate_pyramid
    from segmentation_suite.project_config import save_project_config
    from segmentation_suite.models.architectures import get_pretrained_checkpoint

    project_dir.mkdir(parents=True, exist_ok=True)
    dest = project_dir / "raw_data.zarr"
    dest.mkdir(parents=True, exist_ok=True)

    print("Building raw_data.zarr (base level) ...")
    tiff_to_zarr(str(src_tif), str(dest / "0"), chunk_size=(64, 256, 256), num_workers=2)

    print(f"Generating {num_levels - 1} pyramid levels ...")
    generate_pyramid(str(dest / "0"), str(dest), num_levels=num_levels,
                     factors=(2, 2, 2), method="mean", num_workers=2)

    # OME-Zarr multiscale metadata (matches MOSS's own converter output).
    multiscales = [{
        "version": "0.4", "name": dest.stem,
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [
            {"path": str(lvl), "coordinateTransformations":
                [{"type": "scale", "scale": [1.0, 2 ** lvl, 2 ** lvl, 2 ** lvl]}]}
            for lvl in range(num_levels)
        ],
        "type": "mean",
    }]
    (dest / ".zattrs").write_text(json.dumps({"multiscales": multiscales}, indent=2))
    (dest / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    # Per-slice XY TIFFs: MOSS's multi-view ("MOSS 2D") predictor reads TIFF
    # folders, not Zarr, so ship these for the Combined Segmentation workflow.
    raw_images = project_dir / "raw_images"
    raw_images.mkdir(exist_ok=True)
    vol = tifffile.imread(src_tif)
    for z in range(vol.shape[0]):
        tifffile.imwrite(raw_images / f"slice_{z:05d}.tif", vol[z], photometric="minisblack")
    print(f"Wrote {vol.shape[0]} XY slices to raw_images/")

    # Preconfigure the pretrained LSD boundary model + point checkpoint_path at
    # the bundled checkpoint (its filename contains 'lsd_boundary', which the
    # multi-view predictor uses to select the lsd_boundary_2d architecture).
    ckpt_src = get_pretrained_checkpoint("lsd_boundary_2d")
    checkpoint_path = None
    if ckpt_src and Path(ckpt_src).exists():
        import shutil
        checkpoint_path = project_dir / "checkpoint_lsd_boundary_2d.pth"
        shutil.copy2(ckpt_src, checkpoint_path)

    config = {
        "project_name": "FAFB v14 LSD boundary demo",
        "raw_images_dir": None,
        "tile_size": 256,
        "interactive_mode": True,
        "architecture": "lsd_boundary_2d",
        "prediction_architecture": "lsd_boundary_2d",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "checkpoint.pth",
        "notes": "FAFB v14 CLAHE demo. Live predictions use the bundled pretrained "
                 "LSD MtLSD membrane model.",
    }
    save_project_config(str(project_dir), config)
    print(f"Wrote project.json (architecture=lsd_boundary_2d) at {project_dir}")


def preview_prediction(project_dir: Path) -> None:
    """Run the pretrained LSD model on the middle slice and save a raw|membrane PNG."""
    import torch
    from PIL import Image
    from segmentation_suite.models.unet import load_model, get_device
    from segmentation_suite.models.architectures import get_pretrained_checkpoint

    ckpt = get_pretrained_checkpoint("lsd_boundary_2d")
    if not ckpt or not Path(ckpt).exists():
        print("Pretrained checkpoint not found; skipping preview.")
        return
    dev = get_device()
    model = load_model(ckpt, n_channels=1, device=dev, architecture="lsd_boundary_2d")

    slices = sorted((project_dir / "raw_images").glob("*.tif"))
    mid = slices[len(slices) // 2]
    raw = tifffile.imread(mid).astype(np.float32)
    x = torch.from_numpy(raw / 255.0)[None, None].to(dev)
    with torch.no_grad():
        memb = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    raw8 = raw.astype(np.uint8)
    memb8 = (memb * 255).astype(np.uint8)
    gap = np.full((raw8.shape[0], 8), 255, np.uint8)
    out = project_dir / "lsd_preview_raw_vs_membrane.png"
    Image.fromarray(np.concatenate([raw8, gap, memb8], 1)).save(out)
    print(f"Saved preview (device={dev}): {out}  [membrane mean={memb.mean():.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="quick",
                    help="Block size: quick (~64 MB, default) or demo (~512 MB).")
    ap.add_argument("--out", type=Path, default=Path("./fafb_lsd_demo_project"),
                    help="Project directory to create (default: ./fafb_lsd_demo_project).")
    ap.add_argument("--no-predict", action="store_true",
                    help="Skip the LSD preview render at the end.")
    args = ap.parse_args()

    sx, sy, sz = PRESETS[args.preset]
    project_dir = args.out.resolve()
    src_tif = project_dir / "_source" / f"fafb_v14_clahe_{sx}x{sy}x{sz}.tif"

    if not src_tif.exists():
        download_block(sx, sy, sz, src_tif)
    else:
        print(f"Reusing existing download: {src_tif}")

    build_project(src_tif, project_dir)

    if not args.no_predict:
        preview_prediction(project_dir)

    print("\nDone. Open this project in MOSS:")
    print(f"    moss    ->  Load Project  ->  {project_dir}")
    print("Then Segmentation -> MOSS (Recommended); Browse the Input to the")
    print("project's raw_images folder, check XY, and Run.")


if __name__ == "__main__":
    main()
