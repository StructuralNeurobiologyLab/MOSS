#!/usr/bin/env python3
"""Download the tutorial's raw EM data as a folder of TIFF slices.

Fetches a small centered block of the public Full Adult Fly Brain volume
(FAFB v14 CLAHE, 8 nm; Zheng et al. 2018) and writes it to ``raw_tiffs/`` next to
this script — 256 slices of 512 x 512, about 60 MB. The block is *not* committed to
the repo; run this once before starting the tutorial.

Requires ``cloud-volume`` for the download (public data, no credentials):

    pip install cloud-volume        # or: pip install -r ../requirements.txt

Then:

    python download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

# FAFB v14 CLAHE, mip 0 = 8 x 8 x 40 nm, 8-bit. Public Google Cloud bucket.
CLOUDPATH = "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe"
SX, SY, SZ = 512, 512, 256  # block size in voxels (x, y, z)


def main() -> None:
    out = Path(__file__).resolve().parent / "raw_tiffs"
    if out.is_dir() and len(list(out.glob("*.tif"))) >= SZ:
        print(f"raw_tiffs/ already has {len(list(out.glob('*.tif')))} slices — nothing to do.")
        return

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        sys.exit(
            "cloud-volume is required for the download.\n"
            "  pip install cloud-volume   (or: pip install -r ../requirements.txt)"
        )

    vol = CloudVolume(CLOUDPATH, mip=0, use_https=True, progress=True,
                      fill_missing=True, bounded=False)
    b = vol.bounds
    cx = (b.minpt[0] + b.maxpt[0]) // 2
    cy = (b.minpt[1] + b.maxpt[1]) // 2
    cz = (b.minpt[2] + b.maxpt[2]) // 2
    x0, y0, z0 = cx - SX // 2, cy - SY // 2, cz - SZ // 2

    print(f"Downloading {SX}x{SY}x{SZ} FAFB block (~{SX*SY*SZ/1e6:.0f} MB) ...")
    cutout = vol[x0:x0 + SX, y0:y0 + SY, z0:z0 + SZ]   # (x, y, z, 1)
    arr = np.asarray(cutout)[..., 0]
    arr = np.transpose(arr, (2, 1, 0))                 # -> (z, y, x)

    out.mkdir(parents=True, exist_ok=True)
    for z in range(arr.shape[0]):
        tifffile.imwrite(out / f"slice_{z:04d}.tif", arr[z],
                         photometric="minisblack", compression="deflate")
    print(f"Wrote {arr.shape[0]} slices to {out}")
    print("Now open MOSS, create a project, and point Raw Data at this raw_tiffs/ folder.")


if __name__ == "__main__":
    main()
