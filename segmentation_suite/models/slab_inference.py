#!/usr/bin/env python3
"""Z-tiling helpers for 3D slab models.

A slab model emits a prediction for every Z plane of its input, but only the planes
the Z-jitter ever placed the annotation on were actually supervised: with depth D and
jitter M that is the inclusive range [D//2 - M//2, D//2 + M//2], i.e. M+1 planes in
the middle of the slab. The margins above and below were never trained and typically
saturate, so inference must give them weight zero and step through Z by no more than
the trained width.

Ported from Ais _infer_slab (Ais/core/cli_fn.py), with one deliberate fix: Ais steps
by depth//2, which is larger than the trained width whenever depth > 2*(M//2)+1 and
therefore leaves unpredicted Z bands for its own depth-24 and depth-32 options. Here
the stride is the trained width itself, which is correct at every depth.
"""

import numpy as np


def trained_z_window(patch_depth: int, z_jitter: int):
    """Inclusive (lo, hi) output-plane range that training actually supervised."""
    D, M = int(patch_depth), int(z_jitter)
    if M <= 0:
        return 0, D - 1
    return D // 2 - M // 2, D // 2 + M // 2


def slab_z_weights(patch_depth: int, z_jitter: int) -> np.ndarray:
    """Top-hat blend weights: 1 across the trained window, 0 on the margins.

    Deliberately a hard rectangle rather than a taper. The sharp cut is what makes the
    untrained margins contribute exactly nothing, and what trims the unpredicted band
    at the very top and bottom of a volume instead of filling it with saturated values.
    """
    D = int(patch_depth)
    lo, hi = trained_z_window(D, z_jitter)
    w = np.zeros(D, dtype=np.float32)
    w[lo:hi + 1] = 1.0
    return w


def slab_z_starts(n_z: int, patch_depth: int, z_jitter: int):
    """Slab start indices covering every z in [0, n_z).

    Starts may be negative or run past the end; read the slab with clamped indices
    (see slab_read_indices). The final start is snapped so the tail is covered.
    """
    D = int(patch_depth)
    lo, hi = trained_z_window(D, z_jitter)
    stride = hi - lo + 1
    first = -lo                 # trained window opens at z = 0
    last = (n_z - 1) - hi       # trained window closes at z = n_z - 1
    if last <= first:
        return [first]
    starts = list(range(first, last, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def slab_read_indices(start: int, patch_depth: int, n_z: int) -> np.ndarray:
    """Z indices to read for a slab at `start`, clamped to the volume (as Ais does)."""
    return np.clip(np.arange(start, start + int(patch_depth)), 0, n_z - 1)


def read_slab_geometry(checkpoint_path: str, architecture: str, device=None):
    """Recover (patch_depth, z_jitter) for a slab model.

    The checkpoint is authoritative: a model trained at a different depth or jitter
    than the architecture currently declares must still be run the way it was trained.
    Falls back to the architecture defaults for checkpoints saved before slab support.
    """
    from . import architectures as arch_mod

    patch_depth = arch_mod.get_3d_patch_depth(architecture)
    z_jitter = arch_mod.get_z_jitter(architecture)

    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            if ckpt.get('patch_depth'):
                patch_depth = int(ckpt['patch_depth'])
            if 'z_jitter' in ckpt:
                z_jitter = int(ckpt['z_jitter'])
    except Exception as e:
        print(f"[slab] could not read geometry from {checkpoint_path}: {e}; "
              f"using architecture defaults D={patch_depth}, M={z_jitter}")

    return patch_depth, z_jitter
