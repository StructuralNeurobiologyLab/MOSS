#!/usr/bin/env python3
"""
Dynamic architecture loader.

Drop Python files into this folder to add new model architectures.
Each file should define:
    - MODEL_CLASS: The nn.Module class for the model
    - ARCHITECTURE_ID: A unique string identifier (e.g., 'unet_deep')
    - ARCHITECTURE_NAME: Human-readable name (e.g., 'UNet Deep (Large RF)')
    - ARCHITECTURE_DESCRIPTION (optional): Longer description for UI tooltips
"""

import os
import importlib.util
from pathlib import Path
from typing import Dict, Type, Optional
import torch.nn as nn


# Registry of discovered architectures
_architectures: Dict[str, Type[nn.Module]] = {}
_architecture_names: Dict[str, str] = {}
_architecture_descriptions: Dict[str, str] = {}
_architecture_losses: Dict[str, str] = {}  # Optional preferred loss function
_architecture_checkpoints: Dict[str, str] = {}  # Pretrained checkpoint paths
_architecture_training_v2: Dict[str, bool] = {}  # v2 training improvements flag
_architecture_n_context: Dict[str, int] = {}  # N_CONTEXT_SLICES for 2.5D variants
_architecture_slice_spacing: Dict[str, int] = {}  # SLICE_SPACING for 2.5D variants
_architecture_uses_z_coord: Dict[str, bool] = {}  # USES_Z_COORD flag
_architecture_is_3d: Dict[str, bool] = {}  # IS_3D flag for volumetric models
_architecture_patch_depth: Dict[str, int] = {}  # PATCH_DEPTH for 3D models
_architecture_patch_size: Dict[str, int] = {}  # PATCH_SIZE for 3D models
_architecture_is_slab: Dict[str, bool] = {}  # IS_SLAB flag for slab-output 3D models
_architecture_z_jitter: Dict[str, int] = {}  # Z_JITTER margin for slab models
_architecture_z_downsample: Dict[str, int] = {}  # Z_DOWNSAMPLE (Z pooling factor)
_architecture_hidden: Dict[str, bool] = {}  # HIDDEN flag - kept but not shown in UI
_loaded = False


def _load_architectures():
    """Scan the architectures folder and load all valid architecture files."""
    global _loaded
    if _loaded:
        return

    arch_dir = Path(__file__).parent

    # Load all .py files in the architectures folder (except __init__.py)
    for filepath in arch_dir.glob("*.py"):
        if filepath.name.startswith("_"):
            continue

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check for required attributes
            if not hasattr(module, 'MODEL_CLASS'):
                print(f"Warning: {filepath.name} missing MODEL_CLASS, skipping")
                continue
            if not hasattr(module, 'ARCHITECTURE_ID'):
                print(f"Warning: {filepath.name} missing ARCHITECTURE_ID, skipping")
                continue
            if not hasattr(module, 'ARCHITECTURE_NAME'):
                print(f"Warning: {filepath.name} missing ARCHITECTURE_NAME, skipping")
                continue

            arch_id = module.ARCHITECTURE_ID
            _architectures[arch_id] = module.MODEL_CLASS
            _architecture_names[arch_id] = module.ARCHITECTURE_NAME
            _architecture_descriptions[arch_id] = getattr(
                module, 'ARCHITECTURE_DESCRIPTION', ''
            )
            _architecture_losses[arch_id] = getattr(
                module, 'PREFERRED_LOSS', 'bce'  # Default to BCE
            )
            # Check for pretrained checkpoint path
            pretrained_ckpt = getattr(module, 'PRETRAINED_CHECKPOINT', None)
            if pretrained_ckpt:
                _architecture_checkpoints[arch_id] = pretrained_ckpt

            # Check for v2 training improvements flag
            if getattr(module, 'TRAINING_V2', False):
                _architecture_training_v2[arch_id] = True

            # Store 2.5D context parameters if present
            n_ctx = getattr(module, 'N_CONTEXT_SLICES', None)
            if n_ctx is not None:
                _architecture_n_context[arch_id] = n_ctx
            spacing = getattr(module, 'SLICE_SPACING', None)
            if spacing is not None:
                _architecture_slice_spacing[arch_id] = spacing

            if getattr(module, 'USES_Z_COORD', False):
                _architecture_uses_z_coord[arch_id] = True

            if getattr(module, 'IS_3D', False):
                _architecture_is_3d[arch_id] = True
            patch_depth = getattr(module, 'PATCH_DEPTH', None)
            if patch_depth is not None:
                _architecture_patch_depth[arch_id] = patch_depth
            patch_size = getattr(module, 'PATCH_SIZE', None)
            if patch_size is not None:
                _architecture_patch_size[arch_id] = patch_size

            if getattr(module, 'IS_SLAB', False):
                # A slab model is a 3D model. Callers guard the volumetric setup with
                # IS_3D and then read it under IS_SLAB, so letting the two disagree
                # would raise NameError deep inside training rather than here.
                if not getattr(module, 'IS_3D', False):
                    print(f"Warning: {filepath.name} sets IS_SLAB without IS_3D; "
                          f"treating it as 3D")
                    _architecture_is_3d[arch_id] = True
                _architecture_is_slab[arch_id] = True
            z_jitter = getattr(module, 'Z_JITTER', None)
            if z_jitter is not None:
                _architecture_z_jitter[arch_id] = int(z_jitter)
            # Captured here rather than looked up later: these modules are exec'd from
            # file and never land in sys.modules, so there is no way back to them.
            z_down = getattr(module, 'Z_DOWNSAMPLE', None)
            if z_down is not None:
                _architecture_z_downsample[arch_id] = int(z_down)

            if getattr(module, 'HIDDEN', False):
                _architecture_hidden[arch_id] = True

            # Debug output (commented out for cleaner startup)
            # print(f"Loaded architecture: {arch_id} ({module.ARCHITECTURE_NAME})")

        except Exception as e:
            print(f"Failed to load architecture from {filepath.name}: {e}")

    _loaded = True


def get_available_architectures(include_hidden: bool = False) -> Dict[str, str]:
    """
    Get dict of available architectures: {architecture_id: display_name}

    Always includes built-in architectures plus any discovered from files.
    Hidden architectures are excluded unless include_hidden=True.
    """
    _load_architectures()

    # Start with discovered architectures (file-based)
    result = dict(_architecture_names)

    # Filter out hidden architectures unless requested
    if not include_hidden:
        result = {k: v for k, v in result.items() if k not in _architecture_hidden}

    return result


def get_architecture_description(arch_id: str) -> str:
    """Get the description for an architecture."""
    _load_architectures()
    return _architecture_descriptions.get(arch_id, '')


def get_preferred_loss(arch_id: str) -> str:
    """
    Get the preferred loss function for an architecture.

    Returns: 'bce', 'dice', or 'bce_dice' (combined)
    """
    _load_architectures()
    return _architecture_losses.get(arch_id, 'bce')


def get_model_class(architecture: str) -> Type[nn.Module]:
    """
    Get the model class for the given architecture name.

    Checks discovered architectures first, then falls back to built-ins.
    """
    _load_architectures()

    # Check discovered architectures first
    if architecture in _architectures:
        return _architectures[architecture]

    # Fall back to built-in architectures
    print(f"[Arch] WARNING: {architecture} not found in discovered architectures: {list(_architectures.keys())}")
    from ..unet import ARCHITECTURES as builtin_architectures
    if architecture in builtin_architectures:
        return builtin_architectures[architecture]

    # Last resort - raise an error instead of potential infinite loop
    raise ValueError(f"Unknown architecture: {architecture}. Available: {list(_architectures.keys())}")


def get_checkpoint_filename(architecture: str) -> str:
    """Get the checkpoint filename for the given architecture."""
    if architecture == 'unet':
        return 'checkpoint.pth'
    else:
        return f'checkpoint_{architecture}.pth'


def is_pretrained_architecture(arch_id: str) -> bool:
    """
    Check if an architecture has a pretrained checkpoint available.

    Args:
        arch_id: Architecture identifier (e.g., 'lsd_boundary_2d')

    Returns:
        True if pretrained checkpoint is available
    """
    _load_architectures()
    return arch_id in _architecture_checkpoints


def get_pretrained_checkpoint(arch_id: str) -> Optional[str]:
    """
    Get the path to the pretrained checkpoint for an architecture.

    Args:
        arch_id: Architecture identifier (e.g., 'lsd_boundary_2d')

    Returns:
        Path to pretrained checkpoint, or None if not available
    """
    _load_architectures()
    return _architecture_checkpoints.get(arch_id)


def uses_training_v2(arch_id: str) -> bool:
    """Check if architecture uses v2 training improvements (grad clipping, proper resume)."""
    _load_architectures()
    return _architecture_training_v2.get(arch_id, False)


def get_n_context_slices(arch_id: str) -> int:
    """Get the number of input channels/slices for a 2.5D architecture.

    Returns 3 as default for any 2.5D architecture without explicit metadata.
    Returns 1 for non-2.5D architectures.
    """
    _load_architectures()
    if arch_id in _architecture_n_context:
        return _architecture_n_context[arch_id]
    # Fallback: 3 for any 2.5D, 1 otherwise
    return 3 if '25d' in arch_id.lower() else 1


def get_slice_spacing(arch_id: str) -> int:
    """Get the z-spacing between context slices for a 2.5D architecture.

    Returns 3 as default (z-3, z, z+3) for standard 2.5D.
    """
    _load_architectures()
    if arch_id in _architecture_slice_spacing:
        return _architecture_slice_spacing[arch_id]
    return 3  # Default spacing


def uses_z_coord(arch_id: str) -> bool:
    """Check if architecture uses a z-coordinate input channel."""
    _load_architectures()
    return _architecture_uses_z_coord.get(arch_id, False)


def is_3d_architecture(arch_id: str) -> bool:
    """Check if architecture is a 3D volumetric model."""
    _load_architectures()
    return _architecture_is_3d.get(arch_id, False)


def get_3d_patch_depth(arch_id: str) -> int:
    """Get the Z depth for 3D model patches. Default 32."""
    _load_architectures()
    return _architecture_patch_depth.get(arch_id, 32)


def get_3d_patch_size(arch_id: str) -> int:
    """Get the XY size for 3D model patches. Default 128."""
    _load_architectures()
    return _architecture_patch_size.get(arch_id, 128)


def is_slab_architecture(arch_id: str) -> bool:
    """Check if the architecture emits a full Z-slab rather than a single slice.

    Slab models are trained from sparse 2D annotations: one plane of each slab
    carries a real label and the rest are masked out of the loss.
    """
    _load_architectures()
    return _architecture_is_slab.get(arch_id, False)


def is_hub_trainable(arch_id: str) -> bool:
    """Can the multi-user hub actually train this architecture?

    False for slab models: hub crop transfer carries only the 2D and 2.5D variants, so
    a slab session would start, report itself running, and never train. Worse, switching
    a live session to one stops the current trainer and rebuilds the pool, so the run it
    replaced cannot simply be resumed. Keep them out of the hub's pickers entirely and
    train them locally.
    """
    return not is_slab_architecture(arch_id)


def filter_hub_trainable(architectures: Dict[str, str]) -> Dict[str, str]:
    """Drop everything the hub cannot train from an {arch_id: display_name} map."""
    return {k: v for k, v in architectures.items() if is_hub_trainable(k)}


def get_z_jitter(arch_id: str) -> int:
    """Get the Z margin M for a slab architecture (0 for everything else).

    Training stores slabs of depth PATCH_DEPTH + M and random-crops PATCH_DEPTH
    out of them, so the annotated plane lands anywhere in
    [PATCH_DEPTH//2 - M//2, PATCH_DEPTH//2 + M//2]. Only those M+1 output planes
    are ever supervised, which is exactly the range inference may trust.
    """
    _load_architectures()
    return _architecture_z_jitter.get(arch_id, 0)


def validate_slab_geometry(arch_id: str, patch_depth: int = None,
                           z_jitter: int = None) -> None:
    """Raise ValueError if a slab configuration cannot produce valid labels.

    Two independent constraints:
      - D must be a multiple of the model's Z downsampling, or the U-Net skip
        connections do not align.
      - M must be <= D - 2, or the jittered label index escapes the slab. (This
        is a real crash in the Ais reference at D=8, M=8.)
    """
    _load_architectures()
    D = get_3d_patch_depth(arch_id) if patch_depth is None else int(patch_depth)
    M = get_z_jitter(arch_id) if z_jitter is None else int(z_jitter)

    # Recorded at load time. A slab architecture that omits Z_DOWNSAMPLE is a bug: we
    # would otherwise guess a divisor and let a bad depth through to fail inside
    # forward(), which is exactly what this function exists to prevent.
    if arch_id not in _architecture_z_downsample:
        raise ValueError(
            f"{arch_id}: declares no Z_DOWNSAMPLE, so PATCH_DEPTH cannot be validated. "
            f"Add Z_DOWNSAMPLE (the product of the model's Z pooling strides).")
    z_down = _architecture_z_downsample[arch_id]

    if D % z_down:
        raise ValueError(
            f"{arch_id}: PATCH_DEPTH={D} must be a multiple of {z_down} "
            f"(Z downsampling); skip connections would not align.")
    if M < 0:
        raise ValueError(f"{arch_id}: Z_JITTER={M} must be >= 0.")
    if M and M % 2:
        # An odd margin makes the trained range [D//2-M//2, D//2+(M+1)//2] asymmetric,
        # while the inference top-hat is symmetric about D//2 -- so one trained plane
        # gets discarded and the two windows stop coinciding.
        raise ValueError(
            f"{arch_id}: Z_JITTER={M} must be even, so the trained Z range stays "
            f"symmetric about PATCH_DEPTH//2 and matches the inference window.")
    if M and M > D - 2:
        raise ValueError(
            f"{arch_id}: Z_JITTER={M} must be <= PATCH_DEPTH-2 ({D - 2}); "
            f"otherwise the jittered label plane lands outside the {D}-deep slab.")
