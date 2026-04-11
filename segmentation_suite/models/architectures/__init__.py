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

            # Debug output (commented out for cleaner startup)
            # print(f"Loaded architecture: {arch_id} ({module.ARCHITECTURE_NAME})")

        except Exception as e:
            print(f"Failed to load architecture from {filepath.name}: {e}")

    _loaded = True


def get_available_architectures() -> Dict[str, str]:
    """
    Get dict of available architectures: {architecture_id: display_name}

    Always includes built-in architectures plus any discovered from files.
    """
    _load_architectures()

    # Start with built-in architectures from unet.py
    from ..unet import ARCHITECTURE_NAMES as builtin_names
    result = dict(builtin_names)

    # Add discovered architectures (may override built-ins)
    result.update(_architecture_names)

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
