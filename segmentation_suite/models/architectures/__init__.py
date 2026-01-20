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
