#!/usr/bin/env python3
"""
Project configuration file management.
Saves and loads project.json for seamless project resumption.
"""

import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_CONFIG_FILENAME = "project.json"


def get_default_config() -> dict:
    """Return default project configuration."""
    return {
        "version": "1.0",
        "project_name": "",
        "created_at": "",
        "modified_at": "",

        # Paths (relative to project_dir when possible)
        "raw_images_dir": "",
        "masks_dir": "masks",
        "train_images_dir": "train_images",
        "train_masks_dir": "train_masks",
        "checkpoint_path": "checkpoint.pth",

        # Training parameters
        "num_epochs": 5000,
        "batch_size": 2,
        "tile_size": 256,
        "learning_rate": 0.0001,

        # Session state (for resuming)
        "current_slice_index": 0,
        "edit_count": 0,
        "total_images": 0,

        # Flags
        "interactive_mode": True,
        "training_started": False,
        "training_complete": False,

        # Multi-user collaborative training settings
        "multi_user_enabled": False,
        "multi_user_sync_interval": 5,  # Sync weights every N epochs
        "multi_user_blend_ratio": 0.5,  # How much to blend global model (0=local, 1=global)
        "last_session_host": "",  # Last host IP:port for quick rejoin
        "last_session_room": "",  # Last relay room code for quick rejoin
        "multi_user_display_name": "",  # User's display name in sessions
    }


def save_project_config(project_dir: str, config: dict) -> bool:
    """Save project configuration to project.json.

    Args:
        project_dir: Path to project directory
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        config_path = project_dir / PROJECT_CONFIG_FILENAME

        # Update modification time
        config = config.copy()
        config["modified_at"] = datetime.now().isoformat()

        # Set creation time if not set
        if not config.get("created_at"):
            config["created_at"] = config["modified_at"]

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        print(f"Failed to save project config: {e}")
        return False


def load_project_config(project_dir: str) -> dict | None:
    """Load project configuration from project.json.

    Args:
        project_dir: Path to project directory

    Returns:
        Configuration dictionary, or None if not found/invalid
    """
    try:
        config_path = Path(project_dir) / PROJECT_CONFIG_FILENAME

        if not config_path.exists():
            return None

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Merge with defaults to ensure all keys exist
        default = get_default_config()
        default.update(config)

        return default
    except Exception as e:
        print(f"Failed to load project config: {e}")
        return None


def resolve_path(project_dir: str, path: str) -> str:
    """Resolve a path that may be relative to project_dir.

    Args:
        project_dir: Project directory path
        path: Path (absolute or relative)

    Returns:
        Absolute path
    """
    if not path:
        return ""

    path_obj = Path(path)

    # If already absolute, return as-is
    if path_obj.is_absolute():
        return str(path_obj)

    # Otherwise, make it relative to project_dir
    return str(Path(project_dir) / path_obj)


def make_relative_path(project_dir: str, path: str) -> str:
    """Make a path relative to project_dir if possible.

    Args:
        project_dir: Project directory path
        path: Path to make relative

    Returns:
        Relative path if under project_dir, otherwise absolute path
    """
    if not path:
        return ""

    try:
        path_obj = Path(path).resolve()
        project_obj = Path(project_dir).resolve()

        # Check if path is under project_dir
        rel_path = path_obj.relative_to(project_obj)
        return str(rel_path)
    except ValueError:
        # Path is not under project_dir, return absolute
        return str(path)


def project_exists(project_dir: str) -> bool:
    """Check if a valid project exists at the given path.

    Args:
        project_dir: Path to check

    Returns:
        True if project.json exists or project structure is valid
    """
    project_dir = Path(project_dir)

    # Check for project.json
    if (project_dir / PROJECT_CONFIG_FILENAME).exists():
        return True

    # Check for project structure (masks or train_images folder)
    if (project_dir / "masks").is_dir():
        return True
    if (project_dir / "train_images").is_dir():
        return True

    return False
