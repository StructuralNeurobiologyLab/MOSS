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
        "num_epochs": 50000,
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


# =========================================================================
# Subproject helpers
# =========================================================================

SUBPROJECTS_DIR_NAME = "subprojects"
DEFAULT_SUBPROJECT = "default"

# Directories that belong to a subproject (per-label training data)
SUBPROJECT_DIRS = [
    "masks",
    "train_images",
    "train_masks",
    "train_images_25d",
    "train_masks_25d",
    "train_images_dwarf25d",
    "train_masks_dwarf25d",
    "train_images_3d",
    "train_masks_3d",
    "sam2_features",
]


def get_subprojects_root(project_dir: str) -> Path:
    """Return the subprojects root directory for a project."""
    return Path(project_dir) / SUBPROJECTS_DIR_NAME


def has_subprojects(project_dir: str) -> bool:
    """Check if a project uses subprojects."""
    sp_root = get_subprojects_root(project_dir)
    return sp_root.is_dir() and any(sp_root.iterdir())


def list_subprojects(project_dir: str) -> list[str]:
    """List all subproject names in a project, sorted alphabetically.

    Returns empty list for legacy projects that haven't been migrated.
    """
    sp_root = get_subprojects_root(project_dir)
    if not sp_root.is_dir():
        return []
    return sorted([
        d.name for d in sp_root.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])


def get_active_subproject(project_dir: str) -> str:
    """Get the active subproject name from the root project config.

    Returns DEFAULT_SUBPROJECT if not set or for legacy projects.
    """
    config = load_project_config(project_dir)
    if config:
        return config.get("active_subproject", DEFAULT_SUBPROJECT)
    return DEFAULT_SUBPROJECT


def set_active_subproject(project_dir: str, subproject_name: str) -> bool:
    """Set the active subproject in the root project config."""
    config = load_project_config(project_dir)
    if config is None:
        config = get_default_config()
    config["active_subproject"] = subproject_name
    return save_project_config(project_dir, config)


def get_subproject_dir(project_dir: str, subproject_name: str) -> Path:
    """Get the directory path for a specific subproject."""
    return get_subprojects_root(project_dir) / subproject_name


def create_subproject(project_dir: str, subproject_name: str) -> Path:
    """Create a new subproject directory with all required subdirectories.

    Args:
        project_dir: Root project directory
        subproject_name: Name for the new subproject

    Returns:
        Path to the new subproject directory

    Raises:
        ValueError: If subproject already exists or name is invalid
    """
    # Validate name (filesystem-safe)
    if not subproject_name or '/' in subproject_name or '\\' in subproject_name:
        raise ValueError(f"Invalid subproject name: {subproject_name}")

    sp_dir = get_subproject_dir(project_dir, subproject_name)
    if sp_dir.exists():
        raise ValueError(f"Subproject '{subproject_name}' already exists")

    # Create subproject directory and subdirectories
    sp_dir.mkdir(parents=True, exist_ok=True)
    for dirname in SUBPROJECT_DIRS:
        (sp_dir / dirname).mkdir(exist_ok=True)

    # Create a subproject config
    sp_config = {
        "subproject_name": subproject_name,
        "created_at": datetime.now().isoformat(),
        "current_slice_index": 0,
        "edit_count": 0,
        "architecture": "",
        "prediction_architecture": "",
    }
    with open(sp_dir / PROJECT_CONFIG_FILENAME, 'w') as f:
        json.dump(sp_config, f, indent=2)

    return sp_dir


def migrate_to_subprojects(project_dir: str, subproject_name: str = None) -> str:
    """Migrate a legacy (flat) project to the subprojects layout.

    Moves masks/, train_images/, train_masks/, etc. into
    subprojects/<subproject_name>/.

    Args:
        project_dir: Root project directory
        subproject_name: Name for the migrated subproject (default: DEFAULT_SUBPROJECT)

    Returns:
        The subproject name used
    """
    import shutil

    if subproject_name is None:
        subproject_name = DEFAULT_SUBPROJECT

    project_path = Path(project_dir)
    sp_dir = get_subproject_dir(project_dir, subproject_name)
    sp_dir.mkdir(parents=True, exist_ok=True)

    moved_any = False
    for dirname in SUBPROJECT_DIRS:
        src = project_path / dirname
        dst = sp_dir / dirname
        if src.is_dir() and not dst.exists():
            shutil.move(str(src), str(dst))
            moved_any = True
            print(f"[Migrate] Moved {dirname}/ → subprojects/{subproject_name}/{dirname}/")
        elif not dst.exists():
            dst.mkdir(exist_ok=True)

    # Move checkpoint files (checkpoint_*.pth, refiner_checkpoint.pth)
    for ckpt_file in project_path.glob("checkpoint_*.pth"):
        dst = sp_dir / ckpt_file.name
        if not dst.exists():
            shutil.move(str(ckpt_file), str(dst))
            print(f"[Migrate] Moved {ckpt_file.name} → subprojects/{subproject_name}/")
            moved_any = True

    refiner_ckpt = project_path / "refiner_checkpoint.pth"
    if refiner_ckpt.exists() and not (sp_dir / "refiner_checkpoint.pth").exists():
        shutil.move(str(refiner_ckpt), str(sp_dir / "refiner_checkpoint.pth"))
        moved_any = True

    # Create subproject config from existing project config
    existing_config = load_project_config(project_dir)
    sp_config = {
        "subproject_name": subproject_name,
        "created_at": datetime.now().isoformat(),
        "current_slice_index": existing_config.get("current_slice_index", 0) if existing_config else 0,
        "edit_count": existing_config.get("edit_count", 0) if existing_config else 0,
        "architecture": existing_config.get("architecture", "") if existing_config else "",
        "prediction_architecture": existing_config.get("prediction_architecture", "") if existing_config else "",
    }
    with open(sp_dir / PROJECT_CONFIG_FILENAME, 'w') as f:
        json.dump(sp_config, f, indent=2)

    # Update root config with active subproject
    if existing_config:
        existing_config["active_subproject"] = subproject_name
        save_project_config(project_dir, existing_config)

    if moved_any:
        print(f"[Migrate] Legacy project migrated to subproject '{subproject_name}'")
    else:
        print(f"[Migrate] Created subproject '{subproject_name}' (no legacy data to move)")

    return subproject_name


def load_subproject_config(project_dir: str, subproject_name: str) -> dict | None:
    """Load a subproject's config (subprojects/<name>/project.json)."""
    sp_dir = get_subproject_dir(project_dir, subproject_name)
    config_path = sp_dir / PROJECT_CONFIG_FILENAME
    if not config_path.exists():
        return None
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load subproject config: {e}")
        return None


def save_subproject_config(project_dir: str, subproject_name: str, config: dict) -> bool:
    """Save a subproject's config."""
    sp_dir = get_subproject_dir(project_dir, subproject_name)
    sp_dir.mkdir(parents=True, exist_ok=True)
    config_path = sp_dir / PROJECT_CONFIG_FILENAME
    try:
        config = config.copy()
        config["modified_at"] = datetime.now().isoformat()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save subproject config: {e}")
        return False


def get_subproject_paths(project_dir: str, subproject_name: str) -> dict:
    """Get all relevant paths for a subproject.

    Returns a dict with keys: masks_dir, train_images_dir, train_masks_dir,
    train_images_25d_dir, train_masks_25d_dir, and checkpoint base dir.
    All paths are absolute Path objects.
    """
    sp_dir = get_subproject_dir(project_dir, subproject_name)
    return {
        "subproject_dir": sp_dir,
        "masks_dir": sp_dir / "masks",
        "train_images_dir": sp_dir / "train_images",
        "train_masks_dir": sp_dir / "train_masks",
        "train_images_25d_dir": sp_dir / "train_images_25d",
        "train_masks_25d_dir": sp_dir / "train_masks_25d",
        "train_images_dwarf25d_dir": sp_dir / "train_images_dwarf25d",
        "train_masks_dwarf25d_dir": sp_dir / "train_masks_dwarf25d",
        "train_images_3d_dir": sp_dir / "train_images_3d",
        "train_masks_3d_dir": sp_dir / "train_masks_3d",
        "sam2_features_dir": sp_dir / "sam2_features",
    }


def project_exists(project_dir: str) -> bool:
    """Check if a valid project exists at the given path.

    Args:
        project_dir: Path to check

    Returns:
        True if project.json exists or project structure is valid
    """
    project_dir = Path(project_dir)

    if not project_dir.exists():
        return False

    # Check for project.json
    if (project_dir / PROJECT_CONFIG_FILENAME).exists():
        return True

    # Check for project structure (masks or train_images folder)
    if (project_dir / "masks").is_dir():
        return True
    if (project_dir / "train_images").is_dir():
        return True
    if (project_dir / "train_masks").is_dir():
        return True
    if (project_dir / "labels").is_dir():
        return True

    # Check for TIFF files
    tiff_files = list(project_dir.glob("**/*.tif")) + list(project_dir.glob("**/*.tiff"))
    if tiff_files:
        return True

    # Check for Zarr volumes
    zarr_dirs = list(project_dir.glob("**/*.zarr"))
    if zarr_dirs:
        return True

    return False
