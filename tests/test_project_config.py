"""Unit tests for the pure config/path utilities in ``project_config``.

These functions are UI-independent (CODE_REVIEW.md flags them as the natural first
tests), so they run without Qt, torch, or any data. Run with::

    pytest tests/
"""
from pathlib import Path

import pytest

from segmentation_suite import project_config as pc


# --------------------------------------------------------------------------- #
# Defaults / save-load round-trip / config merge
# --------------------------------------------------------------------------- #
def test_default_config_has_core_keys():
    cfg = pc.get_default_config()
    assert cfg["masks_dir"] == "masks"
    assert cfg["checkpoint_path"] == "checkpoint.pth"
    assert cfg["tile_size"] == 256
    assert cfg["interactive_mode"] is True


def test_save_load_roundtrip(tmp_path):
    assert pc.save_project_config(str(tmp_path), {"project_name": "demo", "tile_size": 128})
    loaded = pc.load_project_config(str(tmp_path))
    assert loaded is not None
    assert loaded["project_name"] == "demo"
    assert loaded["tile_size"] == 128
    # timestamps are stamped on save
    assert loaded["created_at"] and loaded["modified_at"]


def test_load_missing_returns_none(tmp_path):
    assert pc.load_project_config(str(tmp_path)) is None


def test_load_merges_new_default_keys(tmp_path):
    # A minimal/old config on disk must still expose newer default keys after load
    # (guards the "config merge dropping new default keys" bug from CODE_REVIEW.md).
    (tmp_path / pc.PROJECT_CONFIG_FILENAME).write_text('{"project_name": "old"}')
    loaded = pc.load_project_config(str(tmp_path))
    assert loaded["project_name"] == "old"
    assert loaded["masks_dir"] == "masks"        # default filled in
    assert loaded["tile_size"] == 256


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def test_resolve_path_relative(tmp_path):
    got = pc.resolve_path(str(tmp_path), "masks")
    assert Path(got) == tmp_path / "masks"


def test_resolve_path_absolute_unchanged(tmp_path):
    abs_path = tmp_path / "elsewhere" / "vol.zarr"
    got = pc.resolve_path(str(tmp_path), str(abs_path))
    assert Path(got) == abs_path


def test_resolve_path_empty():
    assert pc.resolve_path("/whatever", "") == ""


def test_make_relative_path_under_project(tmp_path):
    p = tmp_path / "masks" / "mask_00000.tif"
    got = pc.make_relative_path(str(tmp_path), str(p))
    assert Path(got) == Path("masks") / "mask_00000.tif"


def test_make_relative_path_outside_project_stays_absolute(tmp_path):
    outside = tmp_path.parent / "other_root" / "vol.tif"
    got = pc.make_relative_path(str(tmp_path), str(outside))
    assert Path(got) == outside


# --------------------------------------------------------------------------- #
# Training-folder naming (crop-size dependent)
# --------------------------------------------------------------------------- #
def test_training_folder_names_default_256():
    names = pc.get_training_folder_names(256)
    assert names["images"] == "train_images"
    assert names["masks"] == "train_masks"


@pytest.mark.parametrize("size", [128, 512])
def test_training_folder_names_suffixed(size):
    names = pc.get_training_folder_names(size)
    assert names["images"] == f"train_images_{size}"
    assert names["masks"] == f"train_masks_{size}"


# --------------------------------------------------------------------------- #
# Subprojects
# --------------------------------------------------------------------------- #
def test_create_and_list_subprojects(tmp_path):
    sp = pc.create_subproject(str(tmp_path), "cellA")
    assert sp.is_dir()
    assert (sp / "masks").is_dir()
    assert (sp / "train_images").is_dir()
    assert (sp / pc.PROJECT_CONFIG_FILENAME).exists()
    assert pc.list_subprojects(str(tmp_path)) == ["cellA"]


def test_create_subproject_duplicate_raises(tmp_path):
    pc.create_subproject(str(tmp_path), "cellA")
    with pytest.raises(ValueError):
        pc.create_subproject(str(tmp_path), "cellA")


@pytest.mark.parametrize("bad", ["a/b", "a\\b", ""])
def test_create_subproject_invalid_name_raises(tmp_path, bad):
    with pytest.raises(ValueError):
        pc.create_subproject(str(tmp_path), bad)


def test_active_subproject_default_and_set(tmp_path):
    assert pc.get_active_subproject(str(tmp_path)) == pc.DEFAULT_SUBPROJECT
    pc.create_subproject(str(tmp_path), "cellA")
    assert pc.set_active_subproject(str(tmp_path), "cellA")
    assert pc.get_active_subproject(str(tmp_path)) == "cellA"


def test_subproject_paths_use_crop_suffix(tmp_path):
    paths = pc.get_subproject_paths(str(tmp_path), "cellA", tile_size=128)
    assert paths["train_images_dir"].name == "train_images_128"
    assert paths["masks_dir"].name == "masks"


# --------------------------------------------------------------------------- #
# project_exists
# --------------------------------------------------------------------------- #
def test_project_exists_false_on_empty(tmp_path):
    assert pc.project_exists(str(tmp_path)) is False


def test_project_exists_true_with_config(tmp_path):
    pc.save_project_config(str(tmp_path), {"project_name": "x"})
    assert pc.project_exists(str(tmp_path)) is True


def test_project_exists_true_with_tiff(tmp_path):
    (tmp_path / "stack.tif").write_bytes(b"II*\x00")  # dummy TIFF-ish file
    assert pc.project_exists(str(tmp_path)) is True
