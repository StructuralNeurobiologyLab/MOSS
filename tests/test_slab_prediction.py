"""Tests for the streaming 3D/slab prediction path in PredictWorker.

The bug these pin down: _predict_folder_3d used to allocate the input volume plus a
sum and a count accumulator at full (total_z, h, w), which asked for 356 GiB *each*
on a 4607x3475x5963 dataset and simply raised before predicting anything. Peak
memory must depend on patch_depth, not on how many slices the volume has.

The blend arithmetic itself is covered in test_slab_training.py against the helpers;
here it is checked through the worker, because streaming is where a plane can be
written before the last slab that owed it a contribution has run.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from segmentation_suite.models.slab_inference import trained_z_window
from segmentation_suite.workers.predict_worker import PredictWorker

D, M = 8, 4
PATCH = 8
PIN_PERIOD = 4  # see write_volume


class Threshold:
    """Stand-in model: bright voxels in, large positive logits out.

    Returns per-voxel logits so sigmoid saturates, making the expected mask exactly
    the thresholded input and any blending error visible as a flipped pixel.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return torch.where(x > 0.5, 20.0, -20.0)


def write_volume(folder, n_z, h=PATCH, w=PATCH, seed=0):
    """Write n_z TIFFs whose every Z column contains both 0 and 255.

    Patches are normalized by their own min/max, so the expected mask is only
    predictable if that range is the same for all of them. Z is always read at full
    patch_depth, so pinning an all-0 and an all-255 plane every PIN_PERIOD planes puts
    both extremes in every possible patch -- whatever the XY grid or Z stride does,
    and including the clamped reads at either end of the volume. Normalization is then
    exactly v/255 everywhere and the expected output is just (v > 127.5).
    """
    assert D >= 2 * PIN_PERIOD, "a patch must span at least two pinning periods"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vol = rng.integers(0, 256, size=(n_z, h, w), dtype=np.uint8)
    vol[0::PIN_PERIOD] = 0
    vol[1::PIN_PERIOD] = 255
    for z in range(n_z):
        Image.fromarray(vol[z]).save(folder / f"s{z:04d}.tif")
    return vol


def read_masks(folder, n_z):
    return np.stack([
        np.array(Image.open(folder / f"s{z:04d}_pred.tif")) for z in range(n_z)
    ])


def predict(tmp_path, n_z, model=None, jitter=M, depth=D, overlap=0,
            h=PATCH, w=PATCH, seed=0, spy=None):
    """Run the worker's 3D path over a freshly written volume."""
    in_dir, out_dir = tmp_path / f"in{n_z}_{seed}", tmp_path / f"out{n_z}_{seed}"
    vol = write_volume(in_dir, n_z, h=h, w=w, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    worker = PredictWorker({})
    model = model if model is not None else Threshold()
    if spy is not None:
        spy(worker)
    worker._predict_folder_3d(model, in_dir, out_dir, PATCH, depth, overlap,
                             'xy', torch.device('cpu'), z_jitter=jitter)
    return vol, read_masks(out_dir, n_z), model


# ------------------------------------------------------------------- memory bounding
class AllocSpy:
    """Records the largest array np.zeros is asked for during a run."""

    def __init__(self, monkeypatch):
        self.max_elems = 0
        self.shapes = []
        real = np.zeros

        def spy(shape, *a, **kw):
            shp = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
            self.max_elems = max(self.max_elems, int(np.prod(shp)))
            if len(shp) == 3:
                self.shapes.append(shp)
            return real(shape, *a, **kw)

        monkeypatch.setattr(np, 'zeros', spy)


@pytest.mark.parametrize("n_z", [40, 160])
def test_peak_allocation_does_not_scale_with_slice_count(tmp_path, monkeypatch, n_z):
    """The regression itself: no allocation may be proportional to total_z."""
    spy = AllocSpy(monkeypatch)
    predict(tmp_path, n_z, h=16, w=16)

    full_volume = n_z * 16 * 16
    assert spy.max_elems < full_volume, (
        f"allocated {spy.max_elems} elements, i.e. whole-volume sized ({full_volume})")
    # A handful of (h, w) planes is the intended working set.
    assert spy.max_elems <= 16 * 16 * 4


def test_no_allocation_spans_the_whole_z_axis(tmp_path, monkeypatch):
    """Stated as a shape check so the failure names the offending array."""
    spy = AllocSpy(monkeypatch)
    n_z = 96
    predict(tmp_path, n_z, h=16, w=16)
    offenders = [s for s in spy.shapes if s[0] >= n_z]
    assert not offenders, f"full-depth volumes allocated: {offenders}"


def test_working_set_is_flat_across_volume_sizes(tmp_path, monkeypatch):
    """40 slices and 160 slices must cost the same peak, not 4x."""
    small = AllocSpy(monkeypatch)
    predict(tmp_path, 40, h=16, w=16, seed=1)
    peak_small = small.max_elems

    big = AllocSpy(monkeypatch)
    predict(tmp_path, 160, h=16, w=16, seed=1)
    assert big.max_elems == peak_small


# --------------------------------------------------------------------- correctness
@pytest.mark.parametrize("n_z", [1, 5, 40, 97])
def test_every_slice_gets_exactly_one_mask(tmp_path, n_z):
    _, masks, _ = predict(tmp_path, n_z)
    assert masks.shape[0] == n_z


@pytest.mark.parametrize("n_z", [12, 40, 97])
@pytest.mark.parametrize("h,w", [(PATCH, PATCH), (24, 24), (20, 12)])
def test_streaming_reproduces_the_thresholded_volume(tmp_path, n_z, h, w):
    """A plane flushed before all its contributing slabs ran would come out wrong.

    Non-multiple sizes are included because the trailing patches are padded, and the
    padding must not be blended back into the output.
    """
    vol, masks, _ = predict(tmp_path, n_z, h=h, w=w)
    expected = ((vol > 127.5) * 255).astype(np.uint8)
    assert np.array_equal(masks, expected)


def test_plain_3d_overlap_averaging_still_reproduces_the_volume(tmp_path):
    """z_jitter=0 with overlap gives many votes per plane and heavy slab overlap,
    which is the case most likely to break early flushing."""
    vol, masks, _ = predict(tmp_path, 40, jitter=0, depth=D, overlap=D - 1)
    expected = ((vol > 127.5) * 255).astype(np.uint8)
    assert np.array_equal(masks, expected)


def test_planes_are_written_only_once(tmp_path):
    """Flushing twice would double-count, or overwrite a finished plane with a
    partially accumulated one."""
    written = []
    real_save = Image.Image.save

    def spy(self, fp, *a, **kw):
        written.append(str(fp))
        return real_save(self, fp, *a, **kw)

    n_z = 40
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    write_volume(in_dir, n_z)
    out_dir.mkdir()

    worker = PredictWorker({})
    Image.Image.save = spy
    try:
        worker._predict_folder_3d(Threshold(), in_dir, out_dir, PATCH, D, 0, 'xy',
                                  torch.device('cpu'), z_jitter=M)
    finally:
        Image.Image.save = real_save

    preds = [p for p in written if p.endswith('_pred.tif')]
    assert len(preds) == len(set(preds)) == n_z


def test_blank_slices_still_produce_a_mask(tmp_path):
    """All-zero patches are skipped, so those planes never enter the accumulator;
    they must still be emitted or the downstream voting step loses alignment."""
    n_z = 24
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    write_volume(in_dir, n_z)
    for z in (5, 6, 7):
        Image.fromarray(np.zeros((PATCH, PATCH), np.uint8)).save(in_dir / f"s{z:04d}.tif")
    out_dir.mkdir()

    worker = PredictWorker({})
    worker._predict_folder_3d(Threshold(), in_dir, out_dir, PATCH, D, 0, 'xy',
                              torch.device('cpu'), z_jitter=M)

    masks = read_masks(out_dir, n_z)
    assert masks.shape[0] == n_z
    assert masks[6].max() == 0


# ------------------------------------------------------------------------- plumbing
def test_trained_window_is_what_gets_written(tmp_path):
    """Guards the flush boundary against the trained window moving."""
    lo, hi = trained_z_window(D, M)
    assert (lo, hi) == (D // 2 - M // 2, D // 2 + M // 2)
    _, masks, _ = predict(tmp_path, 40)
    assert masks.shape[0] == 40


def test_stop_request_aborts_without_writing_the_rest(tmp_path):
    n_z = 60
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    write_volume(in_dir, n_z)
    out_dir.mkdir()

    worker = PredictWorker({})
    worker.should_stop = True
    worker._predict_folder_3d(Threshold(), in_dir, out_dir, PATCH, D, 0, 'xy',
                              torch.device('cpu'), z_jitter=M)
    assert not list(out_dir.glob("*_pred.tif"))
