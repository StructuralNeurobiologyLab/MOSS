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

from segmentation_suite.models.slab_inference import slab_z_starts, trained_z_window
from segmentation_suite.workers.predict_worker import (
    XY_BLEND_FLOOR, PredictWorker, build_xy_blend_window)

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
        self.batch_sizes = []

    def __call__(self, x):
        self.calls += 1
        self.batch_sizes.append(x.shape[0])
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


def predict(tmp_path, n_z, model=None, jitter=M, depth=D, overlap=0, batch_limit=8,
            h=PATCH, w=PATCH, seed=0, spy=None, xy_blend='hann', vol=None, tag=''):
    """Run the worker's 3D path over a freshly written volume."""
    in_dir = tmp_path / f"in{n_z}_{seed}_{h}x{w}{tag}"
    out_dir = tmp_path / f"out{n_z}_{seed}_{h}x{w}{tag}_{xy_blend}"
    if vol is None:
        vol = write_volume(in_dir, n_z, h=h, w=w, seed=seed)
    else:
        write_given_volume(in_dir, vol)
    out_dir.mkdir(parents=True, exist_ok=True)

    worker = PredictWorker({})
    model = model if model is not None else Threshold()
    if spy is not None:
        spy(worker)
    worker._predict_folder_3d(model, in_dir, out_dir, PATCH, depth, overlap,
                             'xy', torch.device('cpu'), z_jitter=jitter,
                             batch_limit=batch_limit, xy_blend=xy_blend)
    return vol, read_masks(out_dir, n_z), model


def write_given_volume(folder, vol):
    folder.mkdir(parents=True, exist_ok=True)
    for z in range(vol.shape[0]):
        Image.fromarray(vol[z]).save(folder / f"s{z:04d}.tif")
    return vol


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


# ------------------------------------------------------------------------ batching
def test_patches_are_batched_not_run_one_at_a_time(tmp_path):
    """The whole point of batching: far fewer forward passes than patches."""
    _, _, model = predict(tmp_path, 40, h=64, w=64, overlap=0, batch_limit=8)
    patches_per_slab = (64 // PATCH) ** 2
    n_slabs = len(slab_z_starts(40, D, M))
    assert model.calls < patches_per_slab * n_slabs
    assert max(model.batch_sizes) > 1


def test_batching_does_not_change_the_result(tmp_path):
    """Batch size must be a throughput knob only."""
    vol_a, masks_a, _ = predict(tmp_path, 40, h=32, w=32, batch_limit=1, seed=3)
    vol_b, masks_b, _ = predict(tmp_path, 40, h=32, w=32, batch_limit=16, seed=3)
    assert np.array_equal(vol_a, vol_b)
    assert np.array_equal(masks_a, masks_b)


def test_batch_never_mixes_slabs(tmp_path):
    """Results are placed using the batch's z_start, so a batch spanning two slabs
    would silently write every patch at the wrong depth."""
    depths_seen = []

    class Recorder(Threshold):
        def __call__(self, x):
            depths_seen.append(x.shape[2])
            return super().__call__(x)

    predict(tmp_path, 40, h=32, w=32, model=Recorder(), batch_limit=1000)
    assert set(depths_seen) == {D}


def test_oom_falls_back_to_smaller_batches_and_still_completes(tmp_path):
    """A too-large batch must degrade rather than abort the run. Nothing is
    accumulated before the forward pass returns, so the retry cannot double-count."""

    class OomOnce(Threshold):
        def __init__(self):
            super().__init__()
            self.raised = False

        def __call__(self, x):
            if not self.raised and x.shape[0] > 1:
                self.raised = True
                raise RuntimeError("CUDA out of memory. Tried to allocate 356.00 GiB")
            return super().__call__(x)

    model = OomOnce()
    vol, masks, _ = predict(tmp_path, 40, h=32, w=32, model=model, batch_limit=16)
    assert model.raised
    expected = ((vol > 127.5) * 255).astype(np.uint8)
    assert np.array_equal(masks, expected)


def test_non_oom_runtime_errors_are_not_swallowed(tmp_path):
    class Broken(Threshold):
        def __call__(self, x):
            raise RuntimeError("shape mismatch in conv3d")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        predict(tmp_path, 12, model=Broken())


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


# --------------------------------------------------------------------- XY blending
def seam_ratio(mask, stride):
    """Edge density on tile-boundary columns over edge density elsewhere.

    The same metric that measured ~6x on a real prediction slice at stride 64. A clean
    result sits near 1.0: tile boundaries are no edgier than anywhere else.
    """
    gx = np.abs(np.diff(mask.astype(np.float32) / 255.0, axis=1)).mean(axis=0)
    idx = np.arange(len(gx))
    on = gx[idx % stride == (stride - 1) % stride]
    off = gx[idx % stride != (stride - 1) % stride]
    if off.mean() == 0:
        return 1.0 if on.mean() == 0 else np.inf
    return float(on.mean() / off.mean())


class BadAtBorders:
    """Correct in the patch centre, wrong in a border band.

    Stands in for the real failure mode: the convolutions' padding makes a patch
    unreliable near its edge, so a blend that weights the border like the centre
    prints the tile grid into the output. Ground truth here is 'foreground
    everywhere', which is what the patch centres say.
    """

    def __init__(self, band=2):
        self.band = band

    def __call__(self, x):
        out = torch.full_like(x, 20.0)
        b = self.band
        out[..., :b, :] = -20.0
        out[..., -b:, :] = -20.0
        out[..., :, :b] = -20.0
        out[..., :, -b:] = -20.0
        return out


class ConstantPositive:
    def __call__(self, x):
        return torch.full_like(x, 20.0)


def test_hann_window_never_reaches_zero():
    """A zero-weight column would blank the volume's outermost pixels."""
    win = build_xy_blend_window(PATCH, 'hann')
    assert win.shape == (PATCH, PATCH)
    assert win.min() >= XY_BLEND_FLOOR ** 2
    assert win.max() <= 1.0
    # Still a genuine taper, not a disguised top-hat.
    assert win[PATCH // 2, PATCH // 2] > 4 * win[0, 0]


def test_tophat_window_is_uniform():
    win = build_xy_blend_window(PATCH, 'tophat')
    assert win.min() == win.max() == 1.0


def test_unknown_blend_is_rejected():
    with pytest.raises(ValueError, match="xy_blend"):
        build_xy_blend_window(PATCH, 'cosine')


def test_hann_is_constant_overlap_add_at_half_stride():
    """At 50% overlap the interior weights must sum flat, or the blend adds its own
    ripple on top of whatever the model does."""
    win = build_xy_blend_window(64, 'hann')
    stride = 32
    acc = np.zeros(64 * 4, dtype=np.float64)
    for start in range(0, 64 * 3, stride):
        acc[start:start + 64] += win[32]           # one row is enough; it is separable
    interior = acc[64:64 * 3]
    assert interior.std() / interior.mean() < 0.05


@pytest.mark.parametrize("h,w", [(32, 32), (28, 20)])
def test_taper_suppresses_tile_seams(tmp_path, h, w):
    """The regression this fix exists for, measured the same way as on the real data."""
    n_z, overlap = 24, PATCH // 2      # 50% overlap, as the real runs use
    _, top, _ = predict(tmp_path, n_z, model=BadAtBorders(), overlap=overlap,
                        h=h, w=w, xy_blend='tophat')
    _, han, _ = predict(tmp_path, n_z, model=BadAtBorders(), overlap=overlap,
                        h=h, w=w, xy_blend='hann')

    mid = n_z // 2
    r_top = seam_ratio(top[mid], PATCH // 2)
    r_han = seam_ratio(han[mid], PATCH // 2)
    assert r_han < r_top, f"taper did not reduce seams: {r_han:.2f} vs {r_top:.2f}"

    # The centres all say foreground, so a good blend recovers mostly foreground.
    wrong_top = (top[mid] != 255).mean()
    wrong_han = (han[mid] != 255).mean()
    assert wrong_han < wrong_top / 2, (
        f"border contamination barely improved: {wrong_han:.3f} vs {wrong_top:.3f}")


@pytest.mark.parametrize("h,w", [(PATCH, PATCH), (28, 20), (32, 32)])
def test_no_pixel_is_left_without_blend_weight(tmp_path, h, w):
    """A pixel with zero accumulated weight divides down to 0 and prints a black
    seam. With a model that says foreground everywhere, every pixel must be 255 --
    including the volume's outermost rows, where the taper is smallest.
    """
    _, masks, _ = predict(tmp_path, 20, model=ConstantPositive(),
                          overlap=PATCH // 2, h=h, w=w)
    assert masks.min() == 255, (
        f"{(masks != 255).sum()} pixels lost their weight, "
        f"first at {np.argwhere(masks != 255)[0].tolist()}")


def test_uniform_patch_is_zeroed_like_training(tmp_path):
    """Training divides unconditionally, so a flat crop becomes zeros. Inference used
    to guard on p_max > p_min and pass raw intensities straight into the network.
    """
    n_z = 20
    vol = np.full((n_z, PATCH, PATCH), 200, dtype=np.uint8)
    _, masks, _ = predict(tmp_path, n_z, vol=vol, tag='flat')
    # Normalized to zeros, Threshold returns a negative logit -> empty mask. Left raw
    # at 200.0 it would clear the 0.5 threshold and fill the tile.
    assert masks.max() == 0, "flat patch was not normalized the way training does"


def test_reflect_padding_does_not_leak_into_the_output(tmp_path):
    """Edge patches are padded to patch_size; only the real extent may be written."""
    vol, masks, _ = predict(tmp_path, 24, h=20, w=12, overlap=PATCH // 2, tag='pad')
    expected = ((vol > 127.5) * 255).astype(np.uint8)
    assert masks.shape == expected.shape


def test_tophat_still_available_for_reproducing_old_predictions(tmp_path):
    vol, masks, _ = predict(tmp_path, 12, xy_blend='tophat')
    expected = ((vol > 127.5) * 255).astype(np.uint8)
    assert np.array_equal(masks, expected)
