"""Tests for 3D slab training from sparse 2D annotations.

The invariants here are the ones that fail silently if broken: a label plane that
drifts out of alignment with the image, an ignore value that leaks into the loss, or
an inference Z-stride that leaves unpredicted bands. None of those raise on their own
-- they just produce a quietly wrong model -- so they are pinned down here.
"""

import numpy as np
import pytest
import torch

from segmentation_suite.models.architectures import (
    get_3d_patch_depth, get_z_jitter, is_slab_architecture, validate_slab_geometry)
from segmentation_suite.models.architectures.unet_3d_slab import (
    UNet3DSlab, Z_DOWNSAMPLE, XY_DOWNSAMPLE)
from segmentation_suite.models.slab_inference import (
    slab_read_indices, slab_z_starts, slab_z_weights, trained_z_window)
from segmentation_suite.workers.train_worker import (
    IGNORE_LABEL, MaskedBCEDiceLoss, SlabPatchDataset)

ARCH = 'unet_3d_slab'
D, M = 16, 8


# --------------------------------------------------------------- registry / geometry
def test_architecture_is_registered_and_visible():
    from segmentation_suite.models.architectures import get_available_architectures
    assert is_slab_architecture(ARCH)
    # A hidden architecture cannot be selected in the UI, which makes it useless.
    assert ARCH in get_available_architectures()


def test_geometry_defaults_are_valid():
    validate_slab_geometry(ARCH)
    assert get_3d_patch_depth(ARCH) % Z_DOWNSAMPLE == 0
    assert get_z_jitter(ARCH) <= get_3d_patch_depth(ARCH) - 2


@pytest.mark.parametrize("depth,jitter", [(8, 8), (16, 20), (14, 8), (6, 4),
                                          (16, 5), (16, 7), (16, -2)])
def test_invalid_geometry_is_rejected(depth, jitter):
    """M >= D-1 puts the label plane outside the slab; D % 4 breaks the skip
    connections; an odd M makes the trained range asymmetric so it no longer
    coincides with the symmetric inference window.

    D=8/M=8 is the specific configuration that crashes the Ais reference.
    """
    with pytest.raises(ValueError):
        validate_slab_geometry(ARCH, depth, jitter)


def test_geometry_validation_refuses_to_guess_the_z_divisor():
    """Architectures are exec'd from file and never enter sys.modules, so Z_DOWNSAMPLE
    must be captured at load time. Silently assuming a divisor would pass a bad depth
    through to fail inside forward()."""
    with pytest.raises(ValueError, match="Z_DOWNSAMPLE"):
        validate_slab_geometry('unet_deep_dice_v2', 16, 8)


def test_slab_checkpoint_name_is_not_mistaken_for_the_plain_3d_arch():
    """'unet_3d' is a substring of 'unet_3d_slab', so a naive substring test picks the
    wrong architecture and builds a model the weights do not fit."""
    from segmentation_suite.models.architectures import get_checkpoint_filename
    name = get_checkpoint_filename(ARCH).lower()
    assert 'unet_3d' in name          # the trap
    # mirrors the ordering in segmentation_combined_page._detect_architecture
    detected = 'unet_3d_slab' if 'unet_3d_slab' in name else (
        'unet_3d' if 'unet_3d' in name else 'unet')
    assert detected == ARCH


def test_unknown_loss_name_raises_instead_of_falling_back_to_bce():
    """A mistyped PREFERRED_LOSS would feed the ignore sentinel into plain BCE."""
    from segmentation_suite.workers.train_worker import get_loss_function
    for name in ('bce', 'dice', 'bce_dice', 'masked_bce_dice'):
        assert get_loss_function(name) is not None
    with pytest.raises(ValueError):
        get_loss_function('masked_bce_dic')


# ------------------------------------------------------------------------ the model
def test_slab_in_slab_out():
    model = UNet3DSlab(n_channels=1, n_classes=1).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, D, 64, 64))
    assert tuple(out.shape) == (1, 1, D, 64, 64)


@pytest.mark.parametrize("depth", [6, 10, 14])
def test_bad_depth_raises_rather_than_silently_reshaping(depth):
    model = UNet3DSlab().eval()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, depth, 32, 32))


def test_bad_xy_raises():
    model = UNet3DSlab().eval()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, D, 30, 32))
    assert XY_DOWNSAMPLE == 16


# ------------------------------------------------------------------------- the loss
def _one_plane_label(z_lab, depth=D, hw=8):
    mask = torch.zeros(hw, hw)
    mask[2:6, 2:6] = 1.0
    label = torch.full((1, 1, depth, hw, hw), IGNORE_LABEL)
    label[0, 0, z_lab] = mask
    return label, mask


def test_ignore_planes_contribute_no_loss_and_no_gradient():
    torch.manual_seed(0)
    label, _ = _one_plane_label(8)
    logits = torch.randn(1, 1, D, 8, 8)
    crit = MaskedBCEDiceLoss()

    base = crit(logits, label)
    perturbed = logits.clone()
    perturbed[:, :, [z for z in range(D) if z != 8]] += 100.0
    assert torch.allclose(base, crit(perturbed, label))

    g = logits.clone().requires_grad_(True)
    crit(g, label).backward()
    planes = torch.nonzero(g.grad[0, 0].abs().sum(dim=(1, 2)) > 0).flatten().tolist()
    assert planes == [8]


def test_slab_loss_equals_the_dense_2d_loss_on_the_labelled_plane():
    torch.manual_seed(1)
    label, mask = _one_plane_label(8)
    logits = torch.randn(1, 1, D, 8, 8)
    crit = MaskedBCEDiceLoss()
    assert torch.allclose(crit(logits, label), crit(logits[:, :, 8], mask[None, None]))


def test_all_ignore_sample_is_finite_and_inert():
    """A slab with nothing labelled must not produce NaN via a zero denominator."""
    crit = MaskedBCEDiceLoss()
    logits = torch.randn(1, 1, D, 8, 8, requires_grad=True)
    loss = crit(logits, torch.full((1, 1, D, 8, 8), IGNORE_LABEL))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) == 0.0


def test_ignore_sentinel_never_reaches_the_bce_target():
    """If 2.0 leaked in as a target, a perfect prediction would score badly."""
    label, mask = _one_plane_label(8)
    perfect = torch.full((1, 1, D, 8, 8), -20.0)
    perfect[0, 0, 8] = torch.where(mask.bool(), 20.0, -20.0)
    assert float(MaskedBCEDiceLoss()(perfect, label)) < 1e-2


def test_masked_loss_still_works_on_fully_labelled_2d_targets():
    torch.manual_seed(2)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    loss = MaskedBCEDiceLoss()(torch.randn(2, 1, 8, 8), target)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------- the dataset
@pytest.fixture
def slab_project(tmp_path):
    """Slab pairs whose every plane is filled with its own index, so the plane the
    label points at can be identified from the pixel values alone."""
    import tifffile
    idir, mdir = tmp_path / 'img', tmp_path / 'msk'
    idir.mkdir(); mdir.mkdir()
    stored = D + M
    for k in range(3):
        slab = np.stack([np.full((32, 32), i, np.float32) for i in range(stored)])
        mask = np.zeros((32, 32), np.uint8)
        mask[8:20, 8:20] = 255
        tifffile.imwrite(idir / f'slice{k:04d}.tif', slab)
        tifffile.imwrite(mdir / f'slice{k:04d}.tif', mask)
    return idir, mdir


def test_dataset_supervises_exactly_one_plane_per_sample(slab_project):
    idir, mdir = slab_project
    ds = SlabPatchDataset(str(idir), str(mdir), patch_depth=D, patch_size=16, z_jitter=M)
    for _ in range(200):
        _, label = ds[0]
        valid = (label[0] != IGNORE_LABEL)
        planes = torch.nonzero(valid.reshape(D, -1).sum(1)).flatten().tolist()
        assert len(planes) == 1


def test_jitter_covers_exactly_the_trained_window(slab_project):
    idir, mdir = slab_project
    ds = SlabPatchDataset(str(idir), str(mdir), patch_depth=D, patch_size=16, z_jitter=M)
    seen = set()
    for _ in range(1500):
        _, label = ds[0]
        valid = (label[0] != IGNORE_LABEL)
        seen.add(int(torch.nonzero(valid.reshape(D, -1).sum(1)).flatten()[0]))
    lo, hi = trained_z_window(D, M)
    assert sorted(seen) == list(range(lo, hi + 1))


def test_label_plane_holds_the_annotated_image_plane(slab_project):
    """The decisive alignment check: the label must sit on the stored centre plane."""
    idir, mdir = slab_project
    ds = SlabPatchDataset(str(idir), str(mdir), patch_depth=D, patch_size=16, z_jitter=M)
    raw = np.stack([np.full((4, 4), i, np.float32) for i in range(D + M)])
    for _ in range(500):
        crop, z_lab = ds._jitter_crop(raw)
        assert crop[z_lab, 0, 0] == (D + M) // 2


def test_validation_uses_a_centred_deterministic_crop(slab_project):
    idir, mdir = slab_project
    ds = SlabPatchDataset(str(idir), str(mdir), patch_depth=D, patch_size=16,
                          z_jitter=M, train=False)
    raw = np.stack([np.full((4, 4), i, np.float32) for i in range(D + M)])
    assert {ds._jitter_crop(raw)[1] for _ in range(50)} == {D // 2}


def test_wrong_depth_slabs_are_skipped_not_silently_cropped(tmp_path):
    import tifffile
    idir, mdir = tmp_path / 'img', tmp_path / 'msk'
    idir.mkdir(); mdir.mkdir()
    tifffile.imwrite(idir / 'good.tif', np.zeros((D + M, 32, 32), np.float32))
    tifffile.imwrite(mdir / 'good.tif', np.zeros((32, 32), np.uint8))
    tifffile.imwrite(idir / 'wrong_depth.tif', np.zeros((D + M + 4, 32, 32), np.float32))
    tifffile.imwrite(mdir / 'wrong_depth.tif', np.zeros((32, 32), np.uint8))
    tifffile.imwrite(idir / 'mask_is_a_slab.tif', np.zeros((D + M, 32, 32), np.float32))
    tifffile.imwrite(mdir / 'mask_is_a_slab.tif', np.zeros((D + M, 32, 32), np.uint8))

    ds = SlabPatchDataset(str(idir), str(mdir), patch_depth=D, patch_size=16, z_jitter=M)
    assert ds.volumes == ['good.tif']


def test_overhanging_crop_pads_the_mask_with_ignore_not_a_reflection(tmp_path):
    """Reflect-padding the mask would assert real labels over image content that was
    itself fabricated by the pad."""
    import tifffile
    idir, mdir = tmp_path / 'img', tmp_path / 'msk'
    idir.mkdir(); mdir.mkdir()
    # stored slab XY (24) smaller than the requested patch (32) forces padding
    tifffile.imwrite(idir / 'a.tif', np.zeros((D + M, 24, 24), np.float32))
    mask = np.full((24, 24), 255, np.uint8)
    tifffile.imwrite(mdir / 'a.tif', mask)

    ds = SlabPatchDataset(str(idir), str(mdir), patch_depth=D, patch_size=32, z_jitter=M)
    for _ in range(20):
        _, label = ds[0]
        plane = label[0][torch.nonzero(
            (label[0] != IGNORE_LABEL).reshape(D, -1).sum(1)).flatten()[0]]
        # XY augmentation relocates the pad ring, so assert its area rather than its
        # position: exactly the 32x32 - 24x24 padded voxels must be ignore, and the
        # real region (mask was all-foreground) must be labelled 1.
        assert int((plane == IGNORE_LABEL).sum()) == 32 * 32 - 24 * 24
        assert int((plane == 1.0).sum()) == 24 * 24


def test_dataset_rejects_impossible_geometry(tmp_path):
    (tmp_path / 'i').mkdir(); (tmp_path / 'm').mkdir()
    with pytest.raises(ValueError):
        SlabPatchDataset(str(tmp_path / 'i'), str(tmp_path / 'm'),
                         patch_depth=8, z_jitter=8)


# --------------------------------------------------------------------- Z inference
def test_only_the_trained_planes_get_weight():
    w = slab_z_weights(D, M)
    lo, hi = trained_z_window(D, M)
    assert w[lo:hi + 1].min() == 1.0
    assert w[:lo].sum() == 0.0 and w[hi + 1:].sum() == 0.0
    assert int(w.sum()) == M + 1


@pytest.mark.parametrize("depth", [12, 16, 24, 32, 48])
@pytest.mark.parametrize("n_z", [1, 3, 9, 17, 64, 200])
def test_z_tiling_covers_every_slice(depth, n_z):
    """Ais steps by depth//2, which gaps for depth >= 24; the stride here is the
    trained width, so coverage holds at every depth."""
    w = slab_z_weights(depth, M)
    covered = np.zeros(n_z)
    for start in slab_z_starts(n_z, depth, M):
        for j in range(depth):
            z = start + j
            if 0 <= z < n_z:
                covered[z] += w[j]
    assert (covered > 0).all()


@pytest.mark.parametrize("depth", [16, 24, 32])
def test_identity_model_reconstructs_the_volume_exactly(depth):
    """Tiling plus blending must be lossless; anything else is a blending bug."""
    rng = np.random.default_rng(0)
    n_z = 70
    vol = rng.random((n_z, 3, 3)).astype(np.float32)
    w = slab_z_weights(depth, M)
    acc = np.zeros_like(vol, dtype=np.float64)
    wsum = np.zeros(n_z)
    for start in slab_z_starts(n_z, depth, M):
        pred = vol[slab_read_indices(start, depth, n_z)]      # identity model
        for j in range(depth):
            z = start + j
            if 0 <= z < n_z and w[j] > 0:
                acc[z] += pred[j] * w[j]
                wsum[z] += w[j]
    assert (wsum > 0).all()
    assert np.abs(acc / wsum[:, None, None] - vol).max() < 1e-6


def test_untrained_margins_cannot_leak_into_the_result():
    """A model returning garbage on the untrained margins must not change the output."""
    rng = np.random.default_rng(1)
    n_z, depth = 70, D
    vol = rng.random((n_z, 3, 3)).astype(np.float32)
    lo, hi = trained_z_window(depth, M)
    w = slab_z_weights(depth, M)
    acc = np.zeros_like(vol, dtype=np.float64)
    wsum = np.zeros(n_z)
    for start in slab_z_starts(n_z, depth, M):
        pred = vol[slab_read_indices(start, depth, n_z)].copy()
        pred[:lo] = 1e4
        pred[hi + 1:] = -1e4
        for j in range(depth):
            z = start + j
            if 0 <= z < n_z and w[j] > 0:
                acc[z] += pred[j] * w[j]
                wsum[z] += w[j]
    assert np.abs(acc / wsum[:, None, None] - vol).max() < 1e-6


def test_plain_3d_still_honours_overlap_in_z():
    """A slab model steps by its trained width, but a plain 3D model has no untrained
    margin and must keep averaging over `overlap`. Stepping by the full depth instead
    silently changed every existing unet_3d prediction from ~32 votes per plane to 1."""
    depth, overlap, n_z = 32, 64, 200
    stride = max(1, depth - overlap)          # what predict_worker passes for z_jitter=0
    w = slab_z_weights(depth, 0)
    assert int(w.sum()) == depth              # no untrained margin when there is no jitter

    votes = np.zeros(n_z)
    for start in slab_z_starts(n_z, depth, 0, stride=stride):
        for j in range(depth):
            z = start + j
            if 0 <= z < n_z:
                votes[z] += w[j]
    assert (votes > 0).all()
    assert votes.mean() > 10, f"lost Z overlap averaging: {votes.mean()} votes/plane"


def test_stride_larger_than_the_trained_width_is_clamped_not_gapped():
    n_z, depth = 200, 16
    for bad in (50, 999):
        covered = np.zeros(n_z)
        w = slab_z_weights(depth, M)
        for start in slab_z_starts(n_z, depth, M, stride=bad):
            for j in range(depth):
                z = start + j
                if 0 <= z < n_z:
                    covered[z] += w[j]
        assert (covered > 0).all(), f"stride {bad} left gaps instead of being clamped"


def test_hub_can_now_train_slab_and_offers_it():
    """Slab was excluded while crop transfer carried only 2D/2.5D variants. It now sends
    a "_slab" variant, so the hub must offer it rather than silently drop it."""
    from segmentation_suite.models.architectures import (
        filter_hub_trainable, get_available_architectures, is_hub_trainable)
    assert is_hub_trainable(ARCH) is True
    visible = get_available_architectures()
    assert set(filter_hub_trainable(visible)) == set(visible)


def test_hub_window_picker_offers_slab():
    from segmentation_suite.hub.hub_window import build_prediction_arch_map
    assert ARCH in build_prediction_arch_map()


def test_hub_routes_each_variant_to_its_own_folder():
    """The host must place a crop by its declared variant, and refuse one it cannot
    place. Inferring from channel count alone sent unknown counts to the plain-2D
    folder, where a 24-plane slab would train as a 24-channel 2D image."""
    from segmentation_suite.hub.hub_server import HubServer
    place = HubServer._variant_for_payload

    class H:
        _KNOWN_VARIANTS = HubServer._KNOWN_VARIANTS
        _VARIANT_BY_NC = HubServer._VARIANT_BY_NC
        _variant_for_nc = HubServer._variant_for_nc

    h = H()
    assert place(h, {"variant": "", "n_channels": 1}) == ("", "tif")
    assert place(h, {"variant": "_25d", "n_channels": 3}) == ("_25d", "tif")
    assert place(h, {"variant": "_dwarf25d", "n_channels": 11}) == ("_dwarf25d", "tif")
    assert place(h, {"variant": "_slab", "n_channels": 24}) == ("_slab", "tif")
    # an unknown declared variant is refused, not guessed
    assert place(h, {"variant": "_bogus", "n_channels": 3}) is None
    # older clients (no declared variant) still infer from channel count
    assert place(h, {"n_channels": 11}) == ("_dwarf25d", "tif")
    # ...but an unrecognized count is now refused instead of silently becoming 2D
    assert place(h, {"n_channels": 24}) is None


def test_session_variant_maps_slab_to_its_own_suffix():
    """If a slab session fell through to ("", "tif"), pooling, per-user counting and
    discard would all operate on the 2D folder instead."""
    from segmentation_suite.hub.hub_server import HubServer

    class H:
        prediction_model = ARCH
        architecture = None
    assert HubServer._session_variant(H()) == ("_slab", "tif")

    class H2:
        prediction_model = 'unet_deep_dice_dwarf25d_v2'
        architecture = None
    assert HubServer._session_variant(H2()) == ("_dwarf25d", "tif")


def test_is_slab_implies_is_3d():
    """Volumetric setup is guarded by IS_3D and then read under IS_SLAB, so the two
    disagreeing would raise NameError deep inside training."""
    from segmentation_suite.models.architectures import is_3d_architecture
    assert is_3d_architecture(ARCH)


def test_z_jitter_survives_a_checkpoint_round_trip(tmp_path):
    """Without this the predictor falls back to weighting every plane, margins included."""
    from segmentation_suite.models.slab_inference import read_slab_geometry
    ckpt = tmp_path / 'c.pth'
    torch.save({'model_state': {}, 'is_slab': True, 'patch_depth': 24, 'z_jitter': 4}, ckpt)
    assert read_slab_geometry(str(ckpt), ARCH) == (24, 4)


def test_geometry_falls_back_to_architecture_defaults_for_old_checkpoints(tmp_path):
    from segmentation_suite.models.slab_inference import read_slab_geometry
    ckpt = tmp_path / 'c.pth'
    torch.save({'model_state': {}}, ckpt)
    assert read_slab_geometry(str(ckpt), ARCH) == (get_3d_patch_depth(ARCH),
                                                   get_z_jitter(ARCH))
