# Changelog

Notable fixes and changes to MOSS. Newest first.

## 2026-09-26 — Fix: 2.5D training silently trained on garbage when crop channels mismatched

**Symptom.** Training a 2.5D architecture (e.g. `unet_deep_dice_25d_v2`) finished
without errors, but the model predicted completely blank images.

**Cause.** The training loader (`NucleiPatchDataset` and `NucleiPatchDatasetSAM2` in
`segmentation_suite/workers/train_worker.py`) "adapted" crops whose channel count did
not match the architecture by slicing the *last* axis. Crops are saved channels-first
`(C, H, W)`, so a 1-channel crop `(1, 512, 512)` became a `(1, 512, 3)` sliver. The
network trained only on those slivers and learned to predict background everywhere.

This loader code dates from the initial commit, but it only became reachable in
v2.0.0 (2026-03-24), when the number of slices saved per 2.5D crop started following
the selected architecture. Before that, 2.5D crops were always saved as 3 slices.
Capturing crops while a 2D architecture is selected (1 slice), and then training a
2.5D one, now produces 1-channel crops in the 2.5D folders and triggers the bug.

**Fix.**
- Crops whose channel count does not match the architecture are **skipped with a
  warning** naming the file, instead of being cropped to a sliver.
- Skipped crops are removed from the sampling list, so they never turn into empty
  tiles (this also applies to the SAM2 dataset).
- If no usable crops remain, training **stops with a clear error** ("No usable
  training crops … re-capture them") instead of training a blank model.

**What to do if you were affected.** Re-capture the crops with the correct
architecture selected, or rebuild the 2.5D stacks. Delete the checkpoint trained on
the bad crops so training starts fresh.
