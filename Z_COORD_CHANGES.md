# Z-Coordinate Architecture Changes

## Goal
Add a new U-Net architecture variant that includes the z-slice position as an
additional input channel. This gives the model explicit spatial context so it can
learn systematic differences across the image stack (e.g., tissue changes, staining
gradients, beam effects that vary with depth).

## Branch
`feature/z-coord-architecture` (based on `main` at 4b1558a)

## Design
- Based on the dwarf 2.5D architecture (11-slice, spacing=2).
- Adds 1 extra channel: normalized z-position (z_index / max_z_index), making
  the total input 12 channels.
- Z-index is parsed from the training crop filename (`slice####_cap*.tif`).
- During prediction, z-index is the file's position in the sorted input list.

## Files Changed

### New files
| File | Purpose |
|------|---------|
| `segmentation_suite/models/architectures/unet_deep_dice_dwarf25d_zcoord.py` | New architecture (12-ch, same U-Net body as dwarf) |

### Modified files
| File | What changed |
|------|-------------|
| `segmentation_suite/models/architectures/__init__.py` | Store/expose `USES_Z_COORD` flag |
| `segmentation_suite/workers/train_worker.py` | Parse z from filenames, append z-channel in dataset |
| `segmentation_suite/workers/predict_worker.py` | Append z-channel during inference |
| `segmentation_suite/workers/viewport_predict_worker.py` | Append z-channel during live viewport prediction; fixed `_n_channels` inflation bug |
| `segmentation_suite/wizard_pages/interactive_training_page.py` | Pass `total_slices` to viewport predictor |
| `segmentation_suite/wizard_pages/segmentation_combined_page.py` | Add zcoord to architecture detection; add zarr support for reslice completeness check |
| `segmentation_suite/workers/reslice_worker.py` | Add zarr volume support for all reslice types (XZ, YZ, diagonal) |

## Bugs found and fixed
1. **viewport _n_channels inflation** — adding +1 to `_n_channels` for z-coord caused
   the caller to try loading 12 image slices instead of 11, breaking live predictions.
2. **architecture detection** — checkpoint name `..._dwarf25d_zcoord.pth` matched
   the generic `dwarf25d` pattern, loading the wrong 11-ch model class.
3. **zarr + reslice** — reslice worker only looked for .tif/.png files, failing
   immediately on zarr input with "No images found".

## How to revert
```bash
git checkout main
# or to undo individual commits:
git log --oneline  # find the commit
git revert <hash>
```
