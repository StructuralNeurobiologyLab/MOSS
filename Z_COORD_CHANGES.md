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

## How to revert
```bash
git checkout main
# or to undo individual commits:
git log --oneline  # find the commit
git revert <hash>
```
