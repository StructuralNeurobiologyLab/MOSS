# MOSS Code Review & Critique

# July 2026 Follow-up

*July 2026 — after a cross-platform install and bug-fix pass (commits `8de0ab6`…`8c006d3`).*

## March concerns — where they stand now

- **Worker coordination scattered** — *addressed.* The stop sequence (worker stop/wait, predict-worker GPU hand-off, UI reset, `training_stopped` signal) is consolidated into one idempotent `InteractiveTrainingPage.stop_training()`; `start_training`, `switch_subproject`, and `cleanup` all route through it. This closed a connected cluster of bugs: the app not quitting cleanly (`MainWindow` had no `closeEvent`, so training/predict QThreads kept the process alive), training continuing after a project switch, and "Stop Training" wrongly reporting "No training data" after a switch (the data check ran before the running-worker check).
- **No tests** — *partially addressed.* Added `tests/test_project_config.py` (22 tests) over the pure config/path utilities flagged in March: default-merge, path resolution, subproject layout, `project_exists`. The stateful worker/subproject logic is still untested.
- **`interactive_training_page.py` god object** — *unchanged.* Still ~3900 lines. Deliberately not refactored this pass (high risk, no rewrite requested).
- **State tracked in too many places** — *unchanged.* The single-source-of-truth redesign was deferred. `stop_training()` removes one class of sync bug, but the redundant checkpoint/architecture state across root config, subproject config, page instance vars, and the predict worker remains.

## New findings this pass

### Cross-platform install was broken on Windows
- PyQt6 wheels ≥6.8 fail to import on Windows (`DLL load failed while importing QtCore: The specified procedure could not be found`). Capped `PyQt6>=6.4.0,<6.8` across `environment.yml` / `requirements.txt` / `pyproject.toml`.
- Recent `conda` mis-detects the Windows build number (`__win=0`), so conda-forge `qtbase` refuses to install — documented as a troubleshooting caveat.
- Added a fully pinned multi-platform `conda-lock.yml` (linux-64 / osx-64 / osx-arm64 / win-64) for reproducible installs; the `<6.8` cap is also what makes the lock resolvable on Linux/macOS.

### GPU multi-view prediction unusable on small cards
- The Combined Segmentation multi-view predictor hardcoded a 1024-px patch. On <8 GB GPUs a batch overflows VRAM and the driver pages GPU memory to system RAM — 100% GPU utilisation but ~seconds per slice (slower than CPU). Now caps patch size to 512 on <8 GB GPUs; larger GPUs and CPU are unchanged.

### Portability / documentation
- LSD 2D preview used a hardcoded absolute developer path; now resolves the bundled checkpoint via the architecture registry (`get_pretrained_checkpoint`).
- Reconciled the citation and `pyproject` Homepage to the active repository, removed stale README content (vestigial `refiner_*` dirs, the removed "Refiner Mode" feature), and added a worked example (`examples/fafb_lsd_demo/`) that downloads a small public FAFB block, builds a MOSS project preconfigured with the bundled pretrained LSD membrane model, and renders a preview — a full pipeline a new user can verify in minutes.

## Still open (tracked in TODO.md)

- **LSD 3D "Run Segmentation" is genuinely broken, not just mis-wired.** `em_pipeline/pipeline.py`'s `SegmentationPipeline` imports `em_pipeline.strategies` and `em_pipeline.data.volume`, and **neither module exists in the repo**. The 3D watershed pipeline is incomplete. (The LSD 2D preview path works and does not use `SegmentationPipeline`.)
- **Two reported runtime bugs still need a live repro** before a confident fix: loaded Zarr volumes becoming unavailable after project load, and trained-model predictions not appearing in the viewer (first suspects: the `show_predictions` toggle and whether the predict worker receives a checkpoint on project load — the latter is a symptom of the redundant-state issue above).
- The structural items (god object, single source of truth) remain the main long-term risk, exactly as the March review predicted — the state-sync bugs fixed this pass are the kind it warned would keep appearing.

---

# March 2026 Review

*March 2026*

## What Works Well

- **The interactive loop** (paint → Tab → train → see predictions live) is a genuinely good workflow for segmentation. Most tools make you leave the annotation environment to train. MOSS keeps you in context.
- **Architecture plugin system** is clean — drop a `.py` file in `models/architectures/` and it's automatically discovered. Metadata-driven (`N_CONTEXT_SLICES`, `TRAINING_V2`, `PREFERRED_LOSS`) so the training pipeline adapts without special-casing.
- **Zarr + TIFF transparency** — supporting both seamlessly is practical for real microscopy data at different scales.
- **Epoch multiplier approach** — fixed-length epochs via random augmented crops gives consistent training feedback regardless of dataset size. Unconventional but fits the interactive use case.

## Architectural Concerns

### `interactive_training_page.py` is a god object (~3900+ lines)

It handles: UI layout, mask I/O, crop saving, training orchestration, prediction management, 3D mode, multi-user sync, subproject switching, zarr rendering, and project config persistence.

Consequences already observed:
- Two separate crop-saving codepaths (`capture_crop_for_training` and `_save_edit_crops`) with duplicated inline 2.5D logic. When dwarf 2.5D was added, only one path was updated.
- Dead code (`_save_edit_to_disk`, `_save_edit_crops`) survived unnoticed because the file is too large to audit at a glance.
- `_save_25d_crop` exists as a proper helper but only the dead codepath used it. The live Tab path has its own inline version.

Potential splits:
- **CropSaver** — all crop extraction, normalization, and saving (2D, 2.5D, dwarf)
- **TrainingController** — start/stop/reset training, architecture state, checkpoint management
- **PredictionController** — predict worker lifecycle, checkpoint loading, overlay management
- **SubprojectManager** — switching, state save/restore, directory resolution

### State tracked in too many places

Architecture and checkpoint state is stored redundantly in:
1. Root `project.json`
2. Subproject `project.json`
3. `interactive_training_page` instance variables (`current_architecture`, `prediction_architecture`)
4. Predict worker's internal state
5. Train worker's config dict

These get out of sync. The "predictions don't work until you start training" bug was caused by the predict worker not receiving the checkpoint path on project load, because only the page's instance variables were set.

A single source of truth (e.g., a `ProjectState` object that workers and UI both reference) would eliminate this class of bug.

### Worker coordination is scattered

`predict_worker.set_training_active(False)` appears in 5+ locations. Missing one instance causes subtle GPU contention bugs. The stop-training sequence (stop flag → wait → update UI → notify predict worker → re-enable controls → emit signal) is duplicated across 4 call sites with slight variations.

A state machine or centralized `TrainingSession` that owns both workers and emits state transitions would consolidate this.

### No tests

For something with this much state management (subproject switching, config merging, path resolution, cache invalidation), even basic unit tests on the non-UI logic would catch issues like:
- Mask cache not cleared on subproject switch
- Config merge dropping new default keys
- Path resolution for nested subproject layouts

The config and path utilities in `project_config.py` are pure functions — easy to test with no UI dependency.

## Minor Issues

- **CUDA thread termination** — `QThread.terminate()` on a thread holding a CUDA context causes a fatal crash. This was present in 4 locations (now fixed with graceful timeout).
- **Broad exception swallowing** — `_save_25d_crop` and several other methods catch `Exception` and print, hiding real failures. The dwarf crops silently not saving would have been caught with proper error propagation.
- **"foo: Not a TIFF" warnings on startup** — some file in the scan path has a `.tif` extension but isn't actually TIFF. Harmless but noisy.

## Summary

MOSS solves a real problem well. The interactive training loop is its killer feature. The main structural risk is that `interactive_training_page.py` is accumulating responsibilities faster than they're being factored out, which leads to duplicated logic and state-sync bugs that are hard to spot. As features grow (subprojects, dwarf 2.5D, multi-user, 3D), this will compound.
