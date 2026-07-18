# MOSS TODO

## Bugs
- [x] Undo (Ctrl+Z) only undoes accepted prediction suggestions, not paint strokes. **FIXED**: OptimizedCanvas.draw_at() was missing undo region snapshot before modifying mask.
- [x] On a fresh z-slice, spacebar to accept doesn't work until after a click action. **FIXED**: Canvas didn't have keyboard focus after slice navigation — added setFocus() in load_current_slice().
- [x] Server didn't handle TRAINING_DATA messages — client crops were rejected as "Unknown message type" and binary PNG data was parsed as gzip weights. **FIXED**: Added `_handle_training_data` to server that relays JSON header + 2 binary frames to host client.
- [x] Architecture dropdown showed wrong value after multi-user lock — subproject switch could overwrite locked architecture. **FIXED**: `_resolve_working_dirs` now skips architecture restore when `_architecture_locked` is True.

## Needs Testing
- [~] Reslicing and segmentation page — the **MOSS 2D** multi-view reslice → predict → vote path was exercised end-to-end this session and works (after the low-VRAM patch-size and Windows fixes). LSD **3D** "Run Segmentation" still blocked (see Bugs). Multi-user paths still untested.
- [ ] Multi-user session — full end-to-end test (LAN host, LAN join, crop sending, disconnect)
- [ ] Undo (Ctrl+Z) now works for paint strokes — needs manual verification
- [ ] Spacebar accept on fresh z-slice — needs manual verification
- [ ] Home page pipeline steps updated (4 steps instead of 6) — verify UI looks correct
- [ ] Multi-user: verify subproject panel is locked during session
- [ ] Multi-user: verify joinees cannot start training
- [ ] Multi-user: verify crops from joinee arrive at host's training directory

## Minor / Cosmetic
- [x] Multi-user now has unified dialog with LAN (default) and Relay options. Session dialog includes architecture + subproject selection for host.
- [x] Multi-user disconnect throws `ConnectionClosedError` and `TimeoutError` traceback — **FIXED** in client.py
- [x] `relay_config.txt` added to `.gitignore` to protect credentials

## UI Cleanup
- [x] Home page pipeline steps updated to match actual workflow: Data Import → Ground Truth → Segmentation → Export
- [x] Removed stale "Proofreading" card and pipeline step from home page

## Bugs
- [ ] LSD "Run Segmentation" (3D watershed) fails with a missing-module error. **Root cause found**: `em_pipeline/pipeline.py`'s `SegmentationPipeline` imports `em_pipeline.strategies` and `em_pipeline.data.volume`, and **neither module exists in the repo** — the 3D pipeline is incomplete. "Test LSD" and the LSD **2D** preview path work (the LSD 2D preview checkpoint path was fixed separately; it does not use `SegmentationPipeline`).
- [x] Review Crops button didn't delete dwarf 2.5D crops (`train_images_dwarf25d`/`train_masks_dwarf25d`). **FIXED**: Added dwarf25d directory awareness and move logic to `TrainingDataReviewer._discard_current()`.

## Work in Progress
- [ ] Proofreading page — skipped in the wizard (`training_wizard.py` auto-advances past it), but the backend code is **not** deleted: `em_pipeline/proofreading/` still contains `moss_bridge.py`, `neuroglancer_state.py`, `viewer.py`. Needs wizard-side reimplementation to re-expose it.
- [ ] Export page — a minimal functional page now exists (`wizard_pages/finish_page.py`, wired as `export_page` with `set_config` + open-output-folder). Re-verify whether the "placeholder / stale references" concern still applies before treating this as a rewrite.

## Investigation Results
- [x] `refiner_images`, `refiner_masks_before`, `refiner_masks_after` — **VESTIGIAL**. Removed from project_config.py. Deleted unused refiner.py model file. (Stale README references — project-structure dirs and the "Refiner Mode" feature bullet — also removed.)
- [x] `sam2_features` — **ACTIVE**. Stores pre-extracted SAM2 embeddings for training crops. SAM2 weights download to `<MOSS_root>/sam2_models/` from HuggingFace (wanglab/MedSAM2) on first use.

## Technical Debt
- [ ] `interactive_training_page.py` is a god object (~3900 lines) — consider splitting into TrainingController, PredictionController, CropSaver, SubprojectManager
- [ ] Checkpoint/architecture state tracked redundantly in root config, subproject config, page instance vars, and predict worker — single source of truth needed
- [x] `predict_worker.set_training_active(False)` scattered across 5+ call sites — **DONE**: consolidated the training-stop sequence into one idempotent `InteractiveTrainingPage.stop_training()`; `start_training` / `switch_subproject` / `cleanup` now route through it (also fixed clean-quit, project-switch stop, and the "Stop = No training data" bug). Remaining `set_training_active` calls are distinct contexts (init, dynamic prediction, start=True, on-finish), not duplicated stop logic.
