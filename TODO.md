# MOSS TODO

## Bugs
- [ ] LSD "Run Segmentation" (3D watershed) fails with a missing-module error. **Root cause found**: `em_pipeline/pipeline.py`'s `SegmentationPipeline` imports `em_pipeline.strategies` and `em_pipeline.data.volume`, and **neither module exists in the repo** — the 3D pipeline is incomplete. "Test LSD" and the LSD **2D** preview path work (the LSD 2D preview checkpoint path was fixed separately; it does not use `SegmentationPipeline`).

## Needs Testing
- [ ] Reslicing and segmentation — the **MOSS 2D** multi-view reslice → predict → vote path was verified this session; still need to verify the LSD **3D** "Run Segmentation" (blocked, see Bugs) and the multi-user paths.
- [ ] Multi-user session — full end-to-end test (LAN host, LAN join, crop sending, disconnect)
- [ ] Undo (Ctrl+Z) for paint strokes — manual verification
- [ ] Spacebar accept on fresh z-slice — manual verification
- [ ] Home page pipeline steps (4 steps) — verify UI looks correct
- [ ] Multi-user: verify subproject panel is locked during session
- [ ] Multi-user: verify joinees cannot start training
- [ ] Multi-user: verify crops from joinee arrive at host's training directory

## Work in Progress
- [ ] Proofreading page — skipped in the wizard (`training_wizard.py` auto-advances past it), but the backend code is **not** deleted: `em_pipeline/proofreading/` still contains `moss_bridge.py`, `neuroglancer_state.py`, `viewer.py`. Needs wizard-side reimplementation to re-expose it.
- [ ] Export page — a minimal functional page now exists (`wizard_pages/finish_page.py`, wired as `export_page` with `set_config` + open-output-folder). Re-verify whether the "placeholder / stale references" concern still applies before treating this as a rewrite.

## Technical Debt
- [ ] `interactive_training_page.py` is a god object (~3900 lines) — consider splitting into TrainingController, PredictionController, CropSaver, SubprojectManager
- [ ] Checkpoint/architecture state tracked redundantly in root config, subproject config, page instance vars, and predict worker — single source of truth needed
