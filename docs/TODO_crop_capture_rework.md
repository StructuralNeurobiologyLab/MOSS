# TODO (major): rework crop capture and storage around a manifest

Status: not started. Raised 2026-08-04 while adding the 3D slab model.

## The problem in one line

Capture writes several redundant pixel copies of the same annotated region and throws
away the one thing that would let us regenerate them: where the crop came from.

## What is stored today

`crop_id = f"slice{idx:04d}_cap{timestamp}"` (interactive_training_page.py:2892).

- `slice4411` — the Z index. The only coordinate kept. `train_worker.py:169` parses it
  with `slice(\d+)` for the z-coord architectures.
- `cap47228` — `int(time.time()*1000) % 100000`. A uniqueness token. It looks like a
  coordinate and is not one.
- The crop box XY (`crop_x`, `crop_y`) exists in memory during capture and is never
  written. No sidecar files exist; the folders hold only `.tif`.

So two crops from the same slice are indistinguishable in origin, and no crop can be
traced back to its location in the volume.

Every capture writes the same region again per architecture variant:

| folder | content | 175 crops @ 512 |
|---|---|---|
| `train_masks_512` | the 2D annotation | **1 MB** |
| `train_images_512` | (H,W) | 34 MB |
| `train_images_512_25d` | (3,H,W) | 292 MB |
| `train_images_512_dwarf25d` | (11,H,W) | 380 MB |
| `train_images_512_slab` | (24,H,W) | ~840 MB extrapolated (53 MB at 11 crops) |

Roughly 1.5 GB of pixels carrying 1 MB of information. Every image byte is a
deterministic function of (source volume, z, y, x, size) — the mask is the only
irreplaceable artifact.

## Proposal

Treat image crops as a **cache**, and the mask plus a manifest row as the **source of
truth**.

Store per capture:
- the 2D mask (as now)
- a manifest row: volume identity, `z`, `y`, `x`, `w`, `h`, crop size, capture time,
  annotator, and the normalization actually applied

Then any image variant is regenerated on demand: 2D, 2.5D at any spacing, dwarf, a slab
at any `D`/`M`. Including variants that do not exist yet.

## What this unlocks

1. **Recapture more context without re-annotating.** Want `D=32` slabs instead of 16?
   Regenerate. Today it means re-visiting every annotation by hand.
2. **Survive lost image crops.** Delete or corrupt the pixel cache and rebuild it from
   the manifest. Only mask loss is fatal.
3. **Storage.** Keep masks plus manifest (~1 MB) and materialise only the variant being
   trained. ~1.5 GB down to tens of MB.
4. **Provenance.** Answer "which region is this crop, and who annotated it" — currently
   unanswerable.
5. **De-duplication.** Detect overlapping or repeat crops of the same region, which is
   invisible today.

## What has to be got right

- **Volume identity, not just a path.** Store something that detects drift (path plus
  shape plus a hash of a few sampled slices). Regenerating against a changed or
  re-sliced volume must fail loudly, never silently produce mismatched pixels.
- **Normalization is currently baked in at capture.** 2D and 2.5D normalize per channel;
  the slab path normalizes across the whole slab. To regenerate byte-identically, the
  manifest must record which rule was used — or, better, cache raw and normalize at
  train time so the rule stops being a stored property.
- **The Hub needs pixels.** A remote annotator's volume is not on the host, so the
  manifest alone is not enough there. Either keep sending pixels for hub sessions, or
  build on the raw-volume sharing already on the `multiuser-hub` branch so the host can
  regenerate too.
- **Zarr vs TIFF-folder sources** resolve coordinates differently; the manifest has to
  say which.
- **No backfill for existing crops.** Their XY is already gone. Migration means: keep old
  crops as-is (opaque, still trainable), and manifest only new ones. Do not pretend old
  crops have coordinates.

## Sketch

`train_manifest.jsonl` in the subproject, one row per capture:

```json
{"id": "slice4411_cap47228", "volume": {"kind": "zarr", "path": "...", "shape": [5000, 8192, 8192], "sig": "ab12..."},
 "z": 4411, "y": 3072, "x": 5120, "w": 512, "h": 512, "norm": "per_slab_minmax",
 "created": "2026-08-04T12:11:03Z", "by": "nmedina"}
```

JSONL because captures are append-only and a partial write must not corrupt earlier rows.

## Cheap first step, if the full rework is not worth it yet

Put the coordinates in the filename:
`slice{idx:04d}_y{crop_y:05d}_x{crop_x:05d}_cap{ts}.tif`

Every current consumer keeps working — they regex `slice(\d+)` and ignore the rest — and
new captures become traceable. Roughly 15 minutes, covers the 2D/2.5D/dwarf/slab siblings
at once since they all inherit `crop_id`. It buys provenance and de-duplication but not
regeneration or the storage win.

## Open question

Is regeneration-on-demand actually wanted, or just provenance? If only provenance, do the
filename change and stop. The manifest only pays for itself if crops will really be
rebuilt from it — and that requires committing to keeping source volumes addressable and
unchanged, which is a bigger promise than it looks.
