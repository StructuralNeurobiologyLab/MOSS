# FAFB + pretrained LSD demo

Verify a complete MOSS pipeline — real EM data in, membrane predictions out — in
well under 30 minutes, using public data and the bundled pretrained model. No
annotation or training required.

The script downloads a small block of the **Full Adult Fly Brain (FAFB v14)**
electron-microscopy volume (Zheng et al. 2018), converts it to the pyramidal
OME-Zarr layout MOSS uses, writes a `project.json` preconfigured for the bundled
**LSD boundary/membrane** model, and renders a preview so you can confirm it works.

## 1. Install

From the repository root (see the top-level README for details):

```bash
conda env create -f environment.yml
conda activate moss
pip install -e .
```

Then the one extra dependency for the download step:

```bash
pip install -r examples/fafb_lsd_demo/requirements.txt
```

## 2. Build the demo project

```bash
cd examples/fafb_lsd_demo

# ~64 MB block — fastest end-to-end check (recommended first run):
python build_demo.py --preset quick --out ./demo_project

# ~0.5 GB block, matching the figures below:
# python build_demo.py --preset demo --out ./demo_project
```

This will:
1. Download a centered FAFB block (`fafb_v14_clahe`, 8×8×40 nm) → a TIFF stack.
2. Build `demo_project/raw_data.zarr` (4-level pyramid) + per-slice `raw_images/`.
3. Write `demo_project/project.json` wired to the `lsd_boundary_2d` model and copy
   the bundled checkpoint in as `checkpoint_lsd_boundary_2d.pth`.
4. Render `demo_project/lsd_preview_raw_vs_membrane.png` (raw EM | predicted membranes).

The preview uses your GPU automatically if PyTorch sees one, otherwise CPU.

## 3. Open in MOSS

```bash
moss
```

- **Load Project** → select the `demo_project` folder.
- **Interactive view:** scroll slices with `A`/`D`, press `S` to toggle live LSD
  membrane predictions (they use the bundled pretrained model — no training needed).
- **Combined Segmentation** → **MOSS (Recommended)**: click **Browse** next to
  *Input* and select the project's `raw_images` folder (the multi-view predictor
  reads TIFF folders, not Zarr), check **XY**, and **Run**. Results land in
  `predictions/` and `heatmap/`.

## Notes

- The block is **not** committed to the repo — it is downloaded on demand from a
  public Google Cloud bucket. Re-running with the same `--out` reuses the download.
- `--preset quick` (512×512×256) is enough to validate the pipeline; `--preset demo`
  (1024×1024×512) gives a more representative field of view.
- Low-VRAM GPUs (<8 GB): the multi-view predictor automatically uses a 512-pixel
  patch to stay within memory.
