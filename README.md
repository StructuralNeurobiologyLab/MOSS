# MOSS - Microscopy Oriented Segmentation with Supervision

A PyQt6-based interactive segmentation tool with U-Net training, prediction, and mask editing capabilities. MOSS provides an intuitive interface for training deep learning models on microscopy images through direct annotation and real-time feedback.

This is the actively maintained MOSS repository: https://github.com/StructuralNeurobiologyLab/MOSS
It supersedes the original *MOSS-lite* prototype; all ongoing work lands here.

## Features

- **Interactive Training**: Paint masks on images and train U-Net models in real-time
- **Multiple Architectures**: Standard UNet plus a family of "Deep Dice v2" 2D and 2.5D models (see [Architectures](#architectures))
- **Large Volumes (OME-Zarr / Pyramids)**: Load multiscale `.zarr` stores with automatic resolution selection by zoom level, or generate pyramids from TIFF stacks (see [Loading Data](#loading-data))
- **Selectable Crop Size**: Train at 128, 256, or 512 px tiles
- **Live Predictions**: See model predictions as you work
- **Subprojects**: Organize multiple labels/targets within a single project
- **Reslice & Voting**: Generate orthogonal/diagonal views and combine predictions across them for consensus segmentation
- **Mask Editing Tools**: Brush, eraser, fill, and component-based editing
- **Multi-User Training**: Collaborate with others anywhere (requires relay server)

## Architectures

MOSS ships several model architectures, selectable from the training UI:

| Architecture | Notes |
|---|---|
| **UNet (Standard)** | Baseline 2D U-Net. |
| **UNet Deep Dice v2 (Stable)** | Recommended default. Combined BCE+Dice loss, gradient clipping, robust resume. |
| **UNet Deep Dice 2.5D v2 (Stable)** | 2.5D variant taking 3 z-slices (z−3, z, z+3) as input channels. |
| **UNet Deep 2.5D v2 (11-slice)** | Wider z-context: 11 channels spanning z−10…z+10 (every 2nd slice). |
| **UNet Deep 2.5D v2 (11-slice + Z-coord)** | As above, plus a z-coordinate channel encoding the slice's normalized depth — useful when appearance varies systematically with depth. |

The "v2" models are recommended for new projects. (A few additional experimental architectures exist in the codebase but are hidden from the UI.)

## Loading Data

MOSS reads both image stacks and chunked volumes:

- **TIFF stacks** (`.tif`, `.tiff`), plus PNG and JPEG. Single-channel grayscale is recommended.
- **OME-Zarr / multiscale pyramids** (`.zarr` directories, zarr v2 and v3). MOSS reads the `multiscales` metadata and automatically renders from the appropriate pyramid level for the current zoom, so very large volumes stay responsive.

### Generating pyramids from TIFF

For large datasets, convert a TIFF stack to a pyramidal OME-Zarr from the home screen via **Generate Pyramids**. This writes a multi-level (4-level) OME-Zarr store; subsequent loads use the downsampled levels when zoomed out. Volumes without a pyramid structure still load via a zarr fallback path.

## Installation

### Recommended: let an AI coding agent set it up

The quickest and most reliable way to get MOSS running — especially across
platforms — is to point an AI coding agent (e.g. Claude Code) at this repository
and ask it to install and launch MOSS. The agent reads `environment.yml` /
`conda-lock.yml`, detects your OS and GPU, and works around the platform-specific
quirks that otherwise trip up a manual install: Windows PyQt6 wheel failures,
conda virtual-package (`__win`) detection, CUDA-vs-CPU PyTorch selection, and so
on. To install by hand instead, use one of the options below.

### Option 1: conda environment (recommended for manual setup)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate moss

# Install the package
pip install -e .
```

For a fully pinned, reproducible environment, install from the committed lock file
instead of solving `environment.yml`:

```bash
conda-lock install --name moss conda-lock.yml
conda activate moss
pip install -e .
```

`conda-lock.yml` pins exact versions for `linux-64`, `osx-64`, `osx-arm64`, and
`win-64`. Regenerate it after changing `environment.yml`:
`conda-lock lock -f environment.yml -p linux-64 -p osx-64 -p osx-arm64 -p win-64`.

### Option 2: pip install

#### Mac (Apple Silicon or Intel)

```bash
python -m venv moss-env
source moss-env/bin/activate
pip install torch torchvision
pip install -e .
```

#### Linux with NVIDIA GPU (CUDA)

```bash
python -m venv moss-env
source moss-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

#### Linux/Windows CPU only

```bash
python -m venv moss-env
source moss-env/bin/activate  # On Windows: moss-env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Option 3: Manual installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 4: HPC / Cluster (conda)

On shared HPC clusters, pip-installed PyQt6 often fails because its pre-compiled binaries
expect newer system libraries than the cluster provides (e.g. `libharfbuzz`, `FreeType`).
The fix is to replace the pip PyQt6 with the conda-forge build.

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate moss

# 2. Replace pip PyQt6 with conda-forge version (kept below 6.8 to match the pins)
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
conda install -c conda-forge "pyqt6<6.8"

# 3. Install the package
pip install -e .

# 4. Launch (requires interactive session with X11 forwarding)
srun --partition=cpu --pty --x11 --cpus-per-task=4 bash
conda activate moss
moss
```

For heavy training on cluster data, use **LAN network mode**: host a session on the cluster
and connect from your laptop via "Advanced (LAN)" in the multi-user dialog. This avoids
X11 lag — you annotate locally and the cluster trains on received crops.

## Usage

After installation, run:

```bash
moss
```

Or run as a Python module:

```bash
python -m segmentation_suite
```

### Try it on real data (no annotation needed)

To verify a complete pipeline end to end in a few minutes, use the worked example
in [`examples/fafb_lsd_demo/`](examples/fafb_lsd_demo/). It downloads a small public
FAFB electron-microscopy block, builds a MOSS project preconfigured with the bundled
pretrained LSD membrane model, and renders a preview:

```bash
pip install -r examples/fafb_lsd_demo/requirements.txt
python examples/fafb_lsd_demo/build_demo.py --preset quick --out ./demo_project
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| A / Left Arrow | Previous slice |
| D / Right Arrow | Next slice |
| B | Brush tool |
| E | Eraser tool |
| H | Hand (pan) tool |
| F + Click | Fill tool |
| S | Toggle predictions |
| Space | Accept hovered prediction component |
| Shift + Space | Accept ALL predictions (replace mask) |
| Tab | Capture crop for training |
| Ctrl + S | Save project |
| Ctrl + Z | Undo |
| +/- | Zoom in/out |
| [ / ] | Decrease / increase brush size |
| Shift + Scroll | Adjust brush size |

## Project Structure

When you create a project, MOSS lays out roughly the following:

```
project_folder/
├── project.json              # Project settings
├── masks/                    # Saved masks (mask_00000.tif, etc.)
├── train_images/             # Training image crops (256 px default)
├── train_masks/              # Training mask crops (256 px default)
├── train_images_128/ 512/    # Crop-size-specific folders (when 128/512 selected)
├── train_masks_128/ 512/     #   (mirrors train_images_*)
├── train_images_25d/ ...     # 2.5D / 11-slice variants, per crop size
└── checkpoint*.pth           # Model checkpoints (per architecture)
```

Notes:

- The project config file is named **`project.json`**.
- Training data is organized **per crop size**: the default 256 px uses `train_images`/`train_masks`, while 128 and 512 px get suffixed folders (`train_images_128`, `train_images_512`, …). 2.5D architectures use parallel `_25d` / `_dwarf25d` folders, and 3D uses `train_images_3d` / `train_masks_3d`.
- Projects that use **subprojects** nest the per-label data under `subprojects/<name>/`, each with its own `project.json`, `masks/`, and training folders.

## Requirements

- Python 3.9+
- PyQt6 6.4–6.7 (`< 6.8`; newer wheels have a Windows load bug and aren't lockable cross-platform)
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- macOS MPS (optional, for Apple Silicon acceleration)

## Troubleshooting

### PyQt6 issues on Mac

If you get errors about PyQt6, try:

```bash
pip install PyQt6 --force-reinstall
```

### PyQt6 on Windows (`DLL load failed while importing QtCore: The specified procedure could not be found`)

PyQt6 wheels **6.8 and newer fail to load on some Windows systems** with this error
(it also appears as a failure importing `QtWidgets`). The dependency files cap PyQt6
below 6.8 on all platforms (also required for a reproducible cross-platform
`conda-lock`), so a fresh install avoids it. If you hit it in an existing
environment, pin the binding **and** the Qt6 runtime together:

```bash
pip install "PyQt6==6.7.1" "PyQt6-Qt6==6.7.1"
```

The conda-forge `pyqt6` build is an alternative, but note that older `conda`
versions mis-detect the Windows build number (`__win=0`) and refuse to install
recent `qtbase`; upgrading `conda` or using the pip pin above avoids that.

### PyTorch MPS issues on Mac

For Apple Silicon Macs, ensure you have a recent PyTorch version:

```bash
pip install --upgrade torch torchvision
```

### CUDA out of memory

Reduce batch size in the training settings or use a smaller tile size.

### Images not loading

Supported formats: TIFF (.tif, .tiff), PNG, JPEG, and OME-Zarr (`.zarr`) stores. For best results, use single-channel grayscale images. See [Loading Data](#loading-data) for large-volume / pyramid handling.

### PyQt6 on Linux clusters (`undefined symbol: FT_Get_Colorline_Stops`)

Pip-installed PyQt6 bundles binaries compiled against newer system libraries than most
clusters have. Replace it with the conda-forge build (kept below 6.8 to match the pins):

```bash
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
conda install -c conda-forge "pyqt6<6.8"
```

## Author

**Nelson Medina** - Creator and primary developer
GitHub: [@nelsmedina](https://github.com/nelsmedina)

## Citation

If you use MOSS in your research, please cite:

```
Medina, N. (2025). MOSS: Microscopy Oriented Segmentation with Supervision [Computer software].
https://github.com/StructuralNeurobiologyLab/MOSS
```

## License

MIT License









