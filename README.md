# MOSS - Microscopy Oriented Segmentation with Supervision

A PyQt6-based interactive segmentation tool with U-Net training, prediction, and mask editing capabilities. MOSS provides an intuitive interface for training deep learning models on microscopy images through direct annotation and real-time feedback.

MOSS-lite is the virst version of MOSS and my personal copy. All future work and updates will be reflected in https://github.com/StructuralNeurobiologyLab/MOSS . 

## Features

- **Interactive Training**: Paint masks on images and train U-Net models in real-time
- **Multiple Architectures**: UNet, UNetDeep, UNetDeepDice, 2.5D models
- **Live Predictions**: See model predictions as you work
- **Refiner Mode**: Train a refinement model that learns from your edits
- **Batch Processing**: Reslice, predict, and vote across multiple views
- **Mask Editing Tools**: Brush, eraser, fill, and component-based editing
- **Multi-User Training**: Collaborate with others anywhere (requires relay server)

## Installation

### Option 1: pip install (recommended)

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

#### Linux/Windows CPU only

```bash
python -m venv moss-env
source moss-env/bin/activate  # On Windows: moss-env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Option 2: conda environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate moss

# Install the package
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

# 2. Replace pip PyQt6 with conda-forge version
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
conda install -c conda-forge pyqt6

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
| Tab | Capture crop for training (refiner mode) |
| Ctrl + S | Save project |
| Ctrl + Z | Undo |
| +/- | Zoom in/out |
| Shift + Scroll | Adjust brush size |

## Project Structure

When you create a project, the following folders are created:

```
project_folder/
├── project_config.json    # Project settings
├── masks/                 # Saved masks (mask_00000.tif, etc.)
├── train_images/          # Training image crops
├── train_masks/           # Training mask crops
├── checkpoint_*.pth       # Model checkpoints (per architecture)
├── refiner_images/        # Refiner training data
├── refiner_masks_before/  # Mask state before edits
├── refiner_masks_after/   # Mask state after edits
└── refiner_checkpoint.pth # Refiner model checkpoint
```

## Requirements

- Python 3.9+
- PyQt6 6.4+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- macOS MPS (optional, for Apple Silicon acceleration)

## Troubleshooting

### PyQt6 issues on Mac

If you get errors about PyQt6, try:

```bash
pip install PyQt6 --force-reinstall
```

### PyTorch MPS issues on Mac

For Apple Silicon Macs, ensure you have a recent PyTorch version:

```bash
pip install --upgrade torch torchvision
```

### CUDA out of memory

Reduce batch size in the training settings or use a smaller tile size.

### Images not loading

Supported formats: TIFF (.tif, .tiff), PNG, JPEG. For best results, use single-channel grayscale TIFF images.

### PyQt6 on Linux clusters (`undefined symbol: FT_Get_Colorline_Stops`)

Pip-installed PyQt6 bundles binaries compiled against newer system libraries than most
clusters have. Replace it with the conda-forge build:

```bash
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
conda install -c conda-forge pyqt6
```

## Author

**Nelson Medina** - Creator and primary developer
GitHub: [@nelsmedina](https://github.com/nelsmedina)

## Citation

If you use MOSS in your research, please cite:

```
Medina, N. (2025). MOSS: Microscopy Oriented Segmentation with Supervision [Computer software].
https://github.com/nelsmedina/MOSS
```

## License

MIT License









