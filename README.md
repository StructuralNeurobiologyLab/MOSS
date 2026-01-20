# MOSS - Microscopy Oriented Segmentation with Supervision

A PyQt6-based interactive segmentation tool with U-Net training, prediction, and mask editing capabilities. MOSS provides an intuitive interface for training deep learning models on microscopy images through direct annotation and real-time feedback.

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

```bash
# Create a virtual environment (recommended)
python -m venv moss-env
source moss-env/bin/activate  # On Windows: moss-env\Scripts\activate

# Install PyTorch first (select appropriate version for your platform)
# For Mac (CPU/MPS):
pip install torch torchvision

# For Linux/Windows with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install MOSS
pip install segmentation_suite-1.0.0-py3-none-any.whl

# Or install from source:
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









