"""
Segmentation Suite - UNet Training Pipeline + Annotation Tool Integration
"""

__version__ = "1.0.0"

# Allow opening large images (high-res EM data) without PIL's decompression bomb check
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
