#!/usr/bin/env python3
"""
ZarrImageSource - provides efficient access to Zarr volumes with pyramid levels.
Reads OME-Zarr multiscales metadata to discover pyramid structure.
"""

import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configure zarr for async concurrency (must be done before opening any zarr arrays)
try:
    import zarr
    import warnings
    # Suppress v2/v3 coexistence warnings (we handle this in _open_zarr_robust)
    warnings.filterwarnings('ignore', category=UserWarning, module='zarr')
    # Try zarr v3 config style first
    if hasattr(zarr, 'config'):
        zarr.config.set({'async.concurrency': 64})
    print("[ZarrImageSource] Zarr async concurrency enabled")
except Exception as e:
    print(f"[ZarrImageSource] Could not configure zarr async: {e}")


class ZarrImageSource:
    """
    Provides access to Zarr volumes with automatic pyramid level selection.

    Supports OME-Zarr format with multiscales metadata.
    """

    def __init__(self, zarr_path: Path):
        """
        Initialize Zarr image source.

        Args:
            zarr_path: Path to the Zarr directory (e.g., 'raw_data.zarr')
        """
        import zarr

        self.zarr_path = Path(zarr_path)
        self.zarr_group = self._open_zarr_robust(zarr_path)

        # Discover pyramid structure
        self.pyramid_paths = []  # List of dataset paths (e.g., ['0', 's1', 's2', 's3'])
        self.downsample_factors = []  # Downsample factor for each level
        self._missing_pyramids = False  # True if only full-res found (no downsampled levels)
        self._discover_pyramid_levels()

        # Store volume metadata
        self.num_slices = 0
        self.height = 0
        self.width = 0
        self._load_metadata()

        # Global intensity stats for consistent normalization
        # Use reasonable defaults immediately for instant startup
        self.global_min = 0
        self.global_max = 255
        self._stats_ready = False

        # Compute real stats in background thread (non-blocking)
        self._stats_thread = threading.Thread(target=self._compute_global_stats_async, daemon=True)
        self._stats_thread.start()

    @staticmethod
    def _open_zarr_robust(zarr_path):
        """Open a Zarr store, handling v2/v3 format conflicts.

        Some stores have v2 root metadata (.zgroup/.zattrs) but v3 arrays,
        or v3 root metadata (zarr.json) with empty attrs. This method tries
        multiple strategies to get a working group with both attrs and arrays.
        """
        import zarr
        zarr_path = str(zarr_path)
        zarr_major = int(zarr.__version__.split('.')[0])

        if zarr_major >= 3:
            # Zarr 3.x: try v3 first, fall back to v2 if arrays not found
            # Strategy 1: open normally (zarr 3 picks format automatically)
            group = zarr.open_group(zarr_path, mode='r')
            if list(group.keys()):
                return group

            # Strategy 2: try forcing v2 format
            try:
                group_v2 = zarr.open_group(zarr_path, mode='r', zarr_format=2)
                if list(group_v2.keys()):
                    return group_v2
            except Exception:
                pass

            # Strategy 3: try forcing v3 format
            try:
                group_v3 = zarr.open_group(zarr_path, mode='r', zarr_format=3)
                if list(group_v3.keys()):
                    # v3 group may have empty attrs if multiscales are in .zattrs (v2)
                    # Merge v2 attrs if available
                    if not group_v3.attrs.get('multiscales'):
                        import json
                        zattrs_path = Path(zarr_path) / '.zattrs'
                        if zattrs_path.exists():
                            v2_attrs = json.loads(zattrs_path.read_text())
                            # Store merged attrs for later use
                            group_v3._v2_attrs = v2_attrs
                    return group_v3
            except Exception:
                pass

            # Last resort: return whatever we got
            return group
        else:
            # Zarr 2.x: just open normally
            return zarr.open(zarr_path, mode='r')

    def _discover_pyramid_levels(self):
        """Discover pyramid levels from Zarr metadata."""
        # Try OME-Zarr multiscales first
        # Check both native attrs and merged v2 attrs (for v2/v3 hybrid stores)
        attrs = {}
        if hasattr(self.zarr_group, 'attrs'):
            attrs = dict(self.zarr_group.attrs)
        if 'multiscales' not in attrs and hasattr(self.zarr_group, '_v2_attrs'):
            attrs = self.zarr_group._v2_attrs

        if 'multiscales' in attrs:
            multiscales = attrs['multiscales'][0]  # First multiscale
            datasets = multiscales.get('datasets', [])

            for i, ds in enumerate(datasets):
                path = ds.get('path', str(i))

                # FIX: Check if path actually exists, otherwise try alternatives
                if path not in self.zarr_group:
                    # Try common naming patterns
                    alternatives = [f's{i}', str(i), f'level_{i}', '0' if i == 0 else None]
                    path_found = False
                    for alt_path in alternatives:
                        if alt_path and alt_path in self.zarr_group:
                            path = alt_path
                            path_found = True
                            break

                    if not path_found:
                        print(f"[ZarrImageSource] Warning: Path '{ds.get('path')}' not found, skipping")
                        continue

                self.pyramid_paths.append(path)

                # Extract downsample factor from coordinateTransformations
                transforms = ds.get('coordinateTransformations', [])
                scale_factor = 1
                for transform in transforms:
                    if transform.get('type') == 'scale':
                        scale = transform.get('scale', [1, 1, 1])
                        # For 3D volumes, scale is [z, y, x], we use y or x
                        scale_factor = int(scale[-1]) if len(scale) > 1 else 1
                        break

                # Fallback: assume power of 2 downsampling
                if scale_factor == 1 and i > 0:
                    scale_factor = 2 ** i

                self.downsample_factors.append(scale_factor)
        else:
            # Fallback: check for common pyramid level names
            common_paths = ['0', 's1', 's2', 's3', 's4']
            for i, path in enumerate(common_paths):
                if path in self.zarr_group:
                    self.pyramid_paths.append(path)
                    self.downsample_factors.append(2 ** i)

            # Last resort: use only level '0' if it exists
            if not self.pyramid_paths and '0' in self.zarr_group:
                self.pyramid_paths = ['0']
                self.downsample_factors = [1]

            # Final fallback: find any array in the group (e.g. 'volume')
            if not self.pyramid_paths:
                import zarr as zarr_module
                for key in self.zarr_group.keys():
                    item = self.zarr_group[key]
                    if isinstance(item, zarr_module.Array) and len(item.shape) >= 2:
                        self.pyramid_paths = [key]
                        self.downsample_factors = [1]
                        self._missing_pyramids = True
                        print(f"[ZarrImageSource] WARNING: No standard pyramid structure found.")
                        print(f"[ZarrImageSource] Using array '{key}' as single full-res level.")
                        print(f"[ZarrImageSource] Performance may be slow without downsampled levels.")
                        break

        if self.pyramid_paths:
            if not getattr(self, '_missing_pyramids', False):
                print(f"[ZarrImageSource] Discovered {len(self.pyramid_paths)} pyramid levels:")
                for path, factor in zip(self.pyramid_paths, self.downsample_factors):
                    print(f"  {path}: {factor}x downsample")

    def _load_metadata(self):
        """Load volume metadata from the first pyramid level."""
        if not self.pyramid_paths:
            raise ValueError("No pyramid levels found in Zarr store")

        level_0 = self.zarr_group[self.pyramid_paths[0]]

        if len(level_0.shape) == 3:
            # 3D volume: (slices, height, width)
            self.num_slices = level_0.shape[0]
            self.height = level_0.shape[1]
            self.width = level_0.shape[2]
        elif len(level_0.shape) == 2:
            # 2D image: (height, width)
            self.num_slices = 1
            self.height = level_0.shape[0]
            self.width = level_0.shape[1]
        else:
            raise ValueError(f"Unexpected Zarr shape: {level_0.shape}")

        print(f"[ZarrImageSource] Volume: {self.num_slices} slices, {self.height}x{self.width} pixels")

    def _compute_global_stats_async(self):
        """Compute global min/max in background thread for non-blocking startup.

        Uses reasonable defaults (0, 255) until this completes.
        """
        try:
            # Use most downsampled level for speed
            stats_level = self.pyramid_paths[-1]
            zarr_array = self.zarr_group[stats_level]

            # Sample a few slices to estimate
            if len(zarr_array.shape) == 3:
                # Use the actual number of slices in this downsampled level, not full-res count
                num_slices_this_level = zarr_array.shape[0]
                sample_indices = np.linspace(0, num_slices_this_level - 1, min(10, num_slices_this_level), dtype=int)

                global_min = float('inf')
                global_max = float('-inf')

                for idx in sample_indices:
                    slice_data = np.array(zarr_array[idx])
                    global_min = min(global_min, slice_data.min())
                    global_max = max(global_max, slice_data.max())

                self.global_min = global_min
                self.global_max = global_max
            else:
                # 2D array
                data = np.array(zarr_array)
                self.global_min = data.min()
                self.global_max = data.max()

            self._stats_ready = True
            print(f"[ZarrImageSource] Global intensity computed: min={self.global_min:.2f}, max={self.global_max:.2f}")
        except Exception as e:
            print(f"[ZarrImageSource] Error computing global stats: {e}")

    def select_pyramid_level(self, zoom_level: float):
        """
        Select the appropriate pyramid level based on zoom.

        VERY AGGRESSIVE: Prioritize performance over sharpness.
        At zoomed-out levels, we just need to see general regions,
        not fine detail. Use the most downsampled level possible.

        Args:
            zoom_level: Current zoom level (1.0 = 100%, 0.5 = 50%, etc.)

        Returns:
            tuple: (level_index, level_path, downsample_factor)
        """
        if not self.pyramid_paths:
            return 0, '0', 1

        # Only use full resolution when zoomed in past 200%
        if zoom_level > 2.0:
            return 0, self.pyramid_paths[0], self.downsample_factors[0]

        # VERY AGGRESSIVE: Allow up to 4x oversampling before switching to higher res
        # This means: at 100% zoom, 4x downsample is fine (just for navigation)
        #             at 50% zoom, 8x downsample is fine
        #             at 25% zoom, 16x+ would be fine (use max available)
        # Formula: target = 4.0 / zoom (allows 4x blurrier than "perfect")
        target_downsample = 4.0 / zoom_level

        # Find the highest downsample level that doesn't exceed target
        # (or just use the max available if target is very high)
        best_level = len(self.pyramid_paths) - 1  # Start with most downsampled
        for i in range(len(self.pyramid_paths)):
            if self.downsample_factors[i] <= target_downsample:
                best_level = i
            # Don't break - keep going to find the most aggressive level within target

        return best_level, self.pyramid_paths[best_level], self.downsample_factors[best_level]

    def get_slice(self, z_index: int, pyramid_level: int = 0):
        """
        Get a full slice at a given z-index and pyramid level.

        Args:
            z_index: Slice index in FULL-RESOLUTION space (0-based)
            pyramid_level: Pyramid level index (0 = full res, 1 = half, etc.)

        Returns:
            numpy.ndarray: 2D array of the slice
        """
        if pyramid_level >= len(self.pyramid_paths):
            pyramid_level = 0

        level_path = self.pyramid_paths[pyramid_level]
        downsample = self.downsample_factors[pyramid_level]
        zarr_array = self.zarr_group[level_path]

        if len(zarr_array.shape) == 3:
            # 3D volume - map z_index to downsampled space
            z_index_ds = z_index // downsample
            z_index_ds = min(z_index_ds, zarr_array.shape[0] - 1)
            slice_data = np.array(zarr_array[z_index_ds], dtype=np.float32)
        else:
            # 2D image
            slice_data = np.array(zarr_array, dtype=np.float32)

        return slice_data

    def get_tile_native(self, z_index: int, y1: int, y2: int, x1: int, x2: int, pyramid_level: int = 0):
        """
        Get a tile/region at native pyramid resolution (no upscaling).

        Coordinates are in FULL-RESOLUTION space. Returns tile at pyramid's native size.

        Args:
            z_index: Slice index in FULL-RESOLUTION space (0-based)
            y1, y2: Y range in full-resolution coordinates
            x1, x2: X range in full-resolution coordinates
            pyramid_level: Pyramid level index (0 = full res, 1 = half, etc.)

        Returns:
            tuple: (tile, downsample_factor) - tile at native resolution
        """
        if pyramid_level >= len(self.pyramid_paths):
            pyramid_level = 0

        level_path = self.pyramid_paths[pyramid_level]
        downsample = self.downsample_factors[pyramid_level]
        zarr_array = self.zarr_group[level_path]

        # Convert coordinates to downsampled space (including Z!)
        ds_z = z_index // downsample
        ds_y1 = y1 // downsample
        ds_y2 = (y2 + downsample - 1) // downsample  # Ceiling division
        ds_x1 = x1 // downsample
        ds_x2 = (x2 + downsample - 1) // downsample

        # Clamp to array bounds
        if len(zarr_array.shape) == 3:
            max_z = zarr_array.shape[0]
            max_y, max_x = zarr_array.shape[1], zarr_array.shape[2]
            ds_z = max(0, min(ds_z, max_z - 1))
        else:
            max_y, max_x = zarr_array.shape[0], zarr_array.shape[1]

        ds_y1 = max(0, min(ds_y1, max_y - 1))
        ds_y2 = max(ds_y1 + 1, min(ds_y2, max_y))  # Ensure at least 1 pixel
        ds_x1 = max(0, min(ds_x1, max_x - 1))
        ds_x2 = max(ds_x1 + 1, min(ds_x2, max_x))  # Ensure at least 1 pixel

        # Load the region at native resolution
        if len(zarr_array.shape) == 3:
            tile = np.array(zarr_array[ds_z, ds_y1:ds_y2, ds_x1:ds_x2], dtype=np.float32)
        else:
            tile = np.array(zarr_array[ds_y1:ds_y2, ds_x1:ds_x2], dtype=np.float32)

        return tile, downsample

    def get_tile(self, z_index: int, y1: int, y2: int, x1: int, x2: int, pyramid_level: int = 0):
        """
        Get a tile/region at a given z-index and pyramid level.

        Coordinates are in FULL-RESOLUTION space. This method automatically
        scales them to the appropriate pyramid level.

        Args:
            z_index: Slice index (0-based)
            y1, y2: Y range in full-resolution coordinates
            x1, x2: X range in full-resolution coordinates
            pyramid_level: Pyramid level index (0 = full res, 1 = half, etc.)

        Returns:
            numpy.ndarray: 2D array of the tile, scaled to match (y2-y1, x2-x1) size
        """
        if pyramid_level >= len(self.pyramid_paths):
            pyramid_level = 0

        level_path = self.pyramid_paths[pyramid_level]
        downsample = self.downsample_factors[pyramid_level]
        zarr_array = self.zarr_group[level_path]

        # Convert coordinates to downsampled space
        ds_y1 = y1 // downsample
        ds_y2 = (y2 + downsample - 1) // downsample  # Ceiling division
        ds_x1 = x1 // downsample
        ds_x2 = (x2 + downsample - 1) // downsample

        # Load the region
        if len(zarr_array.shape) == 3:
            # 3D volume - map z_index to downsampled space
            z_index_ds = z_index // downsample
            z_index_ds = min(z_index_ds, zarr_array.shape[0] - 1)
            tile = np.array(zarr_array[z_index_ds, ds_y1:ds_y2, ds_x1:ds_x2], dtype=np.float32)
        else:
            # 2D image
            tile = np.array(zarr_array[ds_y1:ds_y2, ds_x1:ds_x2], dtype=np.float32)

        # If downsampled, upscale to match expected size
        if downsample > 1:
            from scipy.ndimage import zoom as scipy_zoom

            target_h = y2 - y1
            target_w = x2 - x1

            zoom_factors = (target_h / tile.shape[0], target_w / tile.shape[1])
            tile = scipy_zoom(tile, zoom_factors, order=1)  # Bilinear interpolation

            # Ensure exact size (zoom might be slightly off)
            if tile.shape[0] > target_h:
                tile = tile[:target_h, :]
            if tile.shape[1] > target_w:
                tile = tile[:, :target_w]

            # Pad if needed (shouldn't happen, but be safe)
            if tile.shape[0] < target_h or tile.shape[1] < target_w:
                pad_h = target_h - tile.shape[0]
                pad_w = target_w - tile.shape[1]
                tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='edge')

        return tile
