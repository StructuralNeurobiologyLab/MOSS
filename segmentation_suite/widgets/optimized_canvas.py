#!/usr/bin/env python3
"""
Optimized PaintCanvas subclass for MOSS-full.

Based on original MOSS canvas with added zarr pyramid support.
Key features:
- Cached rendering for better performance
- Zarr pyramid support with zoom-based level selection
- Full-resolution mask painting
- Crop preview for training captures
"""

import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QEvent, QTimer
from PyQt6.QtGui import (
    QPainter, QImage, QPixmap, QPen, QColor, QPaintEvent,
    QBrush, QWheelEvent, QFont
)

from .paint_canvas import PaintCanvas
from ..dpi_scaling import scaled, scaled_font

# Thread pool for async tile loading (shared across canvases)
_tile_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="TileLoader")


class OptimizedCanvas(PaintCanvas):
    """PaintCanvas with cached rendering and zarr pyramid support."""

    # Signal emitted when Tab is pressed (capture crop)
    capture_requested = pyqtSignal()
    # Signal emitted when a suggestion component is accepted (spacebar)
    suggestion_accepted = pyqtSignal()
    # Signal emitted when brush size changes (for updating UI slider)
    brush_size_changed = pyqtSignal(int)
    # 3D mode signals
    save_3d_requested = pyqtSignal()
    cancel_3d_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable focus so we can receive key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Enable mouse tracking to get cursor position even without button press
        self.setMouseTracking(True)

        # Cached composited pixmap for the current viewport
        self._cached_pixmap = None
        self._cache_valid = False
        self._cached_render_key = None  # (slice_idx, x1, y1, x2, y2, level, mask_id, sugg_id)

        # === Zarr source state ===
        self._zarr_source = None
        self._zarr_slice_idx = 0
        self._image_height = None  # Full-res dimensions
        self._image_width = None

        # === Tile cache for zarr ===
        self._cached_tile = None
        self._cached_tile_params = None

        # Crop preview box (for training snapshot)
        self._crop_preview_bounds = None  # (x, y, w, h) in image coordinates
        self._crop_preview_alpha = 0.6
        self._crop_preview_visible = True
        self._crop_size = 256  # Default crop size

        # Track painting bounds for auto-positioning crop box
        self._paint_bounds_min = None  # (x, y)
        self._paint_bounds_max = None  # (x, y)

        # Draggable / lockable crop-box handle (top-right of the yellow box).
        # Unlocked: the box keeps auto-following your painting (old behavior), and you
        # can drag the circle to nudge it. A single click locks it (circle -> lock
        # icon): the box freezes in place, can't be dragged, and persists across
        # captures until you click again to unlock.
        self._crop_locked = False
        # Fixed on-screen size — does NOT scale with the zoom level (only DPI).
        self._crop_handle_radius = scaled(11)
        self._crop_handle_pressed = False    # a press landed on the handle
        self._crop_handle_moved = False      # moved past the click threshold since press
        self._crop_handle_press_pos = None   # widget-space press point (for the threshold)
        self._crop_drag_last_img = None      # last mouse pos in image coords while dragging

        # Suggestion hover state (for accepting specific components)
        self._hovered_component = None  # Binary mask of hovered connected component
        self._labeled_suggestion = None  # Cached labeled array
        self._num_labels = 0

        # F key modifier state (for temporary fill mode)
        self._f_key_held = False

        # Floating hint state
        self._tab_count = 0  # How many times Tab has been pressed
        self._suggestion_accept_count = 0  # How many times suggestion accepted

        # 3D mode state (kept for compatibility with interactive_training_page)
        self._3d_mode_enabled = False
        self._3d_session_active = False
        self._3d_fixed_bounds = None
        self._3d_button_rects = {}  # Stores button rectangles for 3D mode

        # === Async loading state ===
        # LRU cache for tiles keyed by (slice_idx, x1, y1, x2, y2, level)
        self._tile_lru_cache = OrderedDict()
        self._tile_cache_max_size = 10  # Keep up to 10 tiles in memory

        # Pending async load
        self._pending_load_future = None
        self._pending_load_params = None

        # Timer for checking async load completion
        self._async_check_timer = QTimer(self)
        self._async_check_timer.timeout.connect(self._check_async_load)
        self._async_check_timer.setInterval(16)  # ~60fps check rate

    # ==========================================================================
    # ZARR SOURCE MANAGEMENT
    # ==========================================================================

    def set_zarr_source(self, zarr_source, slice_index: int):
        """Set zarr pyramid source and slice index."""
        self._zarr_source = zarr_source
        self._zarr_slice_idx = slice_index
        self._image_height = zarr_source.height
        self._image_width = zarr_source.width

        # Clear raw_image since we're in zarr mode
        self.raw_image = None

        # Invalidate single-tile cache (LRU cache persists)
        self._cache_valid = False
        self._cached_tile = None
        self._cached_tile_params = None

        # Cancel any pending async load for old slice
        if self._pending_load_future is not None:
            self._pending_load_future.cancel()
            self._pending_load_future = None
            self._pending_load_params = None

        self.update()

        # Prefetch adjacent slices in background
        self._prefetch_adjacent_slices(slice_index)

    def _prefetch_adjacent_slices(self, current_slice: int):
        """Prefetch tiles for adjacent slices in background threads."""
        if self._zarr_source is None:
            return

        # Get current viewport bounds
        bounds = self.get_viewport_bounds()
        if bounds is None:
            return

        x1, y1, x2, y2 = bounds
        level = self._select_pyramid_level()

        # Prefetch slices: current-1, current+1, current-2, current+2
        prefetch_offsets = [-1, 1, -2, 2]
        for offset in prefetch_offsets:
            slice_idx = current_slice + offset
            if 0 <= slice_idx < self._zarr_source.num_slices:
                cache_key = (slice_idx, x1, y1, x2, y2, level)
                if cache_key not in self._tile_lru_cache:
                    # Submit prefetch job
                    _tile_executor.submit(
                        self._load_tile_for_cache,
                        slice_idx, y1, y2, x1, x2, level
                    )

    def _load_tile_for_cache(self, slice_idx: int, y1: int, y2: int, x1: int, x2: int, level: int):
        """Load a tile and store in LRU cache (runs in background thread)."""
        if self._zarr_source is None:
            return

        try:
            tile, downsample = self._zarr_source.get_tile_native(
                slice_idx, y1, y2, x1, x2, pyramid_level=level
            )
            cache_key = (slice_idx, x1, y1, x2, y2, level)

            # Store in cache (thread-safe for OrderedDict in CPython due to GIL)
            self._tile_lru_cache[cache_key] = (tile.copy(), downsample)

            # Evict old entries if cache too large
            while len(self._tile_lru_cache) > self._tile_cache_max_size:
                self._tile_lru_cache.popitem(last=False)

        except Exception as e:
            pass  # Silently ignore prefetch errors

    def _get_cached_tile(self, slice_idx: int, x1: int, y1: int, x2: int, y2: int, level: int):
        """Get tile from LRU cache if available, moving to end (most recent)."""
        cache_key = (slice_idx, x1, y1, x2, y2, level)
        if cache_key in self._tile_lru_cache:
            # Move to end (most recently used)
            self._tile_lru_cache.move_to_end(cache_key)
            return self._tile_lru_cache[cache_key]
        return None

    def _check_async_load(self):
        """Check if async load completed and update display."""
        if self._pending_load_future is None:
            self._async_check_timer.stop()
            return

        if self._pending_load_future.done():
            try:
                result = self._pending_load_future.result()
                if result is not None:
                    # Store in cache and trigger repaint
                    tile, downsample, cache_key = result
                    self._tile_lru_cache[cache_key] = (tile, downsample)
                    while len(self._tile_lru_cache) > self._tile_cache_max_size:
                        self._tile_lru_cache.popitem(last=False)
                    self.update()
            except Exception:
                pass
            finally:
                self._pending_load_future = None
                self._pending_load_params = None
                self._async_check_timer.stop()

    def _get_image_dimensions(self):
        """Get image dimensions from zarr source or raw_image."""
        if self._zarr_source is not None:
            return self._image_height, self._image_width
        elif self.raw_image is not None:
            return self.raw_image.shape
        return None, None

    def _select_pyramid_level(self):
        """Select optimal pyramid level based on zoom."""
        if self._zarr_source is None:
            return 0

        num_levels = len(self._zarr_source.pyramid_paths)
        if num_levels <= 1:
            return 0

        # Calculate optimal level: zoom=1 -> level 0, zoom=0.5 -> level 1, etc.
        if self.zoom_level >= 1.0:
            return 0

        optimal_level = int(math.log2(1.0 / self.zoom_level))
        selected_level = min(optimal_level, num_levels - 1)

        # Debug: print level changes
        if not hasattr(self, '_last_reported_level') or self._last_reported_level != selected_level:
            downsample = self._zarr_source.downsample_factors[selected_level] if selected_level < len(self._zarr_source.downsample_factors) else 1
            print(f"[Canvas] Zoom {self.zoom_level:.2f} → Pyramid level {selected_level} (downsample {downsample}x)")
            self._last_reported_level = selected_level

        return selected_level

    # ==========================================================================
    # IMAGE/MASK SETTERS (invalidate cache)
    # ==========================================================================

    def set_image(self, image: np.ndarray):
        """Set the raw grayscale image to display."""
        super().set_image(image)
        self._cache_valid = False
        # Clear zarr mode
        self._zarr_source = None
        self._image_height = None
        self._image_width = None

    def set_mask(self, mask: np.ndarray):
        """Set the segmentation mask."""
        super().set_mask(mask)
        self._cache_valid = False

    def set_suggestion(self, suggestion: np.ndarray):
        """Set the AI suggestion mask."""
        super().set_suggestion(suggestion)
        self._cache_valid = False
        # Clear cached labels - will be recomputed on hover
        self._labeled_suggestion = None
        self._num_labels = 0
        self._hovered_component = None

        # Re-check hover state at current cursor position so spacebar works
        # immediately without requiring a mouse move after new predictions arrive
        if suggestion is not None and self.show_suggestion:
            cursor_pos = self.mapFromGlobal(self.cursor().pos())
            if self.rect().contains(cursor_pos):
                h, w = self._get_image_dimensions()
                if h is not None:
                    scaled_w = int(w * self.zoom_level)
                    scaled_h = int(h * self.zoom_level)
                    img_x = (self.width() - scaled_w) // 2 + self.offset.x()
                    img_y = (self.height() - scaled_h) // 2 + self.offset.y()
                    rel_x = cursor_pos.x() - img_x
                    rel_y = cursor_pos.y() - img_y
                    if 0 <= rel_x < scaled_w and 0 <= rel_y < scaled_h:
                        px = int(rel_x / self.zoom_level)
                        py = int(rel_y / self.zoom_level)
                        self._update_hovered_component(px, py)

    def set_image_alpha(self, alpha: float):
        """Set the base image alpha."""
        super().set_image_alpha(alpha)
        self._cache_valid = False

    def set_mask_alpha(self, alpha: float):
        """Set the mask overlay alpha."""
        super().set_mask_alpha(alpha)
        self._cache_valid = False

    def set_suggestion_alpha(self, alpha: float):
        """Set the suggestion overlay alpha."""
        super().set_suggestion_alpha(alpha)
        self._cache_valid = False

    def toggle_suggestion_visibility(self, visible: bool):
        """Toggle suggestion overlay visibility."""
        super().toggle_suggestion_visibility(visible)
        self._cache_valid = False

    # ==========================================================================
    # CROP PREVIEW
    # ==========================================================================

    def set_crop_preview_alpha(self, alpha: float):
        """Set the crop preview box alpha (0.0 to 1.0)."""
        self._crop_preview_alpha = max(0.0, min(1.0, alpha))
        self.update()

    def set_crop_preview_visible(self, visible: bool):
        """Toggle crop preview box visibility."""
        self._crop_preview_visible = visible
        self.update()

    def set_crop_size(self, size: int):
        """Set the crop size (default 256)."""
        self._crop_size = size
        self._update_crop_preview_from_bounds()
        self.update()

    def clear_paint_bounds(self):
        """Clear the tracked painting bounds (call after capturing crop)."""
        self._paint_bounds_min = None
        self._paint_bounds_max = None
        # A locked box stays put for the next capture; an unlocked one is reset.
        if not self._crop_locked:
            self._crop_preview_bounds = None
        self.update()

    def get_crop_bounds(self):
        """Get current crop preview bounds (x, y, w, h) in image coordinates."""
        return self._crop_preview_bounds

    def _crop_handle_center(self):
        """Screen-space (x, y) of the drag/lock handle at the crop box's top-right,
        or None when there is no draggable box (e.g. hidden, or 3D session)."""
        if not (self._crop_preview_visible and self._crop_preview_bounds is not None):
            return None
        if self._3d_session_active:
            return None
        h, w = self._get_image_dimensions()
        if h is None:
            return None
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()
        crop_x, crop_y, crop_w, crop_h = self._crop_preview_bounds
        return (img_x + int((crop_x + crop_w) * self.zoom_level),
                img_y + int(crop_y * self.zoom_level))

    def _pos_on_crop_handle(self, pos):
        """True if a widget-space QPoint is on the crop handle (with a little slop)."""
        center = self._crop_handle_center()
        if center is None:
            return False
        r = self._crop_handle_radius + scaled(4)  # generous hit target
        return (pos.x() - center[0]) ** 2 + (pos.y() - center[1]) ** 2 <= r * r

    def _screen_to_image(self, pos):
        """Convert a widget-space QPoint to (px, py) image coordinates (may be OOB)."""
        h, w = self._get_image_dimensions()
        if h is None:
            return None
        img_x = (self.width() - int(w * self.zoom_level)) // 2 + self.offset.x()
        img_y = (self.height() - int(h * self.zoom_level)) // 2 + self.offset.y()
        return (int((pos.x() - img_x) / self.zoom_level),
                int((pos.y() - img_y) / self.zoom_level))

    def _update_crop_preview_from_bounds(self):
        """Update crop preview box to center on painted region."""
        # A locked box stays frozen; an unlocked box keeps following the paint stroke.
        if self._crop_locked and self._crop_preview_bounds is not None:
            return
        if self._paint_bounds_min is None or self._paint_bounds_max is None:
            return

        h, w = self._get_image_dimensions()
        if h is None:
            return

        size = self._crop_size

        # Calculate center of painted region
        center_x = (self._paint_bounds_min[0] + self._paint_bounds_max[0]) // 2
        center_y = (self._paint_bounds_min[1] + self._paint_bounds_max[1]) // 2

        # Calculate crop box position (centered on paint center)
        crop_x = center_x - size // 2
        crop_y = center_y - size // 2

        # Clamp to image bounds
        crop_x = max(0, min(crop_x, w - size))
        crop_y = max(0, min(crop_y, h - size))

        # Ensure we have enough space for the crop
        if w >= size and h >= size:
            self._crop_preview_bounds = (crop_x, crop_y, size, size)
        else:
            # Image smaller than crop size - use full image
            self._crop_preview_bounds = (0, 0, min(w, size), min(h, size))

    # ==========================================================================
    # VIEWPORT BOUNDS (for predictions)
    # ==========================================================================

    def get_viewport_bounds(self):
        """Get visible image region in full-resolution pixel coordinates.

        Returns:
            (left, top, right, bottom) or None if no image loaded.
        """
        h, w = self._get_image_dimensions()
        if h is None:
            return None

        widget_w, widget_h = self.width(), self.height()
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)

        # Image position on widget
        img_x = (widget_w - scaled_w) // 2 + self.offset.x()
        img_y = (widget_h - scaled_h) // 2 + self.offset.y()

        # Visible region in widget coords
        vis_left = max(0, -img_x)
        vis_top = max(0, -img_y)
        vis_right = min(scaled_w, widget_w - img_x)
        vis_bottom = min(scaled_h, widget_h - img_y)

        # Convert to image pixel coords
        img_left = int(vis_left / self.zoom_level)
        img_top = int(vis_top / self.zoom_level)
        img_right = int(math.ceil(vis_right / self.zoom_level))
        img_bottom = int(math.ceil(vis_bottom / self.zoom_level))

        # Clamp to image bounds
        img_left = max(0, min(img_left, w))
        img_top = max(0, min(img_top, h))
        img_right = max(0, min(img_right, w))
        img_bottom = max(0, min(img_bottom, h))

        return (img_left, img_top, img_right, img_bottom)

    # ==========================================================================
    # SUGGESTION HOVER
    # ==========================================================================

    def accept_suggestion(self):
        """Accept the current suggestion as the mask."""
        super().accept_suggestion()
        self._cache_valid = False

    def _update_hovered_component(self, img_x: int, img_y: int):
        """Update which suggestion component is being hovered."""
        # Only works when suggestion is visible
        if not self.show_suggestion or self.suggestion is None:
            self._hovered_component = None
            return

        h, w = self.suggestion.shape

        # Check bounds
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            self._hovered_component = None
            return

        # Check if cursor is over a suggestion pixel
        if self.suggestion[img_y, img_x] <= 127:
            self._hovered_component = None
            return

        # Compute labeled components if not cached
        if self._labeled_suggestion is None:
            from scipy import ndimage
            binary = (self.suggestion > 127).astype(np.uint8)
            self._labeled_suggestion, self._num_labels = ndimage.label(binary)

        # Get the label at cursor position
        label = self._labeled_suggestion[img_y, img_x]
        if label == 0:
            self._hovered_component = None
            return

        # Create mask for this component
        self._hovered_component = (self._labeled_suggestion == label)

    def accept_hovered_component(self):
        """Accept the currently hovered suggestion component into the mask."""
        if self._hovered_component is None:
            return False

        # Allocate mask if it doesn't exist yet (fresh slice with no prior edits)
        if self.mask is None:
            self._ensure_mask_exists()
            if self.mask is None:
                return False

        # Compute tight bounds of the component for efficient undo
        rows = np.any(self._hovered_component, axis=1)
        cols = np.any(self._hovered_component, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        y2 += 1
        x2 += 1

        self._edit_bounds = [y1, y2, x1, x2]
        # Snapshot region before modification
        self._undo_region = self.mask[y1:y2, x1:x2].copy()
        self._undo_bounds = [y1, y2, x1, x2]

        # Add hovered component to mask
        self.mask[self._hovered_component] = 255

        # Clear the hovered component
        self._hovered_component = None

        # Emit edit signal
        self.emit_edit()

        self.update()
        return True

    def get_hovered_component(self):
        """Return the currently hovered component mask (or None)."""
        return self._hovered_component

    # ==========================================================================
    # DRAWING
    # ==========================================================================

    def fill_at(self, pos: QPoint):
        """Flood fill at the given position."""
        h, w = self._get_image_dimensions()
        if h is None or self.mask is None:
            return

        from scipy import ndimage

        # Calculate image coordinates
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()

        rel_x = pos.x() - img_x
        rel_y = pos.y() - img_y

        if rel_x < 0 or rel_y < 0 or rel_x >= scaled_w or rel_y >= scaled_h:
            return

        px = int(rel_x / self.zoom_level)
        py = int(rel_y / self.zoom_level)

        # Clamp to valid range
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))

        fill_value = 0 if self.erasing else 255
        current_value = self.mask[py, px]

        # Don't fill if already the target value
        if current_value == fill_value:
            return

        # Find connected component at click point
        if current_value > 127:
            binary = (self.mask > 127)
        else:
            binary = (self.mask <= 127)

        # Label connected components
        labeled, num_labels = ndimage.label(binary)

        # Get the label at the click point
        clicked_label = labeled[py, px]

        if clicked_label == 0:
            return

        # Fill all pixels with this label
        fill_mask = (labeled == clicked_label)
        self.mask[fill_mask] = fill_value
        self._cache_valid = False

        self.update()

    def draw_at(self, pos: QPoint):
        """Draw on the mask - viewport rendering handles display."""
        h, w = self._get_image_dimensions()
        if h is None or self.mask is None:
            return

        # Calculate image coordinates
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()

        rel_x = pos.x() - img_x
        rel_y = pos.y() - img_y

        if rel_x < 0 or rel_y < 0 or rel_x >= scaled_w or rel_y >= scaled_h:
            return

        px = int(rel_x / self.zoom_level)
        py = int(rel_y / self.zoom_level)
        value = 0 if self.erasing else 255
        radius = self.brush_size

        # Update numpy mask using vectorized operation (faster than loop)
        y_min = max(0, py - radius)
        y_max = min(h, py + radius + 1)
        x_min = max(0, px - radius)
        x_max = min(w, px + radius + 1)

        # Create coordinate grids for the bounding box
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        # Create circular mask
        circle_mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
        # Track edit bounds and undo snapshot BEFORE modifying the mask
        if self._edit_bounds is None:
            self._edit_bounds = [y_min, y_max, x_min, x_max]
            # Snapshot the region before first modification of this stroke
            self._undo_region = self.mask[y_min:y_max, x_min:x_max].copy()
            self._undo_bounds = [y_min, y_max, x_min, x_max]
        else:
            new_y1 = min(self._edit_bounds[0], y_min)
            new_y2 = max(self._edit_bounds[1], y_max)
            new_x1 = min(self._edit_bounds[2], x_min)
            new_x2 = max(self._edit_bounds[3], x_max)
            if (new_y1 < self._undo_bounds[0] or new_y2 > self._undo_bounds[1] or
                    new_x1 < self._undo_bounds[2] or new_x2 > self._undo_bounds[3]):
                # Expand: snapshot larger region, merge with old snapshot to preserve originals
                oy1, oy2, ox1, ox2 = self._undo_bounds
                expanded = self.mask[new_y1:new_y2, new_x1:new_x2].copy()
                ey1 = oy1 - new_y1
                ey2 = ey1 + (oy2 - oy1)
                ex1 = ox1 - new_x1
                ex2 = ex1 + (ox2 - ox1)
                expanded[ey1:ey2, ex1:ex2] = self._undo_region
                self._undo_region = expanded
                self._undo_bounds = [new_y1, new_y2, new_x1, new_x2]
            self._edit_bounds[0] = new_y1
            self._edit_bounds[1] = new_y2
            self._edit_bounds[2] = new_x1
            self._edit_bounds[3] = new_x2

        # Apply to mask (after snapshot)
        self.mask[y_min:y_max, x_min:x_max][circle_mask] = value

        self._cache_valid = False

        # Track painting bounds for crop preview
        if self._paint_bounds_min is None:
            self._paint_bounds_min = (x_min, y_min)
            self._paint_bounds_max = (x_max, y_max)
        else:
            self._paint_bounds_min = (
                min(self._paint_bounds_min[0], x_min),
                min(self._paint_bounds_min[1], y_min)
            )
            self._paint_bounds_max = (
                max(self._paint_bounds_max[0], x_max),
                max(self._paint_bounds_max[1], y_max)
            )

        # Update crop preview box position
        self._update_crop_preview_from_bounds()

        # Trigger repaint - viewport rendering will handle display
        self.update()

    def _ensure_mask_exists(self):
        """Ensure mask array exists at full resolution."""
        h, w = self._get_image_dimensions()
        if h is None:
            return

        if self.mask is None:
            self.mask = np.zeros((h, w), dtype=np.uint8)

    # ==========================================================================
    # EVENT HANDLERS
    # ==========================================================================

    def mousePressEvent(self, event):
        """Handle mouse press with button mappings: left=paint, right=erase, middle=pan."""
        # 3D mode buttons
        if event.button() == Qt.MouseButton.LeftButton and self._3d_session_active:
            button = self._get_3d_button_at_pos(event.pos())
            if button == 'check':
                self.save_3d_requested.emit()
                return
            elif button == 'x':
                self.cancel_3d_requested.emit()
                return

        # Crop-box handle: take priority over painting. A drag moves an unlocked box;
        # a click (no drag) toggles the lock. A locked box can't be dragged.
        if (event.button() == Qt.MouseButton.LeftButton
                and self._pos_on_crop_handle(event.pos())):
            self._crop_handle_pressed = True
            self._crop_handle_moved = False
            self._crop_handle_press_pos = event.pos()
            self._crop_drag_last_img = self._screen_to_image(event.pos())
            return

        # Right-click: temporarily switch to eraser
        if event.button() == Qt.MouseButton.RightButton:
            self._ensure_mask_exists()
            if self.mask is not None:
                self._right_click_erasing = True
                self._original_erasing = self.erasing
                self.erasing = True
                self.drawing = True
                self._edit_bounds = None  # Will be tracked by draw_at
                self.draw_at(event.pos())
            return

        # Left-click with fill tool OR F key held (temporary fill)
        if event.button() == Qt.MouseButton.LeftButton and (self.current_tool == 'fill' or self._f_key_held):
            self._ensure_mask_exists()
            if self.mask is not None:
                # Fill affects whole mask potentially — snapshot full mask for undo
                h, w = self.mask.shape
                self._edit_bounds = [0, h, 0, w]
                self._undo_region = self.mask.copy()
                self._undo_bounds = [0, h, 0, w]
                self.fill_at(event.pos())
                self.emit_edit()
            return

        # Middle-click: pan (handled by parent)
        # Left-click: paint (default, handled by parent)
        self._ensure_mask_exists()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing and panning."""
        # Update cursor position for brush preview
        self.cursor_pos = event.pos()

        # Interacting with the crop-box handle.
        if self._crop_handle_pressed:
            if self._crop_handle_press_pos is not None:
                d = event.pos() - self._crop_handle_press_pos
                thr = scaled(4)  # click vs. drag threshold
                if d.x() * d.x() + d.y() * d.y() >= thr * thr:
                    self._crop_handle_moved = True
            # Only an unlocked box can be dragged; a locked box is frozen in place.
            if self._crop_handle_moved and not self._crop_locked:
                cur = self._screen_to_image(event.pos())
                if (cur is not None and self._crop_drag_last_img is not None
                        and self._crop_preview_bounds is not None):
                    dx = cur[0] - self._crop_drag_last_img[0]
                    dy = cur[1] - self._crop_drag_last_img[1]
                    if dx or dy:
                        h, w = self._get_image_dimensions()
                        cx, cy, cw, ch = self._crop_preview_bounds
                        max_x = (w - cw) if (w is not None and w >= cw) else 0
                        max_y = (h - ch) if (h is not None and h >= ch) else 0
                        nx = max(0, min(cx + dx, max_x))
                        ny = max(0, min(cy + dy, max_y))
                        self._crop_preview_bounds = (nx, ny, cw, ch)
                        self._crop_drag_last_img = cur
            self.update()
            return

        if self.drawing:
            self.draw_at(event.pos())
        elif self.last_mouse_pos is not None:
            # Pan
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.viewport_changed.emit()
        else:
            # Not drawing or panning - check for suggestion hover
            h, w = self._get_image_dimensions()
            if h is not None and self.show_suggestion and self.suggestion is not None:
                # Convert screen pos to image coordinates
                scaled_w = int(w * self.zoom_level)
                scaled_h = int(h * self.zoom_level)
                img_x = (self.width() - scaled_w) // 2 + self.offset.x()
                img_y = (self.height() - scaled_h) // 2 + self.offset.y()

                rel_x = event.pos().x() - img_x
                rel_y = event.pos().y() - img_y

                if 0 <= rel_x < scaled_w and 0 <= rel_y < scaled_h:
                    px = int(rel_x / self.zoom_level)
                    py = int(rel_y / self.zoom_level)
                    self._update_hovered_component(px, py)
                else:
                    self._hovered_component = None

        self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release with button mappings."""
        # End a crop-handle interaction. A press+release with no drag is a click,
        # which toggles the lock (circle <-> lock icon).
        if self._crop_handle_pressed:
            self._crop_handle_pressed = False
            if not self._crop_handle_moved:
                self._crop_locked = not self._crop_locked
            self._crop_drag_last_img = None
            self._crop_handle_press_pos = None
            self.update()
            return

        # Right-click release: restore original erasing state
        if event.button() == Qt.MouseButton.RightButton:
            if hasattr(self, '_right_click_erasing') and self._right_click_erasing:
                self._right_click_erasing = False
                self.erasing = getattr(self, '_original_erasing', False)
                if self.drawing:
                    self.drawing = False
                    self.emit_edit()
            return

        super().mouseReleaseEvent(event)

    def event(self, event):
        """Override event to catch Tab, Space, and F keys."""
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Tab:
                # Always emit capture_requested - page will handle 3D vs normal mode
                # Only require crop preview bounds in non-3D mode
                print(f"[Canvas] Tab pressed. 3D mode: {self._3d_mode_enabled}, 3D session: {self._3d_session_active}, crop_bounds: {self._crop_preview_bounds is not None}")
                if self._3d_session_active or self._crop_preview_bounds is not None:
                    self._tab_count += 1
                    print(f"[Canvas] Incremented _tab_count to {self._tab_count}, emitting capture_requested")
                    self.capture_requested.emit()
                else:
                    print(f"[Canvas] NOT emitting - no 3D session and no crop bounds")
                return True  # Event handled
            elif event.key() == Qt.Key.Key_Space:
                # Accept hovered suggestion component
                if self.accept_hovered_component():
                    self._suggestion_accept_count += 1
                    self.suggestion_accepted.emit()
                    return True
            elif event.key() == Qt.Key.Key_F:
                # F key held = temporary fill mode
                self._f_key_held = True
                return True
            # S key is handled at page level to sync with checkbox
        elif event.type() == QEvent.Type.KeyRelease:
            if event.key() == Qt.Key.Key_F:
                # F key released = exit temporary fill mode
                self._f_key_held = False
                return True
        return super().event(event)

    def keyPressEvent(self, event):
        """Handle key press events."""
        # Bracket keys for brush size
        if event.key() == Qt.Key.Key_BracketLeft:
            self.brush_size = max(1, self.brush_size - 5)
            self.brush_size_changed.emit(self.brush_size)
            self.update()
            return

        if event.key() == Qt.Key.Key_BracketRight:
            self.brush_size = min(200, self.brush_size + 5)
            self.brush_size_changed.emit(self.brush_size)
            self.update()
            return

        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel - Shift+scroll changes brush size, otherwise zoom."""
        event.accept()  # Prevent propagation

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+scroll = change brush size
            delta = event.angleDelta().y()
            if delta > 0:
                new_size = min(100, self.brush_size + 2)
            else:
                new_size = max(1, self.brush_size - 2)

            if new_size != self.brush_size:
                self.brush_size = new_size
                self.brush_size_changed.emit(new_size)
                self.update()
        else:
            # Normal scroll = zoom
            cursor_pos = event.position().toPoint()
            if event.angleDelta().y() > 0:
                self.zoom_in(cursor_pos)
            else:
                self.zoom_out(cursor_pos)

    # ==========================================================================
    # ZOOM
    # ==========================================================================

    def zoom_in(self, cursor_pos=None):
        """Zoom in at cursor position."""
        if cursor_pos is None or isinstance(cursor_pos, bool):
            cursor_pos = getattr(self, 'cursor_pos', None)
        self._zoom_at_point(1.2, cursor_pos)

    def zoom_out(self, cursor_pos=None):
        """Zoom out at cursor position."""
        if cursor_pos is None or isinstance(cursor_pos, bool):
            cursor_pos = getattr(self, 'cursor_pos', None)
        self._zoom_at_point(1 / 1.2, cursor_pos)

    def _zoom_at_point(self, factor: float, cursor_pos):
        """Zoom by factor, keeping the point under cursor stationary."""
        h, w = self._get_image_dimensions()
        if h is None:
            return

        old_zoom = self.zoom_level
        new_zoom = max(0.1, min(50.0, old_zoom * factor))

        if cursor_pos is not None:
            # Current image position on screen
            scaled_w = int(w * old_zoom)
            scaled_h = int(h * old_zoom)
            img_x = (self.width() - scaled_w) // 2 + self.offset.x()
            img_y = (self.height() - scaled_h) // 2 + self.offset.y()

            # Cursor position relative to image origin (in screen coords)
            rel_x = cursor_pos.x() - img_x
            rel_y = cursor_pos.y() - img_y

            # Image coordinate under cursor
            img_coord_x = rel_x / old_zoom
            img_coord_y = rel_y / old_zoom

            # After zoom, where would this image coordinate appear?
            new_scaled_w = int(w * new_zoom)
            new_scaled_h = int(h * new_zoom)
            new_img_x = (self.width() - new_scaled_w) // 2 + self.offset.x()
            new_img_y = (self.height() - new_scaled_h) // 2 + self.offset.y()

            # New screen position of the same image coordinate
            new_screen_x = new_img_x + img_coord_x * new_zoom
            new_screen_y = new_img_y + img_coord_y * new_zoom

            # Adjust offset to keep cursor over the same image point
            offset_adjust_x = cursor_pos.x() - new_screen_x
            offset_adjust_y = cursor_pos.y() - new_screen_y

            self.offset.setX(int(self.offset.x() + offset_adjust_x))
            self.offset.setY(int(self.offset.y() + offset_adjust_y))

        self.zoom_level = new_zoom

        # Invalidate tile cache when zoom changes (different pyramid level)
        self._cached_tile = None
        self._cached_tile_params = None

        self.update()
        self.viewport_changed.emit()

    # ==========================================================================
    # 3D MODE
    # ==========================================================================

    def set_3d_mode_enabled(self, enabled: bool):
        """Enable/disable 3D mode buttons."""
        self._3d_mode_enabled = enabled
        self.update()

    def set_3d_session_active(self, active: bool, bounds=None):
        """Set whether a 3D session is currently active."""
        self._3d_session_active = active
        self._3d_fixed_bounds = bounds
        self.update()

    def _get_3d_button_at_pos(self, pos):
        """Check if position is over a 3D mode button."""
        for name, rect in self._3d_button_rects.items():
            if rect.contains(pos):
                return name
        return None

    def get_3d_button_at_pos(self, pos):
        """Public method to check button at position."""
        return self._get_3d_button_at_pos(pos)

    # ==========================================================================
    # RENDERING
    # ==========================================================================

    def paintEvent(self, event: QPaintEvent):
        """Optimized render - only renders visible viewport region."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        h, w = self._get_image_dimensions()
        if h is None:
            painter.end()
            return

        # Calculate where the full image would be positioned
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()

        # Calculate visible region in IMAGE coordinates
        vis_left = max(0, -img_x)
        vis_top = max(0, -img_y)
        vis_right = min(scaled_w, self.width() - img_x)
        vis_bottom = min(scaled_h, self.height() - img_y)

        # Fix: At extreme zoom-out, ensure we always show the full image
        # rather than returning with grey screen
        if vis_right <= vis_left or vis_bottom <= vis_top:
            # Image would be invisible - render full image at current zoom
            vis_left = max(0, img_x) if img_x < 0 else 0
            vis_top = max(0, img_y) if img_y < 0 else 0
            vis_right = min(scaled_w, self.width() if img_x >= 0 else self.width() - img_x)
            vis_bottom = min(scaled_h, self.height() if img_y >= 0 else self.height() - img_y)

        # Convert to image pixel coordinates
        src_left = int(vis_left / self.zoom_level)
        src_top = int(vis_top / self.zoom_level)
        src_right = min(w, int(np.ceil(vis_right / self.zoom_level)) + 1)
        src_bottom = min(h, int(np.ceil(vis_bottom / self.zoom_level)) + 1)

        # Extract and render only the visible region (at native pyramid resolution)
        viewport_pixmap, downsample = self._render_region(src_left, src_top, src_right, src_bottom)

        # Debug: if pixmap is None at low zoom, log it
        if viewport_pixmap is None and self.zoom_level < 0.1:
            print(f"[Canvas] Grey screen debug: zoom={self.zoom_level:.4f}, src=({src_left},{src_top})-({src_right},{src_bottom}), vis=({vis_left},{vis_top})-({vis_right},{vis_bottom})")

        if viewport_pixmap is not None:
            # Calculate destination rectangle on screen
            # The pixmap is at native pyramid resolution, Qt will scale it
            dest_w = int((src_right - src_left) * self.zoom_level)
            dest_h = int((src_bottom - src_top) * self.zoom_level)

            # Draw position
            draw_x = img_x + int(src_left * self.zoom_level)
            draw_y = img_y + int(src_top * self.zoom_level)

            # Ensure dest dimensions are at least 1
            if dest_w <= 0 or dest_h <= 0:
                dest_w = max(1, dest_w)
                dest_h = max(1, dest_h)

            # Pre-scale pixmap with fast nearest-neighbor, then draw at 1:1
            scaled_pixmap = viewport_pixmap.scaled(
                dest_w, dest_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            painter.drawPixmap(draw_x, draw_y, scaled_pixmap)

        # Draw overlays
        self._draw_brush_cursor(painter)
        self._draw_crop_preview(painter, img_x, img_y)
        self._draw_suggestion_hint(painter)

        # Draw 3D mode save/cancel buttons (green checkmark, red X)
        if self._3d_session_active and self._3d_fixed_bounds:
            x, y, crop_w, crop_h = self._3d_fixed_bounds
            scaled_crop_w = int(crop_w * self.zoom_level)
            scaled_crop_h = int(crop_h * self.zoom_level)
            screen_x = img_x + int(x * self.zoom_level)
            screen_y = img_y + int(y * self.zoom_level)
            self._draw_3d_buttons(painter, screen_x, screen_y, scaled_crop_w, scaled_crop_h)

        painter.end()

    def _render_region(self, x1: int, y1: int, x2: int, y2: int) -> tuple:
        """Render a specific region of the image to a pixmap.

        Returns:
            tuple: (QPixmap, downsample_factor) or (None, 1)
            The downsample_factor indicates how much smaller the pixmap is
            compared to the requested region (for Qt scaling).
        """
        h, w = self._get_image_dimensions()
        if h is None:
            return None, 1

        region_h = y2 - y1
        region_w = x2 - x1

        if region_h <= 0 or region_w <= 0:
            if self.zoom_level < 0.1:
                print(f"[Canvas] _render_region: empty region at zoom={self.zoom_level:.4f}")
            return None, 1

        # Check composited pixmap cache — avoid recomputing numpy overlays
        # when nothing has changed (e.g. cursor moves, timer repaints, etc.)
        level = 0
        if self._zarr_source is not None:
            level = self._select_pyramid_level()
        render_key = (
            getattr(self, '_zarr_slice_idx', 0), x1, y1, x2, y2, level,
            id(self.mask) if self.mask is not None else 0,
            id(self.suggestion) if self.suggestion is not None else 0,
            id(self._hovered_component) if self._hovered_component is not None else 0,
        )
        if self._cache_valid and self._cached_render_key == render_key and self._cached_pixmap is not None:
            # Nothing changed — return cached pixmap
            downsample = self._zarr_source.downsample_factors[level] if self._zarr_source and level < len(self._zarr_source.downsample_factors) else 1
            return self._cached_pixmap, downsample

        downsample = 1  # Default for raw image mode

        # === Get image data ===
        if self._zarr_source is not None:
            # Zarr mode: load from pyramid at NATIVE resolution (no upscaling)
            level = self._select_pyramid_level()

            # Check LRU cache first (includes prefetched tiles)
            cached = self._get_cached_tile(self._zarr_slice_idx, x1, y1, x2, y2, level)
            if cached is not None:
                tile, downsample = cached
                tile = tile.copy()
            else:
                # Not in cache - load synchronously (will prefetch neighbors after)
                try:
                    tile, downsample = self._zarr_source.get_tile_native(
                        self._zarr_slice_idx, y1, y2, x1, x2, pyramid_level=level
                    )
                    # Store in LRU cache
                    cache_key = (self._zarr_slice_idx, x1, y1, x2, y2, level)
                    self._tile_lru_cache[cache_key] = (tile.copy(), downsample)
                    while len(self._tile_lru_cache) > self._tile_cache_max_size:
                        self._tile_lru_cache.popitem(last=False)
                except Exception as e:
                    print(f"[Canvas] Error loading tile: {e}")
                    return None, 1

            # Check for empty tile (can happen at extreme zoom levels)
            if tile.size == 0 or tile.shape[0] == 0 or tile.shape[1] == 0:
                if self.zoom_level < 0.1:
                    print(f"[Canvas] _render_region: empty tile from zarr at zoom={self.zoom_level:.4f}, level={level}")
                return None, 1

            # Normalize using global stats
            img_min = self._zarr_source.global_min or 0
            img_max = self._zarr_source.global_max or 255
            if img_max > img_min:
                tile = (tile - img_min) / (img_max - img_min)
            tile = (tile * 255 * self.image_alpha).clip(0, 255).astype(np.uint8)

        elif self.raw_image is not None:
            # Raw image mode - no downsampling
            tile = self.raw_image[y1:y2, x1:x2].copy()
            img_min, img_max = self.raw_image.min(), self.raw_image.max()
            if img_max > img_min:
                tile = (tile - img_min) / (img_max - img_min)
            tile = (tile * 255 * self.image_alpha).astype(np.uint8)
            downsample = 1
        else:
            return None, 1

        tile_h, tile_w = tile.shape[:2]

        # Create RGB from grayscale tile
        img_rgb = np.stack([tile, tile, tile], axis=-1)

        # Overlay mask (red channel) - downsample mask to match tile
        if self.mask is not None and self.mask_alpha > 0:
            # Extract mask region and downsample to tile size
            mask_region = self.mask[y1:y2, x1:x2]
            if downsample > 1:
                # Fast nearest-neighbor downsample using slicing
                mask_region = mask_region[::downsample, ::downsample]

            # Ensure sizes match (may be off by 1 due to rounding)
            mh, mw = mask_region.shape
            if mh > tile_h:
                mask_region = mask_region[:tile_h, :]
            if mw > tile_w:
                mask_region = mask_region[:, :tile_w]
            if mh < tile_h or mw < tile_w:
                # Pad if needed
                padded = np.zeros((tile_h, tile_w), dtype=mask_region.dtype)
                padded[:mh, :mw] = mask_region
                mask_region = padded

            mask_overlay = (mask_region > 127).astype(np.float32)
            img_rgb[:, :, 0] = np.clip(
                img_rgb[:, :, 0] * (1 - mask_overlay * self.mask_alpha) +
                255 * mask_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Overlay RGB overlay (colored visualizations) - downsample to match tile
        if self.rgb_overlay is not None and self.rgb_overlay_alpha > 0:
            rgb_region = self.rgb_overlay[y1:y2, x1:x2, :]
            if downsample > 1:
                rgb_region = rgb_region[::downsample, ::downsample, :]

            # Ensure sizes match
            rh, rw = rgb_region.shape[:2]
            if rh > tile_h:
                rgb_region = rgb_region[:tile_h, :, :]
            if rw > tile_w:
                rgb_region = rgb_region[:, :tile_w, :]
            if rh < tile_h or rw < tile_w:
                padded = np.zeros((tile_h, tile_w, 3), dtype=rgb_region.dtype)
                padded[:rh, :rw, :] = rgb_region
                rgb_region = padded

            # Blend RGB overlay with base image
            alpha = self.rgb_overlay_alpha
            img_rgb = (img_rgb * (1 - alpha) + rgb_region * alpha).astype(np.uint8)

        # Overlay suggestion (green channel) - downsample to match tile
        if self.show_suggestion and self.suggestion is not None and self.mask_alpha > 0:
            sugg_region = self.suggestion[y1:y2, x1:x2]
            if downsample > 1:
                sugg_region = sugg_region[::downsample, ::downsample]

            # Ensure sizes match
            sh, sw = sugg_region.shape
            if sh > tile_h:
                sugg_region = sugg_region[:tile_h, :]
            if sw > tile_w:
                sugg_region = sugg_region[:, :tile_w]
            if sh < tile_h or sw < tile_w:
                padded = np.zeros((tile_h, tile_w), dtype=sugg_region.dtype)
                padded[:sh, :sw] = sugg_region
                sugg_region = padded

            sugg_overlay = (sugg_region > 127).astype(np.float32)
            img_rgb[:, :, 1] = np.clip(
                img_rgb[:, :, 1] * (1 - sugg_overlay * self.mask_alpha) +
                255 * sugg_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Highlight hovered component with subtle red checkerboard
        if self._hovered_component is not None:
            hover_region = self._hovered_component[y1:y2, x1:x2]
            if downsample > 1:
                hover_region = hover_region[::downsample, ::downsample]

            # Ensure sizes match
            hh, hw = hover_region.shape
            if hh > tile_h:
                hover_region = hover_region[:tile_h, :]
            if hw > tile_w:
                hover_region = hover_region[:, :tile_w]

            if hover_region.shape == (tile_h, tile_w) and hover_region.any():
                yy, xx = np.ogrid[:tile_h, :tile_w]
                # Adjust checker pattern for downsampled coordinates
                checker = ((yy * downsample + y1 + xx * downsample + x1) % 3 < 1)
                red_pixels = hover_region & checker
                img_rgb[red_pixels, 0] = np.clip(
                    img_rgb[red_pixels, 0] * (1 - self.mask_alpha) +
                    255 * self.mask_alpha, 0, 255
                ).astype(np.uint8)
                img_rgb[red_pixels, 1] = (img_rgb[red_pixels, 1] * (1 - self.mask_alpha)).astype(np.uint8)
                img_rgb[red_pixels, 2] = (img_rgb[red_pixels, 2] * (1 - self.mask_alpha)).astype(np.uint8)

        # Convert to QPixmap
        img_rgb = np.ascontiguousarray(img_rgb)
        qimg = QImage(img_rgb.data, tile_w, tile_h, 3 * tile_w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        # Cache the composited pixmap
        self._cached_pixmap = pixmap
        self._cached_render_key = render_key
        self._cache_valid = True

        return pixmap, downsample

    def _draw_brush_cursor(self, painter):
        """Draw brush cursor preview."""
        if self.cursor_pos and self.current_tool in ['brush', 'eraser']:
            scaled_radius = int(self.brush_size * self.zoom_level)
            pen = QPen(QColor(255, 255, 255, 180), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.cursor_pos, scaled_radius, scaled_radius)

    def _draw_crop_preview(self, painter, img_x, img_y):
        """Draw crop preview box (yellow dotted or blue solid for 3D session)."""
        # Determine which bounds to use and color scheme
        if self._3d_session_active and self._3d_fixed_bounds is not None:
            # 3D session: use fixed bounds with BLUE solid line
            crop_x, crop_y, crop_w, crop_h = self._3d_fixed_bounds
            is_3d_mode = True
        elif self._crop_preview_visible and self._crop_preview_bounds is not None:
            # Normal mode: use paint bounds with yellow dotted line
            crop_x, crop_y, crop_w, crop_h = self._crop_preview_bounds
            is_3d_mode = False
        else:
            return

        # For 3D mode, always show at full opacity; for normal mode use configurable alpha
        if is_3d_mode:
            alpha = 255  # Always fully visible in 3D mode
        else:
            if self._crop_preview_alpha <= 0:
                return
            alpha = int(255 * self._crop_preview_alpha)

        # Convert image coordinates to screen coordinates
        screen_x = img_x + int(crop_x * self.zoom_level)
        screen_y = img_y + int(crop_y * self.zoom_level)
        screen_w = int(crop_w * self.zoom_level)
        screen_h = int(crop_h * self.zoom_level)

        if is_3d_mode:
            # Blue solid line for 3D session
            pen = QPen(QColor(50, 150, 255, alpha), 3, Qt.PenStyle.SolidLine)
            marker_color = QColor(50, 150, 255, alpha)
        elif self._crop_locked:
            # Locked: blue dotted line so the locked state is unmistakable
            pen = QPen(QColor(60, 160, 255, alpha), 2, Qt.PenStyle.DashLine)
            marker_color = QColor(60, 160, 255, alpha)
        else:
            # Yellow dotted line for normal (unlocked) mode
            pen = QPen(QColor(255, 220, 50, alpha), 2, Qt.PenStyle.DashLine)
            marker_color = QColor(255, 220, 50, alpha)

        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(screen_x, screen_y, screen_w, screen_h)

        # Draw corner markers for better visibility
        marker_size = 8
        painter.setPen(QPen(marker_color, 3, Qt.PenStyle.SolidLine))
        # Top-left
        painter.drawLine(screen_x, screen_y, screen_x + marker_size, screen_y)
        painter.drawLine(screen_x, screen_y, screen_x, screen_y + marker_size)
        # Top-right
        painter.drawLine(screen_x + screen_w, screen_y, screen_x + screen_w - marker_size, screen_y)
        painter.drawLine(screen_x + screen_w, screen_y, screen_x + screen_w, screen_y + marker_size)
        # Bottom-left
        painter.drawLine(screen_x, screen_y + screen_h, screen_x + marker_size, screen_y + screen_h)
        painter.drawLine(screen_x, screen_y + screen_h, screen_x, screen_y + screen_h - marker_size)
        # Bottom-right
        painter.drawLine(screen_x + screen_w, screen_y + screen_h, screen_x + screen_w - marker_size, screen_y + screen_h)
        painter.drawLine(screen_x + screen_w, screen_y + screen_h, screen_x + screen_w, screen_y + screen_h - marker_size)

        # Draggable / lockable handle at the top-right corner (normal mode only):
        # a filled circle you can drag to move the box, or a lock outline when locked.
        if not is_3d_mode:
            hx = screen_x + screen_w
            hy = screen_y
            r = self._crop_handle_radius  # fixed screen size; does not scale with zoom
            if self._crop_locked:
                # Bold blue padlock at full opacity, over a dark halo so it stays
                # legible on any EM background.
                bw = int(r * 1.8)
                bh = int(r * 1.5)
                bx = hx - bw // 2
                by = hy - bh // 2 + int(r * 0.35)
                sw = int(bw * 0.58)
                sh = int(bh * 1.05)
                sx = hx - sw // 2
                sy = by - sh + scaled(3)

                def _draw_lock(color, width):
                    pen = QPen(color, width)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(bx, by, bw, bh, scaled(3), scaled(3))  # body
                    painter.drawArc(sx, sy, sw, sh, 0, 180 * 16)                   # shackle

                _draw_lock(QColor(0, 0, 0, 200), scaled(8))    # dark halo / outline
                _draw_lock(QColor(70, 170, 255), scaled(5))    # bright blue lock
            else:
                # Filled yellow circle drag handle, full opacity with a dark outline.
                painter.setBrush(QColor(255, 220, 50))
                painter.setPen(QPen(QColor(20, 20, 20), scaled(2)))
                painter.drawEllipse(hx - r, hy - r, 2 * r, 2 * r)

        # Draw hints based on mode
        if is_3d_mode:
            # 3D session hint
            hint_text = "3D Session Active\nNavigate slices and paint.\nClick ✓ to save, ✕ to cancel."
            hint_font = scaled_font(12, QFont.Weight.Bold)
            painter.setFont(hint_font)
            painter.setPen(QColor(50, 150, 255, 230))

            padding = scaled(10)
            line_height = scaled(20)
            text_x = screen_x + padding
            text_y = screen_y + padding + line_height
            for line in hint_text.split('\n'):
                painter.drawText(text_x, text_y, line)
                text_y += line_height
        elif self._tab_count < 2 and self.mask_alpha > 0:
            # Normal mode crop hint (only for first 2 captures)
            hint_text = "Once you are happy with\nthe segmentation inside\nthis box, press Tab to\nadd to the training set!"

            hint_font = scaled_font(12, QFont.Weight.Bold)
            painter.setFont(hint_font)
            painter.setPen(QColor(255, 220, 50, 230))

            # Draw text inside box at top-left with padding
            padding = scaled(10)
            line_height = scaled(20)
            text_x = screen_x + padding
            text_y = screen_y + padding + line_height
            for line in hint_text.split('\n'):
                painter.drawText(text_x, text_y, line)
                text_y += line_height

    def _draw_suggestion_hint(self, painter):
        """Draw suggestion acceptance hint."""
        if self._hovered_component is None:
            return
        if self._suggestion_accept_count >= 2:
            return
        if self.mask_alpha <= 0:
            return
        if self.cursor_pos is None:
            return

        hint_text = "Press spacebar to\naccept suggestion!"

        hint_font = scaled_font(12, QFont.Weight.Bold)
        painter.setFont(hint_font)
        painter.setPen(QColor(100, 255, 100, 230))

        # Draw below and to the right of cursor
        offset_x = scaled(35)
        offset_y = scaled(30)
        line_height = scaled(20)
        text_x = self.cursor_pos.x() + offset_x
        text_y = self.cursor_pos.y() + offset_y
        for line in hint_text.split('\n'):
            painter.drawText(text_x, text_y, line)
            text_y += line_height

    def _draw_3d_buttons(self, painter, box_x, box_y, box_w, box_h):
        """Draw save/cancel buttons for 3D Ground Truth mode."""
        btn_size = scaled(32)
        margin = scaled(8)

        # Green checkmark button (save)
        check_x = box_x + box_w - btn_size - margin
        check_y = box_y - btn_size - margin
        check_rect = QRect(check_x, check_y, btn_size, btn_size)
        self._3d_button_rects['check'] = check_rect

        painter.setBrush(QColor(50, 180, 50, 200))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawRoundedRect(check_rect, 4, 4)

        # Draw checkmark symbol
        painter.setPen(QPen(QColor(255, 255, 255), 3))
        cx, cy = check_x + btn_size // 2, check_y + btn_size // 2
        painter.drawLine(cx - 8, cy, cx - 2, cy + 6)
        painter.drawLine(cx - 2, cy + 6, cx + 8, cy - 6)

        # Red X button (cancel)
        x_x = check_x - btn_size - margin
        x_rect = QRect(x_x, check_y, btn_size, btn_size)
        self._3d_button_rects['x'] = x_rect

        painter.setBrush(QColor(180, 50, 50, 200))
        painter.drawRoundedRect(x_rect, 4, 4)

        # Draw X symbol
        xx, xy = x_x + btn_size // 2, check_y + btn_size // 2
        painter.drawLine(xx - 6, xy - 6, xx + 6, xy + 6)
        painter.drawLine(xx + 6, xy - 6, xx - 6, xy + 6)

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def set_tool(self, tool: str):
        """Set the current drawing tool."""
        self.current_tool = tool
        if tool == 'eraser':
            self.erasing = True
        else:
            self.erasing = False

    def set_brush_size(self, size: int):
        """Set brush size."""
        self.brush_size = max(1, min(200, size))
        self.update()

    def queue_directional_preload(self, direction: str):
        """Stub for directional preloading (not implemented)."""
        pass

    def is_slice_cached(self, slice_idx: int) -> bool:
        """Check if a slice is cached (for UI feedback)."""
        return False
