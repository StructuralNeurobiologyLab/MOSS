#!/usr/bin/env python3
"""
Optimized PaintCanvas subclass that caches the rendered image.
Only rebuilds the display when image/mask/alpha actually changes.
During drawing, updates the cache directly without full rebuild.
"""

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QImage, QPixmap, QPen, QColor, QPaintEvent, QBrush, QWheelEvent, QFont

from .paint_canvas import PaintCanvas
from PyQt6.QtCore import pyqtSignal, QEvent

from ..dpi_scaling import scaled, scaled_font


class OptimizedCanvas(PaintCanvas):
    """PaintCanvas with cached rendering for better performance."""

    # Signal emitted when Tab is pressed (capture crop)
    capture_requested = pyqtSignal()
    # Signal emitted when a suggestion component is accepted (spacebar)
    suggestion_accepted = pyqtSignal()
    # Signal emitted when brush size changes (for updating UI slider)
    brush_size_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable focus so we can receive key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Enable mouse tracking to get cursor position even without button press
        self.setMouseTracking(True)

        # Cached display pixmap (unscaled)
        self._cached_pixmap = None
        self._cache_valid = False

        # Crop preview box (for training snapshot)
        self._crop_preview_bounds = None  # (x, y, w, h) in image coordinates
        self._crop_preview_alpha = 0.6
        self._crop_preview_visible = True
        self._crop_size = 256  # Default crop size

        # Track painting bounds for auto-positioning crop box
        self._paint_bounds_min = None  # (x, y)
        self._paint_bounds_max = None  # (x, y)

        # Suggestion hover state (for accepting specific components)
        self._hovered_component = None  # Binary mask of hovered connected component
        self._labeled_suggestion = None  # Cached labeled array
        self._num_labels = 0

        # F key modifier state (for temporary fill mode)
        self._f_key_held = False

        # Floating hint state
        self._tab_count = 0  # How many times Tab has been pressed
        self._suggestion_accept_count = 0  # How many times suggestion accepted

    def set_image(self, image: np.ndarray):
        """Set the raw grayscale image to display."""
        super().set_image(image)
        self._cache_valid = False

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

    # Crop preview methods
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
        self._crop_preview_bounds = None
        self.update()

    def get_crop_bounds(self):
        """Get current crop preview bounds (x, y, w, h) in image coordinates."""
        return self._crop_preview_bounds

    def _update_crop_preview_from_bounds(self):
        """Update crop preview box to center on painted region."""
        if self._paint_bounds_min is None or self._paint_bounds_max is None:
            return
        if self.raw_image is None:
            return

        h, w = self.raw_image.shape
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

        if self.mask is None:
            return False

        # Store for undo
        self.mask_before_edit = self.mask.copy()

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

    def fill_at(self, pos: QPoint):
        """Flood fill at the given position."""
        if self.raw_image is None or self.mask is None:
            return

        from scipy import ndimage

        # Calculate image coordinates
        h, w = self.raw_image.shape
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
        # Create binary mask of pixels matching the clicked value
        if current_value > 127:
            # Clicking on white (mask) - find connected white region
            binary = (self.mask > 127)
        else:
            # Clicking on black (background) - find connected black region
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

        self.update()

    def draw_at(self, pos: QPoint):
        """Draw on the mask - viewport rendering handles display."""
        if self.raw_image is None or self.mask is None:
            return

        # Calculate image coordinates
        h, w = self.raw_image.shape
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
        # Apply to mask
        self.mask[y_min:y_max, x_min:x_max][circle_mask] = value

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

    def mousePressEvent(self, event):
        """Handle mouse press with button mappings: left=paint, right=erase, middle=pan."""
        # Right-click: temporarily switch to eraser
        if event.button() == Qt.MouseButton.RightButton:
            if self.mask is not None:
                self._right_click_erasing = True
                self._original_erasing = self.erasing
                self.erasing = True
                self.drawing = True
                self.mask_before_edit = self.mask.copy()
                self.draw_at(event.pos())
            return

        # Left-click with fill tool OR F key held (temporary fill)
        if event.button() == Qt.MouseButton.LeftButton and (self.current_tool == 'fill' or self._f_key_held):
            if self.mask is not None:
                self.mask_before_edit = self.mask.copy()
                self.fill_at(event.pos())
                self.emit_edit()
            return

        # Middle-click: pan (handled by parent)
        # Left-click: paint (default, handled by parent)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing and panning."""
        # Update cursor position for brush preview
        self.cursor_pos = event.pos()

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
            if self.show_suggestion and self.suggestion is not None and self.raw_image is not None:
                # Convert screen pos to image coordinates
                h, w = self.raw_image.shape
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
                # Only emit if there's a crop box to capture
                if self._crop_preview_bounds is not None:
                    self._tab_count += 1
                    self.capture_requested.emit()
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

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel - Shift+scroll changes brush size, otherwise zoom."""
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+scroll = change brush size
            delta = event.angleDelta().y()
            if delta > 0:
                # Scroll up = increase brush size
                new_size = min(100, self.brush_size + 2)
            else:
                # Scroll down = decrease brush size
                new_size = max(1, self.brush_size - 2)

            if new_size != self.brush_size:
                self.brush_size = new_size
                self.brush_size_changed.emit(new_size)
                self.update()  # Redraw to show new brush cursor
        else:
            # Normal scroll = zoom
            super().wheelEvent(event)

    def zoom_in(self, cursor_pos=None):
        """Zoom in at cursor position (uses tracked cursor if not provided)."""
        # Handle bool from button clicked signal
        if cursor_pos is None or isinstance(cursor_pos, bool):
            cursor_pos = getattr(self, 'cursor_pos', None)
        self._zoom_at_point(1.2, cursor_pos)

    def zoom_out(self, cursor_pos=None):
        """Zoom out at cursor position (uses tracked cursor if not provided)."""
        # Handle bool from button clicked signal
        if cursor_pos is None or isinstance(cursor_pos, bool):
            cursor_pos = getattr(self, 'cursor_pos', None)
        self._zoom_at_point(1 / 1.2, cursor_pos)

    def _zoom_at_point(self, factor: float, cursor_pos):
        """Zoom by factor, keeping the point under cursor stationary."""
        if self.raw_image is None:
            return

        old_zoom = self.zoom_level
        new_zoom = max(0.1, old_zoom * factor)

        if cursor_pos is not None:
            # Get image dimensions
            h, w = self.raw_image.shape

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
        self.update()
        self.viewport_changed.emit()

    def _rebuild_cache(self):
        """Rebuild the cached pixmap from numpy arrays."""
        if self.raw_image is None:
            self._cached_pixmap = None
            return

        h, w = self.raw_image.shape

        # Normalize raw image for display
        img_norm = self.raw_image.copy()
        img_min, img_max = img_norm.min(), img_norm.max()
        if img_max > img_min:
            img_norm = (img_norm - img_min) / (img_max - img_min)
        img_norm = (img_norm * 255 * self.image_alpha).astype(np.uint8)

        # Create RGB image
        img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

        # Overlay mask (red channel)
        if self.mask is not None:
            mask_overlay = (self.mask > 127).astype(np.float32)
            img_rgb[:, :, 0] = np.clip(
                img_rgb[:, :, 0] * (1 - mask_overlay * self.mask_alpha) +
                255 * mask_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Overlay suggestion (green channel) - uses mask_alpha for unified control
        if self.show_suggestion and self.suggestion is not None:
            sugg_overlay = (self.suggestion > 127).astype(np.float32)
            img_rgb[:, :, 1] = np.clip(
                img_rgb[:, :, 1] * (1 - sugg_overlay * self.mask_alpha) +
                255 * sugg_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Convert to QPixmap
        img_rgb = np.ascontiguousarray(img_rgb)
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._cached_pixmap = QPixmap.fromImage(qimg.copy())
        self._cache_valid = True

    def paintEvent(self, event: QPaintEvent):
        """Optimized render - only renders visible viewport region."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self.raw_image is None:
            return

        h, w = self.raw_image.shape

        # Calculate where the full image would be positioned
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()

        # Calculate visible region in IMAGE coordinates
        # Widget viewport is (0, 0) to (widget_width, widget_height)
        vis_left = max(0, -img_x)
        vis_top = max(0, -img_y)
        vis_right = min(scaled_w, self.width() - img_x)
        vis_bottom = min(scaled_h, self.height() - img_y)

        if vis_right <= vis_left or vis_bottom <= vis_top:
            return  # Nothing visible

        # Convert to image pixel coordinates
        src_left = int(vis_left / self.zoom_level)
        src_top = int(vis_top / self.zoom_level)
        src_right = min(w, int(np.ceil(vis_right / self.zoom_level)) + 1)
        src_bottom = min(h, int(np.ceil(vis_bottom / self.zoom_level)) + 1)

        # Extract and render only the visible region
        viewport_pixmap = self._render_region(src_left, src_top, src_right, src_bottom)

        if viewport_pixmap is None:
            return

        # Scale the viewport region
        dest_w = int((src_right - src_left) * self.zoom_level)
        dest_h = int((src_bottom - src_top) * self.zoom_level)

        scaled_pixmap = viewport_pixmap.scaled(
            dest_w, dest_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation
        )

        # Draw at correct position
        draw_x = img_x + int(src_left * self.zoom_level)
        draw_y = img_y + int(src_top * self.zoom_level)

        painter.drawPixmap(draw_x, draw_y, scaled_pixmap)

        # Draw brush cursor preview (lightweight)
        if self.cursor_pos and self.current_tool in ['brush', 'eraser']:
            scaled_radius = int(self.brush_size * self.zoom_level)
            pen = QPen(QColor(255, 255, 255, 180), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.cursor_pos, scaled_radius, scaled_radius)

        # Draw crop preview box (yellow dotted line)
        if (self._crop_preview_visible and
            self._crop_preview_bounds is not None and
            self._crop_preview_alpha > 0):

            crop_x, crop_y, crop_w, crop_h = self._crop_preview_bounds

            # Convert image coordinates to screen coordinates
            screen_x = img_x + int(crop_x * self.zoom_level)
            screen_y = img_y + int(crop_y * self.zoom_level)
            screen_w = int(crop_w * self.zoom_level)
            screen_h = int(crop_h * self.zoom_level)

            # Yellow dotted line with alpha
            alpha = int(255 * self._crop_preview_alpha)
            pen = QPen(QColor(255, 220, 50, alpha), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(screen_x, screen_y, screen_w, screen_h)

            # Draw corner markers for better visibility
            marker_size = 8
            painter.setPen(QPen(QColor(255, 220, 50, alpha), 3, Qt.PenStyle.SolidLine))
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

            # Draw crop hint text (only for first 2 captures, and if mask alpha > 0)
            if self._tab_count < 2 and self.mask_alpha > 0:
                hint_text = "Once you are happy with\nthe segmentation inside\nthis box, press Tab to\nadd to the training set!"

                hint_font = scaled_font(12, QFont.Weight.Bold)
                painter.setFont(hint_font)
                painter.setPen(QColor(255, 220, 50, 230))  # 0.9 alpha

                # Draw text inside box at top-left with padding
                padding = scaled(10)
                line_height = scaled(20)
                text_x = screen_x + padding
                text_y = screen_y + padding + line_height
                for line in hint_text.split('\n'):
                    painter.drawText(text_x, text_y, line)
                    text_y += line_height

        # Draw suggestion hover hint (only for first 2 acceptances, and if mask alpha > 0)
        if self._hovered_component is not None and self._suggestion_accept_count < 2 and self.mask_alpha > 0:
            if self.cursor_pos is not None:
                hint_text = "Press spacebar to\naccept suggestion!"

                hint_font = scaled_font(12, QFont.Weight.Bold)
                painter.setFont(hint_font)
                painter.setPen(QColor(100, 255, 100, 230))  # 0.9 alpha

                # Draw below and to the right of cursor, further away
                offset_x = scaled(35)
                offset_y = scaled(30)
                line_height = scaled(20)
                text_x = self.cursor_pos.x() + offset_x
                text_y = self.cursor_pos.y() + offset_y
                for line in hint_text.split('\n'):
                    painter.drawText(text_x, text_y, line)
                    text_y += line_height

    def _render_region(self, x1: int, y1: int, x2: int, y2: int) -> QPixmap:
        """Render a specific region of the image to a pixmap."""
        if self.raw_image is None:
            return None

        # Extract region from raw image
        region = self.raw_image[y1:y2, x1:x2].copy()
        h, w = region.shape

        # Normalize for display
        img_min, img_max = self.raw_image.min(), self.raw_image.max()
        if img_max > img_min:
            region = (region - img_min) / (img_max - img_min)
        region = (region * 255 * self.image_alpha).astype(np.uint8)

        # Create RGB
        img_rgb = np.stack([region, region, region], axis=-1)

        # Overlay mask (red channel)
        if self.mask is not None:
            mask_region = self.mask[y1:y2, x1:x2]
            mask_overlay = (mask_region > 127).astype(np.float32)
            img_rgb[:, :, 0] = np.clip(
                img_rgb[:, :, 0] * (1 - mask_overlay * self.mask_alpha) +
                255 * mask_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Overlay suggestion (green channel) - uses mask_alpha for unified control
        if self.show_suggestion and self.suggestion is not None:
            sugg_region = self.suggestion[y1:y2, x1:x2]
            sugg_overlay = (sugg_region > 127).astype(np.float32)
            img_rgb[:, :, 1] = np.clip(
                img_rgb[:, :, 1] * (1 - sugg_overlay * self.mask_alpha) +
                255 * sugg_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Highlight hovered component with subtle red checkerboard (preview of acceptance)
        if self._hovered_component is not None:
            hover_region = self._hovered_component[y1:y2, x1:x2]
            if hover_region.any():
                # Create checkerboard pattern for texture
                yy, xx = np.ogrid[:h, :w]
                # Offset by region position for consistent pattern
                checker = ((yy + y1 + xx + x1) % 3 < 1)  # Sparse pattern
                # Red checkerboard pixels (preview of what mask will look like)
                red_pixels = hover_region & checker
                # Apply red overlay with same alpha as mask (unified control)
                img_rgb[red_pixels, 0] = np.clip(
                    img_rgb[red_pixels, 0] * (1 - self.mask_alpha) +
                    255 * self.mask_alpha, 0, 255
                ).astype(np.uint8)
                img_rgb[red_pixels, 1] = (img_rgb[red_pixels, 1] * (1 - self.mask_alpha)).astype(np.uint8)
                img_rgb[red_pixels, 2] = (img_rgb[red_pixels, 2] * (1 - self.mask_alpha)).astype(np.uint8)

        # Convert to QPixmap
        img_rgb = np.ascontiguousarray(img_rgb)
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())
