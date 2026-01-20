#!/usr/bin/env python3
"""
Interactive painting canvas for annotation tool.
Handles image display, zoom, pan, and drawing operations.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QImage, QPixmap, QPen, QColor,
    QWheelEvent, QMouseEvent, QPaintEvent
)


class PaintCanvas(QWidget):
    """Custom widget for displaying and editing images with masks."""

    # Signals
    edit_made = pyqtSignal(np.ndarray, np.ndarray)  # (before_mask, after_mask)
    viewport_changed = pyqtSignal()  # Emitted when zoom or pan changes

    def __init__(self, parent=None):
        super().__init__(parent)

        # Image data
        self.raw_image = None  # Grayscale numpy array [H, W]
        self.mask = None       # Binary mask numpy array [H, W] (0 or 255)
        self.suggestion = None # AI suggestion numpy array [H, W] (0 or 255)

        # Display state
        self.zoom_level = 1.0
        self.offset = QPoint(0, 0)
        self.last_mouse_pos = None
        self.cursor_pos = None  # Track cursor for brush preview

        # Drawing state
        self.drawing = False
        self.erasing = False
        self.panning = False  # Hand tool active
        self.brush_size = 10
        self.mask_before_edit = None  # Store mask state before editing
        self.current_tool = 'brush'  # 'brush', 'eraser', or 'hand'

        # Display settings
        self.show_suggestion = False
        self.image_alpha = 1.0  # Base image opacity
        self.mask_alpha = 0.5
        self.suggestion_alpha = 0.3

        # Mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image: np.ndarray):
        """Set the raw grayscale image to display."""
        if image.ndim == 3:
            image = image.mean(axis=-1)  # Convert to grayscale
        self.raw_image = image.astype(np.float32)
        if self.mask is None or self.mask.shape != image.shape:
            # Initialize empty mask
            self.mask = np.zeros(image.shape, dtype=np.uint8)
        self.update()

    def set_mask(self, mask: np.ndarray):
        """Set the segmentation mask."""
        self.mask = mask.astype(np.uint8)
        self.update()

    def set_suggestion(self, suggestion: np.ndarray):
        """Set the AI suggestion mask."""
        self.suggestion = suggestion.astype(np.uint8) if suggestion is not None else None
        self.update()

    def accept_suggestion(self):
        """Accept the current suggestion as the mask."""
        if self.suggestion is not None:
            self.mask_before_edit = self.mask.copy()
            self.mask = self.suggestion.copy()
            self.emit_edit()
            self.suggestion = None
            self.update()

    def set_brush_size(self, size: int):
        """Set the brush size for drawing."""
        self.brush_size = max(1, size)

    def set_tool(self, tool: str):
        """Set the active tool ('brush', 'eraser', or 'hand')."""
        self.current_tool = tool
        self.erasing = (tool == 'eraser')
        self.panning = (tool == 'hand')

        # Update cursor based on tool
        if tool == 'hand':
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def toggle_suggestion_visibility(self, visible: bool):
        """Toggle suggestion overlay visibility."""
        self.show_suggestion = visible
        self.update()

    def set_image_alpha(self, alpha: float):
        """Set the base image alpha (0.0 to 1.0)."""
        self.image_alpha = max(0.0, min(1.0, alpha))
        self.update()

    def set_mask_alpha(self, alpha: float):
        """Set the mask overlay alpha (0.0 to 1.0)."""
        self.mask_alpha = max(0.0, min(1.0, alpha))
        self.update()

    def set_suggestion_alpha(self, alpha: float):
        """Set the suggestion overlay alpha (0.0 to 1.0)."""
        self.suggestion_alpha = max(0.0, min(1.0, alpha))
        self.update()

    def zoom_in(self, cursor_pos=None):
        """Zoom in by 20% at cursor position."""
        old_zoom = self.zoom_level
        self.zoom_level *= 1.2

        # Adjust offset to zoom at cursor position
        if cursor_pos and self.raw_image is not None:
            # Calculate the position in widget coordinates relative to center
            center_x = self.width() / 2
            center_y = self.height() / 2
            dx = cursor_pos.x() - center_x
            dy = cursor_pos.y() - center_y

            # Adjust offset to keep point under cursor
            zoom_ratio = self.zoom_level / old_zoom - 1
            self.offset.setX(int(self.offset.x() - dx * zoom_ratio))
            self.offset.setY(int(self.offset.y() - dy * zoom_ratio))

        self.update()
        self.viewport_changed.emit()

    def zoom_out(self, cursor_pos=None):
        """Zoom out by 20% at cursor position."""
        old_zoom = self.zoom_level
        self.zoom_level = max(0.1, self.zoom_level / 1.2)

        # Adjust offset to zoom at cursor position
        if cursor_pos and self.raw_image is not None:
            # Calculate the position in widget coordinates relative to center
            center_x = self.width() / 2
            center_y = self.height() / 2
            dx = cursor_pos.x() - center_x
            dy = cursor_pos.y() - center_y

            # Adjust offset to keep point under cursor
            zoom_ratio = self.zoom_level / old_zoom - 1
            self.offset.setX(int(self.offset.x() - dx * zoom_ratio))
            self.offset.setY(int(self.offset.y() - dy * zoom_ratio))

        self.update()
        self.viewport_changed.emit()

    def reset_view(self):
        """Reset zoom and pan to default."""
        self.zoom_level = 1.0
        self.offset = QPoint(0, 0)
        self.update()
        self.viewport_changed.emit()

    def get_viewport_bounds(self) -> tuple:
        """Calculate visible portion of image in pixel coordinates.

        Returns:
            (img_left, img_top, img_right, img_bottom) in image pixel coordinates,
            or None if no image is loaded.
        """
        if self.raw_image is None:
            return None

        h, w = self.raw_image.shape
        widget_w, widget_h = self.width(), self.height()

        # Scaled image dimensions
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)

        # Image top-left corner position in widget coordinates
        img_x = (widget_w - scaled_w) // 2 + self.offset.x()
        img_y = (widget_h - scaled_h) // 2 + self.offset.y()

        # Visible widget bounds
        widget_left, widget_top = 0, 0
        widget_right, widget_bottom = widget_w, widget_h

        # Calculate intersection of widget viewport and scaled image
        visible_left = max(widget_left, img_x)
        visible_top = max(widget_top, img_y)
        visible_right = min(widget_right, img_x + scaled_w)
        visible_bottom = min(widget_bottom, img_y + scaled_h)

        # Convert to image pixel coordinates
        img_left = int((visible_left - img_x) / self.zoom_level)
        img_top = int((visible_top - img_y) / self.zoom_level)
        img_right = int((visible_right - img_x) / self.zoom_level)
        img_bottom = int((visible_bottom - img_y) / self.zoom_level)

        # Clamp to image bounds
        img_left = max(0, min(img_left, w))
        img_top = max(0, min(img_top, h))
        img_right = max(0, min(img_right, w))
        img_bottom = max(0, min(img_bottom, h))

        return (img_left, img_top, img_right, img_bottom)

    def paintEvent(self, event: QPaintEvent):
        """Render the canvas."""
        if self.raw_image is None:
            return

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))  # Dark background

        # Convert numpy arrays to QImage
        h, w = self.raw_image.shape

        # Normalize raw image for display
        img_norm = self.raw_image.copy()
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
        img_norm = (img_norm * 255 * self.image_alpha).astype(np.uint8)  # Apply image alpha

        # Create RGB image
        img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1).copy()

        # Overlay mask (red channel)
        if self.mask is not None:
            mask_overlay = (self.mask > 127).astype(np.uint8)
            img_rgb[:, :, 0] = np.clip(
                img_rgb[:, :, 0] * (1 - mask_overlay * self.mask_alpha) +
                255 * mask_overlay * self.mask_alpha, 0, 255
            ).astype(np.uint8)

        # Overlay suggestion (green channel)
        if self.show_suggestion and self.suggestion is not None:
            sugg_overlay = (self.suggestion > 127).astype(np.uint8)
            img_rgb[:, :, 1] = np.clip(
                img_rgb[:, :, 1] * (1 - sugg_overlay * self.suggestion_alpha) +
                255 * sugg_overlay * self.suggestion_alpha, 0, 255
            ).astype(np.uint8)

        # Convert to QImage
        qimg = QImage(
            img_rgb.data, w, h, 3 * w,
            QImage.Format.Format_RGB888
        )

        # Apply zoom and offset
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)
        pixmap = QPixmap.fromImage(qimg).scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Center the image
        x = (self.width() - pixmap.width()) // 2 + self.offset.x()
        y = (self.height() - pixmap.height()) // 2 + self.offset.y()

        painter.drawPixmap(x, y, pixmap)

        # Draw brush cursor preview
        if self.cursor_pos and self.current_tool in ['brush', 'eraser']:
            # Draw circle showing brush size
            scaled_radius = int(self.brush_size * self.zoom_level)
            pen = QPen(QColor(255, 255, 255, 180), 2)  # White with transparency
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.cursor_pos, scaled_radius, scaled_radius)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for drawing and panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == 'hand':
                # Start panning with hand tool
                self.last_mouse_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                # Start drawing (only if mask exists)
                if self.mask is not None:
                    self.drawing = True
                    self.mask_before_edit = self.mask.copy()
                    self.draw_at(event.pos())
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Start panning (always available)
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
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

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing:
                self.drawing = False
                self.emit_edit()
            elif self.last_mouse_pos is not None:
                # End panning with hand tool
                self.last_mouse_pos = None
                self.setCursor(Qt.CursorShape.OpenHandCursor if self.current_tool == 'hand' else Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.last_mouse_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.current_tool == 'hand' else Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming at cursor position."""
        cursor_pos = event.position().toPoint()
        if event.angleDelta().y() > 0:
            self.zoom_in(cursor_pos)
        else:
            self.zoom_out(cursor_pos)

    def draw_at(self, pos: QPoint):
        """Draw on the mask at the given position."""
        if self.raw_image is None or self.mask is None:
            return

        # Convert screen coordinates to image coordinates
        h, w = self.raw_image.shape
        scaled_w = int(w * self.zoom_level)
        scaled_h = int(h * self.zoom_level)

        # Image position on screen
        img_x = (self.width() - scaled_w) // 2 + self.offset.x()
        img_y = (self.height() - scaled_h) // 2 + self.offset.y()

        # Mouse position relative to image
        rel_x = pos.x() - img_x
        rel_y = pos.y() - img_y

        # Check if click is within image bounds
        if rel_x < 0 or rel_y < 0 or rel_x >= scaled_w or rel_y >= scaled_h:
            return

        # Convert to image pixel coordinates
        px = int(rel_x / self.zoom_level)
        py = int(rel_y / self.zoom_level)

        # Draw circle on mask
        value = 0 if self.erasing else 255
        radius = self.brush_size

        y_min = max(0, py - radius)
        y_max = min(h, py + radius + 1)
        x_min = max(0, px - radius)
        x_max = min(w, px + radius + 1)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (x - px) ** 2 + (y - py) ** 2 <= radius ** 2:
                    self.mask[y, x] = value

        self.update()

    def emit_edit(self):
        """Emit signal that an edit was made."""
        if self.mask_before_edit is not None:
            self.edit_made.emit(self.mask_before_edit, self.mask.copy())
            self.mask_before_edit = None
