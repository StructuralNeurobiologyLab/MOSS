#!/usr/bin/env python3
"""
Interactive training page - follows annotation tool structure with sliding window loading.
Keeps ~200 images in memory, loads more when user approaches the edge.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QToolBar, QSlider, QFileDialog, QSizePolicy, QMessageBox,
    QProgressBar, QCheckBox, QComboBox
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import QApplication

from ..project_config import save_project_config, load_project_config, make_relative_path
from ..dpi_scaling import scaled, scaled_font


def _load_single_image(args):
    """Load a single image (for parallel loading)."""
    # Support both old format (idx, path, mask_path) and new format with max_size
    if len(args) == 4:
        idx, image_path, mask_path, max_size = args
    else:
        idx, image_path, mask_path = args
        max_size = None

    try:
        img = np.array(Image.open(image_path))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32)

        if mask_path and mask_path.exists():
            mask = np.array(Image.open(mask_path))
            mask = mask.astype(np.uint8)
        else:
            mask = np.zeros(img.shape, dtype=np.uint8)

        # Pad to max_size if provided and image is smaller
        if max_size is not None:
            max_h, max_w = max_size
            h, w = img.shape
            if h < max_h or w < max_w:
                pad_h = max_h - h
                pad_w = max_w - w
                img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        return idx, img, mask, None
    except Exception as e:
        return idx, None, None, str(e)

# Import OptimizedCanvas - subclass of PaintCanvas with caching
try:
    from ..widgets.optimized_canvas import OptimizedCanvas
except ImportError:
    OptimizedCanvas = None

# Use our LoadingDialog which has set_progress method
from ..loading_dialog import LoadingDialog


class InteractiveTrainingPage(QWidget):
    """Interactive training page with sliding window image loading."""

    # Signals
    training_complete = pyqtSignal(str)  # checkpoint_path
    architecture_changed = pyqtSignal(str)  # architecture id - emitted when user changes architecture
    training_started = pyqtSignal()  # emitted when training starts
    training_stopped = pyqtSignal()  # emitted when training stops
    loss_updated = pyqtSignal(float, int)  # loss_value, batch_number - forwarded from train_worker

    # Sliding window parameters
    WINDOW_SIZE = 200  # Keep this many images in memory
    LOAD_THRESHOLD = 30  # Load more when within this many images of edge
    BATCH_SIZE = 100  # Load this many images at a time

    def __init__(self):
        super().__init__()

        # Application state
        self.current_slice_index = 0
        self.current_tool = 'brush'
        self.image_source = None

        # Image file list (always complete)
        self.image_files = []  # List of Path objects for ALL images

        # Sliding window of loaded images
        self.images = {}  # {index: numpy array} - sparse dict for loaded images
        self.masks = {}   # {index: numpy array} - sparse dict for masks
        self.window_start = 0  # Start index of currently loaded window
        self.window_end = 0    # End index of currently loaded window
        self._loading_in_progress = False  # Prevent concurrent loads
        self._max_image_size = None  # (height, width) - max dimensions for padding

        # Config from wizard
        self.config = {}
        self.project_dir = None
        self.train_images_dir = None
        self.train_masks_dir = None
        self.train_images_25d_dir = None  # For 2.5D training data
        self.train_masks_25d_dir = None   # For 2.5D training masks
        self.masks_dir = None
        self.user_training_masks_dir = None  # User-provided training masks (optional)
        self._pending_images_dir = None  # For lazy loading when page becomes visible

        # Training state
        self.train_worker = None
        self.edit_count = 0
        self.current_architecture = 'unet'  # Default architecture
        self._arch_name_to_id = {}  # Maps display name -> architecture id
        self._arch_id_to_name = {}  # Maps architecture id -> display name
        self._architecture_locked = False  # True when in multi-user session

        # Prediction state
        self.predict_worker = None
        self.show_predictions = False
        self.prediction_architecture = 'unet'  # Which architecture to use for predictions
        self._pred_name_to_id = {}  # Maps short name -> architecture id (for predictions)
        self._pred_id_to_name = {}  # Maps architecture id -> short name (for predictions)
        self._init_predict_worker()

        # Prediction debounce timer - delays predictions while navigating
        self._prediction_debounce_timer = QTimer()
        self._prediction_debounce_timer.setSingleShot(True)
        self._prediction_debounce_timer.timeout.connect(self._do_delayed_prediction)
        self._prediction_delay_ms = 300  # Wait 300ms after last navigation before predicting

        # Refiner state (learns from user edits)
        self.refiner_worker = None
        self.refiner_mode = False  # When True, refiner handles training/prediction
        self._refiner_debounce_timer = QTimer()
        self._refiner_debounce_timer.setSingleShot(True)
        self._refiner_debounce_timer.timeout.connect(self._do_refiner_prediction)
        self._refiner_mask_at_load = {}  # {slice_idx: mask} - tracks mask state when slice was loaded
        self.refiner_images_dir = None
        self.refiner_masks_before_dir = None
        self.refiner_masks_after_dir = None

        # Undo history
        self.undo_stack = []  # List of (idx, mask_before) tuples
        self.max_undo = 50

        # Auto-save timer (save config every 30 seconds)
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save_config)
        self.auto_save_timer.start(30000)  # 30 seconds

        # Status message timer (for temporary messages)
        self._status_timer = QTimer()
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self._restore_status_message)
        self._last_progress_message = ""

        # Multi-user collaborative training state
        self._sync_client = None  # SyncClient for sending/receiving weights
        self._aggregation_server = None  # AggregationServer (if hosting)
        self._multi_user_enabled = False
        self._is_host = False  # Whether this user is hosting the session
        self._sync_interval_epochs = 5  # Push weights every N epochs
        self._blend_ratio = 0.5  # How much to blend global model (0=local, 1=global)
        self._last_sync_epoch = 0
        self._pending_global_weights = None  # Weights received while training
        self._has_received_global_model = False  # Whether joinee has received model from host

        # Timer for periodic model check (joinee only)
        self._model_check_timer = QTimer(self)
        self._model_check_timer.timeout.connect(self._check_for_global_model)
        self._model_check_attempts = 0

        # Cooldown for model sharing (host only) - prevent spamming
        self._last_model_share_time = 0.0
        self._model_share_cooldown = 30.0  # seconds between model shares

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Set strong focus policy so page can receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        main_layout = QVBoxLayout(self)
        margin = scaled(5)
        main_layout.setContentsMargins(margin, margin, margin, margin)
        main_layout.setSpacing(scaled(5))

        if OptimizedCanvas is None:
            error_label = QLabel(
                "Error: annotation_tool package not found.\n\n"
                "Please install it with:\n"
                "pip install -e /path/to/annotation_tool_package"
            )
            error_label.setFont(scaled_font(14))
            error_label.setStyleSheet("color: red;")
            main_layout.addWidget(error_label)
            return

        # Create canvas FIRST - using OptimizedCanvas for better performance
        self.canvas = OptimizedCanvas()
        self.canvas.edit_made.connect(self.on_edit_made)
        self.canvas.capture_requested.connect(self.capture_crop_for_training)
        self.canvas.suggestion_accepted.connect(self.on_suggestion_accepted)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(scaled(600), scaled(400))

        # Create toolbar
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)

        # Add canvas
        main_layout.addWidget(self.canvas)

        # Create bottom controls
        controls = self.create_controls()
        main_layout.addWidget(controls)

        # Status label
        self.status_label = QLabel("Load images to begin.")
        main_layout.addWidget(self.status_label)

        # Connect viewport changes to prediction updates
        self.canvas.viewport_changed.connect(self._on_viewport_changed)

        # Connect brush size changes from canvas (Shift+scroll) to update slider
        self.canvas.brush_size_changed.connect(self._on_canvas_brush_size_changed)

    def _on_canvas_brush_size_changed(self, size):
        """Update slider when brush size is changed via Shift+scroll."""
        self.brush_slider.blockSignals(True)  # Prevent feedback loop
        self.brush_slider.setValue(size)
        self.brush_slider.blockSignals(False)
        self.brush_size_label.setText(str(size))

    def _init_predict_worker(self):
        """Initialize the viewport prediction worker."""
        from ..workers.viewport_predict_worker import ViewportPredictWorker

        self.predict_worker = ViewportPredictWorker()
        self.predict_worker.prediction_ready.connect(self._on_prediction_ready)
        self.predict_worker.start()

    def save_project(self):
        """Manually save the project (all masks and config)."""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "No project directory set.")
            return

        # Save current slice mask
        self.save_current_slice()

        # Save all modified masks to disk
        saved_count = 0
        for idx, mask in self.masks.items():
            if mask is not None:
                mask_path = self.masks_dir / f"mask_{idx:05d}.tif"
                try:
                    Image.fromarray(mask).save(mask_path, compression='tiff_lzw')
                    saved_count += 1
                except Exception as e:
                    print(f"Failed to save mask {idx}: {e}")

        # Save project config
        self._save_project_config()

        self.status_label.setText(f"Project saved! ({saved_count} masks)")

    def _open_training_reviewer(self):
        """Open the training data reviewer popup."""
        if not self.train_images_dir or not self.train_images_dir.exists():
            QMessageBox.warning(
                self, "No Training Data",
                "No training images directory found. Create some crops first."
            )
            return

        # Check if there are any training images
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        train_images = [
            f for f in self.train_images_dir.iterdir()
            if f.suffix.lower() in extensions
        ]
        if not train_images:
            QMessageBox.information(
                self, "No Training Images",
                "No training images found to review."
            )
            return

        # Stop training if running
        was_training = False
        if self.train_worker and self.train_worker.isRunning():
            was_training = True
            self.train_worker.stop()
            if not self.train_worker.wait(3000):
                self.train_worker.terminate()
                self.train_worker.wait(1000)
            self.train_btn.setText("Start Training")
            self.train_progress.setVisible(False)
            self.arch_combo.setEnabled(True)
            if self.predict_worker:
                self.predict_worker.set_training_active(False)

        # Turn off predictions
        was_showing_predictions = self.show_predictions
        if self.show_predictions:
            self.show_pred_check.setChecked(False)

        # Save current work
        self.save_current_slice()

        # Open reviewer dialog
        from ..widgets.training_data_reviewer import TrainingDataReviewer
        reviewer = TrainingDataReviewer(
            self.train_images_dir,
            self.train_masks_dir,
            self.project_dir,
            parent=self
        )

        # Connect signal to reload data if modified
        reviewer.data_modified.connect(self._on_training_data_modified)

        # Show dialog (blocks until closed)
        reviewer.exec()

        # Report results
        if reviewer.was_modified():
            self.status_label.setText(
                f"Reviewer: Discarded {reviewer.discard_count} crops. "
                f"{len(reviewer.image_files)} remaining."
            )

    def _on_training_data_modified(self):
        """Handle when training data was modified by the reviewer."""
        # Clear any cached training data in the worker
        if self.train_worker:
            self.train_worker = None

        # The training will reload data when started again
        print("[Reviewer] Training data modified - will reload on next training start")

    def on_show_predictions_changed(self, state):
        """Handle Show Predictions checkbox toggle."""
        self.show_predictions = (state == 2)  # Qt.CheckState.Checked
        self.canvas.toggle_suggestion_visibility(self.show_predictions)

        if self.show_predictions:
            self._request_viewport_prediction()
        else:
            self.canvas.set_suggestion(None)

    def _populate_architecture_combo(self):
        """Populate the architecture dropdown with available architectures."""
        from ..models.unet import get_available_architectures

        self.arch_combo.blockSignals(True)
        self.arch_combo.clear()

        # Get available architectures
        architectures = get_available_architectures()

        # Store mapping of display name -> architecture id
        self._arch_name_to_id = {}
        self._arch_id_to_name = {}

        for arch_id, display_name in architectures.items():
            self.arch_combo.addItem(display_name)
            self._arch_name_to_id[display_name] = arch_id
            self._arch_id_to_name[arch_id] = display_name

        # Select current architecture
        if self.current_architecture in self._arch_id_to_name:
            idx = self.arch_combo.findText(self._arch_id_to_name[self.current_architecture])
            if idx >= 0:
                self.arch_combo.setCurrentIndex(idx)

        self.arch_combo.blockSignals(False)

    def on_architecture_changed(self, display_name: str):
        """Handle architecture selection change."""
        if display_name not in self._arch_name_to_id:
            return

        arch_id = self._arch_name_to_id[display_name]
        if arch_id == self.current_architecture:
            return

        # Check if architecture is locked (multi-user session)
        if self._architecture_locked:
            QMessageBox.warning(
                self, "Architecture Locked",
                "Architecture cannot be changed during a multi-user session."
            )
            # Revert selection
            self.arch_combo.blockSignals(True)
            idx = self.arch_combo.findText(self._arch_id_to_name[self.current_architecture])
            if idx >= 0:
                self.arch_combo.setCurrentIndex(idx)
            self.arch_combo.blockSignals(False)
            return

        # Check if training is running
        if self.train_worker is not None and self.train_worker.isRunning():
            QMessageBox.warning(
                self, "Training in Progress",
                "Please stop training before changing architecture."
            )
            # Revert selection
            self.arch_combo.blockSignals(True)
            idx = self.arch_combo.findText(self._arch_id_to_name[self.current_architecture])
            if idx >= 0:
                self.arch_combo.setCurrentIndex(idx)
            self.arch_combo.blockSignals(False)
            return

        self.current_architecture = arch_id
        self._show_temp_status(f"Switched to {display_name}")

        # Notify listeners (e.g., setup page for multi-user)
        self.architecture_changed.emit(arch_id)

        # Check if we have pending weights that can now be applied
        self._try_apply_pending_weights()

        # Update prediction worker with new architecture
        if self.predict_worker:
            self.predict_worker.set_architecture(arch_id)
            # Update checkpoint path for new architecture
            checkpoint_path = self._get_checkpoint_path()
            if checkpoint_path and checkpoint_path.exists():
                self.predict_worker.set_checkpoint(str(checkpoint_path))

        # Update status to show training state for this architecture
        self._update_architecture_status()

        # Request new prediction with new architecture
        if self.show_predictions:
            self._request_viewport_prediction()

    def _get_checkpoint_path(self):
        """Get the checkpoint path for the current architecture."""
        from ..models.unet import get_checkpoint_filename

        if not self.project_dir:
            return None

        filename = get_checkpoint_filename(self.current_architecture)
        return self.project_dir / filename

    def _update_architecture_status(self):
        """Update status to show training state for current architecture."""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path and checkpoint_path.exists():
            # Try to get epoch info from checkpoint
            try:
                import torch
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                epoch = ckpt.get('epoch', 0) + 1
                self._show_temp_status(f"Loaded {self._arch_id_to_name[self.current_architecture]}: Epoch {epoch}")
            except Exception:
                self._show_temp_status(f"Model available: {self._arch_id_to_name[self.current_architecture]}")
        else:
            self._show_temp_status(f"No trained model for {self._arch_id_to_name[self.current_architecture]}")

    def lock_architecture(self, architecture: str):
        """
        Lock the architecture dropdown (used during multi-user session).

        Args:
            architecture: Architecture ID to lock to. If different from current,
                         switches to it first.
        """
        print(f"[Training] Locking architecture to: {architecture}")
        self._architecture_locked = True

        # If we need to switch architecture
        if architecture and architecture != self.current_architecture:
            if architecture in self._arch_id_to_name:
                # Switch to the required architecture
                self.current_architecture = architecture
                self.arch_combo.blockSignals(True)
                idx = self.arch_combo.findText(self._arch_id_to_name[architecture])
                if idx >= 0:
                    self.arch_combo.setCurrentIndex(idx)
                self.arch_combo.blockSignals(False)
                self._show_temp_status(f"Architecture set to {self._arch_id_to_name[architecture]} (locked)")
                # Check if we have pending weights that can now be applied
                self._try_apply_pending_weights()
            else:
                print(f"[Training] Warning: Unknown architecture {architecture}")

        # Style the dropdown to show it's locked (red text, disabled look)
        self.arch_combo.setEnabled(False)
        self.arch_combo.setStyleSheet("""
            QComboBox {
                color: #cc0000;
                background-color: #ffeeee;
            }
            QComboBox:disabled {
                color: #cc0000;
                background-color: #ffeeee;
            }
        """)
        self.arch_combo.setToolTip("Architecture locked during multi-user session")

    def unlock_architecture(self):
        """Unlock the architecture dropdown (when leaving multi-user session)."""
        print("[Training] Unlocking architecture")
        self._architecture_locked = False

        # Reset styling
        self.arch_combo.setEnabled(True)
        self.arch_combo.setStyleSheet("")
        self.arch_combo.setToolTip("Select model architecture")

    def _populate_prediction_model_combo(self):
        """Populate the prediction model dropdown with available trained models."""
        from ..models.unet import get_available_architectures, get_checkpoint_filename

        self.pred_model_combo.blockSignals(True)
        self.pred_model_combo.clear()

        # Store mapping for prediction models
        self._pred_name_to_id = {}
        self._pred_id_to_name = {}

        architectures = get_available_architectures()

        for arch_id, display_name in architectures.items():
            # Use shorter names for the smaller dropdown
            short_name = display_name.replace('UNet ', '').replace('(', '').replace(')', '')
            self.pred_model_combo.addItem(short_name)
            self._pred_name_to_id[short_name] = arch_id
            self._pred_id_to_name[arch_id] = short_name

        # Select current prediction architecture
        if self.prediction_architecture in self._pred_id_to_name:
            idx = self.pred_model_combo.findText(self._pred_id_to_name[self.prediction_architecture])
            if idx >= 0:
                self.pred_model_combo.setCurrentIndex(idx)

        self.pred_model_combo.blockSignals(False)

    def on_prediction_model_changed(self, short_name: str):
        """Handle prediction model selection change."""
        if short_name not in self._pred_name_to_id:
            return

        arch_id = self._pred_name_to_id[short_name]
        if arch_id == self.prediction_architecture:
            return

        self.prediction_architecture = arch_id

        # Update prediction worker
        if self.predict_worker:
            self.predict_worker.set_architecture(arch_id)
            checkpoint_path = self._get_prediction_checkpoint_path()
            if checkpoint_path and checkpoint_path.exists():
                self.predict_worker.set_checkpoint(str(checkpoint_path))
                self._show_temp_status(f"Predictions: {short_name}")
            else:
                self._show_temp_status(f"No trained model for {short_name}")

        # Request new prediction if showing
        if self.show_predictions:
            self._request_viewport_prediction()

    def _get_prediction_checkpoint_path(self):
        """Get the checkpoint path for the prediction architecture.

        When training is active and prediction architecture matches training,
        returns the snapshot checkpoint for real-time preview.
        """
        from ..models.unet import get_checkpoint_filename

        if not self.project_dir:
            return None

        filename = get_checkpoint_filename(self.prediction_architecture)
        checkpoint_path = self.project_dir / filename

        # Use snapshot checkpoint if training is active and architectures match
        training_active = (hasattr(self, 'train_worker') and
                          self.train_worker is not None and
                          self.train_worker.isRunning())

        if training_active and self.prediction_architecture == self.current_architecture:
            snapshot_path = self.project_dir / filename.replace('.pth', '_snapshot.pth')
            if snapshot_path.exists():
                return snapshot_path

        return checkpoint_path

    def _on_viewport_changed(self):
        """Handle viewport pan/zoom changes."""
        if self.show_predictions:
            self._request_viewport_prediction()
        # Also trigger refiner prediction on viewport change
        if self.refiner_mode:
            self._refiner_debounce_timer.start(300)  # 300ms debounce

    def _request_viewport_prediction(self, immediate=False):
        """Request prediction for the current viewport (debounced by default)."""
        if not self.show_predictions or self.predict_worker is None:
            return

        if immediate:
            # Skip debounce, predict now
            self._do_delayed_prediction()
        else:
            # Debounce: restart timer on each request, only predict after delay
            self._prediction_debounce_timer.start(self._prediction_delay_ms)

    def _do_delayed_prediction(self):
        """Actually perform the prediction (called after debounce delay)."""
        if not self.show_predictions or self.predict_worker is None:
            return

        idx = self.current_slice_index
        if idx not in self.images:
            return

        # Check for checkpoint (prediction architecture-specific)
        checkpoint_path = self._get_prediction_checkpoint_path()
        if not checkpoint_path or not checkpoint_path.exists():
            return

        # Set checkpoint and architecture if changed
        self.predict_worker.set_architecture(self.prediction_architecture)
        self.predict_worker.set_checkpoint(str(checkpoint_path))

        # Tell predict worker whether training is active (for GPU/CPU decision)
        training_active = hasattr(self, 'train_worker') and self.train_worker is not None and self.train_worker.isRunning()
        self.predict_worker.set_training_active(training_active)

        # Get viewport bounds
        bounds = self.canvas.get_viewport_bounds()
        if bounds is None:
            return

        # For 2.5D architectures, pass a 3-channel stack (z-3, z, z+3)
        if self.predict_worker.is_25d():
            adjacent = self._load_adjacent_slices(idx)
            if adjacent is not None and len(adjacent) == 3:
                # Stack as (H, W, 3)
                image_stack = np.stack(adjacent, axis=-1)
                self.predict_worker.request_prediction(image_stack, bounds)
            else:
                # Fall back to single slice if adjacent loading fails
                self.predict_worker.request_prediction(self.images[idx], bounds)
        else:
            # Regular 2D prediction
            self.predict_worker.request_prediction(self.images[idx], bounds)

    def _on_prediction_ready(self, prediction: np.ndarray, bounds: tuple):
        """Handle prediction result from worker."""
        if not self.show_predictions:
            return

        idx = self.current_slice_index
        if idx not in self.images:
            return

        # Create full-size suggestion array
        h, w = self.images[idx].shape
        suggestion = np.zeros((h, w), dtype=np.uint8)

        # Place prediction in correct location
        x_min, y_min, x_max, y_max = bounds
        pred_h, pred_w = prediction.shape

        # Ensure bounds match prediction size
        actual_h = min(pred_h, y_max - y_min)
        actual_w = min(pred_w, x_max - x_min)

        suggestion[y_min:y_min+actual_h, x_min:x_min+actual_w] = prediction[:actual_h, :actual_w]

        # Update canvas
        self.canvas.set_suggestion(suggestion)

    def create_toolbar(self):
        """Create the toolbar with two rows for better organization."""
        # Container widget for two toolbar rows
        toolbar_container = QWidget()
        toolbar_layout = QVBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(2)

        # ===== ROW 1: File operations, Tools, Brush Size =====
        row1 = QToolBar()
        row1.setMovable(False)

        load_btn = QPushButton("Load Images")
        load_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        load_btn.clicked.connect(self.load_images)
        row1.addWidget(load_btn)

        load_project_btn = QPushButton("Load Project")
        load_project_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        load_project_btn.clicked.connect(self.load_project)
        row1.addWidget(load_project_btn)

        save_btn = QPushButton("Save")
        save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        save_btn.setToolTip("Save project (Ctrl+S)")
        save_btn.clicked.connect(self.save_project)
        row1.addWidget(save_btn)

        review_btn = QPushButton("Review Crops")
        review_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        review_btn.setToolTip("Review training crops and discard bad ones (A/D to navigate, Space to discard)")
        review_btn.clicked.connect(self._open_training_reviewer)
        row1.addWidget(review_btn)

        row1.addSeparator()

        brush_btn = QPushButton("\U0001F58C Brush")  # 🖌️
        brush_btn.setCheckable(True)
        brush_btn.setChecked(True)
        brush_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        brush_btn.clicked.connect(lambda: self.set_tool('brush', brush_btn))
        row1.addWidget(brush_btn)
        self.brush_btn = brush_btn

        eraser_btn = QPushButton("\U0001F9F9 Eraser")  # 🧹
        eraser_btn.setCheckable(True)
        eraser_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        eraser_btn.clicked.connect(lambda: self.set_tool('eraser', eraser_btn))
        row1.addWidget(eraser_btn)
        self.eraser_btn = eraser_btn

        hand_btn = QPushButton("\u270B Pan")  # ✋
        hand_btn.setCheckable(True)
        hand_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        hand_btn.clicked.connect(lambda: self.set_tool('hand', hand_btn))
        row1.addWidget(hand_btn)
        self.hand_btn = hand_btn

        fill_btn = QPushButton("\U0001FAA3 Fill")  # 🪣
        fill_btn.setCheckable(True)
        fill_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        fill_btn.setToolTip("Fill connected region (click to fill)")
        fill_btn.clicked.connect(lambda: self.set_tool('fill', fill_btn))
        row1.addWidget(fill_btn)
        self.fill_btn = fill_btn

        row1.addSeparator()

        row1.addWidget(QLabel("  Brush: "))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(10)
        self.brush_slider.setMaximumWidth(scaled(120))
        self.brush_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        row1.addWidget(self.brush_slider)

        self.brush_size_label = QLabel("10")
        row1.addWidget(self.brush_size_label)

        row1.addSeparator()

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(scaled(30))
        zoom_in_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        zoom_in_btn.setToolTip("Zoom In")
        zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        row1.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(scaled(30))
        zoom_out_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        zoom_out_btn.setToolTip("Zoom Out")
        zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        row1.addWidget(zoom_out_btn)

        toolbar_layout.addWidget(row1)

        # ===== ROW 2: View controls, Prediction, Training =====
        row2 = QToolBar()
        row2.setMovable(False)

        # Crop preview alpha slider
        row2.addWidget(QLabel("  Crop: "))
        crop_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        crop_alpha_slider.setMinimum(0)
        crop_alpha_slider.setMaximum(100)
        crop_alpha_slider.setValue(60)
        crop_alpha_slider.setMaximumWidth(scaled(70))
        crop_alpha_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        crop_alpha_slider.setToolTip("Crop preview box opacity (Tab to capture)")
        crop_alpha_slider.valueChanged.connect(self.on_crop_alpha_changed)
        row2.addWidget(crop_alpha_slider)

        row2.addWidget(QLabel("  Mask: "))
        mask_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        mask_alpha_slider.setMinimum(0)
        mask_alpha_slider.setMaximum(100)
        mask_alpha_slider.setValue(50)
        mask_alpha_slider.setMaximumWidth(scaled(70))
        mask_alpha_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        mask_alpha_slider.valueChanged.connect(self.on_mask_alpha_changed)
        row2.addWidget(mask_alpha_slider)

        row2.addSeparator()

        # Prediction model selector
        pred_label = QLabel("Predict:")
        row2.addWidget(pred_label)

        self.pred_model_combo = QComboBox()
        self.pred_model_combo.setToolTip("Select model for predictions")
        self.pred_model_combo.setFixedWidth(scaled(100))
        self.pred_model_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._populate_prediction_model_combo()
        self.pred_model_combo.currentTextChanged.connect(self.on_prediction_model_changed)
        row2.addWidget(self.pred_model_combo)

        # Prediction toggle
        self.show_pred_check = QCheckBox("Show")
        self.show_pred_check.setChecked(False)
        self.show_pred_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.show_pred_check.stateChanged.connect(self.on_show_predictions_changed)
        row2.addWidget(self.show_pred_check)

        # Accept All prediction button
        self.accept_all_btn = QPushButton("Accept")
        self.accept_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.accept_all_btn.setToolTip("Replace current mask with the prediction (Shift+Space)")
        self.accept_all_btn.clicked.connect(self._accept_all_prediction)
        row2.addWidget(self.accept_all_btn)

        row2.addSeparator()

        # Architecture selector
        arch_label = QLabel("Train:")
        row2.addWidget(arch_label)

        self.arch_combo = QComboBox()
        self.arch_combo.setToolTip("Select model architecture")
        self.arch_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._populate_architecture_combo()
        self.arch_combo.currentTextChanged.connect(self.on_architecture_changed)
        row2.addWidget(self.arch_combo)

        self.train_btn = QPushButton("Start")
        self.train_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.train_btn.setToolTip("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        row2.addWidget(self.train_btn)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setFixedWidth(scaled(80))
        self.train_progress.setVisible(False)
        self.train_progress.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row2.addWidget(self.train_progress)

        reset_btn = QPushButton("Reset")
        reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset_btn.setToolTip("Archive current model and start fresh")
        reset_btn.clicked.connect(self.reset_model)
        row2.addWidget(reset_btn)

        toolbar_layout.addWidget(row2)

        return toolbar_container

    def create_controls(self):
        """Create bottom navigation controls."""
        controls_widget = QWidget()
        layout = QHBoxLayout(controls_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        prev_btn = QPushButton("\u25C0")  # ◀
        prev_btn.setFixedWidth(scaled(40))
        prev_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        prev_btn.clicked.connect(self.prev_slice)
        layout.addWidget(prev_btn)

        # Navigation slider - only updates on release, not while dragging
        self.nav_slider = QSlider(Qt.Orientation.Horizontal)
        self.nav_slider.setMinimum(0)
        self.nav_slider.setMaximum(0)
        self.nav_slider.setValue(0)
        self.nav_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.nav_slider.sliderReleased.connect(self.on_nav_slider_released)
        self.nav_slider.valueChanged.connect(self.on_nav_slider_preview)  # Preview only
        layout.addWidget(self.nav_slider)

        next_btn = QPushButton("\u25B6")  # ▶
        next_btn.setFixedWidth(scaled(40))
        next_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        next_btn.clicked.connect(self.next_slice)
        layout.addWidget(next_btn)

        self.slice_label = QLabel("0 / 0")
        self.slice_label.setMinimumWidth(scaled(80))
        layout.addWidget(self.slice_label)

        layout.addStretch()

        # Refiner toggle button
        self.refiner_btn = QPushButton("Refiner: OFF")
        self.refiner_btn.setCheckable(True)
        self.refiner_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refiner_btn.setToolTip(
            "Toggle Refiner Mode - learns from your edits in real-time.\n"
            "When ON, stops regular training/prediction and runs the refiner instead.\n"
            "Paint edits, then press Tab to capture crops for training."
        )
        self.refiner_btn.clicked.connect(self._toggle_refiner_mode)
        layout.addWidget(self.refiner_btn)

        # Reset Refiner button
        self.reset_refiner_btn = QPushButton("Reset")
        self.reset_refiner_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.reset_refiner_btn.setToolTip("Reset refiner model weights and restart training")
        self.reset_refiner_btn.setVisible(False)  # Only visible when refiner is ON
        self.reset_refiner_btn.clicked.connect(self._reset_refiner)
        layout.addWidget(self.reset_refiner_btn)

        # Refiner edit count label
        self.refiner_edits_label = QLabel("")
        self.refiner_edits_label.setVisible(False)
        layout.addWidget(self.refiner_edits_label)

        layout.addSpacing(scaled(10))

        self.edits_label = QLabel("Edits: 0")
        layout.addWidget(self.edits_label)

        return controls_widget

    def on_nav_slider_preview(self, value: int):
        """Preview slice number while dragging (doesn't load the image)."""
        if self.image_files:
            # Just update the label to show where we're going
            self.slice_label.setText(f"{value + 1} / {len(self.image_files)}")

    def on_nav_slider_released(self):
        """Handle navigation slider release - only loads when user stops dragging."""
        value = self.nav_slider.value()
        if value != self.current_slice_index and 0 <= value < len(self.image_files):
            self.save_current_slice()
            self.current_slice_index = value
            self._check_and_extend_window()
            self.load_current_slice()

    def set_tool(self, tool: str, button):
        """Set the active drawing tool."""
        self.current_tool = tool
        self.canvas.set_tool(tool)
        self.brush_btn.setChecked(tool == 'brush')
        self.eraser_btn.setChecked(tool == 'eraser')
        self.hand_btn.setChecked(tool == 'hand')
        self.fill_btn.setChecked(tool == 'fill')

    def on_brush_size_changed(self, value: int):
        self.canvas.set_brush_size(value)
        self.brush_size_label.setText(str(value))

    def on_mask_alpha_changed(self, value: int):
        self.canvas.set_mask_alpha(value / 100.0)

    def on_crop_alpha_changed(self, value: int):
        self.canvas.set_crop_preview_alpha(value / 100.0)

    def on_suggestion_accepted(self):
        """Handle when a suggestion component is accepted via spacebar."""
        self.status_label.setText("Suggestion accepted! (added to mask)")

    def _accept_all_prediction(self):
        """Replace current mask with the prediction."""
        if self.canvas.suggestion is None:
            self.status_label.setText("No prediction to accept")
            return

        idx = self.current_slice_index
        if idx not in self.masks:
            self.status_label.setText("No mask loaded")
            return

        # Get the suggestion (prediction)
        suggestion = self.canvas.suggestion

        # Make sure dimensions match
        mask_shape = self.masks[idx].shape
        if suggestion.shape != mask_shape:
            self.status_label.setText(f"Shape mismatch: mask {mask_shape} vs prediction {suggestion.shape}")
            return

        # Replace the mask with the prediction
        self.masks[idx] = suggestion.copy()

        # Update canvas
        self.canvas.set_mask(self.masks[idx])

        # Clear the suggestion overlay since we've accepted it
        self.canvas.set_suggestion(None)

        # Mark as edited
        self.edit_count += 1
        self.edits_label.setText(f"Edits: {self.edit_count}")

        self.status_label.setText("Prediction accepted as new mask")

    def event(self, event):
        """Override event to catch Tab key before focus navigation."""
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab:
            self.capture_crop_for_training()
            return True  # Event handled
        return super().event(event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # Save: Ctrl+S
        if key == Qt.Key.Key_S and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.save_project()
            return

        # Accept all prediction: Shift+Space
        if key == Qt.Key.Key_Space and modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._accept_all_prediction()
            return

        # Toggle predictions: S (without modifiers)
        if key == Qt.Key.Key_S and not modifiers:
            self.show_pred_check.setChecked(not self.show_pred_check.isChecked())
            return

        # Accept suggestion: Space (always forward to canvas)
        if key == Qt.Key.Key_Space:
            # Forward space bar to canvas for accepting hovered component
            if hasattr(self.canvas, 'accept_hovered_component'):
                if self.canvas.accept_hovered_component():
                    self.canvas.suggestion_accepted.emit()
            return

        # Undo: Ctrl+Z
        if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.undo()
            return

        # Navigation: A/D or Left/Right arrows
        if key == Qt.Key.Key_Left or key == Qt.Key.Key_A:
            self.prev_slice()
        elif key == Qt.Key.Key_Right or key == Qt.Key.Key_D:
            self.next_slice()
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.canvas.zoom_in()
        elif key == Qt.Key.Key_Minus:
            self.canvas.zoom_out()
        elif key == Qt.Key.Key_B:
            self.set_tool('brush', self.brush_btn)
        elif key == Qt.Key.Key_E:
            self.set_tool('eraser', self.eraser_btn)
        elif key == Qt.Key.Key_H:
            self.set_tool('hand', self.hand_btn)
        elif key == Qt.Key.Key_F:
            # F is handled as modifier in canvas, not as tool toggle
            pass
        else:
            super().keyPressEvent(event)

    def set_config(self, config: dict):
        """Set configuration from setup page."""
        new_project_dir = Path(config.get('project_dir', ''))

        # Detect if this is a different project (not just a re-set of the same project)
        old_project_dir = getattr(self, 'project_dir', None)
        is_project_change = (old_project_dir is not None
                            and old_project_dir != new_project_dir
                            and str(old_project_dir) != '')

        if is_project_change:
            print(f"[Training] Project changed from {old_project_dir} to {new_project_dir}")
            # Reset state for new project
            self.image_files = []
            self.image_source = None
            self.current_index = 0
            # Canvas will be updated when new images are loaded via scan_and_load_initial

        self.config = config
        self.project_dir = new_project_dir

        if self.project_dir:
            self.train_images_dir = self.project_dir / 'train_images'
            self.train_masks_dir = self.project_dir / 'train_masks'
            self.train_images_25d_dir = self.project_dir / 'train_images_25d'
            self.train_masks_25d_dir = self.project_dir / 'train_masks_25d'
            self.masks_dir = self.project_dir / 'masks'
            self.refiner_images_dir = self.project_dir / 'refiner_images'
            self.refiner_masks_before_dir = self.project_dir / 'refiner_masks_before'
            self.refiner_masks_after_dir = self.project_dir / 'refiner_masks_after'

            self.train_images_dir.mkdir(parents=True, exist_ok=True)
            self.train_masks_dir.mkdir(parents=True, exist_ok=True)
            self.train_images_25d_dir.mkdir(parents=True, exist_ok=True)
            self.train_masks_25d_dir.mkdir(parents=True, exist_ok=True)
            self.masks_dir.mkdir(parents=True, exist_ok=True)
            self.refiner_images_dir.mkdir(parents=True, exist_ok=True)
            self.refiner_masks_before_dir.mkdir(parents=True, exist_ok=True)
            self.refiner_masks_after_dir.mkdir(parents=True, exist_ok=True)

            # Check if we have pending weights from multi-user that can now be applied
            self._try_apply_pending_weights()

        # Store user-provided training masks directory (optional layer to edit)
        # Masks will be copied to project folder in scan_and_load_initial
        user_training_masks = config.get('train_masks')
        if user_training_masks and os.path.isdir(user_training_masks):
            self.user_training_masks_dir = Path(user_training_masks)
        else:
            self.user_training_masks_dir = None

        # Store the raw_images_dir but don't load immediately (lazy loading)
        # Images will be loaded when the page becomes visible
        raw_images_dir = config.get('raw_images_dir') or config.get('train_images')
        self._pending_images_dir = raw_images_dir if raw_images_dir and os.path.isdir(raw_images_dir) else None

    def showEvent(self, event):
        """Handle page becoming visible - load images if pending."""
        super().showEvent(event)
        # Load images only when page is actually shown (lazy loading)
        if hasattr(self, '_pending_images_dir') and self._pending_images_dir:
            # Avoid reloading if already loaded from same directory AND image_files is populated
            if (self.image_source and self.image_source.get('path') == self._pending_images_dir
                    and self.image_files):
                return
            self.scan_and_load_initial(self._pending_images_dir)
            self._pending_images_dir = None  # Clear so we don't reload on re-show

    def load_images(self):
        """Open file dialog to load images."""
        dirpath = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dirpath:
            self.scan_and_load_initial(dirpath)

    def scan_and_load_initial(self, dirpath: str):
        """Scan directory for images and load initial window."""
        folder = Path(dirpath)
        self.image_files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        ])

        if not self.image_files:
            QMessageBox.warning(self, "Error", f"No images found in {dirpath}")
            return

        self.image_source = {'type': 'directory', 'path': dirpath}

        # Scan all images to find max dimensions (for padding mismatched sizes)
        max_h, max_w = 0, 0
        print(f"[Scan] Checking dimensions of {len(self.image_files)} images...")
        for img_path in self.image_files:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)
            except Exception as e:
                print(f"[Scan] Error reading {img_path.name}: {e}")
        self._max_image_size = (max_h, max_w)
        print(f"[Scan] Max image size: {max_h} x {max_w}")

        # Copy user-provided training masks to project's masks folder (if provided)
        # This ensures all masks are in the project and edits are saved properly
        if self.user_training_masks_dir and self.user_training_masks_dir.exists() and self.masks_dir:
            mask_files = sorted([
                f for f in self.user_training_masks_dir.iterdir()
                if f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
            ])
            if len(mask_files) == len(self.image_files):
                # Check if we need to copy (masks_dir empty or incomplete)
                existing_masks = list(self.masks_dir.glob("mask_*.tif"))
                if len(existing_masks) < len(mask_files):
                    print(f"[TrainingMasks] Copying {len(mask_files)} training masks to project...")
                    self._copy_training_masks_to_project(mask_files)
                else:
                    print(f"[TrainingMasks] Project already has {len(existing_masks)} masks, skipping copy")
            elif len(mask_files) > 0:
                print(f"[TrainingMasks] WARNING: Mask count ({len(mask_files)}) != image count ({len(self.image_files)}), ignoring training masks")

        # Clear existing data
        self.images = {}
        self.masks = {}

        # Check for saved state BEFORE loading to center window on saved position
        saved_index = 0
        if self.project_dir:
            config = load_project_config(str(self.project_dir))
            if config:
                saved_index = config.get("current_slice_index", 0)
                self.edit_count = config.get("edit_count", 0)
                if hasattr(self, 'edits_label'):
                    self.edits_label.setText(f"Edits: {self.edit_count}")

                # Restore training architecture selection
                saved_arch = config.get("architecture", "unet")
                if saved_arch != self.current_architecture:
                    self.current_architecture = saved_arch
                    # Update combo box (if populated)
                    if hasattr(self, '_arch_id_to_name') and saved_arch in self._arch_id_to_name:
                        self.arch_combo.blockSignals(True)
                        idx = self.arch_combo.findText(self._arch_id_to_name[saved_arch])
                        if idx >= 0:
                            self.arch_combo.setCurrentIndex(idx)
                        self.arch_combo.blockSignals(False)
                    # Check if we have pending weights that can now be applied
                    self._try_apply_pending_weights()

                # Restore prediction architecture selection
                saved_pred_arch = config.get("prediction_architecture", saved_arch)
                if saved_pred_arch != self.prediction_architecture:
                    self.prediction_architecture = saved_pred_arch
                    # Update prediction combo box (if populated)
                    if hasattr(self, '_pred_id_to_name') and saved_pred_arch in self._pred_id_to_name:
                        self.pred_model_combo.blockSignals(True)
                        idx = self.pred_model_combo.findText(self._pred_id_to_name[saved_pred_arch])
                        if idx >= 0:
                            self.pred_model_combo.setCurrentIndex(idx)
                        self.pred_model_combo.blockSignals(False)
                    # Update prediction worker
                    if self.predict_worker:
                        self.predict_worker.set_architecture(saved_pred_arch)

                # Restore user training masks directory if saved (for re-copying if needed)
                saved_user_masks = config.get("user_training_masks_dir")
                if saved_user_masks and os.path.isdir(saved_user_masks):
                    self.user_training_masks_dir = Path(saved_user_masks)

        # Clamp saved_index to valid range
        saved_index = max(0, min(saved_index, len(self.image_files) - 1))
        self.current_slice_index = saved_index

        # Load window centered on saved position (single load)
        half_window = self.WINDOW_SIZE // 2
        start = max(0, saved_index - half_window)
        end = min(len(self.image_files), start + self.WINDOW_SIZE)
        # Adjust start if end hit the limit
        start = max(0, end - self.WINDOW_SIZE)

        self._load_window(start, end)

        # Display current slice (already in window, no additional load)
        self.load_current_slice()
        self.status_label.setText(
            f"Loaded {len(self.image_files)} images | "
            f"Slice {self.current_slice_index + 1} | "
            f"In memory: {self.window_start + 1}-{self.window_end}"
        )

        # Save config
        self._save_project_config()

    def _load_window(self, start: int, end: int):
        """Load a window of images from start to end index using parallel loading."""
        # Prevent concurrent loads (can happen due to processEvents during loading)
        if self._loading_in_progress:
            return

        start = max(0, start)
        end = min(len(self.image_files), end)

        if start >= end:
            return

        self._loading_in_progress = True

        # Evict images outside the new window (with some buffer)
        buffer = 50
        evict_below = start - buffer
        evict_above = end + buffer

        keys_to_remove = [k for k in self.images.keys() if k < evict_below or k >= evict_above]
        for k in keys_to_remove:
            del self.images[k]
            if k in self.masks:
                del self.masks[k]

        # Build list of images to load (skip already loaded)
        to_load = []
        for i in range(start, end):
            if i not in self.images:
                mask_path = None
                if self.masks_dir:
                    mask_path = self.masks_dir / f"mask_{i:05d}.tif"
                to_load.append((i, self.image_files[i], mask_path, self._max_image_size))

        if not to_load:
            self.window_start = start
            self.window_end = end
            self._loading_in_progress = False
            return

        # Show loading dialog
        loading = LoadingDialog(self, f"Loading {len(to_load)} images...")
        loading.show()
        loading.set_progress(0, len(to_load))
        QApplication.processEvents()

        # Parallel load images using ThreadPoolExecutor
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        loaded_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all load tasks
            futures = executor.map(_load_single_image, to_load)

            # Process results as they complete
            for idx, img, mask, error in futures:
                if error:
                    print(f"Failed to load image {idx}: {error}")
                else:
                    self.images[idx] = img
                    self.masks[idx] = mask

                loaded_count += 1
                loading.set_progress(loaded_count, len(to_load))
                QApplication.processEvents()

        self.window_start = start
        self.window_end = end
        self._loading_in_progress = False
        loading.close()

    def _check_and_extend_window(self):
        """Check if we need to load more images and do so if needed."""
        if not self.image_files:
            return

        idx = self.current_slice_index

        # If completely outside current window, load a new window centered on target
        if idx < self.window_start or idx >= self.window_end:
            half_window = self.WINDOW_SIZE // 2
            new_start = max(0, idx - half_window)
            new_end = min(len(self.image_files), new_start + self.WINDOW_SIZE)
            new_start = max(0, new_end - self.WINDOW_SIZE)  # Adjust if hit end
            self._load_window(new_start, new_end)
            self.status_label.setText(
                f"Slice {idx+1}/{len(self.image_files)} | "
                f"In memory: {self.window_start+1}-{self.window_end}"
            )
            return

        # Check if approaching end of window - extend forward
        if idx >= self.window_end - self.LOAD_THRESHOLD:
            new_end = min(self.window_end + self.BATCH_SIZE, len(self.image_files))
            new_start = max(0, new_end - self.WINDOW_SIZE)
            if new_end > self.window_end:
                self._load_window(new_start, new_end)
                self.status_label.setText(
                    f"Slice {idx+1}/{len(self.image_files)} | "
                    f"In memory: {self.window_start+1}-{self.window_end}"
                )

        # Check if approaching start of window - extend backward
        elif idx <= self.window_start + self.LOAD_THRESHOLD:
            new_start = max(0, self.window_start - self.BATCH_SIZE)
            new_end = min(new_start + self.WINDOW_SIZE, len(self.image_files))
            if new_start < self.window_start:
                self._load_window(new_start, new_end)
                self.status_label.setText(
                    f"Slice {idx+1}/{len(self.image_files)} | "
                    f"In memory: {self.window_start+1}-{self.window_end}"
                )

    def prev_slice(self):
        """Navigate to the previous slice."""
        if self.current_slice_index > 0:
            self.save_current_slice()
            self.current_slice_index -= 1
            self._check_and_extend_window()
            self.load_current_slice()
            self._prefetch_adjacent_slices()

    def next_slice(self):
        """Navigate to the next slice."""
        if self.current_slice_index < len(self.image_files) - 1:
            self.save_current_slice()
            self.current_slice_index += 1
            self._check_and_extend_window()
            self.load_current_slice()
            self._prefetch_adjacent_slices()

    def _prefetch_adjacent_slices(self):
        """Pre-fetch adjacent slices in background for 2.5D prediction."""
        if not self.image_files or not hasattr(self, 'predict_worker'):
            return
        if not self.predict_worker.is_25d():
            return

        import threading
        from concurrent.futures import ThreadPoolExecutor

        def prefetch():
            idx = self.current_slice_index
            slice_spacing = 3  # Distance between slices (z-3, z, z+3)
            total = len(self.image_files)

            # Build list of indices to load (z-3, z, z+3)
            to_load = []
            for offset in [-slice_spacing, 0, slice_spacing]:
                target_idx = idx + offset
                if target_idx < 0:
                    target_idx = 0
                elif target_idx >= total:
                    target_idx = total - 1

                if target_idx not in self.images and target_idx not in to_load:
                    to_load.append(target_idx)

            if not to_load:
                return

            def load_one(target_idx):
                try:
                    img = np.array(Image.open(self.image_files[target_idx]))
                    if img.ndim == 3:
                        img = img.mean(axis=-1)
                    img = img.astype(np.float32)
                    # Pad to max size if needed
                    if self._max_image_size is not None:
                        max_h, max_w = self._max_image_size
                        h, w = img.shape
                        if h < max_h or w < max_w:
                            pad_h = max_h - h
                            pad_w = max_w - w
                            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                    return target_idx, img
                except:
                    return target_idx, None

            # Use 8 workers for parallel loading
            with ThreadPoolExecutor(max_workers=min(len(to_load), 8)) as executor:
                results = list(executor.map(load_one, to_load))

            for target_idx, img in results:
                if img is not None:
                    self.images[target_idx] = img

        thread = threading.Thread(target=prefetch, daemon=True)
        thread.start()

    def _copy_training_masks_to_project(self, mask_files: list):
        """Copy user-provided training masks to project's masks folder.

        This ensures all masks are in the project and edits are saved properly.
        Masks are renamed to mask_{index:05d}.tif format.
        """
        from concurrent.futures import ThreadPoolExecutor

        def copy_single_mask(args):
            idx, src_path = args
            dst_path = self.masks_dir / f"mask_{idx:05d}.tif"

            # Skip if already exists
            if dst_path.exists():
                return idx, True, None

            try:
                # Load and normalize the mask
                mask = np.array(Image.open(src_path))

                # Handle multi-channel masks (take first channel)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]

                # Normalize to 0-255 range (handles binary 0/1 masks)
                if mask.max() > 0 and mask.max() <= 1:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)

                # Save to project masks folder
                Image.fromarray(mask).save(dst_path, compression='tiff_lzw')
                return idx, True, None
            except Exception as e:
                return idx, False, str(e)

        # Show progress dialog
        loading = LoadingDialog(self, f"Copying {len(mask_files)} training masks...")
        loading.show()
        loading.set_progress(0, len(mask_files))
        QApplication.processEvents()

        # Build list of (index, path) tuples
        to_copy = [(i, mask_files[i]) for i in range(len(mask_files))]

        # Copy in parallel
        max_workers = min(16, (os.cpu_count() or 1) + 4)
        copied_count = 0
        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, success, error in executor.map(copy_single_mask, to_copy):
                copied_count += 1
                if error:
                    errors.append(f"Mask {idx}: {error}")
                loading.set_progress(copied_count, len(mask_files))
                QApplication.processEvents()

        loading.close()

        if errors:
            print(f"[TrainingMasks] Errors copying {len(errors)} masks:")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  {err}")

        print(f"[TrainingMasks] Copied {copied_count - len(errors)} masks to {self.masks_dir}")

    def save_current_slice(self):
        """Save the current slice mask to memory and disk."""
        idx = self.current_slice_index
        if idx in self.images and self.canvas.mask is not None:
            mask = self.canvas.mask.copy()
            self.masks[idx] = mask

            # Save to disk when navigating away
            if self.masks_dir:
                mask_path = self.masks_dir / f"mask_{idx:05d}.tif"
                try:
                    Image.fromarray(mask).save(mask_path, compression='tiff_lzw')
                except Exception as e:
                    print(f"Failed to save mask: {e}")

    def load_current_slice(self):
        """Load the current slice to canvas."""
        idx = self.current_slice_index

        # Clear paint bounds when loading new slice
        self.canvas.clear_paint_bounds()

        if idx in self.images:
            self.canvas.set_image(self.images[idx])
            if idx in self.masks:
                self.canvas.set_mask(self.masks[idx])
                # Track mask state at load time for refiner (before any edits)
                if self.refiner_mode and idx not in self._refiner_mask_at_load:
                    self._refiner_mask_at_load[idx] = self.masks[idx].copy()
                    mask_px = np.sum(self.masks[idx] > 127)
                    print(f"[Refiner] Load slice {idx}: tracking 'before' state: {mask_px}px")
            self.update_slice_label()
        else:
            # Need to load this image
            self._load_window(
                max(0, idx - self.WINDOW_SIZE // 2),
                min(len(self.image_files), idx + self.WINDOW_SIZE // 2)
            )
            if idx in self.images:
                self.canvas.set_image(self.images[idx])
                if idx in self.masks:
                    self.canvas.set_mask(self.masks[idx])
                    # Track mask state at load time for refiner
                    if self.refiner_mode and idx not in self._refiner_mask_at_load:
                        self._refiner_mask_at_load[idx] = self.masks[idx].copy()
                        mask_px = np.sum(self.masks[idx] > 127)
                        print(f"[Refiner] Load slice {idx} (window): tracking 'before' state: {mask_px}px")
            self.update_slice_label()

        # Request prediction for new slice if enabled
        if self.show_predictions:
            self._request_viewport_prediction()

        # Refiner prediction on slice change
        if self.refiner_mode:
            self._refiner_debounce_timer.start(300)

    def update_slice_label(self):
        """Update the slice indicator label and slider."""
        total = len(self.image_files)
        current = self.current_slice_index + 1 if total > 0 else 0
        self.slice_label.setText(f"{current} / {total}")

        # Update slider without triggering valueChanged
        self.nav_slider.blockSignals(True)
        self.nav_slider.setMaximum(max(0, total - 1))
        self.nav_slider.setValue(self.current_slice_index)
        self.nav_slider.blockSignals(False)

    def on_edit_made(self, before_mask, after_mask):
        """Handle edit signal from canvas - memory only, no disk I/O."""
        idx = self.current_slice_index
        if idx not in self.images:
            return

        # Store undo history
        self.undo_stack.append((idx, before_mask.copy()))
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

        # Update mask in memory only - NO DISK I/O
        self.masks[idx] = after_mask.copy()

        self.edit_count += 1
        self.edits_label.setText(f"Edits: {self.edit_count}")

        # In refiner mode, we DON'T feed individual brush strokes (too small)
        # Instead, use Tab to capture larger crops with before/after context
        # Just trigger prediction update after edits
        if self.refiner_mode:
            self._refiner_debounce_timer.start(500)  # 500ms delay for prediction

    def capture_crop_for_training(self):
        """Capture the current crop preview area and save for training (Tab key).

        In refiner mode: saves to refiner folders and feeds to refiner worker.
        In normal mode: saves to train_images/train_masks for regular training.
        """
        # Check we have everything needed
        if not self.project_dir:
            self.status_label.setText("No project - cannot capture crop")
            return

        idx = self.current_slice_index
        if idx not in self.images or idx not in self.masks:
            self.status_label.setText("No image/mask loaded")
            return

        # Get crop bounds from canvas
        bounds = self.canvas.get_crop_bounds()
        if bounds is None:
            self.status_label.setText("No crop area - paint something first")
            return

        crop_x, crop_y, crop_w, crop_h = bounds

        # Extract image and mask crops
        image = self.images[idx]
        mask_after = self.masks[idx]

        img_crop = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
        mask_after_crop = mask_after[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()

        # Normalize image to uint8 for saving
        img_min, img_max = img_crop.min(), img_crop.max()
        if img_max > img_min:
            img_crop_uint8 = ((img_crop - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_crop_uint8 = img_crop.astype(np.uint8)

        # Generate unique filename
        import time
        timestamp = int(time.time() * 1000) % 100000
        crop_id = f"slice{idx:04d}_cap{timestamp}"

        # REFINER MODE: Save to refiner folders and feed to refiner
        if self.refiner_mode:
            try:
                # Get mask_before from when slice was loaded
                if idx in self._refiner_mask_at_load:
                    mask_before = self._refiner_mask_at_load[idx]
                    mask_before_crop = mask_before[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
                else:
                    # Fallback: use zeros if no before state tracked
                    mask_before_crop = np.zeros_like(mask_after_crop)
                    print(f"[Refiner] WARNING: No before state for slice {idx}, using zeros")

                # Debug: check diff at capture time
                diff_at_capture = np.sum(mask_before_crop != mask_after_crop)
                before_px = np.sum(mask_before_crop > 127)
                after_px = np.sum(mask_after_crop > 127)
                print(f"[Refiner] CAPTURE {crop_id}: before={before_px}px, after={after_px}px, diff={diff_at_capture}px")

                if diff_at_capture == 0:
                    print(f"[Refiner] WARNING: No difference in capture! Skipping save.")
                    self.status_label.setText("Refiner: No changes to capture - paint something first!")
                    return

                # Save to refiner folders
                Image.fromarray(img_crop_uint8).save(self.refiner_images_dir / f"{crop_id}.tif", compression='tiff_lzw')
                Image.fromarray(mask_before_crop).save(self.refiner_masks_before_dir / f"{crop_id}.tif", compression='tiff_lzw')
                Image.fromarray(mask_after_crop).save(self.refiner_masks_after_dir / f"{crop_id}.tif", compression='tiff_lzw')

                # Update sample count (refiner loads from folders)
                if self.refiner_worker is not None:
                    sample_count = self.refiner_worker.get_edit_count()
                    self.refiner_edits_label.setText(f"({sample_count} samples)")

                # Update the "at load" state to current, so next Tab captures new changes
                self._refiner_mask_at_load[idx] = mask_after.copy()

                # Count refiner samples
                refiner_count = len(list(self.refiner_images_dir.glob("*.tif")))
                self.status_label.setText(
                    f"Refiner: Captured {crop_w}x{crop_h} crop! ({refiner_count} samples)"
                )

                # Clear paint bounds
                self.canvas.clear_paint_bounds()

                # Request prediction after adding edit
                self._refiner_debounce_timer.start(500)

            except Exception as e:
                self.status_label.setText(f"Failed to save refiner crop: {e}")
            return

        # NORMAL MODE: Save to training folders
        try:
            img_path = self.train_images_dir / f"{crop_id}.tif"
            mask_path = self.train_masks_dir / f"{crop_id}.tif"

            Image.fromarray(img_crop_uint8).save(img_path, compression='tiff_lzw')
            Image.fromarray(mask_after_crop).save(mask_path, compression='tiff_lzw')

            # Also save 2.5D version if directories exist
            saved_25d = False
            if self.train_images_25d_dir and self.train_masks_25d_dir:
                adjacent = self._load_adjacent_slices(idx)
                if adjacent is not None and len(adjacent) == 3:
                    try:
                        import tifffile

                        # Crop each adjacent slice at the same position
                        crops_25d = []
                        for slice_img in adjacent:
                            h, w = slice_img.shape
                            if crop_y + crop_h <= h and crop_x + crop_w <= w:
                                crop = slice_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
                                crop_min, crop_max = crop.min(), crop.max()
                                if crop_max > crop_min:
                                    crop = ((crop - crop_min) / (crop_max - crop_min) * 255).astype(np.uint8)
                                else:
                                    crop = crop.astype(np.uint8)
                                crops_25d.append(crop)
                            else:
                                crops_25d = None
                                break

                        if crops_25d and len(crops_25d) == 3:
                            # Stack as (C, H, W) for tifffile - it interprets first dim as pages/channels
                            stack = np.stack(crops_25d, axis=0)
                            tifffile.imwrite(str(self.train_images_25d_dir / f"{crop_id}.tif"), stack, compression='lzw')
                            Image.fromarray(mask_after_crop).save(self.train_masks_25d_dir / f"{crop_id}.tif", compression='tiff_lzw')
                            saved_25d = True

                    except Exception as e:
                        print(f"Failed to save 2.5D crop: {e}")

            # Count existing training files
            train_count = len(list(self.train_images_dir.glob("*.tif")))
            count_25d = len(list(self.train_images_25d_dir.glob("*.tif"))) if self.train_images_25d_dir else 0

            status_msg = f"Captured crop! ({crop_w}x{crop_h}) - {train_count} 2D"
            if saved_25d:
                status_msg += f", {count_25d} 2.5D"
            status_msg += " training samples"

            # Multi-user mode: send crop to host
            if self._multi_user_enabled and self._sync_client and self._sync_client.is_connected:
                if not self._is_host:
                    # Client sends crop to host
                    self._sync_client.send_training_data(img_crop_uint8, mask_after_crop, idx)
                    status_msg += " (sent to host)"
                # Host doesn't need to send - crop is already saved locally

            self.status_label.setText(status_msg)

            # Clear paint bounds so user can start fresh
            self.canvas.clear_paint_bounds()

        except Exception as e:
            self.status_label.setText(f"Failed to save crop: {e}")

    def undo(self):
        """Undo the last edit."""
        if not self.undo_stack:
            return

        idx, old_mask = self.undo_stack.pop()

        # Restore the mask
        self.masks[idx] = old_mask.copy()

        # If we're on the same slice, update canvas
        if idx == self.current_slice_index:
            self.canvas.set_mask(old_mask)

        self.edit_count = max(0, self.edit_count - 1)
        self.edits_label.setText(f"Edits: {self.edit_count}")
        self.status_label.setText(f"Undo: restored mask for slice {idx + 1}")

    def load_project(self):
        """Load an existing project directory."""
        dirpath = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if not dirpath:
            return

        project_dir = Path(dirpath)

        # Check for expected project structure
        masks_dir = project_dir / 'masks'
        train_images_dir = project_dir / 'train_images'
        train_masks_dir = project_dir / 'train_masks'

        if not masks_dir.exists() and not train_images_dir.exists():
            QMessageBox.warning(
                self, "Invalid Project",
                "This doesn't appear to be a valid project directory.\n"
                "Expected 'masks/' or 'train_images/' folder."
            )
            return

        # Set project directories
        self.project_dir = project_dir
        self.masks_dir = masks_dir
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.refiner_images_dir = project_dir / 'refiner_images'
        self.refiner_masks_before_dir = project_dir / 'refiner_masks_before'
        self.refiner_masks_after_dir = project_dir / 'refiner_masks_after'

        # Create directories if they don't exist
        masks_dir.mkdir(parents=True, exist_ok=True)
        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_masks_dir.mkdir(parents=True, exist_ok=True)
        self.refiner_images_dir.mkdir(parents=True, exist_ok=True)
        self.refiner_masks_before_dir.mkdir(parents=True, exist_ok=True)
        self.refiner_masks_after_dir.mkdir(parents=True, exist_ok=True)

        # Look for raw images - check common locations
        raw_images_dir = None
        for candidate in ['raw_images', 'images', 'raw']:
            candidate_dir = project_dir / candidate
            if candidate_dir.exists():
                raw_images_dir = candidate_dir
                break

        if raw_images_dir is None:
            # Ask user to select raw images directory
            QMessageBox.information(
                self, "Select Images",
                "Project loaded. Now select the raw images directory."
            )
            raw_path = QFileDialog.getExistingDirectory(self, "Select Raw Images Directory")
            if raw_path:
                raw_images_dir = Path(raw_path)

        if raw_images_dir and raw_images_dir.exists():
            self.scan_and_load_initial(str(raw_images_dir))
            self.status_label.setText(f"Loaded project: {project_dir.name}")
        else:
            self.status_label.setText(f"Project loaded, no images found")

    def _build_project_config(self) -> dict:
        """Build project configuration dictionary."""
        from ..models.unet import get_checkpoint_filename

        config = {
            "version": "1.0",
            "project_name": self.project_dir.name if self.project_dir else "",

            # Paths (make relative where possible)
            "raw_images_dir": str(self.image_source.get('path', '')) if self.image_source else "",
            "masks_dir": "masks",
            "train_images_dir": "train_images",
            "train_masks_dir": "train_masks",
            "user_training_masks_dir": str(self.user_training_masks_dir) if self.user_training_masks_dir else None,
            "checkpoint_path": get_checkpoint_filename(self.current_architecture),

            # Model architecture
            "architecture": self.current_architecture,
            "prediction_architecture": self.prediction_architecture,

            # Training parameters from config
            "num_epochs": self.config.get("num_epochs", 5000),
            "batch_size": self.config.get("batch_size", 2),
            "tile_size": self.config.get("tile_size", 256),
            "learning_rate": self.config.get("learning_rate", 0.0001),

            # Session state
            "current_slice_index": self.current_slice_index,
            "edit_count": self.edit_count,
            "total_images": len(self.image_files),

            # Flags
            "interactive_mode": True,
            "training_started": self.train_worker is not None,
            "training_complete": False,
        }
        return config

    def _save_project_config(self):
        """Save current project configuration to project.json."""
        if not self.project_dir:
            return

        # Don't save if image_source is not set (project not fully loaded)
        if not self.image_source:
            print("[SaveConfig] Skipping save - no image_source set")
            return

        config = self._build_project_config()
        save_project_config(str(self.project_dir), config)

    def _auto_save_config(self):
        """Auto-save callback for timer."""
        if self.project_dir and self.image_files:
            self._save_project_config()

    def _save_edit_to_disk(self, idx, after_mask, image, before_mask):
        """Save mask and crops to disk (called asynchronously)."""
        print(f"DEBUG _save_edit_to_disk called: idx={idx}")

        # Save mask to disk
        if self.masks_dir:
            mask_path = self.masks_dir / f"mask_{idx:05d}.tif"
            try:
                Image.fromarray(after_mask).save(mask_path, compression='tiff_lzw')
            except Exception as e:
                print(f"Failed to save mask: {e}")

        # Save training crops
        self._save_edit_crops(image, before_mask, after_mask, idx)

    def _save_edit_crops(self, image, before_mask, after_mask, slice_idx):
        """Extract and save crops around the edited region."""
        print(f"DEBUG _save_edit_crops called: slice_idx={slice_idx}")

        if not self.train_images_dir or not self.train_masks_dir:
            print(f"DEBUG _save_edit_crops: train dirs not set, returning")
            return

        diff = (before_mask != after_mask)
        if not diff.any():
            print(f"DEBUG _save_edit_crops: no diff detected, returning")
            return

        print(f"DEBUG _save_edit_crops: diff found, proceeding to save crops")

        rows = np.any(diff, axis=1)
        cols = np.any(diff, axis=0)
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            return

        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
        center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2

        crop_size = 256
        h, w = image.shape

        # Load adjacent slices for 2.5D (if available)
        adjacent_slices = self._load_adjacent_slices(slice_idx)
        print(f"DEBUG: adjacent_slices={'None' if adjacent_slices is None else len(adjacent_slices)}, "
              f"train_images_25d_dir={self.train_images_25d_dir}")

        offsets = [(0, 0), (-64, 0), (64, 0), (0, -64), (0, 64)]
        for i, (dy, dx) in enumerate(offsets):
            py = center_y + dy - crop_size // 2
            px = center_x + dx - crop_size // 2

            py = max(0, min(py, h - crop_size))
            px = max(0, min(px, w - crop_size))

            if py + crop_size > h or px + crop_size > w:
                continue

            img_crop = image[py:py+crop_size, px:px+crop_size].copy()
            mask_crop = after_mask[py:py+crop_size, px:px+crop_size].copy()

            img_min, img_max = img_crop.min(), img_crop.max()
            if img_max > img_min:
                img_crop = ((img_crop - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_crop = img_crop.astype(np.uint8)

            crop_id = f"slice{slice_idx:04d}_edit{self.edit_count:04d}_crop{i}"
            try:
                # Save 2D crop
                Image.fromarray(img_crop).save(self.train_images_dir / f"{crop_id}.tif", compression='tiff_lzw')
                Image.fromarray(mask_crop).save(self.train_masks_dir / f"{crop_id}.tif", compression='tiff_lzw')

                # Save 2.5D stack (11 channels: 5 flanking on each side + center)
                if adjacent_slices is not None and self.train_images_25d_dir:
                    print(f"DEBUG: Saving 2.5D crop {crop_id} with {len(adjacent_slices)} slices")
                    self._save_25d_crop(adjacent_slices, mask_crop, py, px, crop_size, crop_id)
                else:
                    print(f"DEBUG: NOT saving 2.5D crop - adjacent_slices={adjacent_slices is not None}, "
                          f"25d_dir={self.train_images_25d_dir is not None}")

            except Exception as e:
                print(f"Failed to save crop: {e}")

    def _load_adjacent_slices(self, slice_idx: int) -> list:
        """Load adjacent slices for 2.5D training (z-3, z, z+3 = 3 channels)."""
        if not self.image_files or len(self.image_files) < 3:
            return None

        from concurrent.futures import ThreadPoolExecutor

        total_slices = len(self.image_files)
        slice_spacing = 3  # Distance between slices (z-3, z, z+3)

        # Build list of indices we need (z-3, z, z+3)
        indices = []
        for offset in [-slice_spacing, 0, slice_spacing]:
            idx = slice_idx + offset
            # Mirror at boundaries
            if idx < 0:
                idx = 0
            elif idx >= total_slices:
                idx = total_slices - 1
            indices.append(idx)

        # Identify which slices need loading from disk
        to_load = [idx for idx in set(indices) if idx not in self.images]

        # Parallel load missing slices
        if to_load:
            def load_slice(idx):
                try:
                    img = np.array(Image.open(self.image_files[idx]))
                    if img.ndim == 3:
                        img = img.mean(axis=-1)
                    img = img.astype(np.float32)
                    # Pad to max size if needed
                    if self._max_image_size is not None:
                        max_h, max_w = self._max_image_size
                        h, w = img.shape
                        if h < max_h or w < max_w:
                            pad_h = max_h - h
                            pad_w = max_w - w
                            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                    return idx, img
                except Exception as e:
                    print(f"Failed to load slice {idx}: {e}")
                    return idx, None

            with ThreadPoolExecutor(max_workers=min(len(to_load), 8)) as executor:
                results = list(executor.map(load_slice, to_load))

            # Cache loaded slices
            for idx, img in results:
                if img is not None:
                    self.images[idx] = img

        # Build output list in correct order
        slices = []
        for idx in indices:
            if idx in self.images:
                slices.append(self.images[idx])
            else:
                return None  # Failed to load a required slice

        return slices

    def _save_25d_crop(self, slices: list, mask_crop: np.ndarray,
                       py: int, px: int, crop_size: int, crop_id: str):
        """Save a 2.5D crop (11-channel stack) for training."""
        if not self.train_images_25d_dir or not self.train_masks_25d_dir:
            print(f"DEBUG _save_25d_crop: dirs not set")
            return

        try:
            import tifffile

            # Crop each slice at the same position
            crops = []
            for i, slice_img in enumerate(slices):
                h, w = slice_img.shape
                # Handle edge cases where slice might be smaller
                if py + crop_size > h or px + crop_size > w:
                    print(f"DEBUG _save_25d_crop: slice {i} too small ({h}x{w}) for crop at ({py},{px}) size {crop_size}")
                    return

                crop = slice_img[py:py+crop_size, px:px+crop_size].copy()

                # Normalize each channel independently
                crop_min, crop_max = crop.min(), crop.max()
                if crop_max > crop_min:
                    crop = ((crop - crop_min) / (crop_max - crop_min) * 255).astype(np.uint8)
                else:
                    crop = crop.astype(np.uint8)
                crops.append(crop)

            # Stack as (C, H, W) for tifffile - it interprets first dim as pages/channels
            stack = np.stack(crops, axis=0)

            print(f"DEBUG _save_25d_crop: saving stack shape {stack.shape} to {self.train_images_25d_dir / f'{crop_id}.tif'}")

            # Save as multi-channel TIFF using tifffile (handles >3 channels)
            tifffile.imwrite(str(self.train_images_25d_dir / f"{crop_id}.tif"), stack, compression='lzw')
            Image.fromarray(mask_crop).save(self.train_masks_25d_dir / f"{crop_id}.tif", compression='tiff_lzw')

        except Exception as e:
            print(f"Failed to save 2.5D crop: {e}")

    def reset_model(self):
        """Reset the model - archive old checkpoint and start fresh."""
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project directory set")
            return

        checkpoint_path = self._get_checkpoint_path()

        # Check if there's a model to reset
        if not checkpoint_path or not checkpoint_path.exists():
            arch_name = self._arch_id_to_name.get(self.current_architecture, self.current_architecture)
            QMessageBox.information(
                self, "No Model",
                f"No existing model found for {arch_name}. Training will start fresh."
            )
            return

        # Confirmation dialog
        arch_name = self._arch_id_to_name.get(self.current_architecture, self.current_architecture)
        reply = QMessageBox.question(
            self, "Reset Model",
            f"Are you sure you want to reset the {arch_name} model?\n\n"
            "The current model will be archived with a timestamp\n"
            "and training will start from scratch.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Stop training if running
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.stop()
            # Use timeout to avoid freezing UI
            if not self.train_worker.wait(3000):
                print("[Training] Worker didn't stop in 3s, forcing termination")
                self.train_worker.terminate()
                self.train_worker.wait(1000)
            self.train_btn.setText("Start Training")
            self.train_progress.setVisible(False)
            self.arch_combo.setEnabled(True)

        # Clear the worker reference
        self.train_worker = None

        # Small delay to ensure file operations complete
        import time
        time.sleep(0.5)

        # Re-check checkpoint path (it might have been saved during shutdown)
        checkpoint_path = self._get_checkpoint_path()
        if not checkpoint_path.exists():
            self.status_label.setText("Model already reset or not found.")
            return

        # Archive the old model with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = checkpoint_path.stem  # e.g., 'checkpoint' or 'checkpoint_unet_deep'
        archive_name = f"{base_name}_old_{timestamp}.pth"
        archive_path = self.project_dir / archive_name

        try:
            # Rename old checkpoint
            checkpoint_path.rename(archive_path)
            print(f"[Reset] Archived checkpoint to: {archive_path}")

            # Verify the checkpoint is actually gone
            if checkpoint_path.exists():
                print(f"[Reset] WARNING: Checkpoint still exists after rename, deleting...")
                checkpoint_path.unlink()

            # Also archive final checkpoint if it exists
            final_path = self.project_dir / 'checkpoint_final.pth'
            if final_path.exists():
                final_archive = self.project_dir / f"checkpoint_final_old_{timestamp}.pth"
                final_path.rename(final_archive)

            # Final verification
            if checkpoint_path.exists():
                print(f"[Reset] ERROR: Checkpoint STILL exists: {checkpoint_path}")
                QMessageBox.warning(self, "Warning",
                    f"Could not fully remove checkpoint.\nPlease manually delete:\n{checkpoint_path}")
            else:
                print(f"[Reset] Success: Checkpoint removed")
                self.status_label.setText(f"Model archived as {archive_name}. Ready to train fresh!")

                # If in multi-user mode as joinee, restart model check to re-obtain from host
                if self._multi_user_enabled and not self._is_host:
                    print("[Reset] Restarting model check for multi-user joinee")
                    self._has_received_global_model = False
                    self._model_check_attempts = 0
                    # Request immediately, then start periodic check as backup
                    if self._sync_client and self._sync_client.is_connected:
                        print("[Reset] Requesting global model immediately...")
                        self._sync_client.request_global_model()
                    self._model_check_timer.start(3000)

                QMessageBox.information(
                    self, "Model Reset",
                    f"Old model archived as:\n{archive_name}\n\n"
                    "You can now start training a new model from scratch."
                )

        except Exception as e:
            print(f"[Reset] Exception: {e}")
            QMessageBox.warning(self, "Error", f"Failed to archive model:\n{e}")

    def start_training(self):
        """Start training worker."""
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project directory set")
            return

        train_files = list(self.train_images_dir.glob("*.tif")) if self.train_images_dir else []
        if not train_files:
            QMessageBox.warning(self, "No Data", "No training data - paint some masks first!")
            return

        from ..workers.train_worker import TrainWorker

        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.stop()
            # Use timeout to avoid freezing UI - wait max 3 seconds
            if not self.train_worker.wait(3000):
                print("[Training] Worker didn't stop in 3s, forcing termination")
                self.train_worker.terminate()
                self.train_worker.wait(1000)  # Brief wait after terminate
            self.train_btn.setText("Start Training")
            self.train_progress.setVisible(False)
            self.arch_combo.setEnabled(True)  # Re-enable architecture selection
            # Notify predict worker that training stopped - can use GPU now
            if self.predict_worker:
                self.predict_worker.set_training_active(False)
            # Emit training stopped signal
            self.training_stopped.emit()
            # Request fresh prediction with main checkpoint
            if self.show_predictions:
                self._request_viewport_prediction(immediate=True)
            return

        # Get architecture-specific checkpoint path
        checkpoint_path = self._get_checkpoint_path()

        # Check if checkpoint exists - if so, resume from it
        resume_checkpoint = None
        print(f"[Training] Architecture: {self.current_architecture}")
        print(f"[Training] Checkpoint path: {checkpoint_path}")
        print(f"[Training] Checkpoint exists: {checkpoint_path.exists()}")
        if checkpoint_path.exists():
            # Double-check the file is actually readable and has content
            try:
                file_size = checkpoint_path.stat().st_size
                print(f"[Training] Checkpoint size: {file_size} bytes")
                if file_size > 1000:  # Valid checkpoint should be > 1KB
                    resume_checkpoint = str(checkpoint_path)
                    print(f"[Training] Will RESUME from: {resume_checkpoint}")
                else:
                    print(f"[Training] Checkpoint too small, starting fresh")
            except Exception as e:
                print(f"[Training] Error checking checkpoint: {e}")
        else:
            print(f"[Training] No checkpoint found, starting FRESH")

        config = {
            'train_images': str(self.train_images_dir),
            'train_masks': str(self.train_masks_dir),
            'checkpoint_path': str(checkpoint_path),
            'resume_checkpoint': resume_checkpoint,  # Resume from existing checkpoint
            'num_epochs': self.config.get('num_epochs', 5000),
            'batch_size': self.config.get('batch_size', 2),
            'tile_size': 256,
            'learning_rate': self.config.get('learning_rate', 1e-4),
            'architecture': self.current_architecture,  # Pass architecture to worker
        }

        self.train_worker = TrainWorker(config)
        self.train_worker.progress.connect(self._on_training_progress)
        self.train_worker.finished.connect(self._on_training_finished)
        self.train_worker.log.connect(self._on_training_log)
        # Forward loss updates for live plot
        self.train_worker.loss_updated.connect(self.loss_updated.emit)
        # Use snapshots for predictions during training (dual checkpoint system)
        self.train_worker.snapshot_created.connect(self._on_local_snapshot_created)

        # Connect multi-user sync signal
        if self._multi_user_enabled:
            self.train_worker.set_weights_export_interval(self._sync_interval_epochs)
            self.train_worker.weights_exported.connect(self._on_weights_exported)
            # Host: connect snapshot signal to broadcast to clients
            if self._is_host:
                self.train_worker.snapshot_created.connect(self._on_snapshot_created)

        # Connect SAM2 extraction progress signal if this is a SAM2 architecture
        if 'sam2' in self.current_architecture.lower():
            self.train_worker.sam2_extraction_progress.connect(self._on_sam2_extraction_progress)

        self.train_worker.start()

        # Stop refiner worker during main training to avoid GPU conflicts and confusing logs
        if self.refiner_worker is not None:
            print("[Refiner] Stopping worker (main training started)")
            self.refiner_worker.stop()
            if not self.refiner_worker.wait(2000):
                self.refiner_worker.terminate()
                self.refiner_worker.wait(500)
            self.refiner_worker = None
            # Also turn off refiner mode UI
            if self.refiner_mode:
                self.refiner_mode = False
                self.refiner_btn.setText("Refiner: OFF")
                self.refiner_btn.setStyleSheet("")
                self.refiner_edits_label.setVisible(False)
                self.reset_refiner_btn.setVisible(False)
                self.canvas.set_suggestion(None)

        # Notify predict worker that training started - must use CPU
        if self.predict_worker:
            self.predict_worker.set_training_active(True)

        self.train_btn.setText("Stop Training")
        self.train_progress.setVisible(True)
        self.arch_combo.setEnabled(False)  # Disable architecture selection during training

        # Emit training started signal
        self.training_started.emit()

        arch_name = self._arch_id_to_name.get(self.current_architecture, self.current_architecture)
        if resume_checkpoint:
            self._show_temp_status(f"Resuming {arch_name} training...")
        else:
            self._show_temp_status(f"Starting {arch_name} training from scratch...")

    def _on_training_progress(self, epoch, total, train_loss, val_loss):
        self.train_progress.setValue(int(100 * epoch / total))
        self._last_progress_message = f"Training: Epoch {epoch}/{total}, Loss: {train_loss:.4f}"
        self.status_label.setText(self._last_progress_message)

    def _on_training_finished(self, success, result):
        self.train_btn.setText("Start Training")
        self.train_progress.setVisible(False)
        self.arch_combo.setEnabled(True)  # Re-enable architecture selection

        # Notify predict worker that training stopped - can use GPU now
        if self.predict_worker:
            self.predict_worker.set_training_active(False)

        # Emit training stopped signal
        self.training_stopped.emit()

        # Request fresh prediction with main checkpoint
        if self.show_predictions:
            self._request_viewport_prediction(immediate=True)

        if success:
            self.status_label.setText("Training complete!")
            self.training_complete.emit(result)
        else:
            self.status_label.setText(f"Training failed: {result}")

    def _on_training_log(self, message):
        """Handle log messages from training worker."""
        print(f"[Training] {message}")
        # Update status for important messages (temporarily)
        if "Resuming" in message or "device" in message.lower():
            self._show_temp_status(message)

    def _on_sam2_extraction_progress(self, current: int, total: int, message: str):
        """Handle SAM2 feature extraction progress - just update status label."""
        if current >= total:
            self.status_label.setText("SAM2 features ready! Training starting...")
        else:
            self.status_label.setText(f"☕ SAM2: {current}/{total} - {message}")

    def _show_temp_status(self, message, duration_ms=5000):
        """Show a temporary status message that reverts after duration."""
        self.status_label.setText(message)
        self._status_timer.start(duration_ms)

    def _restore_status_message(self):
        """Restore the last training progress message."""
        if self._last_progress_message:
            self.status_label.setText(self._last_progress_message)

    # =========================================================================
    # Multi-User Collaborative Training
    # =========================================================================

    def enable_multi_user(self, is_host: bool, server=None, client=None):
        """
        Enable multi-user collaborative training mode.

        Args:
            is_host: True if this user is hosting the session
            server: AggregationServer instance (if hosting)
            client: SyncClient instance (for all users)
        """
        # Skip if already enabled with same server/client
        if self._multi_user_enabled and self._sync_client == client and self._aggregation_server == server:
            print("[MultiUser] Already enabled, skipping")
            return

        self._multi_user_enabled = True
        self._is_host = is_host
        self._aggregation_server = server
        self._sync_client = client

        # Connect signals (only if not already connected)
        if client:
            try:
                client.global_model_received.connect(self._on_global_model_received)
                client.sync_status.connect(self._on_sync_status)
                client.error.connect(self._on_sync_error)
                # In relay mode, host needs to respond when users join or request model
                if is_host:
                    client.user_joined_room.connect(self._on_user_joined_relay)
                    client.model_requested.connect(self._on_model_requested_relay)
                    # Host receives training data from clients
                    client.training_data_received.connect(self._on_training_data_received)
                else:
                    # Client receives snapshots from host
                    client.snapshot_received.connect(self._on_snapshot_received)
            except TypeError:
                pass  # Already connected

        if server:
            try:
                server.aggregation_complete.connect(self._on_aggregation_complete)
            except TypeError:
                pass  # Already connected

        # Load settings from config
        self._sync_interval_epochs = self.config.get('multi_user_sync_interval', 5)
        self._blend_ratio = self.config.get('multi_user_blend_ratio', 0.5)

        # Update train worker export interval and connect signal if it exists
        if self.train_worker:
            self.train_worker.set_weights_export_interval(self._sync_interval_epochs)
            try:
                self.train_worker.weights_exported.connect(self._on_weights_exported)
            except TypeError:
                pass  # Already connected
            # Host: connect snapshot signal to broadcast to clients
            if is_host:
                try:
                    self.train_worker.snapshot_created.connect(self._on_snapshot_created)
                except TypeError:
                    pass  # Already connected

        # If host, initialize server with current model weights (so joiners receive them)
        if is_host and server:
            self._initialize_server_with_current_weights(server)

        # For relay host: weights are shared when users join (user_joined signal)
        # or when users request the model (model_requested signal)
        # No need to share immediately on room creation

        # Request global model from server (in case we missed it during connection)
        # This is especially important for joinees who connected before going to Training page
        # Host doesn't need to request - they ARE the source of truth
        if not is_host and client and client.is_connected:
            print("[MultiUser] Requesting global model from server...")
            client.request_global_model()

        # Start periodic model check for joinees (stops once model is received)
        if not is_host:
            self._has_received_global_model = False
            self._model_check_attempts = 0
            self._model_check_timer.start(3000)  # Check every 3 seconds
            print("[MultiUser] Started periodic model check for joinee")

        self._show_temp_status("Multi-user mode enabled")
        print(f"[MultiUser] Enabled - Host: {is_host}, Sync every {self._sync_interval_epochs} epochs")

    def _initialize_server_with_current_weights(self, server):
        """Load current checkpoint and set as global weights on server."""
        import torch

        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path and checkpoint_path.exists():
            try:
                print(f"[MultiUser] Loading host weights from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

                # Debug: show checkpoint structure
                if isinstance(checkpoint, dict):
                    print(f"[MultiUser] Checkpoint keys: {list(checkpoint.keys())}")

                # Extract model state dict (handle both key naming conventions)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        weights = checkpoint['model_state_dict']
                    elif 'model_state' in checkpoint:
                        weights = checkpoint['model_state']
                    else:
                        weights = checkpoint  # Assume it's just the state dict
                else:
                    weights = checkpoint

                tensor_count = sum(1 for v in weights.values() if hasattr(v, 'shape'))
                print(f"[MultiUser] Setting global weights: {tensor_count} tensors")
                if tensor_count < 10:
                    print(f"[MultiUser] WARNING: Weight keys: {list(weights.keys())}")

                server.set_global_weights(weights)
                print(f"[MultiUser] Server initialized with host's model weights")

                # Broadcast to any clients that connected before we initialized
                server.broadcast_global_model()
                print(f"[MultiUser] Broadcast global model to {server.client_count} connected clients")
                self._show_temp_status("Sharing model with session")

            except Exception as e:
                print(f"[MultiUser] Warning: Could not load checkpoint for sharing: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[MultiUser] No checkpoint to share - joiners will start from scratch")

    def _check_for_global_model(self):
        """Periodic check for joinees to request global model if they don't have it."""
        # Stop if we've received the model or are no longer in multi-user mode
        if self._has_received_global_model or not self._multi_user_enabled or self._is_host:
            self._model_check_timer.stop()
            return

        # Check if we have a checkpoint file now
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path and checkpoint_path.exists():
            print("[MultiUser] Checkpoint exists, stopping model check")
            self._has_received_global_model = True
            self._model_check_timer.stop()
            return

        # Skip if we're currently receiving a chunked transfer
        if self._sync_client and self._sync_client.is_receiving_chunks:
            print("[MultiUser] Chunked transfer in progress, skipping request")
            return

        # Limit attempts to avoid spamming forever
        self._model_check_attempts += 1
        if self._model_check_attempts > 20:  # Stop after ~60 seconds
            print("[MultiUser] Max model check attempts reached, stopping")
            self._model_check_timer.stop()
            return

        # Request the model again
        if self._sync_client and self._sync_client.is_connected:
            print(f"[MultiUser] Requesting global model (attempt {self._model_check_attempts})...")
            self._sync_client.request_global_model()

    def disable_multi_user(self):
        """Disable multi-user collaborative training mode."""
        self._multi_user_enabled = False
        self._is_host = False
        self._has_received_global_model = False
        self._model_check_timer.stop()

        # Disconnect signals
        if self._sync_client:
            try:
                self._sync_client.global_model_received.disconnect(self._on_global_model_received)
                self._sync_client.sync_status.disconnect(self._on_sync_status)
                self._sync_client.error.disconnect(self._on_sync_error)
            except TypeError:
                pass  # Already disconnected
            try:
                self._sync_client.user_joined_room.disconnect(self._on_user_joined_relay)
            except TypeError:
                pass  # Already disconnected or never connected
            try:
                self._sync_client.model_requested.disconnect(self._on_model_requested_relay)
            except TypeError:
                pass  # Already disconnected or never connected
            try:
                self._sync_client.training_data_received.disconnect(self._on_training_data_received)
            except TypeError:
                pass  # Already disconnected or never connected
            try:
                self._sync_client.snapshot_received.disconnect(self._on_snapshot_received)
            except TypeError:
                pass  # Already disconnected or never connected

        if self._aggregation_server:
            try:
                self._aggregation_server.aggregation_complete.disconnect(self._on_aggregation_complete)
            except TypeError:
                pass

        self._sync_client = None
        self._aggregation_server = None

        # Disable weight export and disconnect signals in train worker
        if self.train_worker:
            self.train_worker.set_weights_export_interval(0)
            try:
                self.train_worker.snapshot_created.disconnect(self._on_snapshot_created)
            except TypeError:
                pass  # Already disconnected or never connected

        self._show_temp_status("Multi-user mode disabled")
        print("[MultiUser] Disabled")

    def _on_weights_exported(self, weights: dict, epoch: int, loss: float):
        """
        Handle weights exported from training worker.

        Called every sync_interval epochs during training.
        Sends weights to the server for aggregation.
        """
        if not self._multi_user_enabled or not self._sync_client:
            return

        # Count training samples (approximate from dataset)
        num_samples = 100  # Default estimate

        # Send weights to server
        self._sync_client.send_weights(weights, epoch, loss, num_samples)
        self._last_sync_epoch = epoch
        print(f"[MultiUser] Sent weights for epoch {epoch} (loss: {loss:.4f})")

    def _try_apply_pending_weights(self):
        """
        Try to apply pending weights if they exist and we now have a valid checkpoint path.

        Called after project_dir or architecture is set, in case weights were received
        before the project was fully configured (e.g., joinee connected before loading project).
        """
        if not self._pending_global_weights:
            return

        checkpoint_path = self._get_checkpoint_path()
        if not checkpoint_path:
            return

        print("[MultiUser] Applying pending weights now that project is configured")
        weights = self._pending_global_weights
        self._pending_global_weights = None
        self._apply_global_model(weights)

    def _on_global_model_received(self, global_weights: dict):
        """
        Handle receiving global model from server.

        The global model is the aggregated result from all users.
        We blend it with our local model to incorporate collective learning.

        Note: The host receives aggregated models via _on_aggregation_complete instead,
        so we skip here to avoid the host applying their own initial model back to themselves.
        """
        if not self._multi_user_enabled:
            return

        # Host receives aggregated models via _on_aggregation_complete signal
        # Skip here to avoid host applying their own initial model
        if getattr(self, '_is_host', False):
            print("[MultiUser] Host skipping global_model_received (uses aggregation_complete instead)")
            return

        print("[MultiUser] Received global model from server")

        # Mark that we've received a model and stop the periodic check
        self._has_received_global_model = True
        self._model_check_timer.stop()

        # If training is in progress, queue the weights for later application
        if self.train_worker and self.train_worker.isRunning():
            self._pending_global_weights = global_weights
            self._show_temp_status("Global model received (will apply after epoch)")
        else:
            # Apply immediately if not training
            self._apply_global_model(global_weights)

    def _on_aggregation_complete(self, global_weights: dict):
        """
        Handle aggregation complete (host only).

        The host receives this when aggregation finishes.
        Apply the global model to our local training as well.
        """
        if not self._multi_user_enabled:
            return

        print("[MultiUser] Aggregation complete (host)")

        # Host also receives the aggregated model
        if self.train_worker and self.train_worker.isRunning():
            self._pending_global_weights = global_weights
        else:
            self._apply_global_model(global_weights)

    def _apply_global_model(self, global_weights: dict):
        """
        Apply global model weights with blending.

        Blends global weights with local weights:
        new_weights = (1 - blend_ratio) * local + blend_ratio * global

        With blend_ratio=0.5 (default):
        Each user keeps 50% their own learning + 50% collective learning

        If no local checkpoint exists, uses global weights directly (100%).
        """
        try:
            from ..network import blend_weights
            import torch

            # Debug: show received weights info
            tensor_count = sum(1 for v in global_weights.values() if hasattr(v, 'shape'))
            print(f"[MultiUser] _apply_global_model received: {tensor_count} tensors")
            if tensor_count < 10:
                print(f"[MultiUser] WARNING: Global weight keys: {list(global_weights.keys())}")

            checkpoint_path = self._get_checkpoint_path()
            print(f"[MultiUser] checkpoint_path={checkpoint_path}, project_dir={self.project_dir}, arch={self.current_architecture}")

            # If no local checkpoint, use global weights directly (joinee starting fresh)
            if not checkpoint_path or not checkpoint_path.exists():
                print(f"[MultiUser] No local checkpoint - using global weights directly (path exists: {checkpoint_path.exists() if checkpoint_path else 'N/A'})")

                # Check if we have a project directory
                if not checkpoint_path:
                    print("[MultiUser] No project directory set yet - storing weights for later")
                    print(f"[MultiUser] project_dir={self.project_dir}, current_architecture={self.current_architecture}")
                    # Store weights to apply when project is loaded
                    self._pending_global_weights = global_weights
                    self._show_temp_status("Model received (will apply when project loads)")
                    return

                # Ensure directory exists
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                # Save global weights as new checkpoint
                new_checkpoint = {
                    'epoch': 0,
                    'model_state_dict': global_weights,
                    'loss': 0.0
                }
                torch.save(new_checkpoint, checkpoint_path)
                print(f"[MultiUser] Saved global model to {checkpoint_path}")

                # Notify prediction worker to reload
                if self.predict_worker:
                    self.predict_worker.set_checkpoint(str(checkpoint_path))
                    self.predict_worker.set_architecture(self.current_architecture)

                self._show_temp_status("Received model from session")
                return

            # Load local weights for blending
            local_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(local_checkpoint, dict) and 'model_state_dict' in local_checkpoint:
                local_weights = local_checkpoint['model_state_dict']
            elif isinstance(local_checkpoint, dict) and 'model_state' in local_checkpoint:
                local_weights = local_checkpoint['model_state']
            else:
                local_weights = local_checkpoint  # Assume it's just the state dict

            # Blend weights
            blended = blend_weights(local_weights, global_weights, self._blend_ratio)

            # Save blended weights back to checkpoint
            if isinstance(local_checkpoint, dict):
                if 'model_state_dict' in local_checkpoint:
                    local_checkpoint['model_state_dict'] = blended
                elif 'model_state' in local_checkpoint:
                    local_checkpoint['model_state'] = blended
                else:
                    local_checkpoint = blended
            else:
                local_checkpoint = blended

            torch.save(local_checkpoint, checkpoint_path)

            # Notify prediction worker to reload
            if self.predict_worker:
                self.predict_worker.set_checkpoint(str(checkpoint_path))

            self._show_temp_status(f"Applied global model (blend: {self._blend_ratio:.0%})")
            print(f"[MultiUser] Applied global model with blend ratio {self._blend_ratio}")

        except Exception as e:
            print(f"[MultiUser] Error applying global model: {e}")
            import traceback
            traceback.print_exc()

    def _on_sync_status(self, message: str):
        """Handle sync status message from client."""
        print(f"[MultiUser] {message}")

    def _on_sync_error(self, error: str):
        """Handle sync error from client."""
        print(f"[MultiUser] Error: {error}")
        self._show_temp_status(f"Sync error: {error}")

    def _on_user_joined_relay(self, display_name: str):
        """
        Handle when a user joins the relay room (host only).

        Share architecture info and current weights with the new user.
        """
        if not self._is_host or not self._sync_client:
            return

        print(f"[MultiUser] User '{display_name}' joined - sharing architecture and weights...")
        self._show_temp_status(f"{display_name} joined - sharing model...")

        # Send architecture info first so joinee can lock to it
        self._sync_client.send_architecture(self.current_architecture)

        # Share current checkpoint weights
        self._share_current_weights_to_relay()

    def _on_model_requested_relay(self, requester_id: str):
        """
        Handle when a user requests the global model (host only, relay mode).

        Share architecture and current weights with the requesting user.
        """
        if not self._is_host or not self._sync_client:
            return

        print(f"[MultiUser] User '{requester_id}' requested model - sharing architecture and weights...")
        self._show_temp_status("Sharing model with user...")

        # Send architecture info first
        self._sync_client.send_architecture(self.current_architecture)

        # Share current checkpoint weights
        self._share_current_weights_to_relay()

    def _share_current_weights_to_relay(self):
        """Share current checkpoint weights via relay (for new joiners)."""
        import torch
        import time

        if not self._sync_client or not self._sync_client.is_connected:
            return

        # Check cooldown to prevent spamming (large models take time to transfer)
        now = time.time()
        time_since_last = now - self._last_model_share_time
        if time_since_last < self._model_share_cooldown:
            remaining = self._model_share_cooldown - time_since_last
            print(f"[MultiUser] Model share cooldown active, skipping ({remaining:.0f}s remaining)")
            return

        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path and checkpoint_path.exists():
            try:
                print(f"[MultiUser] Loading weights from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

                # Extract model weights from checkpoint
                if isinstance(checkpoint, dict):
                    if 'model_state' in checkpoint:
                        weights = checkpoint['model_state']
                        epoch = checkpoint.get('epoch', 0)
                    elif 'model_state_dict' in checkpoint:
                        weights = checkpoint['model_state_dict']
                        epoch = checkpoint.get('epoch', 0)
                    else:
                        weights = checkpoint
                        epoch = 0
                else:
                    weights = checkpoint
                    epoch = 0

                # Send weights to relay (all users in room will receive)
                print(f"[MultiUser] Sending weights (epoch {epoch}) to relay...")
                self._sync_client.send_weights(weights, epoch, 0.0, 1)
                self._last_model_share_time = now  # Update cooldown timestamp
                self._show_temp_status("Shared model with new user")

            except Exception as e:
                print(f"[MultiUser] Error sharing weights: {e}")
                import traceback
                traceback.print_exc()
        else:
            # No checkpoint exists - create fresh initialized model and share it
            # This ensures everyone starts from the same random weights
            print(f"[MultiUser] No checkpoint found - creating fresh {self.current_architecture} model...")
            try:
                from ..models.architectures import get_model_class

                # Create fresh model with random initialization
                model_class = get_model_class(self.current_architecture)
                fresh_model = model_class()
                weights = fresh_model.state_dict()

                # Save as initial checkpoint so host uses same weights
                initial_checkpoint = {
                    'model_state': weights,
                    'epoch': 0,
                    'train_loss': 0.0,
                    'architecture': self.current_architecture
                }
                torch.save(initial_checkpoint, checkpoint_path)
                print(f"[MultiUser] Saved initial checkpoint to {checkpoint_path}")

                # Share the fresh weights with all users
                print(f"[MultiUser] Sharing fresh initialized weights...")
                self._sync_client.send_weights(weights, 0, 0.0, 1)
                self._last_model_share_time = now
                self._show_temp_status("Shared fresh model with session")

            except Exception as e:
                print(f"[MultiUser] Error creating fresh model: {e}")
                import traceback
                traceback.print_exc()

    def _on_training_data_received(self, image_bytes: bytes, mask_bytes: bytes, metadata: dict):
        """
        Handle training data received from a client (host only).

        Saves the crop to the training directory and optionally triggers dataset reload.
        """
        if not self._is_host:
            return

        sender = metadata.get("display_name", "Unknown")
        crop_size = metadata.get("crop_size", 0)
        timestamp = metadata.get("timestamp", 0)
        sender_id = metadata.get("user_id", "unknown")[:8]

        print(f"[MultiUser] Received training crop from {sender} ({crop_size}x{crop_size})")

        if not self.train_images_dir or not self.train_masks_dir:
            print("[MultiUser] No training directories set, cannot save received crop")
            return

        try:
            import io
            from PIL import Image

            # Decode images
            img = Image.open(io.BytesIO(image_bytes))
            mask = Image.open(io.BytesIO(mask_bytes))

            # Generate filename
            crop_id = f"remote_{sender_id}_{timestamp}"

            # Save to training directories
            img_path = self.train_images_dir / f"{crop_id}.tif"
            mask_path = self.train_masks_dir / f"{crop_id}.tif"

            img.save(img_path, compression='tiff_lzw')
            mask.save(mask_path, compression='tiff_lzw')

            print(f"[MultiUser] Saved training crop from {sender} to {crop_id}.tif")
            self._show_temp_status(f"Received crop from {sender}")

            # Request dataset reload in train worker
            if self.train_worker and self.train_worker.isRunning():
                self.train_worker.request_dataset_reload()

        except Exception as e:
            print(f"[MultiUser] Error saving training crop: {e}")
            import traceback
            traceback.print_exc()

    def _on_snapshot_received(self, weights: dict, snapshot_id: int):
        """
        Handle snapshot model received from host (client only).

        Updates the prediction worker to use the new snapshot model.
        """
        if self._is_host:
            return

        print(f"[MultiUser] Received snapshot #{snapshot_id}")

        try:
            import torch

            # Get checkpoint path
            checkpoint_path = self._get_checkpoint_path()
            if not checkpoint_path:
                print("[MultiUser] No checkpoint path set, cannot save snapshot")
                return

            # Save snapshot to checkpoint
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            new_checkpoint = {
                'epoch': 0,
                'model_state_dict': weights,
                'snapshot_id': snapshot_id,
            }
            torch.save(new_checkpoint, checkpoint_path)
            print(f"[MultiUser] Saved snapshot #{snapshot_id} to {checkpoint_path}")

            # Notify prediction worker to reload
            if self.predict_worker:
                self.predict_worker.set_checkpoint(str(checkpoint_path))
                self.predict_worker.set_architecture(self.current_architecture)

            self._show_temp_status(f"Snapshot #{snapshot_id} applied")

        except Exception as e:
            print(f"[MultiUser] Error applying snapshot: {e}")
            import traceback
            traceback.print_exc()

    def _on_local_snapshot_created(self, weights: dict, snapshot_id: int):
        """
        Handle snapshot created by local train worker.

        Triggers a prediction request if predictions are enabled,
        so user sees updated model results immediately.
        """
        print(f"[Training] Snapshot #{snapshot_id} created")

        # Request new prediction immediately if predictions are enabled
        if self.show_predictions:
            self._request_viewport_prediction(immediate=True)

    def _on_snapshot_created(self, weights: dict, snapshot_id: int):
        """
        Handle snapshot created by local train worker (host only).

        Broadcasts the snapshot to all connected clients.
        """
        if not self._is_host:
            return

        if not self._sync_client or not self._sync_client.is_connected:
            return

        print(f"[MultiUser] Broadcasting snapshot #{snapshot_id} to clients")
        self._sync_client.send_snapshot(weights, snapshot_id)

    def _create_session(self):
        """Create a multi-user session from the Training page."""
        from PyQt6.QtWidgets import QMessageBox, QInputDialog
        try:
            from ..network import SyncClient, DEFAULT_RELAY_URL
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        if not DEFAULT_RELAY_URL:
            QMessageBox.warning(
                self, "Relay Not Configured",
                "No relay server configured.\n\n"
                "Please set up relay_config.txt in the network folder.\n"
                "See SETUP_GUIDE.md in relay_server/ for instructions."
            )
            return

        # Get display name
        import os
        default_name = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        name, ok = QInputDialog.getText(
            self, "Create Session",
            "Enter your display name:",
            text=default_name + "_host"
        )
        if not ok or not name.strip():
            return

        # Create client and connect
        self._sync_client = SyncClient(parent=self)
        self._sync_client.display_name = name.strip()
        self._sync_client.room_created.connect(self._on_session_room_created)
        self._sync_client.connected.connect(lambda: self._on_session_connected(True))
        self._sync_client.disconnected.connect(self._on_session_disconnected)
        self._sync_client.error.connect(self._on_session_error)
        self._sync_client.user_list_updated.connect(self._on_session_user_list)
        self._sync_client.sync_status.connect(self._on_session_sync_status)

        self._sync_client.connect_to_relay(DEFAULT_RELAY_URL, create_room=True)
        self.session_create_btn.setEnabled(False)
        self.session_join_btn.setEnabled(False)
        self._show_temp_status("Creating session...")

    def _join_session(self):
        """Join an existing session from the Training page."""
        from PyQt6.QtWidgets import QMessageBox, QInputDialog
        try:
            from ..network import SyncClient, DEFAULT_RELAY_URL
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        if not DEFAULT_RELAY_URL:
            QMessageBox.warning(
                self, "Relay Not Configured",
                "No relay server configured.\n\n"
                "Please set up relay_config.txt in the network folder.\n"
                "See SETUP_GUIDE.md in relay_server/ for instructions."
            )
            return

        # Get room code
        code, ok = QInputDialog.getText(
            self, "Join Session",
            "Enter 6-character session code:"
        )
        if not ok or not code.strip():
            return

        code = code.strip().upper()
        if len(code) != 6:
            QMessageBox.warning(self, "Invalid Code", "Session code must be 6 characters.")
            return

        # Get display name
        import os
        default_name = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        name, ok = QInputDialog.getText(
            self, "Join Session",
            "Enter your display name:",
            text=default_name
        )
        if not ok or not name.strip():
            return

        # Create client and connect
        self._sync_client = SyncClient(parent=self)
        self._sync_client.display_name = name.strip()
        self._sync_client.room_joined.connect(lambda c: self._on_session_room_joined(c))
        self._sync_client.connected.connect(lambda: self._on_session_connected(False))
        self._sync_client.disconnected.connect(self._on_session_disconnected)
        self._sync_client.error.connect(self._on_session_error)
        self._sync_client.user_list_updated.connect(self._on_session_user_list)
        self._sync_client.sync_status.connect(self._on_session_sync_status)

        self._sync_client.connect_to_relay(DEFAULT_RELAY_URL, room_code=code)
        self.session_create_btn.setEnabled(False)
        self.session_join_btn.setEnabled(False)
        self._show_temp_status(f"Joining session {code}...")

    def _disconnect_session(self):
        """Disconnect from the current session."""
        self.disable_multi_user()
        if self._sync_client:
            self._sync_client.disconnect()
            self._sync_client = None
        self._aggregation_server = None
        self._update_session_ui(connected=False)
        self._show_temp_status("Disconnected from session")

    def _on_session_room_created(self, room_code: str):
        """Handle room created - we're the host."""
        print(f"[Training] Room created: {room_code}")
        self.session_status_label.setText(f"Session: {room_code}")
        self.session_status_label.setVisible(True)
        self._update_session_ui(connected=True)
        # Enable multi-user as host
        self.enable_multi_user(is_host=True, server=None, client=self._sync_client)
        self._show_temp_status(f"Session created! Code: {room_code}")

    def _on_session_room_joined(self, room_code: str):
        """Handle room joined - we're a client."""
        print(f"[Training] Joined room: {room_code}")
        self.session_status_label.setText(f"Session: {room_code}")
        self.session_status_label.setVisible(True)
        self._update_session_ui(connected=True)
        # Enable multi-user as joinee
        self.enable_multi_user(is_host=False, server=None, client=self._sync_client)

    def _on_session_connected(self, is_host: bool):
        """Handle successful connection."""
        print(f"[Training] Connected (host={is_host})")

    def _on_session_disconnected(self):
        """Handle disconnection."""
        print("[Training] Disconnected from session")
        self._update_session_ui(connected=False)
        self.disable_multi_user()

    def _on_session_error(self, error: str):
        """Handle session error."""
        from PyQt6.QtWidgets import QMessageBox
        print(f"[Training] Session error: {error}")
        QMessageBox.warning(self, "Session Error", error)
        self._update_session_ui(connected=False)

    def _on_session_user_list(self, users: list):
        """Handle user list update."""
        count = len(users)
        print(f"[Training] {count} user(s) in session")

    def _on_session_sync_status(self, status: str):
        """Handle sync status update."""
        self._show_temp_status(status)

    def _update_session_ui(self, connected: bool):
        """Update UI based on connection state (called by wizard)."""
        # Session UI is now in the wizard sidebar, not the training page
        pass

    def set_multi_user_state(self, server, client, is_relay_host=False):
        """
        Set multi-user state from setup page or wizard.

        Called when transitioning from setup to training or when session is created.
        """
        if server or client:
            # User is host if they have a server OR if they created the relay room
            is_host = (server is not None) or is_relay_host
            self.enable_multi_user(
                is_host=is_host,
                server=server,
                client=client
            )

    # =========================================================================
    # Refiner Mode - learns from user edits in real-time
    # =========================================================================

    def _init_refiner_worker(self):
        """Initialize the refiner worker."""
        from ..workers.refiner_worker import RefinerWorker

        self.refiner_worker = RefinerWorker(project_dir=self.project_dir)
        self.refiner_worker.training_started.connect(self._on_refiner_training_started)
        self.refiner_worker.training_progress.connect(self._on_refiner_training_progress)
        self.refiner_worker.training_complete.connect(self._on_refiner_training_complete)
        self.refiner_worker.prediction_ready.connect(self._on_refiner_prediction_ready)
        self.refiner_worker.start()
        print("[Refiner] Worker initialized and started")

    def _toggle_refiner_mode(self):
        """Toggle refiner mode on/off."""
        self.refiner_mode = not self.refiner_mode

        if self.refiner_mode:
            # Turning ON refiner mode
            self.refiner_btn.setText("Refiner: ON")
            self.refiner_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.refiner_edits_label.setVisible(True)
            self.reset_refiner_btn.setVisible(True)

            # Stop regular training if running
            if self.train_worker and self.train_worker.isRunning():
                self.train_worker.stop()
                if not self.train_worker.wait(3000):
                    self.train_worker.terminate()
                    self.train_worker.wait(1000)
                self.train_btn.setText("Start Training")
                self.train_progress.setVisible(False)
                self.arch_combo.setEnabled(True)

            # Stop regular predictions
            if self.show_predictions:
                self.show_pred_check.setChecked(False)
                self.show_predictions = False

            # Initialize refiner worker if not already done
            if self.refiner_worker is None:
                self._init_refiner_worker()

            # Update sample count display
            if self.refiner_worker:
                sample_count = self.refiner_worker.get_edit_count()
                self.refiner_edits_label.setText(f"({sample_count} samples)")

            # Clear mask tracking for fresh start
            self._refiner_mask_at_load.clear()

            # Track current slice's mask as the "before" state
            idx = self.current_slice_index
            if idx in self.masks:
                self._refiner_mask_at_load[idx] = self.masks[idx].copy()
                mask_px = np.sum(self.masks[idx] > 127)
                print(f"[Refiner] Toggle ON: tracking slice {idx} 'before' state: {mask_px}px")

            self.status_label.setText("Refiner ON - Paint edits, then press Tab to capture crop")

        else:
            # Turning OFF refiner mode
            self.refiner_btn.setText("Refiner: OFF")
            self.refiner_btn.setStyleSheet("")
            self.refiner_edits_label.setVisible(False)
            self.reset_refiner_btn.setVisible(False)

            # Stop refiner worker to prevent background training
            if self.refiner_worker is not None:
                print("[Refiner] Stopping worker (refiner mode OFF)")
                self.refiner_worker.stop()
                if not self.refiner_worker.wait(2000):
                    self.refiner_worker.terminate()
                    self.refiner_worker.wait(500)
                self.refiner_worker = None

            # Clear suggestion overlay
            self.canvas.set_suggestion(None)

            self.status_label.setText("Refiner mode OFF")

    def _reset_refiner(self):
        """Reset the refiner model weights and clear training data."""
        if self.refiner_worker is None:
            return

        # Reset model weights
        self.refiner_worker.reset_model()

        # Also clear training data folders
        self.refiner_worker.clear_edits()

        self.refiner_edits_label.setText("(0 samples)")

        # Clear mask tracking for fresh start
        self._refiner_mask_at_load.clear()

        # Track current slice's mask as the new "before" state
        idx = self.current_slice_index
        if idx in self.masks:
            self._refiner_mask_at_load[idx] = self.masks[idx].copy()

        # Clear suggestion overlay
        self.canvas.set_suggestion(None)

        self.status_label.setText("Refiner reset - model and training data cleared")

    def _do_refiner_prediction(self):
        """Request a prediction from the refiner."""
        print(f"[Refiner] _do_refiner_prediction called, mode={self.refiner_mode}, worker={self.refiner_worker is not None}")
        if not self.refiner_mode or self.refiner_worker is None:
            return

        idx = self.current_slice_index
        if idx not in self.images or idx not in self.masks:
            return

        # Get viewport bounds
        bounds = self.canvas.get_viewport_bounds()
        if bounds is None:
            return

        raw_image = self.images[idx]
        current_mask = self.masks[idx]

        self.refiner_worker.request_prediction(raw_image, current_mask, bounds)

    def _on_refiner_training_started(self):
        """Handle refiner training started."""
        self.status_label.setText("Refiner: Training...")

    def _on_refiner_training_progress(self, epoch: int, total: int, loss: float):
        """Handle refiner training progress."""
        self.status_label.setText(f"Refiner: Epoch {epoch}/{total}, Loss: {loss:.4f}")

    def _on_refiner_training_complete(self):
        """Handle refiner training complete."""
        self.status_label.setText("Refiner: Training complete - predicting...")
        # Request prediction after training
        self._do_refiner_prediction()

    def _on_refiner_prediction_ready(self, prediction: np.ndarray, bounds: tuple):
        """Handle refiner prediction ready."""
        print(f"[Refiner] Prediction received in UI, shape={prediction.shape}, refiner_mode={self.refiner_mode}")
        if not self.refiner_mode:
            return

        # Show as suggestion overlay
        pred_pixels = np.sum(prediction > 127)
        print(f"[Refiner] Setting suggestion with {pred_pixels} foreground pixels")
        self.canvas.set_suggestion(prediction)
        self.canvas.toggle_suggestion_visibility(True)
        self.status_label.setText(f"Refiner: Prediction ready ({pred_pixels}px) - Space to accept")

    def cleanup(self):
        """Clean up workers when page is closed."""
        # Stop auto-save timer
        if self.auto_save_timer:
            self.auto_save_timer.stop()

        # Save current state and mask before closing
        self.save_current_slice()
        self._save_project_config()

        # Stop refiner worker
        if self.refiner_worker:
            self.refiner_worker.stop()
            if not self.refiner_worker.wait(2000):
                self.refiner_worker.terminate()
                self.refiner_worker.wait(500)
            self.refiner_worker = None

        # Stop workers with timeouts to avoid hangs
        if self.train_worker:
            self.train_worker.stop()
            if not self.train_worker.wait(3000):
                self.train_worker.terminate()
                self.train_worker.wait(1000)
            self.train_worker = None
            # Notify predict worker that training stopped
            if self.predict_worker:
                self.predict_worker.set_training_active(False)

        if self.predict_worker:
            self.predict_worker.stop()
            if not self.predict_worker.wait(2000):
                self.predict_worker.terminate()
                self.predict_worker.wait(500)
            self.predict_worker = None

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
