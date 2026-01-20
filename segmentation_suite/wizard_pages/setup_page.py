#!/usr/bin/env python3
"""
Setup page for the training wizard - folder selection and model configuration.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QFormLayout, QCheckBox, QListWidget, QDialog,
    QDialogButtonBox, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, QSettings
from PyQt6.QtGui import QFont


class SetupPage(QWidget):
    """Setup page for configuring training parameters."""

    # Signal emitted when configuration is complete
    config_ready = pyqtSignal(dict)
    # Signal emitted when a project is loaded (auto-advance to next step)
    project_loaded = pyqtSignal()
    # Signal emitted when architecture should be locked (joining a session)
    architecture_locked = pyqtSignal(str)  # architecture name
    # Signal emitted when architecture should be unlocked (disconnected)
    architecture_unlocked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title and Load Project buttons
        title_row = QHBoxLayout()
        title = QLabel("Training Setup")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title_row.addWidget(title)
        title_row.addStretch()

        # Resume Last Project button (disabled if no last project)
        self.resume_last_btn = QPushButton("Resume Last Project")
        self.resume_last_btn.setStyleSheet("QPushButton { padding: 8px 16px; }")
        self.resume_last_btn.clicked.connect(self._resume_last_project)
        title_row.addWidget(self.resume_last_btn)

        # Check if there's a last project to resume
        settings = QSettings("MOSS", "SegmentationSuite")
        last_project = settings.value("last_project_dir", "")
        if last_project and os.path.isdir(last_project):
            # Show project name in tooltip
            self.resume_last_btn.setToolTip(f"Resume: {os.path.basename(last_project)}")
        else:
            self.resume_last_btn.setEnabled(False)
            self.resume_last_btn.setToolTip("No recent project")

        load_project_btn = QPushButton("Load Existing Project")
        load_project_btn.setStyleSheet("QPushButton { padding: 8px 16px; font-weight: bold; }")
        load_project_btn.clicked.connect(self._load_existing_project)
        title_row.addWidget(load_project_btn)

        layout.addLayout(title_row)

        # Project name
        project_group = QGroupBox("Project")
        project_layout = QFormLayout(project_group)

        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("Enter project name")
        project_layout.addRow("Project Name:", self.project_name)

        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Select output directory")
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir)
        output_row.addWidget(output_btn)
        project_layout.addRow("Output Directory:", output_row)

        layout.addWidget(project_group)

        # Data paths
        data_group = QGroupBox("Training Data")
        data_layout = QFormLayout(data_group)

        # Train images
        self.train_images = QLineEdit()
        self.train_images.setPlaceholderText("Folder containing training images")
        train_img_btn = QPushButton("Browse...")
        train_img_btn.clicked.connect(lambda: self._browse_folder(self.train_images, "Select Training Images Folder"))
        train_img_row = QHBoxLayout()
        train_img_row.addWidget(self.train_images)
        train_img_row.addWidget(train_img_btn)
        data_layout.addRow("Training Images:", train_img_row)

        # Interactive painting checkbox
        self.interactive_check = QCheckBox("Paint ground truth interactively (masks optional)")
        self.interactive_check.setChecked(True)
        self.interactive_check.stateChanged.connect(self._on_interactive_changed)
        data_layout.addRow("", self.interactive_check)

        # Train masks
        self.train_masks = QLineEdit()
        self.train_masks.setPlaceholderText("(Optional) Folder containing training masks")
        self.train_mask_btn = QPushButton("Browse...")
        self.train_mask_btn.clicked.connect(lambda: self._browse_folder(self.train_masks, "Select Training Masks Folder"))
        train_mask_row = QHBoxLayout()
        train_mask_row.addWidget(self.train_masks)
        train_mask_row.addWidget(self.train_mask_btn)
        data_layout.addRow("Training Masks:", train_mask_row)

        # Validation images (optional)
        self.val_images = QLineEdit()
        self.val_images.setPlaceholderText("(Optional) Folder containing validation images")
        val_img_btn = QPushButton("Browse...")
        val_img_btn.clicked.connect(lambda: self._browse_folder(self.val_images, "Select Validation Images Folder"))
        val_img_row = QHBoxLayout()
        val_img_row.addWidget(self.val_images)
        val_img_row.addWidget(val_img_btn)
        data_layout.addRow("Validation Images:", val_img_row)

        # Validation masks (optional)
        self.val_masks = QLineEdit()
        self.val_masks.setPlaceholderText("(Optional) Folder containing validation masks")
        val_mask_btn = QPushButton("Browse...")
        val_mask_btn.clicked.connect(lambda: self._browse_folder(self.val_masks, "Select Validation Masks Folder"))
        val_mask_row = QHBoxLayout()
        val_mask_row.addWidget(self.val_masks)
        val_mask_row.addWidget(val_mask_btn)
        data_layout.addRow("Validation Masks:", val_mask_row)

        layout.addWidget(data_group)

        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)

        self.model_type = QComboBox()
        self.model_type.addItems(["New Model", "Resume from Checkpoint"])
        self.model_type.currentIndexChanged.connect(self._on_model_type_changed)
        model_layout.addRow("Model:", self.model_type)

        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Path to checkpoint file (.pth)")
        self.checkpoint_path.setEnabled(False)
        checkpoint_btn = QPushButton("Browse...")
        checkpoint_btn.clicked.connect(self._browse_checkpoint)
        self.checkpoint_btn = checkpoint_btn
        checkpoint_btn.setEnabled(False)
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint_path)
        checkpoint_row.addWidget(checkpoint_btn)
        model_layout.addRow("Checkpoint:", checkpoint_row)

        layout.addWidget(model_group)

        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)

        self.num_epochs = QSpinBox()
        self.num_epochs.setRange(10, 10000)
        self.num_epochs.setValue(5000)
        params_layout.addRow("Epochs:", self.num_epochs)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 32)
        self.batch_size.setValue(2)
        params_layout.addRow("Batch Size:", self.batch_size)

        self.tile_size = QComboBox()
        self.tile_size.addItems(["256", "512", "1024"])
        self.tile_size.setCurrentText("512")
        params_layout.addRow("Tile Size:", self.tile_size)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(6)
        self.learning_rate.setRange(0.000001, 0.01)
        self.learning_rate.setValue(0.0001)
        self.learning_rate.setSingleStep(0.00001)
        params_layout.addRow("Learning Rate:", self.learning_rate)

        layout.addWidget(params_group)

        # Multi-user training (optional)
        multiuser_group = QGroupBox("Multi-User Training (Optional)")
        multiuser_layout = QVBoxLayout(multiuser_group)

        # Description
        desc_label = QLabel("Train collaboratively with others anywhere (internet)")
        desc_label.setStyleSheet("color: #666;")
        multiuser_layout.addWidget(desc_label)

        # Status label
        self.session_status_label = QLabel("Not connected")
        self.session_status_label.setStyleSheet("color: gray;")
        multiuser_layout.addWidget(self.session_status_label)

        # Room code display (when hosting)
        self.room_code_label = QLabel("")
        self.room_code_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3; padding: 10px;")
        self.room_code_label.setVisible(False)
        multiuser_layout.addWidget(self.room_code_label)

        # Button row
        btn_row = QHBoxLayout()

        # Session buttons moved to wizard sidebar - keeping references for compatibility
        self.host_session_btn = QPushButton("Create Session")
        self.host_session_btn.setVisible(False)  # Hidden - use wizard sidebar instead
        btn_row.addWidget(self.host_session_btn)

        self.join_session_btn = QPushButton("Join Session")
        self.join_session_btn.setVisible(False)  # Hidden - use wizard sidebar instead
        btn_row.addWidget(self.join_session_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setToolTip("Leave the current session")
        self.disconnect_btn.clicked.connect(self._disconnect_session)
        self.disconnect_btn.setVisible(False)
        btn_row.addWidget(self.disconnect_btn)

        btn_row.addStretch()
        multiuser_layout.addLayout(btn_row)

        # Advanced options (LAN direct connect)
        self.advanced_btn = QPushButton("Advanced (LAN)")
        self.advanced_btn.setToolTip("Direct LAN connection (requires same network)")
        self.advanced_btn.setStyleSheet("QPushButton { color: #666; }")
        self.advanced_btn.clicked.connect(self._show_advanced_options)
        multiuser_layout.addWidget(self.advanced_btn)

        # Connected users list (visible when connected)
        self.user_list_widget = QListWidget()
        self.user_list_widget.setMaximumHeight(80)
        self.user_list_widget.setVisible(False)
        multiuser_layout.addWidget(self.user_list_widget)

        layout.addWidget(multiuser_group)

        # Multi-user state
        self._aggregation_server = None
        self._sync_client = None
        self._room_code = None
        self._is_relay_mode = False
        self._is_relay_host = False  # True if user created the relay room
        self._current_architecture = ""  # Set by interactive_training_page

        layout.addStretch()

    def _browse_folder(self, line_edit, title):
        """Browse for a folder."""
        folder = QFileDialog.getExistingDirectory(self, title)
        if folder:
            line_edit.setText(folder)

    def _browse_output(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir.setText(folder)

    def _browse_checkpoint(self):
        """Browse for checkpoint file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint",
            "", "PyTorch Checkpoints (*.pth);;All Files (*)"
        )
        if filepath:
            self.checkpoint_path.setText(filepath)

    def _on_model_type_changed(self, index):
        """Handle model type selection change."""
        is_resume = index == 1
        self.checkpoint_path.setEnabled(is_resume)
        self.checkpoint_btn.setEnabled(is_resume)

    def _on_interactive_changed(self, state):
        """Handle interactive painting checkbox change."""
        interactive = state == 2  # Qt.CheckState.Checked
        # When interactive is enabled, masks folder becomes optional
        if interactive:
            self.train_masks.setPlaceholderText("(Optional) Folder containing training masks")
        else:
            self.train_masks.setPlaceholderText("Folder containing training masks - REQUIRED")

    def get_config(self) -> dict:
        """Get the current configuration."""
        output_dir = self.output_dir.text() or os.path.expanduser("~/segmentation_projects")
        project_name = self.project_name.text() or "untitled_project"

        # Use directly loaded project dir if available (more reliable than reconstructing)
        if hasattr(self, '_loaded_project_dir') and self._loaded_project_dir:
            project_dir = self._loaded_project_dir
            print(f"[get_config] Using loaded project_dir: {project_dir}")
        else:
            project_dir = os.path.join(output_dir, project_name)
            print(f"[get_config] Constructed project_dir: {project_dir}")

        return {
            'project_name': project_name,
            'project_dir': project_dir,
            'train_images': self.train_images.text(),
            'train_masks': self.train_masks.text() or None,
            'raw_images_dir': self.train_images.text(),  # For interactive page
            'val_images': self.val_images.text() or None,
            'val_masks': self.val_masks.text() or None,
            'interactive_mode': self.interactive_check.isChecked(),
            'resume_checkpoint': self.checkpoint_path.text() if self.model_type.currentIndex() == 1 else None,
            'checkpoint_path': os.path.join(project_dir, 'checkpoint.pth'),
            'num_epochs': self.num_epochs.value(),
            'batch_size': self.batch_size.value(),
            'tile_size': int(self.tile_size.currentText()),
            'learning_rate': self.learning_rate.value(),
        }

    def validate(self) -> tuple[bool, str]:
        """Validate the configuration."""
        if not self.train_images.text():
            return False, "Please select a training images folder"
        if not os.path.isdir(self.train_images.text()):
            return False, "Training images folder does not exist"

        # Masks are required only if not using interactive mode
        if not self.interactive_check.isChecked():
            if not self.train_masks.text():
                return False, "Please select a training masks folder (or enable interactive painting)"
            if not os.path.isdir(self.train_masks.text()):
                return False, "Training masks folder does not exist"
        elif self.train_masks.text() and not os.path.isdir(self.train_masks.text()):
            return False, "Training masks folder does not exist"

        if self.model_type.currentIndex() == 1 and not os.path.isfile(self.checkpoint_path.text()):
            return False, "Checkpoint file does not exist"
        return True, ""

    def _resume_last_project(self):
        """Resume the last opened project."""
        from PyQt6.QtWidgets import QMessageBox
        from ..project_config import load_project_config, project_exists

        settings = QSettings("MOSS", "SegmentationSuite")
        project_dir = settings.value("last_project_dir", "")

        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(
                self, "No Recent Project",
                "No recent project found to resume."
            )
            return

        # Check if this looks like a valid project
        if not project_exists(project_dir):
            QMessageBox.warning(
                self, "Invalid Project",
                f"The last project no longer exists or is invalid:\n{project_dir}"
            )
            # Clear the invalid setting
            settings.remove("last_project_dir")
            self.resume_last_btn.hide()
            return

        # Clear ALL fields first to prevent leftover data
        self._clear_all_fields()

        # Try to load project.json
        config = load_project_config(project_dir)

        if config:
            self._apply_loaded_config(project_dir, config)
        else:
            self._load_from_folder_structure(project_dir)

    def _clear_all_fields(self):
        """Clear all UI fields to prevent leftover data from previous projects."""
        self.project_name.clear()
        self.output_dir.clear()
        self.train_images.clear()
        self.train_masks.clear()
        self.val_images.clear()
        self.val_masks.clear()
        self.checkpoint_path.clear()
        self.model_type.setCurrentIndex(0)  # New model
        self.num_epochs.setValue(5000)
        self.batch_size.setValue(2)
        self.learning_rate.setValue(0.0001)
        self.tile_size.setCurrentIndex(0)  # Default tile size
        self.interactive_check.setChecked(True)
        self._loaded_project_dir = None
        self._loaded_project_config = None
        # Disable session buttons until a project is loaded
        self.host_session_btn.setEnabled(False)
        self.host_session_btn.setToolTip("Load a project first to enable multi-user sessions")
        self.join_session_btn.setEnabled(False)
        self.join_session_btn.setToolTip("Load a project first to enable multi-user sessions")
        print("[LoadProject] Cleared all fields")

    def _load_existing_project(self):
        """Load an existing project and auto-fill all paths."""
        from PyQt6.QtWidgets import QMessageBox
        from ..project_config import load_project_config, resolve_path, project_exists

        # Clear ALL fields first to prevent leftover data
        self._clear_all_fields()

        project_dir = QFileDialog.getExistingDirectory(self, "Select Existing Project Directory")
        print(f"[LoadProject] User selected directory: {project_dir}")
        if not project_dir:
            return

        # Check if this looks like a valid project
        if not project_exists(project_dir):
            QMessageBox.warning(
                self, "Invalid Project",
                "This doesn't appear to be a valid project directory.\n"
                "Expected 'project.json' or 'masks/'/'train_images/' folder."
            )
            return

        # Try to load project.json
        config = load_project_config(project_dir)

        if config:
            # Load from project.json - smooth loading!
            self._apply_loaded_config(project_dir, config)
        else:
            # Fallback: infer from folder structure
            self._load_from_folder_structure(project_dir)

    def _apply_loaded_config(self, project_dir: str, config: dict):
        """Apply loaded project configuration to UI and auto-advance."""
        from ..project_config import resolve_path

        # Save as last project for resume functionality
        settings = QSettings("MOSS", "SegmentationSuite")
        settings.setValue("last_project_dir", project_dir)

        # Project info
        self.project_name.setText(config.get("project_name", os.path.basename(project_dir)))
        self.output_dir.setText(os.path.dirname(project_dir))

        # Resolve and set paths
        raw_images_config = config.get("raw_images_dir", "")
        raw_images = resolve_path(project_dir, raw_images_config)
        print(f"[LoadProject] raw_images_dir from config: '{raw_images_config}'")
        print(f"[LoadProject] resolved to: '{raw_images}'")
        if raw_images and os.path.isdir(raw_images):
            self.train_images.setText(raw_images)
            print(f"[LoadProject] Set train_images to: '{raw_images}'")
        else:
            # Ask user to select (only prompt needed)
            raw_path = QFileDialog.getExistingDirectory(self, "Select Raw Images Directory")
            if raw_path:
                self.train_images.setText(raw_path)
                print(f"[LoadProject] User selected raw_path: '{raw_path}'")

        # Training masks (the train_masks_dir with crops)
        train_masks = resolve_path(project_dir, config.get("train_masks_dir", "train_masks"))
        if train_masks and os.path.isdir(train_masks) and os.listdir(train_masks):
            self.train_masks.setText(train_masks)

        # Training parameters
        self.num_epochs.setValue(config.get("num_epochs", 5000))
        self.batch_size.setValue(config.get("batch_size", 2))
        self.learning_rate.setValue(config.get("learning_rate", 0.0001))

        tile_size = str(config.get("tile_size", 256))
        idx = self.tile_size.findText(tile_size)
        if idx >= 0:
            self.tile_size.setCurrentIndex(idx)

        # Interactive mode
        self.interactive_check.setChecked(config.get("interactive_mode", True))

        # Checkpoint - auto-resume if exists (no dialog)
        checkpoint = resolve_path(project_dir, config.get("checkpoint_path", "checkpoint.pth"))
        if checkpoint and os.path.isfile(checkpoint):
            self.model_type.setCurrentIndex(1)  # Resume from checkpoint
            self.checkpoint_path.setText(checkpoint)

        # Store loaded config for passing to interactive page
        self._loaded_project_config = config
        self._loaded_project_dir = project_dir

        print(f"[LoadProject] Applied config for: {project_dir}")
        print(f"[LoadProject] UI shows: output_dir='{self.output_dir.text()}', project_name='{self.project_name.text()}'")

        # Enable multi-user session buttons now that project is loaded
        self._enable_session_buttons()

        # Auto-advance to workflow
        self.project_loaded.emit()

    def _load_from_folder_structure(self, project_dir: str):
        """Fallback: Load project by inferring from folder structure and auto-advance."""
        # Save as last project for resume functionality
        settings = QSettings("MOSS", "SegmentationSuite")
        settings.setValue("last_project_dir", project_dir)

        project_name = os.path.basename(project_dir)
        parent_dir = os.path.dirname(project_dir)

        self.project_name.setText(project_name)
        self.output_dir.setText(parent_dir)

        # Look for raw images directory
        raw_images_dir = None
        for candidate in ['raw_images', 'images', 'raw']:
            candidate_path = os.path.join(project_dir, candidate)
            if os.path.isdir(candidate_path):
                raw_images_dir = candidate_path
                break

        if raw_images_dir:
            self.train_images.setText(raw_images_dir)
        else:
            # Only prompt needed
            raw_path = QFileDialog.getExistingDirectory(self, "Select Raw Images Directory")
            if raw_path:
                self.train_images.setText(raw_path)

        # Set masks folder if exists
        train_masks_dir = os.path.join(project_dir, 'train_masks')
        if os.path.isdir(train_masks_dir) and os.listdir(train_masks_dir):
            self.train_masks.setText(train_masks_dir)

        # Checkpoint - auto-resume if exists (no dialog)
        checkpoint_path = os.path.join(project_dir, 'checkpoint.pth')
        if os.path.isfile(checkpoint_path):
            self.model_type.setCurrentIndex(1)  # Resume from checkpoint
            self.checkpoint_path.setText(checkpoint_path)

        self.interactive_check.setChecked(True)
        self._loaded_project_config = None
        self._loaded_project_dir = project_dir

        # Enable multi-user session buttons now that project is loaded
        self._enable_session_buttons()

        # Auto-advance to workflow
        self.project_loaded.emit()

    def get_loaded_project_config(self) -> tuple:
        """Return loaded project config if available."""
        return (
            getattr(self, '_loaded_project_config', None),
            getattr(self, '_loaded_project_dir', None)
        )

    # =========================================================================
    # Multi-User Session Management
    # =========================================================================

    def set_current_architecture(self, architecture: str):
        """Set the current architecture (called by interactive_training_page)."""
        self._current_architecture = architecture
        print(f"[Setup] Current architecture set to: {architecture}")

    def _enable_session_buttons(self):
        """Enable multi-user session buttons after project is loaded."""
        self.host_session_btn.setEnabled(True)
        self.host_session_btn.setToolTip("Create a session and get a code for others to join (works over internet)")
        self.join_session_btn.setEnabled(True)
        self.join_session_btn.setToolTip("Enter a 6-character code to join someone's session")
        print("[Setup] Multi-user session buttons enabled (project loaded)")

    def _host_session(self):
        """Start hosting a multi-user session via relay server."""
        try:
            from ..network import SyncClient, DEFAULT_RELAY_URL
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        # Check if relay is configured
        if not DEFAULT_RELAY_URL:
            QMessageBox.warning(
                self, "Relay Not Configured",
                "No relay server is configured.\n\n"
                "To use internet-based sessions, deploy your own relay server:\n"
                "1. See relay_server/SETUP_GUIDE.md for instructions\n"
                "2. Set DEFAULT_RELAY_URL in network/client.py\n\n"
                "For LAN-only sessions, use 'Advanced (LAN)' instead."
            )
            return

        # Get display name
        name, ok = self._get_display_name()
        if not ok:
            return

        self._is_relay_mode = True

        # Show status
        self.session_status_label.setText("Creating room...")
        self.session_status_label.setStyleSheet("color: orange;")

        # Create sync client and connect to relay
        try:
            self._sync_client = SyncClient(parent=self)
            self._sync_client.room_created.connect(self._on_room_created)
            self._sync_client.connected.connect(self._on_relay_connected)
            self._sync_client.disconnected.connect(self._on_client_disconnected)
            self._sync_client.error.connect(self._on_session_error)
            self._sync_client.user_list_updated.connect(self._on_user_list_updated)
            self._sync_client.sync_status.connect(self._on_sync_status)

            # Create room on relay server
            success = self._sync_client.create_relay_room(name)
            if not success:
                QMessageBox.critical(self, "Error", "Failed to connect to relay server")
                self._sync_client = None
                self._reset_session_ui()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create room:\n{e}")
            self._sync_client = None
            self._reset_session_ui()

    def _on_room_created(self, room_code: str):
        """Handle room created on relay server."""
        self._room_code = room_code
        self._is_relay_host = True  # We created the room, so we're the host
        print(f"[Setup] Room created: {room_code} (we are host)")

        # Show room code prominently
        self.session_status_label.setText("Session active! Share this code:")
        self.session_status_label.setStyleSheet("color: green;")
        self.room_code_label.setText(room_code)
        self.room_code_label.setVisible(True)

        # Show dialog with the code
        QMessageBox.information(
            self, "Session Started",
            f"Your session is now active!\n\n"
            f"Share this code with others:\n\n"
            f"   {room_code}\n\n"
            f"They can join from anywhere with an internet connection."
        )

        self.host_session_btn.setVisible(False)
        self.join_session_btn.setVisible(False)
        self.advanced_btn.setVisible(False)
        self.disconnect_btn.setVisible(True)
        self.user_list_widget.setVisible(True)

    def _on_relay_connected(self):
        """Handle connected to relay server."""
        print("[Setup] Connected to relay")

    def _on_room_joined(self, room_code: str):
        """Handle successfully joined a relay room."""
        self._room_code = room_code
        print(f"[Setup] Joined room: {room_code}")

        self.session_status_label.setText(f"Connected to room {room_code}")
        self.session_status_label.setStyleSheet("color: green; font-weight: bold;")

        self.host_session_btn.setVisible(False)
        self.join_session_btn.setVisible(False)
        self.advanced_btn.setVisible(False)
        self.disconnect_btn.setVisible(True)
        self.user_list_widget.setVisible(True)

    def _on_server_started(self, connection_string: str):
        """Handle LAN server started."""
        # Create a client for the host to participate in federated learning
        self._create_host_client()

        # LAN mode - show IP address
        self.session_status_label.setText(f"Hosting on {connection_string}")
        self.session_status_label.setStyleSheet("color: green; font-weight: bold;")
        QMessageBox.information(
            self, "Session Started",
            f"Your session is now active!\n\n"
            f"Share this with others on your network:\n{connection_string}\n\n"
            f"(Others click 'Advanced (LAN)' and enter this address)"
        )

        self.host_session_btn.setVisible(False)
        self.join_session_btn.setVisible(False)
        self.advanced_btn.setVisible(False)
        self.disconnect_btn.setVisible(True)
        self.user_list_widget.setVisible(True)

    def _create_host_client(self):
        """Create a client for the host to participate in federated learning."""
        from ..network.client import HostClient

        try:
            name = getattr(self, '_host_display_name', 'Host')
            self._sync_client = HostClient(parent=self)
            self._sync_client.connected.connect(self._on_host_client_connected)
            self._sync_client.error.connect(self._on_session_error)
            self._sync_client.user_list_updated.connect(self._on_user_list_updated)

            # Connect to local server
            self._sync_client.connect_to_local_server(port=8765, user_name=name)
            print(f"[Setup] Host client connecting to local server as '{name}'")

        except Exception as e:
            print(f"[Setup] Warning: Could not create host client: {e}")

    def _on_host_client_connected(self):
        """Handle host client connected to local server."""
        print("[Setup] Host client connected to local server")

    def _join_session(self):
        """Join an existing multi-user session."""
        try:
            from ..network import SyncClient, DEFAULT_RELAY_URL
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        # Get code or address
        code_or_address, ok = self._get_room_code_or_address()
        if not ok or not code_or_address:
            return

        # Get display name
        name, ok = self._get_display_name()
        if not ok:
            return

        # Determine if this is a room code or IP address
        # Room codes are 6 uppercase alphanumeric chars
        # IP addresses contain dots or colons
        is_ip_address = '.' in code_or_address or ':' in code_or_address

        if is_ip_address:
            # Direct LAN connection
            self._join_direct(code_or_address, name)
        else:
            # Room code - requires relay server
            if not DEFAULT_RELAY_URL:
                QMessageBox.warning(
                    self, "Relay Not Configured",
                    "Room codes require a relay server.\n\n"
                    "Either:\n"
                    "1. Deploy your own relay (see relay_server/SETUP_GUIDE.md)\n"
                    "2. Use an IP address for LAN connections (e.g., 192.168.1.5:8765)"
                )
                return
            self._join_by_code(code_or_address.upper(), name)

    def _join_direct(self, host_address: str, name: str):
        """Join via direct LAN connection."""
        from ..network import SyncClient

        # Parse host:port
        try:
            if ':' in host_address:
                host_ip, port_str = host_address.split(':')
                port = int(port_str)
            else:
                host_ip = host_address
                port = 8765
        except ValueError:
            QMessageBox.warning(self, "Invalid Address",
                "Please enter a valid address (e.g., 192.168.1.5:8765)")
            return

        print(f"[Setup] Joining direct: {host_ip}:{port}")

        # Create client
        try:
            self._sync_client = SyncClient(parent=self)
            self._sync_client.connected.connect(self._on_client_connected)
            self._sync_client.disconnected.connect(self._on_client_disconnected)
            self._sync_client.error.connect(self._on_session_error)
            self._sync_client.user_list_updated.connect(self._on_user_list_updated)
            self._sync_client.architecture_received.connect(self._on_architecture_received)

            self._sync_client.connect_direct(host_ip, port, name)
            self._is_relay_mode = False

            self.session_status_label.setText(f"Connecting to {host_ip}:{port}...")
            self.session_status_label.setStyleSheet("color: orange;")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect:\n{e}")

    def _join_by_code(self, room_code: str, name: str):
        """Join via room code (relay server)."""
        from ..network import SyncClient

        print(f"[Setup] Joining by code: {room_code}")

        try:
            self._sync_client = SyncClient(parent=self)
            self._sync_client.room_joined.connect(self._on_room_joined)
            self._sync_client.connected.connect(self._on_client_connected)
            self._sync_client.disconnected.connect(self._on_client_disconnected)
            self._sync_client.error.connect(self._on_session_error)
            self._sync_client.user_list_updated.connect(self._on_user_list_updated)
            self._sync_client.sync_status.connect(self._on_sync_status)

            # Connect to relay server with room code
            success = self._sync_client.connect_relay(room_code, name)
            if not success:
                QMessageBox.critical(self, "Error", "Failed to connect to relay server")
                self._sync_client = None
                return

            self._room_code = room_code
            self._is_relay_mode = True

            self.session_status_label.setText(f"Joining room {room_code}...")
            self.session_status_label.setStyleSheet("color: orange;")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect:\n{e}")

    def _on_architecture_received(self, architecture: str):
        """Handle architecture received from host - lock the architecture dropdown."""
        print(f"[Setup] Received architecture from host: {architecture}")
        self.architecture_locked.emit(architecture)

    def _on_sync_status(self, status: str):
        """Handle sync status updates."""
        print(f"[Setup] Sync status: {status}")

    def _on_client_connected(self):
        """Handle successful connection to host."""
        if self._room_code:
            self.session_status_label.setText(f"Connected to room {self._room_code}")
        else:
            self.session_status_label.setText(f"Connected to {self._sync_client.connection_info}")
        self.session_status_label.setStyleSheet("color: green; font-weight: bold;")

        self.host_session_btn.setVisible(False)
        self.join_session_btn.setVisible(False)
        self.advanced_btn.setVisible(False)
        self.disconnect_btn.setVisible(True)
        self.user_list_widget.setVisible(True)

    def _on_client_disconnected(self):
        """Handle disconnection from host."""
        self._reset_session_ui()
        self.session_status_label.setText("Disconnected from session")
        self.session_status_label.setStyleSheet("color: orange;")

    def _disconnect_session(self):
        """Disconnect from the current session."""
        if self._aggregation_server:
            self._aggregation_server.stop()
            self._aggregation_server = None

        if self._sync_client:
            self._sync_client.disconnect()
            self._sync_client = None

        self._reset_session_ui()

    def _reset_session_ui(self):
        """Reset the session UI to initial state."""
        self.session_status_label.setText("Not connected")
        self.session_status_label.setStyleSheet("color: gray;")

        self.room_code_label.setVisible(False)
        self.room_code_label.setText("")
        self.host_session_btn.setVisible(True)
        self.join_session_btn.setVisible(True)
        self.advanced_btn.setVisible(True)
        self.disconnect_btn.setVisible(False)
        self.user_list_widget.setVisible(False)
        self.user_list_widget.clear()

        self._room_code = None
        self._is_relay_mode = False
        self._is_relay_host = False

        # Unlock the architecture dropdown
        self.architecture_unlocked.emit()

    def _on_user_connected(self, user_id: str, display_name: str):
        """Handle when a user connects to our session."""
        self.user_list_widget.addItem(f"{display_name}")
        self.session_status_label.setText(
            f"Hosting ({self._aggregation_server.client_count} users)"
        )

    def _on_user_disconnected(self, user_id: str):
        """Handle when a user disconnects from our session."""
        # Update user list (simplified - just refresh count)
        if self._aggregation_server:
            self.session_status_label.setText(
                f"Hosting ({self._aggregation_server.client_count} users)"
            )

    def _on_user_list_updated(self, users: list):
        """Handle updated user list from server."""
        self.user_list_widget.clear()
        for user in users:
            name = user.get('display_name', 'Unknown')
            self.user_list_widget.addItem(name)

    def _on_session_error(self, error: str):
        """Handle session error."""
        QMessageBox.warning(self, "Session Error", error)
        self.session_status_label.setText(f"Error: {error}")
        self.session_status_label.setStyleSheet("color: red;")

    def _get_display_name(self) -> tuple:
        """Get display name from user."""
        from PyQt6.QtWidgets import QInputDialog

        settings = QSettings("MOSS", "SegmentationSuite")
        last_name = settings.value("multi_user_display_name", "")

        name, ok = QInputDialog.getText(
            self, "Display Name",
            "Enter your display name:",
            text=last_name or os.environ.get('USER', 'User')
        )

        if ok and name:
            settings.setValue("multi_user_display_name", name)

        return name, ok

    def _get_host_address(self) -> tuple:
        """Get host address from user."""
        from PyQt6.QtWidgets import QInputDialog

        settings = QSettings("MOSS", "SegmentationSuite")
        last_host = settings.value("last_session_host", "")

        address, ok = QInputDialog.getText(
            self, "Join Session",
            "Enter host address (e.g., 192.168.1.5:8765):",
            text=last_host
        )

        if ok and address:
            settings.setValue("last_session_host", address)

        return address, ok

    def _get_room_code_or_address(self) -> tuple:
        """Get room code or address from user."""
        from PyQt6.QtWidgets import QInputDialog

        settings = QSettings("MOSS", "SegmentationSuite")
        last_code = settings.value("last_session_room", "")

        code, ok = QInputDialog.getText(
            self, "Join Session",
            "Enter room code (e.g., ABC123)\nor IP address (e.g., 192.168.1.5:8765):",
            text=last_code
        )

        if ok and code:
            settings.setValue("last_session_room", code)

        return code, ok

    def _show_advanced_options(self):
        """Show advanced LAN connection dialog."""
        from PyQt6.QtWidgets import QInputDialog

        choices = ["Host LAN Session (others connect to your IP)",
                   "Join LAN Session (enter host IP)"]
        choice, ok = QInputDialog.getItem(
            self, "Advanced LAN Options",
            "Select an option:",
            choices, 0, False
        )

        if ok:
            if choice == choices[0]:
                # Host LAN session (without room code)
                self._room_code = None  # No room code for pure LAN
                self._host_lan_session()
            else:
                # Join LAN session
                host_address, ok = self._get_host_address()
                if ok and host_address:
                    name, ok = self._get_display_name()
                    if ok:
                        self._join_direct(host_address, name)

    def _select_architecture_dialog(self) -> str:
        """Show dialog to select model architecture. Returns architecture ID or empty string."""
        from ..models.architectures import get_available_architectures, get_checkpoint_filename

        # Get available architectures
        architectures = get_available_architectures()
        if not architectures:
            QMessageBox.warning(
                self, "No Architectures",
                "No model architectures found."
            )
            return ""

        # Check which architectures have checkpoints in the project directory
        project_dir = getattr(self, '_loaded_project_dir', None)
        archs_with_checkpoints = set()
        default_arch = None

        if project_dir and os.path.isdir(project_dir):
            for arch_id in architectures.keys():
                checkpoint_name = get_checkpoint_filename(arch_id)
                checkpoint_path = os.path.join(project_dir, checkpoint_name)
                if os.path.isfile(checkpoint_path):
                    archs_with_checkpoints.add(arch_id)
                    if default_arch is None:
                        default_arch = arch_id  # First found checkpoint

        # Create list of display names, marking those with checkpoints
        arch_items = list(architectures.items())  # [(id, name), ...]
        display_names = []
        default_index = 0

        for i, (arch_id, name) in enumerate(arch_items):
            if arch_id in archs_with_checkpoints:
                display_names.append(f"{name} (has checkpoint)")
                if arch_id == default_arch:
                    default_index = i
            else:
                display_names.append(name)

        from PyQt6.QtWidgets import QInputDialog
        selected_name, ok = QInputDialog.getItem(
            self, "Select Model Architecture",
            "Choose the model architecture for this session:\n\n"
            "All participants will train using this architecture.",
            display_names, default_index, False
        )

        if not ok:
            return ""

        # Find the architecture ID for the selected name (strip checkpoint indicator)
        selected_base = selected_name.replace(" (has checkpoint)", "")
        for arch_id, name in arch_items:
            if name == selected_base:
                return arch_id

        return ""

    def _host_lan_session(self):
        """Host a LAN-only session (no room code)."""
        try:
            from ..network import AggregationServer, get_local_ip
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        # If no architecture set, prompt user to select one
        if not self._current_architecture:
            architecture = self._select_architecture_dialog()
            if not architecture:
                return
            self._current_architecture = architecture

        name, ok = self._get_display_name()
        if not ok:
            return

        # Save display name for creating host client after server starts
        self._host_display_name = name

        self._is_relay_mode = False
        self.session_status_label.setText("Starting LAN session...")
        self.session_status_label.setStyleSheet("color: orange;")

        try:
            self._aggregation_server = AggregationServer(parent=self)
            self._aggregation_server.user_connected.connect(self._on_user_connected)
            self._aggregation_server.user_disconnected.connect(self._on_user_disconnected)
            self._aggregation_server.error.connect(self._on_session_error)
            self._aggregation_server.server_started.connect(self._on_server_started)

            connection_string = self._aggregation_server.start(
                port=8765,
                architecture=self._current_architecture
            )
            print(f"[Setup] Started LAN server with architecture: {self._current_architecture}")

            # Lock the host's architecture while hosting
            self.architecture_locked.emit(self._current_architecture)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start server:\n{e}")

    def get_multi_user_state(self) -> tuple:
        """Return the current multi-user state (server, client, is_relay_host)."""
        return (self._aggregation_server, self._sync_client, self._is_relay_host)
