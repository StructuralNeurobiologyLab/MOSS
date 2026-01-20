#!/usr/bin/env python3
"""
Prediction page for the training wizard - run prediction on multiple views.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QCheckBox, QLineEdit,
    QFileDialog, QSpinBox, QFormLayout, QComboBox
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class PredictionPage(QWidget):
    """Prediction page for running UNet inference on all views."""

    # Signals
    prediction_complete = pyqtSignal(dict)  # output_dirs
    busy_changed = pyqtSignal(bool)  # True when predicting, False when done

    def __init__(self):
        super().__init__()
        self.worker = None
        self.config = {}
        self.current_architecture = 'unet'
        self._arch_name_to_id = {}
        self._arch_id_to_name = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("Multi-View Prediction")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        # Architecture selector
        self.arch_combo = QComboBox()
        self.arch_combo.currentTextChanged.connect(self._on_architecture_changed)
        model_layout.addRow("Architecture:", self.arch_combo)

        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Path to trained model checkpoint")
        checkpoint_btn = QPushButton("Browse...")
        checkpoint_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint_path)
        checkpoint_row.addWidget(checkpoint_btn)
        model_layout.addRow("Checkpoint:", checkpoint_row)

        layout.addWidget(model_group)

        # Populate architecture dropdown
        self._populate_architecture_combo()

        # View selection
        views_group = QGroupBox("Views to Predict")
        views_layout = QVBoxLayout(views_group)

        self.xy_check = QCheckBox("XY (original orientation)")
        self.xy_check.setChecked(True)
        views_layout.addWidget(self.xy_check)

        self.xz_check = QCheckBox("XZ reslice")
        self.xz_check.setChecked(True)
        views_layout.addWidget(self.xz_check)

        self.yz_check = QCheckBox("YZ reslice")
        self.yz_check.setChecked(True)
        views_layout.addWidget(self.yz_check)

        views_layout.addSpacing(10)

        self.diag1_check = QCheckBox("Diagonal ZX 45")
        views_layout.addWidget(self.diag1_check)

        self.diag2_check = QCheckBox("Diagonal ZY 45")
        views_layout.addWidget(self.diag2_check)

        self.diag3_check = QCheckBox("Diagonal ZX 30")
        views_layout.addWidget(self.diag3_check)

        layout.addWidget(views_group)

        # Prediction options
        options_group = QGroupBox("Options")
        options_layout = QFormLayout(options_group)

        self.patch_size = QSpinBox()
        self.patch_size.setRange(128, 1024)
        self.patch_size.setValue(512)
        self.patch_size.setSingleStep(128)
        options_layout.addRow("Patch Size:", self.patch_size)

        self.overlap = QSpinBox()
        self.overlap.setRange(0, 256)
        self.overlap.setValue(64)
        self.overlap.setSingleStep(16)
        options_layout.addRow("Overlap:", self.overlap)

        layout.addWidget(options_group)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        progress_layout.addWidget(self.log_text)

        layout.addWidget(progress_group)

        # Buttons
        button_row = QHBoxLayout()

        self.start_btn = QPushButton("Start Prediction")
        self.start_btn.clicked.connect(self._on_start)
        button_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        layout.addStretch()

    def _browse_checkpoint(self):
        """Browse for checkpoint file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint",
            "", "PyTorch Checkpoints (*.pth);;All Files (*)"
        )
        if filepath:
            self.checkpoint_path.setText(filepath)

    def set_config(self, config: dict):
        """Set configuration from previous steps."""
        self.config = config

        # Pre-fill checkpoint path from config, or try to find one for current architecture
        if 'checkpoint_path' in config:
            self.checkpoint_path.setText(config['checkpoint_path'])
        else:
            # Try to find checkpoint for current architecture
            checkpoint = self._find_most_recent_checkpoint(self.current_architecture)
            if checkpoint:
                self.checkpoint_path.setText(checkpoint)

        # Get reslice_dirs from config, or auto-detect from project folder
        reslice_dirs = config.get('reslice_dirs', {})
        if not reslice_dirs:
            reslice_dirs = self._detect_reslice_dirs()
            # Store back in config so _get_views_config can use it
            self.config['reslice_dirs'] = reslice_dirs

        print(f"PredictionPage.set_config: reslice_dirs = {reslice_dirs}")

        # Check if reslice dirs exist - used to set default checked state
        xz_exists = bool(reslice_dirs.get('xz')) and os.path.isdir(reslice_dirs.get('xz', ''))
        yz_exists = bool(reslice_dirs.get('yz')) and os.path.isdir(reslice_dirs.get('yz', ''))
        diag1_exists = bool(reslice_dirs.get('diag_zx45')) and os.path.isdir(reslice_dirs.get('diag_zx45', ''))
        diag2_exists = bool(reslice_dirs.get('diag_zy45')) and os.path.isdir(reslice_dirs.get('diag_zy45', ''))
        diag3_exists = bool(reslice_dirs.get('diag_zx30')) and os.path.isdir(reslice_dirs.get('diag_zx30', ''))

        print(f"  xz_exists={xz_exists}, yz_exists={yz_exists}")

        # All checkboxes are always enabled (optional) - just set checked state based on existence
        self.xz_check.setEnabled(True)
        self.yz_check.setEnabled(True)
        self.diag1_check.setEnabled(True)
        self.diag2_check.setEnabled(True)
        self.diag3_check.setEnabled(True)

        # Only check if the directories exist
        self.xz_check.setChecked(xz_exists)
        self.yz_check.setChecked(yz_exists)
        self.diag1_check.setChecked(diag1_exists)
        self.diag2_check.setChecked(diag2_exists)
        self.diag3_check.setChecked(diag3_exists)

    def _detect_reslice_dirs(self) -> dict:
        """Auto-detect reslice directories from project folder structure."""
        reslice_dirs = {}
        project_dir = self.config.get('project_dir', '')
        if not project_dir or not os.path.isdir(project_dir):
            return reslice_dirs

        reslice_base = os.path.join(project_dir, 'reslices')
        if not os.path.isdir(reslice_base):
            return reslice_dirs

        # Check for standard reslice directory names
        reslice_mapping = {
            'xz': 'xz_reslice',
            'yz': 'yz_reslice',
            'diag_zx45': 'diag_zx45',
            'diag_zy45': 'diag_zy45',
            'diag_zx30': 'diag_zx30',
        }

        for key, dirname in reslice_mapping.items():
            dir_path = os.path.join(reslice_base, dirname)
            if os.path.isdir(dir_path) and os.listdir(dir_path):
                reslice_dirs[key] = dir_path
                print(f"  Auto-detected reslice: {key} -> {dir_path}")

        return reslice_dirs

    def _get_views_config(self) -> list:
        """Build list of view configurations."""
        views = []
        project_dir = self.config.get('project_dir', os.path.expanduser('~/predictions'))
        pred_base = os.path.join(project_dir, 'predictions')
        reslice_dirs = self.config.get('reslice_dirs', {})

        if self.xy_check.isChecked():
            xy_input = self.config.get('raw_images_dir', '')
            if not xy_input or not os.path.isdir(xy_input):
                self._log("Error: raw_images_dir not set or invalid. Please configure it in project settings.")
                return []
            views.append({
                'name': 'xy',
                'input_dir': xy_input,
                'output_dir': os.path.join(pred_base, 'xy_predictions'),
            })

        if self.xz_check.isChecked():
            xz_dir = reslice_dirs.get('xz', '')
            if xz_dir and os.path.isdir(xz_dir):
                views.append({
                    'name': 'xz',
                    'input_dir': xz_dir,
                    'output_dir': os.path.join(pred_base, 'xz_predictions'),
                })
            else:
                self._log("Warning: XZ checked but reslice directory not found, skipping")

        if self.yz_check.isChecked():
            yz_dir = reslice_dirs.get('yz', '')
            if yz_dir and os.path.isdir(yz_dir):
                views.append({
                    'name': 'yz',
                    'input_dir': yz_dir,
                    'output_dir': os.path.join(pred_base, 'yz_predictions'),
                })
            else:
                self._log("Warning: YZ checked but reslice directory not found, skipping")

        if self.diag1_check.isChecked():
            diag1_dir = reslice_dirs.get('diag_zx45', '')
            if diag1_dir and os.path.isdir(diag1_dir):
                views.append({
                    'name': 'diag_zx45',
                    'input_dir': diag1_dir,
                    'output_dir': os.path.join(pred_base, 'diag_zx45_predictions'),
                })
            else:
                self._log("Warning: Diagonal ZX 45 checked but directory not found, skipping")

        if self.diag2_check.isChecked():
            diag2_dir = reslice_dirs.get('diag_zy45', '')
            if diag2_dir and os.path.isdir(diag2_dir):
                views.append({
                    'name': 'diag_zy45',
                    'input_dir': diag2_dir,
                    'output_dir': os.path.join(pred_base, 'diag_zy45_predictions'),
                })
            else:
                self._log("Warning: Diagonal ZY 45 checked but directory not found, skipping")

        if self.diag3_check.isChecked():
            diag3_dir = reslice_dirs.get('diag_zx30', '')
            if diag3_dir and os.path.isdir(diag3_dir):
                views.append({
                    'name': 'diag_zx30',
                    'input_dir': diag3_dir,
                    'output_dir': os.path.join(pred_base, 'diag_zx30_predictions'),
                })
            else:
                self._log("Warning: Diagonal ZX 30 checked but directory not found, skipping")

        return views

    def _on_start(self):
        """Start prediction."""
        if self.worker is not None:
            return

        if not self.checkpoint_path.text() or not os.path.isfile(self.checkpoint_path.text()):
            self._log("Please select a valid checkpoint file")
            return

        from ..workers.predict_worker import PredictWorker

        views = self._get_views_config()
        if not views:
            self._log("Please select at least one view to predict")
            return

        config = {
            'checkpoint_path': self.checkpoint_path.text(),
            'architecture': self.current_architecture,
            'views': views,
            'patch_size': self.patch_size.value(),
            'overlap': self.overlap.value(),
        }

        self.worker = PredictWorker(config)
        self.worker.started.connect(self._on_started)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_stop(self):
        """Stop prediction."""
        print("Stop button clicked (Prediction)")
        if self.worker:
            self.worker.stop()
            self._log("Stopping...")

    def _on_started(self):
        """Handle started signal."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Prediction in progress...")
        self.busy_changed.emit(True)

    def _on_progress(self, view: str, current: int, total: int):
        """Handle progress signal."""
        self.status_label.setText(f"Predicting {view}: {current}/{total}")
        self.progress_bar.setValue(int(100 * current / total))

    def _log(self, message: str):
        """Add log message."""
        self.log_text.append(message)

    def _on_finished(self, success: bool, result: dict):
        """Handle finished signal."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Wait for thread to fully finish before clearing reference
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)

        self.worker = None
        self.busy_changed.emit(False)

        if success:
            self.status_label.setText("Prediction complete!")
            self._log("Prediction complete!")
            self.config['prediction_dirs'] = result
            self.prediction_complete.emit(result)
        else:
            error_msg = result.get('error', 'Unknown error')
            if "Stopped" in error_msg:
                self.status_label.setText("Prediction stopped")
            else:
                self.status_label.setText("Prediction failed")
                self._log(f"Error: {error_msg}")

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

    def _on_architecture_changed(self, display_name: str):
        """Handle architecture selection change - auto-update checkpoint path."""
        if display_name not in self._arch_name_to_id:
            return

        arch_id = self._arch_name_to_id[display_name]
        self.current_architecture = arch_id

        # Try to find the most recent checkpoint for this architecture
        checkpoint = self._find_most_recent_checkpoint(arch_id)
        if checkpoint:
            self.checkpoint_path.setText(checkpoint)
        else:
            # Clear checkpoint path - don't use a checkpoint from a different architecture
            self.checkpoint_path.clear()
            self._log(f"No checkpoint found for {display_name}. Please select one manually.")

    def _find_most_recent_checkpoint(self, architecture: str) -> str:
        """Find the most recent checkpoint for the given architecture."""
        from ..models.architectures import get_checkpoint_filename

        project_dir = self.config.get('project_dir', '')
        if not project_dir or not os.path.isdir(project_dir):
            return ''

        checkpoints_dir = os.path.join(project_dir, 'checkpoints')
        if not os.path.isdir(checkpoints_dir):
            return ''

        # Get the expected checkpoint filename for this architecture
        checkpoint_filename = get_checkpoint_filename(architecture)
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)

        if os.path.isfile(checkpoint_path):
            return checkpoint_path

        return ''
