#!/usr/bin/env python3
"""
Reslice page for the training wizard - create XZ, YZ, and diagonal views.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QCheckBox, QLineEdit,
    QFileDialog, QSpinBox, QFormLayout
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class ReslicePage(QWidget):
    """Reslice page for creating multi-view reslices."""

    # Signals
    reslice_complete = pyqtSignal(dict)  # output_dirs
    busy_changed = pyqtSignal(bool)  # True when reslicing, False when done

    def __init__(self):
        super().__init__()
        self.worker = None
        self.config = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("Reslicing")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Input selection
        input_group = QGroupBox("Input")
        input_layout = QFormLayout(input_group)

        self.input_dir = QLineEdit()
        self.input_dir.setPlaceholderText("Folder containing raw z-stack images")
        input_btn = QPushButton("Browse...")
        input_btn.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_dir)
        input_row.addWidget(input_btn)
        input_layout.addRow("Raw Images:", input_row)

        layout.addWidget(input_group)

        # Reslice options
        options_group = QGroupBox("Reslice Options")
        options_layout = QVBoxLayout(options_group)

        self.xz_check = QCheckBox("XZ reslice (side view along Y)")
        self.xz_check.setChecked(True)
        options_layout.addWidget(self.xz_check)

        self.yz_check = QCheckBox("YZ reslice (side view along X)")
        self.yz_check.setChecked(True)
        options_layout.addWidget(self.yz_check)

        options_layout.addSpacing(10)

        # Diagonal options
        diag_label = QLabel("Diagonal views (optional):")
        options_layout.addWidget(diag_label)

        self.diag1_check = QCheckBox("Diagonal ZX 45 degrees")
        options_layout.addWidget(self.diag1_check)

        self.diag2_check = QCheckBox("Diagonal ZY 45 degrees")
        options_layout.addWidget(self.diag2_check)

        self.diag3_check = QCheckBox("Diagonal ZX 30 degrees")
        options_layout.addWidget(self.diag3_check)

        layout.addWidget(options_group)

        # Processing options
        proc_group = QGroupBox("Processing Options")
        proc_layout = QFormLayout(proc_group)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(50, 500)
        self.batch_size.setValue(200)
        proc_layout.addRow("Batch Size:", self.batch_size)

        self.workers = QSpinBox()
        self.workers.setRange(1, 32)
        self.workers.setValue(8)
        proc_layout.addRow("Workers:", self.workers)

        layout.addWidget(proc_group)

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

        self.start_btn = QPushButton("Start Reslicing")
        self.start_btn.clicked.connect(self._on_start)
        button_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        layout.addStretch()

    def _browse_input(self):
        """Browse for input folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Raw Images Folder")
        if folder:
            self.input_dir.setText(folder)

    def set_config(self, config: dict):
        """Set configuration from previous steps."""
        self.config = config
        # Pre-fill input directory if available
        if 'raw_images_dir' in config:
            self.input_dir.setText(config['raw_images_dir'])

    def _get_reslice_config(self) -> dict:
        """Build reslice configuration."""
        output_dir = self.config.get('project_dir', os.path.expanduser('~/reslice_output'))

        diagonals = []
        if self.diag1_check.isChecked():
            diagonals.append({'angle': 45, 'axes': (0, 2), 'name': 'diag_zx45'})
        if self.diag2_check.isChecked():
            diagonals.append({'angle': 45, 'axes': (0, 1), 'name': 'diag_zy45'})
        if self.diag3_check.isChecked():
            diagonals.append({'angle': 30, 'axes': (0, 2), 'name': 'diag_zx30'})

        return {
            'input_dir': self.input_dir.text(),
            'output_dir': os.path.join(output_dir, 'reslices'),
            'create_xz': self.xz_check.isChecked(),
            'create_yz': self.yz_check.isChecked(),
            'diagonals': diagonals,
            'batch_size': self.batch_size.value(),
            'max_workers': self.workers.value(),
        }

    def _on_start(self):
        """Start reslicing."""
        print(f"Start clicked, worker={self.worker}")

        if self.worker is not None:
            print("Worker still exists, cannot start")
            return

        if not self.input_dir.text() or not os.path.isdir(self.input_dir.text()):
            self._log("Please select a valid input directory")
            return

        from ..workers.reslice_worker import ResliceWorker

        print("Creating new ResliceWorker...")
        config = self._get_reslice_config()
        self.worker = ResliceWorker(config)
        self.worker.started.connect(self._on_started)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)

        # Reset UI
        self.progress_bar.setValue(0)
        self.log_text.clear()

        print("Starting worker...")
        self.worker.start()

    def _on_stop(self):
        """Stop reslicing."""
        print("Stop button clicked (Reslice)")
        if self.worker:
            print("Stopping...")
            self.worker.stop()
            self._log("Stopping... please wait")
            self.stop_btn.setEnabled(False)
            # Wait for worker to finish (with timeout)
            if not self.worker.wait(5000):  # 5 second timeout
                self._log("Worker taking too long, forcing termination")
                print("Forcing termination...")
                self.worker.terminate()
                self.worker.wait(1000)
            print("Stopped")
            self._log("Stopped")
            self.worker = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopped")
            self.progress_bar.setValue(0)
            self.busy_changed.emit(False)

    def _on_started(self):
        """Handle started signal."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Reslicing in progress...")
        self.busy_changed.emit(True)

    def _on_progress(self, plane: str, current: int, total: int):
        """Handle progress signal."""
        self.status_label.setText(f"Processing {plane}: {current}/{total}")
        self.progress_bar.setValue(int(100 * current / total))

    def _log(self, message: str):
        """Add log message."""
        self.log_text.append(message)

    def _on_finished(self, success: bool, result: dict):
        """Handle finished signal."""
        # Ignore if worker was already cleared (e.g., by stop button)
        if self.worker is None:
            return

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Wait for thread to fully finish before clearing reference
        if self.worker.isRunning():
            self.worker.wait(2000)

        self.worker = None
        self.busy_changed.emit(False)

        if success:
            self.status_label.setText("Reslicing complete!")
            self._log("Reslicing complete!")
            self.config['reslice_dirs'] = result
            self.reslice_complete.emit(result)
        else:
            error_msg = result.get('error', 'Unknown error')
            if "Stopped" in error_msg:
                self.status_label.setText("Reslicing stopped")
            else:
                self.status_label.setText("Reslicing failed")
                self._log(f"Error: {error_msg}")

    def cleanup(self):
        """Clean up worker when page is closed."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)
            self.worker = None

    def closeEvent(self, event):
        """Handle page close."""
        self.cleanup()
        super().closeEvent(event)
