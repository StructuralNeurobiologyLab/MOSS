#!/usr/bin/env python3
"""
Voting page for the training wizard - combine predictions into consensus heatmap.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QCheckBox, QLineEdit,
    QFileDialog, QSpinBox, QFormLayout
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class VotingPage(QWidget):
    """Voting page for combining multi-view predictions."""

    # Signals
    voting_complete = pyqtSignal(str)  # output_dir

    def __init__(self):
        super().__init__()
        self.worker = None
        self.rotate_worker = None
        self.config = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("Consensus Voting")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Info
        info = QLabel(
            "This step combines predictions from all views into a consensus heatmap.\n"
            "Diagonal predictions will be rotated back to XY orientation first."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888888;")
        layout.addWidget(info)

        # View selection
        views_group = QGroupBox("Views to Include")
        views_layout = QVBoxLayout(views_group)

        self.xy_check = QCheckBox("XY predictions")
        self.xy_check.setChecked(True)
        views_layout.addWidget(self.xy_check)

        self.xz_check = QCheckBox("XZ predictions")
        self.xz_check.setChecked(True)
        views_layout.addWidget(self.xz_check)

        self.yz_check = QCheckBox("YZ predictions")
        self.yz_check.setChecked(True)
        views_layout.addWidget(self.yz_check)

        views_layout.addSpacing(10)

        self.diag1_check = QCheckBox("Diagonal ZX 45 predictions")
        views_layout.addWidget(self.diag1_check)

        self.diag2_check = QCheckBox("Diagonal ZY 45 predictions")
        views_layout.addWidget(self.diag2_check)

        self.diag3_check = QCheckBox("Diagonal ZX 30 predictions")
        views_layout.addWidget(self.diag3_check)

        layout.addWidget(views_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QFormLayout(options_group)

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(50, 500)
        self.chunk_size.setValue(200)
        options_layout.addRow("Chunk Size:", self.chunk_size)

        self.workers = QSpinBox()
        self.workers.setRange(1, 32)
        self.workers.setValue(8)
        options_layout.addRow("Workers:", self.workers)

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

        self.start_btn = QPushButton("Generate Consensus")
        self.start_btn.clicked.connect(self._on_start)
        button_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        layout.addStretch()

    def set_config(self, config: dict):
        """Set configuration from previous steps."""
        self.config = config

        # Get prediction_dirs from config, or auto-detect from project folder
        pred_dirs = config.get('prediction_dirs', {})
        if not pred_dirs:
            pred_dirs = self._detect_prediction_dirs()
            self.config['prediction_dirs'] = pred_dirs

        print(f"VotingPage.set_config: prediction_dirs = {pred_dirs}")

        # Enable/disable checkboxes based on available predictions
        self.xy_check.setEnabled(bool(pred_dirs.get('xy')) and os.path.isdir(pred_dirs.get('xy', '')))
        self.xz_check.setEnabled(bool(pred_dirs.get('xz')) and os.path.isdir(pred_dirs.get('xz', '')))
        self.yz_check.setEnabled(bool(pred_dirs.get('yz')) and os.path.isdir(pred_dirs.get('yz', '')))
        self.diag1_check.setEnabled(bool(pred_dirs.get('diag_zx45')) and os.path.isdir(pred_dirs.get('diag_zx45', '')))
        self.diag2_check.setEnabled(bool(pred_dirs.get('diag_zy45')) and os.path.isdir(pred_dirs.get('diag_zy45', '')))
        self.diag3_check.setEnabled(bool(pred_dirs.get('diag_zx30')) and os.path.isdir(pred_dirs.get('diag_zx30', '')))

        # Check enabled boxes by default
        for check in [self.xy_check, self.xz_check, self.yz_check,
                      self.diag1_check, self.diag2_check, self.diag3_check]:
            check.setChecked(check.isEnabled())

    def _detect_prediction_dirs(self) -> dict:
        """Auto-detect prediction directories from project folder structure."""
        pred_dirs = {}
        project_dir = self.config.get('project_dir', '')
        if not project_dir or not os.path.isdir(project_dir):
            return pred_dirs

        pred_base = os.path.join(project_dir, 'predictions')
        if not os.path.isdir(pred_base):
            return pred_dirs

        # Check for standard prediction directory names
        pred_mapping = {
            'xy': 'xy_predictions',
            'xz': 'xz_predictions',
            'yz': 'yz_predictions',
            'diag_zx45': 'diag_zx45_predictions',
            'diag_zy45': 'diag_zy45_predictions',
            'diag_zx30': 'diag_zx30_predictions',
        }

        for key, dirname in pred_mapping.items():
            dir_path = os.path.join(pred_base, dirname)
            if os.path.isdir(dir_path) and os.listdir(dir_path):
                pred_dirs[key] = dir_path
                print(f"  Auto-detected prediction: {key} -> {dir_path}")

        return pred_dirs

    def _on_start(self):
        """Start the voting process."""
        # First, rotate diagonal predictions back to XY
        self._rotate_diagonals()

    def _rotate_diagonals(self):
        """Rotate diagonal predictions back to XY orientation."""
        pred_dirs = self.config.get('prediction_dirs', {})
        project_dir = self.config.get('project_dir', os.path.expanduser('~/'))

        diagonals = []
        if self.diag1_check.isChecked() and 'diag_zx45' in pred_dirs:
            diagonals.append({
                'name': 'diag_zx45',
                'input_dir': pred_dirs['diag_zx45'],
                'output_dir': os.path.join(project_dir, 'rotated_predictions', 'diag_zx45_rotated'),
                'angle': 45,
                'axes': (0, 2),
            })
        if self.diag2_check.isChecked() and 'diag_zy45' in pred_dirs:
            diagonals.append({
                'name': 'diag_zy45',
                'input_dir': pred_dirs['diag_zy45'],
                'output_dir': os.path.join(project_dir, 'rotated_predictions', 'diag_zy45_rotated'),
                'angle': 45,
                'axes': (0, 1),
            })
        if self.diag3_check.isChecked() and 'diag_zx30' in pred_dirs:
            diagonals.append({
                'name': 'diag_zx30',
                'input_dir': pred_dirs['diag_zx30'],
                'output_dir': os.path.join(project_dir, 'rotated_predictions', 'diag_zx30_rotated'),
                'angle': 30,
                'axes': (0, 2),
            })

        if diagonals:
            from ..workers.rotate_worker import RotateWorker

            xy_dir = pred_dirs.get('xy', '')
            rotate_config = {
                'xy_dir': xy_dir,
                'diagonals': diagonals,
            }

            self._log("Rotating diagonal predictions back to XY orientation...")
            self.rotate_worker = RotateWorker(rotate_config)
            self.rotate_worker.started.connect(self._on_rotate_started)
            self.rotate_worker.progress.connect(self._on_progress)
            self.rotate_worker.log.connect(self._log)
            self.rotate_worker.finished.connect(self._on_rotate_finished)
            self.rotate_worker.start()
        else:
            # No diagonals, go straight to voting
            self._start_voting({})

    def _on_rotate_started(self):
        """Handle rotate worker started."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Rotating diagonal predictions...")

    def _on_rotate_finished(self, success: bool, result: dict):
        """Handle rotate worker finished."""
        self.rotate_worker = None

        if success:
            self._log("Diagonal rotation complete!")
            self._start_voting(result)
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Rotation failed")
            self._log(f"Error: {result.get('error', 'Unknown error')}")

    def _start_voting(self, rotated_dirs: dict):
        """Start the voting worker."""
        from ..workers.voting_worker import VotingWorker

        pred_dirs = self.config.get('prediction_dirs', {})
        project_dir = self.config.get('project_dir', os.path.expanduser('~/'))

        xy_dir = pred_dirs.get('xy', '')
        yz_dir = pred_dirs.get('yz', '') if self.yz_check.isChecked() else None
        xz_dir = pred_dirs.get('xz', '') if self.xz_check.isChecked() else None

        diag_dirs = list(rotated_dirs.values()) if rotated_dirs else []

        voting_config = {
            'xy_dir': xy_dir,
            'yz_dir': yz_dir,
            'xz_dir': xz_dir,
            'diag_dirs': diag_dirs,
            'output_dir': os.path.join(project_dir, 'heatmap_consensus'),
            'chunk_size': self.chunk_size.value(),
            'max_workers': self.workers.value(),
        }

        self._log("Starting consensus voting...")
        self.worker = VotingWorker(voting_config)
        self.worker.started.connect(self._on_voting_started)
        self.worker.progress.connect(self._on_voting_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_voting_finished)
        self.worker.start()

    def _on_voting_started(self):
        """Handle voting worker started."""
        self.status_label.setText("Computing consensus...")

    def _on_voting_progress(self, current: int, total: int):
        """Handle voting progress."""
        self.status_label.setText(f"Processing chunk {current}/{total}")
        self.progress_bar.setValue(int(100 * current / total))

    def _on_voting_finished(self, success: bool, result: str):
        """Handle voting worker finished."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Wait for thread to fully finish before clearing reference
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)

        self.worker = None

        if success:
            self.status_label.setText("Consensus complete!")
            self._log(f"Heatmap saved to: {result}")
            self.config['heatmap_dir'] = result
            self.voting_complete.emit(result)
        else:
            self.status_label.setText("Voting failed")
            self._log(f"Error: {result}")

    def _on_stop(self):
        """Stop the current operation."""
        print("Stop button clicked (Voting)")
        if self.rotate_worker:
            self.rotate_worker.stop()
        if self.worker:
            self.worker.stop()
        self._log("Stopping...")

    def _on_progress(self, name: str, current: int, total: int):
        """Handle progress signal."""
        self.status_label.setText(f"Processing {name}: {current}/{total}")
        self.progress_bar.setValue(int(100 * current / total))

    def _log(self, message: str):
        """Add log message."""
        self.log_text.append(message)
