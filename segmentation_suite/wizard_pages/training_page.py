#!/usr/bin/env python3
"""
Training page for the training wizard - shows training progress.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


class TrainingPage(QWidget):
    """Training page showing training progress."""

    # Signals
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()
    training_complete = pyqtSignal(str)  # checkpoint path

    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("Training")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Epoch progress
        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Epoch:"))
        self.epoch_label = QLabel("0 / 0")
        epoch_row.addWidget(self.epoch_label)
        epoch_row.addStretch()
        progress_layout.addLayout(epoch_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Loss display
        loss_row = QHBoxLayout()
        loss_row.addWidget(QLabel("Train Loss:"))
        self.train_loss_label = QLabel("--")
        loss_row.addWidget(self.train_loss_label)
        loss_row.addSpacing(20)
        loss_row.addWidget(QLabel("Val Loss:"))
        self.val_loss_label = QLabel("--")
        loss_row.addWidget(self.val_loss_label)
        loss_row.addStretch()
        progress_layout.addLayout(loss_row)

        layout.addWidget(progress_group)

        # Control buttons
        button_row = QHBoxLayout()

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self._on_start)
        button_row.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        self.pause_btn.setEnabled(False)
        button_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        layout.addStretch()

    def set_config(self, config: dict):
        """Set the training configuration."""
        self.config = config

    def start_training(self):
        """Start the training worker."""
        if self.worker is not None:
            return

        from ..workers.train_worker import TrainWorker

        self.worker = TrainWorker(self.config)
        self.worker.started.connect(self._on_worker_started)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_start(self):
        """Handle start button click."""
        self.start_training()

    def _on_pause(self):
        """Handle pause button click."""
        if self.worker:
            if self.worker.is_paused:
                self.worker.resume()
                self.pause_btn.setText("Pause")
            else:
                self.worker.pause()
                self.pause_btn.setText("Resume")

    def _on_stop(self):
        """Handle stop button click."""
        print("Stop button clicked (Training)")
        if self.worker:
            self.worker.stop()
            self._on_log("Stopping training...")

    def _on_worker_started(self):
        """Handle worker started signal."""
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.training_started.emit()

    def _on_progress(self, epoch: int, total: int, train_loss: float, val_loss: float):
        """Handle progress signal."""
        self.epoch_label.setText(f"{epoch} / {total}")
        self.progress_bar.setValue(int(100 * epoch / total))
        self.train_loss_label.setText(f"{train_loss:.4f}")
        if val_loss > 0:
            self.val_loss_label.setText(f"{val_loss:.4f}")

    def _on_log(self, message: str):
        """Handle log signal."""
        self.log_text.append(message)

    def _on_finished(self, success: bool, result: str):
        """Handle finished signal."""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.worker = None

        if success:
            self._on_log(f"Training complete! Checkpoint: {result}")
            self.training_complete.emit(result)
        else:
            self._on_log(f"Training failed: {result}")
