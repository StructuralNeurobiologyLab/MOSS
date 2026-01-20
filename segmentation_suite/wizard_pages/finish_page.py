#!/usr/bin/env python3
"""
Finish page for the training wizard - summary and launch annotation tool.
"""

import os
import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class FinishPage(QWidget):
    """Finish page showing summary and options to open annotation tool."""

    # Signals
    open_annotation_tool = pyqtSignal(str, str)  # images_dir, masks_dir

    def __init__(self):
        super().__init__()
        self.config = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("Workflow Complete")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Summary
        summary_group = QGroupBox("Summary")
        summary_layout = QFormLayout(summary_group)

        self.project_label = QLabel("--")
        summary_layout.addRow("Project:", self.project_label)

        self.checkpoint_label = QLabel("--")
        summary_layout.addRow("Model Checkpoint:", self.checkpoint_label)

        self.heatmap_label = QLabel("--")
        summary_layout.addRow("Consensus Heatmap:", self.heatmap_label)

        layout.addWidget(summary_group)

        # Output locations
        output_group = QGroupBox("Output Locations")
        output_layout = QFormLayout(output_group)

        self.predictions_label = QLabel("--")
        output_layout.addRow("Predictions:", self.predictions_label)

        self.reslices_label = QLabel("--")
        output_layout.addRow("Reslices:", self.reslices_label)

        layout.addWidget(output_group)

        # Info text
        info = QLabel(
            "The consensus heatmap has been generated. You can now:\n"
            "1. Threshold the heatmap to create binary masks\n"
            "2. Use the annotation tool to proofread and refine the results\n"
            "3. Use the refined masks to retrain the model"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888888; padding: 10px;")
        layout.addWidget(info)

        # Action buttons
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout(button_group)

        # Open in annotation tool
        open_btn = QPushButton("Open Heatmap in Annotation Tool")
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                padding: 15px 30px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        open_btn.clicked.connect(self._open_annotation_tool)
        button_layout.addWidget(open_btn)

        # Open output folder
        folder_btn = QPushButton("Open Output Folder")
        folder_btn.clicked.connect(self._open_output_folder)
        button_layout.addWidget(folder_btn)

        layout.addWidget(button_group)

        layout.addStretch()

    def set_config(self, config: dict):
        """Set configuration from previous steps."""
        self.config = config

        # Update summary labels
        self.project_label.setText(config.get('project_name', 'Unknown'))

        checkpoint = config.get('checkpoint_path', '')
        if checkpoint:
            self.checkpoint_label.setText(os.path.basename(checkpoint))
            self.checkpoint_label.setToolTip(checkpoint)

        heatmap_dir = config.get('heatmap_dir', '')
        if heatmap_dir:
            self.heatmap_label.setText(os.path.basename(heatmap_dir))
            self.heatmap_label.setToolTip(heatmap_dir)

        project_dir = config.get('project_dir', '')
        if project_dir:
            pred_dir = os.path.join(project_dir, 'predictions')
            if os.path.isdir(pred_dir):
                self.predictions_label.setText(pred_dir)
            else:
                self.predictions_label.setText("Not created")

            reslice_dir = os.path.join(project_dir, 'reslices')
            if os.path.isdir(reslice_dir):
                self.reslices_label.setText(reslice_dir)
            else:
                self.reslices_label.setText("Not created")

    def _open_annotation_tool(self):
        """Open the annotation tool with the heatmap loaded."""
        heatmap_dir = self.config.get('heatmap_dir', '')
        raw_images_dir = self.config.get('raw_images_dir', self.config.get('train_images', ''))

        if heatmap_dir and os.path.isdir(heatmap_dir):
            # Try to launch annotation-tool command
            try:
                # Launch annotation tool as separate process
                # The annotation tool should accept command line arguments
                # for loading images and masks
                subprocess.Popen([
                    'annotation-tool',
                    '--images', raw_images_dir,
                    '--masks', heatmap_dir,
                ])
            except FileNotFoundError:
                # If annotation-tool command not found, try importing directly
                try:
                    from annotation_tool.main_window import MainWindow
                    from PyQt6.QtWidgets import QApplication
                    # Note: This would need the main window to be shown
                    # For now, just emit the signal
                    self.open_annotation_tool.emit(raw_images_dir, heatmap_dir)
                except ImportError:
                    self.open_annotation_tool.emit(raw_images_dir, heatmap_dir)
        else:
            self.open_annotation_tool.emit(raw_images_dir, heatmap_dir)

    def _open_output_folder(self):
        """Open the project output folder in file manager."""
        project_dir = self.config.get('project_dir', '')
        if project_dir and os.path.isdir(project_dir):
            import platform
            if platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', project_dir])
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', project_dir])
            elif platform.system() == 'Windows':
                subprocess.Popen(['explorer', project_dir])
