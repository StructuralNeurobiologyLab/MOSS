#!/usr/bin/env python3
"""
Export / finish page for the training wizard.
"""

import os
import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout
)
from PyQt6.QtGui import QFont


class FinishPage(QWidget):
    """Finish page showing project summary and output folder access."""

    def __init__(self):
        super().__init__()
        self.config = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("Export")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Summary
        summary_group = QGroupBox("Project Summary")
        summary_layout = QFormLayout(summary_group)

        self.project_label = QLabel("--")
        summary_layout.addRow("Project:", self.project_label)

        self.project_dir_label = QLabel("--")
        self.project_dir_label.setWordWrap(True)
        summary_layout.addRow("Location:", self.project_dir_label)

        self.checkpoint_label = QLabel("--")
        summary_layout.addRow("Model Checkpoint:", self.checkpoint_label)

        layout.addWidget(summary_group)

        # Actions
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout(button_group)

        folder_btn = QPushButton("Open Project Folder")
        folder_btn.setStyleSheet("""
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
        folder_btn.clicked.connect(self._open_output_folder)
        button_layout.addWidget(folder_btn)

        layout.addWidget(button_group)

        layout.addStretch()

    def set_config(self, config: dict):
        """Set configuration from previous steps."""
        self.config = config

        self.project_label.setText(config.get('project_name', 'Unknown'))

        project_dir = config.get('project_dir', '')
        if project_dir:
            self.project_dir_label.setText(project_dir)
            self.project_dir_label.setToolTip(project_dir)

        checkpoint = config.get('checkpoint_path', '')
        if checkpoint:
            self.checkpoint_label.setText(os.path.basename(checkpoint))
            self.checkpoint_label.setToolTip(checkpoint)

    def _open_output_folder(self):
        """Open the project folder in the system file manager."""
        project_dir = self.config.get('project_dir', '')
        if project_dir and os.path.isdir(project_dir):
            import platform
            if platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', project_dir])
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', project_dir])
            elif platform.system() == 'Windows':
                subprocess.Popen(['explorer', project_dir])
