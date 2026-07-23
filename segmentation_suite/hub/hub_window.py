#!/usr/bin/env python3
"""
HubWindow — the Hub controller GUI.

This is the operator's view of a joint multi-user training session (distinct
from the per-user MOSS wizard). It shows the session code, the project the
owner registered, a live grid of connected users (each a colored, animal-badged
tile), and the controls that make the hub authoritative: include/exclude a
user's crops, reset the model, and dictate the prediction model every client
must use.

The window is driven entirely by a backend object exposing the signals/methods
documented in mock_backend.MockHubBackend, so it runs on a laptop against the
mock and against the real HubServer later without change.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QComboBox, QFileDialog, QGridLayout, QGroupBox
)

from .animals import animal_for_index, color_for_index
from .user_tile import UserTile
from .crop_gallery import CropGallery


# Prediction models the hub can dictate to clients (mirrors MOSS arch names).
PREDICTION_MODELS = [
    "unet_deep_dice_25d",
    "unet_deep_dice",
    "unet_increased_rf",
    "mtlsd_25d",
]


class HubWindow(QMainWindow):
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self._tiles: dict[str, UserTile] = {}
        self._galleries: dict[str, CropGallery] = {}
        self._crop_buffer: dict[str, list] = {}   # user_id -> [(QImage, caption)]
        self._crop_buffer_cap = 200
        self._join_index = 0

        self.setWindowTitle("MOSS Hub — Multi-User Session Controller")
        self.resize(1040, 720)
        self._build_ui()
        self._connect_backend()

    # --------------------------------------------------------------- build UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())

        body = QHBoxLayout()
        body.setContentsMargins(14, 14, 14, 14)
        body.setSpacing(14)
        body.addWidget(self._build_user_area(), stretch=3)
        body.addWidget(self._build_control_panel(), stretch=1)
        body_w = QWidget()
        body_w.setLayout(body)
        root.addWidget(body_w, stretch=1)

        self.setStyleSheet("""
            QMainWindow { background:#161618; }
            QLabel { color:#e8e8e8; }
            QGroupBox {
                color:#c0c0c0; border:1px solid #333; border-radius:8px;
                margin-top:10px; padding:10px; font-weight:bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; }
            QPushButton {
                background:#2f6fd0; color:white; border:none;
                padding:9px 14px; border-radius:6px; font-weight:600;
            }
            QPushButton:hover { background:#3a7ee0; }
            QPushButton#danger { background:#b8433a; }
            QPushButton#danger:hover { background:#cc4d43; }
            QComboBox {
                background:#2a2a2e; color:#eee; border:1px solid #444;
                padding:6px; border-radius:5px;
            }
        """)

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("QFrame { background:#0f0f10; border-bottom:1px solid #2a2a2a; }")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(24)

        title = QLabel("MOSS · HUB")
        title.setStyleSheet("font-size:20px; font-weight:800; color:#4d90e6;")
        lay.addWidget(title)

        # Session code (big, copyable)
        code_box = QVBoxLayout()
        code_box.setSpacing(0)
        code_cap = QLabel("SESSION CODE")
        code_cap.setStyleSheet("color:#777; font-size:10px; font-weight:bold;")
        code_row = QHBoxLayout()
        code_row.setSpacing(6)
        self.code_label = QLabel("——————")
        self.code_label.setStyleSheet(
            "font-size:22px; font-weight:800; color:#f4c542; letter-spacing:3px;"
        )
        copy_btn = QPushButton("copy")
        copy_btn.setFixedHeight(24)
        copy_btn.setStyleSheet("background:#333; padding:2px 8px; font-size:11px;")
        copy_btn.clicked.connect(self._copy_code)
        code_row.addWidget(self.code_label)
        code_row.addWidget(copy_btn)
        code_box.addWidget(code_cap)
        code_box.addLayout(code_row)
        lay.addLayout(code_box)

        # Project
        proj_box = QVBoxLayout()
        proj_box.setSpacing(0)
        proj_cap = QLabel("PROJECT")
        proj_cap.setStyleSheet("color:#777; font-size:10px; font-weight:bold;")
        self.project_label = QLabel("— waiting for owner —")
        self.project_label.setStyleSheet("font-size:15px; font-weight:700; color:#e8e8e8;")
        self.subproj_label = QLabel("")
        self.subproj_label.setStyleSheet("font-size:11px; color:#8a8a8a;")
        proj_box.addWidget(proj_cap)
        proj_box.addWidget(self.project_label)
        proj_box.addWidget(self.subproj_label)
        lay.addLayout(proj_box)

        lay.addStretch()

        # Connected count
        self.count_label = QLabel("0 connected")
        self.count_label.setStyleSheet("font-size:14px; color:#5fbf6a; font-weight:700;")
        lay.addWidget(self.count_label)
        return bar

    def _build_user_area(self) -> QWidget:
        wrap = QGroupBox("Connected users — click a tile to view their crops")
        outer = QVBoxLayout(wrap)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        self._tiles_container = QWidget()
        self._tiles_container.setStyleSheet("background:transparent;")
        scroll.viewport().setStyleSheet("background:transparent;")
        self._tiles_grid = QGridLayout(self._tiles_container)
        self._tiles_grid.setSpacing(14)
        self._tiles_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self._tiles_container)
        outer.addWidget(scroll)

        self._empty_label = QLabel("No users connected yet.\nShare the session code to invite collaborators.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color:#666; font-size:14px;")
        self._tiles_grid.addWidget(self._empty_label, 0, 0)
        return wrap

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(300)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        # Storage
        store = QGroupBox("Storage")
        s_lay = QVBoxLayout(store)
        self.datadir_label = QLabel("—")
        self.datadir_label.setWordWrap(True)
        self.datadir_label.setStyleSheet("color:#bbb; font-size:11px; font-weight:normal;")
        change_btn = QPushButton("Change data folder…")
        change_btn.clicked.connect(self._choose_data_dir)
        s_lay.addWidget(QLabel("Main folder (crops, models):"))
        s_lay.addWidget(self.datadir_label)
        s_lay.addWidget(change_btn)
        lay.addWidget(store)

        # Training
        train = QGroupBox("Cluster training")
        t_lay = QVBoxLayout(train)
        self.round_label = QLabel("Round: 0")
        self.loss_label = QLabel("Loss: —")
        self.contrib_label = QLabel("Contributing users: 0")
        for w in (self.round_label, self.loss_label, self.contrib_label):
            w.setStyleSheet("color:#ccc; font-size:12px; font-weight:normal;")
            t_lay.addWidget(w)
        reset_btn = QPushButton("Reset model")
        reset_btn.setObjectName("danger")
        reset_btn.clicked.connect(self.backend.reset_model)
        t_lay.addWidget(reset_btn)
        lay.addWidget(train)

        # Prediction authority
        pred = QGroupBox("Prediction model (authoritative)")
        p_lay = QVBoxLayout(pred)
        p_lay.addWidget(QLabel("All clients predict with:"))
        self.pred_combo = QComboBox()
        self.pred_combo.addItems(PREDICTION_MODELS)
        p_lay.addWidget(self.pred_combo)
        apply_btn = QPushButton("Push to all clients")
        apply_btn.clicked.connect(self._push_prediction_model)
        p_lay.addWidget(apply_btn)
        note = QLabel("Clients' prediction dropdown locks (red) to this choice.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:10px; font-weight:normal;")
        p_lay.addWidget(note)
        lay.addWidget(pred)

        lay.addStretch()
        return panel

    # ------------------------------------------------------------- backend io
    def _connect_backend(self):
        self.backend.session_started.connect(self._on_session_started)
        self.backend.project_registered.connect(self._on_project_registered)
        self.backend.user_connected.connect(self._on_user_connected)
        self.backend.user_disconnected.connect(self._on_user_disconnected)
        self.backend.crop_received.connect(self._on_crop_received)
        self.backend.training_status.connect(self._on_training_status)

    def _on_session_started(self, code: str, data_dir: str):
        self.code_label.setText(code)
        self.datadir_label.setText(data_dir)

    def _on_project_registered(self, name: str, subprojects: list):
        self.project_label.setText(name)
        self.subproj_label.setText("subprojects: " + ", ".join(subprojects))

    def _on_user_connected(self, user_id: str, display_name: str, is_owner: bool):
        if self._empty_label is not None:
            self._empty_label.hide()
        idx = self._join_index
        self._join_index += 1
        tile = UserTile(
            user_id, display_name,
            animal=animal_for_index(idx), color=color_for_index(idx),
            is_owner=is_owner,
        )
        tile.clicked.connect(self._open_gallery)
        tile.toggled.connect(self.backend.set_user_included)
        self._tiles[user_id] = tile
        self._relayout_tiles()
        self._update_count()

    def _on_user_disconnected(self, user_id: str):
        tile = self._tiles.pop(user_id, None)
        if tile:
            tile.setParent(None)
            tile.deleteLater()
        self._relayout_tiles()
        self._update_count()

    def _on_crop_received(self, user_id: str, image, caption: str):
        tile = self._tiles.get(user_id)
        if tile:
            tile.increment_crops()
        buf = self._crop_buffer.setdefault(user_id, [])
        buf.append((image, caption))
        if len(buf) > self._crop_buffer_cap:
            del buf[0]
        gallery = self._galleries.get(user_id)
        if gallery and gallery.isVisible():
            gallery.add_crop(image, caption)

    def _on_training_status(self, rnd: int, loss: float, contributors: int):
        self.round_label.setText(f"Round: {rnd}")
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.contrib_label.setText(f"Contributing users: {contributors}")

    # ----------------------------------------------------------------- helpers
    def _relayout_tiles(self):
        cols = max(1, (self._tiles_container.width() or 700) // 168)
        for i, tile in enumerate(self._tiles.values()):
            self._tiles_grid.addWidget(tile, i // cols, i % cols)

    def _update_count(self):
        n = len(self._tiles)
        self.count_label.setText(f"{n} connected")

    def _open_gallery(self, user_id: str):
        tile = self._tiles.get(user_id)
        if not tile:
            return
        gallery = self._galleries.get(user_id)
        if gallery is None:
            gallery = CropGallery(user_id, tile.display_name, tile.color, self)
            # Backfill crops already received before the gallery was opened.
            for image, caption in self._crop_buffer.get(user_id, []):
                gallery.add_crop(image, caption)
            self._galleries[user_id] = gallery
        gallery.show()
        gallery.raise_()
        gallery.activateWindow()

    def _copy_code(self):
        QGuiApplication.clipboard().setText(self.code_label.text())

    def _choose_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose the Hub's main data folder")
        if path:
            self.datadir_label.setText(path)
            self.backend.set_data_dir(path)

    def _push_prediction_model(self):
        self.backend.set_prediction_model(self.pred_combo.currentText())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_tiles()
