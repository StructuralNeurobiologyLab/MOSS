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

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QComboBox, QFileDialog, QGridLayout, QGroupBox
)

from .animals import animal_for_index, color_for_index
from .user_tile import UserTile


def build_prediction_arch_map() -> dict:
    """Return {arch_id: display_name} identical to MOSS's prediction dropdown set.

    Mirrors interactive_training_page._populate_prediction_model_combo: the
    non-hidden file architectures plus any hidden ones that ship a pretrained
    checkpoint (e.g. the LSD model). Keeps the hub's model list in lock-step
    with what users actually see.
    """
    from ..models.unet import get_available_architectures
    from ..models.architectures import (
        get_available_architectures as _registry_architectures,
        is_pretrained_architecture,
    )
    architectures = get_available_architectures()
    for arch_id, name in _registry_architectures(include_hidden=True).items():
        if arch_id not in architectures and is_pretrained_architecture(arch_id):
            architectures[arch_id] = name
    return architectures


class HubWindow(QMainWindow):
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self._tiles: dict[str, UserTile] = {}
        self._data_dir = None

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
            QComboBox QAbstractItemView {
                background:#2a2a2e; color:#eee;
                selection-background-color:#2f6fd0; selection-color:#ffffff;
                border:1px solid #444; outline:none;
            }
        """)

    _CODE_PLACEHOLDER = "—" * 6  # em-dashes shown before a session starts

    def _make_copy_button(self, source_label, empty_guard=None):
        """A small 'copy' button with hover/pressed feedback + a transient 'copied!'.

        Its pseudo-state rules live on the button's OWN stylesheet so they win
        over the global QPushButton style (Qt cascade). Uses one reusable
        window-parented single-shot timer, so rapid clicks don't stack.
        """
        btn = QPushButton("copy")
        btn.setFixedHeight(24)
        btn.setMinimumWidth(64)  # keep width steady across 'copy' / 'copied!'
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton { background:#333; color:#ddd; border:none;
                          padding:2px 8px; font-size:11px; border-radius:4px; }
            QPushButton:hover   { background:#454545; }
            QPushButton:pressed { background:#f4c542; color:#161618; }
        """)
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: btn.setText("copy"))

        def do_copy():
            text = source_label.text().strip()
            if not text or (empty_guard and empty_guard(text)):
                return
            QGuiApplication.clipboard().setText(text)
            btn.setText("copied!")
            timer.start(1200)

        btn.clicked.connect(do_copy)
        return btn

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("QFrame { background:#0f0f10; border-bottom:1px solid #2a2a2a; }")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(24)

        title = QLabel("MOSS · HUB")
        title.setStyleSheet("font-size:20px; font-weight:800; color:#4d90e6;")
        lay.addWidget(title)

        # Connect address (big, copyable) — THIS is what LAN clients type to join.
        addr_box = QVBoxLayout()
        addr_box.setSpacing(0)
        addr_cap = QLabel("CONNECT ADDRESS  ·  share this")
        addr_cap.setStyleSheet("color:#777; font-size:10px; font-weight:bold;")
        addr_row = QHBoxLayout()
        addr_row.setSpacing(6)
        self.address_label = QLabel("—")
        self.address_label.setStyleSheet(
            "font-size:20px; font-weight:800; color:#f4c542; font-family:monospace;"
        )
        addr_row.addWidget(self.address_label)
        addr_row.addWidget(self._make_copy_button(self.address_label))
        self.address_hint = QLabel("clients: Multi-User → Join → LAN → paste this")
        self.address_hint.setStyleSheet("color:#8a8a8a; font-size:10px;")
        addr_box.addWidget(addr_cap)
        addr_box.addLayout(addr_row)
        addr_box.addWidget(self.address_hint)
        lay.addLayout(addr_box)

        # Session code — a session label/identifier, NOT the LAN connect string.
        code_box = QVBoxLayout()
        code_box.setSpacing(0)
        code_cap = QLabel("SESSION CODE")
        code_cap.setStyleSheet("color:#777; font-size:10px; font-weight:bold;")
        code_row = QHBoxLayout()
        code_row.setSpacing(6)
        self.code_label = QLabel(self._CODE_PLACEHOLDER)
        self.code_label.setStyleSheet(
            "font-size:15px; font-weight:700; color:#b9a24a; letter-spacing:2px;"
        )
        code_row.addWidget(self.code_label)
        code_row.addWidget(self._make_copy_button(
            self.code_label, empty_guard=lambda t: t.startswith("—")))
        code_note = QLabel("session label — not for joining")
        code_note.setStyleSheet("color:#6a6a6a; font-size:9px;")
        code_box.addWidget(code_cap)
        code_box.addLayout(code_row)
        code_box.addWidget(code_note)
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
        # Same model set the user sees in MOSS; arch_id stored as item data.
        try:
            arch_map = build_prediction_arch_map()
        except Exception as e:
            print(f"[Hub] could not load architectures: {e}")
            arch_map = {}
        for arch_id, display_name in arch_map.items():
            short = display_name.replace('UNet ', '').replace('(', '').replace(')', '')
            self.pred_combo.addItem(short, arch_id)
        # Authoritative: changing this broadcasts to every client immediately,
        # and each client that joins is locked to the current choice on connect.
        self.pred_combo.currentIndexChanged.connect(self._on_prediction_changed)
        p_lay.addWidget(self.pred_combo)
        note = QLabel("Applied automatically — clients' prediction dropdown "
                      "locks (red) to this choice on join and on change.")
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
        if hasattr(self.backend, "user_restored"):
            self.backend.user_restored.connect(self._on_user_restored)
        self.backend.crop_received.connect(self._on_crop_received)
        self.backend.training_status.connect(self._on_training_status)
        if hasattr(self.backend, "prediction_model_set"):
            self.backend.prediction_model_set.connect(self._select_prediction_model)

    def _on_session_started(self, code: str, data_dir: str, connect_addr: str = ""):
        self.code_label.setText(code)
        self.datadir_label.setText(data_dir)
        self._data_dir = data_dir
        if connect_addr:
            self.address_label.setText(connect_addr)
            if connect_addr.startswith("127.0.0.1") or connect_addr.startswith("localhost"):
                self.address_label.setStyleSheet(
                    "font-size:20px; font-weight:800; color:#e6a33c; font-family:monospace;")
                self.address_hint.setText("⚠ localhost only — other machines can't reach this")
                self.address_hint.setStyleSheet("color:#e6a33c; font-size:10px;")

    def _on_project_registered(self, name: str, subprojects: list):
        self.project_label.setText(name)
        self.subproj_label.setText("subprojects: " + ", ".join(subprojects))

    def _ensure_tile(self, user_id, display_name, is_owner, join_index):
        """Create the tile if new (stable animal/color by join_index), else reuse."""
        if self._empty_label is not None:
            self._empty_label.hide()
        tile = self._tiles.get(user_id)
        if tile is None:
            tile = UserTile(
                user_id, display_name,
                animal=animal_for_index(join_index), color=color_for_index(join_index),
                is_owner=is_owner,
            )
            tile.clicked.connect(self._open_gallery)
            tile.toggled.connect(self.backend.set_user_included)
            self._tiles[user_id] = tile
            self._relayout_tiles()
        return tile

    def _on_user_connected(self, user_id: str, display_name: str, is_owner: bool, join_index: int):
        tile = self._ensure_tile(user_id, display_name, is_owner, join_index)
        tile.set_online(True)
        self._update_count()

    def _on_user_restored(self, user_id: str, display_name: str, is_owner: bool,
                          join_index: int, crop_count: int, included: bool):
        """Rebuild a tile for a resumed session — offline until the user reconnects."""
        tile = self._ensure_tile(user_id, display_name, is_owner, join_index)
        tile.set_crop_count(crop_count)
        tile.set_included(included)
        tile.set_online(False)
        self._update_count()

    def _on_user_disconnected(self, user_id: str):
        # Keep the tile — the user's crops persist and they may reconnect.
        tile = self._tiles.get(user_id)
        if tile:
            tile.set_online(False)
        self._update_count()

    def _on_crop_received(self, user_id: str, png_bytes: bytes, caption: str):
        # Crops are persisted to disk by the hub; the tile just tracks the count.
        tile = self._tiles.get(user_id)
        if tile:
            tile.increment_crops()

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
        online = sum(1 for t in self._tiles.values() if getattr(t, "_online", True))
        total = len(self._tiles)
        if online == total:
            self.count_label.setText(f"{online} connected")
        else:
            self.count_label.setText(f"{online} online · {total} total")

    def _open_gallery(self, user_id: str):
        """Open the user's crops in the same Review Crops tool used in MOSS."""
        from pathlib import Path

        tile = self._tiles.get(user_id)
        if not tile or not self._data_dir:
            return
        base = Path(self._data_dir) / "incoming" / user_id
        images = base / "train_images"
        masks = base / "train_masks"
        if not images.exists() or not any(images.glob("*.png")):
            # Keep the guard (a missing dir crashes the reviewer; an empty one
            # opens a useless window) but surface it non-modally.
            self.statusBar().showMessage(
                f"No crops received from {tile.display_name} yet.", 4000)
            return

        from ..widgets.training_data_reviewer import TrainingDataReviewer
        reviewer = TrainingDataReviewer(images, masks, base, parent=self)
        reviewer.setWindowTitle(f"Review crops — {tile.display_name}")
        reviewer.exec()
        # Discards may have removed crops — refresh the tile count from disk.
        tile.set_crop_count(len(list(images.glob("*.png"))))

    def _choose_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose the Hub's main data folder")
        if path:
            self.datadir_label.setText(path)
            self.backend.set_data_dir(path)

    def _on_prediction_changed(self, index: int):
        # Broadcast the arch_id to all clients immediately; joiners synced on connect.
        arch_id = self.pred_combo.itemData(index)
        if arch_id:
            self.backend.set_prediction_model(arch_id)

    def _select_prediction_model(self, arch_id: str):
        """Reflect the owner's registered/authoritative model without re-broadcasting."""
        if not arch_id:
            return
        idx = self.pred_combo.findData(arch_id)
        if idx >= 0:
            self.pred_combo.blockSignals(True)
            self.pred_combo.setCurrentIndex(idx)
            self.pred_combo.blockSignals(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_tiles()
