#!/usr/bin/env python3
"""
MockHubBackend — a fake backend that simulates a live multi-user session so the
Hub GUI can be developed and demoed on a laptop with no cluster and no network.

It mimics the signal surface the real HubServer will expose, so the GUI can be
wired against this now and swapped for the real server later without changes.

Backend -> GUI signals:
    session_started(code, data_dir)
    project_registered(project_name, subprojects)
    user_connected(user_id, display_name, is_owner)
    user_disconnected(user_id)
    crop_received(user_id, QImage, caption)
    training_status(round, loss, num_samples)

GUI -> Backend methods:
    set_user_included(user_id, included)
    reset_model()
    set_prediction_model(arch)
    set_data_dir(path)
"""

from __future__ import annotations

import random

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QBuffer, QIODevice
from PyQt6.QtGui import QImage, qRgb

from ..network.session import get_local_ip


def _qimage_to_png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


_NAMES = ["Nelson", "Sofia", "Amir", "Yuki", "Priya", "Leo", "Marta", "Kwame"]


def _make_fake_crop(seed: int, size: int = 96) -> QImage:
    """Generate a small grayscale 'EM-ish' crop with a bright blob (mock)."""
    rng = random.Random(seed)
    img = QImage(size, size, QImage.Format.Format_RGB32)
    base = rng.randint(40, 90)
    cx, cy = rng.randint(20, size - 20), rng.randint(20, size - 20)
    r = rng.randint(10, 22)
    for y in range(size):
        for x in range(size):
            v = base + rng.randint(-15, 15)
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                v = min(255, v + 120)  # the labelled object
            v = max(0, min(255, v))
            img.setPixel(x, y, qRgb(v, v, v))
    return img


class MockHubBackend(QObject):
    session_started = pyqtSignal(str, str, str)
    project_registered = pyqtSignal(str, list)
    user_connected = pyqtSignal(str, str, bool)
    user_disconnected = pyqtSignal(str)
    crop_received = pyqtSignal(str, bytes, str)
    training_status = pyqtSignal(int, float, int)
    prediction_model_set = pyqtSignal(str)

    def __init__(self, data_dir: str = "~/ceph/moss_hub_demo", parent=None):
        super().__init__(parent)
        self._data_dir = data_dir
        self._code = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
        self._users: dict[str, str] = {}          # user_id -> name
        self._included: dict[str, bool] = {}
        self._crop_seq = 0
        self._round = 0
        self._loss = 1.4
        self._name_pool = list(_NAMES)
        random.shuffle(self._name_pool)

        self._join_timer = QTimer(self)
        self._join_timer.timeout.connect(self._maybe_add_user)
        self._crop_timer = QTimer(self)
        self._crop_timer.timeout.connect(self._maybe_emit_crop)
        self._train_timer = QTimer(self)
        self._train_timer.timeout.connect(self._tick_training)

    # ------------------------------------------------------------------ start
    def start(self):
        self.session_started.emit(self._code, self._data_dir, f"{get_local_ip()}:8765")
        # First user joins shortly and becomes owner, registering the project.
        QTimer.singleShot(600, self._add_owner)
        self._join_timer.start(3500)
        self._crop_timer.start(1400)
        self._train_timer.start(2500)

    # ------------------------------------------------------------------ users
    def _add_owner(self):
        uid = "user0"
        name = self._name_pool.pop() if self._name_pool else "Owner"
        self._users[uid] = name
        self._included[uid] = True
        self.user_connected.emit(uid, name, True)
        self.project_registered.emit(
            "songbird_em_v3", ["mitochondria", "synapses", "nuclei"]
        )
        self.prediction_model_set.emit("unet_deep_dice_dwarf25d_v2")

    def _maybe_add_user(self):
        if len(self._users) >= 6 or not self._name_pool:
            return
        if random.random() < 0.7:
            uid = f"user{len(self._users)}"
            name = self._name_pool.pop()
            self._users[uid] = name
            self._included[uid] = True
            self.user_connected.emit(uid, name, False)

    # ------------------------------------------------------------------ crops
    def _maybe_emit_crop(self):
        if not self._users:
            return
        uid = random.choice(list(self._users.keys()))
        self._crop_seq += 1
        img = _make_fake_crop(self._crop_seq)
        self.crop_received.emit(uid, _qimage_to_png_bytes(img), f"#{self._crop_seq} · z=1{self._crop_seq:03d}")

    # --------------------------------------------------------------- training
    def _tick_training(self):
        self._round += 1
        self._loss = max(0.02, self._loss * random.uniform(0.9, 0.99))
        included_users = sum(1 for v in self._included.values() if v)
        self.training_status.emit(self._round, self._loss, included_users)

    # ----------------------------------------------------- GUI -> backend API
    def set_user_included(self, user_id: str, included: bool):
        self._included[user_id] = included
        print(f"[MockBackend] user {user_id} included={included}")

    def reset_model(self):
        self._round = 0
        self._loss = 1.4
        print("[MockBackend] model reset")

    def set_prediction_model(self, arch: str):
        print(f"[MockBackend] prediction model -> {arch} (would broadcast to all clients)")

    def set_data_dir(self, path: str):
        self._data_dir = path
        print(f"[MockBackend] data dir -> {path}")
