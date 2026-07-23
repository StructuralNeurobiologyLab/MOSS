#!/usr/bin/env python3
"""
HubServer — the authoritative multi-user hub backend.

Unlike the legacy AggregationServer (where the "host" was a MOSS wizard secretly
connected to its own server, and crops were relayed to it by a fragile
"first-non-sender-client = host" heuristic), the HubServer is the *terminal*
receiver: connected users are pure clients, and every crop lands directly on the
hub's disk under incoming/<user_id>/. It speaks the same wire protocol the
existing SyncClient already uses (HELLO / WELCOME / TRAINING_DATA / GLOBAL_MODEL),
plus the hub additions (PROJECT_REGISTER, SET_PREDICTION_MODEL, extended WELCOME).

It exposes the exact signal/method surface that MockHubBackend does, so the Hub
GUI is unchanged whether driven by the mock or by this real server.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from ..network.protocol import (
    Message, MessageType,
    create_welcome_message, create_user_list_message,
    create_set_prediction_model_message, create_training_data_ack_message,
)
from ..network.session import generate_session_id, get_local_ip


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [Hub] {msg}")


class _User:
    __slots__ = ("user_id", "display_name", "is_owner", "included", "crop_count", "ws")

    def __init__(self, user_id, display_name, is_owner, ws):
        self.user_id = user_id
        self.display_name = display_name
        self.is_owner = is_owner
        self.included = True
        self.crop_count = 0
        self.ws = ws


class HubServer(QObject):
    # Signal surface mirrors MockHubBackend exactly.
    session_started = pyqtSignal(str, str, str)         # code, data_dir, connect_addr
    project_registered = pyqtSignal(str, list)          # project_name, subprojects
    user_connected = pyqtSignal(str, str, bool)         # user_id, name, is_owner
    user_disconnected = pyqtSignal(str)                 # user_id
    crop_received = pyqtSignal(str, bytes, str)         # user_id, png_bytes, caption
    training_status = pyqtSignal(int, float, int)       # round, loss, contributors

    def __init__(self, data_dir: str, host: str = "0.0.0.0", port: int = 8765,
                 parent=None):
        super().__init__(parent)
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets not installed. pip install websockets")

        self.data_dir = Path(data_dir).expanduser()
        self.host = host
        self.port = port
        self._adv_host = host  # resolved to a routable IP in start()
        self.code = generate_session_id()

        # Authoritative session identity (set when the owner registers).
        self.project_name: str = ""
        self.owner_subproject: str = ""
        self.session_subproject: str = ""
        self.architecture: str = ""
        self.prediction_model: str = ""
        self.subprojects: list = []

        self._users: Dict[str, _User] = {}        # user_id -> _User
        self._ws_to_uid: Dict[object, str] = {}
        self._pending_td: Dict[str, dict] = {}     # user_id -> in-flight training-data frames
        self._crop_seq = 0

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._running = False

    def connect_address(self) -> str:
        """The bare IP:port a LAN client types to join (what the GUI shows)."""
        return f"{self._adv_host}:{self.port}"

    # ================================================================= startup
    def start(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "incoming").mkdir(exist_ok=True)
        # Resolve the address to advertise once. When bound to all interfaces,
        # fall back to the routable LAN IP so clients get something reachable.
        self._adv_host = self.host if self.host not in ("0.0.0.0", "::", "") else get_local_ip()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.session_started.emit(self.code, str(self.data_dir), self.connect_address())
        _log(f"session {self.code} · data_dir={self.data_dir} · connect at ws://{self.connect_address()}")
        if self._adv_host == "127.0.0.1":
            _log("WARNING: no LAN route detected — only localhost clients can connect")

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:  # pragma: no cover
            if self._running:  # suppress expected exceptions during stop()
                _log(f"server error: {e}")
        finally:
            self._loop.close()

    async def _serve(self):
        async with websockets.serve(
            self._handle_client, self.host, self.port,
            ping_interval=20, ping_timeout=60,
            max_size=500 * 1024 * 1024,
        ):
            _log(f"listening on {self.host}:{self.port}")
            while self._running:
                await asyncio.sleep(0.1)

    # ============================================================ client I/O
    async def _handle_client(self, websocket):
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._on_json(websocket, message)
                elif isinstance(message, bytes):
                    uid = self._ws_to_uid.get(websocket)
                    if uid:
                        await self._on_binary(uid, message)
        except Exception as e:
            _log(f"client loop ended: {e}")
        finally:
            await self._drop(websocket)

    async def _on_json(self, websocket, data: str):
        try:
            msg = Message.from_json(data)
        except Exception as e:
            _log(f"bad json: {e}")
            return

        if msg.type == MessageType.HELLO:
            await self._on_hello(websocket, msg)
        elif msg.type == MessageType.PROJECT_REGISTER:
            await self._on_project_register(websocket, msg)
        elif msg.type == MessageType.TRAINING_DATA:
            uid = self._ws_to_uid.get(websocket)
            if uid:
                self._pending_td[uid] = {"payload": msg.payload, "frames": []}
        elif msg.type == MessageType.GOODBYE:
            await self._drop(websocket)

    async def _on_hello(self, websocket, msg: Message):
        uid = msg.payload.get("user_id")
        name = msg.payload.get("display_name", "User")
        if not uid:
            return
        is_owner = len(self._users) == 0
        user = _User(uid, name, is_owner, websocket)
        self._users[uid] = user
        self._ws_to_uid[websocket] = uid
        (self.data_dir / "incoming" / uid / "train_images").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "incoming" / uid / "train_masks").mkdir(parents=True, exist_ok=True)
        _log(f"HELLO {name} ({uid}) owner={is_owner}")

        await self._send_welcome(user)
        await self._broadcast_user_list()
        self.user_connected.emit(uid, name, is_owner)

    async def _send_welcome(self, user: _User):
        # The owner keeps their own authoritative subproject; only joinees adopt
        # the namespaced session subproject (to avoid colliding with their own).
        session_subproject = None if user.is_owner else (self.session_subproject or None)
        welcome = create_welcome_message(
            session_id=self.code,
            user_list=self._user_list(),
            architecture=self.architecture or None,
            session_subproject=session_subproject,
            prediction_model=self.prediction_model or None,
            is_owner=user.is_owner,
        )
        await self._safe_send(user.ws, welcome.to_json())

    async def _on_project_register(self, websocket, msg: Message):
        uid = self._ws_to_uid.get(websocket)
        user = self._users.get(uid)
        if not user or not user.is_owner:
            _log("ignoring PROJECT_REGISTER from non-owner")
            return
        self.project_name = msg.payload.get("project_name", "")
        self.owner_subproject = msg.payload.get("subproject", "") or "default"
        self.architecture = msg.payload.get("architecture", "") or self.architecture
        self.prediction_model = msg.payload.get("prediction_model", "") or self.prediction_model
        self.subprojects = msg.payload.get("subprojects", [])
        # Stable, human-readable, quarantined session subproject name.
        safe_proj = self.project_name.replace("/", "-").replace("\\", "-")
        self.session_subproject = f"{self.owner_subproject}__{safe_proj}"
        _log(f"PROJECT_REGISTER project={self.project_name} subproject={self.owner_subproject} "
             f"-> session_subproject={self.session_subproject} arch={self.architecture}")
        self.project_registered.emit(self.project_name, self.subprojects)
        # Re-welcome everyone so late-arriving identity reaches earlier joiners.
        for u in list(self._users.values()):
            await self._send_welcome(u)

    async def _on_binary(self, uid: str, data: bytes):
        td = self._pending_td.get(uid)
        if td is None:
            return  # unexpected binary (hub does not accept client weights)
        td["frames"].append(data)
        if len(td["frames"]) >= 2:
            img_bytes, mask_bytes = td["frames"][0], td["frames"][1]
            payload = td["payload"]
            del self._pending_td[uid]
            self._store_crop(uid, img_bytes, mask_bytes, payload)
            ack = create_training_data_ack_message(uid, received=True)
            await self._safe_send(self._users[uid].ws, ack.to_json())

    def _store_crop(self, uid: str, img_bytes: bytes, mask_bytes: bytes, payload: dict):
        user = self._users.get(uid)
        if not user:
            return
        self._crop_seq += 1
        ts = payload.get("timestamp", int(time.time() * 1000))
        stem = f"{ts}_{self._crop_seq}"
        base = self.data_dir / "incoming" / uid
        (base / "train_images" / f"{stem}.png").write_bytes(img_bytes)
        (base / "train_masks" / f"{stem}.png").write_bytes(mask_bytes)
        user.crop_count += 1
        slice_idx = payload.get("slice_index", 0)
        caption = f"#{self._crop_seq} · z={slice_idx}"
        self.crop_received.emit(uid, img_bytes, caption)
        _log(f"crop from {user.display_name} -> incoming/{uid}/ ({len(img_bytes)+len(mask_bytes)} B)")

    async def _drop(self, websocket):
        uid = self._ws_to_uid.pop(websocket, None)
        if not uid:
            return
        user = self._users.pop(uid, None)
        self._pending_td.pop(uid, None)
        if user:
            _log(f"disconnect {user.display_name} ({uid})")
            self.user_disconnected.emit(uid)
        await self._broadcast_user_list()

    # ================================================================ helpers
    def _user_list(self) -> list:
        return [{"user_id": u.user_id, "display_name": u.display_name,
                 "is_owner": u.is_owner} for u in self._users.values()]

    async def _broadcast_user_list(self):
        msg = create_user_list_message(self._user_list()).to_json()
        for u in list(self._users.values()):
            await self._safe_send(u.ws, msg)

    async def _safe_send(self, ws, data):
        try:
            await ws.send(data)
        except Exception:
            pass

    # =================================================== GUI -> backend API
    def set_user_included(self, user_id: str, included: bool):
        user = self._users.get(user_id)
        if user:
            user.included = included
        _log(f"user {user_id} included={included}")

    def reset_model(self):
        _log("reset_model requested (trainer not yet wired)")

    def set_prediction_model(self, arch: str):
        self.prediction_model = arch
        _log(f"prediction model -> {arch} (broadcasting to all clients)")
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(self._broadcast_prediction(arch), self._loop)

    async def _broadcast_prediction(self, arch: str):
        msg = create_set_prediction_model_message(arch).to_json()
        for u in list(self._users.values()):
            await self._safe_send(u.ws, msg)

    def set_data_dir(self, path: str):
        self.data_dir = Path(path).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "incoming").mkdir(exist_ok=True)
        _log(f"data dir -> {self.data_dir}")
