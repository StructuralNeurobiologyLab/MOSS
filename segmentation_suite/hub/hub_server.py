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
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from ..network.protocol import (
    Message, MessageType,
    create_welcome_message, create_user_list_message,
    create_set_prediction_model_message, create_training_data_ack_message,
    serialize_weights, create_global_model_message,
    needs_chunking, chunk_data, create_chunk_start_message, create_chunk_end_message,
)
from ..network.session import generate_session_id, get_local_ip


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [Hub] {msg}")


class _User:
    __slots__ = ("user_id", "display_name", "is_owner", "included", "crop_count",
                 "join_index", "ws")

    def __init__(self, user_id, display_name, is_owner, ws, join_index=0, included=True):
        self.user_id = user_id
        self.display_name = display_name
        self.is_owner = is_owner
        self.included = included
        self.crop_count = 0
        self.join_index = join_index
        self.ws = ws


class HubServer(QObject):
    # Signal surface mirrors MockHubBackend exactly.
    session_started = pyqtSignal(str, str, str)         # code, data_dir, connect_addr
    project_registered = pyqtSignal(str, list)          # project_name, subprojects
    user_connected = pyqtSignal(str, str, bool, int)    # user_id, name, is_owner, join_index
    user_disconnected = pyqtSignal(str)                 # user_id
    user_restored = pyqtSignal(str, str, bool, int, int, bool)  # uid, name, is_owner, join_index, crop_count, included
    crop_received = pyqtSignal(str, bytes, str)         # user_id, png_bytes, caption
    training_status = pyqtSignal(int, float, int)       # round, loss, contributors
    prediction_model_set = pyqtSignal(str)              # owner's authoritative model (arch_id)
    training_loss = pyqtSignal(float, int)              # per-batch loss, global_batch (loss plot)

    def __init__(self, data_dir: str, host: str = "0.0.0.0", port: int = 8765,
                 resume: bool = False, parent=None):
        super().__init__(parent)
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets not installed. pip install websockets")

        self.data_dir = Path(data_dir).expanduser()
        self.host = host
        self.port = port
        self._adv_host = host  # resolved to a routable IP in start()
        self.code = generate_session_id()

        # Resume / persistence
        self._resume = resume       # requested (or auto-detected in start())
        self._resumed = False       # actually loaded a manifest
        self._owner_id = ""
        self._known_users: dict = {}   # uid -> {display_name, join_index, is_owner, included}
        self._next_join_index = 0

        # Authoritative session identity (set when the owner registers).
        self.project_name: str = ""
        self.owner_subproject: str = ""
        self.session_subproject: str = ""
        self.architecture: str = ""
        self.prediction_model: str = ""
        self.crop_size: int = 0   # single hub-wide crop/tile size (0 = unset)
        self.subprojects: list = []

        self._users: Dict[str, _User] = {}        # user_id -> _User
        self._ws_to_uid: Dict[object, str] = {}
        self._pending_td: Dict[str, dict] = {}     # user_id -> in-flight training-data frames
        self._crop_seq = 0
        self._registered = False                    # first PROJECT_REGISTER wins; ignore dupes

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._running = False

        # --- trainer state (also exposed via training_state() for a web/HTTP view) ---
        self._trainer = None                # HubTrainer
        self._round = 0                     # last epoch reported
        self._last_loss = 0.0
        self._contrib_count = 0             # enabled users with >=1 crop
        self._agg_round = 0                 # GLOBAL_MODEL broadcast counter
        self._last_weights = None           # last exported state_dict (for late joiners)
        self._loss_history = deque(maxlen=5000)   # (batch, loss) for the loss plot / HTTP
        # configurable knobs (set from __main__)
        self.train_epochs = 50000
        self.broadcast_interval = 5
        self.train_batch_size = 2
        self.train_lr = 1e-4
        self.force_cpu = False
        # debounce destructive pool rebuilds (rapid include/exclude toggles)
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(600)
        self._rebuild_timer.timeout.connect(self._destructive_rebuild)

    def connect_address(self) -> str:
        """The bare IP:port a LAN client types to join (what the GUI shows)."""
        return f"{self._adv_host}:{self.port}"

    # ============================================================ persistence
    def _manifest_path(self) -> Path:
        return self.data_dir / "session.json"

    def _save_manifest(self):
        import json
        try:
            data = {
                "code": self.code,
                "project_name": self.project_name,
                "owner_subproject": self.owner_subproject,
                "session_subproject": self.session_subproject,
                "architecture": self.architecture,
                "prediction_model": self.prediction_model,
                "crop_size": self.crop_size,
                "subprojects": self.subprojects,
                "owner_id": self._owner_id,
                "registered": self._registered,
                "users": self._known_users,
            }
            self._manifest_path().write_text(json.dumps(data, indent=2))
        except Exception as e:
            _log(f"manifest save failed: {e}")

    def _load_manifest(self) -> bool:
        import json
        p = self._manifest_path()
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            _log(f"manifest load failed: {e}")
            return False
        self.code = data.get("code", self.code)
        self.project_name = data.get("project_name", "")
        self.owner_subproject = data.get("owner_subproject", "")
        self.session_subproject = data.get("session_subproject", "")
        self.architecture = data.get("architecture", "")
        self.prediction_model = data.get("prediction_model", "")
        self.crop_size = int(data.get("crop_size", 0))
        self.subprojects = data.get("subprojects", [])
        self._owner_id = data.get("owner_id", "")
        self._known_users = data.get("users", {}) or {}
        self._registered = bool(data.get("registered", False))
        self._next_join_index = max(
            [int(u.get("join_index", 0)) for u in self._known_users.values()],
            default=-1) + 1
        return True

    def _disk_crop_count(self, uid: str) -> int:
        d = self.data_dir / "incoming" / uid / "train_images"
        return len(list(d.glob("*.png"))) if d.exists() else 0

    # ============================================================ crop pool
    # The trainer trains on the UNION of ENABLED users' crops. We present that
    # union as a single merged directory of symlinks (train_pool/), files named
    # "<uid>__<stem>.png" so cross-user stems can't collide. Additive changes are
    # linked into the live pool (worker re-scans at epoch end); destructive
    # changes do a full atomic-swap rebuild + trainer restart.
    def _pool_root(self):
        return self.data_dir / "train_pool"

    def _pool_images(self):
        return self._pool_root() / "train_images"

    def _pool_masks(self):
        return self._pool_root() / "train_masks"

    def _enabled_uids(self):
        return [uid for uid, info in self._known_users.items() if info.get("included", True)]

    def _link(self, src: Path, dst: Path):
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
        except OSError:
            import shutil
            shutil.copy2(src, dst)   # fallback where symlinks aren't permitted

    def _pool_append(self, uid: str, stem: str):
        """Race-free additive: link ONE crop pair into the live pool."""
        self._pool_images().mkdir(parents=True, exist_ok=True)
        self._pool_masks().mkdir(parents=True, exist_ok=True)
        img = self.data_dir / "incoming" / uid / "train_images" / f"{stem}.png"
        msk = self.data_dir / "incoming" / uid / "train_masks" / f"{stem}.png"
        if img.exists() and msk.exists():
            name = f"{uid}__{stem}.png"
            self._link(img, self._pool_images() / name)
            self._link(msk, self._pool_masks() / name)

    def _pool_rebuild_full(self):
        """Full rebuild into a temp dir + atomic swap; recomputes _contrib_count."""
        import shutil
        tmp = self.data_dir / "train_pool.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "train_images").mkdir(parents=True)
        (tmp / "train_masks").mkdir(parents=True)
        contrib = 0
        for uid in self._enabled_uids():
            imgs = self.data_dir / "incoming" / uid / "train_images"
            msks = self.data_dir / "incoming" / uid / "train_masks"
            if not imgs.exists():
                continue
            n = 0
            for f in imgs.glob("*.png"):
                m = msks / f.name
                if not m.exists():
                    continue
                name = f"{uid}__{f.name}"
                self._link(f, tmp / "train_images" / name)
                self._link(m, tmp / "train_masks" / name)
                n += 1
            if n:
                contrib += 1
        self._contrib_count = contrib
        root = self._pool_root()
        if root.exists():
            shutil.rmtree(root)
        os.replace(tmp, root)

    # ============================================================ trainer wiring
    def _recount_contrib(self):
        """Cheap: count enabled users that have at least one crop on disk."""
        self._contrib_count = sum(
            1 for uid in self._enabled_uids() if self._disk_crop_count(uid) > 0)

    def start_training(self, fresh: bool = False):
        """Operator-controlled start (from the web console / Qt). Training does NOT
        auto-start on crops — the operator decides when to begin."""
        self._pool_rebuild_full()   # fresh pool; also sets _contrib_count
        if not (self.crop_size and (self.prediction_model or self.architecture)):
            _log("cannot start training: session not configured yet")
            return
        if self._contrib_count == 0:
            _log("cannot start training: no enabled users with crops")
            return
        if self._trainer is None:
            from .hub_trainer import HubTrainer
            self._trainer = HubTrainer(self)
        if not self._trainer.is_running():
            _log("start training")
            self._trainer.start(resume=not fresh)
        self.training_status.emit(self._round, self._last_loss, self._contrib_count)

    def stop_training(self):
        _log("stop training")
        if self._trainer:
            self._trainer.stop()
        self.training_status.emit(self._round, self._last_loss, self._contrib_count)

    def _on_loss(self, loss: float, batch: int):
        self._loss_history.append((batch, loss))
        self.training_loss.emit(loss, batch)

    def _on_train_progress(self, epoch: int, total: int, train_loss: float, val_loss: float):
        self._round = epoch
        self._last_loss = float(train_loss)
        self.training_status.emit(epoch, float(train_loss), self._contrib_count)

    def _on_train_finished(self, ok: bool, msg: str):
        _log(f"trainer finished ok={ok} msg={msg}")

    def _on_weights_exported(self, weights: dict, epoch: int, loss: float):
        self.broadcast_global_model(weights)

    def _destructive_rebuild(self):
        self._pool_rebuild_full()
        if self._trainer and self._trainer.is_running():
            self._trainer.restart_resume()   # new worker lists a fresh, consistent pool
        # if not running, leave it stopped (operator-controlled)
        self.training_status.emit(self._round, self._last_loss, self._contrib_count)

    def training_state(self) -> dict:
        """Plain-data snapshot the GUI shows — also the contract for a web/HTTP view."""
        return {
            "round": self._round,
            "loss": self._last_loss,
            "contributors": self._contrib_count,
            "agg_round": self._agg_round,
            "running": bool(self._trainer and self._trainer.is_running()),
            "loss_history": list(self._loss_history),
        }

    # ============================================================ weight broadcast
    def broadcast_global_model(self, weights: dict):
        self._last_weights = weights
        self._agg_round += 1
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_global_model(weights, self._agg_round), self._loop)

    async def _broadcast_global_model(self, weights: dict, agg_round: int):
        header = create_global_model_message(
            aggregation_round=agg_round, contributor_count=self._contrib_count).to_json()
        data = serialize_weights(weights)
        for u in list(self._users.values()):
            await self._send_model_frames(u.ws, header, data)
        _log(f"broadcast global model round={agg_round} to {len(self._users)} clients "
             f"({len(data)/1024/1024:.1f}MB)")

    async def _send_model_frames(self, ws, header: str, data: bytes):
        if needs_chunking(data):
            import uuid
            tid = uuid.uuid4().hex
            chunks = chunk_data(data)
            await self._safe_send(ws, create_chunk_start_message(
                tid, len(chunks), len(data), original_type="global_model").to_json())
            for c in chunks:
                await self._safe_send(ws, c)
            await self._safe_send(ws, create_chunk_end_message(tid).to_json())
        else:
            await self._safe_send(ws, header)
            await self._safe_send(ws, data)

    # ================================================================= startup
    def start(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "incoming").mkdir(exist_ok=True)
        # Resume an existing session if requested (or auto-detected) and a manifest exists.
        if self._resume:
            self._resumed = self._load_manifest()
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
        if self._resumed:
            _log(f"RESUMED session: project={self.project_name} subproject={self.session_subproject} "
                 f"crop_size={self.crop_size} users={len(self._known_users)}")
            # Re-populate the GUI: project header, prediction model, and OFFLINE tiles.
            if self.project_name:
                self.project_registered.emit(self.project_name, self.subprojects)
            if self.prediction_model:
                self.prediction_model_set.emit(self.prediction_model)
            for uid, info in self._known_users.items():
                self.user_restored.emit(
                    uid, info.get("display_name", "User"), bool(info.get("is_owner")),
                    int(info.get("join_index", 0)), self._disk_crop_count(uid),
                    bool(info.get("included", True)))
            # Rebuild the training pool from restored crops (training is
            # operator-controlled, so it does not auto-start on resume).
            self._pool_rebuild_full()

    def stop(self):
        if self._trainer:
            self._trainer.stop()
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
        known = self._known_users.get(uid)
        if known is not None:
            # Returning user (reconnect or resume): keep their role/index/inclusion.
            is_owner = bool(known.get("is_owner"))
            join_index = int(known.get("join_index", 0))
            included = bool(known.get("included", True))
            known["display_name"] = name
        else:
            # New user. Only a FRESH, unregistered session's first user is owner;
            # on a registered/resumed session a newcomer is always a joinee.
            is_owner = (not self._registered) and (len(self._known_users) == 0)
            join_index = self._next_join_index
            self._next_join_index += 1
            included = True
            self._known_users[uid] = {"display_name": name, "join_index": join_index,
                                      "is_owner": is_owner, "included": included}
        if is_owner and not self._owner_id:
            self._owner_id = uid
        user = _User(uid, name, is_owner, websocket, join_index=join_index, included=included)
        self._users[uid] = user
        self._ws_to_uid[websocket] = uid
        (self.data_dir / "incoming" / uid / "train_images").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "incoming" / uid / "train_masks").mkdir(parents=True, exist_ok=True)
        _log(f"HELLO {name} ({uid}) owner={is_owner} idx={join_index}")

        await self._send_welcome(user)
        # Late joiner: hand them the current trained weights so they predict with them.
        if self._last_weights is not None:
            header = create_global_model_message(
                aggregation_round=self._agg_round,
                contributor_count=self._contrib_count).to_json()
            await self._send_model_frames(user.ws, header, serialize_weights(self._last_weights))
        await self._broadcast_user_list()
        self.user_connected.emit(uid, name, is_owner, join_index)
        self._save_manifest()

    async def _send_welcome(self, user: _User):
        # The owner keeps their own authoritative subproject; only joinees adopt
        # the namespaced session subproject (to avoid colliding with their own).
        # Owner keeps their own subproject on a FRESH session (they pick it in the
        # setup dialog). On a RESUMED session the owner is told their subproject too.
        if user.is_owner:
            session_subproject = self.owner_subproject if self._resumed else None
        else:
            session_subproject = self.session_subproject or None
        welcome = create_welcome_message(
            session_id=self.code,
            user_list=self._user_list(),
            architecture=self.architecture or None,
            session_subproject=session_subproject,
            prediction_model=self.prediction_model or None,
            crop_size=self.crop_size or None,   # global — sent to owner too
            session_configured=self._registered,  # owner skips setup popup if already configured
            is_owner=user.is_owner,
        )
        await self._safe_send(user.ws, welcome.to_json())

    async def _on_project_register(self, websocket, msg: Message):
        uid = self._ws_to_uid.get(websocket)
        user = self._users.get(uid)
        if not user or not user.is_owner:
            _log("ignoring PROJECT_REGISTER from non-owner")
            return
        if self._registered:
            return  # identity already set — ignore duplicate registers (loop guard)
        self._registered = True
        self._owner_id = uid
        self.project_name = msg.payload.get("project_name", "")
        self.owner_subproject = msg.payload.get("subproject", "") or "default"
        self.architecture = msg.payload.get("architecture", "") or self.architecture
        self.prediction_model = msg.payload.get("prediction_model", "") or self.prediction_model
        self.crop_size = int(msg.payload.get("crop_size", 0)) or self.crop_size
        self.subprojects = msg.payload.get("subprojects", [])
        # Stable, human-readable, quarantined session subproject name.
        safe_proj = self.project_name.replace("/", "-").replace("\\", "-")
        self.session_subproject = f"{self.owner_subproject}__{safe_proj}"
        _log(f"PROJECT_REGISTER project={self.project_name} subproject={self.owner_subproject} "
             f"-> session_subproject={self.session_subproject} arch={self.architecture} "
             f"crop_size={self.crop_size}")
        self.project_registered.emit(self.project_name, self.subprojects)
        if self.prediction_model:
            self.prediction_model_set.emit(self.prediction_model)
        self._save_manifest()
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
        # Backstop: the hub accepts a single crop size. Reject a stray mismatched
        # crop (e.g. from a client that connected mid-change) before it hits disk.
        incoming_cs = int(payload.get("crop_size", 0))
        if self.crop_size and incoming_cs and incoming_cs != self.crop_size:
            _log(f"SKIP crop from {user.display_name}: {incoming_cs} != locked {self.crop_size}")
            return
        self._crop_seq += 1
        ts = payload.get("timestamp", int(time.time() * 1000))
        stem = f"{ts}_{self._crop_seq}"
        base = self.data_dir / "incoming" / uid
        (base / "train_images" / f"{stem}.png").write_bytes(img_bytes)
        (base / "train_masks" / f"{stem}.png").write_bytes(mask_bytes)
        user.crop_count += 1
        # Feed the training pool only while training is running (additive link,
        # worker re-scans at epoch end). When stopped, just keep the contributor
        # count fresh — the operator starts training explicitly.
        if self._known_users.get(uid, {}).get("included", True):
            if self._trainer and self._trainer.is_running():
                self._pool_append(uid, stem)
                self._trainer.request_reload()
            else:
                self._recount_contrib()
            self.training_status.emit(self._round, self._last_loss, self._contrib_count)
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
        # NB: we intentionally do NOT clear self._registered when the session
        # empties — the config is persisted (session.json) so the session can be
        # resumed. A returning owner keeps their role via _known_users.
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
        if user_id in self._known_users:
            self._known_users[user_id]["included"] = included
        user = self._users.get(user_id)
        if user:
            user.included = included
        _log(f"user {user_id} included={included}")
        self._save_manifest()
        self._rebuild_timer.start()   # debounce rapid toggles -> _destructive_rebuild

    def reset_model(self):
        """Archive the checkpoint and clear loss history. If training was running,
        restart it fresh; if it was stopped, stay stopped (operator restarts)."""
        from datetime import datetime
        from ..models.unet import get_checkpoint_filename
        was_running = bool(self._trainer and self._trainer.is_running())
        if self._trainer:
            self._trainer.stop()
        arch = self.prediction_model or self.architecture or "unet"
        model_dir = self.data_dir / "model"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for fname in (get_checkpoint_filename(arch), "checkpoint_final.pth"):
            p = model_dir / fname
            if p.exists():
                try:
                    p.rename(model_dir / f"{p.stem}_old_{ts}.pth")
                except OSError as e:
                    _log(f"reset archive failed for {p}: {e}")
        self._last_weights = None
        self._round = 0
        self._last_loss = 0.0
        self._loss_history.clear()
        _log("model reset")
        self._pool_rebuild_full()
        self.training_status.emit(0, 0.0, self._contrib_count)
        if was_running:
            self.start_training(fresh=True)

    def set_prediction_model(self, arch: str):
        changed = arch != self.prediction_model
        self.prediction_model = arch
        _log(f"session model -> {arch} (broadcasting to all clients)")
        self._save_manifest()
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(self._broadcast_prediction(arch), self._loop)
        # New architecture => different checkpoint shape; reset + restart fresh.
        if changed and self._trainer and self._trainer.is_running():
            self.reset_model()

    async def _broadcast_prediction(self, arch: str):
        msg = create_set_prediction_model_message(arch).to_json()
        for u in list(self._users.values()):
            await self._safe_send(u.ws, msg)

    def set_data_dir(self, path: str):
        self.data_dir = Path(path).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "incoming").mkdir(exist_ok=True)
        _log(f"data dir -> {self.data_dir}")

    def discard_crop(self, uid: str, name: str):
        """Move a crop pair to incoming/<uid>/discarded/ and rebuild the pool.

        Called (marshaled to the main thread) from the web console's review view.
        """
        import shutil
        if not uid or "/" in uid or "\\" in uid or ".." in name or "/" in name:
            return
        base = self.data_dir / "incoming" / uid
        for sub in ("train_images", "train_masks"):
            src = base / sub / name
            if src.exists():
                dst_dir = base / "discarded" / sub
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst_dir / name))
        _log(f"discarded crop {uid}/{name}")
        self._rebuild_timer.start()   # debounced destructive rebuild

    # ================================================================ web view
    def web_state(self) -> dict:
        """Full snapshot for the web console. Defensive copies — safe to read
        from the HTTP thread while the main thread mutates state."""
        from .animals import animal_for_index, color_for_index, animal_svg
        users = []
        online = set(self._users.keys())
        for uid, info in list(self._known_users.items()):
            idx = int(info.get("join_index", 0))
            users.append({
                "uid": uid,
                "name": info.get("display_name", "User"),
                "is_owner": bool(info.get("is_owner")),
                "online": uid in online,
                "included": bool(info.get("included", True)),
                "crops": self._disk_crop_count(uid),
                "color": color_for_index(idx),
                "animal_svg": animal_svg(animal_for_index(idx), color_for_index(idx)),
            })
        models = []
        try:
            from ..models.unet import get_available_architectures
            from ..models.architectures import (
                get_available_architectures as _reg, is_pretrained_architecture)
            archm = get_available_architectures()
            for aid, nm in _reg(include_hidden=True).items():
                if aid not in archm and is_pretrained_architecture(aid):
                    archm[aid] = nm
            for aid, disp in archm.items():
                short = disp.replace("UNet ", "").replace("(", "").replace(")", "")
                models.append({"id": aid, "name": short})
        except Exception as e:
            _log(f"model list error: {e}")
        return {
            "code": self.code,
            "connect_address": self.connect_address(),
            "data_dir": str(self.data_dir),
            "project_name": self.project_name,
            "session_subproject": self.session_subproject,
            "crop_size": self.crop_size,
            "model": self.prediction_model or self.architecture,
            "models": models,
            "users": users,
            "online_count": len(online),
            "total_count": len(self._known_users),
            "training": self.training_state(),
        }
