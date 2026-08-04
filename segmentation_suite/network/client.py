#!/usr/bin/env python3
"""
WebSocket client for multi-user collaborative training.

Connects to a host (direct) or relay server to sync model weights.
"""

import asyncio
import json
import socket
import threading
from typing import Optional, Callable
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketClientProtocol = None

from .protocol import (
    Message, MessageType, serialize_weights, deserialize_weights,
    create_hello_message, create_weights_push_message, create_goodbye_message,
    create_chunk_start_message, create_chunk_end_message,
    create_training_data_message, create_project_register_message,
    create_raw_register_message,
    serialize_training_data,
    chunk_data, needs_chunking, MAX_CHUNK_SIZE
)
from .session import UserInfo, generate_user_id


# Load relay server URL from config file
def _load_relay_url():
    """Load relay URL from relay_config.txt file."""
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'relay_config.txt')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        return line
        except Exception:
            pass
    return None

DEFAULT_RELAY_URL = _load_relay_url()


def _log(msg: str):
    """Print debug log with timestamp."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Client] {msg}")


class SyncClient(QObject):
    """
    WebSocket client for syncing model weights.

    Can connect to either:
    - Direct host: ws://192.168.1.5:8765
    - Relay server: wss://relay.example.com/room/ABC123

    Signals:
        connected(): Emitted when connection is established
        disconnected(): Emitted when disconnected
        global_model_received(dict): Emitted when global model is received
        user_list_updated(list): Emitted when user list changes
        sync_status(str): Status messages
        error(str): Error messages
    """

    # Signals
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    global_model_received = pyqtSignal(dict)  # weights
    user_list_updated = pyqtSignal(list)  # list of user dicts
    sync_status = pyqtSignal(str)
    error = pyqtSignal(str)
    architecture_received = pyqtSignal(str)  # architecture name from host
    room_created = pyqtSignal(str)  # room_code - emitted when room is created on relay
    room_joined = pyqtSignal(str)  # room_code - emitted when joined relay room
    model_requested = pyqtSignal(str)  # user_id - emitted when another user requests model (relay mode)
    user_joined_room = pyqtSignal(str)  # display_name - emitted when a user joins (relay mode)

    # Ceiling on queued crop-variant sends. Counted in variants: with 2D + 2.5D + dwarf
    # + slab that is 4 per capture, so 32 is roughly 8 captures in flight. Sized so the
    # worst case (all slab, ~14 MB of array + bytes each) stays well under a GB.
    MAX_PENDING_CROP_SENDS = 32

    # New signals for multi-user redesign
    training_data_received = pyqtSignal(bytes, bytes, dict)  # image_bytes, mask_bytes, metadata (host only)

    # Hub redesign signals (authoritative standalone hub)
    owner_assigned = pyqtSignal(bool)              # True if this client is the session owner
    session_subproject_received = pyqtSignal(str)  # local subproject to adopt for this session
    prediction_model_received = pyqtSignal(str)    # authoritative prediction model (locked)
    crop_size_received = pyqtSignal(int)           # authoritative crop/tile size (locked)
    raw_available = pyqtSignal(dict)               # hub is serving a raw volume (descriptor); {} = none
    raw_upload_requested = pyqtSignal(str)         # hub can't see owner's data -> upload it (http_url)
    raw_upload_denied = pyqtSignal(dict)           # hub refused the upload (need_bytes/free_bytes)

    def __init__(self, parent=None):
        super().__init__(parent)

        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library not installed. Run: pip install websockets")

        # Connection state
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

        # User info
        self.user_id = generate_user_id()
        self.display_name = "User"
        self.session_id: Optional[str] = None
        self.session_configured = False  # True if the hub session is already set up (resume/post-register)

        # Connection info
        self._host_ip: Optional[str] = None
        self._port: Optional[int] = None
        self._relay_url: Optional[str] = None
        self._room_code: Optional[str] = None
        self._create_room_on_connect: bool = False  # True if creating, False if joining

        # Pending weights to receive
        self._expecting_weights = False

        # Chunk reassembly state
        self._chunk_transfer_id: Optional[str] = None
        self._chunk_buffer: list = []
        self._chunk_total: int = 0
        self._chunk_original_type: str = ""

        # Reconnection settings
        self.auto_reconnect = True
        self.reconnect_delay = 5.0  # seconds
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10

        # Training data reception state (multi-user redesign)
        self._expecting_training_data = False
        self._training_data_metadata: Optional[dict] = None
        self._training_data_buffer: list = []  # Holds [image_bytes, mask_bytes]

        # Serializes every websocket send. A logical message can be several frames
        # (a JSON header followed by binary image+mask, or header+weights, or a
        # chunked stream). Multiple sends are scheduled as independent coroutines on
        # one event loop (run_coroutine_threadsafe), so without this lock their
        # frames interleave at await points on the single socket — cross-pairing a
        # header with another message's binary frames. That is exactly how a crop's
        # image and mask got swapped when the 2D / 2.5D / dwarf variants of one
        # capture were sent back-to-back. Hold this across all frames of a message.
        self._send_lock: Optional[asyncio.Lock] = None
        # Crop-send backpressure. One capture queues every architecture variant, so this
        # counts variant sends, not captures.
        self._pending_crop_sends = 0
        self._dropped_crop_sends = 0

    def _sendlock(self) -> asyncio.Lock:
        """Lazily create the send lock on the client's event-loop thread. Safe to
        create lazily: all sends run on that single loop, so the None-check and
        assignment never race (no await between them)."""
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        return self._send_lock

    def connect_direct(self, host_ip: str, port: int, user_name: str) -> bool:
        """
        Connect directly to a host.

        Args:
            host_ip: IP address of the host
            port: Port the host is listening on
            user_name: Display name for this user

        Returns:
            True if connection initiated (not yet established)
        """
        if self._running:
            return False

        self._host_ip = host_ip
        self._port = port
        self._relay_url = None
        self._room_code = None
        self.display_name = user_name

        return self._start_connection()

    def create_relay_room(self, user_name: str,
                          relay_url: Optional[str] = None) -> bool:
        """
        Create a new room on the relay server.

        The room_created signal will be emitted with the room code once created.

        Args:
            user_name: Display name for this user
            relay_url: Relay server URL (uses default if not specified)

        Returns:
            True if connection initiated
        """
        if self._running:
            return False

        relay = relay_url or DEFAULT_RELAY_URL
        if not relay:
            self.error.emit("No relay server configured. Deploy relay to Glitch first.")
            return False

        self._host_ip = None
        self._port = None
        self._relay_url = relay
        self._room_code = None
        self._create_room_on_connect = True
        self.display_name = user_name

        return self._start_connection()

    def connect_relay(self, room_code: str, user_name: str,
                      relay_url: Optional[str] = None) -> bool:
        """
        Connect via relay server using a room code.

        Args:
            room_code: Room code to join
            user_name: Display name for this user
            relay_url: Relay server URL (uses default if not specified)

        Returns:
            True if connection initiated
        """
        if self._running:
            return False

        relay = relay_url or DEFAULT_RELAY_URL
        if not relay:
            self.error.emit("No relay server configured. Deploy relay to Glitch first.")
            return False

        self._host_ip = None
        self._port = None
        self._relay_url = relay
        self._room_code = room_code.upper()
        self._create_room_on_connect = False
        self.display_name = user_name

        return self._start_connection()

    def _start_connection(self) -> bool:
        """Start the connection in a background thread."""
        _log(f"Starting connection thread...")
        self._running = True
        self._reconnect_attempts = 0
        self._thread = threading.Thread(target=self._run_client, daemon=True)
        self._thread.start()
        return True

    def disconnect(self):
        """Disconnect from the server."""
        self._running = False
        self.auto_reconnect = False  # Don't reconnect after manual disconnect

        if self._loop and self._loop.is_running():
            # Send goodbye and close
            asyncio.run_coroutine_threadsafe(
                self._close_connection(),
                self._loop
            )

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._connected = False
        self.disconnected.emit()

    async def _close_connection(self):
        """Close the WebSocket connection gracefully."""
        if self._websocket:
            try:
                goodbye = create_goodbye_message(self.user_id, "disconnect")
                await self._websocket.send(goodbye.to_json())
                await self._websocket.close()
            except Exception:
                pass

    def _run_client(self):
        """Run the async client (called in background thread)."""
        _log("Client thread started")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            _log(f"Client error: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Client error: {e}")
        finally:
            _log("Client thread ending")
            self._loop.close()

    @staticmethod
    def _is_permanent_connection_error(exc) -> bool:
        """True if the error can't be fixed by retrying (e.g. name resolution).

        A bad/mistyped host (the classic "typed the session code into the
        IP field" mistake) raises socket.gaierror — retrying 10x just spams
        the UI, so we fail fast instead. Walk the exception chain since
        websockets may wrap it.
        """
        seen = set()
        cur = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, socket.gaierror):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    async def _connect_loop(self):
        """Connection loop with reconnection support."""
        _log("Starting connect loop")
        while self._running:
            try:
                await self._connect_and_listen()
                # Clean exit from listen loop (e.g., _running set to False)
                break
            except Exception as e:
                _log(f"Connection error: {e}")
                self._connected = False
                self.disconnected.emit()

                permanent = self._is_permanent_connection_error(e)

                # Only show error if this wasn't a deliberate disconnect
                if self._running:
                    self.error.emit(f"Connection failed: {e}")

                if not self._running or not self.auto_reconnect or permanent:
                    if permanent:
                        _log("Permanent connection error (name resolution) — not reconnecting")
                    else:
                        _log(f"Not reconnecting (running={self._running}, auto_reconnect={self.auto_reconnect})")
                    self._running = False  # mark dead so this client can't silently no-op a reuse
                    break

                self._reconnect_attempts += 1
                if self._reconnect_attempts > self._max_reconnect_attempts:
                    _log("Max reconnection attempts reached")
                    self.error.emit("Max reconnection attempts reached")
                    self._running = False
                    break

                _log(f"Reconnecting in {self.reconnect_delay}s (attempt {self._reconnect_attempts})...")
                self.sync_status.emit(f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

    async def _connect_and_listen(self):
        """Connect to server and listen for messages."""
        url = self._get_websocket_url()
        _log(f"Connecting to {url}...")
        self.sync_status.emit(f"Connecting to {url}...")

        try:
            async with websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=120,
                close_timeout=10,
                max_size=500 * 1024 * 1024,  # 500MB for large model weights
            ) as websocket:
                _log("WebSocket connected!")
                self._websocket = websocket
                self._connected = True
                self._reconnect_attempts = 0

                # Different handshake for relay vs direct connection
                if self._relay_url:
                    # Relay mode: send create_room or join_room
                    if self._create_room_on_connect:
                        msg = {
                            "type": "create_room",
                            "payload": {
                                "user_id": self.user_id,
                                "display_name": self.display_name
                            }
                        }
                        _log(f"Creating room on relay...")
                        self.sync_status.emit("Creating room...")
                    else:
                        msg = {
                            "type": "join_room",
                            "payload": {
                                "room_code": self._room_code,
                                "user_id": self.user_id,
                                "display_name": self.display_name
                            }
                        }
                        _log(f"Joining room {self._room_code}...")
                        self.sync_status.emit(f"Joining room {self._room_code}...")
                    await websocket.send(json.dumps(msg))
                else:
                    # Direct mode: send hello
                    hello = create_hello_message(self.user_id, self.display_name)
                    _log(f"Sending hello: {hello.to_json()}")
                    await websocket.send(hello.to_json())
                    self.connected.emit()
                    self.sync_status.emit("Connected!")

                # Listen for messages
                _log("Starting message loop...")
                async for message in websocket:
                    if not self._running:
                        _log("Stopping message loop (not running)")
                        break

                    if isinstance(message, str):
                        _log(f"Received JSON: {message[:100]}...")
                        await self._handle_json_message(message)
                    elif isinstance(message, bytes):
                        _log(f"Received binary: {len(message)} bytes")
                        await self._handle_binary_message(message)

                _log("Message loop ended")
        except websockets.exceptions.ConnectionClosedOK:
            _log("Connection closed normally")
        except websockets.exceptions.ConnectionClosedError as e:
            # Normal disconnect: we sent 1000 but server didn't reply with close frame
            if (e.sent and e.sent.code == 1000) or (e.rcvd and e.rcvd.code == 1000):
                _log("Connection closed (clean disconnect)")
            else:
                _log(f"Connection closed with error: {e}")
                raise
        except Exception as e:
            _log(f"Connection error in _connect_and_listen: {e}")
            raise

    def _get_websocket_url(self) -> str:
        """Get the WebSocket URL to connect to."""
        if self._relay_url:
            # Relay mode: just connect to relay URL, room is handled via messages
            return self._relay_url
        else:
            # Direct mode: ws://192.168.1.5:8765
            return f"ws://{self._host_ip}:{self._port}"

    async def _handle_json_message(self, data: str):
        """Handle a JSON message from the server."""
        try:
            # Try to parse as raw JSON first (for relay messages)
            raw = json.loads(data)
            msg_type = raw.get("type", "")
            payload = raw.get("payload", {})

            # Handle relay-specific messages
            if msg_type == "room_created":
                self._room_code = payload.get("room_code", "")
                # After room is created, switch to join mode for reconnects
                self._create_room_on_connect = False
                _log(f"Room created: {self._room_code}")
                self.room_created.emit(self._room_code)
                self.connected.emit()
                self.sync_status.emit(f"Room created: {self._room_code}")
                return

            elif msg_type == "room_joined":
                self._room_code = payload.get("room_code", "")
                users = payload.get("users", [])
                _log(f"Joined room: {self._room_code}")
                self.room_joined.emit(self._room_code)
                self.user_list_updated.emit(users)
                self.connected.emit()
                self.sync_status.emit(f"Joined room: {self._room_code}")
                return

            elif msg_type == "user_joined":
                users = payload.get("users", [])
                display_name = payload.get("display_name", "Someone")
                self.user_list_updated.emit(users)
                self.sync_status.emit(f"{display_name} joined")
                # Emit signal so training page can share weights with new user
                self.user_joined_room.emit(display_name)
                return

            elif msg_type == "user_left":
                users = payload.get("users", [])
                display_name = payload.get("display_name", "Someone")
                self.user_list_updated.emit(users)
                self.sync_status.emit(f"{display_name} left")
                return

            elif msg_type == "request_model":
                # Another user is requesting the global model (relay mode)
                requester = payload.get("user_id", "unknown")
                if requester != self.user_id:  # Don't emit for our own requests
                    _log(f"User {requester} is requesting the model")
                    self.model_requested.emit(requester)
                return

            elif msg_type == "session_info":
                # Host sent session info (e.g., architecture)
                architecture = payload.get("architecture", "")
                if architecture:
                    _log(f"Received session architecture: {architecture}")
                    self.architecture_received.emit(architecture)
                return

            elif msg_type == "error":
                error_msg = payload.get("error", "Unknown error")
                self.error.emit(error_msg)
                return

            # Handle training data messages (multi-user redesign)
            elif msg_type == "training_data":
                self._expecting_training_data = True
                self._training_data_metadata = payload
                self._training_data_buffer = []
                sender = payload.get("display_name", "Unknown")
                crop_size = payload.get("crop_size", 0)
                _log(f"Receiving training data from {sender} ({crop_size}x{crop_size})")
                self.sync_status.emit(f"Receiving crop from {sender}...")
                return

            # Handle relay-forwarded weight messages
            elif msg_type == "weights_push":
                self._expecting_weights = True
                sender = payload.get("display_name", "Unknown")
                epoch = payload.get("epoch", 0)
                self.sync_status.emit(f"Receiving weights from {sender} (epoch {epoch})")
                return

            elif msg_type == "global_model":
                self._expecting_weights = True
                round_num = payload.get("aggregation_round", 0)
                contributors = payload.get("contributor_count", 0)
                self.sync_status.emit(f"Receiving global model (round {round_num})")
                return

            # Handle chunked transfers
            elif msg_type == "chunk_start":
                # Ignore new chunk_start if we're already receiving a transfer
                # This prevents interleaving issues when multiple transfers overlap
                if self._chunk_transfer_id:
                    new_id = payload.get("transfer_id", "")
                    _log(f"Ignoring chunk_start {new_id} - already receiving transfer {self._chunk_transfer_id}")
                    return

                self._chunk_transfer_id = payload.get("transfer_id", "")
                self._chunk_total = payload.get("total_chunks", 0)
                self._chunk_original_type = payload.get("original_type", "")
                self._chunk_buffer = []
                total_size = payload.get("total_size", 0)
                sender = payload.get("display_name", "Unknown")
                epoch = payload.get("epoch", 0)
                _log(f"Starting chunked receive: {self._chunk_total} chunks, {total_size} bytes from {sender}")
                self.sync_status.emit(f"Receiving from {sender} (epoch {epoch}, {self._chunk_total} chunks)...")
                return

            elif msg_type == "chunk_end":
                transfer_id = payload.get("transfer_id", "")
                if transfer_id != self._chunk_transfer_id:
                    _log(f"Ignoring chunk_end for {transfer_id} - expecting {self._chunk_transfer_id}")
                    return

                if self._chunk_buffer:
                    _log(f"Chunk transfer complete, reassembling {len(self._chunk_buffer)} chunks")
                    # Reassemble chunks
                    complete_data = b''.join(self._chunk_buffer)
                    _log(f"Reassembled {len(complete_data)} bytes")

                    # Process as weights
                    try:
                        weights = deserialize_weights(complete_data)
                        self.global_model_received.emit(weights)
                        self.sync_status.emit("Weights received (chunked)")
                    except Exception as e:
                        _log(f"Error deserializing chunked weights: {e}")
                        self.error.emit(f"Failed to deserialize weights: {e}")

                # Clear chunk state
                self._chunk_transfer_id = None
                self._chunk_buffer = []
                self._chunk_total = 0
                return

            # Try to parse as protocol Message for direct connection
            msg = Message.from_json(data)

            if msg.type == MessageType.WELCOME:
                self.session_id = msg.payload.get("session_id")
                user_list = msg.payload.get("user_list", [])
                architecture = msg.payload.get("architecture", "")
                self.user_list_updated.emit(user_list)
                if architecture:
                    _log(f"Session architecture: {architecture}")
                    self.architecture_received.emit(architecture)
                # Hub extensions: ownership, session subproject, prediction lock.
                self.session_configured = bool(msg.payload.get("session_configured", False))
                self.owner_assigned.emit(bool(msg.payload.get("is_owner", False)))
                session_subproject = msg.payload.get("session_subproject", "")
                if session_subproject:
                    _log(f"Session subproject: {session_subproject}")
                    self.session_subproject_received.emit(session_subproject)
                prediction_model = msg.payload.get("prediction_model", "")
                if prediction_model:
                    _log(f"Authoritative prediction model: {prediction_model}")
                    self.prediction_model_received.emit(prediction_model)
                crop_size = msg.payload.get("crop_size", 0)
                if crop_size:
                    _log(f"Authoritative crop size: {crop_size}")
                    self.crop_size_received.emit(int(crop_size))
                # Raw-volume availability: the hub is serving a shared volume the
                # joinee can download/stream. {} / absent = nothing shared.
                raw = msg.payload.get("raw") or {}
                if raw.get("ready"):
                    _log(f"Hub raw volume available: {raw.get('num_slices')} slices "
                         f"@ {raw.get('http_url')}")
                self.raw_available.emit(raw)
                self.sync_status.emit(f"Joined session {self.session_id}")

            elif msg.type == MessageType.SET_PREDICTION_MODEL:
                arch = msg.payload.get("architecture", "")
                if arch:
                    _log(f"Prediction model set by hub: {arch}")
                    self.prediction_model_received.emit(arch)

            elif msg.type == MessageType.RAW_UPLOAD_REQUEST:
                url = msg.payload.get("http_url", "")
                _log(f"Hub requests raw upload -> {url}")
                self.raw_upload_requested.emit(url)

            elif msg.type == MessageType.RAW_UPLOAD_DENIED:
                _log(f"Hub denied raw upload: {msg.payload}")
                self.raw_upload_denied.emit(dict(msg.payload or {}))

            elif msg.type == MessageType.USER_LIST:
                user_list = msg.payload.get("users", [])
                self.user_list_updated.emit(user_list)

            elif msg.type == MessageType.GLOBAL_MODEL:
                # Weights will follow as binary message
                self._expecting_weights = True
                round_num = msg.payload.get("aggregation_round", 0)
                contributors = msg.payload.get("contributor_count", 0)
                self.sync_status.emit(f"Receiving global model (round {round_num}, {contributors} contributors)")

            elif msg.type == MessageType.WEIGHTS_ACK:
                received = msg.payload.get("received", False)
                if received:
                    self.sync_status.emit("Weights received by server")
                else:
                    self.error.emit("Server rejected weights")

            elif msg.type == MessageType.ERROR:
                error = msg.payload.get("error", "Unknown error")
                self.error.emit(error)

        except Exception as e:
            print(f"[Client] Error parsing message: {e}")

    async def _handle_binary_message(self, data: bytes):
        """Handle binary weight data from the server."""
        # Check if we're in chunked receive mode
        if self._chunk_transfer_id:
            received = len(self._chunk_buffer)
            # Only accept chunks until we have the expected number
            # This prevents chunks from overlapping transfers from mixing in
            if received >= self._chunk_total:
                _log(f"Ignoring extra binary chunk (already have {received}/{self._chunk_total})")
                return
            self._chunk_buffer.append(data)
            received = len(self._chunk_buffer)
            if received % 2 == 0 or received == self._chunk_total:
                _log(f"Received chunk {received}/{self._chunk_total}")
            return

        # Handle training data (two binary messages: image then mask)
        if self._expecting_training_data:
            self._training_data_buffer.append(data)
            if len(self._training_data_buffer) == 2:
                # Got both image and mask
                img_bytes, mask_bytes = self._training_data_buffer
                metadata = self._training_data_metadata or {}
                self._expecting_training_data = False
                self._training_data_buffer = []
                self._training_data_metadata = None
                _log(f"Training data received: {len(img_bytes)}+{len(mask_bytes)} bytes")
                self.training_data_received.emit(img_bytes, mask_bytes, metadata)
                self.sync_status.emit("Training crop received")
            return

        # Regular (non-chunked) binary message (weights)
        if self._expecting_weights:
            try:
                weights = deserialize_weights(data)
                self._expecting_weights = False
                self.global_model_received.emit(weights)
                self.sync_status.emit("Global model received")
            except Exception as e:
                self.error.emit(f"Failed to deserialize weights: {e}")

    def send_architecture(self, architecture: str):
        """
        Send architecture info to the room (host only, for new joiners).

        Args:
            architecture: Architecture identifier (e.g., 'unet', 'unet_deep_dice')
        """
        if not self._connected or not self._loop:
            return

        asyncio.run_coroutine_threadsafe(
            self._send_architecture_async(architecture),
            self._loop
        )

    async def _send_architecture_async(self, architecture: str):
        """Async implementation of send_architecture."""
        if not self._websocket:
            return

        try:
            msg = Message(
                type=MessageType.SESSION_INFO,
                payload={"architecture": architecture}
            )
            async with self._sendlock():
                await self._websocket.send(msg.to_json())
            _log(f"Sent architecture info: {architecture}")
        except Exception as e:
            _log(f"Failed to send architecture: {e}")

    def send_project_register(self, project_name: str, subproject: str,
                              architecture: str = "", prediction_model: str = "",
                              subprojects: list = None, crop_size: int = 0):
        """Owner -> Hub: register the authoritative project identity for the session."""
        if not self._connected or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_project_register_async(
                project_name, subproject, architecture, prediction_model,
                subprojects or [], crop_size),
            self._loop
        )

    async def _send_project_register_async(self, project_name, subproject,
                                           architecture, prediction_model, subprojects,
                                           crop_size):
        if not self._websocket:
            return
        try:
            msg = create_project_register_message(
                project_name, subproject, architecture, prediction_model, subprojects,
                crop_size)
            async with self._sendlock():
                await self._websocket.send(msg.to_json())
            _log(f"Sent PROJECT_REGISTER: {project_name}/{subproject}")
        except Exception as e:
            _log(f"Failed to send project register: {e}")

    def send_raw_register(self, path: str, fmt: str = "zarr", size_bytes: int = 0,
                          n_files: int = 0):
        """Owner -> Hub: offer to share the session's raw volume (the hub references
        it in place if it can see `path`, else it needs an upload)."""
        if not self._connected or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_raw_register_async(path, fmt, size_bytes, n_files), self._loop)

    async def _send_raw_register_async(self, path, fmt, size_bytes, n_files=0):
        if not self._websocket:
            return
        try:
            msg = create_raw_register_message(path, fmt, size_bytes, n_files)
            async with self._sendlock():
                await self._websocket.send(msg.to_json())
            _log(f"Sent RAW_REGISTER: {path} ({fmt}, {size_bytes} B)")
        except Exception as e:
            _log(f"Failed to send raw register: {e}")

    def send_raw_upload_complete(self):
        """Owner -> Hub: all raw files uploaded; attach + serve."""
        if not self._connected or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._send_raw_upload_complete_async(), self._loop)

    async def _send_raw_upload_complete_async(self):
        if not self._websocket:
            return
        try:
            from .protocol import create_raw_upload_complete_message
            msg = create_raw_upload_complete_message()
            async with self._sendlock():
                await self._websocket.send(msg.to_json())
            _log("Sent RAW_UPLOAD_COMPLETE")
        except Exception as e:
            _log(f"Failed to send raw upload complete: {e}")

    def send_weights(self, weights: dict, epoch: int, loss: float,
                    num_samples: int = 1):
        """
        Send model weights to the server.

        Args:
            weights: PyTorch model state dict
            epoch: Current training epoch
            loss: Current training loss
            num_samples: Number of training samples (for weighting)
        """
        if not self._connected or not self._loop:
            self.error.emit("Not connected")
            return

        asyncio.run_coroutine_threadsafe(
            self._send_weights_async(weights, epoch, loss, num_samples),
            self._loop
        )

    async def _send_weights_async(self, weights: dict, epoch: int, loss: float,
                                  num_samples: int):
        """Async implementation of send_weights."""
        if not self._websocket:
            return

        try:
            # Serialize weights
            weights_data = serialize_weights(weights)
            _log(f"Serialized weights: {len(weights_data)} bytes")

            # Check if chunking is needed
            if needs_chunking(weights_data):
                await self._send_chunked_weights(weights_data, epoch, loss, num_samples)
            else:
                # Send header
                header = create_weights_push_message(
                    self.user_id, epoch, loss, num_samples
                )
                async with self._sendlock():
                    await self._websocket.send(header.to_json())
                    # Send weights as binary
                    await self._websocket.send(weights_data)

                self.sync_status.emit(f"Sent weights (epoch {epoch})")

        except Exception as e:
            self.error.emit(f"Failed to send weights: {e}")
            import traceback
            traceback.print_exc()

    async def _send_chunked_weights(self, weights_data: bytes, epoch: int, loss: float,
                                     num_samples: int):
        """Send weights in chunks for large models."""
        import uuid

        transfer_id = str(uuid.uuid4())[:8]
        chunks = chunk_data(weights_data)
        total_chunks = len(chunks)

        _log(f"Sending weights in {total_chunks} chunks ({len(weights_data)} bytes total)")
        self.sync_status.emit(f"Sending weights in {total_chunks} chunks...")

        # Send chunk start message with metadata
        start_msg = create_chunk_start_message(
            transfer_id=transfer_id,
            total_chunks=total_chunks,
            total_size=len(weights_data),
            original_type="weights_push"
        )
        # Include weights_push metadata in the start message
        start_msg.payload["epoch"] = epoch
        start_msg.payload["loss"] = loss
        start_msg.payload["num_samples"] = num_samples
        start_msg.payload["user_id"] = self.user_id
        start_msg.payload["display_name"] = self.display_name

        # Hold the send lock across the whole chunked stream (start + all chunks +
        # end) so no other message's frames can interleave into it.
        async with self._sendlock():
            await self._websocket.send(start_msg.to_json())

            # Send each chunk
            for i, chunk in enumerate(chunks):
                await self._websocket.send(chunk)
                if (i + 1) % 2 == 0 or i == total_chunks - 1:
                    _log(f"Sent chunk {i + 1}/{total_chunks}")

            # Send chunk end message
            end_msg = create_chunk_end_message(transfer_id)
            await self._websocket.send(end_msg.to_json())

        self.sync_status.emit(f"Sent weights (epoch {epoch}, {total_chunks} chunks)")

    def request_global_model(self):
        """Request the current global model from the server."""
        if not self._connected or not self._loop:
            return

        asyncio.run_coroutine_threadsafe(
            self._request_global_model_async(),
            self._loop
        )

    async def _request_global_model_async(self):
        """Async implementation of request_global_model."""
        if not self._websocket:
            return

        try:
            from .protocol import Message, MessageType
            msg = Message(
                type=MessageType.REQUEST_MODEL,
                payload={"user_id": self.user_id}
            )
            _log("Requesting global model from server")
            async with self._sendlock():
                await self._websocket.send(msg.to_json())
        except Exception as e:
            _log(f"Error requesting global model: {e}")

    def send_training_data(self, image_array, mask_array, slice_index: int = 0):
        """
        Send training crop data to the host (client only).

        In the new multi-user architecture, clients send training crops
        to the host instead of model weights.

        Args:
            image_array: numpy array of the image crop (uint8)
            mask_array: numpy array of the mask crop (uint8)
            slice_index: Source slice index for metadata
        """
        if not self._connected or not self._loop:
            self.error.emit("Not connected")
            return

        # Bound the backlog. Sends are fire-and-forget onto the event loop, so capturing
        # faster than the link drains used to grow memory without limit -- one capture
        # queues every architecture variant, and a slab variant is ~6 MB of array plus
        # ~8 MB of compressed bytes. Refuse rather than accumulate, and say so: a
        # dropped crop the user knows about is recoverable, a silent 1 GB backlog is not.
        if self._pending_crop_sends >= self.MAX_PENDING_CROP_SENDS:
            self._dropped_crop_sends += 1
            self.sync_status.emit(
                f"Crop send backlog full ({self._pending_crop_sends} queued) — "
                f"{self._dropped_crop_sends} not sent. Pause capturing to let it drain.")
            _log(f"DROPPED crop send: {self._pending_crop_sends} already queued "
                 f"(total dropped {self._dropped_crop_sends})")
            return

        self._pending_crop_sends += 1
        asyncio.run_coroutine_threadsafe(
            self._send_training_data_async(image_array, mask_array, slice_index),
            self._loop
        )

    async def _send_training_data_async(self, image_array, mask_array, slice_index: int):
        """Async implementation of send_training_data."""
        if not self._websocket:
            self._pending_crop_sends = max(0, self._pending_crop_sends - 1)
            return

        try:
            import numpy as np
            image_array = np.asarray(image_array)
            # A 2.5D/dwarf crop is a multi-channel (C,H,W) stack; a plain crop is 2D (H,W).
            if image_array.ndim == 3:
                n_channels = int(image_array.shape[0])
                crop_size = int(image_array.shape[-1])
            else:
                n_channels = 1
                crop_size = int(image_array.shape[0]) if image_array.ndim >= 2 else 256

            # Send header + image + mask as one atomic unit so a concurrent
            # variant send can't interleave and swap image/mask frames.
            #
            # Compress INSIDE the lock. Every queued send used to serialize on arrival
            # and then wait here holding its compressed bytes as well as its source
            # array, so a burst of captures held both copies of every pending variant at
            # once. Deep slab variants make that expensive fast (~8 MB of bytes on top of
            # a 6 MB array each), and only the send actually in flight needs its bytes.
            async with self._sendlock():
                img_bytes, mask_bytes = serialize_training_data(image_array, mask_array)
                header = create_training_data_message(
                    user_id=self.user_id,
                    display_name=self.display_name,
                    crop_size=crop_size,
                    slice_index=slice_index,
                    n_channels=n_channels,
                )
                await self._websocket.send(header.to_json())
                await self._websocket.send(img_bytes)
                await self._websocket.send(mask_bytes)

            total_kb = (len(img_bytes) + len(mask_bytes)) / 1024
            self.sync_status.emit(f"Sent training crop ({total_kb:.1f}KB)")
            _log(f"Sent training data: {n_channels}ch {crop_size}x{crop_size}, {total_kb:.1f}KB")

        except Exception as e:
            self.error.emit(f"Failed to send training data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Must decrement on every path, or one failed send permanently shrinks the
            # window and enough of them wedge sending shut.
            self._pending_crop_sends = max(0, self._pending_crop_sends - 1)

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    @property
    def connection_info(self) -> str:
        """Get connection info string."""
        if self._relay_url:
            return f"Room: {self._room_code}"
        elif self._host_ip:
            return f"{self._host_ip}:{self._port}"
        return "Not connected"

    @property
    def room_code(self) -> Optional[str]:
        """Get the current room code (relay mode only)."""
        return self._room_code

    @property
    def is_relay_mode(self) -> bool:
        """Check if connected via relay."""
        return self._relay_url is not None

    @property
    def is_receiving_chunks(self) -> bool:
        """Check if currently receiving a chunked transfer."""
        return self._chunk_transfer_id is not None and self._chunk_transfer_id != ""


class HostClient(SyncClient):
    """
    Special client for the host that connects to their own server.

    The host needs a client to participate in the federated learning,
    since they also contribute weights and receive the global model.
    """

    def connect_to_local_server(self, port: int, user_name: str) -> bool:
        """
        Connect to the local server running on this machine.

        Args:
            port: Port the server is listening on
            user_name: Display name for the host

        Returns:
            True if connection initiated
        """
        return self.connect_direct("127.0.0.1", port, user_name)


def create_relay_room(relay_url: Optional[str] = None) -> Optional[str]:
    """
    Create a new room on the relay server.

    Args:
        relay_url: Relay server URL (uses default if not specified)

    Returns:
        Room code, or None if failed
    """
    import requests

    relay = relay_url or DEFAULT_RELAY_URL
    if not relay:
        return None

    try:
        # Convert wss:// to https:// for REST API
        api_url = relay.replace("wss://", "https://").replace("ws://", "http://")
        response = requests.post(f"{api_url}/create-room", timeout=10)
        if response.ok:
            return response.json().get("room_code")
    except Exception as e:
        print(f"[Client] Failed to create relay room: {e}")

    return None
