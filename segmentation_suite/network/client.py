#!/usr/bin/env python3
"""
WebSocket client for multi-user collaborative training.

Connects to a host (direct) or relay server to sync model weights.
"""

import asyncio
import json
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

    async def _connect_loop(self):
        """Connection loop with reconnection support."""
        _log("Starting connect loop")
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                _log(f"Connection error: {e}")
                import traceback
                traceback.print_exc()
                self._connected = False
                self.disconnected.emit()
                self.error.emit(f"Connection failed: {e}")

                if not self._running or not self.auto_reconnect:
                    _log(f"Not reconnecting (running={self._running}, auto_reconnect={self.auto_reconnect})")
                    break

                self._reconnect_attempts += 1
                if self._reconnect_attempts > self._max_reconnect_attempts:
                    _log("Max reconnection attempts reached")
                    self.error.emit("Max reconnection attempts reached")
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
        except Exception as e:
            _log(f"Connection error in _connect_and_listen: {e}")
            import traceback
            traceback.print_exc()
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

            elif msg_type == "error":
                error_msg = payload.get("error", "Unknown error")
                self.error.emit(error_msg)
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
                self.sync_status.emit(f"Joined session {self.session_id}")

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
            self._chunk_buffer.append(data)
            received = len(self._chunk_buffer)
            if received % 2 == 0 or received == self._chunk_total:
                _log(f"Received chunk {received}/{self._chunk_total}")
            return

        # Regular (non-chunked) binary message
        if self._expecting_weights:
            try:
                weights = deserialize_weights(data)
                self._expecting_weights = False
                self.global_model_received.emit(weights)
                self.sync_status.emit("Global model received")
            except Exception as e:
                self.error.emit(f"Failed to deserialize weights: {e}")

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
            await self._websocket.send(msg.to_json())
        except Exception as e:
            _log(f"Error requesting global model: {e}")

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
