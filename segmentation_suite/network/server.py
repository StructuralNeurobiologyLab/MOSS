#!/usr/bin/env python3
"""
WebSocket server for multi-user collaborative training.

Runs as the host's aggregation server, accepting client connections
and coordinating model weight synchronization.
"""

import asyncio
import threading
from typing import Dict, Optional, Set
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

from .protocol import (
    Message, MessageType, serialize_weights, deserialize_weights,
    create_welcome_message, create_weights_ack_message,
    create_global_model_message, create_user_list_message,
    create_error_message, create_training_data_ack_message
)
from .session import MultiUserSession, UserInfo, get_local_ip
from .aggregator import FedAvgAggregator


def _log(msg: str):
    """Print debug log with timestamp."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Server] {msg}")


class AggregationServer(QObject):
    """
    WebSocket server that accepts client connections and aggregates models.

    Runs in a separate thread to avoid blocking the Qt event loop.

    Signals:
        user_connected(str, str): Emitted when a user connects (user_id, name)
        user_disconnected(str): Emitted when a user disconnects (user_id)
        weights_received(str): Emitted when weights are received (user_id)
        aggregation_complete(dict): Emitted after aggregation (new global weights)
        error(str): Emitted on error
        server_started(str): Emitted when server starts (connection string)
        server_stopped(): Emitted when server stops
    """

    # Signals
    user_connected = pyqtSignal(str, str)  # user_id, display_name
    user_disconnected = pyqtSignal(str)    # user_id
    weights_received = pyqtSignal(str)     # user_id
    aggregation_complete = pyqtSignal(dict)  # new global weights
    error = pyqtSignal(str)
    server_started = pyqtSignal(str)  # connection string
    server_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library not installed. Run: pip install websockets")

        self.session: Optional[MultiUserSession] = None
        self.aggregator = FedAvgAggregator()

        # WebSocket server state
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Connected clients
        self._clients: Dict[str, WebSocketServerProtocol] = {}  # user_id -> websocket
        self._websocket_to_user: Dict[WebSocketServerProtocol, str] = {}  # websocket -> user_id

        # Pending training data (per-user binary chunk tracking)
        self._pending_training_data = {}  # user_id -> {"json_header": str, "chunks": []}

        # Aggregation settings
        self.min_contributors = 1  # Minimum clients needed to trigger aggregation
        self.aggregation_timeout = 30.0  # Seconds to wait for updates before aggregating

    def start(self, port: int = 8765, architecture: str = "") -> str:
        """
        Start the WebSocket server.

        Args:
            port: Port to listen on
            architecture: Model architecture for this session

        Returns:
            Connection string (IP:port) for clients to connect
        """
        _log(f"Starting server on port {port} with architecture: {architecture}")
        if self._running:
            _log("Server already running")
            return self.session.connection_string if self.session else ""

        # Create session
        host_ip = get_local_ip()
        _log(f"Local IP: {host_ip}")
        self.session = MultiUserSession(
            host_ip=host_ip,
            port=port,
            is_host=True,
            architecture=architecture
        )

        # Start server in background thread
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait a moment for server to start
        import time
        time.sleep(0.5)

        _log(f"Server started: {self.session.connection_string}")
        return self.session.connection_string

    def stop(self):
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        # Signal the event loop to stop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._clients.clear()
        self._websocket_to_user.clear()
        self.session = None

        self.server_stopped.emit()

    def _run_server(self):
        """Run the async server (called in background thread)."""
        _log("Server thread started")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            _log(f"Server error: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Server error: {e}")
        finally:
            _log("Server thread ending")
            self._loop.close()

    async def _serve(self):
        """Async server main loop."""
        _log(f"Starting WebSocket server on 0.0.0.0:{self.session.port}")
        try:
            async with websockets.serve(
                self._handle_client,
                "0.0.0.0",  # Listen on all interfaces
                self.session.port,
                ping_interval=20,
                ping_timeout=60,
                max_size=500 * 1024 * 1024,  # 500MB for large model weights
            ) as server:
                self._server = server
                _log("WebSocket server running!")
                self.server_started.emit(self.session.connection_string)

                # Keep running until stopped
                while self._running:
                    await asyncio.sleep(0.1)

                _log("Server loop ended")

        except OSError as e:
            _log(f"OSError starting server: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Failed to start server: {e}")
        except Exception as e:
            _log(f"Exception in _serve: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Server error: {e}")

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a client connection."""
        _log(f"New client connected from {websocket.remote_address}")
        user_id = None

        try:
            async for message in websocket:
                if isinstance(message, str):
                    # JSON message
                    _log(f"Received JSON from client: {message[:100]}...")
                    await self._handle_json_message(websocket, message)
                elif isinstance(message, bytes):
                    # Binary message (weights)
                    _log(f"Received binary from client: {len(message)} bytes")
                    user_id = self._websocket_to_user.get(websocket)
                    if user_id:
                        await self._handle_binary_message(user_id, message)

        except websockets.exceptions.ConnectionClosed as e:
            _log(f"Client connection closed: {e}")
        except Exception as e:
            _log(f"Error handling client: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up disconnected client
            user_id = self._websocket_to_user.get(websocket)
            _log(f"Cleaning up client: {user_id}")
            if user_id:
                self._clients.pop(user_id, None)
                self._websocket_to_user.pop(websocket, None)
                if self.session:
                    self.session.remove_user(user_id)
                self.user_disconnected.emit(user_id)
                await self._broadcast_user_list()

    async def _handle_json_message(self, websocket: WebSocketServerProtocol, data: str):
        """Handle a JSON message from a client."""
        try:
            msg = Message.from_json(data)

            if msg.type == MessageType.HELLO:
                await self._handle_hello(websocket, msg)
            elif msg.type == MessageType.WEIGHTS_PUSH:
                await self._handle_weights_push_header(websocket, msg)
            elif msg.type == MessageType.GOODBYE:
                await self._handle_goodbye(websocket, msg)
            elif msg.type == MessageType.REQUEST_MODEL:
                await self._handle_request_model(websocket, msg)
            elif msg.type == MessageType.TRAINING_DATA:
                await self._handle_training_data_header(websocket, msg, data)
            else:
                print(f"[Server] Unknown message type: {msg.type}")

        except Exception as e:
            print(f"[Server] Error parsing message: {e}")
            error_msg = create_error_message("Invalid message format", str(e))
            await websocket.send(error_msg.to_json())

    async def _handle_hello(self, websocket: WebSocketServerProtocol, msg: Message):
        """Handle a HELLO message (client introduction)."""
        user_id = msg.payload.get("user_id")
        display_name = msg.payload.get("display_name", "Unknown")
        _log(f"HELLO from {display_name} (id: {user_id})")

        if not user_id:
            _log("Error: Missing user_id in HELLO")
            error_msg = create_error_message("Missing user_id")
            await websocket.send(error_msg.to_json())
            return

        # Register client
        self._clients[user_id] = websocket
        self._websocket_to_user[websocket] = user_id
        _log(f"Registered client {user_id}, total clients: {len(self._clients)}")

        # Add to session
        user_info = UserInfo(user_id=user_id, display_name=display_name)
        if self.session:
            self.session.add_user(user_info)

        # Send welcome message with architecture
        welcome = create_welcome_message(
            session_id=self.session.session_id if self.session else "",
            user_list=self.session.get_user_list() if self.session else [],
            architecture=self.session.architecture if self.session else ""
        )
        _log(f"Sending WELCOME to {user_id} with architecture: {self.session.architecture if self.session else 'none'}")
        await websocket.send(welcome.to_json())

        # Emit signal and broadcast updated user list
        self.user_connected.emit(user_id, display_name)
        await self._broadcast_user_list()

        # Send current global model if available
        if self.aggregator.global_weights is not None:
            _log(f"Sending global model to new client {user_id}")
            await self._send_global_model_to_client(websocket)

    async def _handle_weights_push_header(self, websocket: WebSocketServerProtocol, msg: Message):
        """
        Handle a WEIGHTS_PUSH message header.

        The actual weights follow as a binary message.
        """
        user_id = msg.payload.get("user_id")
        epoch = msg.payload.get("epoch", 0)
        loss = msg.payload.get("loss", 0.0)
        num_samples = msg.payload.get("num_samples", 1)

        if not user_id:
            return

        # Store metadata for when binary data arrives
        if not hasattr(self, '_pending_weights'):
            self._pending_weights = {}

        self._pending_weights[user_id] = {
            "epoch": epoch,
            "loss": loss,
            "num_samples": num_samples
        }

    async def _handle_binary_message(self, user_id: str, data: bytes):
        """Handle binary data from a client (training data PNGs or weight data)."""

        # Check if we're expecting training data chunks from this user
        if user_id in self._pending_training_data:
            await self._handle_training_data_chunk(user_id, data)
            return

        # Otherwise, handle as weight data
        try:
            # Deserialize weights
            weights = deserialize_weights(data)

            # Get metadata
            metadata = getattr(self, '_pending_weights', {}).get(user_id, {})
            epoch = metadata.get("epoch", 0)
            loss = metadata.get("loss", 0.0)
            num_samples = metadata.get("num_samples", 1)

            # Add to aggregator
            self.aggregator.add_update(weights, contribution_weight=num_samples)

            # Update session info
            if self.session:
                self.session.update_user_sync(user_id, epoch, loss)

            # Emit signal
            self.weights_received.emit(user_id)

            # Send acknowledgment
            websocket = self._clients.get(user_id)
            if websocket:
                ack = create_weights_ack_message(user_id, received=True)
                await websocket.send(ack.to_json())

            # Check if we should aggregate
            if self.aggregator.update_count >= self.min_contributors:
                await self._aggregate_and_broadcast()

        except Exception as e:
            print(f"[Server] Error handling weights from {user_id}: {e}")

    async def _handle_training_data_chunk(self, user_id: str, data: bytes):
        """
        Handle a training data binary chunk (image, mask, or 2.5D stack PNG).

        Accumulates chunks until expected count is reached (2 for 2D, 3 if 2.5D),
        then forwards the complete training data to all other clients.
        """
        pending = self._pending_training_data[user_id]
        pending["chunks"].append(data)

        expected = pending.get("expected_chunks", 2)
        if len(pending["chunks"]) < expected:
            return

        # Got all chunks — forward to all other clients
        json_header = pending["json_header"]
        chunks = pending["chunks"]

        # Clean up pending state
        del self._pending_training_data[user_id]

        sender_ws = self._clients.get(user_id)
        display_name = Message.from_json(json_header).payload.get("display_name", "Unknown")
        sizes = "+".join(str(len(c)) for c in chunks)
        _log(f"Forwarding training data from {display_name} to other clients "
             f"({sizes} bytes, {len(chunks)} chunks)")

        # Forward to all clients except the sender
        for uid, ws in list(self._clients.items()):
            if ws == sender_ws:
                continue
            try:
                await ws.send(json_header)
                for chunk in chunks:
                    await ws.send(chunk)
            except Exception as e:
                _log(f"Error forwarding training data to {uid}: {e}")

        # Send ACK to sender
        if sender_ws:
            try:
                ack = create_training_data_ack_message(user_id, received=True,
                                                        message="Forwarded to host")
                await sender_ws.send(ack.to_json())
            except Exception:
                pass

    async def _handle_goodbye(self, websocket: WebSocketServerProtocol, msg: Message):
        """Handle a GOODBYE message (graceful disconnect)."""
        user_id = msg.payload.get("user_id")
        if user_id:
            self._clients.pop(user_id, None)
            self._websocket_to_user.pop(websocket, None)
            if self.session:
                self.session.remove_user(user_id)
            self.user_disconnected.emit(user_id)
            await self._broadcast_user_list()

    async def _handle_request_model(self, websocket: WebSocketServerProtocol, msg: Message):
        """Handle a REQUEST_MODEL message - send global model to requesting client."""
        user_id = msg.payload.get("user_id", "unknown")
        _log(f"REQUEST_MODEL from {user_id}")

        if self.aggregator.global_weights is not None:
            _log(f"Sending global model to {user_id}")
            await self._send_global_model_to_client(websocket)
        else:
            _log(f"No global model available to send to {user_id}")

    async def _handle_training_data_header(self, websocket: WebSocketServerProtocol,
                                            msg: Message, raw_json: str):
        """
        Handle a TRAINING_DATA message header.

        Stores metadata and prepares to receive binary messages (image + mask PNGs,
        and optionally a 2.5D stack PNG). The expected_chunks field in the payload
        determines how many binary messages to buffer (2 or 3).
        """
        user_id = msg.payload.get("user_id")
        display_name = msg.payload.get("display_name", "Unknown")
        crop_size = msg.payload.get("crop_size", 0)
        expected_chunks = msg.payload.get("expected_chunks", 2)
        has_25d = msg.payload.get("has_25d", False)

        if not user_id:
            return

        suffix = " +2.5D" if has_25d else ""
        _log(f"TRAINING_DATA from {display_name} ({crop_size}x{crop_size}{suffix}, {expected_chunks} chunks)")

        self._pending_training_data[user_id] = {
            "json_header": raw_json,
            "expected_chunks": expected_chunks,
            "chunks": []
        }

    async def _broadcast_user_list(self):
        """Broadcast updated user list to all clients."""
        if not self.session:
            return

        msg = create_user_list_message(self.session.get_user_list())
        json_data = msg.to_json()

        for websocket in list(self._clients.values()):
            try:
                await websocket.send(json_data)
            except Exception:
                pass

    async def _aggregate_and_broadcast(self):
        """Aggregate weights and broadcast to all clients."""
        # Perform aggregation
        global_weights = self.aggregator.aggregate(min_updates=1)
        if global_weights is None:
            return

        if self.session:
            self.session.aggregation_round += 1

        # Emit signal with new weights
        self.aggregation_complete.emit(global_weights)

        # Broadcast to all clients
        await self._broadcast_global_model()

    async def _broadcast_global_model(self):
        """Broadcast global model to all connected clients."""
        if self.aggregator.global_weights is None:
            return

        # Create message header
        msg = create_global_model_message(
            aggregation_round=self.session.aggregation_round if self.session else 0,
            contributor_count=len(self._clients)
        )
        json_data = msg.to_json()

        # Serialize weights
        weights_data = serialize_weights(self.aggregator.global_weights)

        # Send to all clients
        for websocket in list(self._clients.values()):
            try:
                await websocket.send(json_data)
                await websocket.send(weights_data)
            except Exception as e:
                print(f"[Server] Error broadcasting to client: {e}")

    async def _send_global_model_to_client(self, websocket: WebSocketServerProtocol):
        """Send current global model to a specific client."""
        if self.aggregator.global_weights is None:
            return

        msg = create_global_model_message(
            aggregation_round=self.session.aggregation_round if self.session else 0,
            contributor_count=len(self._clients)
        )

        try:
            await websocket.send(msg.to_json())
            await websocket.send(serialize_weights(self.aggregator.global_weights))
        except Exception as e:
            print(f"[Server] Error sending global model: {e}")

    def set_global_weights(self, weights: dict):
        """
        Set the global weights (e.g., from host's local model).

        This allows the host to initialize the global model.

        Args:
            weights: PyTorch model state dict
        """
        self.aggregator.set_global_weights(weights)

    def broadcast_global_model(self):
        """Broadcast current global model to all clients (thread-safe)."""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_global_model(),
                self._loop
            )

    def trigger_aggregation(self):
        """Force aggregation even if min_contributors not reached (thread-safe)."""
        if self._loop and self._running and self.aggregator.update_count > 0:
            asyncio.run_coroutine_threadsafe(
                self._aggregate_and_broadcast(),
                self._loop
            )

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running

    @property
    def client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self._clients)
