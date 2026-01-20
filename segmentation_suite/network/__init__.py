#!/usr/bin/env python3
"""
Multi-user collaborative training network module.

Provides peer-to-peer model synchronization for federated learning.

Basic usage:

    # Host starts a session
    from segmentation_suite.network import AggregationServer, SyncClient

    server = AggregationServer()
    connection_string = server.start(port=8765)
    print(f"Share this with others: {connection_string}")

    # Host also needs a client to participate
    host_client = SyncClient()
    host_client.connect_direct("127.0.0.1", 8765, "Host")

    # Clients join
    client = SyncClient()
    client.connect_direct("192.168.1.5", 8765, "Client1")

    # Send weights
    client.send_weights(model.state_dict(), epoch=10, loss=0.05)

    # Receive global model
    client.global_model_received.connect(lambda weights: model.load_state_dict(weights))

Relay mode (for internet, no port forwarding needed):

    # Create a room (host)
    client = SyncClient()
    client.room_created.connect(lambda code: print(f"Share this code: {code}"))
    client.create_relay_room("MyName", relay_url="wss://segmentation-relay.yourname.workers.dev/ws")

    # Join a room (others)
    client = SyncClient()
    client.connect_relay("ABC123", "MyName", relay_url="wss://segmentation-relay.yourname.workers.dev/ws")
"""

from .protocol import (
    MessageType,
    Message,
    serialize_weights,
    deserialize_weights,
)

from .session import (
    MultiUserSession,
    UserInfo,
    SessionManager,
    get_session_manager,
    get_local_ip,
    generate_session_id,
    generate_user_id,
)

from .aggregator import (
    FedAvgAggregator,
    MomentumAggregator,
    blend_weights,
)

from .server import AggregationServer
from .client import SyncClient, HostClient, DEFAULT_RELAY_URL


__all__ = [
    # Protocol
    "MessageType",
    "Message",
    "serialize_weights",
    "deserialize_weights",

    # Session
    "MultiUserSession",
    "UserInfo",
    "SessionManager",
    "get_session_manager",
    "get_local_ip",
    "generate_session_id",
    "generate_user_id",

    # Aggregation
    "FedAvgAggregator",
    "MomentumAggregator",
    "blend_weights",

    # Server/Client
    "AggregationServer",
    "SyncClient",
    "HostClient",
    "DEFAULT_RELAY_URL",
]
