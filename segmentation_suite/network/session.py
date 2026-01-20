#!/usr/bin/env python3
"""
Session management for multi-user collaborative training.

Handles session creation, user tracking, and connection state.
"""

import random
import socket
import string
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


def generate_session_id(length: int = 6) -> str:
    """Generate a random session ID (room code)."""
    # Use uppercase letters and digits, excluding ambiguous characters
    chars = string.ascii_uppercase.replace('O', '').replace('I', '')
    chars += string.digits.replace('0', '').replace('1', '')
    return ''.join(random.choices(chars, k=length))


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))


def get_local_ip() -> str:
    """
    Get the local IP address that other machines on the LAN can reach.

    Returns:
        Local IP address (e.g., "192.168.1.5")
    """
    try:
        # Create a socket and connect to an external address
        # We don't actually send any data, just get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        # Use Google's DNS as a target (doesn't actually connect)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost
        return "127.0.0.1"


@dataclass
class UserInfo:
    """Information about a connected user."""
    user_id: str
    display_name: str
    connected_at: datetime = field(default_factory=datetime.now)
    last_sync: Optional[datetime] = None
    contribution_count: int = 0
    last_epoch: int = 0
    last_loss: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "connected_at": self.connected_at.isoformat(),
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "contribution_count": self.contribution_count,
            "last_epoch": self.last_epoch,
            "last_loss": self.last_loss
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserInfo":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            display_name=data["display_name"],
            connected_at=datetime.fromisoformat(data["connected_at"]) if data.get("connected_at") else datetime.now(),
            last_sync=datetime.fromisoformat(data["last_sync"]) if data.get("last_sync") else None,
            contribution_count=data.get("contribution_count", 0),
            last_epoch=data.get("last_epoch", 0),
            last_loss=data.get("last_loss", 0.0)
        )


@dataclass
class MultiUserSession:
    """
    Manages a multi-user training session.

    A session represents a collaborative training room where multiple users
    can connect and share model weights.
    """
    session_id: str = field(default_factory=generate_session_id)
    host_ip: str = field(default_factory=get_local_ip)
    port: int = 8765
    is_host: bool = False
    connected_users: List[UserInfo] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    aggregation_round: int = 0
    architecture: str = ""  # Model architecture for this session (e.g., "unet_deep_dice_25d")

    @property
    def connection_string(self) -> str:
        """Get the connection string for clients to join."""
        return f"{self.host_ip}:{self.port}"

    @property
    def websocket_url(self) -> str:
        """Get the WebSocket URL for clients to connect."""
        return f"ws://{self.host_ip}:{self.port}"

    @property
    def user_count(self) -> int:
        """Get the number of connected users."""
        return len(self.connected_users)

    def add_user(self, user: UserInfo) -> None:
        """Add a user to the session."""
        # Remove any existing user with same ID
        self.connected_users = [u for u in self.connected_users if u.user_id != user.user_id]
        self.connected_users.append(user)

    def remove_user(self, user_id: str) -> Optional[UserInfo]:
        """Remove a user from the session by ID."""
        for i, user in enumerate(self.connected_users):
            if user.user_id == user_id:
                return self.connected_users.pop(i)
        return None

    def get_user(self, user_id: str) -> Optional[UserInfo]:
        """Get a user by ID."""
        for user in self.connected_users:
            if user.user_id == user_id:
                return user
        return None

    def update_user_sync(self, user_id: str, epoch: int, loss: float) -> None:
        """Update a user's sync information."""
        user = self.get_user(user_id)
        if user:
            user.last_sync = datetime.now()
            user.contribution_count += 1
            user.last_epoch = epoch
            user.last_loss = loss

    def get_user_list(self) -> List[dict]:
        """Get list of users as dictionaries."""
        return [user.to_dict() for user in self.connected_users]

    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "host_ip": self.host_ip,
            "port": self.port,
            "is_host": self.is_host,
            "connected_users": self.get_user_list(),
            "created_at": self.created_at.isoformat(),
            "aggregation_round": self.aggregation_round
        }


class SessionManager:
    """
    Manages session state and provides utilities for session discovery.
    """

    def __init__(self):
        self.current_session: Optional[MultiUserSession] = None
        self.user_info: Optional[UserInfo] = None

    def create_session(self, display_name: str, port: int = 8765) -> MultiUserSession:
        """
        Create a new session as host.

        Args:
            display_name: Name to display for this user
            port: Port to listen on

        Returns:
            New MultiUserSession instance
        """
        self.user_info = UserInfo(
            user_id=generate_user_id(),
            display_name=display_name
        )

        self.current_session = MultiUserSession(
            port=port,
            is_host=True
        )
        # Add self as first user
        self.current_session.add_user(self.user_info)

        return self.current_session

    def join_session(self, host_ip: str, port: int, display_name: str) -> MultiUserSession:
        """
        Join an existing session as client.

        Args:
            host_ip: IP address of the host
            port: Port the host is listening on
            display_name: Name to display for this user

        Returns:
            New MultiUserSession instance (as client)
        """
        self.user_info = UserInfo(
            user_id=generate_user_id(),
            display_name=display_name
        )

        self.current_session = MultiUserSession(
            host_ip=host_ip,
            port=port,
            is_host=False
        )
        # Don't add self to user list - server will do that

        return self.current_session

    def leave_session(self) -> None:
        """Leave the current session."""
        self.current_session = None
        self.user_info = None

    @property
    def is_connected(self) -> bool:
        """Check if currently in a session."""
        return self.current_session is not None

    @property
    def is_host(self) -> bool:
        """Check if this user is the host."""
        return self.current_session is not None and self.current_session.is_host


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
