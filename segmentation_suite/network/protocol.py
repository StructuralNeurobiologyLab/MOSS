#!/usr/bin/env python3
"""
Network protocol for multi-user collaborative training.

Defines message types and serialization for WebSocket communication.
"""

import gzip
import io
import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Optional
from datetime import datetime

import torch


class MessageType(Enum):
    """Types of messages exchanged between client and server."""
    # Connection management
    HELLO = "hello"           # Client introduces itself
    WELCOME = "welcome"       # Server acknowledges, sends session info
    GOODBYE = "goodbye"       # Graceful disconnect

    # Weight synchronization
    WEIGHTS_PUSH = "weights_push"    # Client sends weights to server
    WEIGHTS_ACK = "weights_ack"      # Server acknowledges receipt
    GLOBAL_MODEL = "global_model"    # Server broadcasts updated model
    REQUEST_MODEL = "request_model"  # Client requests current global model

    # Chunked transfer (for large weights)
    CHUNK_START = "chunk_start"      # Start of chunked transfer
    CHUNK_END = "chunk_end"          # End of chunked transfer

    # Session info
    USER_LIST = "user_list"          # Server sends list of connected users

    # Errors
    ERROR = "error"                  # Error message

    # Training data transfer (multi-user redesign)
    TRAINING_DATA = "training_data"        # Client -> Host: image+mask crop
    TRAINING_DATA_ACK = "training_data_ack"  # Host -> Client: received confirmation

    # Snapshot model (multi-user redesign)
    SNAPSHOT_MODEL = "snapshot_model"  # Host -> All: model snapshot (every 30s)


# Maximum chunk size for WebSocket messages (16MB to stay under Cloudflare's 32MB limit)
MAX_CHUNK_SIZE = 16 * 1024 * 1024


@dataclass
class Message:
    """Base message structure."""
    type: MessageType
    payload: dict
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp
        })

    @classmethod
    def from_json(cls, data: str) -> "Message":
        """Deserialize message from JSON string."""
        obj = json.loads(data)
        return cls(
            type=MessageType(obj["type"]),
            payload=obj["payload"],
            timestamp=obj.get("timestamp", "")
        )


def serialize_weights(state_dict: dict, use_half_precision: bool = True) -> bytes:
    """
    Compress and serialize PyTorch state dict to bytes.

    Uses float16 conversion + gzip compression for efficient transfer.
    Typically achieves 4-8x size reduction.

    Args:
        state_dict: PyTorch model state dictionary
        use_half_precision: Convert to float16 for smaller size (default True)

    Returns:
        Compressed bytes
    """
    # Check if this is accidentally a full checkpoint instead of just model weights
    if len(state_dict) < 10:
        if 'model_state_dict' in state_dict:
            print(f"[Protocol] WARNING: serialize_weights received full checkpoint, extracting model_state_dict")
            state_dict = state_dict['model_state_dict']
        elif 'model_state' in state_dict:
            print(f"[Protocol] WARNING: serialize_weights received full checkpoint, extracting model_state")
            state_dict = state_dict['model_state']

    print(f"[Protocol] Serializing {len(state_dict)} weight keys")

    # Convert to half precision for smaller transfer size
    if use_half_precision:
        compressed_state = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                compressed_state[key] = value.half()  # float32 -> float16
            else:
                compressed_state[key] = value  # Keep non-tensors and non-float32 as-is
    else:
        compressed_state = state_dict

    # Save to bytes buffer
    buffer = io.BytesIO()
    torch.save(compressed_state, buffer)
    buffer.seek(0)

    # Compress with gzip (level 1 for speed - still gets good compression on float16 data)
    compressed = gzip.compress(buffer.read(), compresslevel=1)

    tensor_count = sum(1 for v in compressed_state.values() if isinstance(v, torch.Tensor))
    print(f"[Protocol] Serialized weights: {len(compressed) / 1024 / 1024:.1f}MB (half={use_half_precision}, {tensor_count} tensors)")
    return compressed


def deserialize_weights(data: bytes) -> dict:
    """
    Decompress and deserialize bytes to PyTorch state dict.

    Converts float16 back to float32 for training compatibility.

    Args:
        data: Compressed bytes from serialize_weights

    Returns:
        PyTorch model state dictionary (float32)
    """
    # Decompress
    decompressed = gzip.decompress(data)

    # Load state dict
    buffer = io.BytesIO(decompressed)
    state_dict = torch.load(buffer, map_location='cpu', weights_only=False)

    # Debug: show what was loaded
    print(f"[Protocol] Loaded state_dict with {len(state_dict)} keys")
    if len(state_dict) < 10:
        print(f"[Protocol] Keys are: {list(state_dict.keys())}")
        # Check if this is accidentally a full checkpoint instead of just model weights
        if 'model_state_dict' in state_dict:
            print(f"[Protocol] WARNING: This looks like a full checkpoint, extracting model_state_dict")
            state_dict = state_dict['model_state_dict']
            print(f"[Protocol] Extracted model_state_dict with {len(state_dict)} keys")
        elif 'model_state' in state_dict:
            print(f"[Protocol] WARNING: This looks like a full checkpoint, extracting model_state")
            state_dict = state_dict['model_state']
            print(f"[Protocol] Extracted model_state with {len(state_dict)} keys")

    # Convert back to float32 for training
    restored_state = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.float16:
            restored_state[key] = value.float()  # float16 -> float32
        else:
            restored_state[key] = value  # Keep non-tensors and non-float16 as-is

    tensor_count = sum(1 for v in restored_state.values() if isinstance(v, torch.Tensor))
    print(f"[Protocol] Deserialized weights: {tensor_count} tensors, {len(restored_state)} total keys")
    return restored_state


def create_hello_message(user_id: str, display_name: str) -> Message:
    """Create a HELLO message for client introduction."""
    return Message(
        type=MessageType.HELLO,
        payload={
            "user_id": user_id,
            "display_name": display_name
        }
    )


def create_welcome_message(session_id: str, user_list: list,
                           architecture: str = None) -> Message:
    """Create a WELCOME message with session info."""
    payload = {
        "session_id": session_id,
        "user_list": user_list
    }
    if architecture:
        payload["architecture"] = architecture
    return Message(
        type=MessageType.WELCOME,
        payload=payload
    )


def create_weights_push_message(user_id: str, epoch: int, loss: float,
                                 num_samples: int) -> Message:
    """
    Create a WEIGHTS_PUSH message header.

    Note: The actual weights are sent as binary data following the JSON message.
    """
    return Message(
        type=MessageType.WEIGHTS_PUSH,
        payload={
            "user_id": user_id,
            "epoch": epoch,
            "loss": loss,
            "num_samples": num_samples
        }
    )


def create_weights_ack_message(user_id: str, received: bool) -> Message:
    """Create a WEIGHTS_ACK message."""
    return Message(
        type=MessageType.WEIGHTS_ACK,
        payload={
            "user_id": user_id,
            "received": received
        }
    )


def create_global_model_message(aggregation_round: int, contributor_count: int) -> Message:
    """
    Create a GLOBAL_MODEL message header.

    Note: The actual weights are sent as binary data following the JSON message.
    """
    return Message(
        type=MessageType.GLOBAL_MODEL,
        payload={
            "aggregation_round": aggregation_round,
            "contributor_count": contributor_count
        }
    )


def create_user_list_message(users: list) -> Message:
    """Create a USER_LIST message."""
    return Message(
        type=MessageType.USER_LIST,
        payload={
            "users": users
        }
    )


def create_goodbye_message(user_id: str, reason: str = "disconnect") -> Message:
    """Create a GOODBYE message."""
    return Message(
        type=MessageType.GOODBYE,
        payload={
            "user_id": user_id,
            "reason": reason
        }
    )


def create_error_message(error: str, details: Optional[str] = None) -> Message:
    """Create an ERROR message."""
    return Message(
        type=MessageType.ERROR,
        payload={
            "error": error,
            "details": details or ""
        }
    )


def create_chunk_start_message(transfer_id: str, total_chunks: int, total_size: int,
                                original_type: str) -> Message:
    """Create a CHUNK_START message to begin a chunked transfer."""
    return Message(
        type=MessageType.CHUNK_START,
        payload={
            "transfer_id": transfer_id,
            "total_chunks": total_chunks,
            "total_size": total_size,
            "original_type": original_type  # "weights_push" or "global_model"
        }
    )


def create_chunk_end_message(transfer_id: str) -> Message:
    """Create a CHUNK_END message to signal transfer complete."""
    return Message(
        type=MessageType.CHUNK_END,
        payload={
            "transfer_id": transfer_id
        }
    )


def chunk_data(data: bytes, chunk_size: int = MAX_CHUNK_SIZE) -> list:
    """Split data into chunks of specified size."""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def needs_chunking(data: bytes) -> bool:
    """Check if data needs to be chunked."""
    return len(data) > MAX_CHUNK_SIZE


def create_training_data_message(user_id: str, display_name: str,
                                  crop_size: int, slice_index: int,
                                  chunk_index: int = 0, total_chunks: int = 1) -> Message:
    """
    Create a TRAINING_DATA message header.

    Note: The actual image and mask data are sent as binary following the JSON message.
    """
    import time
    return Message(
        type=MessageType.TRAINING_DATA,
        payload={
            "user_id": user_id,
            "display_name": display_name,
            "crop_size": crop_size,
            "slice_index": slice_index,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "timestamp": int(time.time() * 1000)
        }
    )


def create_training_data_ack_message(user_id: str, received: bool,
                                      message: str = "") -> Message:
    """Create a TRAINING_DATA_ACK message."""
    return Message(
        type=MessageType.TRAINING_DATA_ACK,
        payload={
            "user_id": user_id,
            "received": received,
            "message": message
        }
    )


def create_snapshot_model_message(snapshot_id: int, user_id: str) -> Message:
    """
    Create a SNAPSHOT_MODEL message header.

    Note: The actual weights are sent as binary data following the JSON message.
    """
    return Message(
        type=MessageType.SNAPSHOT_MODEL,
        payload={
            "snapshot_id": snapshot_id,
            "user_id": user_id
        }
    )


def serialize_training_data(image_array, mask_array) -> tuple:
    """
    Serialize image and mask arrays to compressed bytes.

    Args:
        image_array: numpy array of the image crop (uint8)
        mask_array: numpy array of the mask crop (uint8)

    Returns:
        Tuple of (image_bytes, mask_bytes)
    """
    import io
    from PIL import Image

    # Convert to PIL and save as PNG (lossless compression)
    img_buffer = io.BytesIO()
    Image.fromarray(image_array).save(img_buffer, format='PNG', compress_level=6)
    img_bytes = img_buffer.getvalue()

    mask_buffer = io.BytesIO()
    Image.fromarray(mask_array).save(mask_buffer, format='PNG', compress_level=6)
    mask_bytes = mask_buffer.getvalue()

    return img_bytes, mask_bytes


def deserialize_training_data(image_bytes: bytes, mask_bytes: bytes) -> tuple:
    """
    Deserialize image and mask bytes to numpy arrays.

    Args:
        image_bytes: PNG encoded image data
        mask_bytes: PNG encoded mask data

    Returns:
        Tuple of (image_array, mask_array) as numpy arrays
    """
    import io
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    mask = Image.open(io.BytesIO(mask_bytes))

    return np.array(img), np.array(mask)
