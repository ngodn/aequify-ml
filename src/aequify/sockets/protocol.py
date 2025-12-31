"""
Message framing protocol for socket communication.

This module provides length-prefixed message framing to handle
TCP stream boundaries properly.

Optimized for Python 3.14+ free-threaded (GIL-disabled) builds.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

# Message header format: 4-byte unsigned int (big-endian) for message length
HEADER_FORMAT = ">I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16 MB default max


@dataclass(frozen=True, slots=True)
class Message:
    """Immutable message container for thread-safe passing between threads."""

    data: bytes

    def __len__(self) -> int:
        return len(self.data)


class ProtocolError(Exception):
    """Raised when protocol violations occur."""


class MessageTooLargeError(ProtocolError):
    """Raised when message exceeds maximum allowed size."""


def encode_message(data: bytes, max_size: int = MAX_MESSAGE_SIZE) -> bytes:
    """
    Encode a message with length prefix for transmission.

    Args:
        data: Raw bytes to encode
        max_size: Maximum allowed message size

    Returns:
        Length-prefixed message bytes

    Raises:
        MessageTooLargeError: If data exceeds max_size
    """
    if len(data) > max_size:
        raise MessageTooLargeError(
            f"Message size {len(data)} exceeds maximum {max_size}"
        )
    header = struct.pack(HEADER_FORMAT, len(data))
    return header + data


def decode_header(header_bytes: bytes) -> int:
    """
    Decode message length from header bytes.

    Args:
        header_bytes: Exactly HEADER_SIZE bytes

    Returns:
        Message length as integer

    Raises:
        ProtocolError: If header is invalid
    """
    if len(header_bytes) != HEADER_SIZE:
        raise ProtocolError(
            f"Invalid header size: expected {HEADER_SIZE}, got {len(header_bytes)}"
        )
    return struct.unpack(HEADER_FORMAT, header_bytes)[0]


class MessageBuffer:
    """
    Buffer for accumulating and extracting framed messages from a byte stream.

    This class is NOT thread-safe by design. Each connection should have
    its own MessageBuffer instance. For free-threaded Python, this avoids
    lock contention by isolating state per connection.
    """

    __slots__ = ("_buffer", "_max_message_size")

    def __init__(self, max_message_size: int = MAX_MESSAGE_SIZE) -> None:
        self._buffer = bytearray()
        self._max_message_size = max_message_size

    def append(self, data: bytes) -> None:
        """Append received data to the buffer."""
        self._buffer.extend(data)

    def extract_messages(self) -> Generator[Message, None, None]:
        """
        Extract all complete messages from the buffer.

        Yields:
            Complete Message objects

        Raises:
            MessageTooLargeError: If incoming message exceeds max size
        """
        while len(self._buffer) >= HEADER_SIZE:
            # Peek at the header to get message length
            msg_len = decode_header(bytes(self._buffer[:HEADER_SIZE]))

            # Validate message size
            if msg_len > self._max_message_size:
                raise MessageTooLargeError(
                    f"Incoming message size {msg_len} exceeds maximum {self._max_message_size}"
                )

            # Check if we have the complete message
            total_len = HEADER_SIZE + msg_len
            if len(self._buffer) < total_len:
                break

            # Extract the complete message
            data = bytes(self._buffer[HEADER_SIZE:total_len])
            del self._buffer[:total_len]
            yield Message(data=data)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def pending_bytes(self) -> int:
        """Number of bytes currently in the buffer."""
        return len(self._buffer)
