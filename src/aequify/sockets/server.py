"""
Thread-safe socket server implementation.

Uses explicit locks and thread-safe patterns for robust concurrency.
"""

from __future__ import annotations

import socket
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from aequify.logging import get_logger

from .protocol import (
    Message,
    MessageBuffer,
    ProtocolError,
    encode_message,
)

if TYPE_CHECKING:
    from collections.abc import Generator

logger = get_logger(__name__)


class ServerState(Enum):
    """Server lifecycle states."""

    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()


@dataclass(slots=True)
class ClientConnection:
    """
    Represents a connected client.

    Each client has its own socket and message buffer.
    Thread-safe for concurrent access patterns.
    """

    socket: socket.socket
    address: tuple[str, int]
    buffer: MessageBuffer = field(default_factory=MessageBuffer)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _closed: bool = field(default=False)

    def send(self, data: bytes) -> bool:
        """
        Send data to the client in a thread-safe manner.

        Args:
            data: Raw bytes to send (will be framed automatically)

        Returns:
            True if sent successfully, False if connection is closed
        """
        with self._lock:
            if self._closed:
                return False
            try:
                message = encode_message(data)
                self.socket.sendall(message)
                return True
            except OSError as e:
                logger.debug(f"Send failed to {self.address}: {e}")
                return False

    def recv(self, buffer_size: int = 4096) -> bytes | None:
        """
        Receive data from the client.

        Args:
            buffer_size: Maximum bytes to receive in one call

        Returns:
            Received bytes, empty bytes if connection closed, None on error
        """
        try:
            return self.socket.recv(buffer_size)
        except OSError as e:
            logger.debug(f"Recv failed from {self.address}: {e}")
            return None

    def close(self) -> None:
        """Close the client connection."""
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # Socket may already be disconnected
                self.socket.close()

    @property
    def is_closed(self) -> bool:
        """Check if the connection is closed."""
        with self._lock:
            return self._closed


class MessageHandler(ABC):
    """
    Abstract base class for message handlers.

    Implement this to define how messages are processed.
    Handlers should be thread-safe as they may be called concurrently.
    """

    @abstractmethod
    def on_message(
        self, client: ClientConnection, message: Message
    ) -> bytes | None:
        """
        Handle a received message.

        Args:
            client: The client that sent the message
            message: The received message

        Returns:
            Response bytes to send back, or None for no response
        """

    def on_connect(self, client: ClientConnection) -> None:
        """Called when a client connects. Override to customize."""

    def on_disconnect(self, client: ClientConnection) -> None:
        """Called when a client disconnects. Override to customize."""

    def on_error(self, client: ClientConnection, error: Exception) -> None:
        """Called when an error occurs. Override to customize."""
        logger.error(f"Error handling client {client.address}: {error}")


class EchoHandler(MessageHandler):
    """Simple echo handler for testing."""

    def on_message(
        self, client: ClientConnection, message: Message
    ) -> bytes | None:
        return message.data


class SocketServer:
    """
    Thread-safe socket server with per-client threading.

    Features:
    - Uses explicit locks for thread safety
    - Each client connection handled in separate thread
    - Thread-safe client tracking
    - Graceful shutdown support

    Example:
        class MyHandler(MessageHandler):
            def on_message(self, client, message):
                return b"Response: " + message.data

        server = SocketServer("0.0.0.0", 8080, MyHandler())
        server.start()  # Blocking
        # Or use server.start_background() for non-blocking
    """

    def __init__(
        self,
        host: str,
        port: int,
        handler: MessageHandler,
        *,
        backlog: int = 128,
        recv_buffer_size: int = 4096,
        reuse_address: bool = True,
        reuse_port: bool = False,
        timeout: float | None = None,
        thread_name_prefix: str | None = None,
    ) -> None:
        """
        Initialize the socket server.

        Args:
            host: Host address to bind to
            port: Port number to bind to
            handler: Message handler instance
            backlog: Maximum queued connections
            recv_buffer_size: Buffer size for receiving data
            reuse_address: Enable SO_REUSEADDR
            reuse_port: Enable SO_REUSEPORT (Linux/BSD)
            timeout: Socket timeout for accept (None for blocking)
            thread_name_prefix: Custom prefix for thread names (default: "server")
        """
        self._host = host
        self._port = port
        self._handler = handler
        self._backlog = backlog
        self._recv_buffer_size = recv_buffer_size
        self._reuse_address = reuse_address
        self._reuse_port = reuse_port
        self._timeout = timeout
        self._thread_name_prefix = thread_name_prefix or "server"

        # Thread-safe state management
        self._state_lock = threading.Lock()
        self._state = ServerState.CREATED

        # Client tracking with lock
        self._clients_lock = threading.Lock()
        self._clients: dict[tuple[str, int], ClientConnection] = {}

        # Server socket and thread
        self._socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

    @property
    def state(self) -> ServerState:
        """Get current server state."""
        with self._state_lock:
            return self._state

    @property
    def address(self) -> tuple[str, int]:
        """Get the bound address."""
        return (self._host, self._port)

    @property
    def client_count(self) -> int:
        """Get the number of connected clients."""
        with self._clients_lock:
            return len(self._clients)

    def _set_state(self, state: ServerState) -> None:
        """Set server state (internal use)."""
        with self._state_lock:
            self._state = state

    def _create_socket(self) -> socket.socket:
        """Create and configure the server socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if self._reuse_address:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if self._reuse_port:
            # SO_REUSEPORT is not available on all platforms
            if hasattr(socket, "SO_REUSEPORT"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        if self._timeout is not None:
            sock.settimeout(self._timeout)

        sock.bind((self._host, self._port))
        sock.listen(self._backlog)

        return sock

    def _add_client(self, client: ClientConnection) -> None:
        """Add a client to the tracking dict."""
        with self._clients_lock:
            self._clients[client.address] = client

    def _remove_client(self, client: ClientConnection) -> None:
        """Remove a client from the tracking dict."""
        with self._clients_lock:
            self._clients.pop(client.address, None)

    def _handle_client(self, client: ClientConnection) -> None:
        """
        Handle a single client connection in its own thread.

        This method runs in a separate thread for each client.
        """
        try:
            self._handler.on_connect(client)

            while not self._shutdown_event.is_set() and not client.is_closed:
                data = client.recv(self._recv_buffer_size)

                if data is None:
                    # Error occurred
                    break

                if not data:
                    # Client disconnected gracefully
                    break

                # Add data to buffer and process complete messages
                client.buffer.append(data)

                try:
                    for message in client.buffer.extract_messages():
                        response = self._handler.on_message(client, message)
                        if response is not None:
                            if not client.send(response):
                                break
                except ProtocolError as e:
                    self._handler.on_error(client, e)
                    break

        except Exception as e:
            self._handler.on_error(client, e)
        finally:
            self._handler.on_disconnect(client)
            client.close()
            self._remove_client(client)

    def _accept_loop(self) -> None:
        """Main accept loop running in its own thread."""
        while not self._shutdown_event.is_set():
            try:
                if self._socket is None:
                    break

                try:
                    conn, addr = self._socket.accept()
                except socket.timeout:
                    continue

                client = ClientConnection(socket=conn, address=addr)
                self._add_client(client)

                # Start client handler thread
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True,
                    name=f"{self._thread_name_prefix}-handler",
                )
                thread.start()

            except OSError as e:
                if not self._shutdown_event.is_set():
                    logger.error(f"Accept error: {e}")
                break

    def start(self) -> None:
        """
        Start the server and block until stopped.

        Raises:
            RuntimeError: If server is not in CREATED state
        """
        with self._state_lock:
            if self._state != ServerState.CREATED:
                raise RuntimeError(f"Cannot start server in state {self._state}")
            self._state = ServerState.STARTING

        try:
            self._socket = self._create_socket()
            self._shutdown_event.clear()
            self._set_state(ServerState.RUNNING)

            logger.info(f"Server listening on {self._host}:{self._port}")
            self._accept_loop()

        finally:
            self._cleanup()

    def start_background(self) -> threading.Thread:
        """
        Start the server in a background thread.

        Returns:
            The thread running the server

        Raises:
            RuntimeError: If server is not in CREATED state
        """
        with self._state_lock:
            if self._state != ServerState.CREATED:
                raise RuntimeError(f"Cannot start server in state {self._state}")
            self._state = ServerState.STARTING

        self._socket = self._create_socket()
        self._shutdown_event.clear()
        self._set_state(ServerState.RUNNING)

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name=f"{self._thread_name_prefix}-accept",
        )
        self._accept_thread.start()

        logger.info(f"Server listening on {self._host}:{self._port} (background)")
        return self._accept_thread

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the server gracefully.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        with self._state_lock:
            if self._state not in (ServerState.RUNNING, ServerState.STARTING):
                return
            self._state = ServerState.STOPPING

        self._shutdown_event.set()

        # Close all client connections
        with self._clients_lock:
            clients = list(self._clients.values())

        for client in clients:
            client.close()

        # Wait for accept thread to finish
        if self._accept_thread is not None and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=timeout)

        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up server resources."""
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

        self._set_state(ServerState.STOPPED)
        logger.info("Server stopped")

    def broadcast(self, data: bytes, exclude: ClientConnection | None = None) -> int:
        """
        Send data to all connected clients.

        Args:
            data: Data to broadcast
            exclude: Optional client to exclude from broadcast

        Returns:
            Number of clients the message was sent to
        """
        with self._clients_lock:
            clients = list(self._clients.values())

        count = 0
        for client in clients:
            if client is not exclude and client.send(data):
                count += 1

        return count

    @contextmanager
    def running(self) -> Generator[SocketServer, None, None]:
        """
        Context manager for running the server.

        Example:
            with server.running():
                # Server is running
                do_something()
            # Server is stopped
        """
        self.start_background()
        try:
            yield self
        finally:
            self.stop()


# Type alias for callback-style handlers
MessageCallback = Callable[[ClientConnection, Message], bytes | None]


class CallbackHandler(MessageHandler):
    """Handler that wraps a callback function."""

    def __init__(
        self,
        callback: MessageCallback,
        on_connect: Callable[[ClientConnection], None] | None = None,
        on_disconnect: Callable[[ClientConnection], None] | None = None,
    ) -> None:
        self._callback = callback
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect

    def on_message(
        self, client: ClientConnection, message: Message
    ) -> bytes | None:
        return self._callback(client, message)

    def on_connect(self, client: ClientConnection) -> None:
        if self._on_connect:
            self._on_connect(client)

    def on_disconnect(self, client: ClientConnection) -> None:
        if self._on_disconnect:
            self._on_disconnect(client)
