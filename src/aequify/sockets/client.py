"""
Thread-safe socket client implementation.

Uses explicit locks and thread-safe patterns for robust concurrency.
"""

from __future__ import annotations

import queue
import socket
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

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


class ClientState(Enum):
    """Client lifecycle states."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()


class ConnectionError(Exception):
    """Raised when connection fails."""


class ClientEventHandler(ABC):
    """
    Abstract base class for client event handlers.

    Implement this for async message handling.
    Handlers should be thread-safe.
    """

    @abstractmethod
    def on_message(self, message: Message) -> None:
        """
        Handle a received message.

        Called from the receiver thread.

        Args:
            message: The received message
        """

    def on_connect(self) -> None:
        """Called when connected to server."""

    def on_disconnect(self, error: Exception | None = None) -> None:
        """Called when disconnected from server."""

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        logger.error(f"Client error: {error}")


class SocketClient:
    """
    Thread-safe socket client with automatic reconnection support.

    Features:
    - Uses explicit locks for thread safety
    - Background thread for receiving messages
    - Thread-safe send operations
    - Supports both sync and async patterns

    Example (synchronous):
        client = SocketClient("localhost", 8080)
        client.connect()
        response = client.send_receive(b"Hello")
        client.disconnect()

    Example (asynchronous with handler):
        class MyHandler(ClientEventHandler):
            def on_message(self, message):
                print(f"Received: {message.data}")

        client = SocketClient("localhost", 8080, handler=MyHandler())
        with client.connected():
            client.send(b"Hello")
            # Messages arrive via handler
    """

    def __init__(
        self,
        host: str,
        port: int,
        handler: ClientEventHandler | None = None,
        *,
        connect_timeout: float = 10.0,
        recv_timeout: float | None = None,
        recv_buffer_size: int = 4096,
        auto_reconnect: bool = False,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: int = 5,
        thread_name_prefix: str | None = None,
    ) -> None:
        """
        Initialize the socket client.

        Args:
            host: Server host address
            port: Server port number
            handler: Optional event handler for async message handling
            connect_timeout: Timeout for connection attempts
            recv_timeout: Timeout for receive operations (None for blocking)
            recv_buffer_size: Buffer size for receiving data
            auto_reconnect: Enable automatic reconnection
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts (0 for infinite)
            thread_name_prefix: Custom prefix for thread names (default: "client")
        """
        self._host = host
        self._port = port
        self._handler = handler
        self._connect_timeout = connect_timeout
        self._recv_timeout = recv_timeout
        self._recv_buffer_size = recv_buffer_size
        self._auto_reconnect = auto_reconnect
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts
        self._thread_name_prefix = thread_name_prefix or "client"

        # Thread-safe state management
        self._state_lock = threading.Lock()
        self._state = ClientState.DISCONNECTED

        # Socket and communication
        self._socket: socket.socket | None = None
        self._socket_lock = threading.Lock()

        # Message buffer (per-connection, not shared)
        self._buffer = MessageBuffer()

        # Background receiver thread
        self._recv_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # For synchronous receive pattern
        self._response_queue: queue.Queue[Message | Exception] = queue.Queue()
        self._waiting_for_response = threading.Event()

    @property
    def state(self) -> ClientState:
        """Get current client state."""
        with self._state_lock:
            return self._state

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        with self._state_lock:
            return self._state == ClientState.CONNECTED

    @property
    def address(self) -> tuple[str, int]:
        """Get server address."""
        return (self._host, self._port)

    def _set_state(self, state: ClientState) -> None:
        """Set client state (internal use)."""
        with self._state_lock:
            self._state = state

    def _create_socket(self) -> socket.socket:
        """Create and configure a client socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        return sock

    def connect(self) -> None:
        """
        Connect to the server.

        Raises:
            ConnectionError: If connection fails
            RuntimeError: If already connected
        """
        with self._state_lock:
            if self._state == ClientState.CONNECTED:
                raise RuntimeError("Already connected")
            if self._state == ClientState.CONNECTING:
                raise RuntimeError("Connection in progress")
            self._state = ClientState.CONNECTING

        try:
            self._socket = self._create_socket()
            logger.debug(f"Attempting socket.connect to {self._host}:{self._port}...")
            self._socket.connect((self._host, self._port))
            logger.debug(f"Socket connected to {self._host}:{self._port}")

            # Set receive timeout after connection
            if self._recv_timeout is not None:
                self._socket.settimeout(self._recv_timeout)
            else:
                self._socket.setblocking(True)

            self._buffer.clear()
            self._shutdown_event.clear()
            self._set_state(ClientState.CONNECTED)

            # Start receiver thread if handler is set
            if self._handler:
                self._recv_thread = threading.Thread(
                    target=self._receive_loop,
                    daemon=True,
                    name=f"{self._thread_name_prefix}-recv",
                )
                self._recv_thread.start()
                self._handler.on_connect()

            logger.info(f"Connected to {self._host}:{self._port}")

        except OSError as e:
            self._cleanup()
            raise ConnectionError(f"Failed to connect to {self._host}:{self._port}: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the server."""
        with self._state_lock:
            if self._state == ClientState.DISCONNECTED:
                return
            if self._state == ClientState.DISCONNECTING:
                return
            self._state = ClientState.DISCONNECTING

        self._shutdown_event.set()

        # Wait for receiver thread
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)

        self._cleanup()

        if self._handler:
            self._handler.on_disconnect()

        logger.info(f"Disconnected from {self._host}:{self._port}")

    def _cleanup(self) -> None:
        """Clean up client resources."""
        with self._socket_lock:
            if self._socket:
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None

        self._buffer.clear()
        self._set_state(ClientState.DISCONNECTED)

    def send(self, data: bytes) -> bool:
        """
        Send data to the server.

        Args:
            data: Raw bytes to send (will be framed automatically)

        Returns:
            True if sent successfully

        Raises:
            RuntimeError: If not connected
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        with self._socket_lock:
            if self._socket is None:
                return False
            try:
                message = encode_message(data)
                self._socket.sendall(message)
                return True
            except OSError as e:
                logger.debug(f"Send failed: {e}")
                return False

    def receive(self, timeout: float | None = None) -> Message | None:
        """
        Receive a single message from the server (blocking).

        This method is for synchronous usage without a handler.
        Do not use if a handler is set.

        Args:
            timeout: Receive timeout (None for blocking)

        Returns:
            Received Message or None if disconnected/timeout

        Raises:
            RuntimeError: If not connected or handler is set
            ProtocolError: If protocol error occurs
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")
        if self._handler:
            raise RuntimeError("Cannot use receive() when handler is set")

        with self._socket_lock:
            sock = self._socket
            if sock is None:
                return None

            original_timeout = sock.gettimeout()
            if timeout is not None:
                sock.settimeout(timeout)

            try:
                while True:
                    # Check for complete messages in buffer first
                    for message in self._buffer.extract_messages():
                        return message

                    # Need more data
                    try:
                        data = sock.recv(self._recv_buffer_size)
                    except socket.timeout:
                        return None

                    if not data:
                        # Server disconnected
                        return None

                    self._buffer.append(data)

            finally:
                if timeout is not None:
                    sock.settimeout(original_timeout)

    def send_receive(
        self,
        data: bytes,
        timeout: float | None = None,
    ) -> Message | None:
        """
        Send data and wait for a response (request-response pattern).

        This method is for synchronous usage without a handler.

        Args:
            data: Data to send
            timeout: Response timeout

        Returns:
            Response Message or None if timeout/error

        Raises:
            RuntimeError: If not connected
        """
        if not self.send(data):
            return None
        return self.receive(timeout=timeout)

    def _receive_loop(self) -> None:
        """Background receive loop for async handler pattern."""
        logger.debug(f"Receiver thread started for {self._host}:{self._port}")
        reconnect_attempts = 0

        while not self._shutdown_event.is_set():
            try:
                with self._socket_lock:
                    sock = self._socket
                    if sock is None:
                        logger.debug("Receiver loop: socket is None, exiting")
                        break

                    try:
                        sock.settimeout(0.5)  # Short timeout for shutdown check
                        data = sock.recv(self._recv_buffer_size)
                    except socket.timeout:
                        continue
                    except OSError as e:
                        if not self._shutdown_event.is_set():
                            logger.debug(f"Receive error: {e}")
                        break

                if not data:
                    # Server disconnected
                    logger.info(f"Server {self._host}:{self._port} disconnected (empty recv)")
                    break

                self._buffer.append(data)

                # Process complete messages
                try:
                    for message in self._buffer.extract_messages():
                        if self._waiting_for_response.is_set():
                            self._response_queue.put(message)
                        elif self._handler:
                            self._handler.on_message(message)
                except ProtocolError as e:
                    if self._handler:
                        self._handler.on_error(e)
                    break

                reconnect_attempts = 0  # Reset on successful receive

            except Exception as e:
                if self._handler:
                    self._handler.on_error(e)
                break

        # Handle disconnection
        logger.debug(f"Receiver loop ended for {self._host}:{self._port}, shutdown={self._shutdown_event.is_set()}")
        if not self._shutdown_event.is_set():
            if self._auto_reconnect:
                logger.debug("Starting auto-reconnect...")
                self._attempt_reconnect(reconnect_attempts)
            else:
                logger.debug("No auto-reconnect, cleaning up")
                self._cleanup()
                if self._handler:
                    self._handler.on_disconnect()

    def _attempt_reconnect(self, attempts: int) -> None:
        """Attempt to reconnect to the server."""
        max_attempts = self._max_reconnect_attempts or float("inf")

        while attempts < max_attempts and not self._shutdown_event.is_set():
            attempts += 1
            logger.info(f"Reconnection attempt {attempts}...")

            self._cleanup()
            self._shutdown_event.wait(self._reconnect_delay)

            if self._shutdown_event.is_set():
                break

            try:
                self._set_state(ClientState.CONNECTING)
                self._socket = self._create_socket()
                self._socket.connect((self._host, self._port))

                if self._recv_timeout is not None:
                    self._socket.settimeout(self._recv_timeout)
                else:
                    self._socket.setblocking(True)

                self._buffer.clear()
                self._set_state(ClientState.CONNECTED)

                if self._handler:
                    self._handler.on_connect()

                logger.info(f"Reconnected to {self._host}:{self._port}")
                # Restart receive loop
                self._receive_loop()
                return

            except OSError as e:
                logger.warning(f"Reconnection failed: {e}")

        # Max attempts reached
        self._cleanup()
        if self._handler:
            self._handler.on_disconnect(
                ConnectionError(f"Failed to reconnect after {attempts} attempts")
            )

    @contextmanager
    def connected(self) -> Generator[SocketClient, None, None]:
        """
        Context manager for connection lifecycle.

        Example:
            with client.connected():
                client.send(b"Hello")
        """
        self.connect()
        try:
            yield self
        finally:
            self.disconnect()


class QueuedClient(SocketClient):
    """
    Socket client that queues received messages.

    Useful when you want to process messages in your own thread
    rather than in a handler callback.

    Example:
        client = QueuedClient("localhost", 8080)
        with client.connected():
            client.send(b"Hello")
            message = client.get_message(timeout=5.0)
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_queue_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        self._message_queue: queue.Queue[Message] = queue.Queue(
            maxsize=max_queue_size
        )

        # Create internal handler
        handler = _QueueHandler(self._message_queue)
        super().__init__(host, port, handler=handler, **kwargs)

    def get_message(self, timeout: float | None = None) -> Message | None:
        """
        Get the next message from the queue.

        Args:
            timeout: How long to wait (None for blocking)

        Returns:
            Message or None if timeout
        """
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_all_messages(self) -> list[Message]:
        """Get all currently queued messages."""
        messages = []
        while True:
            try:
                messages.append(self._message_queue.get_nowait())
            except queue.Empty:
                break
        return messages


class _QueueHandler(ClientEventHandler):
    """Internal handler that queues messages."""

    def __init__(self, message_queue: queue.Queue[Message]) -> None:
        self._queue = message_queue

    def on_message(self, message: Message) -> None:
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            logger.warning("Message queue full, dropping message")
