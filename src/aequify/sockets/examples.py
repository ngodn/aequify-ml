"""
Example usage of the aequify socket module.

Run with Python 3.14+:
    python3.14 -m aequify.socket.examples demo
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from aequify.logging import get_logger

from . import (
    CallbackHandler,
    ClientConnection,
    ClientEventHandler,
    Message,
    MessageHandler,
    QueuedClient,
    SocketClient,
    SocketServer,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# =============================================================================
# Example 1: Simple Echo Server
# =============================================================================


class EchoHandler(MessageHandler):
    """Echo server that returns messages back to sender."""

    def on_message(
        self, client: ClientConnection, message: Message
    ) -> bytes | None:
        logger.info(f"Echo: {message.data!r} from {client.address}")
        return message.data

    def on_connect(self, client: ClientConnection) -> None:
        logger.info(f"Client connected: {client.address}")

    def on_disconnect(self, client: ClientConnection) -> None:
        logger.info(f"Client disconnected: {client.address}")


def example_echo_server() -> None:
    """Run a simple echo server."""
    server = SocketServer(
        "127.0.0.1",
        9000,
        EchoHandler(),
        thread_name_prefix="aq-echo-9000",
    )
    logger.info("Starting echo server on port 9000...")
    server.start()


# =============================================================================
# Example 2: Chat Server with Broadcasting
# =============================================================================


class ChatHandler(MessageHandler):
    """Chat server that broadcasts messages to all clients."""

    def __init__(self, server: SocketServer) -> None:
        self._server = server
        self._lock = threading.Lock()
        self._usernames: dict[tuple[str, int], str] = {}

    def on_message(
        self, client: ClientConnection, message: Message
    ) -> bytes | None:
        text = message.data.decode("utf-8", errors="replace")

        # Handle username setting
        if text.startswith("/name "):
            username = text[6:].strip()
            with self._lock:
                self._usernames[client.address] = username
            return f"Username set to: {username}".encode()

        # Broadcast message
        with self._lock:
            username = self._usernames.get(client.address, str(client.address))

        broadcast_msg = f"[{username}] {text}".encode()
        count = self._server.broadcast(broadcast_msg, exclude=client)
        logger.info(f"Broadcasted to {count} clients: {text}")

        return None  # No direct response

    def on_connect(self, client: ClientConnection) -> None:
        with self._lock:
            self._usernames[client.address] = f"User-{client.address[1]}"
        logger.info(f"New chat client: {client.address}")

    def on_disconnect(self, client: ClientConnection) -> None:
        with self._lock:
            self._usernames.pop(client.address, None)
        logger.info(f"Chat client left: {client.address}")


def example_chat_server() -> None:
    """Run a chat server."""
    server = SocketServer("127.0.0.1", 9001, handler=None)  # type: ignore
    handler = ChatHandler(server)
    server._handler = handler
    logger.info("Starting chat server on port 9001...")
    server.start()


# =============================================================================
# Example 3: Synchronous Client
# =============================================================================


def example_sync_client() -> None:
    """Demonstrate synchronous client usage."""
    client = SocketClient(
        "127.0.0.1",
        9000,
        connect_timeout=5.0,
        thread_name_prefix="aq-sync-9000",
    )

    with client.connected():
        # Send and receive in one call
        response = client.send_receive(b"Hello, Server!", timeout=5.0)
        if response:
            logger.info(f"Received: {response.data!r}")

        # Send multiple messages
        for i in range(3):
            response = client.send_receive(f"Message {i}".encode(), timeout=5.0)
            if response:
                logger.info(f"Response {i}: {response.data!r}")


# =============================================================================
# Example 4: Asynchronous Client with Handler
# =============================================================================


class AsyncClientHandler(ClientEventHandler):
    """Handler for async message processing."""

    def __init__(self) -> None:
        self._message_count = 0
        self._lock = threading.Lock()

    def on_message(self, message: Message) -> None:
        with self._lock:
            self._message_count += 1
            count = self._message_count
        logger.info(f"Async received #{count}: {message.data!r}")

    def on_connect(self) -> None:
        logger.info("Async client connected")

    def on_disconnect(self, error: Exception | None = None) -> None:
        if error:
            logger.error(f"Async client disconnected with error: {error}")
        else:
            logger.info("Async client disconnected")


def example_async_client() -> None:
    """Demonstrate asynchronous client usage."""
    handler = AsyncClientHandler()
    client = SocketClient(
        "127.0.0.1",
        9000,
        handler=handler,
        thread_name_prefix="aq-async-9000",
    )

    with client.connected():
        # Send messages (responses handled by handler)
        for i in range(5):
            client.send(f"Async message {i}".encode())
            time.sleep(0.1)

        # Wait for responses
        time.sleep(0.5)


# =============================================================================
# Example 5: Queued Client
# =============================================================================


def example_queued_client() -> None:
    """Demonstrate queued client for custom message processing."""
    client = QueuedClient(
        "127.0.0.1",
        9000,
        max_queue_size=100,
        thread_name_prefix="aq-queued-9000",
    )

    with client.connected():
        # Send several messages
        for i in range(3):
            client.send(f"Queued message {i}".encode())

        # Process messages in our own loop
        time.sleep(0.3)  # Wait for responses
        messages = client.get_all_messages()

        logger.info(f"Received {len(messages)} messages from queue:")
        for msg in messages:
            logger.info(f"  - {msg.data!r}")


# =============================================================================
# Example 6: Callback-based Server
# =============================================================================


def example_callback_server() -> None:
    """Demonstrate callback-based handler."""
    connection_count = 0
    count_lock = threading.Lock()

    def on_connect(client: ClientConnection) -> None:
        nonlocal connection_count
        with count_lock:
            connection_count += 1
            logger.info(f"Connection #{connection_count} from {client.address}")

    def handle_message(
        client: ClientConnection, message: Message
    ) -> bytes | None:
        text = message.data.decode()
        logger.info(f"Callback handler received: {text}")
        return f"Processed: {text.upper()}".encode()

    handler = CallbackHandler(
        callback=handle_message,
        on_connect=on_connect,
    )

    server = SocketServer(
        "127.0.0.1",
        9002,
        handler,
        thread_name_prefix="aq-callback-9002",
    )
    logger.info("Starting callback server on port 9002...")
    server.start()


# =============================================================================
# Example 7: Server with Context Manager
# =============================================================================


def example_server_context() -> None:
    """Demonstrate server context manager for background operation."""
    server = SocketServer(
        "127.0.0.1",
        9003,
        EchoHandler(),
        thread_name_prefix="aq-ctx-9003",
    )

    with server.running():
        logger.info("Server running in background...")

        # Do other work while server runs
        time.sleep(2)

        logger.info(f"Current clients: {server.client_count}")

    logger.info("Server stopped via context manager")


# =============================================================================
# Example 8: Full Integration Demo
# =============================================================================


def example_integration_demo() -> None:
    """
    Run a full integration demo with server and multiple clients.

    This demonstrates:
    - Background server operation
    - Multiple concurrent clients
    - Different client patterns (sync, async, queued)
    - Thread-safe operation
    """
    logger.info("=== Integration Demo ===")

    # Start server in background
    server = SocketServer(
        "127.0.0.1",
        9999,
        EchoHandler(),
        thread_name_prefix="aq-core-9999",
    )

    with server.running():
        logger.info("Server started in background")
        time.sleep(0.1)  # Let server start

        # Sync client
        sync_client = SocketClient(
            "127.0.0.1",
            9999,
            thread_name_prefix="aq-tui-9999",
        )
        with sync_client.connected():
            resp = sync_client.send_receive(b"Sync hello!")
            logger.info(f"Sync client got: {resp.data if resp else 'None'}")

        # Async client
        async_handler = AsyncClientHandler()
        async_client = SocketClient(
            "127.0.0.1",
            9999,
            handler=async_handler,
            thread_name_prefix="aq-tui-9999",
        )
        with async_client.connected():
            async_client.send(b"Async hello!")
            time.sleep(0.2)

        # Queued client
        queued = QueuedClient(
            "127.0.0.1",
            9999,
            thread_name_prefix="aq-tui-9999",
        )
        with queued.connected():
            queued.send(b"Queued hello!")
            time.sleep(0.1)
            msg = queued.get_message(timeout=1.0)
            logger.info(f"Queued client got: {msg.data if msg else 'None'}")

        # Multiple concurrent clients
        logger.info("Testing concurrent clients...")
        results: list[bytes] = []
        results_lock = threading.Lock()

        def client_worker(client_id: int) -> None:
            client = SocketClient(
                "127.0.0.1",
                9999,
                thread_name_prefix=f"aq-worker{client_id}-9999",
            )
            with client.connected():
                resp = client.send_receive(f"Client {client_id}".encode())
                if resp:
                    with results_lock:
                        results.append(resp.data)

        threads = [
            threading.Thread(
                target=client_worker,
                args=(i,),
                name=f"aq-worker{i}-9999--aqsockclient",
            )
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logger.info(f"Concurrent results: {len(results)} responses")

    logger.info("=== Demo Complete ===")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    from aequify.logging import setup_logging

    setup_logging()

    examples = {
        "echo": ("Echo Server", example_echo_server),
        "chat": ("Chat Server", example_chat_server),
        "sync": ("Sync Client (needs echo server)", example_sync_client),
        "async": ("Async Client (needs echo server)", example_async_client),
        "queued": ("Queued Client (needs echo server)", example_queued_client),
        "callback": ("Callback Server", example_callback_server),
        "context": ("Server Context Manager", example_server_context),
        "demo": ("Full Integration Demo", example_integration_demo),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in examples:
        print("Usage: python -m aequify.socket.examples <example>")
        print("\nAvailable examples:")
        for name, (desc, _) in examples.items():
            print(f"  {name:10} - {desc}")
        sys.exit(1)

    name = sys.argv[1]
    desc, func = examples[name]
    logger.info(f"Running example: {desc}")
    func()
