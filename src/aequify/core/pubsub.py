"""
Aequify PubSub - Topic-based publish/subscribe over sockets.

Provides real-time message delivery for decoupled components.
Built on top of aequify.sockets for reliable communication.

Port: Default is 9901, configurable via config.yaml pubsub.port.

Example Publisher (Engine):
    server = PubSubServer("127.0.0.1", 9901)
    server.start_background()

    # Publish state changes
    server.publish("engine.state", {"tick": 42, "status": "running"})
    server.publish("rate_limit", {"weight": 1200, "limit": 2400})

Example Subscriber (TUI):
    subscriber = Subscriber("127.0.0.1", 9901, topics=["engine.state"])
    subscriber.connect()

    # Receive messages
    topic, data = subscriber.receive(timeout=1.0)
    if topic:
        print(f"{topic}: {data}")
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aequify.logging import get_logger
from aequify.sockets import (
    CallbackHandler,
    ClientConnection,
    ClientEventHandler,
    Message,
    QueuedClient,
    SocketServer,
)

logger = get_logger(__name__)


@dataclass
class PubSubMessage:
    """A pub/sub message with topic and payload."""

    topic: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps({
            "topic": self.topic,
            "data": self.data,
            "ts": self.timestamp,
        }).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "PubSubMessage":
        """Deserialize from JSON bytes."""
        parsed = json.loads(data)
        return cls(
            topic=parsed["topic"],
            data=parsed["data"],
            timestamp=parsed.get("ts", time.time()),
        )


class PubSubServer:
    """
    Publish/Subscribe server for broadcasting messages by topic.

    Features:
    - Topic-based message routing
    - Client subscription filtering (optional)
    - Broadcast to all or filtered clients
    - Thread-safe publishing

    Example:
        server = PubSubServer("0.0.0.0", 9901)
        server.start_background()

        server.publish("engine.state", {"tick": 1, "status": "running"})
        server.publish("market.btc", {"price": 50000})

        server.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9901,
        thread_name_prefix: str = "pubsub",
    ) -> None:
        """
        Initialize the pub/sub server.

        Args:
            host: Host address to bind to.
            port: Port number to bind to (default: 9901, configurable via config.yaml).
            thread_name_prefix: Prefix for thread names.
        """
        self._host = host
        self._port = port

        # Track client subscriptions
        self._subscriptions: dict[tuple[str, int], set[str]] = {}
        self._subscriptions_lock = threading.Lock()

        # Create handler that processes subscription requests
        handler = CallbackHandler(
            callback=self._handle_message,
            on_connect=self._on_client_connect,
            on_disconnect=self._on_client_disconnect,
        )

        self._server = SocketServer(
            host=host,
            port=port,
            handler=handler,
            thread_name_prefix=thread_name_prefix,
            timeout=1.0,  # For shutdown responsiveness
        )

    @property
    def address(self) -> tuple[str, int]:
        """Get server address."""
        return (self._host, self._port)

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return self._server.client_count

    def _on_client_connect(self, client: ClientConnection) -> None:
        """Handle client connection."""
        with self._subscriptions_lock:
            # Default: subscribe to all topics (empty set = all)
            self._subscriptions[client.address] = set()
        logger.debug(f"PubSub client connected: {client.address}")

    def _on_client_disconnect(self, client: ClientConnection) -> None:
        """Handle client disconnection."""
        with self._subscriptions_lock:
            self._subscriptions.pop(client.address, None)
        logger.debug(f"PubSub client disconnected: {client.address}")

    def _handle_message(
        self, client: ClientConnection, message: Message
    ) -> bytes | None:
        """
        Handle incoming messages from clients.

        Clients can send subscription requests:
        {"subscribe": ["topic1", "topic2"]}
        {"unsubscribe": ["topic1"]}
        {"subscribe_all": true}
        """
        try:
            data = json.loads(message.data)

            if "subscribe" in data:
                topics = set(data["subscribe"])
                with self._subscriptions_lock:
                    if client.address in self._subscriptions:
                        self._subscriptions[client.address].update(topics)
                logger.debug(f"Client {client.address} subscribed to: {topics}")
                return json.dumps({"ok": True, "subscribed": list(topics)}).encode()

            if "unsubscribe" in data:
                topics = set(data["unsubscribe"])
                with self._subscriptions_lock:
                    if client.address in self._subscriptions:
                        self._subscriptions[client.address] -= topics
                return json.dumps({"ok": True, "unsubscribed": list(topics)}).encode()

            if data.get("subscribe_all"):
                with self._subscriptions_lock:
                    self._subscriptions[client.address] = set()  # Empty = all
                return json.dumps({"ok": True, "subscribed": "all"}).encode()

        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON"}).encode()
        except Exception as e:
            return json.dumps({"error": str(e)}).encode()

        return None

    def publish(self, topic: str, data: dict[str, Any]) -> int:
        """
        Publish a message to a topic.

        Args:
            topic: Topic name (e.g., "engine.state", "rate_limit").
            data: Message payload as dictionary.

        Returns:
            Number of clients the message was sent to.
        """
        message = PubSubMessage(topic=topic, data=data)
        payload = message.to_json()

        # Get clients subscribed to this topic
        with self._subscriptions_lock:
            # Copy to avoid holding lock during send
            subscriptions = dict(self._subscriptions)

        count = 0
        for address, topics in subscriptions.items():
            # Empty set = subscribed to all topics
            if not topics or topic in topics or self._matches_pattern(topic, topics):
                # Send via server's broadcast mechanism
                # We need direct client access, so use internal server state
                with self._server._clients_lock:
                    client = self._server._clients.get(address)
                    if client and client.send(payload):
                        count += 1

        return count

    @staticmethod
    def _matches_pattern(topic: str, patterns: set[str]) -> bool:
        """Check if topic matches any wildcard patterns."""
        for pattern in patterns:
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if topic.startswith(prefix + ".") or topic == prefix:
                    return True
            elif pattern.endswith("*"):
                prefix = pattern[:-1]
                if topic.startswith(prefix):
                    return True
        return False

    def start(self) -> None:
        """Start the server (blocking)."""
        logger.info(f"PubSub server starting on {self._host}:{self._port}")
        self._server.start()

    def start_background(self) -> threading.Thread:
        """Start the server in background thread."""
        logger.info(f"PubSub server starting on {self._host}:{self._port} (background)")
        return self._server.start_background()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the server."""
        self._server.stop(timeout)
        logger.info("PubSub server stopped")


class Subscriber:
    """
    Pub/Sub subscriber client.

    Connects to a PubSubServer and receives messages filtered by topic.

    Example:
        subscriber = Subscriber("127.0.0.1", 9901)
        subscriber.connect()
        subscriber.subscribe(["engine.state", "rate_limit"])

        while True:
            msg = subscriber.receive(timeout=1.0)
            if msg:
                print(f"{msg.topic}: {msg.data}")
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9901,
        topics: list[str] | None = None,
        auto_reconnect: bool = True,
    ) -> None:
        """
        Initialize the subscriber.

        Args:
            host: Server host address.
            port: Server port number (default: 9901, configurable via config.yaml).
            topics: Topics to subscribe to (None = all).
            auto_reconnect: Enable automatic reconnection.
        """
        self._host = host
        self._port = port
        self._initial_topics = topics
        self._subscribed_topics: set[str] = set(topics) if topics else set()

        self._client = QueuedClient(
            host=host,
            port=port,
            auto_reconnect=auto_reconnect,
            thread_name_prefix="aeq-sub",
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._client.is_connected

    def connect(self) -> None:
        """Connect to the pub/sub server."""
        logger.debug(f"Subscriber connecting to {self._host}:{self._port}...")
        self._client.connect()
        logger.debug(f"Subscriber connected, is_connected={self._client.is_connected}")

        # Send initial subscription
        if self._initial_topics:
            result = self.subscribe(self._initial_topics)
            logger.debug(f"Initial subscription sent: {result}")
        else:
            result = self._subscribe_all()
            logger.debug(f"Subscribe all sent: {result}")

    def disconnect(self) -> None:
        """Disconnect from the server."""
        self._client.disconnect()

    def subscribe(self, topics: list[str]) -> bool:
        """
        Subscribe to additional topics.

        Args:
            topics: Topics to subscribe to.

        Returns:
            True if subscription was sent.
        """
        if not self._client.is_connected:
            return False

        self._subscribed_topics.update(topics)
        msg = json.dumps({"subscribe": topics}).encode()
        return self._client.send(msg)

    def unsubscribe(self, topics: list[str]) -> bool:
        """
        Unsubscribe from topics.

        Args:
            topics: Topics to unsubscribe from.

        Returns:
            True if unsubscription was sent.
        """
        if not self._client.is_connected:
            return False

        self._subscribed_topics -= set(topics)
        msg = json.dumps({"unsubscribe": topics}).encode()
        return self._client.send(msg)

    def _subscribe_all(self) -> bool:
        """Subscribe to all topics."""
        if not self._client.is_connected:
            return False
        msg = json.dumps({"subscribe_all": True}).encode()
        return self._client.send(msg)

    def receive(self, timeout: float | None = None) -> PubSubMessage | None:
        """
        Receive the next message.

        Args:
            timeout: How long to wait (None for blocking).

        Returns:
            PubSubMessage or None if timeout.
        """
        message = self._client.get_message(timeout=timeout)
        if message is None:
            return None

        try:
            return PubSubMessage.from_json(message.data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse message: {e}")
            return None

    def receive_all(self) -> list[PubSubMessage]:
        """Get all queued messages."""
        messages = self._client.get_all_messages()
        result = []
        for msg in messages:
            try:
                result.append(PubSubMessage.from_json(msg.data))
            except (json.JSONDecodeError, KeyError):
                continue
        return result

    def __enter__(self) -> "Subscriber":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        """Context manager exit."""
        self.disconnect()


# Type alias for message handlers
MessageHandler = Callable[[PubSubMessage], None]


class SubscriberWorker:
    """
    Subscriber that processes messages in a callback.

    Useful for integrating with event loops or UI frameworks.

    Example:
        def handle_message(msg: PubSubMessage):
            print(f"{msg.topic}: {msg.data}")

        worker = SubscriberWorker("127.0.0.1", 9901, handler=handle_message)
        worker.start()  # Runs in background thread

        # Later
        worker.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9901,
        handler: MessageHandler | None = None,
        topics: list[str] | None = None,
    ) -> None:
        """
        Initialize the subscriber worker.

        Args:
            host: Server host address.
            port: Server port number (default: 9901, configurable via config.yaml).
            handler: Callback for received messages.
            topics: Topics to subscribe to.
        """
        self._subscriber = Subscriber(host, port, topics=topics)
        self._handler = handler
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._thread is not None and self._thread.is_alive()

    def set_handler(self, handler: MessageHandler) -> None:
        """Set the message handler."""
        self._handler = handler

    def start(self) -> None:
        """Start the worker in a background thread."""
        if self._thread is not None:
            raise RuntimeError("Worker already running")

        self._stop_event.clear()
        self._subscriber.connect()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="pubsub-worker",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._subscriber.disconnect()
        self._thread = None

    def _run(self) -> None:
        """Worker loop."""
        while not self._stop_event.is_set():
            try:
                msg = self._subscriber.receive(timeout=0.5)
                if msg and self._handler:
                    try:
                        self._handler(msg)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Subscriber error: {e}")
                break
