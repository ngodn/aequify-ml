"""
Aequify Socket Module

Thread-safe socket server and client implementation for Python 3.14+.

This module provides:
- SocketServer: Multi-threaded TCP server with per-client handling
- SocketClient: Thread-safe TCP client with sync/async patterns
- QueuedClient: Client variant that queues messages for processing
- Message framing protocol for reliable message boundaries

Example Server:
    from aequify.socket import SocketServer, MessageHandler

    class EchoHandler(MessageHandler):
        def on_message(self, client, message):
            return message.data  # Echo back

    server = SocketServer("0.0.0.0", 8080, EchoHandler())
    server.start()

Example Client (synchronous):
    from aequify.socket import SocketClient

    client = SocketClient("localhost", 8080)
    with client.connected():
        response = client.send_receive(b"Hello, World!")
        print(response.data)

Example Client (asynchronous):
    from aequify.socket import SocketClient, ClientEventHandler

    class MyHandler(ClientEventHandler):
        def on_message(self, message):
            print(f"Received: {message.data}")

    client = SocketClient("localhost", 8080, handler=MyHandler())
    with client.connected():
        client.send(b"Hello")
        time.sleep(1)  # Wait for response
"""

from .client import (
    ClientEventHandler,
    ClientState,
    ConnectionError,
    QueuedClient,
    SocketClient,
)
from .protocol import (
    HEADER_SIZE,
    MAX_MESSAGE_SIZE,
    Message,
    MessageBuffer,
    MessageTooLargeError,
    ProtocolError,
    decode_header,
    encode_message,
)
from .server import (
    CallbackHandler,
    ClientConnection,
    EchoHandler,
    MessageHandler,
    ServerState,
    SocketServer,
)

__all__ = [
    # Server
    "SocketServer",
    "ServerState",
    "ClientConnection",
    "MessageHandler",
    "CallbackHandler",
    "EchoHandler",
    # Client
    "SocketClient",
    "QueuedClient",
    "ClientState",
    "ClientEventHandler",
    "ConnectionError",
    # Protocol
    "Message",
    "MessageBuffer",
    "ProtocolError",
    "MessageTooLargeError",
    "encode_message",
    "decode_header",
    "HEADER_SIZE",
    "MAX_MESSAGE_SIZE",
]
