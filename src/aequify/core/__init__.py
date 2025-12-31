"""
Aequify Core - ML and Trading Utilities

Core functionality for the Aequify trading platform including:
- ML models and data processing
- Trading engine and exchange connectors
- Database integrations (TigerBeetle, QuestDB)

Optimized for Python 3.14+ free-threaded builds.
"""

from .exchange import BinanceFuturesRateLimiter, RateLimitState
from aequify.logging import get_logger, setup_logging, shutdown_logging
from .ports import PUBSUB_PORT
from .pubsub import PubSubMessage, PubSubServer, Subscriber, SubscriberWorker
from .system import SystemMonitor, SystemStats

__all__ = [
    # Exchange
    "BinanceFuturesRateLimiter",
    "RateLimitState",
    # Logging
    "get_logger",
    "setup_logging",
    "shutdown_logging",
    # Ports
    "PUBSUB_PORT",
    # PubSub
    "PubSubServer",
    "PubSubMessage",
    "Subscriber",
    "SubscriberWorker",
    # System monitoring
    "SystemMonitor",
    "SystemStats",
]
