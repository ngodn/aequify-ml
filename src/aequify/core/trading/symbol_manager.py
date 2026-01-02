"""
Symbol Manager - Per-symbol trading context with isolated async loop.

Each symbol gets its own SymbolManager instance running in an IsolatedLoop.
This ensures:
- Trade streams don't block the TUI
- Each symbol operates independently
- Trades flow to HotStore without contention

Architecture:
    SymbolManager (one per symbol)
        └─ IsolatedLoop (own thread + event loop)
            └─ BinanceFuturesTradeStream (WebSocket)
                └─ on_trade callback
                    └─ HotStore.add_trade()

Usage:
    from aequify.core.trading import SymbolManager

    # Create manager for a symbol
    manager = SymbolManager("BTC/USDT:USDT", demo=True)

    # Set trade callback (connects to HotStore from Mojo)
    manager.on_trade = hot_store.add_trade

    # Start streaming
    manager.start()

    # ... runs independently, doesn't block TUI ...

    # Stop when done
    manager.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from aequify.logging import get_logger
from aequify.runtime import IsolatedLoop

logger = get_logger(__name__)


# Trade dict type for callbacks (matches CCXT trade format passed to HotStore)
TradeDict = dict[str, Any]

# Trade callback type - receives trade dict, can be sync or async
TradeCallback = Callable[[TradeDict], None]


def _make_hot_store_handler(
    hot_registry: Any,
    symbol: str,
    publisher: Any = None,
) -> TradeCallback:
    """
    Create a trade handler that adds trades to HotStore and publishes via PubSub.

    Args:
        hot_registry: Mojo HotStoreRegistry instance.
        symbol: Trading pair in CCXT format (e.g., "BTC/USDT:USDT").
        publisher: Optional TradingPublisher for PubSub publishing.

    Returns:
        Callback function for on_trade.
    """
    # Normalize symbol for storage (remove slashes/colons)
    storage_symbol = symbol.replace("/", "").replace(":", "")

    def handler(trade: TradeDict) -> None:
        """Add trade to HotStore and publish via PubSub."""
        try:
            hot_registry.add_trade_raw(
                storage_symbol,
                int(trade["id"]),
                float(trade["price"]),
                float(trade["amount"]),
                int(trade["timestamp"]),
                trade.get("side") == "sell",  # is_buyer_maker
            )
        except Exception as e:
            logger.error(f"[{symbol}] Failed to add trade to HotStore: {e}")

        # Publish to PubSub for TUI
        if publisher is not None:
            try:
                publisher.publish_trade(symbol, trade)
            except Exception as e:
                logger.debug(f"[{symbol}] Failed to publish trade: {e}")

    return handler


@dataclass
class SymbolManagerConfig:
    """Configuration for SymbolManager."""

    # Use demo mode (Binance testnet)
    demo: bool = False

    # Stream reconnection settings
    max_retries: int = 0  # 0 = unlimited

    # Rate limiting
    rate_limit: bool = True

    # API receive window in ms
    recv_window: int = 5000


@dataclass
class SymbolManager:
    """
    Manages all trading operations for a single symbol.

    Runs in its own IsolatedLoop to ensure non-blocking operation.
    Subscribes to trade stream and passes trades to HotStore.

    Each symbol should have exactly one SymbolManager instance.

    Attributes:
        symbol: Trading pair in CCXT format (e.g., "BTC/USDT:USDT").
        config: Manager configuration.
        on_trade: Callback for each trade (connect to HotStore).
    """

    symbol: str
    config: SymbolManagerConfig = field(default_factory=SymbolManagerConfig)

    # Callbacks
    on_trade: TradeCallback | None = None
    on_connect: Callable[[], None] | None = None
    on_disconnect: Callable[[Exception | None], None] | None = None

    # Internal state
    _loop: IsolatedLoop | None = field(default=None, init=False, repr=False)
    _stream: Any = field(default=None, init=False, repr=False)  # BinanceFuturesTradeStream
    _running: bool = field(default=False, init=False)
    _trade_count: int = field(default=0, init=False)
    _connected: bool = field(default=False, init=False)
    _start_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the isolated loop."""
        # Create unique loop name from symbol
        loop_name = f"sym-{self.symbol.replace('/', '-').replace(':', '-').lower()}"
        self._loop = IsolatedLoop(loop_name)

    @property
    def is_running(self) -> bool:
        """Whether manager is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Whether trade stream is connected."""
        return self._connected

    @property
    def trade_count(self) -> int:
        """Total trades received since start."""
        return self._trade_count

    @property
    def uptime(self) -> float:
        """Seconds since start (0 if not running)."""
        if not self._running or self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def start(self) -> None:
        """
        Start the symbol manager.

        Starts the isolated event loop and trade stream.
        Non-blocking - returns immediately.
        """
        with self._lock:
            if self._running:
                logger.warning(f"[{self.symbol}] SymbolManager already running")
                return

            logger.info(f"[{self.symbol}] Starting SymbolManager")

            # Start the isolated loop
            self._loop.start()

            # Schedule stream start in the isolated loop
            self._loop.schedule(self._start_stream())

            self._running = True
            self._start_time = time.time()

            logger.info(f"[{self.symbol}] SymbolManager started")

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the symbol manager.

        Stops trade stream and isolated event loop.

        Args:
            timeout: Maximum time to wait for graceful shutdown.
        """
        with self._lock:
            if not self._running:
                return

            logger.info(f"[{self.symbol}] Stopping SymbolManager (trades: {self._trade_count})")

            self._running = False

            # Stop stream in the isolated loop
            if self._stream and self._loop and self._loop.is_running:
                try:
                    self._loop.run(self._stop_stream(), timeout=timeout)
                except Exception as e:
                    logger.warning(f"[{self.symbol}] Error stopping stream: {e}")

            # Stop the isolated loop
            if self._loop:
                self._loop.stop(timeout=timeout)

            self._connected = False
            self._stream = None

            logger.info(f"[{self.symbol}] SymbolManager stopped")

    async def _start_stream(self) -> None:
        """Start the trade stream (runs in isolated loop)."""
        from aequify.core.rtds.trade_streams import (
            BinanceFuturesTradeStream,
            StreamConfig,
        )

        stream_config = StreamConfig(
            demo=self.config.demo,
            max_retries=self.config.max_retries,
            rate_limit=self.config.rate_limit,
            recv_window=self.config.recv_window,
        )

        self._stream = BinanceFuturesTradeStream(
            symbol=self.symbol,
            on_trade=self._handle_trade,
            on_connect=self._handle_connect,
            on_disconnect=self._handle_disconnect,
            config=stream_config,
        )

        await self._stream.start()

    async def _stop_stream(self) -> None:
        """Stop the trade stream (runs in isolated loop)."""
        if self._stream:
            await self._stream.stop()

    async def _handle_trade(self, trade: TradeDict) -> None:
        """
        Handle incoming trade from stream.

        Increments counter and calls on_trade callback.
        This runs in the isolated loop's thread.
        """
        self._trade_count += 1

        if self.on_trade:
            try:
                self.on_trade(trade)
            except Exception as e:
                logger.error(f"[{self.symbol}] Trade callback error: {e}")

    async def _handle_connect(self) -> None:
        """Handle stream connection."""
        self._connected = True
        logger.info(f"[{self.symbol}] Trade stream connected")

        if self.on_connect:
            try:
                self.on_connect()
            except Exception as e:
                logger.error(f"[{self.symbol}] Connect callback error: {e}")

    async def _handle_disconnect(self, error: Exception | None) -> None:
        """Handle stream disconnection."""
        self._connected = False
        logger.warning(f"[{self.symbol}] Trade stream disconnected: {error}")

        if self.on_disconnect:
            try:
                self.on_disconnect(error)
            except Exception as e:
                logger.error(f"[{self.symbol}] Disconnect callback error: {e}")

    def stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "symbol": self.symbol,
            "is_running": self._running,
            "is_connected": self._connected,
            "trade_count": self._trade_count,
            "uptime": round(self.uptime, 1),
            "demo": self.config.demo,
        }

    def __enter__(self) -> "SymbolManager":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class SymbolManagerRegistry:
    """
    Registry for managing multiple SymbolManager instances.

    Thread-safe singleton for accessing managers by symbol.
    Manages HotStore integration - trades automatically flow to HotStore.
    Manages PubSub integration - trades are published for TUI consumption.

    Usage:
        # Get registry with HotStore and PubSub integration
        import hot  # Mojo compiled module
        from aequify.core.trading import get_trading_publisher

        registry = SymbolManagerRegistry()
        registry.set_hot_registry(hot.HotStoreRegistry())
        registry.set_publisher(get_trading_publisher())

        # Add a symbol (auto-wired to HotStore and PubSub)
        manager = registry.add("BTC/USDT:USDT", demo=True, auto_start=True)

        # Trades now flow: Stream → HotStore → PubSub (TUI)

        # Stop all
        registry.stop_all()
    """

    _instance: "SymbolManagerRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SymbolManagerRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._managers = {}
                    cls._instance._managers_lock = threading.Lock()
                    cls._instance._hot_registry = None
                    cls._instance._publisher = None
        return cls._instance

    def set_hot_registry(self, hot_registry: Any) -> None:
        """
        Set the HotStore registry for trade storage.

        When set, all new SymbolManagers will automatically wire
        their on_trade callback to store trades in HotStore.

        Args:
            hot_registry: Mojo HotStoreRegistry instance.
        """
        self._hot_registry = hot_registry
        logger.info("HotStore registry configured for trade storage")

    def set_publisher(self, publisher: Any) -> None:
        """
        Set the TradingPublisher for PubSub publishing.

        When set, all new SymbolManagers will automatically wire
        their on_trade callback to publish trades via PubSub.

        Args:
            publisher: TradingPublisher instance.
        """
        self._publisher = publisher
        logger.info("TradingPublisher configured for PubSub publishing")

    @property
    def hot_registry(self) -> Any | None:
        """Get the HotStore registry."""
        return self._hot_registry

    @property
    def publisher(self) -> Any | None:
        """Get the TradingPublisher."""
        return self._publisher

    def add(
        self,
        symbol: str,
        demo: bool = False,
        on_trade: TradeCallback | None = None,
        auto_start: bool = False,
    ) -> SymbolManager:
        """
        Add a new symbol manager.

        If hot_registry is set, trades are automatically stored in HotStore.

        Args:
            symbol: Trading pair in CCXT format.
            demo: Use demo mode.
            on_trade: Trade callback (optional, overrides auto HotStore wiring).
            auto_start: Start manager immediately.

        Returns:
            The created SymbolManager.

        Raises:
            ValueError: If symbol already exists.
        """
        with self._managers_lock:
            if symbol in self._managers:
                raise ValueError(f"Symbol {symbol} already registered")

            # Auto-wire to HotStore and PubSub if available and no custom callback
            if on_trade is None and self._hot_registry is not None:
                on_trade = _make_hot_store_handler(
                    self._hot_registry,
                    symbol,
                    publisher=self._publisher,
                )
                logger.debug(f"[{symbol}] Auto-wired to HotStore and PubSub")

            config = SymbolManagerConfig(demo=demo)
            manager = SymbolManager(symbol=symbol, config=config, on_trade=on_trade)
            self._managers[symbol] = manager

            if auto_start:
                manager.start()

            logger.info(f"Registered SymbolManager for {symbol}")
            return manager

    def get(self, symbol: str) -> SymbolManager | None:
        """Get manager for a symbol."""
        with self._managers_lock:
            return self._managers.get(symbol)

    def remove(self, symbol: str, timeout: float = 10.0) -> bool:
        """
        Remove and stop a symbol manager.

        Args:
            symbol: Symbol to remove.
            timeout: Stop timeout.

        Returns:
            True if removed, False if not found.
        """
        with self._managers_lock:
            manager = self._managers.pop(symbol, None)

        if manager:
            manager.stop(timeout=timeout)
            logger.info(f"Removed SymbolManager for {symbol}")
            return True
        return False

    def symbols(self) -> list[str]:
        """Get list of registered symbols."""
        with self._managers_lock:
            return list(self._managers.keys())

    def all(self) -> list[SymbolManager]:
        """Get all managers."""
        with self._managers_lock:
            return list(self._managers.values())

    def start_all(self) -> None:
        """Start all managers."""
        for manager in self.all():
            if not manager.is_running:
                manager.start()

    def stop_all(self, timeout: float = 10.0) -> None:
        """Stop all managers."""
        managers = self.all()
        for manager in managers:
            manager.stop(timeout=timeout)

    def stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        managers = self.all()
        return {
            "symbol_count": len(managers),
            "running_count": sum(1 for m in managers if m.is_running),
            "connected_count": sum(1 for m in managers if m.is_connected),
            "total_trades": sum(m.trade_count for m in managers),
            "symbols": {m.symbol: m.stats() for m in managers},
        }


# Convenience function
def get_symbol_registry() -> SymbolManagerRegistry:
    """Get the global SymbolManagerRegistry singleton."""
    return SymbolManagerRegistry()
