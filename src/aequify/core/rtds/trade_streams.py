"""
Binance Futures Trade Stream using CCXT Pro.

CCXT Pro (now free in CCXT 1.95+) handles:
- WebSocket connection management
- Auto-reconnection
- Ping/pong keep-alive
- Message parsing to unified format

Each stream runs independently for a single symbol.
Later can add different stream types (kline, orderbook, etc.) per symbol.

Usage:
------

# Import the Mojo HotStore bridge (compiled as Python extension)
import hot  # from aequify.core.db.hot_bridge

# Create callback that adds trades to HotStore
def on_trade(trade_dict):
    hot.add_trade("BTCUSDT", trade_dict)

stream = BinanceFuturesTradeStream(
    symbol="BTC/USDT:USDT",
    on_trade=on_trade,
)

await stream.start()
# ... runs until stopped
await stream.stop()

# Or use context manager
async with BinanceFuturesTradeStream("BTC/USDT:USDT", on_trade=...) as stream:
    await asyncio.sleep(60)  # Run for 1 minute
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)


# Trade dict type for callbacks (matches CCXT trade format)
TradeDict = dict[str, Any]


@dataclass
class StreamConfig:
    """Configuration for a trade stream."""

    # Use demo mode (Binance testnet)
    demo: bool = False

    # Reconnection settings (CCXT handles this, but we can configure)
    max_retries: int = 0  # 0 = unlimited

    # Rate limiting
    rate_limit: bool = True

    # API receive window in ms (for timestamp validation)
    recv_window: int = 5000


@dataclass
class BinanceFuturesTradeStream:
    """
    Binance Futures trade stream for a single symbol using CCXT Pro.

    CCXT Pro handles all WebSocket complexity:
    - Connection management
    - Auto-reconnection with backoff
    - Ping/pong keep-alive
    - Message parsing to unified format

    Attributes:
        symbol: Trading pair in CCXT format (e.g., "BTC/USDT:USDT").
        on_trade: Callback for each trade (sync or async).
        on_connect: Optional callback when connected.
        on_disconnect: Optional callback when disconnected.
        config: Stream configuration.
        shared_exchange: Optional shared CCXT exchange instance to avoid
            multiple load_markets() API calls. If provided, this exchange
            will be used instead of creating a new one.
    """

    symbol: str
    on_trade: Callable[[TradeDict], None] | Callable[[TradeDict], Coroutine[Any, Any, None]]
    on_connect: Callable[[], Coroutine[Any, Any, None]] | None = None
    on_disconnect: Callable[[Exception | None], Coroutine[Any, Any, None]] | None = None
    config: StreamConfig = field(default_factory=StreamConfig)
    shared_exchange: ccxtpro.binanceusdm | None = None

    # Internal state
    _exchange: ccxtpro.binanceusdm | None = field(default=None, init=False)
    _owns_exchange: bool = field(default=False, init=False)  # Whether we created the exchange
    _task: asyncio.Task | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _trade_count: int = field(default=0, init=False)
    _connected: bool = field(default=False, init=False)

    @property
    def is_running(self) -> bool:
        """Whether stream is currently running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Whether WebSocket is connected."""
        return self._connected

    @property
    def trade_count(self) -> int:
        """Total trades received since start."""
        return self._trade_count

    async def start(self) -> None:
        """Start the trade stream."""
        if self._running:
            logger.warning(f"[{self.symbol}] Stream already running")
            return

        logger.debug(f"[{self.symbol}] Starting trade stream")

        # Use shared exchange if provided, otherwise create our own
        if self.shared_exchange is not None:
            self._exchange = self.shared_exchange
            self._owns_exchange = False
            logger.debug(f"[{self.symbol}] Using shared exchange instance")
        else:
            # Create exchange instance (will trigger load_markets on first call)
            exchange_config = {
                "enableRateLimit": self.config.rate_limit,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                    "recvWindow": self.config.recv_window,
                },
            }

            self._exchange = ccxtpro.binanceusdm(exchange_config)
            self._owns_exchange = True

            # For demo mode, use CCXT's built-in enable_demo_trading()
            # =============================================================================
            # BINANCE USDS-M FUTURES DEMO ENDPOINTS (as of Dec 2025):
            #   REST API:  https://demo-fapi.binance.com (NOT testnet.binancefuture.com!)
            #   WebSocket: wss://fstream.binancefuture.com
            # Demo API keys from: https://testnet.binancefuture.com (same keys work for demo-fapi)
            # Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
            # =============================================================================
            if self.config.demo:
                self._exchange.enable_demo_trading(True)
                logger.debug(f"[{self.symbol}] Configured for demo mode")

        self._running = True
        self._trade_count = 0
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the trade stream gracefully."""
        if not self._running:
            return

        logger.debug(f"[{self.symbol}] Stopping trade stream (received {self._trade_count} trades)")

        self._running = False

        # Cancel task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Only close exchange if we created it (not shared)
        if self._exchange and self._owns_exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.error(f"[{self.symbol}] Error closing exchange: {e}")
        self._exchange = None

        self._connected = False
        logger.debug(f"[{self.symbol}] Trade stream stopped")

    async def _run_loop(self) -> None:
        """Main loop watching trades."""
        retries = 0

        while self._running:
            try:
                # Watch trades - CCXT Pro handles reconnection internally
                trades = await self._exchange.watch_trades(self.symbol)

                # First successful fetch = connected
                if not self._connected:
                    self._connected = True
                    retries = 0
                    logger.info(f"[{self.symbol}] Trade stream connected")

                    if self.on_connect:
                        try:
                            await self.on_connect()
                        except Exception as e:
                            logger.error(f"[{self.symbol}] on_connect error: {e}")

                # Process trades
                for ccxt_trade in trades:
                    await self._handle_trade(ccxt_trade)

            except asyncio.CancelledError:
                break

            except Exception as e:
                was_connected = self._connected
                self._connected = False

                if was_connected:
                    logger.warning(f"[{self.symbol}] Disconnected: {e}")

                    if self.on_disconnect:
                        try:
                            await self.on_disconnect(e)
                        except Exception as cb_err:
                            logger.error(f"[{self.symbol}] on_disconnect error: {cb_err}")
                else:
                    # Log first connection failure
                    logger.warning(f"[{self.symbol}] Connection failed: {type(e).__name__}: {e}")

                # Check retry limit
                retries += 1
                if self.config.max_retries > 0 and retries > self.config.max_retries:
                    logger.error(
                        f"[{self.symbol}] Max retries ({self.config.max_retries}) exceeded"
                    )
                    break

                # Exponential backoff
                delay = min(2**retries, 60)
                logger.info(f"[{self.symbol}] Reconnecting in {delay}s (attempt {retries})")
                await asyncio.sleep(delay)

    async def _handle_trade(self, ccxt_trade: dict) -> None:
        """Process CCXT trade and call callback."""
        try:
            # Guard: ignore invalid trades (bad data from exchange)
            price = ccxt_trade.get("price", 0)
            if not price or price <= 0:
                logger.debug(f"[{self.symbol}] Ignoring trade with invalid price: {price}")
                return

            # Pass CCXT trade dict directly to callback
            # Callback should use hot.add_trade(symbol, trade_dict) for HotStore
            self._trade_count += 1

            # Call callback (sync or async)
            result = self.on_trade(ccxt_trade)
            if hasattr(result, "__await__"):
                await result

        except Exception as e:
            logger.error(f"[{self.symbol}] Failed to handle trade: {e}")

    def stats(self) -> dict:
        """Get stream statistics."""
        return {
            "symbol": self.symbol,
            "is_running": self._running,
            "is_connected": self._connected,
            "trade_count": self._trade_count,
            "demo": self.config.demo,
        }

    # Context manager support
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
