"""
Binance Futures Position Stream using CCXT Pro.

Monitors account positions in real-time via user data stream.
Uses ACCOUNT_UPDATE events from Binance WebSocket.

CCXT Pro handles:
- Listen key management
- Auto-reconnection
- Keep-alive pings

Usage:
------

from aequify.rtds import BinanceFuturesPositionStream

stream = BinanceFuturesPositionStream(
    api_key="your_api_key",
    api_secret="your_api_secret",
    on_position=lambda positions: print(positions),
)

await stream.start()
# ... runs until stopped
await stream.stop()

# Or use context manager
async with stream:
    await asyncio.sleep(60)
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import ccxt
import ccxt.pro as ccxtpro

# Transient errors that should be retried with backoff
TRANSIENT_ERRORS = (
    ccxt.ExchangeNotAvailable,
    ccxt.NetworkError,
    ccxt.RequestTimeout,
)

if TYPE_CHECKING:
    from aequify.runtime import IsolatedLoop

logger = logging.getLogger(__name__)


@dataclass
class PositionData:
    """
    Position data for display.

    Simplified position info from CCXT unified format.
    """

    symbol: str
    side: str  # "long" or "short"
    contracts: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    notional: float
    leverage: int
    margin_mode: str  # "cross" or "isolated"
    initial_margin: float
    timestamp_ms: int

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.side.lower() == "long"

    @property
    def is_profitable(self) -> bool:
        """Check if position is in profit."""
        return self.unrealized_pnl > 0

    @property
    def base_symbol(self) -> str:
        """Extract base symbol (e.g., 'BTC' from 'BTC/USDT:USDT')."""
        if "/" in self.symbol:
            return self.symbol.split("/")[0]
        return self.symbol

    @property
    def timestamp(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp_ms / 1000)

    @classmethod
    def from_ccxt_position(cls, pos: dict[str, Any]) -> PositionData:
        """Create from CCXT unified position format."""
        # Get leverage from position or calculate from margin percentage
        leverage = pos.get("leverage")
        if leverage is None:
            init_margin_pct = pos.get("initialMarginPercentage", 0)
            if init_margin_pct and init_margin_pct > 0:
                leverage = int(1 / init_margin_pct)

        return cls(
            symbol=pos.get("symbol", ""),
            side=(pos.get("side") or "").lower(),
            contracts=float(pos.get("contracts", 0) or 0),
            entry_price=float(pos.get("entryPrice", 0) or 0),
            mark_price=float(pos.get("markPrice", 0) or 0),
            liquidation_price=float(pos.get("liquidationPrice", 0) or 0),
            unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
            unrealized_pnl_pct=float(pos.get("percentage", 0) or 0),
            notional=float(pos.get("notional", 0) or 0),
            leverage=leverage or 0,
            margin_mode=pos.get("marginMode", "cross") or "cross",
            initial_margin=float(pos.get("initialMargin", 0) or 0),
            timestamp_ms=pos.get("timestamp", 0) or int(datetime.now().timestamp() * 1000),
        )


@dataclass
class PositionStreamConfig:
    """Configuration for position stream."""

    # Use demo mode (Binance testnet)
    demo: bool = False

    # API credentials
    api_key: str = ""
    api_secret: str = ""

    # Fetch initial snapshot before streaming
    fetch_snapshot: bool = True

    # Reconnection settings
    max_retries: int = 0  # 0 = unlimited

    # Delay in seconds after learning completes before next position poll
    # Flow: poll positions -> run learning -> wait delay -> repeat
    poll_delay: float = 10.0

    # API receive window in ms (for timestamp validation)
    recv_window: int = 5000


# Callback types
PositionCallback = (
    Callable[[list[PositionData]], None] | Callable[[list[PositionData]], Coroutine[Any, Any, None]]
)
ConnectCallback = Callable[[], Coroutine[Any, Any, None]] | None
DisconnectCallback = Callable[[Exception | None], Coroutine[Any, Any, None]] | None

# Learning callback - receives list of positions, returns when learning is complete
# This is called after each position poll, before the delay
LearningCallback = Callable[[list[PositionData]], Coroutine[Any, Any, None]] | None


@dataclass
class BinanceFuturesPositionStream:
    """
    Binance Futures position stream using CCXT Pro.

    Monitors account positions via user data stream (ACCOUNT_UPDATE events).
    Fetches initial snapshot then provides real-time updates.

    Attributes:
        on_position: Callback for position updates (receives list of all positions).
        on_connect: Optional callback when connected.
        on_disconnect: Optional callback when disconnected.
        config: Stream configuration.
    """

    on_position: PositionCallback
    on_connect: ConnectCallback = None
    on_disconnect: DisconnectCallback = None
    on_learning: LearningCallback = None  # Called after position poll, before delay
    config: PositionStreamConfig = field(default_factory=PositionStreamConfig)

    # Internal state
    _exchange: ccxtpro.binanceusdm | None = field(default=None, init=False)
    _poll_exchange: ccxtpro.binanceusdm | None = field(default=None, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    _poll_task: asyncio.Task | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _connected: bool = field(default=False, init=False)
    _update_count: int = field(default=0, init=False)
    _positions: dict[str, PositionData] = field(default_factory=dict, init=False)
    _last_update_time: float = field(default=0.0, init=False)

    # Isolated event loop for REST polling (prevents blocking by main loop)
    _isolated_loop: "IsolatedLoop | None" = field(default=None, init=False)
    _main_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)

    @property
    def is_running(self) -> bool:
        """Whether stream is currently running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Whether WebSocket is connected."""
        return self._connected

    @property
    def update_count(self) -> int:
        """Total position updates received since start."""
        return self._update_count

    @property
    def positions(self) -> list[PositionData]:
        """Current positions."""
        return list(self._positions.values())

    async def start(self) -> None:
        """Start the position stream."""
        if self._running:
            logger.warning("Position stream already running")
            return

        if not self.config.api_key or not self.config.api_secret:
            raise ValueError("API credentials required for position stream")

        logger.info("Starting position stream")

        # Create authenticated exchange instance
        # For demo mode, disable snapshot fetch as it uses sapi endpoints that don't exist
        fetch_snapshot = self.config.fetch_snapshot and not self.config.demo
        exchange_config = {
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,  # Auto-sync time with Binance servers
                "recvWindow": self.config.recv_window,
                # Fetch snapshot and await it before streaming (disabled on demo)
                "watchPositions": {
                    "fetchPositionsSnapshot": fetch_snapshot,
                    "awaitPositionsSnapshot": fetch_snapshot,
                },
            },
        }

        self._exchange = ccxtpro.binanceusdm(exchange_config)

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
            logger.debug("Position stream configured for demo mode")

        # Create a separate exchange for REST polling (without sandbox mode issues)
        self._poll_exchange = ccxtpro.binanceusdm(
            {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": True,
                "timeout": 30000,  # 30s HTTP timeout for slow testnet
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,  # Auto-sync time with Binance servers
                    "recvWindow": self.config.recv_window,
                },
            }
        )
        if self.config.demo:
            self._poll_exchange.enable_demo_trading(True)

        self._running = True
        self._update_count = 0
        self._positions = {}
        self._last_update_time = 0.0

        # Store main loop reference for callbacks from isolated loop
        self._main_loop = asyncio.get_running_loop()

        # Get isolated event loop for REST polling (critical operations)
        from aequify.runtime import get_exchange_loop

        self._isolated_loop = get_exchange_loop()

        # WebSocket stream runs on main loop (CCXT Pro handles this well)
        self._task = asyncio.create_task(self._run_loop())

        # REST polling runs on isolated loop (prevents blocking by GPU/DB operations)
        self._poll_task = self._isolated_loop.create_task(self._poll_loop_isolated())

        logger.info(
            "Position stream tasks created (run_loop on main, poll_loop on isolated thread)"
        )

    async def stop(self) -> None:
        """Stop the position stream gracefully."""
        if not self._running:
            return

        logger.info(f"Stopping position stream (received {self._update_count} updates)")

        self._running = False

        # Cancel main loop task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

        # Cancel isolated loop task (runs on different thread)
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            # Don't await - it's on a different loop
        self._poll_task = None

        # Close main exchange (on main loop)
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket exchange: {e}")
        self._exchange = None

        # Close poll exchange (on isolated loop)
        if self._poll_exchange and self._isolated_loop:
            try:
                await self._isolated_loop.run(self._poll_exchange.close())
            except Exception as e:
                logger.error(f"Error closing poll exchange: {e}")
        self._poll_exchange = None

        self._connected = False
        self._positions = {}
        self._main_loop = None
        logger.info("Position stream stopped")

    async def _run_loop(self) -> None:
        """Main loop watching positions."""
        retries = 0

        while self._running:
            try:
                # Watch positions - CCXT Pro handles user data stream
                positions = await self._exchange.watch_positions()

                # First successful fetch = connected
                if not self._connected:
                    self._connected = True
                    retries = 0
                    logger.info("Position stream connected")

                    if self.on_connect:
                        try:
                            await self.on_connect()
                        except Exception as e:
                            logger.error(f"on_connect error: {e}")

                # Process positions
                await self._handle_positions(positions)

            except asyncio.CancelledError:
                break

            except Exception as e:
                was_connected = self._connected
                self._connected = False

                if was_connected:
                    logger.warning(f"Position stream disconnected: {e}")

                    if self.on_disconnect:
                        try:
                            await self.on_disconnect(e)
                        except Exception as cb_err:
                            logger.error(f"on_disconnect error: {cb_err}")
                else:
                    # Log first connection failure
                    logger.warning(f"Position stream connection failed: {type(e).__name__}: {e}")

                # Check retry limit
                retries += 1
                if self.config.max_retries > 0 and retries > self.config.max_retries:
                    logger.error(f"Max retries ({self.config.max_retries}) exceeded")
                    break

                # Exponential backoff
                delay = min(2**retries, 60)
                logger.info(f"Reconnecting in {delay}s (attempt {retries})")
                await asyncio.sleep(delay)

    async def _handle_positions(self, ccxt_positions: list[dict], from_poll: bool = False) -> None:
        """Convert CCXT positions to PositionData and call callback."""
        try:
            import time

            # Filter to active positions (non-zero contracts)
            active_positions: list[PositionData] = []

            for pos in ccxt_positions:
                contracts = float(pos.get("contracts", 0) or 0)
                if contracts != 0:
                    position_data = PositionData.from_ccxt_position(pos)
                    active_positions.append(position_data)
                    self._positions[position_data.symbol] = position_data
                else:
                    # Remove closed position from cache
                    symbol = pos.get("symbol", "")
                    if symbol in self._positions:
                        del self._positions[symbol]

            self._update_count += 1
            self._last_update_time = time.time()

            if from_poll:
                logger.debug(f"Position poll: {len(active_positions)} active positions")
            else:
                logger.debug(f"Position stream update: {len(active_positions)} active positions")

            # Call callback with all active positions
            result = self.on_position(active_positions)
            if hasattr(result, "__await__"):
                await result

        except Exception as e:
            logger.error(f"Failed to handle positions: {e}")

    async def _poll_loop(self) -> None:
        """
        REST API fallback polling for position updates.

        Polls REST API regularly since WebSocket ACCOUNT_UPDATE only fires on
        position changes (open/close/modify), not on mark price changes that affect PnL.
        """
        # Wait for exchange to be ready
        while self._running and not self._exchange:
            await asyncio.sleep(1)

        if not self._running:
            return

        # Sync time with Binance servers before first poll to avoid InvalidNonce errors
        # adjustForTimeDifference only syncs on first authenticated request, so we force it
        if self._poll_exchange:
            try:
                await self._poll_exchange.load_time_difference()
                logger.debug("Position poll exchange time synced with Binance")
            except Exception as e:
                logger.warning(f"Failed to sync time for position poll: {e}")

        logger.debug(f"Position poll loop started, delay={self.config.poll_delay}s")

        # Exponential backoff state for transient errors
        consecutive_errors = 0
        max_backoff = 120  # Max 2 minutes between retries

        while self._running:
            try:
                # Wait for the polling delay
                await asyncio.sleep(self.config.poll_delay)

                if not self._running or not self._poll_exchange:
                    break

                # Poll REST API for updated positions using direct fapi endpoint
                # (fetch_positions() calls sapi endpoints that don't exist on demo)
                raw_positions = await self._poll_exchange.fapiPrivateV2GetPositionRisk()

                # Reset error counter on success
                consecutive_errors = 0

                # Parse raw positions into CCXT-like format
                positions = self._parse_position_risk(raw_positions)

                logger.debug(f"Position poll: got {len(positions)} active positions")
                await self._handle_positions(positions, from_poll=True)

            except asyncio.CancelledError:
                break
            except TRANSIENT_ERRORS as e:
                # Transient errors (network issues, exchange unavailable) - use exponential backoff
                consecutive_errors += 1
                backoff = min(2**consecutive_errors + random.random(), max_backoff)
                logger.warning(
                    f"Position poll transient error ({consecutive_errors}x): "
                    f"{type(e).__name__}, retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
            except Exception as e:
                # Non-transient errors - log and use fixed delay
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"Position poll error: {error_msg}")
                await asyncio.sleep(10)

    async def _poll_loop_isolated(self) -> None:
        """
        REST API polling for position updates - runs on ISOLATED event loop.

        This method runs on a dedicated thread with its own event loop to ensure
        position polling is never blocked by GPU operations or heavy database
        queries on the main event loop.

        Flow: poll positions -> handle positions -> run learning -> wait delay -> repeat

        Polls REST API regularly since WebSocket ACCOUNT_UPDATE only fires on
        position changes (open/close/modify), not on mark price changes that affect PnL.
        """
        # Wait for main loop to be ready
        while self._running and not self._exchange:
            await asyncio.sleep(1)

        if not self._running:
            return

        # Sync time with Binance servers before first poll to avoid InvalidNonce errors
        if self._poll_exchange:
            try:
                await self._poll_exchange.load_time_difference()
                logger.debug("Position poll exchange time synced with Binance (isolated loop)")
            except Exception as e:
                logger.warning(f"Failed to sync time for position poll: {e}")

        logger.info(
            f"Position poll loop started on isolated thread, delay={self.config.poll_delay}s"
        )

        # Exponential backoff state for transient errors
        consecutive_errors = 0
        max_backoff = 120  # Max 2 minutes between retries

        while self._running:
            try:
                if not self._running or not self._poll_exchange:
                    break

                # 1. Poll REST API for updated positions using direct fapi endpoint
                # (fetch_positions() calls sapi endpoints that don't exist on demo)
                raw_positions = await self._poll_exchange.fapiPrivateV2GetPositionRisk()

                # Reset error counter on success
                consecutive_errors = 0

                # Parse raw positions into CCXT-like format (CPU-bound, runs here)
                ccxt_positions = self._parse_position_risk(raw_positions)

                logger.debug(
                    f"Position poll (isolated): got {len(ccxt_positions)} active positions"
                )

                # 2. Convert to PositionData and update cache
                active_positions = self._update_position_cache(ccxt_positions)

                # 3. Handle positions on the main loop (for thread-safe TUI updates)
                if self._main_loop and not self._main_loop.is_closed():
                    self._main_loop.call_soon_threadsafe(
                        self._schedule_position_callback, active_positions
                    )

                # 4. Run learning callback on main loop if configured
                if self.on_learning and active_positions:
                    await self._run_learning_on_main_loop(active_positions)

                # 5. Wait delay before next poll
                await asyncio.sleep(self.config.poll_delay)

            except asyncio.CancelledError:
                break
            except TRANSIENT_ERRORS as e:
                # Transient errors (network issues, exchange unavailable) - use exponential backoff
                consecutive_errors += 1
                # Exponential backoff with jitter: 2^n + random(0, 1) seconds, capped at max_backoff
                backoff = min(2**consecutive_errors + random.random(), max_backoff)
                logger.warning(
                    f"Position poll transient error ({consecutive_errors}x): "
                    f"{type(e).__name__}, retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
            except Exception as e:
                # Non-transient errors - log and use fixed delay
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"Position poll error (isolated): {error_msg}")
                await asyncio.sleep(10)

        logger.info("Position poll loop stopped (isolated thread)")

    def _schedule_handle_positions(self, positions: list[dict]) -> None:
        """
        Schedule position handling on the main event loop.

        Called from isolated loop via call_soon_threadsafe.
        """
        if self._main_loop and not self._main_loop.is_closed():
            asyncio.ensure_future(
                self._handle_positions(positions, from_poll=True),
                loop=self._main_loop,
            )

    def _update_position_cache(self, ccxt_positions: list[dict]) -> list[PositionData]:
        """
        Convert CCXT positions to PositionData and update internal cache.

        Returns list of active positions (non-zero contracts).
        Called from isolated loop - only updates internal state.
        """
        import time

        active_positions: list[PositionData] = []

        for pos in ccxt_positions:
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts != 0:
                position_data = PositionData.from_ccxt_position(pos)
                active_positions.append(position_data)
                self._positions[position_data.symbol] = position_data
            else:
                # Remove closed position from cache
                symbol = pos.get("symbol", "")
                if symbol in self._positions:
                    del self._positions[symbol]

        self._update_count += 1
        self._last_update_time = time.time()

        return active_positions

    def _schedule_position_callback(self, positions: list[PositionData]) -> None:
        """
        Schedule the on_position callback on the main event loop.

        Called from isolated loop via call_soon_threadsafe.
        """
        if self._main_loop and not self._main_loop.is_closed():
            asyncio.ensure_future(
                self._invoke_position_callback(positions),
                loop=self._main_loop,
            )

    async def _invoke_position_callback(self, positions: list[PositionData]) -> None:
        """Invoke the on_position callback with error handling."""
        try:
            result = self.on_position(positions)
            if hasattr(result, "__await__"):
                await result
        except Exception as e:
            logger.error(f"on_position callback error: {e}")

    async def _run_learning_on_main_loop(self, positions: list[PositionData]) -> None:
        """
        Run the learning callback on the main event loop and wait for completion.

        This runs the learning callback synchronously from the isolated loop's
        perspective - it waits for learning to complete before returning.
        """
        if not self.on_learning or not self._main_loop:
            return

        # Create a future to wait for learning completion
        loop = asyncio.get_running_loop()
        learning_done = loop.create_future()

        async def run_learning() -> None:
            try:
                await self.on_learning(positions)
                # Signal completion back to isolated loop
                if not learning_done.done():
                    loop.call_soon_threadsafe(learning_done.set_result, None)
            except Exception as e:
                logger.error(f"Learning callback error: {e}")
                if not learning_done.done():
                    loop.call_soon_threadsafe(learning_done.set_result, None)

        # Schedule learning on main loop
        if not self._main_loop.is_closed():
            self._main_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(run_learning(), loop=self._main_loop)
            )

        # Wait for learning to complete (with timeout)
        try:
            await asyncio.wait_for(learning_done, timeout=300.0)  # 5 min max
            logger.debug("Learning callback completed")
        except asyncio.TimeoutError:
            logger.warning("Learning callback timed out after 5 minutes")

    def _parse_position_risk(self, raw_positions: list[dict]) -> list[dict]:
        """
        Parse raw Binance positionRisk response into CCXT unified format.

        Only returns positions with non-zero positionAmt.
        """
        positions = []
        for pos in raw_positions:
            position_amt = float(pos.get("positionAmt", 0) or 0)
            if position_amt == 0:
                continue

            # Determine side from positionAmt and positionSide
            position_side = pos.get("positionSide", "BOTH")
            if position_side == "LONG":
                side = "long"
            elif position_side == "SHORT":
                side = "short"
            else:
                # BOTH mode - determine from position amount
                side = "long" if position_amt > 0 else "short"

            entry_price = float(pos.get("entryPrice", 0) or 0)
            mark_price = float(pos.get("markPrice", 0) or 0)
            liquidation_price = float(pos.get("liquidationPrice", 0) or 0)
            unrealized_pnl = float(pos.get("unRealizedProfit", 0) or 0)
            notional = float(pos.get("notional", 0) or 0)
            leverage = int(float(pos.get("leverage", 1) or 1))
            margin_type = pos.get("marginType", "cross")
            isolated_margin = float(pos.get("isolatedMargin", 0) or 0)
            isolated_wallet = float(pos.get("isolatedWallet", 0) or 0)

            # Calculate percentage PnL
            initial_margin = (
                isolated_wallet if margin_type == "isolated" else abs(notional) / leverage
            )
            pnl_pct = (unrealized_pnl / initial_margin * 100) if initial_margin > 0 else 0

            # Convert symbol format (e.g., BTCUSDT -> BTC/USDT:USDT)
            raw_symbol = pos.get("symbol", "")
            # Simple conversion - strip USDT suffix and add format
            if raw_symbol.endswith("USDT"):
                base = raw_symbol[:-4]
                symbol = f"{base}/USDT:USDT"
            else:
                symbol = raw_symbol

            positions.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "contracts": abs(position_amt),
                    "entryPrice": entry_price,
                    "markPrice": mark_price,
                    "liquidationPrice": liquidation_price,
                    "unrealizedPnl": unrealized_pnl,
                    "percentage": pnl_pct,
                    "notional": abs(notional),
                    "leverage": leverage,
                    "marginMode": "isolated" if margin_type == "isolated" else "cross",
                    "initialMargin": initial_margin,
                    "timestamp": int(datetime.now().timestamp() * 1000),
                }
            )

        return positions

    def stats(self) -> dict:
        """Get stream statistics."""
        return {
            "is_running": self._running,
            "is_connected": self._connected,
            "update_count": self._update_count,
            "position_count": len(self._positions),
            "demo": self.config.demo,
        }

    # Context manager support
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
