"""
Hot Store Flush Manager - Flush trades to cold store at canonical time windows.

Trades are buffered in hot store (Mojo) and flushed to cold store (QuestDB)
at synchronized time boundaries.

Canonical Time Windows:
- Aligned to clock time (e.g., for 15min: 00:00, 00:15, 00:30, 00:45)
- Configured via database.hot.flush_window_ms in config.yaml
- Default: 900000ms (15 minutes)

Flow:
    Trade Stream → HotStore (buffer) → [canonical window] → ColdStore (persist)

Usage:
    from aequify.core.db.hot.flush import FlushManager

    # Create manager with hot store registry
    manager = FlushManager(registry, flush_window_ms=900000)

    # Start background flush loop
    await manager.start()

    # Manual flush (for testing)
    await manager.flush_all()

    # Stop
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.db.cold import ColdStore
    from aequify.core.db.cold.models import Trade

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

MS_PER_MINUTE = 60_000
MS_PER_HOUR = 3_600_000
MS_PER_DAY = 86_400_000
MINUTES_PER_DAY = 1440


# =============================================================================
# Flush Config
# =============================================================================


@dataclass
class FlushConfig:
    """Configuration for flush manager."""

    flush_window_ms: int = 900_000  # 15 minutes default
    enabled: bool = True
    max_trades_per_flush: int = 100_000  # Safety limit

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FlushConfig:
        """Create from config dict."""
        db_config = config.get("database", {})
        hot_config = db_config.get("hot", {})

        return cls(
            flush_window_ms=hot_config.get("flush_window_ms", 900_000),
            enabled=hot_config.get("flush_enabled", True),
        )

    @property
    def flush_window_minutes(self) -> int:
        """Flush window in minutes."""
        return self.flush_window_ms // MS_PER_MINUTE


# =============================================================================
# Flush Stats
# =============================================================================


@dataclass
class FlushStats:
    """Statistics for flush operations."""

    total_flushes: int = 0
    total_trades_flushed: int = 0
    last_flush_time: float = 0.0
    last_flush_count: int = 0
    flush_errors: int = 0
    symbols_flushed: dict[str, int] = field(default_factory=dict)

    def record_flush(self, symbol: str, count: int) -> None:
        """Record a successful flush."""
        self.total_flushes += 1
        self.total_trades_flushed += count
        self.last_flush_time = time.time()
        self.last_flush_count = count
        self.symbols_flushed[symbol] = self.symbols_flushed.get(symbol, 0) + count

    def record_error(self) -> None:
        """Record a flush error."""
        self.flush_errors += 1


# =============================================================================
# Time Utilities
# =============================================================================


def get_current_minute_of_day() -> int:
    """Get current minute of day (0-1439) in UTC."""
    now = datetime.now(timezone.utc)
    return now.hour * 60 + now.minute


def get_next_canonical_minute(window_minutes: int) -> int:
    """
    Get the next canonical flush minute.

    For window_minutes=15, if current is 10:07, next is 10:15 (minute 615).

    Args:
        window_minutes: Size of flush window in minutes.

    Returns:
        Minute of day for next flush (0-1439).
    """
    current = get_current_minute_of_day()
    # Round up to next window boundary
    next_window = ((current // window_minutes) + 1) * window_minutes
    # Handle day rollover
    return next_window % MINUTES_PER_DAY


def get_previous_canonical_minute(window_minutes: int) -> int:
    """
    Get the previous canonical minute boundary.

    For window_minutes=15, if current is 10:07, previous is 10:00 (minute 600).

    Args:
        window_minutes: Size of flush window in minutes.

    Returns:
        Minute of day for previous boundary (0-1439).
    """
    current = get_current_minute_of_day()
    return (current // window_minutes) * window_minutes


def seconds_until_next_canonical(window_minutes: int) -> float:
    """
    Get seconds until next canonical flush time.

    Args:
        window_minutes: Size of flush window in minutes.

    Returns:
        Seconds until next flush.
    """
    now = datetime.now(timezone.utc)
    current_minute = now.hour * 60 + now.minute
    current_second = now.second + now.microsecond / 1_000_000

    next_minute = get_next_canonical_minute(window_minutes)

    # Handle day rollover
    if next_minute <= current_minute:
        minutes_until = (MINUTES_PER_DAY - current_minute) + next_minute
    else:
        minutes_until = next_minute - current_minute

    # Convert to seconds, accounting for current position within minute
    return (minutes_until * 60) - current_second


# =============================================================================
# Flush Manager
# =============================================================================


class FlushManager:
    """
    Manages flushing trades from hot store to cold store at canonical intervals.

    Runs a background task that waits for canonical time boundaries and
    flushes all accumulated trades for each symbol.
    """

    def __init__(
        self,
        hot_registry: Any,  # HotStoreRegistry from Mojo bridge
        cold_store: "ColdStore",
        config: FlushConfig | None = None,
    ) -> None:
        """
        Initialize flush manager.

        Args:
            hot_registry: Mojo HotStoreRegistry instance.
            cold_store: ColdStore instance for persistence.
            config: Flush configuration.
        """
        self._hot = hot_registry
        self._cold = cold_store
        self._config = config or FlushConfig()
        self._stats = FlushStats()
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def stats(self) -> FlushStats:
        """Get flush statistics."""
        return self._stats

    @property
    def config(self) -> FlushConfig:
        """Get flush configuration."""
        return self._config

    async def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            return

        if not self._config.enabled:
            logger.info("FlushManager disabled by config")
            return

        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(
            f"FlushManager started (window={self._config.flush_window_minutes}m)"
        )

    async def stop(self) -> None:
        """Stop the background flush loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("FlushManager stopped")

    async def _flush_loop(self) -> None:
        """Background loop that flushes at canonical times."""
        window_minutes = self._config.flush_window_minutes

        while self._running:
            try:
                # Wait until next canonical time
                wait_seconds = seconds_until_next_canonical(window_minutes)
                logger.debug(f"Next flush in {wait_seconds:.1f}s")

                await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                # Flush all symbols
                await self.flush_all()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")
                self._stats.record_error()
                # Wait a bit before retrying
                await asyncio.sleep(60)

    async def flush_all(self) -> dict[str, int]:
        """
        Flush all symbols from hot store to cold store.

        Returns:
            Dict of symbol -> trades flushed count.
        """
        from aequify.core.db.cold.models import Trade

        results: dict[str, int] = {}
        window_minutes = self._config.flush_window_minutes

        # Get the window to flush (previous canonical window)
        end_minute = get_previous_canonical_minute(window_minutes)
        start_minute = end_minute - window_minutes
        if start_minute < 0:
            start_minute += MINUTES_PER_DAY

        logger.debug(
            f"Flushing window: minutes {start_minute}-{end_minute} "
            f"({start_minute // 60:02d}:{start_minute % 60:02d} - "
            f"{end_minute // 60:02d}:{end_minute % 60:02d})"
        )

        # Get all symbols
        try:
            symbols = list(self._hot.list_symbols())
        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
            return results

        # Flush each symbol
        for symbol in symbols:
            try:
                count = await self._flush_symbol(symbol, start_minute, end_minute)
                if count > 0:
                    results[symbol] = count
                    self._stats.record_flush(symbol, count)
            except Exception as e:
                logger.error(f"Failed to flush {symbol}: {e}")
                self._stats.record_error()

        if results:
            total = sum(results.values())
            logger.info(f"Flushed {total} trades from {len(results)} symbols")

        return results

    async def _flush_symbol(
        self,
        symbol: str,
        start_minute: int,
        end_minute: int,
    ) -> int:
        """
        Flush a single symbol's trades to cold store.

        Args:
            symbol: Trading pair.
            start_minute: Start of window (minute of day).
            end_minute: End of window (minute of day).

        Returns:
            Number of trades flushed.
        """
        from aequify.core.db.cold.models import Trade

        # Use flush_window which gets trades and clears them
        result = self._hot.flush_window(symbol, start_minute, end_minute)
        trades_data = result.get("trades", [])
        count = result.get("count", 0)

        if count == 0:
            return 0

        # Convert to Trade objects
        trades = []
        for t in trades_data:
            trades.append(
                Trade(
                    trade_id=int(t["trade_id"]),
                    symbol=symbol,
                    price=float(t["price"]),
                    quantity=float(t["quantity"]),
                    timestamp_ms=int(t["timestamp_ms"]),
                    is_buyer_maker=bool(t["is_buyer_maker"]),
                )
            )

        # Insert to cold store
        if trades:
            inserted = await self._cold.insert_trades(trades)
            logger.debug(f"Flushed {inserted} trades for {symbol}")
            return inserted

        return 0

    async def flush_symbol_now(self, symbol: str) -> int:
        """
        Immediately flush a symbol (all available data).

        For manual/testing use.

        Args:
            symbol: Trading pair to flush.

        Returns:
            Number of trades flushed.
        """
        # Flush entire day range
        return await self._flush_symbol(symbol, 0, MINUTES_PER_DAY)


# =============================================================================
# Factory Function
# =============================================================================


async def create_flush_manager(
    hot_registry: Any,
    cold_store: "ColdStore",
    config_path: str = "config/config.yaml",
) -> FlushManager:
    """
    Create a flush manager from config file.

    Args:
        hot_registry: Mojo HotStoreRegistry instance.
        cold_store: ColdStore instance.
        config_path: Path to config.yaml.

    Returns:
        Configured FlushManager.
    """
    import yaml

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    flush_config = FlushConfig.from_config(config_data)
    return FlushManager(hot_registry, cold_store, flush_config)
