"""
QuestDB Cold Storage.

Provides persistent storage for:
- trades: Historical trade data from backfill and hot store
- bootstrap_results: APEX engine optimization results
- bootstrap_bounds: Optimized parameter bounds
- levels: Session imbalance levels

Uses ILP for high-speed ingestion and PostgreSQL for queries.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from questdb.ingress import TimestampNanos

from aequify.logging import get_logger

from .client import QuestDBClient, QuestDBConfig
from .models import (
    BootstrapBounds,
    BootstrapResult,
    Direction,
    SessionLevels,
    Trade,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


# =============================================================================
# Table Schemas
# =============================================================================

# Trades table schema
TRADES_SCHEMA = """
    symbol SYMBOL,
    trade_id LONG,
    price DOUBLE,
    quantity DOUBLE,
    notional DOUBLE,
    is_buyer_maker BOOLEAN,
    timestamp TIMESTAMP
"""

# Bootstrap results table schema
BOOTSTRAP_RESULTS_SCHEMA = """
    symbol SYMBOL,
    direction SYMBOL,
    price_move DOUBLE,
    min_dca_distance DOUBLE,
    imbalance_threshold DOUBLE,
    time_window INT,
    take_profit DOUBLE,
    stop_loss DOUBLE,
    max_hold_time INT,
    entries INT,
    winners INT,
    win_rate DOUBLE,
    avg_pnl DOUBLE,
    loss DOUBLE,
    timestamp TIMESTAMP
"""

# Bootstrap bounds table schema
BOOTSTRAP_BOUNDS_SCHEMA = """
    symbol SYMBOL,
    direction SYMBOL,
    price_move_min DOUBLE,
    price_move_max DOUBLE,
    min_dca_distance_min DOUBLE,
    min_dca_distance_max DOUBLE,
    imbalance_threshold_min DOUBLE,
    imbalance_threshold_max DOUBLE,
    take_profit_min DOUBLE,
    take_profit_max DOUBLE,
    stop_loss_min DOUBLE,
    stop_loss_max DOUBLE,
    max_hold_time_min INT,
    max_hold_time_max INT,
    timestamp TIMESTAMP
"""

# Session levels table schema
LEVELS_SCHEMA = """
    symbol SYMBOL,
    direction SYMBOL,
    session_date STRING,
    session_name STRING,
    level_index INT,
    price DOUBLE,
    volume DOUBLE,
    delta_pct DOUBLE,
    imbalance_pct DOUBLE,
    filled BOOLEAN,
    fill_time_ms LONG,
    used BOOLEAN,
    timestamp TIMESTAMP
"""


# =============================================================================
# Cold Store
# =============================================================================


class ColdStore:
    """
    QuestDB cold storage for historical data.

    Provides async interface for:
    - Inserting trades (with deduplication)
    - Storing bootstrap results and bounds
    - Storing session levels
    - Querying historical data

    Usage:
        async with ColdStore() as store:
            # Insert trades
            await store.insert_trades(trades)

            # Query trades
            trades = await store.get_trades("BTC/USDT:USDT", limit=1000)

            # Store bootstrap result
            await store.insert_bootstrap_result(result)
    """

    # Table names
    TABLE_TRADES = "trades"
    TABLE_BOOTSTRAP_RESULTS = "bootstrap_results"
    TABLE_BOOTSTRAP_BOUNDS = "bootstrap_bounds"
    TABLE_LEVELS = "levels"

    def __init__(self, config: QuestDBConfig | None = None) -> None:
        """
        Initialize cold store.

        Args:
            config: QuestDB configuration. Uses defaults if None.
        """
        self._client = QuestDBClient(config)
        self._tables_initialized = False

    async def initialize(self) -> None:
        """Initialize client and create tables."""
        await self._client.initialize()
        await self._ensure_tables()
        logger.info("ColdStore initialized")

    async def close(self) -> None:
        """Close client connection."""
        await self._client.close()
        logger.info("ColdStore closed")

    async def __aenter__(self) -> ColdStore:
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()

    async def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        if self._tables_initialized:
            return

        await self._client.create_table_if_not_exists(
            self.TABLE_TRADES, TRADES_SCHEMA, partition_by="DAY"
        )
        await self._client.create_table_if_not_exists(
            self.TABLE_BOOTSTRAP_RESULTS, BOOTSTRAP_RESULTS_SCHEMA, partition_by="MONTH"
        )
        await self._client.create_table_if_not_exists(
            self.TABLE_BOOTSTRAP_BOUNDS, BOOTSTRAP_BOUNDS_SCHEMA, partition_by="MONTH"
        )
        await self._client.create_table_if_not_exists(
            self.TABLE_LEVELS, LEVELS_SCHEMA, partition_by="DAY"
        )

        self._tables_initialized = True

    # =========================================================================
    # Trades
    # =========================================================================

    async def insert_trades(self, trades: Sequence[Trade]) -> int:
        """
        Insert trades using ILP (high-speed ingestion).

        Automatically deduplicates by trade_id within the batch.
        QuestDB handles timestamp-based deduplication.

        Args:
            trades: Sequence of Trade objects.

        Returns:
            Number of trades inserted.
        """
        if not trades:
            return 0

        # Deduplicate by trade_id within batch
        seen_ids: set[int] = set()
        unique_trades = []
        for trade in trades:
            if trade.trade_id not in seen_ids:
                seen_ids.add(trade.trade_id)
                unique_trades.append(trade)

        # Insert via ILP
        with self._client.create_sender() as sender:
            for trade in unique_trades:
                sender.row(
                    self.TABLE_TRADES,
                    symbols={"symbol": trade.symbol},
                    columns=trade.to_ilp_dict(),
                    at=TimestampNanos(trade.timestamp_ns),
                )
            sender.flush()

        logger.debug(f"Inserted {len(unique_trades)} trades")
        return len(unique_trades)

    async def get_trades(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 10000,
    ) -> list[Trade]:
        """
        Get trades for a symbol within time range.

        Args:
            symbol: Trading pair.
            start_ms: Start timestamp (ms). None = no lower bound.
            end_ms: End timestamp (ms). None = no upper bound.
            limit: Maximum number of trades to return.

        Returns:
            List of Trade objects.
        """
        conditions = ["symbol = $1"]
        params: list[Any] = [symbol]

        if start_ms is not None:
            conditions.append("timestamp >= $2")
            params.append(start_ms * 1_000_000)  # Convert to ns for comparison

        if end_ms is not None:
            idx = len(params) + 1
            conditions.append(f"timestamp <= ${idx}")
            params.append(end_ms * 1_000_000)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT symbol, trade_id, price, quantity, timestamp, is_buyer_maker
            FROM {self.TABLE_TRADES}
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """

        rows = await self._client.query(sql, *params)

        trades = []
        for row in rows:
            # QuestDB returns timestamp as datetime, convert to ms
            ts = row["timestamp"]
            if hasattr(ts, "timestamp"):
                ts_ms = int(ts.timestamp() * 1000)
            else:
                ts_ms = int(ts)

            trades.append(
                Trade(
                    trade_id=int(row["trade_id"]),
                    symbol=str(row["symbol"]),
                    price=float(row["price"]),
                    quantity=float(row["quantity"]),
                    timestamp_ms=ts_ms,
                    is_buyer_maker=bool(row["is_buyer_maker"]),
                )
            )

        return trades

    async def get_latest_trade_id(self, symbol: str) -> int | None:
        """
        Get the latest trade_id for a symbol.

        Useful for knowing where to resume backfill.

        Args:
            symbol: Trading pair.

        Returns:
            Latest trade_id or None if no trades exist.
        """
        sql = f"""
            SELECT MAX(trade_id) as max_id
            FROM {self.TABLE_TRADES}
            WHERE symbol = $1
        """
        row = await self._client.query_one(sql, symbol)
        if row and row.get("max_id") is not None:
            return int(row["max_id"])
        return None

    async def get_trade_count(self, symbol: str) -> int:
        """
        Get total trade count for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            Number of trades.
        """
        sql = f"""
            SELECT COUNT(*) as count
            FROM {self.TABLE_TRADES}
            WHERE symbol = $1
        """
        row = await self._client.query_one(sql, symbol)
        return int(row["count"]) if row else 0

    async def get_timestamp_range(self, symbol: str) -> tuple[int, int] | None:
        """
        Get the min and max timestamps for a symbol's trades.

        Args:
            symbol: Trading pair.

        Returns:
            Tuple of (min_timestamp_ms, max_timestamp_ms) or None if no trades.
        """
        sql = f"""
            SELECT MIN(timestamp_ms) as min_ts, MAX(timestamp_ms) as max_ts
            FROM {self.TABLE_TRADES}
            WHERE symbol = $1
        """
        row = await self._client.query_one(sql, symbol)
        if row and row.get("min_ts") is not None and row.get("max_ts") is not None:
            return (int(row["min_ts"]), int(row["max_ts"]))
        return None

    async def trade_exists(self, symbol: str, trade_id: int) -> bool:
        """
        Check if a trade already exists.

        Args:
            symbol: Trading pair.
            trade_id: Trade ID to check.

        Returns:
            True if trade exists.
        """
        sql = f"""
            SELECT 1
            FROM {self.TABLE_TRADES}
            WHERE symbol = $1 AND trade_id = $2
            LIMIT 1
        """
        row = await self._client.query_one(sql, symbol, trade_id)
        return row is not None

    # =========================================================================
    # Bootstrap Results
    # =========================================================================

    async def insert_bootstrap_result(self, result: BootstrapResult) -> None:
        """
        Insert a bootstrap result.

        Args:
            result: BootstrapResult object.
        """
        timestamp_ns = result.timestamp_ns or (int(time.time() * 1000) * 1_000_000)

        with self._client.create_sender() as sender:
            sender.row(
                self.TABLE_BOOTSTRAP_RESULTS,
                symbols={
                    "symbol": result.symbol,
                    "direction": result.direction.value,
                },
                columns=result.to_ilp_dict(),
                at=TimestampNanos(timestamp_ns),
            )
            sender.flush()

        logger.debug(f"Inserted bootstrap result for {result.symbol}/{result.direction.value}")

    async def get_latest_bootstrap_result(
        self,
        symbol: str,
        direction: Direction,
    ) -> BootstrapResult | None:
        """
        Get the latest bootstrap result for a symbol/direction.

        Args:
            symbol: Trading pair.
            direction: Trading direction.

        Returns:
            BootstrapResult or None if not found.
        """
        sql = f"""
            SELECT *
            FROM {self.TABLE_BOOTSTRAP_RESULTS}
            WHERE symbol = $1 AND direction = $2
            ORDER BY timestamp DESC
            LIMIT 1
        """
        row = await self._client.query_one(sql, symbol, direction.value)
        if not row:
            return None

        ts = row["timestamp"]
        if hasattr(ts, "timestamp"):
            ts_ms = int(ts.timestamp() * 1000)
        else:
            ts_ms = int(ts)

        return BootstrapResult(
            symbol=str(row["symbol"]),
            direction=direction,
            price_move=float(row["price_move"]),
            min_dca_distance=float(row["min_dca_distance"]),
            imbalance_threshold=float(row["imbalance_threshold"]),
            time_window=int(row["time_window"]),
            take_profit=float(row["take_profit"]),
            stop_loss=float(row["stop_loss"]),
            max_hold_time=int(row["max_hold_time"]),
            entries=int(row["entries"]),
            winners=int(row["winners"]),
            win_rate=float(row["win_rate"]),
            avg_pnl=float(row["avg_pnl"]),
            loss=float(row["loss"]),
            timestamp_ms=ts_ms,
        )

    async def get_bootstrap_results_history(
        self,
        symbol: str,
        direction: Direction,
        limit: int = 100,
    ) -> list[BootstrapResult]:
        """
        Get bootstrap results history for a symbol/direction.

        Args:
            symbol: Trading pair.
            direction: Trading direction.
            limit: Maximum number of results.

        Returns:
            List of BootstrapResult objects (newest first).
        """
        sql = f"""
            SELECT *
            FROM {self.TABLE_BOOTSTRAP_RESULTS}
            WHERE symbol = $1 AND direction = $2
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        rows = await self._client.query(sql, symbol, direction.value)

        results = []
        for row in rows:
            ts = row["timestamp"]
            if hasattr(ts, "timestamp"):
                ts_ms = int(ts.timestamp() * 1000)
            else:
                ts_ms = int(ts)

            results.append(
                BootstrapResult(
                    symbol=str(row["symbol"]),
                    direction=direction,
                    price_move=float(row["price_move"]),
                    min_dca_distance=float(row["min_dca_distance"]),
                    imbalance_threshold=float(row["imbalance_threshold"]),
                    time_window=int(row["time_window"]),
                    take_profit=float(row["take_profit"]),
                    stop_loss=float(row["stop_loss"]),
                    max_hold_time=int(row["max_hold_time"]),
                    entries=int(row["entries"]),
                    winners=int(row["winners"]),
                    win_rate=float(row["win_rate"]),
                    avg_pnl=float(row["avg_pnl"]),
                    loss=float(row["loss"]),
                    timestamp_ms=ts_ms,
                )
            )

        return results

    # =========================================================================
    # Bootstrap Bounds
    # =========================================================================

    async def insert_bootstrap_bounds(self, bounds: BootstrapBounds) -> None:
        """
        Insert bootstrap bounds.

        Args:
            bounds: BootstrapBounds object.
        """
        timestamp_ns = bounds.timestamp_ns or (int(time.time() * 1000) * 1_000_000)

        with self._client.create_sender() as sender:
            sender.row(
                self.TABLE_BOOTSTRAP_BOUNDS,
                symbols={
                    "symbol": bounds.symbol,
                    "direction": bounds.direction.value,
                },
                columns=bounds.to_ilp_dict(),
                at=TimestampNanos(timestamp_ns),
            )
            sender.flush()

        logger.debug(f"Inserted bootstrap bounds for {bounds.symbol}/{bounds.direction.value}")

    async def get_latest_bootstrap_bounds(
        self,
        symbol: str,
        direction: Direction,
    ) -> BootstrapBounds | None:
        """
        Get the latest bootstrap bounds for a symbol/direction.

        Args:
            symbol: Trading pair.
            direction: Trading direction.

        Returns:
            BootstrapBounds or None if not found.
        """
        sql = f"""
            SELECT *
            FROM {self.TABLE_BOOTSTRAP_BOUNDS}
            WHERE symbol = $1 AND direction = $2
            ORDER BY timestamp DESC
            LIMIT 1
        """
        row = await self._client.query_one(sql, symbol, direction.value)
        if not row:
            return None

        ts = row["timestamp"]
        if hasattr(ts, "timestamp"):
            ts_ms = int(ts.timestamp() * 1000)
        else:
            ts_ms = int(ts)

        return BootstrapBounds(
            symbol=str(row["symbol"]),
            direction=direction,
            optimized_bound_price_move=(
                float(row["price_move_min"]),
                float(row["price_move_max"]),
            ),
            optimized_bound_min_dca_distance=(
                float(row["min_dca_distance_min"]),
                float(row["min_dca_distance_max"]),
            ),
            optimized_bound_imbalance_threshold=(
                float(row["imbalance_threshold_min"]),
                float(row["imbalance_threshold_max"]),
            ),
            optimized_bound_take_profit=(
                float(row["take_profit_min"]),
                float(row["take_profit_max"]),
            ),
            optimized_bound_stop_loss=(
                float(row["stop_loss_min"]),
                float(row["stop_loss_max"]),
            ),
            optimized_bound_max_hold_time=(
                int(row["max_hold_time_min"]),
                int(row["max_hold_time_max"]),
            ),
            timestamp_ms=ts_ms,
        )

    # =========================================================================
    # Session Levels
    # =========================================================================

    async def insert_session_levels(self, levels: SessionLevels) -> None:
        """
        Insert session levels.

        Args:
            levels: SessionLevels object.
        """
        if not levels.imbalance_levels:
            return

        timestamp_ns = levels.timestamp_ns or (int(time.time() * 1000) * 1_000_000)

        with self._client.create_sender() as sender:
            for level_dict in levels.to_level_dicts():
                sender.row(
                    self.TABLE_LEVELS,
                    symbols={
                        "symbol": levels.symbol,
                        "direction": levels.direction.value,
                    },
                    columns={
                        "session_date": levels.session_date,
                        "session_name": levels.session_name,
                        **level_dict,
                    },
                    at=TimestampNanos(timestamp_ns),
                )
            sender.flush()

        logger.debug(
            f"Inserted {len(levels.imbalance_levels)} levels for "
            f"{levels.symbol}/{levels.direction.value}/{levels.session_name}"
        )

    async def get_session_levels(
        self,
        symbol: str,
        direction: Direction,
        session_date: str | None = None,
        session_name: str | None = None,
    ) -> list[SessionLevels]:
        """
        Get session levels for a symbol/direction.

        Args:
            symbol: Trading pair.
            direction: Trading direction.
            session_date: Optional filter by date.
            session_name: Optional filter by session name.

        Returns:
            List of SessionLevels objects.
        """
        conditions = ["symbol = $1", "direction = $2"]
        params: list[Any] = [symbol, direction.value]

        if session_date:
            conditions.append(f"session_date = ${len(params) + 1}")
            params.append(session_date)

        if session_name:
            conditions.append(f"session_name = ${len(params) + 1}")
            params.append(session_name)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT *
            FROM {self.TABLE_LEVELS}
            WHERE {where}
            ORDER BY session_date DESC, session_name, level_index
        """

        rows = await self._client.query(sql, *params)
        if not rows:
            return []

        # Group by session_date + session_name
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            key = (row["session_date"], row["session_name"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(row)

        # Convert to SessionLevels
        results = []
        for level_rows in grouped.values():
            session = SessionLevels.from_rows(level_rows)
            if session:
                results.append(session)

        return results

    async def get_unfilled_levels(
        self,
        symbol: str,
        direction: Direction,
    ) -> list[SessionLevels]:
        """
        Get unfilled session levels for a symbol/direction.

        Unfilled levels are those where price hasn't returned and not yet used.

        Args:
            symbol: Trading pair.
            direction: Trading direction.

        Returns:
            List of SessionLevels with only unfilled levels.
        """
        sql = f"""
            SELECT *
            FROM {self.TABLE_LEVELS}
            WHERE symbol = $1
              AND direction = $2
              AND filled = false
              AND used = false
            ORDER BY session_date DESC, session_name, level_index
        """

        rows = await self._client.query(sql, symbol, direction.value)
        if not rows:
            return []

        # Group and convert
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            key = (row["session_date"], row["session_name"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(row)

        results = []
        for level_rows in grouped.values():
            session = SessionLevels.from_rows(level_rows)
            if session:
                results.append(session)

        return results


# =============================================================================
# Sync Store Wrapper
# =============================================================================


class SyncColdStore:
    """
    Synchronous wrapper for ColdStore.

    Uses aequify.runtime.IsolatedLoop to run async operations from sync code.

    Usage:
        from aequify.core.db.cold import SyncColdStore

        store = SyncColdStore()
        store.start()

        try:
            trades = store.get_trades("BTC/USDT:USDT", limit=1000)
        finally:
            store.stop()
    """

    def __init__(
        self,
        config: QuestDBConfig | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize sync store wrapper.

        Args:
            config: QuestDB configuration.
            timeout: Default timeout for operations in seconds.
        """
        from aequify.runtime import IsolatedLoop

        self._config = config
        self._timeout = timeout
        self._loop = IsolatedLoop("aeq-cold-store")
        self._store: ColdStore | None = None

    def start(self) -> None:
        """Start the store and its isolated event loop."""
        self._loop.start()
        self._store = ColdStore(self._config)
        self._loop.run(self._store.initialize(), timeout=self._timeout)
        logger.info("SyncColdStore started")

    def stop(self) -> None:
        """Stop the store and its isolated event loop."""
        if self._store:
            try:
                self._loop.run(self._store.close(), timeout=self._timeout)
            except Exception as e:
                logger.warning(f"Error closing store: {e}")
            self._store = None
        self._loop.stop()
        logger.info("SyncColdStore stopped")

    def __enter__(self) -> SyncColdStore:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def _ensure_running(self) -> ColdStore:
        """Ensure store is running."""
        if not self._store or not self._loop.is_running:
            raise RuntimeError("Store not started. Call start() first.")
        return self._store

    # =========================================================================
    # Trades
    # =========================================================================

    def insert_trades(self, trades: Sequence[Trade]) -> int:
        """Insert trades."""
        store = self._ensure_running()
        return self._loop.run(store.insert_trades(trades), timeout=self._timeout)

    def get_trades(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 10000,
    ) -> list[Trade]:
        """Get trades for a symbol."""
        store = self._ensure_running()
        return self._loop.run(
            store.get_trades(symbol, start_ms, end_ms, limit),
            timeout=self._timeout,
        )

    def get_latest_trade_id(self, symbol: str) -> int | None:
        """Get latest trade_id for a symbol."""
        store = self._ensure_running()
        return self._loop.run(store.get_latest_trade_id(symbol), timeout=self._timeout)

    def get_trade_count(self, symbol: str) -> int:
        """Get trade count for a symbol."""
        store = self._ensure_running()
        return self._loop.run(store.get_trade_count(symbol), timeout=self._timeout)

    def get_timestamp_range(self, symbol: str) -> tuple[int, int] | None:
        """Get min/max timestamps for a symbol's trades."""
        store = self._ensure_running()
        return self._loop.run(store.get_timestamp_range(symbol), timeout=self._timeout)

    # =========================================================================
    # Bootstrap Results
    # =========================================================================

    def insert_bootstrap_result(self, result: BootstrapResult) -> None:
        """Insert a bootstrap result."""
        store = self._ensure_running()
        self._loop.run(store.insert_bootstrap_result(result), timeout=self._timeout)

    def get_latest_bootstrap_result(
        self,
        symbol: str,
        direction: Direction,
    ) -> BootstrapResult | None:
        """Get latest bootstrap result."""
        store = self._ensure_running()
        return self._loop.run(
            store.get_latest_bootstrap_result(symbol, direction),
            timeout=self._timeout,
        )

    # =========================================================================
    # Bootstrap Bounds
    # =========================================================================

    def insert_bootstrap_bounds(self, bounds: BootstrapBounds) -> None:
        """Insert bootstrap bounds."""
        store = self._ensure_running()
        self._loop.run(store.insert_bootstrap_bounds(bounds), timeout=self._timeout)

    def get_latest_bootstrap_bounds(
        self,
        symbol: str,
        direction: Direction,
    ) -> BootstrapBounds | None:
        """Get latest bootstrap bounds."""
        store = self._ensure_running()
        return self._loop.run(
            store.get_latest_bootstrap_bounds(symbol, direction),
            timeout=self._timeout,
        )

    # =========================================================================
    # Session Levels
    # =========================================================================

    def insert_session_levels(self, levels: SessionLevels) -> None:
        """Insert session levels."""
        store = self._ensure_running()
        self._loop.run(store.insert_session_levels(levels), timeout=self._timeout)

    def get_session_levels(
        self,
        symbol: str,
        direction: Direction,
        session_date: str | None = None,
        session_name: str | None = None,
    ) -> list[SessionLevels]:
        """Get session levels."""
        store = self._ensure_running()
        return self._loop.run(
            store.get_session_levels(symbol, direction, session_date, session_name),
            timeout=self._timeout,
        )

    def get_unfilled_levels(
        self,
        symbol: str,
        direction: Direction,
    ) -> list[SessionLevels]:
        """Get unfilled session levels."""
        store = self._ensure_running()
        return self._loop.run(
            store.get_unfilled_levels(symbol, direction),
            timeout=self._timeout,
        )
