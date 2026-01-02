"""
Non-blocking trade backfill for cold storage.

Downloads historical trades from Binance Public Data (CSV) and stores in QuestDB.
Uses CSV files from data.binance.vision - NO API RATE LIMITS!

URL pattern:
    https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{DATE}.zip

Uses aequify.runtime patterns for async execution.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from aequify.logging import get_logger
from aequify.runtime import run_async_fire_and_forget

from .models import Trade
from .store import ColdStore

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


# =============================================================================
# Backfill Status
# =============================================================================


class BackfillStatus(str, Enum):
    """Backfill job status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackfillProgress:
    """
    Progress tracking for a backfill job.

    Attributes:
        symbol: Trading pair being backfilled.
        status: Current job status.
        total_trades: Total trades fetched so far.
        days_processed: Number of days processed.
        total_days: Total days to process.
        current_date: Current date being processed.
        start_time_ms: When backfill started.
        end_time_ms: When backfill ended (0 if still running).
        error: Error message if failed.
    """

    symbol: str
    status: BackfillStatus = BackfillStatus.PENDING
    total_trades: int = 0
    days_processed: int = 0
    total_days: int = 0
    current_date: str = ""
    start_time_ms: int = 0
    end_time_ms: int = 0
    error: str = ""

    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        if self.end_time_ms > 0:
            return self.end_time_ms - self.start_time_ms
        if self.start_time_ms > 0:
            return int(time.time() * 1000) - self.start_time_ms
        return 0

    @property
    def trades_per_second(self) -> float:
        """Average trades fetched per second."""
        duration_s = self.duration_ms / 1000
        if duration_s > 0:
            return self.total_trades / duration_s
        return 0.0

    @property
    def progress_pct(self) -> float:
        """Progress percentage (0-100)."""
        if self.total_days > 0:
            return (self.days_processed / self.total_days) * 100
        return 0.0


# =============================================================================
# Backfill Configuration
# =============================================================================


@dataclass
class BackfillConfig:
    """
    Backfill job configuration.

    Attributes:
        lookback_days: Number of days of history to fetch.
        data_dir: Directory to cache CSV files. None = project/data/csv
        skip_existing: Skip download if CSV already exists locally.
    """

    lookback_days: int = 90
    data_dir: str | None = None
    skip_existing: bool = True


# Default configuration
DEFAULT_BACKFILL_CONFIG = BackfillConfig()


# =============================================================================
# Symbol Normalization
# =============================================================================


def normalize_symbol(symbol: str) -> tuple[str, str]:
    """
    Normalize symbol for CSV filename and DB storage.

    Args:
        symbol: Symbol in CCXT format (e.g., "BTC/USDT:USDT")

    Returns:
        Tuple of (filename_symbol, db_symbol)
        e.g., ("BTCUSDT", "BTC/USDT:USDT")
    """
    if "/" in symbol:
        # CCXT format: "BTC/USDT:USDT" -> "BTCUSDT"
        base = symbol.split("/")[0]
        filename_symbol = f"{base}USDT"
        db_symbol = symbol
    else:
        # Already normalized or simple format
        filename_symbol = symbol.replace(":", "")
        if not filename_symbol.endswith("USDT"):
            filename_symbol = f"{filename_symbol}USDT"
        # Convert back to CCXT format for DB
        base = filename_symbol.replace("USDT", "")
        db_symbol = f"{base}/USDT:USDT"

    return filename_symbol, db_symbol


# =============================================================================
# CSV Import Helper
# =============================================================================


async def _import_csv_to_store(
    csv_path: Path,
    symbol: str,
    store: ColdStore,
) -> int:
    """
    Import aggTrades CSV file to cold store.

    CSV format: agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker

    Runs CSV parsing in thread to avoid blocking.

    Args:
        csv_path: Path to CSV file.
        symbol: Symbol for DB (e.g., "BTC/USDT:USDT").
        store: ColdStore instance.

    Returns:
        Number of trades imported.
    """
    import csv as csv_module

    def _parse_csv() -> list[Trade]:
        """Parse CSV file into Trade objects."""
        trades = []

        with open(csv_path) as f:
            reader = csv_module.reader(f)
            for row in reader:
                if len(row) < 7:
                    continue

                try:
                    # agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker
                    trade_id = int(row[0])
                    price = float(row[1])
                    quantity = float(row[2])
                    timestamp_ms = int(row[5])
                    is_buyer_maker = row[6].lower() in ("true", "1")

                    trades.append(
                        Trade(
                            trade_id=trade_id,
                            symbol=symbol,
                            price=price,
                            quantity=quantity,
                            timestamp_ms=timestamp_ms,
                            is_buyer_maker=is_buyer_maker,
                        )
                    )
                except (ValueError, IndexError):
                    continue

        return trades

    # Parse CSV in thread
    trades = await asyncio.to_thread(_parse_csv)

    if not trades:
        return 0

    # Insert to cold store
    return await store.insert_trades(trades)


# =============================================================================
# Backfill Manager
# =============================================================================


class BackfillManager:
    """
    Manages background backfill jobs for multiple symbols.

    Downloads CSV files from Binance public data, caches locally,
    and imports to cold store. Runs asynchronously without blocking.

    Usage:
        manager = BackfillManager()

        # Start backfill for a symbol (runs in background)
        manager.start_backfill("BTC/USDT:USDT")

        # Check progress
        progress = manager.get_progress("BTC/USDT:USDT")
        print(f"Fetched {progress.total_trades} trades")

        # Stop backfill
        manager.stop_backfill("BTC/USDT:USDT")

        # Cleanup
        await manager.close()
    """

    def __init__(
        self,
        store: ColdStore | None = None,
        config: BackfillConfig | None = None,
    ) -> None:
        """
        Initialize backfill manager.

        Args:
            store: ColdStore instance. Creates new one if None.
            config: Backfill configuration.
        """
        self._store = store
        self._owns_store = store is None
        self._config = config or DEFAULT_BACKFILL_CONFIG
        self._progress: dict[str, BackfillProgress] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_flags: dict[str, bool] = {}
        self._callbacks: list[Callable[[str, BackfillProgress], None]] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the manager and store."""
        if self._initialized:
            return

        if self._store is None:
            self._store = ColdStore()
            await self._store.initialize()

        self._initialized = True
        logger.info("BackfillManager initialized")

    async def close(self) -> None:
        """Close the manager and cancel all running jobs."""
        # Cancel all running tasks
        for symbol in list(self._tasks.keys()):
            await self.stop_backfill(symbol)

        # Close store if we own it
        if self._owns_store and self._store:
            await self._store.close()
            self._store = None

        self._initialized = False
        logger.info("BackfillManager closed")

    async def __aenter__(self) -> BackfillManager:
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()

    def on_progress(self, callback: Callable[[str, BackfillProgress], None]) -> None:
        """
        Register callback for progress updates.

        Args:
            callback: Function(symbol, progress) called on updates.
        """
        self._callbacks.append(callback)

    def _notify_progress(self, symbol: str) -> None:
        """Notify callbacks of progress update."""
        progress = self._progress.get(symbol)
        if progress:
            for callback in self._callbacks:
                try:
                    callback(symbol, progress)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

    # =========================================================================
    # Backfill Control
    # =========================================================================

    def start_backfill(
        self,
        symbol: str,
        config: BackfillConfig | None = None,
    ) -> asyncio.Future | asyncio.Task:
        """
        Start a background backfill for a symbol.

        Non-blocking - returns immediately. Use get_progress() to monitor.

        Args:
            symbol: Trading pair (CCXT format).
            config: Override default config for this job.

        Returns:
            Task/Future that resolves when backfill completes.
        """
        if symbol in self._tasks and not self._tasks[symbol].done():
            logger.warning(f"Backfill already running for {symbol}")
            return self._tasks[symbol]

        cfg = config or self._config

        # Initialize progress
        self._progress[symbol] = BackfillProgress(
            symbol=symbol,
            status=BackfillStatus.PENDING,
            total_days=cfg.lookback_days,
        )
        self._cancel_flags[symbol] = False

        # Start background task on the current event loop if available,
        # otherwise use the main runtime loop
        try:
            loop = asyncio.get_running_loop()
            # Running in async context - use create_task on current loop
            task = loop.create_task(self._run_backfill(symbol, cfg))
            self._tasks[symbol] = task
        except RuntimeError:
            # No running loop - use main runtime loop
            future = run_async_fire_and_forget(self._run_backfill(symbol, cfg))
            self._tasks[symbol] = future

        logger.info(f"Started backfill for {symbol}")
        return self._tasks[symbol]

    async def stop_backfill(self, symbol: str) -> None:
        """
        Stop a running backfill.

        Args:
            symbol: Trading pair.
        """
        if symbol not in self._tasks:
            return

        self._cancel_flags[symbol] = True

        task = self._tasks[symbol]
        if not task.done():
            task.cancel()
            try:
                # Handle both asyncio.Task and concurrent.futures.Future
                if isinstance(task, asyncio.Task):
                    await task
                else:
                    # concurrent.futures.Future from run_async_fire_and_forget
                    await asyncio.wrap_future(task)
            except (asyncio.CancelledError, Exception):
                pass

        if symbol in self._progress:
            self._progress[symbol].status = BackfillStatus.CANCELLED
            self._progress[symbol].end_time_ms = int(time.time() * 1000)

        logger.info(f"Stopped backfill for {symbol}")

    def get_progress(self, symbol: str) -> BackfillProgress | None:
        """
        Get progress for a symbol's backfill.

        Args:
            symbol: Trading pair.

        Returns:
            BackfillProgress or None if not started.
        """
        return self._progress.get(symbol)

    def get_all_progress(self) -> dict[str, BackfillProgress]:
        """Get progress for all symbols."""
        return dict(self._progress)

    def is_running(self, symbol: str) -> bool:
        """Check if backfill is running or pending for a symbol."""
        progress = self._progress.get(symbol)
        # Include PENDING status - task is scheduled but may not have started yet
        return progress is not None and progress.status in (
            BackfillStatus.PENDING,
            BackfillStatus.RUNNING,
        )

    # =========================================================================
    # Backfill Execution
    # =========================================================================

    async def _run_backfill(self, symbol: str, config: BackfillConfig) -> None:
        """
        Execute backfill for a symbol using CSV files.

        Downloads from Binance public data, caches locally, imports to cold store.

        Args:
            symbol: Trading pair.
            config: Backfill configuration.
        """
        import io
        import zipfile
        from datetime import datetime, timedelta, timezone

        import httpx

        progress = self._progress[symbol]
        progress.status = BackfillStatus.RUNNING
        progress.start_time_ms = int(time.time() * 1000)
        self._notify_progress(symbol)

        filename_symbol, db_symbol = normalize_symbol(symbol)

        # Setup data directory
        if config.data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "csv"
        else:
            data_dir = Path(config.data_dir)

        symbol_dir = data_dir / filename_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Calculate date range (yesterday backwards)
        end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
        start_date = end_date - timedelta(days=config.lookback_days - 1)

        logger.info(f"[{symbol}] CSV backfill: {start_date} to {end_date} ({config.lookback_days} days)")

        base_url = "https://data.binance.vision/data/futures/um/daily/aggTrades"

        try:
            # Ensure manager is initialized
            if not self._initialized:
                await self.initialize()

            async with httpx.AsyncClient(timeout=120.0) as client:
                current_date = start_date

                while current_date <= end_date:
                    # Check cancellation
                    if self._cancel_flags.get(symbol, False):
                        logger.info(f"Backfill cancelled for {symbol}")
                        break

                    progress.days_processed += 1
                    date_str = current_date.strftime("%Y-%m-%d")
                    progress.current_date = date_str
                    self._notify_progress(symbol)

                    filename = f"{filename_symbol}-aggTrades-{date_str}.zip"
                    csv_filename = filename.replace(".zip", ".csv")
                    csv_path = symbol_dir / csv_filename
                    url = f"{base_url}/{filename_symbol}/{filename}"

                    # Skip if CSV already exists
                    if config.skip_existing and csv_path.exists():
                        logger.debug(f"[{symbol}] {date_str}: using cached CSV")
                        trades_imported = await _import_csv_to_store(csv_path, db_symbol, self._store)
                        progress.total_trades += trades_imported
                        current_date += timedelta(days=1)
                        continue

                    try:
                        # Download ZIP
                        response = await client.get(url)

                        if response.status_code == 404:
                            logger.debug(f"[{symbol}] {date_str}: not found (404)")
                            current_date += timedelta(days=1)
                            continue

                        response.raise_for_status()

                        # Extract CSV from ZIP
                        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                            if not csv_names:
                                logger.warning(f"[{symbol}] {date_str}: no CSV in ZIP")
                                current_date += timedelta(days=1)
                                continue

                            # Save CSV to local cache
                            with zf.open(csv_names[0]) as src:
                                csv_path.write_bytes(src.read())

                        # Import CSV to cold store
                        trades_imported = await _import_csv_to_store(csv_path, db_symbol, self._store)
                        progress.total_trades += trades_imported

                        logger.debug(f"[{symbol}] {date_str}: imported {trades_imported:,} trades")

                    except httpx.TimeoutException:
                        logger.warning(f"[{symbol}] {date_str}: timeout")
                    except httpx.HTTPStatusError as e:
                        logger.warning(f"[{symbol}] {date_str}: HTTP {e.response.status_code}")
                    except Exception as e:
                        logger.warning(f"[{symbol}] {date_str}: error - {e}")

                    current_date += timedelta(days=1)

            # Completed successfully
            progress.status = BackfillStatus.COMPLETED
            progress.end_time_ms = int(time.time() * 1000)

            logger.info(
                f"[{symbol}] Backfill complete: {progress.total_trades:,} trades "
                f"in {progress.duration_ms / 1000:.1f}s"
            )

        except asyncio.CancelledError:
            progress.status = BackfillStatus.CANCELLED
            progress.end_time_ms = int(time.time() * 1000)
            raise

        except Exception as e:
            progress.status = BackfillStatus.FAILED
            progress.error = str(e)
            progress.end_time_ms = int(time.time() * 1000)
            logger.error(f"Backfill failed for {symbol}: {e}")

        finally:
            self._notify_progress(symbol)


# =============================================================================
# Convenience Functions
# =============================================================================


def start_background_backfill(
    symbols: list[str],
    config: BackfillConfig | None = None,
    on_progress: Callable[[str, BackfillProgress], None] | None = None,
) -> BackfillManager:
    """
    Start background backfill for multiple symbols.

    Non-blocking - returns immediately after starting.

    Args:
        symbols: List of trading pairs.
        config: Backfill configuration.
        on_progress: Optional callback for progress updates.

    Returns:
        BackfillManager for monitoring and control.

    Usage:
        manager = start_background_backfill(
            ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            on_progress=lambda s, p: print(f"{s}: {p.total_trades} trades")
        )

        # ... do other work ...

        # Check progress later
        for symbol, progress in manager.get_all_progress().items():
            print(f"{symbol}: {progress.status.value}")
    """
    manager = BackfillManager(config=config)

    if on_progress:
        manager.on_progress(on_progress)

    # Start all backfills
    async def _start_all():
        await manager.initialize()
        for symbol in symbols:
            manager.start_backfill(symbol)

    run_async_fire_and_forget(_start_all())

    return manager


async def backfill_symbol(
    symbol: str,
    config: BackfillConfig | None = None,
) -> BackfillProgress:
    """
    Backfill a single symbol and wait for completion.

    Blocking - waits until backfill is complete.

    Args:
        symbol: Trading pair.
        config: Backfill configuration.

    Returns:
        Final BackfillProgress.
    """
    async with BackfillManager(config=config) as manager:
        manager.start_backfill(symbol)

        # Wait for completion
        while manager.is_running(symbol):
            await asyncio.sleep(1)

        return manager.get_progress(symbol)
