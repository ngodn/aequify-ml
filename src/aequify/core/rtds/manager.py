"""
Stream Manager - Spawns and manages one process per symbol stream.

Each symbol runs in its own process for:
- Isolation: One crash doesn't affect others
- Modularity: Easy to add/remove symbols at runtime
- Future: Different stream types per symbol (trade, kline, depth)

Uses multiprocessing with shared memory queues for trade data.

Usage:
------

from aequify.rtds import StreamManager

manager = StreamManager(
    symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
    on_trade=store.add_trade,
)

await manager.start()
# ... runs until stopped
await manager.stop()

# Add/remove symbols at runtime
await manager.add_symbol("SOL/USDT:USDT")
await manager.remove_symbol("ETH/USDT:USDT")
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import signal
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any, Callable, Coroutine

from aequify.db.models import Trade
from aequify.rtds.streams import BinanceFuturesTradeStream, StreamConfig

logger = logging.getLogger(__name__)


def _run_stream_process(
    symbol: str,
    trade_queue: Queue,
    stop_event: mp.Event,
    config_dict: dict,
) -> None:
    """
    Entry point for stream subprocess.

    Runs in a separate process, sends trades via queue.
    """
    import signal

    from aequify.loop import run_with_uvloop

    # Ignore SIGINT in child process (parent handles it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    async def run():
        config = StreamConfig(
            demo=config_dict.get("demo", False),
            max_retries=config_dict.get("max_retries", 0),
        )

        def on_trade(trade: Trade):
            # Send trade to parent via queue
            trade_queue.put(
                {
                    "symbol": trade.symbol,
                    "trade_id": trade.trade_id,
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "timestamp_ms": trade.timestamp_ms,
                    "is_buyer_maker": trade.is_buyer_maker,
                }
            )

        stream = BinanceFuturesTradeStream(
            symbol=symbol,
            on_trade=on_trade,
            config=config,
        )

        await stream.start()

        # Wait for stop signal
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        await stream.stop()

    try:
        # Use uvloop in subprocess
        run_with_uvloop(run())
    except Exception as e:
        logger.error(f"[{symbol}] Process error: {e}")


@dataclass
class SymbolProcess:
    """Tracks a single symbol's process."""

    symbol: str
    process: mp.Process
    stop_event: mp.Event
    trade_queue: Queue


@dataclass
class StreamManager:
    """
    Manages multiple symbol streams, one process per symbol.

    Attributes:
        symbols: Initial list of symbols to stream.
        on_trade: Callback for each trade (called in main process).
        config: Stream configuration for all streams.
    """

    symbols: list[str]
    on_trade: Callable[[Trade], None] | Callable[[Trade], Coroutine[Any, Any, None]]
    config: StreamConfig = field(default_factory=StreamConfig)

    # Internal state
    _processes: dict[str, SymbolProcess] = field(default_factory=dict, init=False)
    _trade_queue: Queue = field(default_factory=Queue, init=False)
    _consumer_task: asyncio.Task | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)

    async def start(self) -> None:
        """Start all symbol streams."""
        if self._running:
            logger.warning("StreamManager already running")
            return

        logger.info(f"Starting StreamManager with {len(self.symbols)} symbols")

        self._running = True

        # Start consumer task (processes trade queue)
        self._consumer_task = asyncio.create_task(self._consume_trades())

        # Start all symbol processes
        for symbol in self.symbols:
            await self._start_symbol(symbol)

        logger.info(f"StreamManager started: {list(self._processes.keys())}")

    async def stop(self) -> None:
        """Stop all symbol streams gracefully."""
        if not self._running:
            return

        logger.info("Stopping StreamManager...")
        self._running = False

        # Stop all processes
        symbols = list(self._processes.keys())
        for symbol in symbols:
            await self._stop_symbol(symbol)

        # Stop consumer task
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        logger.info("StreamManager stopped")

    async def add_symbol(self, symbol: str) -> bool:
        """
        Add a new symbol stream at runtime.

        Args:
            symbol: Symbol to add.

        Returns:
            True if added, False if already exists.
        """
        if symbol in self._processes:
            logger.warning(f"Symbol {symbol} already streaming")
            return False

        await self._start_symbol(symbol)
        logger.info(f"Added symbol: {symbol}")
        return True

    async def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol stream at runtime.

        Args:
            symbol: Symbol to remove.

        Returns:
            True if removed, False if not found.
        """
        if symbol not in self._processes:
            logger.warning(f"Symbol {symbol} not found")
            return False

        await self._stop_symbol(symbol)
        logger.info(f"Removed symbol: {symbol}")
        return True

    async def _start_symbol(self, symbol: str) -> None:
        """Start a single symbol's process."""
        stop_event = mp.Event()

        config_dict = {
            "demo": self.config.demo,
            "max_retries": self.config.max_retries,
        }

        process = mp.Process(
            target=_run_stream_process,
            args=(symbol, self._trade_queue, stop_event, config_dict),
            name=f"stream-{symbol.replace('/', '-').replace(':', '-')}",
            daemon=True,
        )

        process.start()

        self._processes[symbol] = SymbolProcess(
            symbol=symbol,
            process=process,
            stop_event=stop_event,
            trade_queue=self._trade_queue,
        )

        logger.debug(f"[{symbol}] Process started (PID: {process.pid})")

    async def _stop_symbol(self, symbol: str) -> None:
        """Stop a single symbol's process."""
        if symbol not in self._processes:
            return

        sp = self._processes[symbol]

        # Signal stop
        sp.stop_event.set()

        # Wait for process to exit (with timeout)
        for _ in range(50):  # 5 seconds
            if not sp.process.is_alive():
                break
            await asyncio.sleep(0.1)

        # Force kill if still running
        if sp.process.is_alive():
            logger.warning(f"[{symbol}] Force killing process")
            sp.process.terminate()
            sp.process.join(timeout=1)

        del self._processes[symbol]
        logger.debug(f"[{symbol}] Process stopped")

    async def _consume_trades(self) -> None:
        """Consume trades from queue and call callback."""
        while self._running:
            try:
                # Non-blocking check for trades
                while not self._trade_queue.empty():
                    try:
                        trade_dict = self._trade_queue.get_nowait()
                        trade = Trade(
                            symbol=trade_dict["symbol"],
                            trade_id=trade_dict["trade_id"],
                            price=trade_dict["price"],
                            quantity=trade_dict["quantity"],
                            timestamp_ms=trade_dict["timestamp_ms"],
                            is_buyer_maker=trade_dict["is_buyer_maker"],
                        )

                        # Call callback
                        result = self.on_trade(trade)
                        if hasattr(result, "__await__"):
                            await result

                    except Exception as e:
                        logger.error(f"Error processing trade: {e}")

                await asyncio.sleep(0.001)  # 1ms poll interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(0.1)

    def get_symbols(self) -> list[str]:
        """Get list of currently streaming symbols."""
        return list(self._processes.keys())

    def stats(self) -> dict:
        """Get manager statistics."""
        process_stats = {}
        for symbol, sp in self._processes.items():
            process_stats[symbol] = {
                "pid": sp.process.pid,
                "is_alive": sp.process.is_alive(),
            }

        return {
            "running": self._running,
            "symbol_count": len(self._processes),
            "symbols": list(self._processes.keys()),
            "processes": process_stats,
            "queue_size": self._trade_queue.qsize(),
        }

    # Context manager support
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


@dataclass
class AsyncStreamManager:
    """
    Async-only stream manager using asyncio tasks instead of processes.

    Simpler than StreamManager, runs all streams in the same process
    but in separate asyncio tasks. Good for when process isolation
    isn't needed.

    Attributes:
        symbols: Initial list of symbols to stream.
        on_trade: Callback for each trade.
        config: Stream configuration.
    """

    symbols: list[str]
    on_trade: Callable[[Trade], None] | Callable[[Trade], Coroutine[Any, Any, None]]
    config: StreamConfig = field(default_factory=StreamConfig)

    _streams: dict[str, BinanceFuturesTradeStream] = field(default_factory=dict, init=False)
    _tasks: dict[str, asyncio.Task] = field(default_factory=dict, init=False)
    _running: bool = field(default=False, init=False)

    async def start(self) -> None:
        """Start all symbol streams."""
        if self._running:
            return

        logger.info(f"Starting AsyncStreamManager with {len(self.symbols)} symbols")
        self._running = True

        for symbol in self.symbols:
            await self._start_symbol(symbol)

    async def stop(self) -> None:
        """Stop all symbol streams."""
        if not self._running:
            return

        logger.info("Stopping AsyncStreamManager...")
        self._running = False

        symbols = list(self._streams.keys())
        for symbol in symbols:
            await self._stop_symbol(symbol)

        logger.info("AsyncStreamManager stopped")

    async def add_symbol(self, symbol: str) -> bool:
        """Add a symbol stream at runtime."""
        if symbol in self._streams:
            return False
        await self._start_symbol(symbol)
        return True

    async def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol stream at runtime."""
        if symbol not in self._streams:
            return False
        await self._stop_symbol(symbol)
        return True

    async def _start_symbol(self, symbol: str) -> None:
        """Start a single symbol stream."""
        stream = BinanceFuturesTradeStream(
            symbol=symbol,
            on_trade=self.on_trade,
            config=self.config,
        )

        self._streams[symbol] = stream
        self._tasks[symbol] = asyncio.create_task(stream.start())

    async def _stop_symbol(self, symbol: str) -> None:
        """Stop a single symbol stream."""
        if symbol in self._streams:
            await self._streams[symbol].stop()
            del self._streams[symbol]

        if symbol in self._tasks:
            task = self._tasks[symbol]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self._tasks[symbol]

    def get_symbols(self) -> list[str]:
        """Get list of streaming symbols."""
        return list(self._streams.keys())

    def stats(self) -> dict:
        """Get manager statistics."""
        stream_stats = {symbol: stream.stats() for symbol, stream in self._streams.items()}

        total_trades = sum(s.get("trade_count", 0) for s in stream_stats.values())

        return {
            "running": self._running,
            "symbol_count": len(self._streams),
            "symbols": list(self._streams.keys()),
            "total_trades": total_trades,
            "streams": stream_stats,
        }

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
