"""
Aequify Engine - Async processing loop with PubSub state publishing.

The engine runs in an isolated async event loop and publishes state
via PubSub for real-time TUI updates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aequify.core.pubsub import PubSubServer
from aequify.core.system import SystemMonitor
from aequify.runtime import IsolatedLoop

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

# Default PubSub port
DEFAULT_PUBSUB_PORT = 9100


@dataclass
class EngineState:
    """Observable state for the engine."""

    tick_count: int = 0
    running: bool = False
    status: str = "idle"
    last_tick_time: float = 0.0
    task_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dict for publishing."""
        return {
            "tick_count": self.tick_count,
            "running": self.running,
            "status": self.status,
            "last_tick_time": self.last_tick_time,
            "task_count": self.task_count,
            "errors": self.errors.copy(),
        }


class Engine:
    """
    Async engine with PubSub state publishing.

    The engine processes work asynchronously and publishes state changes
    via PubSub for real-time updates to TUI and other subscribers.

    Topics published:
    - engine.state: Full engine state on each tick
    - engine.error: Error events
    - system.stats: System statistics (CPU, RAM, GPU, Net, Disk I/O)

    Example:
        engine = Engine()
        engine.start()

        # Schedule async work
        engine.schedule(some_coroutine())

        # Later
        engine.stop()
    """

    def __init__(
        self,
        tick_interval: float = 1.0,
        system_stats_interval: float = 2.0,
        pubsub_host: str = "127.0.0.1",
        pubsub_port: int = DEFAULT_PUBSUB_PORT,
    ) -> None:
        """
        Initialize the engine.

        Args:
            tick_interval: Time between ticks in seconds.
            system_stats_interval: Time between system stats updates in seconds.
            pubsub_host: Host for PubSub server.
            pubsub_port: Port for PubSub server.
        """
        self.tick_interval = tick_interval
        self.system_stats_interval = system_stats_interval
        self.pubsub_host = pubsub_host
        self.pubsub_port = pubsub_port

        self._state = EngineState()
        self._loop = IsolatedLoop("engine")
        self._tick_task: asyncio.Task | None = None
        self._system_stats_task: asyncio.Task | None = None

        # PubSub server for state publishing
        self._pubsub: PubSubServer | None = None

        # System monitor for stats collection
        self._system_monitor = SystemMonitor(collect_interval=system_stats_interval)

    @property
    def state(self) -> EngineState:
        """Get a snapshot of current engine state (thread-safe)."""
        return EngineState(
            tick_count=self._state.tick_count,
            running=self._state.running,
            status=self._state.status,
            last_tick_time=self._state.last_tick_time,
            task_count=self._state.task_count,
            errors=self._state.errors.copy(),
        )

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._loop.is_running

    @property
    def pubsub_address(self) -> tuple[str, int]:
        """Get the PubSub server address."""
        return (self.pubsub_host, self.pubsub_port)

    def _publish_state(self) -> None:
        """Publish current state via PubSub."""
        if self._pubsub:
            try:
                self._pubsub.publish("engine.state", self._state.to_dict())
            except Exception as e:
                logger.debug(f"Failed to publish state: {e}")

    def _publish_error(self, error: str) -> None:
        """Publish error event via PubSub."""
        if self._pubsub:
            try:
                self._pubsub.publish("engine.error", {
                    "error": error,
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.debug(f"Failed to publish error: {e}")

    def _publish_system_stats(self) -> None:
        """Collect and publish system statistics via PubSub."""
        if self._pubsub:
            try:
                stats = self._system_monitor.collect()
                self._pubsub.publish("system.stats", stats.to_dict())
            except Exception as e:
                logger.debug(f"Failed to publish system stats: {e}")

    def start(self) -> None:
        """Start the engine and PubSub server."""
        if self._loop.is_running:
            raise RuntimeError("Engine already running")

        self._state.running = True
        self._state.status = "starting"

        # Start PubSub server
        self._pubsub = PubSubServer(
            host=self.pubsub_host,
            port=self.pubsub_port,
            thread_name_prefix="engine-pubsub",
        )
        self._pubsub.start_background()
        logger.info(f"Engine PubSub server started on {self.pubsub_host}:{self.pubsub_port}")

        # Start the isolated event loop
        self._loop.start()

        # Schedule the tick loop
        self._tick_task = self._loop.schedule(self._tick_loop())

        # Schedule the system stats loop
        self._system_stats_task = self._loop.schedule(self._system_stats_loop())

        self._state.status = "running"
        self._publish_state()

        logger.info("Engine started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the engine and PubSub server.

        Args:
            timeout: Maximum time to wait for shutdown.
        """
        if not self._loop.is_running:
            return

        self._state.running = False
        self._state.status = "stopping"
        self._publish_state()

        # Cancel tick task
        if self._tick_task is not None:
            self._tick_task.cancel()
            self._tick_task = None

        # Cancel system stats task
        if self._system_stats_task is not None:
            self._system_stats_task.cancel()
            self._system_stats_task = None

        # Stop the event loop
        self._loop.stop(timeout)

        # Stop PubSub server
        if self._pubsub:
            self._pubsub.stop(timeout)
            self._pubsub = None

        self._state.status = "stopped"
        logger.info("Engine stopped")

    async def _tick_loop(self) -> None:
        """Main tick loop (runs as async task)."""
        while self._state.running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = str(e)
                self._state.errors.append(error_msg)
                self._state.status = "error"
                self._publish_error(error_msg)
                self._publish_state()
                logger.error(f"Engine tick error: {e}")

            await asyncio.sleep(self.tick_interval)

    async def _tick(self) -> None:
        """Single engine tick."""
        self._state.tick_count += 1
        self._state.last_tick_time = time.time()

        # Count active tasks in the loop
        if self._loop.loop is not None:
            self._state.task_count = len(asyncio.all_tasks(self._loop.loop))

        # Publish state update
        self._publish_state()

    async def _system_stats_loop(self) -> None:
        """System stats collection loop (runs as async task)."""
        while self._state.running:
            try:
                self._publish_system_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"System stats error: {e}")

            await asyncio.sleep(self.system_stats_interval)

    def publish(self, topic: str, data: dict[str, Any]) -> int:
        """
        Publish a custom message via PubSub.

        Args:
            topic: Topic name.
            data: Message payload.

        Returns:
            Number of clients the message was sent to.
        """
        if self._pubsub:
            return self._pubsub.publish(topic, data)
        return 0

    def schedule(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Future[Any]:
        """
        Schedule a coroutine to run in the engine's loop.

        Args:
            coro: Coroutine to schedule.

        Returns:
            Future for the result.
        """
        if not self._loop.is_running:
            raise RuntimeError("Engine not running")
        return self._loop.schedule(coro)

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """
        Run a coroutine and wait for result.

        Args:
            coro: Coroutine to run.
            timeout: Maximum time to wait.

        Returns:
            The coroutine's result.
        """
        if not self._loop.is_running:
            raise RuntimeError("Engine not running")
        return self._loop.run(coro, timeout)


# Global engine instance
_engine: Engine | None = None


def get_engine() -> Engine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def start_engine() -> Engine:
    """Start the global engine and return it."""
    engine = get_engine()
    if not engine.is_running:
        engine.start()
    return engine


def stop_engine(timeout: float = 5.0) -> None:
    """Stop the global engine."""
    global _engine
    if _engine is not None:
        _engine.stop(timeout)
