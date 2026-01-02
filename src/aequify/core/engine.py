"""
Aequify Engine - Async processing loop with PubSub state publishing.

The engine runs in an isolated async event loop and publishes state
via PubSub for real-time TUI updates.

Architecture:
    Main Thread (TUI)
        └─ Engine (background thread with IsolatedLoop)
            ├─ PubSub server (publishes state)
            ├─ System monitor (CPU, RAM, GPU stats)
            └─ TradingManager (own IsolatedLoop for trading)
                ├─ BinanceFuturesClient (exchange API)
                ├─ FilterPipeline (symbol selection)
                ├─ RiskManager (pre-trade checks)
                ├─ PositionManager (lifecycle tracking)
                └─ Background tasks (exit monitor, reconciliation)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv

# Load .env file for API credentials
load_dotenv()

from aequify.logging import get_logger
from aequify.core.pubsub import PubSubServer
from aequify.core.system import SystemMonitor
from aequify.runtime import IsolatedLoop

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from aequify.core.exchange.binance_futures import BinanceFuturesClient
    from aequify.core.trading import TradingManager

logger = get_logger(__name__)


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
        config_path: str = "config/config.yaml",
        tick_interval: float = 1.0,
        system_stats_interval: float = 2.0,
    ) -> None:
        """
        Initialize the engine.

        Args:
            config_path: Path to the config.yaml file.
            tick_interval: Time between ticks in seconds.
            system_stats_interval: Time between system stats updates in seconds.
        """
        self.config_path = config_path
        self.tick_interval = tick_interval
        self.system_stats_interval = system_stats_interval

        # Load config first to get pubsub settings
        self._config = self._load_config(config_path)
        pubsub_cfg = self._config.get("pubsub", {})
        self.pubsub_host = pubsub_cfg.get("host", "localhost")
        self.pubsub_port = pubsub_cfg.get("port", 9901)

        self._state = EngineState()
        self._loop = IsolatedLoop("engine")
        self._tick_task: asyncio.Task | None = None
        self._system_stats_task: asyncio.Task | None = None

        # PubSub server for state publishing
        self._pubsub: PubSubServer | None = None

        # System monitor for stats collection
        self._system_monitor = SystemMonitor(collect_interval=system_stats_interval)

        # Trading components (initialized in _start_trading)
        self._client: BinanceFuturesClient | None = None
        self._trading_manager: TradingManager | None = None
        self._trading_initialized: bool = False

        # Position stream for real-time updates
        self._position_stream: Any = None  # BinanceFuturesPositionStream
        self._position_stream_connected: bool = False

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load config from yaml file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        with open(config_file) as f:
            return yaml.safe_load(f) or {}

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

    @property
    def trading_manager(self) -> TradingManager | None:
        """Get the trading manager (if initialized)."""
        return self._trading_manager

    @property
    def client(self) -> BinanceFuturesClient | None:
        """Get the exchange client (if initialized)."""
        return self._client

    @property
    def config(self) -> dict[str, Any]:
        """Get the loaded config."""
        return self._config

    @property
    def is_trading_initialized(self) -> bool:
        """Check if trading components are initialized."""
        return self._trading_initialized

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
            thread_name_prefix="aeq-pubsub",
        )
        self._pubsub.start_background()
        logger.info(f"Engine PubSub server started on {self.pubsub_host}:{self.pubsub_port}")

        # Connect TradingPublisher to PubSub server
        from aequify.core.trading.publisher import get_trading_publisher
        trading_publisher = get_trading_publisher()
        trading_publisher.set_pubsub_server(self._pubsub)

        # Start the isolated event loop
        self._loop.start()

        # Schedule the tick loop
        self._tick_task = self._loop.schedule(self._tick_loop())

        # Schedule the system stats loop
        self._system_stats_task = self._loop.schedule(self._system_stats_loop())

        # Schedule trading startup (runs async in the engine's loop)
        self._loop.schedule(self._start_trading())

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

        # Shutdown trading first (before stopping the loop)
        if self._trading_initialized:
            try:
                self._loop.run(self._stop_trading(), timeout)
            except Exception as e:
                logger.error(f"Error during trading shutdown: {e}")

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

    async def _start_trading(self) -> None:
        """
        Initialize and start trading components.

        Runs in the engine's IsolatedLoop. This method:
        1. Loads config from yaml
        2. Determines demo/live mode
        3. Creates BinanceFuturesClient with appropriate API keys
        4. Creates and initializes TradingManager
        5. TradingManager runs in its own IsolatedLoop for full isolation

        Demo mode uses AEQUIFY_DEMO_API_KEY and AEQUIFY_DEMO_API_SECRET.
        Live mode uses AEQUIFY_API_KEY and AEQUIFY_API_SECRET.
        """
        from aequify.core.exchange.binance_futures import (
            BinanceFuturesClient,
            ClientConfig,
        )
        from aequify.core.trading import TradingConfig, TradingManager

        try:
            self._state.status = "initializing_trading"
            self._publish_state()

            # 1. Load config
            logger.info(f"Loading config from {self.config_path}")
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            with open(config_file) as f:
                self._config = yaml.safe_load(f)

            # 2. Determine demo/live mode
            # Priority: AEQUIFY_DEMO_MODE env var > config.trading.mode
            trading_cfg = self._config.get("trading", {})
            if os.environ.get("AEQUIFY_DEMO_MODE") == "1":
                demo = True
                logger.info("Demo mode enabled via AEQUIFY_DEMO_MODE env var")
            else:
                mode = trading_cfg.get("mode", "demo")
                demo = mode.lower() == "demo"
                logger.info(f"Trading mode from config: {mode}")

            # 3. Get API credentials
            if demo:
                api_key = os.environ.get("AEQUIFY_DEMO_API_KEY", "")
                api_secret = os.environ.get("AEQUIFY_DEMO_API_SECRET", "")
                key_name = "AEQUIFY_DEMO_API_KEY"
            else:
                api_key = os.environ.get("AEQUIFY_API_KEY", "")
                api_secret = os.environ.get("AEQUIFY_API_SECRET", "")
                key_name = "AEQUIFY_API_KEY"

            # Skip trading if no API key
            if not api_key:
                mode_str = "DEMO" if demo else "LIVE"
                logger.warning(
                    f"{key_name} not set - trading disabled. "
                    f"Set {key_name} and {key_name.replace('_KEY', '_SECRET')} to enable {mode_str} trading."
                )
                self._state.status = "running"
                self._publish_state()
                return

            # 4. Create exchange client
            client_config = ClientConfig(
                api_key=api_key,
                api_secret=api_secret,
                demo=demo,
            )
            self._client = BinanceFuturesClient(client_config)
            await self._client.initialize()

            # Connect rate limiter to TradingPublisher for TUI updates
            from aequify.core.trading.publisher import get_trading_publisher
            trading_pub = get_trading_publisher()
            self._client.on_rate_limit_update(trading_pub.make_rate_limit_callback())

            mode_str = "DEMO" if demo else "LIVE"
            logger.info(f"BinanceFuturesClient initialized ({mode_str})")

            # 5. Create and initialize TradingManager
            trading_config = TradingConfig.from_config(self._config)
            self._trading_manager = TradingManager(self._client, trading_config)
            self._trading_manager.set_config_path(self.config_path)
            await self._trading_manager.initialize()

            # 6. Start position stream for real-time updates
            await self._start_position_stream(api_key, api_secret, demo)

            self._trading_initialized = True
            self._state.status = "running"
            self._publish_state()

            # Publish initial stream status
            self._publish_stream_status()

            logger.info(
                f"Trading initialized: {len(self._trading_manager.active_symbols)} active symbols"
            )

        except Exception as e:
            error_msg = f"Failed to initialize trading: {e}"
            logger.error(error_msg)
            self._state.errors.append(error_msg)
            self._state.status = "trading_error"
            self._publish_error(error_msg)
            self._publish_state()

    async def _stop_trading(self) -> None:
        """Shutdown trading components gracefully."""
        # Stop position stream first
        await self._stop_position_stream()

        if self._trading_manager:
            try:
                logger.info("Shutting down TradingManager...")
                await self._trading_manager.shutdown()
                logger.info("TradingManager shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down TradingManager: {e}")

        if self._client:
            try:
                logger.debug("Closing exchange client...")
                await self._client.close()
                logger.debug("Exchange client closed")
            except Exception as e:
                logger.error(f"Error closing exchange client: {e}")

        self._trading_initialized = False

    async def _start_position_stream(
        self, api_key: str, api_secret: str, demo: bool
    ) -> None:
        """Start position stream for real-time position updates."""
        from aequify.core.rtds.position_stream import (
            BinanceFuturesPositionStream,
            PositionStreamConfig,
        )
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            # Get poll delay from config
            trading_cfg = self._config.get("trading", {})
            poll_delay = float(trading_cfg.get("position_poll_delay", 60))
            recv_window = self._config.get("developer", {}).get("recvWindow", 15000)

            stream_config = PositionStreamConfig(
                demo=demo,
                api_key=api_key,
                api_secret=api_secret,
                fetch_snapshot=True,
                poll_delay=poll_delay,
                recv_window=recv_window,
            )

            self._position_stream = BinanceFuturesPositionStream(
                on_position=self._on_position_update,
                on_connect=self._on_position_stream_connect,
                on_disconnect=self._on_position_stream_disconnect,
                config=stream_config,
            )

            await self._position_stream.start()
            logger.info(f"Position stream started (poll_delay={poll_delay}s)")

        except Exception as e:
            logger.error(f"Failed to start position stream: {e}")
            self._position_stream = None

    async def _stop_position_stream(self) -> None:
        """Stop position stream."""
        if self._position_stream:
            try:
                await self._position_stream.stop()
                logger.info("Position stream stopped")
            except Exception as e:
                logger.error(f"Error stopping position stream: {e}")
            self._position_stream = None
            self._position_stream_connected = False

    def _on_position_update(self, positions: list) -> None:
        """Handle position update from stream - publish via PubSub."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Convert PositionData objects to dicts
            positions_data = []
            for pos in positions:
                positions_data.append({
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "contracts": pos.contracts,
                    "entryPrice": pos.entry_price,
                    "markPrice": pos.mark_price,
                    "liquidationPrice": pos.liquidation_price,
                    "unrealizedPnl": pos.unrealized_pnl,
                    "percentage": pos.unrealized_pnl_pct,
                    "notional": pos.notional,
                    "leverage": pos.leverage,
                    "marginMode": pos.margin_mode,
                })

            publisher.publish_positions(positions_data)
        except Exception as e:
            logger.debug(f"Failed to publish positions from stream: {e}")

    async def _on_position_stream_connect(self) -> None:
        """Handle position stream connection."""
        self._position_stream_connected = True
        self._publish_stream_status()
        logger.info("Position stream connected")

    async def _on_position_stream_disconnect(self, error: Exception | None) -> None:
        """Handle position stream disconnection."""
        self._position_stream_connected = False
        self._publish_stream_status()
        if error:
            logger.warning(f"Position stream disconnected: {error}")
        else:
            logger.info("Position stream disconnected")

    def _publish_stream_status(self) -> None:
        """Publish WebSocket stream status via PubSub."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Get trade stream stats from TradingManager's SymbolManagerRegistry
            trade_streams_connected = 0
            trade_streams_total = 0
            if self._trading_manager:
                registry = getattr(self._trading_manager, "_symbol_registry", None)
                if registry:
                    stats = registry.stats()
                    trade_streams_connected = stats.get("connected_count", 0)
                    trade_streams_total = stats.get("symbol_count", 0)

            publisher.publish_stream_status(
                trade_streams_connected=trade_streams_connected,
                trade_streams_total=trade_streams_total,
                position_stream_connected=self._position_stream_connected,
            )
        except Exception as e:
            logger.debug(f"Failed to publish stream status: {e}")

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
