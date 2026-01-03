"""
Trading Manager - Orchestrates all trading components.

Main entry point for trading operations. Coordinates:
- Filter pipeline for symbol selection
- Startup position recovery (existing positions, orphaned orders)
- Risk checks before entry
- Entry validation and execution
- Exit monitoring and execution
- Position lifecycle management

Uses IsolatedLoop for non-blocking async execution from Mojo.
The trading manager runs entirely in its own thread to avoid blocking
the main Mojo/TUI thread.

Architecture:
    TradingManager
        └─ IsolatedLoop (own thread + event loop)
            ├─ FilterPipeline (symbol selection)
            ├─ RiskManager (pre-trade checks)
            ├─ PositionManager (lifecycle tracking)
            ├─ EntryExecutor (order placement)
            ├─ ExitExecutor (position closing)
            └─ Background tasks:
                ├─ _exit_monitor_loop() (exit conditions)
                ├─ _filter_update_loop() (symbol updates)
                └─ _position_reconciliation_loop() (sync with exchange)
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from aequify.logging import get_logger
from aequify.runtime import IsolatedLoop

if TYPE_CHECKING:
    from aequify.core.apex import APEX, BootstrapResult
    from aequify.core.apex.live import APEXLiveDetector, APEXLiveDetectorManager
    from aequify.core.apex.signal_handler import APEXSignalHandler
    from aequify.core.db.cold.backfill import BackfillProgress
    from aequify.core.exchange.binance_futures import BinanceFuturesClient, PositionSide

logger = get_logger(__name__)


@dataclass
class TradingConfig:
    """
    Complete trading configuration.

    Loads from config dict and provides defaults.
    """

    # Mode
    demo: bool = True  # Use demo/testnet mode

    # Risk
    max_positions: int = 3
    max_daily_loss_pct: float = 5.0

    # LONG config
    long_enabled: bool = True
    long_initial_size_usdt: float = 6.5
    long_max_size_usdt: float = 100.0
    long_tp_pct: float = 1.0
    long_sl_pct: float = 0.5
    long_dca_distance_pct: float = -0.5  # Negative for LONG (price must drop)
    long_dca_multiplier: float = 4.25  # DCA size multiplier

    # SHORT config
    short_enabled: bool = True
    short_initial_size_usdt: float = 6.5
    short_max_size_usdt: float = 100.0
    short_tp_pct: float = 1.0
    short_sl_pct: float = 0.5
    short_dca_distance_pct: float = 0.5  # Positive for SHORT (price must rise)
    short_dca_multiplier: float = 4.25  # DCA size multiplier

    # General
    leverage: int = 10
    margin_type: str = "isolated"
    timeout_seconds: float = 120.0
    exit_check_interval_ms: int = 1000

    # Filter pipeline
    filter_configs: list[dict[str, Any]] = field(default_factory=list)
    filter_update_interval_seconds: float = 300.0  # 5 minutes

    # Position reconciliation (sync with exchange)
    position_reconciliation_interval_seconds: float = 15.0

    # Stream connection staggering (avoid rate limiting on startup)
    stream_connection_delay_ms: int = 200  # Delay between WebSocket connections

    # Startup behavior
    cancel_orphaned_orders_on_startup: bool = True
    restore_tp_sl_on_startup: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TradingConfig:
        """Create TradingConfig from config dict."""
        import os

        trading = config.get("trading", {})
        risk = trading.get("risk", {})
        futures = trading.get("futures", {})
        long_cfg = futures.get("long", {})
        short_cfg = futures.get("short", {})

        # Determine demo mode (env var overrides config)
        if os.environ.get("AEQUIFY_DEMO_MODE") == "1":
            demo = True
        else:
            mode = trading.get("mode", "demo")
            demo = mode.lower() == "demo"

        return cls(
            demo=demo,
            max_positions=risk.get("max_positions", 3),
            max_daily_loss_pct=risk.get("max_daily_loss_pct", 5.0),
            long_enabled=long_cfg.get("enabled", True),
            long_initial_size_usdt=long_cfg.get("initial_entry_size_usdt", 6.5),
            long_max_size_usdt=long_cfg.get("maximum_position_size_usdt", 100.0),
            long_tp_pct=long_cfg.get("take_profit_pct", 1.0),
            long_sl_pct=long_cfg.get("stop_loss_pct", 0.5),
            long_dca_distance_pct=long_cfg.get("dca_distance_pct", -0.5),
            long_dca_multiplier=long_cfg.get("dca_multiplier", 4.25),
            short_enabled=short_cfg.get("enabled", True),
            short_initial_size_usdt=short_cfg.get("initial_entry_size_usdt", 6.5),
            short_max_size_usdt=short_cfg.get("maximum_position_size_usdt", 100.0),
            short_tp_pct=short_cfg.get("take_profit_pct", 1.0),
            short_sl_pct=short_cfg.get("stop_loss_pct", 0.5),
            short_dca_distance_pct=short_cfg.get("dca_distance_pct", 0.5),
            short_dca_multiplier=short_cfg.get("dca_multiplier", 4.25),
            leverage=futures.get("leverage", 10),
            margin_type=(futures.get("margin_mode") or "isolated").lower(),
            timeout_seconds=trading.get("timeout_seconds", 120.0),
            exit_check_interval_ms=trading.get("exit_check_interval_ms", 1000),
            filter_configs=config.get("filters", []),
            filter_update_interval_seconds=trading.get("filter_update_interval_seconds", 300.0),
            position_reconciliation_interval_seconds=trading.get(
                "order_reconciliation_interval", 15.0
            ),
            stream_connection_delay_ms=trading.get(
                "stream_connection_delay_ms", 200
            ),
            cancel_orphaned_orders_on_startup=trading.get(
                "cancel_orphaned_orders_on_startup", True
            ),
            restore_tp_sl_on_startup=trading.get("restore_tp_sl_on_startup", True),
        )


# Type alias for signal callbacks
SignalCallback = Callable[["TradingSignal"], None]


@dataclass
class TradingSignal:
    """
    A trading signal to be processed.

    Attributes:
        symbol: Trading pair.
        side: LONG or SHORT.
        price: Signal price.
        timestamp: Signal timestamp.
        params: Additional signal parameters.
    """

    symbol: str
    side: "PositionSide"
    price: float
    timestamp: float
    params: dict[str, Any] | None = None


class TradingManager:
    """
    Main trading orchestrator.

    Coordinates all trading components:
    - FilterPipeline: Symbol selection based on filters
    - RiskManager: Pre-trade risk checks
    - EntryValidator: Entry condition validation
    - EntryExecutor: Order placement
    - ExitValidator: Exit condition checking
    - ExitExecutor: Position closing
    - PositionManager: Position lifecycle tracking

    Runs entirely in an IsolatedLoop (separate thread) for non-blocking
    operation from Mojo code. This ensures the TUI remains responsive.

    On startup:
    1. Syncs positions with exchange
    2. Cancels orphaned TP/SL orders from previous sessions
    3. Tracks existing positions in PositionManager
    4. Optionally restores TP/SL orders for existing positions
    5. Runs filter pipeline to determine active symbols

    Usage:
        manager = TradingManager(client, config)
        await manager.initialize()

        # Process signal
        await manager.process_signal(signal)

        # Or use non-blocking from Mojo
        manager.process_signal_async(signal)

        # Get active symbols (from filter pipeline)
        symbols = manager.active_symbols

        # Cleanup
        await manager.shutdown()
    """

    def __init__(
        self,
        client: "BinanceFuturesClient",
        config: TradingConfig,
    ) -> None:
        """
        Initialize trading manager.

        Args:
            client: Exchange client.
            config: Trading configuration.
        """
        self.client = client
        self.config = config

        # Components (initialized in initialize())
        self._position_manager: Any = None
        self._risk_manager: Any = None
        self._entry_validator: Any = None
        self._entry_executor: Any = None
        self._exit_validator: Any = None
        self._exit_executor: Any = None
        self._filter_pipeline: Any = None

        # IsolatedLoop for async operations
        self._loop = IsolatedLoop("trading-manager")

        # Background tasks (running in isolated loop)
        self._exit_monitor_task: asyncio.Task | None = None
        self._filter_update_task: asyncio.Task | None = None
        self._reconciliation_task: asyncio.Task | None = None
        self._running = False

        # Active symbols from filter pipeline
        self._active_symbols: set[str] = set()
        self._position_sides: dict[str, set[str]] = {}  # symbol -> {"LONG", "SHORT"}
        self._price_changes: dict[str, float] = {}  # symbol -> 24h price change %

        # Symbol manager registry for trade streams
        self._symbol_registry: Any = None

        # Signal callbacks
        self._signal_callbacks: list[SignalCallback] = []

        # Startup time tracking
        self._start_time: datetime | None = None

        # Sequential symbol initialization (1 at a time)
        self._symbol_init_lock = threading.Lock()
        self._bootstrapped_symbols: dict[str, "BootstrapResult"] = {}
        self._config_path: str = "config/config.yaml"

        # APEX live detection components
        self._signal_handler: "APEXSignalHandler | None" = None
        self._detector_manager: "APEXLiveDetectorManager | None" = None
        self._apex_instances: dict[str, "APEX"] = {}  # symbol -> APEX instance
        self._live_metrics_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """
        Initialize all trading components.

        Initialization sequence:
        1. Create component configs
        2. Initialize core components (position manager, risk manager, etc.)
        3. Initialize filter pipeline
        4. Cancel orphaned orders from previous sessions
        5. Sync and recover existing positions
        6. Run initial filter pipeline
        7. Start background tasks in IsolatedLoop
        """
        from aequify.core.filters.pipeline import FilterPipeline
        from aequify.core.trading.entry.entry_executor import EntryExecutor
        from aequify.core.trading.entry.entry_validator import EntryConfig, EntryValidator
        from aequify.core.trading.exit.exit_executor import ExitExecutor
        from aequify.core.trading.exit.exit_validator import ExitConfig, ExitValidator
        from aequify.core.trading.position_manager import PositionManager
        from aequify.core.trading.risk_manager import RiskConfig, RiskManager

        self._start_time = datetime.now()

        # Create configs
        long_entry_config = EntryConfig(
            initial_entry_size_usdt=self.config.long_initial_size_usdt,
            maximum_position_size_usdt=self.config.long_max_size_usdt,
            leverage=self.config.leverage,
            margin_type=self.config.margin_type,
            dca_multiplier=self.config.long_dca_multiplier,
        )

        short_entry_config = EntryConfig(
            initial_entry_size_usdt=self.config.short_initial_size_usdt,
            maximum_position_size_usdt=self.config.short_max_size_usdt,
            leverage=self.config.leverage,
            margin_type=self.config.margin_type,
            dca_multiplier=self.config.short_dca_multiplier,
        )

        long_exit_config = ExitConfig(
            take_profit_pct=self.config.long_tp_pct,
            stop_loss_pct=self.config.long_sl_pct,
            timeout_seconds=self.config.timeout_seconds,
        )

        short_exit_config = ExitConfig(
            take_profit_pct=self.config.short_tp_pct,
            stop_loss_pct=self.config.short_sl_pct,
            timeout_seconds=self.config.timeout_seconds,
        )

        risk_config = RiskConfig(
            max_positions=self.config.max_positions,
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            long_enabled=self.config.long_enabled,
            short_enabled=self.config.short_enabled,
        )

        # Initialize core components
        self._position_manager = PositionManager()
        self._risk_manager = RiskManager(risk_config, self.client)
        self._entry_validator = EntryValidator(
            self.client, long_entry_config, short_entry_config
        )
        self._entry_executor = EntryExecutor(
            self.client,
            self._position_manager,
            self._risk_manager,
            long_entry_config,
            short_entry_config,
        )
        self._exit_validator = ExitValidator(
            self.client, long_exit_config, short_exit_config
        )
        self._exit_executor = ExitExecutor(
            self.client, self._position_manager, self._risk_manager
        )

        # Initialize filter pipeline
        if self.config.filter_configs:
            self._filter_pipeline = FilterPipeline(
                filter_configs=self.config.filter_configs,
                client=self.client,
            )
            await self._filter_pipeline.initialize()
            logger.info("FilterPipeline initialized")

        # Initialize APEX live detection components
        self._init_apex_live_detection()

        # Initialize risk manager
        await self._risk_manager.initialize()

        # Step 1: Cancel orphaned orders from previous sessions
        if self.config.cancel_orphaned_orders_on_startup:
            await self._cancel_all_orders_on_startup()

        # Step 2: Sync with exchange positions and recover existing positions
        await self._risk_manager.sync_positions()
        await self._recover_existing_positions()

        # Step 3: Run initial filter pipeline
        if self._filter_pipeline:
            await self._update_active_symbols()

        # Step 4: Initialize trade streams for active symbols
        # Start isolated loop first (needed for staggered stream connections)
        self._loop.start()
        self._init_symbol_registry()
        self._sync_trade_streams()

        # Start background tasks on the CURRENT loop (engine's loop)
        # IMPORTANT: These tasks call CCXT methods which are tied to the loop
        # where the client was initialized. Running them on a different loop
        # causes "Timeout context manager should be used inside a task" errors.
        self._running = True
        await self._start_background_tasks()

        logger.info(
            f"TradingManager initialized "
            f"(positions: {len(self._risk_manager._open_positions)}, "
            f"symbols: {len(self._active_symbols)})"
        )

        # Step 5: Initialize active symbols (backfill + bootstrap) in background
        # Position symbols first (they need TP/SL immediately)
        if self._active_symbols:
            position_symbols = list(self._position_sides.keys())
            other_symbols = [s for s in self._active_symbols if s not in position_symbols]
            all_symbols = position_symbols + other_symbols
            self._loop.schedule(self.initialize_symbols(all_symbols))

    async def _start_background_tasks(self) -> None:
        """Start all background monitoring tasks in the isolated loop."""
        # Use asyncio.create_task() which is the modern idiom in Python 3.7+
        # This properly handles task context in Python 3.10+

        # Exit monitoring loop
        self._exit_monitor_task = asyncio.create_task(
            self._exit_monitor_loop(), name="exit-monitor"
        )

        # Filter update loop (if filter pipeline is configured)
        if self._filter_pipeline:
            self._filter_update_task = asyncio.create_task(
                self._filter_update_loop(), name="filter-update"
            )

        # Position reconciliation loop
        self._reconciliation_task = asyncio.create_task(
            self._position_reconciliation_loop(), name="position-reconciliation"
        )

        # Live metrics publishing loop (for APEX TUI)
        self._live_metrics_task = asyncio.create_task(
            self._live_metrics_loop(), name="live-metrics"
        )

        logger.debug("Background tasks started")

    async def _cancel_all_orders_on_startup(self) -> None:
        """
        Cancel all open orders on startup.

        This prevents orphaned TP/SL orders from previous sessions
        from causing unexpected behavior.
        """
        try:
            # Collect all symbols that need order cleanup
            symbols_to_clean: set[str] = set()

            # 1. Get symbols from open algo orders (TP/SL)
            try:
                algo_orders = await self.client.fetch_algo_open_orders(None)
                for order in algo_orders:
                    # Algo orders return Binance format symbol (e.g., "BTCUSDT")
                    # Convert to CCXT format for cancel_all_orders
                    binance_symbol = order.get("symbol", "")
                    if binance_symbol:
                        # Convert "BTCUSDT" -> "BTC/USDT:USDT"
                        if binance_symbol.endswith("USDT"):
                            base = binance_symbol[:-4]
                            ccxt_symbol = f"{base}/USDT:USDT"
                            symbols_to_clean.add(ccxt_symbol)
                if algo_orders:
                    logger.debug(f"Found {len(algo_orders)} algo orders to clean up")
            except Exception as e:
                logger.debug(f"Could not fetch algo orders: {e}")

            # 2. Get symbols from open positions
            try:
                positions = await self.client.fetch_positions()
                for pos in positions:
                    position_amt = float(pos.get("contracts", 0) or 0)
                    if position_amt != 0:
                        symbol = pos.get("symbol", "")
                        if symbol:
                            symbols_to_clean.add(symbol)
            except Exception as e:
                logger.debug(f"Could not fetch positions: {e}")

            # 3. Cancel all orders for each symbol
            for symbol in symbols_to_clean:
                try:
                    await self.client.cancel_all_orders(symbol)
                    logger.debug(f"Canceled orders for {symbol}")
                except Exception as e:
                    logger.debug(f"Could not cancel orders for {symbol}: {e}")

            logger.info("Startup order cleanup complete")

        except Exception as e:
            logger.error(f"Failed to cancel orders on startup: {e}")

    async def _recover_existing_positions(self) -> None:
        """
        Recover existing positions from the exchange.

        For each open position:
        1. Create a TrackedPosition in PositionManager
        2. Optionally restore TP/SL orders

        This ensures positions opened in previous sessions are tracked.
        """
        from aequify.core.exchange.binance_futures import PositionSide

        try:
            positions = await self.client.get_positions()
            recovered_count = 0

            for pos in positions:
                symbol = pos.symbol
                contracts = abs(pos.contracts) if pos.contracts else 0.0
                if contracts == 0:
                    continue

                # Determine side
                side = pos.side if pos.side else (
                    PositionSide.LONG if pos.contracts > 0 else PositionSide.SHORT
                )

                # Create tracked position
                tracked = await self._position_manager.create_position(
                    symbol=symbol,
                    side=side,
                    signal_params={"recovered": True},
                    max_hold_time_ms=int(self.config.timeout_seconds * 1000),
                )

                # Update with entry details from exchange
                entry_price = pos.entry_price if pos.entry_price else 0.0
                tracked.entry_price = entry_price
                tracked.entry_quantity = contracts
                tracked.entry_time = datetime.now()  # We don't know actual entry time

                # Mark as open
                from aequify.core.trading.position_manager import TradeStatus
                tracked.status = TradeStatus.OPEN

                # Optionally restore TP/SL orders
                if self.config.restore_tp_sl_on_startup and entry_price > 0:
                    await self._restore_tp_sl_for_position(tracked, side)

                recovered_count += 1
                logger.info(
                    f"Recovered position: {symbol} {side.value} "
                    f"{contracts} @ {entry_price:.6f}"
                )

            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} existing positions")

        except Exception as e:
            logger.error(f"Failed to recover existing positions: {e}")

    async def _restore_tp_sl_for_position(
        self,
        position: Any,  # TrackedPosition
        side: "PositionSide",
    ) -> None:
        """
        Restore TP/SL orders for a recovered position.

        Uses default TP/SL percentages from config.
        """
        from aequify.core.exchange.binance_futures import PositionSide

        symbol = position.symbol
        entry_price = position.entry_price
        quantity = position.entry_quantity

        # Get TP/SL percentages based on side
        if side == PositionSide.LONG:
            tp_pct = self.config.long_tp_pct
            sl_pct = self.config.long_sl_pct
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
        else:
            tp_pct = self.config.short_tp_pct
            sl_pct = self.config.short_sl_pct
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)

        try:
            # Place TP order
            tp_order = await self._entry_executor._place_tp_order(
                symbol, side, quantity, tp_price
            )
            if tp_order:
                position.tp_order = tp_order
                position.target_profit_price = tp_price
                logger.debug(f"Restored TP for {symbol}: {tp_price:.6f}")

            # Place SL order
            sl_order = await self._entry_executor._place_sl_order(
                symbol, side, quantity, sl_price
            )
            if sl_order:
                position.sl_order = sl_order
                position.stop_loss_price = sl_price
                logger.debug(f"Restored SL for {symbol}: {sl_price:.6f}")

        except Exception as e:
            logger.warning(f"Failed to restore TP/SL for {symbol}: {e}")

    async def _update_active_symbols(self) -> None:
        """
        Run filter pipeline and update active symbols.

        Active positions are always included (handled by pipeline).
        Publishes symbol updates to PubSub for TUI consumption.
        """
        if not self._filter_pipeline:
            return

        try:
            result = await self._filter_pipeline.run()
            self._active_symbols = result.symbols
            self._position_sides = result.metadata.get("position_sides", {})
            self._price_changes = result.metadata.get("price_changes", {})

            # Publish symbols to PubSub for TUI
            self._publish_symbols_to_pubsub()

            # Publish market data to PubSub for TUI (symbol info pane)
            self._publish_markets_to_pubsub()

            # Sync trade streams with new active symbols
            self._sync_trade_streams()

            logger.info(
                f"Active symbols updated: {len(self._active_symbols)} symbols "
                f"(positions: {len(result.metadata.get('active_positions', []))})"
            )

        except Exception as e:
            logger.error(f"Failed to update active symbols: {e}")

    def _publish_symbols_to_pubsub(self) -> None:
        """Publish active symbols to PubSub for TUI consumption."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Build symbol list with position metadata
            symbols_data = []
            for symbol in self._active_symbols:
                sides = self._position_sides.get(symbol, set())
                symbols_data.append({
                    "symbol": symbol,
                    "has_position": len(sides) > 0,
                    "has_long": "LONG" in sides,
                    "has_short": "SHORT" in sides,
                    "price_change_pct": self._price_changes.get(symbol, 0.0),
                    "position_pnl": 0.0,  # TODO: enrich from position manager
                })

            publisher.publish_symbols(symbols_data, source="filter")
        except Exception as e:
            logger.debug(f"Failed to publish symbols to PubSub: {e}")

    def _publish_markets_to_pubsub(self) -> None:
        """Publish market data to PubSub for TUI symbol info display."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Get markets from exchange
            if not hasattr(self.client, "_exchange") or not self.client._exchange.markets:
                return

            markets = self.client._exchange.markets
            publisher.publish_markets(markets, symbols=self._active_symbols)
            logger.debug(f"Published market data for {len(self._active_symbols)} symbols")
        except Exception as e:
            logger.debug(f"Failed to publish markets to PubSub: {e}")

    def _publish_positions_to_pubsub(self, positions: list[Any]) -> None:
        """Publish positions to PubSub for TUI consumption."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Convert Position objects to dicts for publishing
            positions_data = []
            for pos in positions:
                # Handle Position objects (from client.get_positions())
                contracts = getattr(pos, "size", 0) or getattr(pos, "contracts", 0) or 0
                if contracts == 0:
                    continue

                positions_data.append({
                    "symbol": pos.symbol,
                    "side": pos.side.value if hasattr(pos.side, "value") else str(pos.side),
                    "contracts": contracts,
                    "entryPrice": pos.entry_price,
                    "markPrice": getattr(pos, "mark_price", 0),
                    "liquidationPrice": getattr(pos, "liquidation_price", 0),
                    "unrealizedPnl": getattr(pos, "unrealized_pnl", 0),
                    "percentage": getattr(pos, "percentage", 0),
                    "notional": getattr(pos, "notional", 0),
                    "leverage": getattr(pos, "leverage", 1),
                    "marginMode": getattr(pos, "margin_type", "cross"),
                })

            publisher.publish_positions(positions_data)
        except Exception as e:
            logger.debug(f"Failed to publish positions to PubSub: {e}")

    def _publish_apex_state(
        self, symbol: str, result: "BootstrapResult", source: str = "bootstrap"
    ) -> None:
        """Publish APEX bootstrap state to PubSub for TUI consumption."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Convert BootstrapResult to dict for TUI
            state = {
                "is_bootstrapped": True,
                "source": source,
                "init_stage": "complete",
                "long_params": None,
                "short_params": None,
            }

            if result.long:
                # is_default: True if no valid params found (entries=0, using defaults)
                is_default = result.long.entries == 0
                state["long_params"] = {
                    "price_move": result.long.params.price_move,
                    "time_window": result.long.params.time_window,
                    "delta_threshold": result.long.params.delta_threshold,
                    "dca_distance_pct": getattr(result.long.params, "dca_distance_pct", -3.0),
                    "target_profit": result.long.params.target_profit,
                    "stop_loss": result.long.params.stop_loss,
                    "max_hold_time_ms": result.long.params.max_hold_time,
                    "win_rate": result.long.win_rate,
                    "entries": result.long.entries,
                    "avg_pnl": result.long.avg_pnl,
                    "is_default": is_default,
                }

            if result.short:
                is_default = result.short.entries == 0
                state["short_params"] = {
                    "price_move": result.short.params.price_move,
                    "time_window": result.short.params.time_window,
                    "delta_threshold": result.short.params.delta_threshold,
                    "dca_distance_pct": getattr(result.short.params, "dca_distance_pct", 3.0),
                    "target_profit": result.short.params.target_profit,
                    "stop_loss": result.short.params.stop_loss,
                    "max_hold_time_ms": result.short.params.max_hold_time,
                    "win_rate": result.short.win_rate,
                    "entries": result.short.entries,
                    "avg_pnl": result.short.avg_pnl,
                    "is_default": is_default,
                }

            publisher.publish_apex_state(symbol, state)
            logger.debug(f"[{symbol}] Published APEX state to TUI (source={source})")
        except Exception as e:
            logger.debug(f"Failed to publish APEX state to PubSub: {e}")

    def _generate_override_file(
        self,
        symbol: str,
        result: "BootstrapResult",
        trade_count: int = 0,
        config: "APEXConfig | None" = None,
    ) -> None:
        """Generate override YAML file with narrowed bounds after bootstrap."""
        import os
        from aequify.core.apex.bootstrap import generate_override_for_symbol
        from aequify.core.apex.config import load_apex_config_for_symbol

        try:
            # Load config if not provided
            if config is None:
                config = load_apex_config_for_symbol(self._config_path, symbol)

            # Derive overrides directory from config path
            config_dir = os.path.dirname(os.path.abspath(self._config_path))
            overrides_dir = os.path.join(config_dir, "overrides")

            # Generate override file with narrowed bounds
            filepath = generate_override_for_symbol(
                symbol=symbol,
                config=config,
                result=result,
                overrides_dir=overrides_dir,
                trade_count=trade_count,
                overwrite=True,  # Always overwrite with fresh bootstrap results
            )

            if filepath:
                logger.info(f"[{symbol}] Override file generated: {filepath}")
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to generate override file: {e}")

    def _publish_backfill_progress(self, symbol: str, progress: "BackfillProgress") -> None:
        """Publish backfill progress to TUI via PubSub."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Publish as APEX state with backfill info
            state = {
                "is_bootstrapped": False,
                "source": "backfill",
                "init_stage": "backfilling",
                "backfill_status": f"Importing {progress.current_date}...",
                "backfill_progress": int(progress.progress_pct),
                "backfill_current_day": progress.days_processed,
                "backfill_total_days": progress.total_days,
                "trade_count": progress.total_trades,
            }

            publisher.publish_apex_state(symbol, state)
        except Exception as e:
            logger.debug(f"Failed to publish backfill progress: {e}")

    def _clear_backfill_progress(self, symbol: str) -> None:
        """Clear backfill progress in TUI (called after backfill completes)."""
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Publish state with cleared backfill status only
            # NOTE: Do NOT reset is_bootstrapped here - this is a partial update
            # that should only clear backfill progress fields. The TUI handler
            # will preserve existing bootstrap state when receiving this.
            state = {
                "source": "bootstrap",
                "backfill_status": "",
                "backfill_progress": 0,
                "backfill_current_day": 0,
                "backfill_total_days": 0,
            }

            publisher.publish_apex_state(symbol, state)
        except Exception as e:
            logger.debug(f"Failed to clear backfill progress: {e}")

    def _publish_init_stage(
        self,
        symbol: str,
        stage: str,
        queue_position: int = 0,
        queue_total: int = 0,
    ) -> None:
        """
        Publish initialization stage to TUI via PubSub.

        Args:
            symbol: Trading pair.
            stage: One of: "queued", "checking_cache", "backfilling", "bootstrapping", "complete"
            queue_position: Position in initialization queue (1-based).
            queue_total: Total symbols in queue.
        """
        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            state = {
                "is_bootstrapped": stage == "complete",
                "init_stage": stage,
                "init_queue_position": queue_position,
                "init_queue_total": queue_total,
            }

            publisher.publish_apex_state(symbol, state)
            logger.debug(f"[{symbol}] Init stage: {stage}")
        except Exception as e:
            logger.debug(f"Failed to publish init stage: {e}")

    def _init_apex_live_detection(self) -> None:
        """Initialize APEX live detection components (signal handler + detector manager)."""
        from aequify.core.apex.live import APEXLiveDetectorManager, LiveDetectorConfig
        from aequify.core.apex.signal_handler import APEXSignalHandler, SignalConfig

        try:
            # Load config for signal handler
            import yaml
            with open(self._config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # Create signal handler
            signal_config = SignalConfig.from_config(config)
            self._signal_handler = APEXSignalHandler(signal_config)
            self._signal_handler.set_trading_manager(self)

            # Create detector manager
            detector_config = LiveDetectorConfig.from_config(config)
            self._detector_manager = APEXLiveDetectorManager(
                signal_handler=self._signal_handler,
                default_config=detector_config,
            )

            logger.info("APEX live detection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize APEX live detection: {e}")

    def _create_live_detector(self, symbol: str, result: "BootstrapResult") -> None:
        """
        Create APEXLiveDetector for a bootstrapped symbol.

        Args:
            symbol: Trading pair.
            result: Bootstrap result with optimized parameters.
        """
        if not self._detector_manager:
            return

        try:
            from aequify.core.apex import APEX, load_apex_config_for_symbol

            # Load config for this symbol
            config = load_apex_config_for_symbol(self._config_path, symbol)

            # Create APEX instance with bootstrap result
            apex = APEX(symbol=symbol, config=config)
            apex.set_bootstrap_result(result)

            # Store APEX instance
            self._apex_instances[symbol] = apex

            # Create detector (will receive trades from SymbolManager)
            self._detector_manager.add(symbol, apex)
            logger.info(f"[{symbol}] Live detector created")

            # Wire trades from SymbolManager to detector
            self._wire_detector_to_stream(symbol)

        except ValueError:
            # Detector already exists
            logger.debug(f"[{symbol}] Detector already exists")
        except Exception as e:
            logger.error(f"[{symbol}] Failed to create live detector: {e}")

    def _wire_detector_to_stream(self, symbol: str) -> None:
        """
        Wire trade stream to live detector for a symbol.

        Wraps the SymbolManager's on_trade callback to also feed the detector.

        Args:
            symbol: Trading pair.
        """
        if not self._symbol_registry or not self._detector_manager:
            return

        detector = self._detector_manager.get(symbol)
        if not detector:
            return

        manager = self._symbol_registry.get(symbol)
        if not manager:
            return

        # Get the existing callback
        original_callback = manager.on_trade

        def wrapped_callback(trade: dict) -> None:
            """Wrap callback to also feed detector."""
            # Call original callback (HotStore + PubSub)
            if original_callback:
                try:
                    original_callback(trade)
                except Exception as e:
                    logger.debug(f"[{symbol}] Original callback error: {e}")

            # Feed trade to detector
            try:
                detector.on_trade(
                    price=float(trade["price"]),
                    quantity=float(trade["amount"]),
                    timestamp_ms=int(trade["timestamp"]),
                    is_buyer_maker=trade.get("side") == "sell",
                )
            except Exception as e:
                logger.debug(f"[{symbol}] Detector feed error: {e}")

        # Replace callback with wrapped version
        manager.on_trade = wrapped_callback
        logger.debug(f"[{symbol}] Detector wired to trade stream")

    def _init_symbol_registry(self) -> None:
        """Initialize SymbolManagerRegistry for trade streams."""
        from aequify.core.trading.publisher import get_trading_publisher
        from aequify.core.trading.symbol_manager import get_symbol_registry

        try:
            self._symbol_registry = get_symbol_registry()
            publisher = get_trading_publisher()
            if publisher.is_connected:
                self._symbol_registry.set_publisher(publisher)
            logger.debug("SymbolManagerRegistry initialized")
        except Exception as e:
            logger.error(f"Failed to initialize symbol registry: {e}")

    def _sync_trade_streams(self) -> None:
        """Sync trade streams with current active symbols."""
        if not self._symbol_registry:
            return

        try:
            current_symbols = set(self._symbol_registry.symbols())
            target_symbols = self._active_symbols

            # Stop streams for symbols no longer active
            symbols_to_remove = current_symbols - target_symbols
            for symbol in symbols_to_remove:
                self._symbol_registry.remove(symbol)

            # Add new symbols (without starting connections yet)
            symbols_to_add = target_symbols - current_symbols
            managers_to_start = []
            for symbol in symbols_to_add:
                try:
                    manager = self._symbol_registry.add(
                        symbol, demo=self.config.demo, auto_start=False
                    )
                    managers_to_start.append(manager)
                except ValueError:
                    pass  # Already exists

            if symbols_to_add or symbols_to_remove:
                logger.info(
                    f"Trade streams synced: +{len(symbols_to_add)} -{len(symbols_to_remove)} "
                    f"(total: {len(self._symbol_registry.symbols())})"
                )

            # Schedule staggered connection starts to avoid rate limiting
            if managers_to_start and self._loop.is_running:
                delay_ms = self.config.stream_connection_delay_ms
                self._loop.schedule(
                    self._staggered_stream_connect(managers_to_start, delay_ms)
                )

        except Exception as e:
            logger.error(f"Failed to sync trade streams: {e}")

    async def _staggered_stream_connect(
        self, managers: list, delay_ms: int
    ) -> None:
        """
        Connect trade streams with staggered delays to avoid rate limiting.

        Args:
            managers: List of SymbolManager instances to start.
            delay_ms: Delay between connections in milliseconds.
        """
        delay_s = delay_ms / 1000.0
        total = len(managers)

        logger.info(f"Starting staggered stream connections: {total} symbols, {delay_ms}ms delay")

        for i, manager in enumerate(managers, 1):
            try:
                manager.start()
                logger.debug(f"[{i}/{total}] Connected stream: {manager.symbol}")
            except Exception as e:
                logger.warning(f"Failed to start stream for {manager.symbol}: {e}")

            # Don't delay after the last connection
            if i < total:
                await asyncio.sleep(delay_s)

    async def _filter_update_loop(self) -> None:
        """
        Background loop that periodically updates active symbols.

        Runs filter pipeline at configured interval.
        """
        interval_s = self.config.filter_update_interval_seconds

        while self._running:
            try:
                await asyncio.sleep(interval_s)
                await self._update_active_symbols()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Filter update error: {e}")
                await asyncio.sleep(interval_s)

    async def _position_reconciliation_loop(self) -> None:
        """
        Background loop that reconciles positions with exchange.

        Periodically syncs with exchange to:
        - Detect externally closed positions
        - Update position sizes for partial fills
        - Re-create TP/SL if they were hit/canceled
        """
        interval_s = self.config.position_reconciliation_interval_seconds

        while self._running:
            try:
                await asyncio.sleep(interval_s)
                await self._reconcile_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position reconciliation error: {e}")
                await asyncio.sleep(interval_s)

    async def _reconcile_positions(self) -> None:
        """
        Reconcile tracked positions with exchange state.

        - Closes positions that no longer exist on exchange
        - Updates position sizes if they changed
        - Ensures TP/SL orders are correctly placed (order reconciliation)
        """
        from aequify.core.exchange.binance_futures import PositionSide
        from aequify.core.trading.order_reconciliation import reconcile_position_orders

        try:
            # Get current exchange positions
            exchange_positions = await self.client.get_positions()

            # Publish positions to PubSub for TUI
            self._publish_positions_to_pubsub(exchange_positions)

            # Get our tracked positions
            tracked_positions = await self._position_manager.get_open_positions()

            for tracked in tracked_positions:
                symbol = tracked.symbol

                # Check if position still exists on exchange
                exchange_pos = next(
                    (p for p in exchange_positions if p.symbol == symbol and p.side == tracked.side),
                    None
                )

                if exchange_pos is None or abs(exchange_pos.contracts or 0) == 0:
                    # Position closed externally
                    logger.info(f"Position {symbol} closed externally")
                    await self._exit_executor._handle_external_close(tracked, "EXTERNAL")
                    await self._risk_manager.register_position_close(symbol, 0.0)

            # Order reconciliation for all exchange positions (not just tracked)
            for pos in exchange_positions:
                contracts = abs(pos.contracts) if pos.contracts else 0.0
                if contracts == 0:
                    continue

                symbol = pos.symbol

                # Get bootstrap params if available
                bootstrap_result = self._bootstrapped_symbols.get(symbol)
                bootstrap_params = None
                if bootstrap_result:
                    # Get params for the correct direction
                    if pos.side == PositionSide.LONG and bootstrap_result.long:
                        bootstrap_params = bootstrap_result.long.params
                    elif pos.side == PositionSide.SHORT and bootstrap_result.short:
                        bootstrap_params = bootstrap_result.short.params

                # Get max position size for DCA check
                max_size = (
                    self.config.long_max_size_usdt
                    if pos.side == PositionSide.LONG
                    else self.config.short_max_size_usdt
                )

                # Reconcile TP/SL orders
                try:
                    position_closed = await reconcile_position_orders(
                        client=self.client,
                        pos=pos,
                        config_path=self._config_path,
                        bootstrap_params=bootstrap_params,
                        max_position_size_usdt=max_size,
                    )

                    if position_closed:
                        # Position was closed because it was already in profit
                        logger.info(f"Position {symbol} closed during reconciliation (TP triggered)")
                        await self._risk_manager.register_position_close(symbol, 0.0)
                except Exception as e:
                    logger.warning(f"Order reconciliation failed for {symbol}: {e}")

            # Sync risk manager
            await self._risk_manager.sync_positions()

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")

    async def _live_metrics_loop(self) -> None:
        """
        Background loop that publishes live APEX metrics to TUI.

        Periodically collects metrics from all detectors and publishes
        via PubSub for real-time TUI updates.
        """
        interval_s = 0.5  # 500ms update rate

        while self._running:
            try:
                await asyncio.sleep(interval_s)
                await self._publish_live_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Live metrics error: {e}")
                await asyncio.sleep(interval_s)

    async def _publish_live_metrics(self) -> None:
        """Publish live metrics from all detectors to TUI."""
        if not self._detector_manager:
            return

        from aequify.core.trading.publisher import get_trading_publisher

        try:
            publisher = get_trading_publisher()
            if not publisher.is_connected:
                return

            # Get metrics from all detectors
            for detector in self._detector_manager.all():
                try:
                    metrics = detector.get_current_metrics()
                    if not metrics:
                        continue

                    # Add bootstrap params for TUI context
                    bootstrap_result = self._bootstrapped_symbols.get(detector.symbol)
                    if bootstrap_result:
                        if bootstrap_result.long:
                            metrics["long_params"] = {
                                "price_move": bootstrap_result.long.params.price_move,
                                "delta_threshold": bootstrap_result.long.params.delta_threshold,
                                "is_default": bootstrap_result.long.entries == 0,
                            }
                        if bootstrap_result.short:
                            metrics["short_params"] = {
                                "price_move": bootstrap_result.short.params.price_move,
                                "delta_threshold": bootstrap_result.short.params.delta_threshold,
                                "is_default": bootstrap_result.short.entries == 0,
                            }

                    # Add trade count
                    metrics["trade_count"] = detector._trade_count
                    metrics["is_bootstrapped"] = detector.apex.is_bootstrapped

                    publisher.publish_apex_live_metrics(detector.symbol, metrics)

                except Exception as e:
                    logger.debug(f"[{detector.symbol}] Metrics publish error: {e}")

        except Exception as e:
            logger.debug(f"Live metrics publish failed: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown trading manager gracefully.

        Cancels all background tasks and stops the isolated loop.
        """
        logger.info("Shutting down TradingManager...")
        self._running = False

        # Cancel background tasks
        tasks_to_cancel = [
            self._exit_monitor_task,
            self._filter_update_task,
            self._reconciliation_task,
            self._live_metrics_task,
        ]

        for task in tasks_to_cancel:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop all trade streams
        if self._symbol_registry:
            try:
                self._symbol_registry.stop_all()
                logger.debug("Trade streams stopped")
            except Exception as e:
                logger.error(f"Error stopping trade streams: {e}")

        # Stop the isolated loop
        self._loop.stop()

        logger.info("TradingManager shutdown complete")

    async def process_signal(self, signal: TradingSignal) -> dict[str, Any]:
        """
        Process a trading signal.

        Full flow:
        1. Risk check
        2. Entry validation
        3. Entry execution (if valid)

        Args:
            signal: Trading signal to process.

        Returns:
            Result dict with status and details.
        """
        from aequify.core.exchange.binance_futures import PositionSide

        symbol = signal.symbol
        side = signal.side
        price = signal.price

        logger.info(f"Processing signal: {symbol} {side.value} @ {price:.6f}")

        # Notify callbacks
        for callback in self._signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

        # 0. Bootstrap gating - symbol must be bootstrapped before entry
        if not self.is_bootstrapped(symbol):
            logger.warning(f"Signal rejected: {symbol} not bootstrapped yet")
            return {
                "status": "rejected",
                "reason": "not_bootstrapped",
                "details": f"{symbol} must complete bootstrap before trading",
            }

        # 1. Risk check
        can_trade, risk_results = await self._risk_manager.can_trade(symbol, side)
        if not can_trade:
            failed = [r for r in risk_results if not r.passed]
            reasons = ", ".join(r.reason for r in failed)
            logger.warning(f"Risk check failed for {symbol}: {reasons}")
            return {
                "status": "rejected",
                "reason": "risk_check",
                "details": reasons,
            }

        # 2. Get current position size for DCA validation
        current_size = self._risk_manager.get_position_size(symbol)
        dca_distance = (
            self.config.long_dca_distance_pct
            if side == PositionSide.LONG
            else self.config.short_dca_distance_pct
        )

        # 3. Entry validation
        validation = await self._entry_validator.validate(
            symbol=symbol,
            side=side,
            current_price=price,
            current_position_size=current_size,
            dca_distance_pct=dca_distance,
        )

        if not validation.valid:
            logger.info(f"Entry validation failed: {validation.reason}")
            return {
                "status": "rejected",
                "reason": "validation",
                "details": validation.reason,
            }

        # 4. Execute entry
        tp_pct = (
            self.config.long_tp_pct
            if side == PositionSide.LONG
            else self.config.short_tp_pct
        )
        sl_pct = (
            self.config.long_sl_pct
            if side == PositionSide.LONG
            else self.config.short_sl_pct
        )

        # Check if DCA or new entry
        existing_position = await self._position_manager.get_position_by_symbol(symbol)
        if existing_position and existing_position.is_open:
            # DCA entry
            result = await self._entry_executor.execute_dca(
                position=existing_position,
                current_price=price,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
            )
        else:
            # New entry
            result = await self._entry_executor.execute(
                symbol=symbol,
                side=side,
                current_price=price,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                signal_params=signal.params,
                max_hold_time_ms=int(self.config.timeout_seconds * 1000),
            )

        if result.success:
            # Invalidate entry validator cache
            self._entry_validator.invalidate_cache(symbol)
            return {
                "status": "executed",
                "position_id": result.position_id,
                "details": result.details,
            }
        else:
            return {
                "status": "failed",
                "reason": "execution",
                "details": result.error,
            }

    def process_signal_async(self, signal: TradingSignal) -> None:
        """
        Process a trading signal asynchronously (non-blocking).

        For use from Mojo code - fires and forgets.
        Schedules the signal processing in the IsolatedLoop.

        Args:
            signal: Trading signal to process.
        """
        if self._loop and self._loop.is_running:
            self._loop.schedule(self.process_signal(signal))

    async def close_position(
        self,
        symbol: str,
        reason: str = "MANUAL",
    ) -> dict[str, Any]:
        """
        Close a position.

        Args:
            symbol: Trading pair to close.
            reason: Exit reason.

        Returns:
            Result dict.
        """
        result = await self._exit_executor.execute_by_symbol(symbol, reason)

        if result.success:
            return {
                "status": "closed",
                "pnl": result.realized_pnl,
                "pnl_pct": result.realized_pnl_pct,
            }
        else:
            return {
                "status": "failed",
                "error": result.error,
            }

    def close_position_async(self, symbol: str, reason: str = "MANUAL") -> None:
        """
        Close a position asynchronously (non-blocking).

        Schedules the position close in the IsolatedLoop.
        """
        if self._loop and self._loop.is_running:
            self._loop.schedule(self.close_position(symbol, reason))

    async def close_all_positions(self, reason: str = "MANUAL") -> list[dict[str, Any]]:
        """
        Close all open positions.

        Args:
            reason: Exit reason.

        Returns:
            List of result dicts.
        """
        results = await self._exit_executor.execute_close_all(reason)
        return [
            {
                "symbol": r.details.get("symbol") if r.details else None,
                "status": "closed" if r.success else "failed",
                "pnl": r.realized_pnl,
                "error": r.error if not r.success else None,
            }
            for r in results
        ]

    async def _exit_monitor_loop(self) -> None:
        """
        Background loop that monitors positions for exit conditions.

        Checks:
        - Timeout
        - Take profit / Stop loss levels
        - Smart close conditions
        """
        interval_s = self.config.exit_check_interval_ms / 1000

        while self._running:
            try:
                await self._check_exits()
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Exit monitor error: {e}")
                await asyncio.sleep(interval_s)

    async def _check_exits(self) -> None:
        """Check all open positions for exit conditions."""
        positions = await self._position_manager.get_open_positions()

        for position in positions:
            try:
                # Get current price
                ticker = await self.client.fetch_ticker(position.symbol)
                current_price = float(ticker.get("last", 0))

                if current_price <= 0:
                    continue

                # Get max size for smart close check
                from aequify.core.exchange.binance_futures import PositionSide

                max_size = (
                    self.config.long_max_size_usdt
                    if position.side == PositionSide.LONG
                    else self.config.short_max_size_usdt
                )

                # Check exit conditions
                result = self._exit_validator.validate(
                    symbol=position.symbol,
                    side=position.side,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    entry_time=position.entry_time.timestamp() if position.entry_time else time.time(),
                    current_position_size=self._risk_manager.get_position_size(position.symbol),
                    max_position_size=max_size,
                )

                if result.should_exit:
                    logger.info(
                        f"Exit condition met for {position.symbol}: {result.exit_type} - {result.reason}"
                    )
                    await self._exit_executor.execute(
                        position, result.exit_type.upper()
                    )
                    # Reset trailing stop state
                    self._exit_validator.reset_trailing_stop(position.symbol, position.side)

            except Exception as e:
                logger.error(f"Exit check failed for {position.symbol}: {e}")

    def on_signal(self, callback: SignalCallback) -> None:
        """
        Register callback for incoming signals.

        Args:
            callback: Function to call with each signal.
        """
        self._signal_callbacks.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get trading statistics."""
        position_stats = self._position_manager.get_stats() if self._position_manager else {}
        risk_stats = self._risk_manager.get_stats() if self._risk_manager else {}

        return {
            "positions": position_stats,
            "risk": risk_stats,
            "running": self._running,
            "active_symbols_count": len(self._active_symbols),
            "active_symbols": list(self._active_symbols),
            "position_sides": {k: list(v) for k, v in self._position_sides.items()},
            "uptime_seconds": self.uptime_seconds,
            "filter_enabled": self._filter_pipeline is not None,
        }

    @property
    def position_manager(self):
        """Get position manager."""
        return self._position_manager

    @property
    def risk_manager(self):
        """Get risk manager."""
        return self._risk_manager

    @property
    def entry_validator(self):
        """Get entry validator."""
        return self._entry_validator

    @property
    def exit_validator(self):
        """Get exit validator."""
        return self._exit_validator

    @property
    def filter_pipeline(self):
        """Get filter pipeline."""
        return self._filter_pipeline

    @property
    def active_symbols(self) -> set[str]:
        """
        Get current active symbols from filter pipeline.

        These are symbols that passed all filters and include
        any symbols with existing positions.
        """
        return self._active_symbols.copy()

    @property
    def position_sides(self) -> dict[str, set[str]]:
        """
        Get position sides for symbols with open positions.

        Returns dict of symbol -> set of sides ("LONG", "SHORT").
        In hedge mode, a symbol can have both.
        """
        return {k: v.copy() for k, v in self._position_sides.items()}

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds since initialization."""
        if not self._start_time:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def is_symbol_active(self, symbol: str) -> bool:
        """Check if a symbol is in the active set."""
        return symbol in self._active_symbols

    async def force_filter_update(self) -> None:
        """
        Force an immediate filter pipeline update.

        Useful when you need to refresh symbols immediately.
        """
        await self._update_active_symbols()

    # =========================================================================
    # Sequential Symbol Initialization (Backfill → Bootstrap)
    # =========================================================================

    def set_config_path(self, config_path: str) -> None:
        """Set the config path for loading APEX configs."""
        self._config_path = config_path

    def is_bootstrapped(self, symbol: str) -> bool:
        """Check if a symbol has completed bootstrap."""
        return symbol in self._bootstrapped_symbols

    def get_bootstrap_result(self, symbol: str) -> "BootstrapResult | None":
        """Get bootstrap result for a symbol."""
        return self._bootstrapped_symbols.get(symbol)

    async def initialize_symbol(
        self,
        symbol: str,
        force_backfill: bool = False,
        force_bootstrap: bool = False,
    ) -> "BootstrapResult | None":
        """
        Initialize a symbol for trading (backfill → bootstrap).

        Uses a lock to ensure only 1 symbol initializes at a time.
        This prevents memory spikes and QuestDB contention.

        Flow:
        1. Check cold store for cached bootstrap results (LONG + SHORT)
        2. If cache valid (< max_cache_valid_days), use it
        3. Else: backfill from CSV → load trades → bootstrap → save to cold store

        Args:
            symbol: Trading pair (CCXT format).
            force_backfill: Force backfill even if data exists.
            force_bootstrap: Force bootstrap even if cache is valid.

        Returns:
            BootstrapResult if successful, None if failed.
        """
        import numpy as np

        from aequify.core.apex import (
            BootstrapResult,
            DirectionResult,
            bootstrap_symbol,
            load_apex_config_for_symbol,
        )
        from aequify.core.db.cold import ColdStore
        from aequify.core.db.cold.backfill import BackfillConfig
        from aequify.core.db.cold.models import Direction

        # Acquire lock (blocking - only 1 symbol at a time)
        with self._symbol_init_lock:
            logger.info(f"[{symbol}] Starting symbol initialization...")

            # Publish checking_cache stage
            self._publish_init_stage(symbol, "checking_cache")

            try:
                # Load config for this symbol (with overrides)
                config = load_apex_config_for_symbol(self._config_path, symbol)

                async with ColdStore() as store:
                    # 1. Check cold store for cached bootstrap results (both directions)
                    if not force_bootstrap:
                        cached_long = await store.get_latest_bootstrap_result(
                            symbol, Direction.LONG
                        )
                        cached_short = await store.get_latest_bootstrap_result(
                            symbol, Direction.SHORT
                        )

                        # Check if at least one direction has valid cache
                        if cached_long or cached_short:
                            # Use the older timestamp to check validity
                            oldest_ts = min(
                                cached_long.timestamp_ms if cached_long else float("inf"),
                                cached_short.timestamp_ms if cached_short else float("inf"),
                            )
                            cache_age_days = (time.time() * 1000 - oldest_ts) / (
                                1000 * 60 * 60 * 24
                            )

                            if cache_age_days < config.max_cache_valid_days:
                                # Construct combined BootstrapResult from cached data
                                result = BootstrapResult(
                                    long=DirectionResult.from_cold_store(cached_long)
                                    if cached_long
                                    else None,
                                    short=DirectionResult.from_cold_store(cached_short)
                                    if cached_short
                                    else None,
                                )
                                logger.info(
                                    f"[{symbol}] Using cached bootstrap "
                                    f"(age: {cache_age_days:.1f} days, "
                                    f"LONG={result.long is not None}, "
                                    f"SHORT={result.short is not None})"
                                )
                                self._bootstrapped_symbols[symbol] = result
                                self._publish_apex_state(symbol, result, source="cached")
                                self._create_live_detector(symbol, result)
                                return result

                            logger.info(
                                f"[{symbol}] Cache expired "
                                f"({cache_age_days:.1f} > {config.max_cache_valid_days} days)"
                            )

                    # 2. Backfill from CSV with progress publishing
                    from aequify.core.db.cold.backfill import BackfillManager

                    backfill_config = BackfillConfig(
                        lookback_days=config.bootstrap.trades_lookback_days,
                        skip_existing=not force_backfill,
                    )

                    # Publish backfill progress to TUI via PubSub
                    def on_backfill_progress(sym: str, progress: BackfillProgress) -> None:
                        self._publish_backfill_progress(sym, progress)

                    logger.info(f"[{symbol}] Starting CSV backfill...")

                    # Use BackfillManager directly to get progress callbacks
                    async with BackfillManager(store=store, config=backfill_config) as manager:
                        manager.on_progress(on_backfill_progress)
                        manager.start_backfill(symbol)

                        # Wait for completion
                        while manager.is_running(symbol):
                            await asyncio.sleep(0.5)

                        backfill_result = manager.get_progress(symbol)

                    if backfill_result is None or backfill_result.total_trades == 0:
                        logger.warning(f"[{symbol}] No trades backfilled")

                    # Clear backfill progress in TUI
                    self._clear_backfill_progress(symbol)

                    logger.info(
                        f"[{symbol}] Backfill complete: {backfill_result.total_trades if backfill_result else 0:,} trades"
                    )

                    # 3. Validate min_lookback_days
                    min_lookback_days = config.bootstrap.min_lookback_days
                    if min_lookback_days > 0:
                        ts_range = await store.get_timestamp_range(symbol)
                        if ts_range is None:
                            logger.warning(
                                f"[{symbol}] No trades in cold store, cannot validate min_lookback_days"
                            )
                            return None

                        min_ts, max_ts = ts_range
                        MS_PER_DAY = 1000 * 60 * 60 * 24
                        available_days = (max_ts - min_ts) / MS_PER_DAY

                        if available_days < min_lookback_days:
                            logger.warning(
                                f"[{symbol}] Insufficient data for bootstrap: "
                                f"{available_days:.1f} days < {min_lookback_days} days required"
                            )
                            return None

                        logger.info(
                            f"[{symbol}] Data range validated: {available_days:.1f} days available"
                        )

                    # 4. Load trades from cold store as numpy arrays
                    trades = await store.get_trades(
                        symbol, limit=config.bootstrap.max_trades or 500000
                    )

                    if not trades:
                        logger.error(f"[{symbol}] No trades in cold store")
                        return None

                    # Convert to numpy arrays for bootstrap
                    timestamps = np.array([t.timestamp_ms for t in trades], dtype=np.int64)
                    prices = np.array([t.price for t in trades], dtype=np.float64)
                    quantities = np.array([t.quantity for t in trades], dtype=np.float64)
                    # sides: 1=buy, -1=sell (is_buyer_maker means seller was taker)
                    sides = np.array(
                        [-1 if t.is_buyer_maker else 1 for t in trades], dtype=np.int8
                    )

                    logger.info(f"[{symbol}] Loaded {len(trades):,} trades for bootstrap")

                    # 5. Run bootstrap
                    self._publish_init_stage(symbol, "bootstrapping")
                    logger.info(f"[{symbol}] Starting bootstrap...")
                    bootstrap_result = await bootstrap_symbol(
                        symbol=symbol,
                        timestamps=timestamps,
                        prices=prices,
                        quantities=quantities,
                        sides=sides,
                        config=config,
                    )

                    # 6. Save both directions to cold store
                    if bootstrap_result.long:
                        await store.insert_bootstrap_result(
                            bootstrap_result.long.to_cold_store(symbol)
                        )
                    if bootstrap_result.short:
                        await store.insert_bootstrap_result(
                            bootstrap_result.short.to_cold_store(symbol)
                        )
                    logger.info(f"[{symbol}] Bootstrap saved to cold store")

                    # 6b. Generate override file (for persistence across restarts)
                    self._generate_override_file(symbol, bootstrap_result, trade_count=len(timestamps))

                    # 7. Store in memory and publish to TUI
                    self._bootstrapped_symbols[symbol] = bootstrap_result
                    self._publish_apex_state(symbol, bootstrap_result, source="bootstrap")
                    self._create_live_detector(symbol, bootstrap_result)

                    logger.info(
                        f"[{symbol}] Initialization complete: "
                        f"LONG={bootstrap_result.long is not None}, "
                        f"SHORT={bootstrap_result.short is not None}"
                    )

                    return bootstrap_result

            except Exception as e:
                logger.error(f"[{symbol}] Initialization failed: {e}")
                return None

    async def initialize_symbols(
        self,
        symbols: list[str],
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, "BootstrapResult | None"]:
        """
        Initialize multiple symbols sequentially.

        Args:
            symbols: List of trading pairs.
            on_progress: Optional callback(symbol, current, total).

        Returns:
            Dict of symbol -> BootstrapResult (or None if failed).
        """
        results: dict[str, "BootstrapResult | None"] = {}
        total = len(symbols)

        # Publish "queued" status for all symbols at the start
        for i, symbol in enumerate(symbols, 1):
            self._publish_init_stage(symbol, "queued", queue_position=i, queue_total=total)

        for i, symbol in enumerate(symbols, 1):
            if on_progress:
                on_progress(symbol, i, total)

            results[symbol] = await self.initialize_symbol(symbol)

        return results


# Convenience function to create manager from config dict
async def create_trading_manager(
    client: "BinanceFuturesClient",
    config: dict[str, Any],
) -> TradingManager:
    """
    Create and initialize a trading manager.

    Args:
        client: Exchange client.
        config: Full config dict.

    Returns:
        Initialized TradingManager.
    """
    trading_config = TradingConfig.from_config(config)
    manager = TradingManager(client, trading_config)
    await manager.initialize()
    return manager
