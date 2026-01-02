"""
APEX Signal Handler - Routes APEX signals to TradingManager.

Processes real-time signal state from APEXLiveDetector and routes valid
signals to the trading engine for execution.

Signal Flow:
    Trade Stream -> APEXLiveDetector -> APEXSignalHandler -> TradingManager

Entry Rules:
- Respects long_enabled / short_enabled config
- Prevents duplicate entries (cooldown period)
- Validates minimum trades for statistical significance
- Validates bootstrap completion

Usage:
    handler = APEXSignalHandler(config)
    handler.set_trading_manager(trading_manager)

    # Called from APEXLiveDetector when signal state changes
    await handler.on_signal_state(state)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.apex.config import VADParameters
    from aequify.core.trading import TradingManager, TradingSignal

logger = get_logger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal handling."""

    # Enable/disable per direction
    long_enabled: bool = True
    short_enabled: bool = True

    # Cooldown between signals for same symbol/side (seconds)
    signal_cooldown_seconds: float = 60.0

    # Minimum trades before signal is considered valid
    min_trades_for_signal: int = 100

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SignalConfig:
        """Create from config dict."""
        trading = config.get("trading", {})
        futures = trading.get("futures", {})
        long_cfg = futures.get("long", {})
        short_cfg = futures.get("short", {})
        apex = config.get("engines", {}).get("apex", {})

        return cls(
            long_enabled=long_cfg.get("enabled", True),
            short_enabled=short_cfg.get("enabled", True),
            signal_cooldown_seconds=apex.get("signal_cooldown_seconds", 60.0),
            min_trades_for_signal=apex.get("min_trades_for_signal", 100),
        )


@dataclass
class SignalState:
    """
    Real-time signal state from APEXLiveDetector.

    Contains computed metrics and signal readiness flags.

    Attributes:
        symbol: Trading pair symbol.
        is_bootstrapped: Whether APEX bootstrap is complete.
        trade_count: Total trades processed.
        current_price: Current trade price.

        # LONG signal state
        long_signal_ready: Whether LONG conditions are met.
        long_params: Optimized LONG parameters (if bootstrapped).
        long_volume_delta: Volume delta % for LONG (from recent high).
        price_move_from_high: Price move % from rolling high.

        # SHORT signal state
        short_signal_ready: Whether SHORT conditions are met.
        short_params: Optimized SHORT parameters (if bootstrapped).
        short_volume_delta: Volume delta % for SHORT (from recent low).
        price_move_from_low: Price move % from rolling low.
    """

    symbol: str
    is_bootstrapped: bool = False
    trade_count: int = 0
    current_price: float = 0.0

    # LONG signal
    long_signal_ready: bool = False
    long_params: "VADParameters | None" = None
    long_volume_delta: float = 0.0
    price_move_from_high: float = 0.0

    # SHORT signal
    short_signal_ready: bool = False
    short_params: "VADParameters | None" = None
    short_volume_delta: float = 0.0
    price_move_from_low: float = 0.0


@dataclass
class SignalRecord:
    """Record of a signal for history tracking."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    timestamp: float
    executed: bool = False
    price: float = 0.0
    params: dict[str, Any] | None = None


# Callback type for signal execution
SignalCallback = Callable[
    ["TradingSignal"], Coroutine[Any, Any, dict[str, Any]]
]


class APEXSignalHandler:
    """
    Handles APEX signals and routes to TradingManager.

    Thread-safe signal processing with cooldown management.

    Usage:
        handler = APEXSignalHandler(config)
        handler.set_trading_manager(trading_manager)

        # Called from APEXLiveDetector
        await handler.on_signal_state(state)
    """

    def __init__(self, config: SignalConfig | None = None) -> None:
        """
        Initialize signal handler.

        Args:
            config: Signal handling configuration.
        """
        self.config = config or SignalConfig()

        # Cooldown tracking: symbol -> {side -> last_signal_time}
        self._last_signal_time: dict[str, dict[str, float]] = {}

        # Signal history for debugging
        self._signal_history: list[SignalRecord] = []
        self._max_history = 100

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Trading manager reference
        self._trading_manager: "TradingManager | None" = None

        # Stats
        self._signals_received = 0
        self._signals_executed = 0
        self._signals_rejected_cooldown = 0
        self._signals_rejected_disabled = 0
        self._signals_rejected_min_trades = 0

    def set_trading_manager(self, manager: "TradingManager") -> None:
        """
        Set the trading manager for signal execution.

        Args:
            manager: TradingManager instance.
        """
        self._trading_manager = manager

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> APEXSignalHandler:
        """
        Create handler from config dict.

        Args:
            config: Full config dict.

        Returns:
            Configured APEXSignalHandler.
        """
        signal_config = SignalConfig.from_config(config)
        return cls(signal_config)

    async def on_signal_state(self, state: SignalState) -> None:
        """
        Process signal state update from APEXLiveDetector.

        Called on signal state changes (not every trade).
        Checks for valid signals and executes if conditions met.

        Args:
            state: Current signal state snapshot.
        """
        self._signals_received += 1

        # CRITICAL: Bootstrap must be complete before any signals are valid
        if not state.is_bootstrapped:
            return

        # Check for LONG signal
        if state.long_signal_ready and state.long_params:
            await self._process_signal(
                symbol=state.symbol,
                side="LONG",
                params=state.long_params,
                price=state.current_price,
                trade_count=state.trade_count,
                volume_delta=state.long_volume_delta,
                price_move=state.price_move_from_high,
            )

        # Check for SHORT signal
        if state.short_signal_ready and state.short_params:
            await self._process_signal(
                symbol=state.symbol,
                side="SHORT",
                params=state.short_params,
                price=state.current_price,
                trade_count=state.trade_count,
                volume_delta=state.short_volume_delta,
                price_move=state.price_move_from_low,
            )

    async def _process_signal(
        self,
        symbol: str,
        side: str,
        params: "VADParameters",
        price: float,
        trade_count: int,
        volume_delta: float,
        price_move: float,
    ) -> None:
        """
        Process a single signal.

        Args:
            symbol: Trading pair.
            side: "LONG" or "SHORT".
            params: Optimized VAD parameters.
            price: Current price.
            trade_count: Trade count for validation.
            volume_delta: Volume delta percentage.
            price_move: Price move percentage.
        """
        # Check if direction is enabled
        if side == "LONG" and not self.config.long_enabled:
            self._signals_rejected_disabled += 1
            return
        if side == "SHORT" and not self.config.short_enabled:
            self._signals_rejected_disabled += 1
            return

        # Check minimum trades
        if trade_count < self.config.min_trades_for_signal:
            self._signals_rejected_min_trades += 1
            logger.debug(
                f"Signal ignored for {symbol} {side}: "
                f"only {trade_count} trades (need {self.config.min_trades_for_signal})"
            )
            return

        # Check cooldown
        if not await self._check_cooldown(symbol, side):
            self._signals_rejected_cooldown += 1
            return

        # Execute signal via TradingManager
        if self._trading_manager:
            try:
                from aequify.core.exchange.binance_futures import PositionSide
                from aequify.core.trading import TradingSignal

                position_side = PositionSide.LONG if side == "LONG" else PositionSide.SHORT

                signal = TradingSignal(
                    symbol=symbol,
                    side=position_side,
                    price=price,
                    timestamp=time.time(),
                    params={
                        "volume_delta": volume_delta,
                        "price_move": price_move,
                        "trade_count": trade_count,
                        "target_profit": params.target_profit,
                        "stop_loss": params.stop_loss,
                        "max_hold_time_ms": params.max_hold_time,
                        "dca_distance_pct": params.dca_distance_pct,
                    },
                )

                logger.info(
                    f"APEX Signal: {side} {symbol} @ {price:.6f} | "
                    f"delta={volume_delta:+.1f}% | "
                    f"pm={price_move:+.2f}% | "
                    f"TP={params.target_profit:.1f}% SL={params.stop_loss:.1f}%"
                )

                # Execute via TradingManager (non-blocking if using IsolatedLoop)
                result = await self._trading_manager.process_signal(signal)

                if result.get("status") == "executed":
                    self._signals_executed += 1
                    await self._record_signal(symbol, side, executed=True, price=price)
                    logger.info(f"Signal executed: {side} {symbol}")
                else:
                    await self._record_signal(symbol, side, executed=False, price=price)
                    logger.info(
                        f"Signal rejected by trading manager: {side} {symbol} - "
                        f"{result.get('reason', 'unknown')}: {result.get('details', '')}"
                    )

            except Exception as e:
                logger.error(f"Signal execution error for {symbol} {side}: {e}")
                await self._record_signal(symbol, side, executed=False, price=price)
        else:
            logger.warning("No trading manager set, signal ignored")

    async def _check_cooldown(self, symbol: str, side: str) -> bool:
        """
        Check if signal is allowed (not in cooldown).

        Args:
            symbol: Trading pair.
            side: "LONG" or "SHORT".

        Returns:
            True if signal is allowed, False if in cooldown.
        """
        async with self._lock:
            now = time.time()

            if symbol not in self._last_signal_time:
                self._last_signal_time[symbol] = {}

            last_time = self._last_signal_time[symbol].get(side, 0)
            elapsed = now - last_time

            if elapsed < self.config.signal_cooldown_seconds:
                logger.debug(
                    f"Signal cooldown for {symbol} {side}: "
                    f"{elapsed:.1f}s < {self.config.signal_cooldown_seconds}s"
                )
                return False

            # Update last signal time
            self._last_signal_time[symbol][side] = now
            return True

    async def _record_signal(
        self,
        symbol: str,
        side: str,
        executed: bool,
        price: float = 0.0,
    ) -> None:
        """Record signal in history."""
        async with self._lock:
            record = SignalRecord(
                symbol=symbol,
                side=side,
                timestamp=time.time(),
                executed=executed,
                price=price,
            )
            self._signal_history.append(record)

            # Trim history
            if len(self._signal_history) > self._max_history:
                self._signal_history = self._signal_history[-self._max_history:]

    def get_stats(self) -> dict[str, Any]:
        """Get signal handler statistics."""
        return {
            "signals_received": self._signals_received,
            "signals_executed": self._signals_executed,
            "signals_rejected_cooldown": self._signals_rejected_cooldown,
            "signals_rejected_disabled": self._signals_rejected_disabled,
            "signals_rejected_min_trades": self._signals_rejected_min_trades,
            "recent_signals": len(self._signal_history),
            "config": {
                "long_enabled": self.config.long_enabled,
                "short_enabled": self.config.short_enabled,
                "cooldown_seconds": self.config.signal_cooldown_seconds,
                "min_trades": self.config.min_trades_for_signal,
            },
        }

    def reset_cooldown(self, symbol: str | None = None, side: str | None = None) -> None:
        """
        Reset cooldown for symbol/side.

        Args:
            symbol: Symbol to reset, or None for all.
            side: Side to reset, or None for both.
        """
        if symbol is None:
            self._last_signal_time.clear()
        elif symbol in self._last_signal_time:
            if side is None:
                self._last_signal_time[symbol].clear()
            elif side in self._last_signal_time[symbol]:
                del self._last_signal_time[symbol][side]

    def get_recent_signals(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent signal history.

        Args:
            symbol: Filter by symbol, or None for all.
            limit: Maximum number of signals to return.

        Returns:
            List of signal records as dicts.
        """
        signals = self._signal_history
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return [
            {
                "symbol": s.symbol,
                "side": s.side,
                "timestamp": s.timestamp,
                "executed": s.executed,
                "price": s.price,
            }
            for s in signals[-limit:]
        ]

    def update_config(
        self,
        long_enabled: bool | None = None,
        short_enabled: bool | None = None,
        cooldown_seconds: float | None = None,
        min_trades: int | None = None,
    ) -> None:
        """
        Update configuration at runtime.

        Args:
            long_enabled: Enable/disable LONG signals.
            short_enabled: Enable/disable SHORT signals.
            cooldown_seconds: Update cooldown period.
            min_trades: Update minimum trades requirement.
        """
        if long_enabled is not None:
            self.config.long_enabled = long_enabled
            logger.info(f"LONG signals {'enabled' if long_enabled else 'disabled'}")

        if short_enabled is not None:
            self.config.short_enabled = short_enabled
            logger.info(f"SHORT signals {'enabled' if short_enabled else 'disabled'}")

        if cooldown_seconds is not None:
            self.config.signal_cooldown_seconds = cooldown_seconds
            logger.info(f"Signal cooldown updated to {cooldown_seconds}s")

        if min_trades is not None:
            self.config.min_trades_for_signal = min_trades
            logger.info(f"Minimum trades updated to {min_trades}")
