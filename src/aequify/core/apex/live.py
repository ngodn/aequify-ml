"""
APEX Live Detector - Real-time signal detection from trade stream.

Signal detection using same logic as bootstrap GPU kernels:
1. Track rolling high/low per time window
2. Calculate volume delta over SAME time window
3. Compare against bootstrap thresholds
4. Emit signal when conditions met

Volume delta is calculated as: (buy_volume - sell_volume) / total_volume * 100
This matches the gpu_volume_delta_multi kernel in bootstrap.

Architecture:
    Trade Stream (via SymbolManager)
        └─ APEXLiveDetector.on_trade()
            ├─ Update rolling high/low (per time window)
            ├─ Update volume delta (per time window - SAME window)
            └─ Check signals (compare vs bootstrap thresholds)
                └─ APEXSignalHandler.on_signal_state()

Usage:
    detector = APEXLiveDetector(
        symbol="BTC/USDT:USDT",
        apex=apex,  # Bootstrapped APEX instance
        signal_handler=signal_handler,
        config=config,
    )

    # Called on each trade from SymbolManager
    detector.on_trade(trade)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.apex import APEX
    from aequify.core.apex.config import VADParameters
    from aequify.core.apex.signal_handler import APEXSignalHandler

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LiveDetectorConfig:
    """Configuration for live signal detection."""

    # Maximum trades to keep in memory for volume delta calculation
    max_trades_buffer: int = 500000

    # Signal offset (triggers slightly before/after threshold)
    long_signal_offset: float = 0.0  # Added to price_move threshold
    short_signal_offset: float = 0.0  # Subtracted from price_move threshold

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LiveDetectorConfig:
        """Create from config dict."""
        apex_cfg = config.get("engines", {}).get("apex", {})

        return cls(
            max_trades_buffer=apex_cfg.get("max_trades_buffer", 500000),
            long_signal_offset=apex_cfg.get("bootstrap", {}).get("long", {}).get("signal_offset", 0.0),
            short_signal_offset=apex_cfg.get("bootstrap", {}).get("short", {}).get("signal_offset", 0.0),
        )


# =============================================================================
# Trade Record
# =============================================================================


@dataclass(slots=True)
class TradeRecord:
    """Minimal trade record for rolling calculations."""

    timestamp_ms: int
    price: float
    buy_qty: float  # Quantity if taker bought (is_buyer_maker=False)
    sell_qty: float  # Quantity if taker sold (is_buyer_maker=True)


# =============================================================================
# Rolling Window Tracker (combines extreme + volume delta)
# =============================================================================


class RollingWindowTracker:
    """
    Tracks rolling metrics within a time window.

    Tracks both:
    - Rolling extreme (high or low)
    - Volume delta over the SAME window

    This matches how bootstrap GPU kernels work:
    - rolling_high/rolling_low uses window X
    - volume_delta_multi uses the SAME window X
    """

    def __init__(self, window_ms: int, is_high: bool = True):
        """
        Args:
            window_ms: Time window in milliseconds.
            is_high: True for rolling high, False for rolling low.
        """
        self.window_ms = window_ms
        self.is_high = is_high

        # Store (timestamp_ms, price, buy_qty, sell_qty)
        self._data: deque[tuple[int, float, float, float]] = deque()

        # Cached values
        self._extreme_value: float = 0.0 if is_high else float("inf")
        self._total_buy: float = 0.0
        self._total_sell: float = 0.0

    def update(self, timestamp_ms: int, price: float, buy_qty: float, sell_qty: float) -> None:
        """Update with new trade."""
        # Remove stale entries and adjust totals
        cutoff = timestamp_ms - self.window_ms
        while self._data and self._data[0][0] < cutoff:
            old = self._data.popleft()
            self._total_buy -= old[2]
            self._total_sell -= old[3]

        # Add new entry
        self._data.append((timestamp_ms, price, buy_qty, sell_qty))
        self._total_buy += buy_qty
        self._total_sell += sell_qty

        # Recompute extreme (could optimize with monotonic deque but simple is fine)
        if self._data:
            if self.is_high:
                self._extreme_value = max(p for _, p, _, _ in self._data)
            else:
                self._extreme_value = min(p for _, p, _, _ in self._data)
        else:
            self._extreme_value = 0.0 if self.is_high else float("inf")

    @property
    def extreme_value(self) -> float:
        """Current rolling high/low value."""
        return self._extreme_value

    @property
    def volume_delta(self) -> float:
        """
        Volume delta over this window.

        Returns: (buy - sell) / total * 100
        - Negative = selling pressure (more sells than buys)
        - Positive = buying pressure (more buys than sells)
        """
        total = self._total_buy + self._total_sell
        if total <= 0:
            return 0.0
        return ((self._total_buy - self._total_sell) / total) * 100.0

    def price_move_pct(self, current_price: float) -> float:
        """
        Calculate price move percentage from extreme.

        For high: (current - high) / high * 100 (negative when below high)
        For low: (current - low) / low * 100 (positive when above low)
        """
        if self._extreme_value <= 0 or self._extreme_value == float("inf"):
            return 0.0
        return (current_price - self._extreme_value) / self._extreme_value * 100.0


# =============================================================================
# APEX Live Detector
# =============================================================================


class APEXLiveDetector:
    """
    Real-time signal detector using same logic as bootstrap.

    On each trade:
    1. Update rolling high/low per time window
    2. Update volume delta per SAME time window
    3. Compare against bootstrap thresholds
    4. Emit signal if conditions met
    """

    def __init__(
        self,
        symbol: str,
        apex: "APEX",
        signal_handler: "APEXSignalHandler | None" = None,
        config: LiveDetectorConfig | None = None,
    ) -> None:
        """
        Initialize live detector.

        Args:
            symbol: Trading pair symbol.
            apex: Bootstrapped APEX instance with optimized parameters.
            signal_handler: Signal handler for emitting signals.
            config: Detection configuration.
        """
        self.symbol = symbol
        self.apex = apex
        self.signal_handler = signal_handler
        self.config = config or LiveDetectorConfig()

        # Rolling trackers - one HIGH and one LOW per time window
        # HIGH tracker for LONG signals (price drop from high + selling pressure)
        # LOW tracker for SHORT signals (price rise from low + buying pressure)
        self._high_trackers: dict[int, RollingWindowTracker] = {}
        self._low_trackers: dict[int, RollingWindowTracker] = {}

        # Initialize trackers for each time window from APEX config
        for tw_ms in apex.config.time_windows_ms:
            self._high_trackers[tw_ms] = RollingWindowTracker(tw_ms, is_high=True)
            self._low_trackers[tw_ms] = RollingWindowTracker(tw_ms, is_high=False)

        # Current state
        self._current_price: float = 0.0
        self._current_timestamp_ms: int = 0
        self._trade_count: int = 0

        # Stats
        self._signals_emitted: int = 0

        logger.info(f"[{symbol}] Live detector initialized with {len(apex.config.time_windows_ms)} time windows")

    def on_trade(
        self,
        price: float,
        quantity: float,
        timestamp_ms: int,
        is_buyer_maker: bool,
    ) -> None:
        """
        Process a trade and check for signals.

        Args:
            price: Trade price.
            quantity: Trade quantity.
            timestamp_ms: Trade timestamp in milliseconds.
            is_buyer_maker: True if buyer was maker (taker sold).
        """
        self._current_price = price
        self._current_timestamp_ms = timestamp_ms
        self._trade_count += 1

        # is_buyer_maker=True means taker sold (selling pressure)
        buy_qty = quantity if not is_buyer_maker else 0.0
        sell_qty = quantity if is_buyer_maker else 0.0

        # Update all trackers with the trade
        for tracker in self._high_trackers.values():
            tracker.update(timestamp_ms, price, buy_qty, sell_qty)
        for tracker in self._low_trackers.values():
            tracker.update(timestamp_ms, price, buy_qty, sell_qty)

        # Check signals if bootstrapped
        if self.apex.is_bootstrapped:
            self._check_signals()

    def _check_signals(self) -> None:
        """Check if LONG or SHORT signal conditions are met."""
        long_params = self.apex.params.long
        short_params = self.apex.params.short

        # Check LONG signal
        # Uses the time_window from bootstrap result
        if long_params:
            tw_ms = long_params.time_window
            tracker = self._high_trackers.get(tw_ms)

            if tracker:
                pm = tracker.price_move_pct(self._current_price)
                delta = tracker.volume_delta

                # LONG: price drops (negative pm) + selling pressure (negative delta)
                long_trigger = long_params.price_move + self.config.long_signal_offset
                price_ok = pm <= long_trigger
                delta_ok = delta <= long_params.delta_threshold

                if price_ok and delta_ok:
                    self._emit_signal(
                        side="LONG",
                        price_move=pm,
                        volume_delta=delta,
                        time_window_ms=tw_ms,
                        params=long_params,
                    )

        # Check SHORT signal
        if short_params:
            tw_ms = short_params.time_window
            tracker = self._low_trackers.get(tw_ms)

            if tracker:
                pm = tracker.price_move_pct(self._current_price)
                delta = tracker.volume_delta

                # SHORT: price rises (positive pm) + buying pressure (positive delta)
                short_trigger = short_params.price_move - self.config.short_signal_offset
                price_ok = pm >= short_trigger
                delta_ok = delta >= short_params.delta_threshold

                if price_ok and delta_ok:
                    self._emit_signal(
                        side="SHORT",
                        price_move=pm,
                        volume_delta=delta,
                        time_window_ms=tw_ms,
                        params=short_params,
                    )

    def _emit_signal(
        self,
        side: str,
        price_move: float,
        volume_delta: float,
        time_window_ms: int,
        params: "VADParameters",
    ) -> None:
        """Emit signal to the handler."""
        if not self.signal_handler:
            return

        from aequify.core.apex.signal_handler import SignalState

        state = SignalState(
            symbol=self.symbol,
            is_bootstrapped=True,
            trade_count=self._trade_count,
            current_price=self._current_price,
            long_signal_ready=(side == "LONG"),
            long_params=params if side == "LONG" else None,
            long_volume_delta=volume_delta if side == "LONG" else 0.0,
            price_move_from_high=price_move if side == "LONG" else 0.0,
            short_signal_ready=(side == "SHORT"),
            short_params=params if side == "SHORT" else None,
            short_volume_delta=volume_delta if side == "SHORT" else 0.0,
            price_move_from_low=price_move if side == "SHORT" else 0.0,
        )

        # Use asyncio.create_task if in async context, else just call
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.signal_handler.on_signal_state(state))
        except RuntimeError:
            # No running loop, call sync (signal handler will handle it)
            asyncio.run(self.signal_handler.on_signal_state(state))

        self._signals_emitted += 1

        logger.info(
            f"[{self.symbol}] {side} signal | "
            f"pm={price_move:+.2f}% (tw={time_window_ms}ms) | "
            f"delta={volume_delta:+.1f}% | "
            f"price={self._current_price:.6f}"
        )

    # =========================================================================
    # Stats & Info
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "symbol": self.symbol,
            "is_bootstrapped": self.apex.is_bootstrapped,
            "trade_count": self._trade_count,
            "current_price": self._current_price,
            "signals_emitted": self._signals_emitted,
            "time_windows": list(self._high_trackers.keys()),
        }

    def get_current_metrics(self) -> dict[str, Any]:
        """
        Get current live metrics for TUI display.

        Returns dict with fields matching APEXTUIState.
        Volume delta is now calculated per-direction using the same
        time window as the bootstrap result.
        """
        if self._current_price <= 0:
            return {}

        long_params = self.apex.params.long
        short_params = self.apex.params.short

        # LONG metrics - from the bootstrap time_window
        long_tw = long_params.time_window if long_params else 0
        long_tracker = self._high_trackers.get(long_tw)

        if long_tracker:
            pm_from_high = long_tracker.price_move_pct(self._current_price)
            rolling_high = long_tracker.extreme_value
            long_delta = long_tracker.volume_delta
        else:
            pm_from_high = 0.0
            rolling_high = self._current_price
            long_delta = 0.0

        # SHORT metrics - from the bootstrap time_window
        short_tw = short_params.time_window if short_params else 0
        short_tracker = self._low_trackers.get(short_tw)

        if short_tracker:
            pm_from_low = short_tracker.price_move_pct(self._current_price)
            rolling_low = short_tracker.extreme_value
            short_delta = short_tracker.volume_delta
        else:
            pm_from_low = 0.0
            rolling_low = self._current_price
            short_delta = 0.0

        # Check signal readiness
        long_ready = False
        short_ready = False

        if long_params and long_tracker:
            long_trigger = long_params.price_move + self.config.long_signal_offset
            long_ready = (
                pm_from_high <= long_trigger
                and long_delta <= long_params.delta_threshold
            )

        if short_params and short_tracker:
            short_trigger = short_params.price_move - self.config.short_signal_offset
            short_ready = (
                pm_from_low >= short_trigger
                and short_delta >= short_params.delta_threshold
            )

        return {
            "symbol": self.symbol,
            "current_price": self._current_price,
            "rolling_high": rolling_high if rolling_high > 0 else self._current_price,
            "rolling_low": rolling_low if rolling_low < float("inf") else self._current_price,
            "price_move_from_high": pm_from_high,
            "price_move_from_low": pm_from_low,
            # Volume delta is now per-direction, using same window as rolling extreme
            "long_volume_delta": long_delta,
            "short_volume_delta": short_delta,
            # Signal state
            "long_signal_ready": long_ready,
            "short_signal_ready": short_ready,
            "long_signal_offset": self.config.long_signal_offset,
            "short_signal_offset": self.config.short_signal_offset,
            # Time windows from bootstrap
            "long_time_window": long_tw,
            "short_time_window": short_tw,
        }


# =============================================================================
# Live Detector Manager
# =============================================================================


class APEXLiveDetectorManager:
    """Manager for multiple APEXLiveDetector instances."""

    def __init__(
        self,
        signal_handler: "APEXSignalHandler | None" = None,
        default_config: LiveDetectorConfig | None = None,
    ) -> None:
        """
        Initialize detector manager.

        Args:
            signal_handler: Signal handler for all detectors.
            default_config: Default configuration for new detectors.
        """
        self.signal_handler = signal_handler
        self._config = default_config or LiveDetectorConfig()
        self._detectors: dict[str, APEXLiveDetector] = {}

    def add(
        self,
        symbol: str,
        apex: "APEX",
        config: LiveDetectorConfig | None = None,
    ) -> APEXLiveDetector:
        """Add a detector for a symbol."""
        if symbol in self._detectors:
            raise ValueError(f"Detector for {symbol} already exists")

        detector = APEXLiveDetector(
            symbol=symbol,
            apex=apex,
            signal_handler=self.signal_handler,
            config=config or self._config,
        )
        self._detectors[symbol] = detector
        logger.info(f"Added live detector for {symbol}")
        return detector

    def get(self, symbol: str) -> APEXLiveDetector | None:
        """Get detector for a symbol."""
        return self._detectors.get(symbol)

    def remove(self, symbol: str) -> bool:
        """Remove detector for a symbol."""
        if symbol in self._detectors:
            del self._detectors[symbol]
            logger.info(f"Removed live detector for {symbol}")
            return True
        return False

    def symbols(self) -> list[str]:
        """Get list of symbols with detectors."""
        return list(self._detectors.keys())

    def all(self) -> list[APEXLiveDetector]:
        """Get all detector instances."""
        return list(self._detectors.values())


# =============================================================================
# Exports
# =============================================================================


def is_live_kernels_available() -> bool:
    """Check if Mojo live kernels are available (deprecated, always False now)."""
    return False


# Keep LiveMetrics for backward compatibility but simplified
@dataclass
class LiveMetrics:
    """Real-time computed metrics (simplified)."""

    symbol: str
    timestamp_ms: int = 0
    current_price: float = 0.0
    trade_count: int = 0
    volume_delta: float = 0.0
    best_pm_from_high: float = 0.0
    best_pm_from_low: float = 0.0
