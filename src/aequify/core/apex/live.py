"""
APEX Live Detector - Real-time signal detection from trade stream.

Simple incremental detection like the old project:
1. Track rolling high/low per time window on each trade
2. Calculate volume imbalance at current price
3. Compare against bootstrap thresholds
4. Emit signal when conditions met

Architecture:
    Trade Stream (via SymbolManager)
        └─ APEXLiveDetector.on_trade()
            ├─ Update rolling high/low (per time window)
            ├─ Calculate volume delta at current price
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

    # Price tolerance for volume imbalance (order flow style)
    # Only sum buy/sell at trades within X% of current price
    imbalance_price_tolerance_pct: float = 1.0

    # Maximum trades to keep in memory for imbalance calculation
    max_trades_for_imbalance: int = 100000

    # Signal offset (triggers slightly before/after threshold)
    long_signal_offset: float = 0.0  # Added to price_move threshold
    short_signal_offset: float = 0.0  # Subtracted from price_move threshold

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LiveDetectorConfig:
        """Create from config dict."""
        apex_cfg = config.get("engines", {}).get("apex", {})

        return cls(
            imbalance_price_tolerance_pct=apex_cfg.get("imbalance_price_tolerance_pct", 1.0),
            max_trades_for_imbalance=apex_cfg.get("max_trades_for_imbalance", 100000),
            long_signal_offset=apex_cfg.get("bootstrap", {}).get("long", {}).get("signal_offset", 0.0),
            short_signal_offset=apex_cfg.get("bootstrap", {}).get("short", {}).get("signal_offset", 0.0),
        )


# =============================================================================
# Trade Record for Imbalance Calculation
# =============================================================================


@dataclass
class TradeRecord:
    """Minimal trade record for imbalance calculation."""

    timestamp_ms: int
    price: float
    buy_qty: float  # Quantity if taker bought
    sell_qty: float  # Quantity if taker sold


# =============================================================================
# Rolling High/Low Tracker
# =============================================================================


class RollingExtreme:
    """Tracks rolling high or low within a time window."""

    def __init__(self, window_ms: int, is_high: bool = True):
        """
        Args:
            window_ms: Time window in milliseconds.
            is_high: True for rolling high, False for rolling low.
        """
        self.window_ms = window_ms
        self.is_high = is_high
        self._data: deque[tuple[int, float]] = deque()  # (timestamp_ms, price)
        self._value: float = 0.0 if is_high else float("inf")

    def update(self, timestamp_ms: int, price: float) -> None:
        """Update with new trade."""
        # Remove stale entries
        cutoff = timestamp_ms - self.window_ms
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

        # Add new entry
        self._data.append((timestamp_ms, price))

        # Recompute extreme (could optimize with monotonic deque but simple is fine)
        if self.is_high:
            self._value = max(p for _, p in self._data) if self._data else 0.0
        else:
            self._value = min(p for _, p in self._data) if self._data else float("inf")

    @property
    def value(self) -> float:
        """Current rolling high/low value."""
        return self._value

    def price_move_pct(self, current_price: float) -> float:
        """Calculate price move percentage from extreme.

        For high: (current - high) / high * 100 (negative when below high)
        For low: (current - low) / low * 100 (positive when above low)
        """
        if self._value <= 0 or self._value == float("inf"):
            return 0.0
        return (current_price - self._value) / self._value * 100.0


# =============================================================================
# APEX Live Detector
# =============================================================================


class APEXLiveDetector:
    """
    Real-time signal detector - simple incremental approach.

    On each trade:
    1. Update rolling high/low per time window
    2. Calculate volume imbalance at current price
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

        # Rolling high trackers (one per time window, for LONG)
        # Price drop from high triggers LONG
        self._rolling_highs: dict[int, RollingExtreme] = {}

        # Rolling low trackers (one per time window, for SHORT)
        # Price rise from low triggers SHORT
        self._rolling_lows: dict[int, RollingExtreme] = {}

        # Initialize trackers for each time window from APEX config
        for tw_ms in apex.config.time_windows_ms:
            self._rolling_highs[tw_ms] = RollingExtreme(tw_ms, is_high=True)
            self._rolling_lows[tw_ms] = RollingExtreme(tw_ms, is_high=False)

        # Trade buffer for imbalance calculation
        self._trades: deque[TradeRecord] = deque(maxlen=self.config.max_trades_for_imbalance)

        # Current state
        self._current_price: float = 0.0
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

        Called on each trade from the stream.

        Args:
            price: Trade price.
            quantity: Trade quantity.
            timestamp_ms: Trade timestamp in milliseconds.
            is_buyer_maker: True if buyer was maker (taker sold).
        """
        self._current_price = price
        self._trade_count += 1

        # Update rolling highs and lows
        for rolling_high in self._rolling_highs.values():
            rolling_high.update(timestamp_ms, price)
        for rolling_low in self._rolling_lows.values():
            rolling_low.update(timestamp_ms, price)

        # Add to trade buffer for imbalance calculation
        # is_buyer_maker=True means taker sold
        buy_qty = quantity if not is_buyer_maker else 0.0
        sell_qty = quantity if is_buyer_maker else 0.0

        self._trades.append(TradeRecord(
            timestamp_ms=timestamp_ms,
            price=price,
            buy_qty=buy_qty,
            sell_qty=sell_qty,
        ))

        # Check signals if bootstrapped
        if self.apex.is_bootstrapped:
            self._check_signals()

    def _calculate_imbalance_at_price(self, target_price: float) -> float:
        """
        Calculate volume imbalance at trades near target price.

        Footprint/order flow style:
        - Sum buy/sell volumes at trades within price tolerance
        - Returns (buy - sell) / total * 100

        Args:
            target_price: Price to calculate imbalance around.

        Returns:
            Volume delta percentage (-100 to +100).
            Negative = selling pressure, Positive = buying pressure.
        """
        if target_price <= 0 or not self._trades:
            return 0.0

        tolerance = target_price * (self.config.imbalance_price_tolerance_pct / 100.0)
        price_low = target_price - tolerance
        price_high = target_price + tolerance

        total_buy = 0.0
        total_sell = 0.0

        for trade in self._trades:
            if price_low <= trade.price <= price_high:
                total_buy += trade.buy_qty
                total_sell += trade.sell_qty

        total_volume = total_buy + total_sell
        if total_volume <= 0:
            return 0.0

        return ((total_buy - total_sell) / total_volume) * 100.0

    def _check_signals(self) -> None:
        """Check if LONG or SHORT signal conditions are met."""
        long_params = self.apex.params.long
        short_params = self.apex.params.short

        # Get volume delta at current price
        volume_delta = self._calculate_imbalance_at_price(self._current_price)

        # Check LONG signal
        if long_params:
            # Find best price move from high (across all time windows)
            # LONG wants price to DROP, so we look for most negative price_move
            best_pm_from_high = 0.0
            best_tw_high = 0

            for tw_ms, rolling_high in self._rolling_highs.items():
                pm = rolling_high.price_move_pct(self._current_price)
                if pm < best_pm_from_high:
                    best_pm_from_high = pm
                    best_tw_high = tw_ms

            # Check conditions
            # price_move threshold is negative (e.g., -2.0 means drop 2%)
            # volume_delta threshold is negative (e.g., -70 means selling pressure)
            long_trigger = long_params.price_move + self.config.long_signal_offset
            price_ok = best_pm_from_high <= long_trigger
            imbalance_ok = volume_delta <= long_params.delta_threshold

            if price_ok and imbalance_ok:
                self._emit_signal(
                    side="LONG",
                    price_move=best_pm_from_high,
                    volume_delta=volume_delta,
                    time_window_ms=best_tw_high,
                    params=long_params,
                )

        # Check SHORT signal
        if short_params:
            # Find best price move from low (across all time windows)
            # SHORT wants price to RISE, so we look for most positive price_move
            best_pm_from_low = 0.0
            best_tw_low = 0

            for tw_ms, rolling_low in self._rolling_lows.items():
                pm = rolling_low.price_move_pct(self._current_price)
                if pm > best_pm_from_low:
                    best_pm_from_low = pm
                    best_tw_low = tw_ms

            # Check conditions
            # price_move threshold is positive (e.g., +2.0 means rise 2%)
            # volume_delta threshold is positive (e.g., +70 means buying pressure)
            short_trigger = short_params.price_move - self.config.short_signal_offset
            price_ok = best_pm_from_low >= short_trigger
            imbalance_ok = volume_delta >= short_params.delta_threshold

            if price_ok and imbalance_ok:
                self._emit_signal(
                    side="SHORT",
                    price_move=best_pm_from_low,
                    volume_delta=volume_delta,
                    time_window_ms=best_tw_low,
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
            "trades_buffered": len(self._trades),
            "time_windows": list(self._rolling_highs.keys()),
        }

    def get_current_metrics(self) -> dict[str, Any]:
        """
        Get current live metrics for TUI display.

        Returns dict with fields matching APEXTUIState:
            current_price, rolling_high, rolling_low,
            price_move_from_high, price_move_from_low,
            long_volume_delta, short_volume_delta,
            long_signal_ready, short_signal_ready,
            long_signal_offset, short_signal_offset,
            price_window, imbalance_price_pct
        """
        if self._current_price <= 0:
            return {}

        volume_delta = self._calculate_imbalance_at_price(self._current_price)

        # Get best price moves (best means closest to triggering)
        # LONG: most negative from high, SHORT: most positive from low
        best_pm_from_high = 0.0
        best_rolling_high = 0.0
        best_tw_high = 0

        for tw_ms, rh in self._rolling_highs.items():
            pm = rh.price_move_pct(self._current_price)
            if pm < best_pm_from_high:
                best_pm_from_high = pm
                best_rolling_high = rh.value
                best_tw_high = tw_ms

        best_pm_from_low = 0.0
        best_rolling_low = float("inf")
        best_tw_low = 0

        for tw_ms, rl in self._rolling_lows.items():
            pm = rl.price_move_pct(self._current_price)
            if pm > best_pm_from_low:
                best_pm_from_low = pm
                best_rolling_low = rl.value
                best_tw_low = tw_ms

        # Get params for signal readiness check
        long_params = self.apex.params.long
        short_params = self.apex.params.short

        # Check signal readiness
        long_ready = False
        short_ready = False

        if long_params:
            long_trigger = long_params.price_move + self.config.long_signal_offset
            long_ready = (
                best_pm_from_high <= long_trigger
                and volume_delta <= long_params.delta_threshold
            )

        if short_params:
            short_trigger = short_params.price_move - self.config.short_signal_offset
            short_ready = (
                best_pm_from_low >= short_trigger
                and volume_delta >= short_params.delta_threshold
            )

        return {
            "symbol": self.symbol,
            "current_price": self._current_price,
            "rolling_high": best_rolling_high if best_rolling_high > 0 else self._current_price,
            "rolling_low": best_rolling_low if best_rolling_low < float("inf") else self._current_price,
            "price_move_from_high": best_pm_from_high,
            "price_move_from_low": best_pm_from_low,
            "price_window": best_tw_high or best_tw_low,
            # Volume delta - same value for both since it's at current price
            "long_volume_delta": volume_delta,
            "short_volume_delta": volume_delta,
            "imbalance_price_pct": self.config.imbalance_price_tolerance_pct,
            # Signal state
            "long_signal_ready": long_ready,
            "short_signal_ready": short_ready,
            "long_signal_offset": self.config.long_signal_offset,
            "short_signal_offset": self.config.short_signal_offset,
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
