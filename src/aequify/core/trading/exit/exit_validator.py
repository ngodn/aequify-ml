"""
Exit Validator - Validates trade exit conditions.

Checks timeout thresholds, TP/SL levels, and smart close conditions
before allowing a position exit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.exchange.binance_futures import BinanceFuturesClient, PositionSide

logger = get_logger(__name__)


@dataclass
class ExitValidationResult:
    """
    Result of exit validation.

    Attributes:
        should_exit: Whether position should be exited.
        reason: Exit reason if should_exit is True.
        exit_type: Type of exit (timeout, tp, sl, smart_close, manual).
        details: Additional validation details.
    """

    should_exit: bool
    reason: str = ""
    exit_type: str = ""
    details: dict[str, Any] | None = None


@dataclass
class ExitConfig:
    """
    Exit configuration per direction.

    Attributes:
        take_profit_pct: Take profit percentage from entry.
        stop_loss_pct: Stop loss percentage from entry.
        timeout_seconds: Position timeout in seconds (0 = disabled).
        smart_close_fee_threshold: Min profit % to smart close (covers fees).
        trailing_stop_enabled: Whether trailing stop is active.
        trailing_stop_activation_pct: Profit % to activate trailing.
        trailing_stop_callback_pct: Callback % from peak.
    """

    take_profit_pct: float = 1.0
    stop_loss_pct: float = 1.0
    timeout_seconds: float = 0.0
    smart_close_fee_threshold: float = 0.1
    trailing_stop_enabled: bool = False
    trailing_stop_activation_pct: float = 0.5
    trailing_stop_callback_pct: float = 0.3

    @classmethod
    def from_config(cls, config: dict[str, Any], side: str = "long") -> ExitConfig:
        """Create ExitConfig from config dict."""
        trading = config.get("trading", {})
        futures = trading.get("futures", {})
        side_config = futures.get(side, {})

        return cls(
            take_profit_pct=side_config.get("take_profit_pct", 1.0),
            stop_loss_pct=side_config.get("stop_loss_pct", 1.0),
            timeout_seconds=side_config.get("timeout_seconds", 0.0),
            smart_close_fee_threshold=side_config.get("smart_close_fee_threshold", 0.1),
            trailing_stop_enabled=side_config.get("trailing_stop_enabled", False),
            trailing_stop_activation_pct=side_config.get("trailing_stop_activation_pct", 0.5),
            trailing_stop_callback_pct=side_config.get("trailing_stop_callback_pct", 0.3),
        )


class ExitValidator:
    """
    Validates trade exit conditions.

    Checks:
    - Timeout (position held too long)
    - Take profit level reached
    - Stop loss level reached
    - Smart close (small profit covers fees, no more DCA available)
    - Trailing stop conditions

    Usage:
        validator = ExitValidator(client, long_config, short_config)

        result = validator.validate(
            symbol="BTC/USDT:USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            current_price=50500.0,
            entry_time=time.time() - 3600,
            current_position_size=50.0,
            max_position_size=100.0,
        )

        if result.should_exit:
            # Execute exit
            ...
    """

    def __init__(
        self,
        client: "BinanceFuturesClient",
        long_config: ExitConfig,
        short_config: ExitConfig,
    ) -> None:
        """
        Initialize exit validator.

        Args:
            client: Exchange client for market info.
            long_config: Configuration for LONG exits.
            short_config: Configuration for SHORT exits.
        """
        self.client = client
        self.long_config = long_config
        self.short_config = short_config

        # Trailing stop state per symbol
        self._trailing_peaks: dict[str, float] = {}

    def _get_config(self, side: "PositionSide") -> ExitConfig:
        """Get config for position side."""
        from aequify.core.exchange.binance_futures import PositionSide

        return self.long_config if side == PositionSide.LONG else self.short_config

    def validate(
        self,
        symbol: str,
        side: "PositionSide",
        entry_price: float,
        current_price: float,
        entry_time: float,
        current_position_size: float = 0.0,
        max_position_size: float = 100.0,
    ) -> ExitValidationResult:
        """
        Validate exit conditions.

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.
            entry_price: Average entry price.
            current_price: Current market price.
            entry_time: Position entry timestamp.
            current_position_size: Current position size in USDT.
            max_position_size: Maximum position size in USDT.

        Returns:
            ExitValidationResult indicating if exit is warranted.
        """
        config = self._get_config(side)

        # Calculate PnL percentage
        pnl_pct = self._calculate_pnl_pct(side, entry_price, current_price)

        # Check stop loss first (highest priority)
        sl_result = self._check_stop_loss(symbol, side, pnl_pct, config)
        if sl_result.should_exit:
            return sl_result

        # Check take profit
        tp_result = self._check_take_profit(symbol, side, pnl_pct, config)
        if tp_result.should_exit:
            return tp_result

        # Check trailing stop
        if config.trailing_stop_enabled:
            ts_result = self._check_trailing_stop(symbol, side, pnl_pct, config)
            if ts_result.should_exit:
                return ts_result

        # Check timeout
        timeout_result = self._check_timeout(symbol, entry_time, config)
        if timeout_result.should_exit:
            return timeout_result

        # Check smart close
        smart_result = self._check_smart_close(
            symbol, side, pnl_pct, current_position_size, max_position_size, config
        )
        if smart_result.should_exit:
            return smart_result

        return ExitValidationResult(
            should_exit=False,
            details={
                "pnl_pct": pnl_pct,
                "tp_pct": config.take_profit_pct,
                "sl_pct": config.stop_loss_pct,
            },
        )

    def _calculate_pnl_pct(
        self,
        side: "PositionSide",
        entry_price: float,
        current_price: float,
    ) -> float:
        """Calculate PnL percentage based on position side."""
        from aequify.core.exchange.binance_futures import PositionSide

        if entry_price <= 0:
            return 0.0

        price_change_pct = ((current_price - entry_price) / entry_price) * 100

        if side == PositionSide.LONG:
            return price_change_pct
        else:  # SHORT
            return -price_change_pct

    def _check_stop_loss(
        self,
        symbol: str,
        side: "PositionSide",
        pnl_pct: float,
        config: ExitConfig,
    ) -> ExitValidationResult:
        """Check if stop loss is triggered."""
        if config.stop_loss_pct <= 0:
            return ExitValidationResult(should_exit=False)

        if pnl_pct <= -config.stop_loss_pct:
            return ExitValidationResult(
                should_exit=True,
                reason=f"Stop loss triggered: {pnl_pct:+.2f}% <= -{config.stop_loss_pct:.1f}%",
                exit_type="sl",
                details={
                    "symbol": symbol,
                    "side": side.value,
                    "pnl_pct": pnl_pct,
                    "threshold": -config.stop_loss_pct,
                },
            )

        return ExitValidationResult(should_exit=False)

    def _check_take_profit(
        self,
        symbol: str,
        side: "PositionSide",
        pnl_pct: float,
        config: ExitConfig,
    ) -> ExitValidationResult:
        """Check if take profit is triggered."""
        if config.take_profit_pct <= 0:
            return ExitValidationResult(should_exit=False)

        if pnl_pct >= config.take_profit_pct:
            return ExitValidationResult(
                should_exit=True,
                reason=f"Take profit triggered: {pnl_pct:+.2f}% >= {config.take_profit_pct:.1f}%",
                exit_type="tp",
                details={
                    "symbol": symbol,
                    "side": side.value,
                    "pnl_pct": pnl_pct,
                    "threshold": config.take_profit_pct,
                },
            )

        return ExitValidationResult(should_exit=False)

    def _check_timeout(
        self,
        symbol: str,
        entry_time: float,
        config: ExitConfig,
    ) -> ExitValidationResult:
        """Check if position has timed out."""
        if config.timeout_seconds <= 0:
            return ExitValidationResult(should_exit=False)

        elapsed = time.time() - entry_time

        if elapsed >= config.timeout_seconds:
            return ExitValidationResult(
                should_exit=True,
                reason=f"Position timeout: {elapsed:.0f}s >= {config.timeout_seconds:.0f}s",
                exit_type="timeout",
                details={
                    "symbol": symbol,
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": config.timeout_seconds,
                },
            )

        return ExitValidationResult(should_exit=False)

    def _check_smart_close(
        self,
        symbol: str,
        side: "PositionSide",
        pnl_pct: float,
        current_position_size: float,
        max_position_size: float,
        config: ExitConfig,
    ) -> ExitValidationResult:
        """
        Check smart close conditions.

        Smart close exits when:
        - Position is in small profit (covers fees)
        - No more DCA available (at max position size)

        This prevents holding a full-size position waiting for TP
        when the market might reverse.
        """
        # Only smart close if in profit (above fee threshold)
        if pnl_pct < config.smart_close_fee_threshold:
            return ExitValidationResult(should_exit=False)

        # Only smart close if at max position (no more DCA available)
        if current_position_size < max_position_size * 0.95:  # 95% threshold
            return ExitValidationResult(should_exit=False)

        return ExitValidationResult(
            should_exit=True,
            reason=(
                f"Smart close: profit {pnl_pct:+.2f}% >= {config.smart_close_fee_threshold:.1f}% "
                f"and position at max ({current_position_size:.2f}/{max_position_size:.2f} USDT)"
            ),
            exit_type="smart_close",
            details={
                "symbol": symbol,
                "side": side.value,
                "pnl_pct": pnl_pct,
                "fee_threshold": config.smart_close_fee_threshold,
                "current_size": current_position_size,
                "max_size": max_position_size,
            },
        )

    def _check_trailing_stop(
        self,
        symbol: str,
        side: "PositionSide",
        pnl_pct: float,
        config: ExitConfig,
    ) -> ExitValidationResult:
        """
        Check trailing stop conditions.

        Trailing stop activates when profit exceeds activation threshold,
        then triggers if price retraces by callback percentage from peak.
        """
        key = f"{symbol}_{side.value}"

        # Check if trailing stop should activate
        if pnl_pct < config.trailing_stop_activation_pct:
            # Not activated yet - clear any peak
            self._trailing_peaks.pop(key, None)
            return ExitValidationResult(should_exit=False)

        # Update peak
        current_peak = self._trailing_peaks.get(key, 0.0)
        if pnl_pct > current_peak:
            self._trailing_peaks[key] = pnl_pct
            current_peak = pnl_pct

        # Check callback from peak
        callback = current_peak - pnl_pct
        if callback >= config.trailing_stop_callback_pct:
            # Clear peak on trigger
            self._trailing_peaks.pop(key, None)
            return ExitValidationResult(
                should_exit=True,
                reason=(
                    f"Trailing stop triggered: peak {current_peak:+.2f}% -> current {pnl_pct:+.2f}% "
                    f"(callback {callback:.2f}% >= {config.trailing_stop_callback_pct:.1f}%)"
                ),
                exit_type="trailing_stop",
                details={
                    "symbol": symbol,
                    "side": side.value,
                    "pnl_pct": pnl_pct,
                    "peak_pct": current_peak,
                    "callback_pct": callback,
                    "callback_threshold": config.trailing_stop_callback_pct,
                },
            )

        return ExitValidationResult(should_exit=False)

    def reset_trailing_stop(self, symbol: str, side: "PositionSide") -> None:
        """Reset trailing stop state for a position."""
        key = f"{symbol}_{side.value}"
        self._trailing_peaks.pop(key, None)

    def get_exit_prices(
        self,
        side: "PositionSide",
        entry_price: float,
    ) -> tuple[float, float]:
        """
        Calculate TP and SL prices from entry.

        Args:
            side: LONG or SHORT.
            entry_price: Average entry price.

        Returns:
            Tuple of (take_profit_price, stop_loss_price).
        """
        from aequify.core.exchange.binance_futures import PositionSide

        config = self._get_config(side)

        if side == PositionSide.LONG:
            tp_price = entry_price * (1 + config.take_profit_pct / 100)
            sl_price = entry_price * (1 - config.stop_loss_pct / 100)
        else:  # SHORT
            tp_price = entry_price * (1 - config.take_profit_pct / 100)
            sl_price = entry_price * (1 + config.stop_loss_pct / 100)

        return tp_price, sl_price
