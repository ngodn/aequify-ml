"""
Risk Manager - Pre-trade risk checks and daily loss tracking.

Thread-safe with asyncio locks. Prevents excessive losses through
configurable limits and checks.

Checks performed:
- Max concurrent positions
- Per-symbol position limits
- Daily loss limit
- Direction enabled (LONG/SHORT)
- Trading window restrictions
- Hedge allowance (relaxed checks)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.exchange.binance_futures import BinanceFuturesClient, PositionSide

logger = get_logger(__name__)


class RiskCheck(str, Enum):
    """Risk check types."""

    MAX_POSITIONS = "max_positions"
    MAX_POSITION_PER_SYMBOL = "max_position_per_symbol"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DISABLED_DIRECTION = "disabled_direction"
    TRADING_WINDOW = "trading_window"
    HEDGE_ALLOWED = "hedge_allowed"


@dataclass
class RiskCheckResult:
    """
    Result of a risk check.

    Attributes:
        passed: Whether the check passed.
        check: The check that was performed.
        reason: Reason for failure (if failed).
        details: Additional details.
    """

    passed: bool
    check: RiskCheck
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """
    Risk management configuration.

    Attributes:
        max_positions: Maximum concurrent positions.
        max_position_per_symbol: Maximum positions per symbol (usually 1).
        max_daily_loss_pct: Maximum daily loss as % of balance.
        long_enabled: Whether LONG trades are enabled.
        short_enabled: Whether SHORT trades are enabled.
        disable_trading_windows: UTC time windows to disable trading (HHMM format).
    """

    max_positions: int = 3
    max_position_per_symbol: int = 1
    max_daily_loss_pct: float = 5.0
    long_enabled: bool = True
    short_enabled: bool = True
    disable_trading_windows: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RiskConfig:
        """Create RiskConfig from config dict."""
        trading = config.get("trading", {})
        risk = trading.get("risk", {})
        futures = trading.get("futures", {})

        # Parse trading windows
        windows = []
        for window in trading.get("disable_trading_on_window_utc", []):
            if isinstance(window, list) and len(window) == 2:
                windows.append((int(window[0]), int(window[1])))

        return cls(
            max_positions=risk.get("max_positions", 3),
            max_position_per_symbol=risk.get("max_position_per_symbol", 1),
            max_daily_loss_pct=risk.get("max_daily_loss_pct", 5.0),
            long_enabled=futures.get("long", {}).get("enabled", True),
            short_enabled=futures.get("short", {}).get("enabled", True),
            disable_trading_windows=windows,
        )


class RiskManager:
    """
    Manages trading risk through pre-trade checks.

    Thread-safe with asyncio locks. Tracks daily PnL and enforces limits.
    Supports DCA with position size tracking.

    Usage:
        manager = RiskManager(config, client)
        await manager.initialize()

        # Check if trade is allowed
        can_trade, results = await manager.can_trade(symbol, side)
        if can_trade:
            # Execute trade
            ...
            await manager.register_position_open(symbol, size_usdt)

        # On close
        await manager.register_position_close(symbol, pnl)
    """

    def __init__(
        self,
        config: RiskConfig,
        client: "BinanceFuturesClient",
    ) -> None:
        """
        Initialize risk manager.

        Args:
            config: Risk configuration.
            client: Exchange client for balance queries.
        """
        self.config = config
        self.client = client
        self._lock = asyncio.Lock()

        # Daily PnL tracking
        self._daily_pnl: float = 0.0
        self._daily_start_balance: float = 0.0
        self._daily_reset_date: datetime = datetime.min

        # Position tracking
        self._open_positions: set[str] = set()  # symbols with open positions
        self._position_sizes: dict[str, float] = {}  # symbol -> current size in USDT

    async def initialize(self) -> None:
        """Initialize risk manager state."""
        await self._reset_daily_if_needed()

    async def _reset_daily_if_needed(self) -> None:
        """Reset daily stats if new day."""
        now = datetime.utcnow()

        if now.date() > self._daily_reset_date.date():
            async with self._lock:
                try:
                    balance = await self.client.fetch_balance()
                    usdt_balance = balance.get("USDT", {})
                    self._daily_start_balance = float(usdt_balance.get("total", 0) or 0)
                    self._daily_pnl = 0.0
                    self._daily_reset_date = now
                    logger.info(
                        f"Daily reset: starting balance {self._daily_start_balance:.2f} USDT"
                    )
                except Exception as e:
                    logger.error(f"Failed to reset daily stats: {e}")

    async def check_all(
        self,
        symbol: str,
        side: "PositionSide",
    ) -> list[RiskCheckResult]:
        """
        Run all risk checks for a potential trade.

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.

        Returns:
            List of RiskCheckResult for each check.
        """
        await self._reset_daily_if_needed()

        results = []

        # Check trading window
        results.append(await self._check_trading_window())

        # Check direction enabled
        results.append(self._check_direction_enabled(side))

        # Check max positions
        results.append(await self._check_max_positions())

        # Check symbol position (always passes for DCA support)
        results.append(await self._check_symbol_position(symbol))

        # Check daily loss
        results.append(await self._check_daily_loss())

        return results

    async def can_trade(
        self,
        symbol: str,
        side: "PositionSide",
        is_hedge: bool = False,
    ) -> tuple[bool, list[RiskCheckResult]]:
        """
        Check if a trade is allowed.

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.
            is_hedge: Whether this is a hedge position (bypasses some checks).

        Returns:
            Tuple of (can_trade, list of check results).
        """
        if is_hedge:
            results = await self._check_hedge_allowed(symbol, side)
        else:
            results = await self.check_all(symbol, side)

        passed = all(r.passed for r in results)
        return passed, results

    async def _check_hedge_allowed(
        self,
        symbol: str,
        side: "PositionSide",
    ) -> list[RiskCheckResult]:
        """
        Check if a hedge trade is allowed.

        Hedge positions have relaxed checks:
        - Max positions check is bypassed (hedge is protective)
        - Per-symbol limit is bypassed (hedge requires 2 positions)
        - Direction must still be enabled
        - Trading window still applies
        - Daily loss limit is checked but with 2x tolerance

        Args:
            symbol: Trading pair.
            side: LONG or SHORT (the hedge side).

        Returns:
            List of RiskCheckResult.
        """
        results = []

        # Trading window still applies
        results.append(await self._check_trading_window())

        # Direction must be enabled
        results.append(self._check_direction_enabled(side))

        # Daily loss limit - more lenient for hedges (allow up to 2x limit)
        daily_check = await self._check_daily_loss()
        if not daily_check.passed:
            daily_loss_pct = daily_check.details.get("daily_loss_pct", 0)
            hedge_limit = self.config.max_daily_loss_pct * 2
            if daily_loss_pct < hedge_limit:
                results.append(
                    RiskCheckResult(
                        passed=True,
                        check=RiskCheck.HEDGE_ALLOWED,
                        reason="Hedge allowed despite daily loss (protective)",
                        details={"daily_loss_pct": daily_loss_pct, "hedge_limit": hedge_limit},
                    )
                )
            else:
                results.append(
                    RiskCheckResult(
                        passed=False,
                        check=RiskCheck.HEDGE_ALLOWED,
                        reason=f"Daily loss too high for hedge ({daily_loss_pct:.1f}% > {hedge_limit:.1f}%)",
                        details={"daily_loss_pct": daily_loss_pct, "hedge_limit": hedge_limit},
                    )
                )
        else:
            results.append(
                RiskCheckResult(
                    passed=True,
                    check=RiskCheck.HEDGE_ALLOWED,
                    details={"reason": "Hedge allowed"},
                )
            )

        logger.debug(
            f"Hedge check for {symbol} {side.value}: "
            f"{'PASSED' if all(r.passed for r in results) else 'FAILED'}"
        )

        return results

    async def _check_trading_window(self) -> RiskCheckResult:
        """Check if current time is within disabled trading window."""
        now = datetime.utcnow()
        current_time = now.hour * 100 + now.minute  # HHMM format

        for start, end in self.config.disable_trading_windows:
            if start <= current_time <= end:
                return RiskCheckResult(
                    passed=False,
                    check=RiskCheck.TRADING_WINDOW,
                    reason=f"Trading disabled {start:04d}-{end:04d} UTC",
                    details={"current_time": current_time},
                )

        return RiskCheckResult(passed=True, check=RiskCheck.TRADING_WINDOW)

    def _check_direction_enabled(self, side: "PositionSide") -> RiskCheckResult:
        """Check if trading direction is enabled."""
        from aequify.core.exchange.binance_futures import PositionSide

        if side == PositionSide.LONG and not self.config.long_enabled:
            return RiskCheckResult(
                passed=False,
                check=RiskCheck.DISABLED_DIRECTION,
                reason="LONG trading is disabled",
            )

        if side == PositionSide.SHORT and not self.config.short_enabled:
            return RiskCheckResult(
                passed=False,
                check=RiskCheck.DISABLED_DIRECTION,
                reason="SHORT trading is disabled",
            )

        return RiskCheckResult(passed=True, check=RiskCheck.DISABLED_DIRECTION)

    async def _check_max_positions(self) -> RiskCheckResult:
        """Check maximum concurrent positions."""
        async with self._lock:
            current = len(self._open_positions)

        if current >= self.config.max_positions:
            return RiskCheckResult(
                passed=False,
                check=RiskCheck.MAX_POSITIONS,
                reason=f"Max positions reached ({current}/{self.config.max_positions})",
                details={"current": current, "max": self.config.max_positions},
            )

        return RiskCheckResult(
            passed=True,
            check=RiskCheck.MAX_POSITIONS,
            details={"current": current, "max": self.config.max_positions},
        )

    async def _check_symbol_position(self, symbol: str) -> RiskCheckResult:
        """Check if symbol position allows new entry (DCA supported)."""
        async with self._lock:
            has_position = symbol in self._open_positions
            current_size = self._position_sizes.get(symbol, 0.0)

        # DCA is allowed - position check always passes
        # Size limit is checked separately via calculate_dca_entry_size
        return RiskCheckResult(
            passed=True,
            check=RiskCheck.MAX_POSITION_PER_SYMBOL,
            details={"symbol": symbol, "has_position": has_position, "current_size": current_size},
        )

    async def _check_daily_loss(self) -> RiskCheckResult:
        """Check daily loss limit."""
        if self._daily_start_balance <= 0:
            return RiskCheckResult(passed=True, check=RiskCheck.DAILY_LOSS_LIMIT)

        async with self._lock:
            daily_loss_pct = (-self._daily_pnl / self._daily_start_balance) * 100

        if daily_loss_pct >= self.config.max_daily_loss_pct:
            return RiskCheckResult(
                passed=False,
                check=RiskCheck.DAILY_LOSS_LIMIT,
                reason=f"Daily loss limit reached ({daily_loss_pct:.1f}%)",
                details={
                    "daily_pnl": self._daily_pnl,
                    "daily_loss_pct": daily_loss_pct,
                    "max_daily_loss_pct": self.config.max_daily_loss_pct,
                },
            )

        return RiskCheckResult(
            passed=True,
            check=RiskCheck.DAILY_LOSS_LIMIT,
            details={"daily_loss_pct": daily_loss_pct},
        )

    async def register_position_open(self, symbol: str, size_usdt: float = 0.0) -> None:
        """
        Register that a position was opened or DCA'd.

        Args:
            symbol: Trading pair.
            size_usdt: Position size added in USDT.
        """
        async with self._lock:
            self._open_positions.add(symbol)
            current = self._position_sizes.get(symbol, 0.0)
            self._position_sizes[symbol] = current + size_usdt
            logger.debug(
                f"Position registered: {symbol}, size: {self._position_sizes[symbol]:.2f} USDT"
            )

    async def register_position_close(
        self,
        symbol: str,
        pnl: float,
    ) -> None:
        """
        Register that a position was closed.

        Args:
            symbol: Trading pair.
            pnl: Realized PnL in USDT.
        """
        async with self._lock:
            self._open_positions.discard(symbol)
            self._position_sizes.pop(symbol, None)
            self._daily_pnl += pnl
            logger.debug(f"Position closed: {symbol}, PnL: {pnl:+.2f} USDT")

    def get_position_size(self, symbol: str) -> float:
        """Get current position size in USDT for a symbol."""
        return self._position_sizes.get(symbol, 0.0)

    def calculate_dca_entry_size(
        self,
        symbol: str,
        initial_size: float,
        max_size: float,
        multiplier: float = 4.25,
    ) -> float | None:
        """
        Calculate the next DCA entry size.

        DCA formula: next_entry = current_total * multiplier
        Stops when total >= max_size.

        Args:
            symbol: Trading pair.
            initial_size: Initial entry size in USDT.
            max_size: Maximum position size in USDT.
            multiplier: DCA multiplier (default 4.25x cumulative).

        Returns:
            Next entry size in USDT, or None if max reached.
        """
        current_size = self._position_sizes.get(symbol, 0.0)

        # No position yet - use initial size
        if current_size == 0:
            return initial_size

        # Check if max already reached
        if current_size >= max_size:
            return None

        # Calculate next entry: current_total * multiplier
        next_entry = current_size * multiplier

        # Cap to not exceed max
        remaining = max_size - current_size
        if next_entry > remaining:
            next_entry = remaining

        # Don't enter if too small (less than 1 USDT)
        if next_entry < 1.0:
            return None

        return next_entry

    async def sync_positions(self) -> None:
        """Sync open positions with exchange, including sizes."""
        try:
            positions = await self.client.get_positions()
            async with self._lock:
                self._open_positions = {p.symbol for p in positions}
                self._position_sizes = {}
                for p in positions:
                    # notional is the position value in USDT
                    self._position_sizes[p.symbol] = abs(p.notional) if p.notional else 0.0
                logger.info(
                    f"Synced {len(self._open_positions)} positions from exchange: "
                    f"{', '.join(f'{s}={self._position_sizes.get(s, 0):.2f}' for s in self._open_positions) or 'none'}"
                )
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get risk manager statistics."""
        return {
            "open_positions": len(self._open_positions),
            "max_positions": self.config.max_positions,
            "daily_pnl": self._daily_pnl,
            "daily_start_balance": self._daily_start_balance,
            "daily_loss_pct": (
                (-self._daily_pnl / self._daily_start_balance) * 100
                if self._daily_start_balance > 0
                else 0
            ),
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
            "long_enabled": self.config.long_enabled,
            "short_enabled": self.config.short_enabled,
            "position_sizes": dict(self._position_sizes),
        }
