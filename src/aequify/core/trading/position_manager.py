"""
Position Manager - Tracks position lifecycle from entry to exit.

Thread-safe position tracking with async locks. Integrates with runtime.py
for non-blocking operation.

Architecture:
    PositionManager
        └─ TrackedPosition (per trade)
            ├─ Entry details (order, price, quantity)
            ├─ Exit details (order, price, reason)
            ├─ TP/SL orders
            └─ PnL tracking
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.exchange.binance_futures import Order, PositionSide

logger = get_logger(__name__)


class TradeStatus(str, Enum):
    """Trade lifecycle status."""

    PENDING = "pending"  # Signal received, not yet executed
    OPEN = "open"  # Entry filled, position open
    CLOSING = "closing"  # TP/SL triggered, closing
    CLOSED = "closed"  # Position closed
    CANCELED = "canceled"  # Trade canceled before entry
    FAILED = "failed"  # Execution failed


@dataclass
class TrackedPosition:
    """
    A tracked position with full lifecycle state.

    Includes entry/exit orders, PnL tracking, and metadata.

    Attributes:
        id: Unique trade ID.
        symbol: Trading pair (e.g., "BTC/USDT:USDT").
        side: LONG or SHORT.
        status: Current trade status.
    """

    # Identity
    id: str
    symbol: str
    side: "PositionSide"

    # Entry
    entry_signal_time: datetime = field(default_factory=datetime.now)
    entry_order: "Order | None" = None
    entry_price: float = 0.0
    entry_quantity: float = 0.0
    entry_time: datetime | None = None

    # Target levels
    target_profit_price: float | None = None
    stop_loss_price: float | None = None
    max_hold_time_ms: int = 120000  # Max hold time in ms (default 2 minutes)

    # Exit orders
    tp_order: "Order | None" = None
    sl_order: "Order | None" = None

    # Exit
    exit_order: "Order | None" = None
    exit_price: float = 0.0
    exit_quantity: float = 0.0
    exit_time: datetime | None = None
    exit_reason: str | None = None  # "TP", "SL", "MANUAL", "TIMEOUT"

    # Status
    status: TradeStatus = TradeStatus.PENDING

    # PnL
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    fees_paid: float = 0.0

    # Metadata
    signal_params: dict[str, Any] = field(default_factory=dict)
    raw_data: dict[str, Any] = field(default_factory=dict)

    # Hedge fields
    is_hedge: bool = False
    hedge_of_position_id: str | None = None
    hedged_by_position_id: str | None = None

    @property
    def is_open(self) -> bool:
        """Check if position is currently open."""
        return self.status == TradeStatus.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.status in (TradeStatus.CLOSED, TradeStatus.CANCELED, TradeStatus.FAILED)

    @property
    def duration_seconds(self) -> float | None:
        """Get trade duration in seconds."""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return None

    @property
    def hold_time_ms(self) -> int:
        """Get current hold time in milliseconds (for open positions)."""
        if not self.entry_time:
            return 0
        now = datetime.now()
        return int((now - self.entry_time).total_seconds() * 1000)

    @property
    def is_timed_out(self) -> bool:
        """Check if position has exceeded max hold time."""
        if not self.is_open or not self.entry_time:
            return False
        return self.hold_time_ms >= self.max_hold_time_ms

    @property
    def time_remaining_ms(self) -> int:
        """Get remaining time before timeout in milliseconds."""
        if not self.is_open or not self.entry_time:
            return 0
        remaining = self.max_hold_time_ms - self.hold_time_ms
        return max(0, remaining)

    def calculate_pnl(self, exit_price: float) -> tuple[float, float]:
        """
        Calculate PnL for a given exit price.

        Args:
            exit_price: Price at which position would be closed.

        Returns:
            Tuple of (pnl_usdt, pnl_pct).
        """
        from aequify.core.exchange.binance_futures import PositionSide

        if self.entry_price <= 0 or self.entry_quantity <= 0:
            return 0.0, 0.0

        if self.side == PositionSide.LONG:
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100

        pnl_usdt = (pnl_pct / 100) * (self.entry_price * self.entry_quantity)

        return pnl_usdt, pnl_pct

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value if self.side else None,
            "status": self.status.value,
            "entry_price": self.entry_price,
            "entry_quantity": self.entry_quantity,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "realized_pnl": self.realized_pnl,
            "realized_pnl_pct": self.realized_pnl_pct,
            "hold_time_ms": self.hold_time_ms,
            "is_hedge": self.is_hedge,
        }


# Callback type for position updates
PositionCallback = Callable[[TrackedPosition], Coroutine[Any, Any, None]]


class PositionManager:
    """
    Manages all tracked positions with thread-safe operations.

    Uses asyncio locks for safe concurrent access. Provides callbacks
    for position updates to notify other components.

    Usage:
        manager = PositionManager()

        # Create position
        position = await manager.create_position("BTC/USDT:USDT", PositionSide.LONG)

        # Update with entry
        position = await manager.update_entry(position.id, entry_order, tp_order, sl_order)

        # Update with exit
        position = await manager.update_exit(position.id, exit_order, "TP")

        # Query positions
        open_positions = await manager.get_open_positions()
    """

    def __init__(self) -> None:
        """Initialize position manager."""
        self._positions: dict[str, TrackedPosition] = {}  # id -> position
        self._by_symbol: dict[str, str] = {}  # symbol -> position id
        self._lock = asyncio.Lock()
        self._callbacks: list[PositionCallback] = []
        self._trade_counter = 0
        self._counter_lock = threading.Lock()

    def on_update(self, callback: PositionCallback) -> None:
        """
        Register callback for position updates.

        Args:
            callback: Async function called on position changes.
        """
        self._callbacks.append(callback)

    async def _notify(self, position: TrackedPosition) -> None:
        """Notify all callbacks of position update."""
        for callback in self._callbacks:
            try:
                await callback(position)
            except Exception as e:
                logger.error(f"Position callback error: {e}")

    def _generate_id(self) -> str:
        """Generate unique trade ID."""
        with self._counter_lock:
            self._trade_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"T{timestamp}_{self._trade_counter:04d}"

    async def create_position(
        self,
        symbol: str,
        side: "PositionSide",
        signal_params: dict[str, Any] | None = None,
        max_hold_time_ms: int = 120000,
        is_hedge: bool = False,
        hedge_of_position_id: str | None = None,
    ) -> TrackedPosition:
        """
        Create a new tracked position (pending entry).

        For DCA: returns existing position if one exists for the symbol.

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.
            signal_params: Signal parameters that triggered this trade.
            max_hold_time_ms: Maximum hold time in milliseconds.
            is_hedge: Whether this is a hedge position.
            hedge_of_position_id: ID of the position being hedged (if hedge).

        Returns:
            New or existing TrackedPosition.
        """
        async with self._lock:
            # Check if symbol already has open position - return it for DCA
            # But NOT if we're creating a hedge (hedges are separate positions)
            if not is_hedge and symbol in self._by_symbol:
                existing_id = self._by_symbol[symbol]
                existing = self._positions.get(existing_id)
                if existing and existing.is_open:
                    logger.info(f"Returning existing position {existing_id} for {symbol} (DCA)")
                    return existing

            position = TrackedPosition(
                id=self._generate_id(),
                symbol=symbol,
                side=side,
                entry_signal_time=datetime.now(),
                status=TradeStatus.PENDING,
                signal_params=signal_params or {},
                max_hold_time_ms=max_hold_time_ms,
                is_hedge=is_hedge,
                hedge_of_position_id=hedge_of_position_id,
            )

            self._positions[position.id] = position

            # For hedges, use a different key pattern to allow both positions
            if is_hedge:
                self._by_symbol[f"{symbol}_hedge"] = position.id
            else:
                self._by_symbol[symbol] = position.id

            # Link the original position to this hedge
            if is_hedge and hedge_of_position_id:
                original = self._positions.get(hedge_of_position_id)
                if original:
                    original.hedged_by_position_id = position.id
                    logger.info(
                        f"Created hedge position {position.id} for {symbol} {side.value} "
                        f"(hedging {hedge_of_position_id})"
                    )
            else:
                logger.info(f"Created position {position.id} for {symbol} {side.value}")

            await self._notify(position)

            return position

    async def update_entry(
        self,
        position_id: str,
        entry_order: "Order",
        tp_order: "Order | None" = None,
        sl_order: "Order | None" = None,
    ) -> TrackedPosition:
        """
        Update position with entry order details.

        Args:
            position_id: Position ID.
            entry_order: Filled entry order.
            tp_order: Take profit order.
            sl_order: Stop loss order.

        Returns:
            Updated TrackedPosition.

        Raises:
            ValueError: If position not found.
        """
        async with self._lock:
            position = self._positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            position.entry_order = entry_order
            position.entry_price = entry_order.average or 0.0
            position.entry_quantity = entry_order.filled
            position.entry_time = datetime.now()
            position.tp_order = tp_order
            position.sl_order = sl_order
            position.status = TradeStatus.OPEN

            # Calculate TP/SL prices from orders
            if tp_order and tp_order.stop_price:
                position.target_profit_price = tp_order.stop_price
            if sl_order and sl_order.stop_price:
                position.stop_loss_price = sl_order.stop_price

            logger.info(
                f"Position {position_id} OPEN: {position.side.value} "
                f"{position.entry_quantity} @ {position.entry_price}"
            )
            await self._notify(position)

            return position

    async def update_exit(
        self,
        position_id: str,
        exit_order: "Order | None",
        exit_reason: str,
    ) -> TrackedPosition:
        """
        Update position with exit details.

        Args:
            position_id: Position ID.
            exit_order: Filled exit order (None if already closed externally).
            exit_reason: Reason for exit (TP, SL, MANUAL, TIMEOUT, etc).

        Returns:
            Updated TrackedPosition.

        Raises:
            ValueError: If position not found.
        """
        async with self._lock:
            position = self._positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            position.exit_order = exit_order
            if exit_order:
                position.exit_price = exit_order.average or 0.0
                position.exit_quantity = exit_order.filled
            position.exit_time = datetime.now()
            position.exit_reason = exit_reason
            position.status = TradeStatus.CLOSED

            # Calculate realized PnL
            if position.exit_price > 0:
                pnl_usdt, pnl_pct = position.calculate_pnl(position.exit_price)
                position.realized_pnl = pnl_usdt
                position.realized_pnl_pct = pnl_pct

            # Remove from by_symbol mapping
            if position.is_hedge:
                hedge_key = f"{position.symbol}_hedge"
                if hedge_key in self._by_symbol:
                    del self._by_symbol[hedge_key]
            else:
                if position.symbol in self._by_symbol:
                    del self._by_symbol[position.symbol]

            hedge_info = " (hedge)" if position.is_hedge else ""
            logger.info(
                f"Position {position_id} CLOSED{hedge_info} ({exit_reason}): "
                f"PnL {position.realized_pnl_pct:+.2f}% ({position.realized_pnl:+.2f} USDT)"
            )
            await self._notify(position)

            return position

    async def cancel_position(self, position_id: str) -> TrackedPosition:
        """
        Cancel a pending position.

        Args:
            position_id: Position ID.

        Returns:
            Updated TrackedPosition.

        Raises:
            ValueError: If position not found or not in PENDING state.
        """
        async with self._lock:
            position = self._positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            if position.status != TradeStatus.PENDING:
                raise ValueError(f"Cannot cancel position in {position.status} state")

            position.status = TradeStatus.CANCELED

            # Remove from by_symbol mapping
            if position.symbol in self._by_symbol:
                del self._by_symbol[position.symbol]

            logger.info(f"Position {position_id} CANCELED")
            await self._notify(position)

            return position

    async def fail_position(
        self,
        position_id: str,
        reason: str,
    ) -> TrackedPosition:
        """
        Mark position as failed.

        Args:
            position_id: Position ID.
            reason: Failure reason.

        Returns:
            Updated TrackedPosition.

        Raises:
            ValueError: If position not found.
        """
        async with self._lock:
            position = self._positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            position.status = TradeStatus.FAILED
            position.raw_data["failure_reason"] = reason

            # Remove from by_symbol mapping
            if position.symbol in self._by_symbol:
                del self._by_symbol[position.symbol]

            logger.error(f"Position {position_id} FAILED: {reason}")
            await self._notify(position)

            return position

    async def get_position(self, position_id: str) -> TrackedPosition | None:
        """Get position by ID."""
        async with self._lock:
            return self._positions.get(position_id)

    async def get_position_by_symbol(self, symbol: str) -> TrackedPosition | None:
        """Get open position by symbol."""
        async with self._lock:
            position_id = self._by_symbol.get(symbol)
            if position_id:
                return self._positions.get(position_id)
            return None

    async def get_open_positions(self) -> list[TrackedPosition]:
        """Get all open positions."""
        async with self._lock:
            return [p for p in self._positions.values() if p.status == TradeStatus.OPEN]

    async def get_all_positions(self) -> list[TrackedPosition]:
        """Get all positions (including closed)."""
        async with self._lock:
            return list(self._positions.values())

    async def get_hedge_position(self, symbol: str) -> TrackedPosition | None:
        """Get open hedge position by symbol."""
        async with self._lock:
            hedge_key = f"{symbol}_hedge"
            position_id = self._by_symbol.get(hedge_key)
            if position_id:
                return self._positions.get(position_id)
            return None

    async def get_hedged_pair(
        self, symbol: str
    ) -> tuple[TrackedPosition | None, TrackedPosition | None]:
        """
        Get both positions for a hedged symbol.

        Returns:
            Tuple of (original_position, hedge_position).
        """
        async with self._lock:
            original_id = self._by_symbol.get(symbol)
            hedge_id = self._by_symbol.get(f"{symbol}_hedge")

            original = self._positions.get(original_id) if original_id else None
            hedge = self._positions.get(hedge_id) if hedge_id else None

            return original, hedge

    def get_stats(self) -> dict[str, Any]:
        """Get position statistics (synchronous for TUI)."""
        all_positions = list(self._positions.values())
        closed = [p for p in all_positions if p.status == TradeStatus.CLOSED]
        open_positions = [p for p in all_positions if p.status == TradeStatus.OPEN]

        winners = [p for p in closed if p.realized_pnl > 0]
        losers = [p for p in closed if p.realized_pnl < 0]

        total_pnl = sum(p.realized_pnl for p in closed)
        total_fees = sum(p.fees_paid for p in closed)

        return {
            "total_trades": len(all_positions),
            "open_positions": len(open_positions),
            "closed_trades": len(closed),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(closed) * 100 if closed else 0,
            "total_pnl_usdt": total_pnl,
            "total_fees_usdt": total_fees,
            "net_pnl_usdt": total_pnl - total_fees,
        }

    def clear_closed(self, max_age_hours: float = 24.0) -> int:
        """
        Remove old closed positions from memory.

        Args:
            max_age_hours: Maximum age in hours for closed positions.

        Returns:
            Number of positions cleared.
        """
        now = datetime.now()
        cleared = 0

        for position_id, position in list(self._positions.items()):
            if position.is_closed and position.exit_time:
                age_hours = (now - position.exit_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    del self._positions[position_id]
                    cleared += 1

        if cleared > 0:
            logger.info(f"Cleared {cleared} old closed positions")

        return cleared
