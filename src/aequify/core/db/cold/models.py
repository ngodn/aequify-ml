"""
Cold storage data models.

Python dataclasses that mirror the Mojo models and support QuestDB persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Direction Enum
# =============================================================================


class Direction(str, Enum):
    """Trading direction."""

    LONG = "long"
    SHORT = "short"


# =============================================================================
# Trade Model
# =============================================================================


@dataclass(slots=True)
class Trade:
    """
    Single trade from exchange stream.

    Mirrors the Mojo Trade struct for compatibility.
    Memory: ~33 bytes per trade (in Mojo).

    Attributes:
        trade_id: Exchange trade ID.
        symbol: Trading pair (e.g., "BTC/USDT:USDT").
        price: Execution price.
        quantity: Trade quantity.
        timestamp_ms: Unix timestamp in milliseconds.
        is_buyer_maker: True if buyer was maker (taker sold).
    """

    trade_id: int
    symbol: str
    price: float
    quantity: float
    timestamp_ms: int
    is_buyer_maker: bool

    @property
    def notional(self) -> float:
        """Price * Quantity."""
        return self.price * self.quantity

    @property
    def side(self) -> str:
        """Taker's side: 'buy' if taker bought, 'sell' if taker sold."""
        return "sell" if self.is_buyer_maker else "buy"

    @property
    def timestamp_ns(self) -> int:
        """Timestamp in nanoseconds (for QuestDB ILP)."""
        return self.timestamp_ms * 1_000_000

    def to_ilp_dict(self) -> dict[str, Any]:
        """Convert to dict for ILP ingestion."""
        return {
            "trade_id": self.trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "notional": self.notional,
            "is_buyer_maker": self.is_buyer_maker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trade:
        """Create from dictionary."""
        return cls(
            trade_id=int(data["trade_id"]),
            symbol=str(data["symbol"]),
            price=float(data["price"]),
            quantity=float(data["quantity"]),
            timestamp_ms=int(data["timestamp_ms"]),
            is_buyer_maker=bool(data["is_buyer_maker"]),
        )

    @classmethod
    def from_ccxt(cls, symbol: str, trade: dict[str, Any]) -> Trade:
        """
        Create from CCXT trade dict.

        Args:
            symbol: Trading pair.
            trade: CCXT trade dict with keys: id, price, amount, timestamp, side.

        Returns:
            Trade instance.
        """
        return cls(
            trade_id=int(trade["id"]),
            symbol=symbol,
            price=float(trade["price"]),
            quantity=float(trade["amount"]),
            timestamp_ms=int(trade["timestamp"]),
            is_buyer_maker=trade.get("side") == "sell",
        )


# =============================================================================
# Bootstrap Result Model
# =============================================================================


@dataclass(slots=True)
class BootstrapResult:
    """
    Bootstrap optimization result for a symbol and direction.

    Stores the optimized parameters from APEX engine's bootstrap process.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT:USDT").
        direction: Trading direction (long/short).
        price_move: Price move threshold (%).
        min_dca_distance: Minimum DCA distance (%).
        imbalance_threshold: Volume imbalance threshold (%).
        time_window: Lookback time window (ms).
        take_profit: Take profit target (%).
        stop_loss: Stop loss threshold (%).
        max_hold_time: Maximum hold time (ms).
        entries: Number of signal entries in backtest.
        winners: Number of profitable trades.
        win_rate: Win rate (0-1).
        avg_pnl: Average PnL (%).
        loss: Optimization loss function value.
        timestamp_ms: When this result was computed.
    """

    symbol: str
    direction: Direction
    price_move: float
    min_dca_distance: float
    imbalance_threshold: float
    time_window: int
    take_profit: float
    stop_loss: float
    max_hold_time: int
    entries: int = 0
    winners: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    loss: float = 0.0
    timestamp_ms: int = 0

    @property
    def timestamp_ns(self) -> int:
        """Timestamp in nanoseconds (for QuestDB ILP)."""
        return self.timestamp_ms * 1_000_000

    def to_ilp_dict(self) -> dict[str, Any]:
        """Convert to dict for ILP ingestion."""
        return {
            "price_move": self.price_move,
            "min_dca_distance": self.min_dca_distance,
            "imbalance_threshold": self.imbalance_threshold,
            "time_window": self.time_window,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "max_hold_time": self.max_hold_time,
            "entries": self.entries,
            "winners": self.winners,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "loss": self.loss,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BootstrapResult:
        """Create from dictionary."""
        direction = data.get("direction", "long")
        if isinstance(direction, str):
            direction = Direction(direction.lower())

        return cls(
            symbol=str(data["symbol"]),
            direction=direction,
            price_move=float(data["price_move"]),
            min_dca_distance=float(data["min_dca_distance"]),
            imbalance_threshold=float(data["imbalance_threshold"]),
            time_window=int(data["time_window"]),
            take_profit=float(data["take_profit"]),
            stop_loss=float(data["stop_loss"]),
            max_hold_time=int(data["max_hold_time"]),
            entries=int(data.get("entries", 0)),
            winners=int(data.get("winners", 0)),
            win_rate=float(data.get("win_rate", 0.0)),
            avg_pnl=float(data.get("avg_pnl", 0.0)),
            loss=float(data.get("loss", 0.0)),
            timestamp_ms=int(data.get("timestamp_ms", 0)),
        )


# =============================================================================
# Bootstrap Bounds Model
# =============================================================================


@dataclass(slots=True)
class BootstrapBounds:
    """
    Optimized parameter bounds for bootstrap grid search.

    These bounds are dynamically adjusted based on historical performance.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT:USDT").
        direction: Trading direction (long/short).
        optimized_bound_price_move: Optimized price move range (min, max).
        optimized_bound_min_dca_distance: Optimized DCA distance range.
        optimized_bound_imbalance_threshold: Optimized imbalance range.
        optimized_bound_take_profit: Optimized take profit range.
        optimized_bound_stop_loss: Optimized stop loss range.
        optimized_bound_max_hold_time: Optimized max hold time range.
        timestamp_ms: When these bounds were computed.
    """

    symbol: str
    direction: Direction
    optimized_bound_price_move: tuple[float, float]
    optimized_bound_min_dca_distance: tuple[float, float]
    optimized_bound_imbalance_threshold: tuple[float, float]
    optimized_bound_take_profit: tuple[float, float]
    optimized_bound_stop_loss: tuple[float, float]
    optimized_bound_max_hold_time: tuple[int, int]
    timestamp_ms: int = 0

    @property
    def timestamp_ns(self) -> int:
        """Timestamp in nanoseconds (for QuestDB ILP)."""
        return self.timestamp_ms * 1_000_000

    def to_ilp_dict(self) -> dict[str, Any]:
        """Convert to dict for ILP ingestion."""
        return {
            "price_move_min": self.optimized_bound_price_move[0],
            "price_move_max": self.optimized_bound_price_move[1],
            "min_dca_distance_min": self.optimized_bound_min_dca_distance[0],
            "min_dca_distance_max": self.optimized_bound_min_dca_distance[1],
            "imbalance_threshold_min": self.optimized_bound_imbalance_threshold[0],
            "imbalance_threshold_max": self.optimized_bound_imbalance_threshold[1],
            "take_profit_min": self.optimized_bound_take_profit[0],
            "take_profit_max": self.optimized_bound_take_profit[1],
            "stop_loss_min": self.optimized_bound_stop_loss[0],
            "stop_loss_max": self.optimized_bound_stop_loss[1],
            "max_hold_time_min": self.optimized_bound_max_hold_time[0],
            "max_hold_time_max": self.optimized_bound_max_hold_time[1],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BootstrapBounds:
        """Create from dictionary."""
        direction = data.get("direction", "long")
        if isinstance(direction, str):
            direction = Direction(direction.lower())

        return cls(
            symbol=str(data["symbol"]),
            direction=direction,
            optimized_bound_price_move=(
                float(data["price_move_min"]),
                float(data["price_move_max"]),
            ),
            optimized_bound_min_dca_distance=(
                float(data["min_dca_distance_min"]),
                float(data["min_dca_distance_max"]),
            ),
            optimized_bound_imbalance_threshold=(
                float(data["imbalance_threshold_min"]),
                float(data["imbalance_threshold_max"]),
            ),
            optimized_bound_take_profit=(
                float(data["take_profit_min"]),
                float(data["take_profit_max"]),
            ),
            optimized_bound_stop_loss=(
                float(data["stop_loss_min"]),
                float(data["stop_loss_max"]),
            ),
            optimized_bound_max_hold_time=(
                int(data["max_hold_time_min"]),
                int(data["max_hold_time_max"]),
            ),
            timestamp_ms=int(data.get("timestamp_ms", 0)),
        )


# =============================================================================
# Imbalance Level Model
# =============================================================================


@dataclass(slots=True)
class ImbalanceLevel:
    """
    Single imbalance level from session analysis.

    Attributes:
        price: Price level.
        volume: Total volume at this level.
        delta_pct: Volume delta percentage.
        imbalance_pct: Buy/sell imbalance percentage.
        filled: Whether price has returned to this level.
        fill_time_ms: When the level was filled (0 if not filled).
        used: Whether this level was used for TP/DCA trigger.
    """

    price: float
    volume: float
    delta_pct: float
    imbalance_pct: float
    filled: bool = False
    fill_time_ms: int = 0
    used: bool = False


@dataclass(slots=True)
class SessionLevels:
    """
    Session imbalance levels for a symbol and direction.

    Stores the detected imbalance levels from APEX engine session analysis.

    Attributes:
        symbol: Trading pair (e.g., "BTC/USDT:USDT").
        direction: Trading direction (long/short).
        session_date: Session date (ISO format: "2025-12-28").
        session_name: Forex session name (e.g., "London", "Tokyo").
        imbalance_levels: List of imbalance levels.
        timestamp_ms: When these levels were computed.
    """

    symbol: str
    direction: Direction
    session_date: str
    session_name: str
    imbalance_levels: list[ImbalanceLevel] = field(default_factory=list)
    timestamp_ms: int = 0

    @property
    def timestamp_ns(self) -> int:
        """Timestamp in nanoseconds (for QuestDB ILP)."""
        return self.timestamp_ms * 1_000_000

    def to_level_dicts(self) -> list[dict[str, Any]]:
        """
        Convert to list of dicts for ILP ingestion.

        Each level becomes a separate row in QuestDB.
        """
        result = []
        for i, level in enumerate(self.imbalance_levels):
            result.append(
                {
                    "level_index": i,
                    "price": level.price,
                    "volume": level.volume,
                    "delta_pct": level.delta_pct,
                    "imbalance_pct": level.imbalance_pct,
                    "filled": level.filled,
                    "fill_time_ms": level.fill_time_ms,
                    "used": level.used,
                }
            )
        return result

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> SessionLevels | None:
        """
        Create from list of QuestDB rows.

        Assumes all rows belong to the same symbol/direction/session.
        """
        if not rows:
            return None

        first = rows[0]
        direction = first.get("direction", "long")
        if isinstance(direction, str):
            direction = Direction(direction.lower())

        levels = []
        for row in sorted(rows, key=lambda r: r.get("level_index", 0)):
            levels.append(
                ImbalanceLevel(
                    price=float(row["price"]),
                    volume=float(row["volume"]),
                    delta_pct=float(row["delta_pct"]),
                    imbalance_pct=float(row["imbalance_pct"]),
                    filled=bool(row.get("filled", False)),
                    fill_time_ms=int(row.get("fill_time_ms", 0)),
                    used=bool(row.get("used", False)),
                )
            )

        return cls(
            symbol=str(first["symbol"]),
            direction=direction,
            session_date=str(first["session_date"]),
            session_name=str(first["session_name"]),
            imbalance_levels=levels,
            timestamp_ms=int(first.get("timestamp_ms", 0)),
        )
