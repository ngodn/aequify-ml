"""
Data models for Binance Futures trading.

All models use dataclasses for immutability and type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT_MARKET = "take_profit_market"
    STOP = "stop"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order status."""

    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class PositionSide(str, Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # One-way mode


@dataclass(frozen=True)
class Fill:
    """
    A single fill (execution) of an order.

    Attributes:
        price: Execution price.
        quantity: Executed quantity.
        fee: Fee amount.
        fee_currency: Fee currency (e.g., "USDT").
        timestamp: Execution timestamp.
    """

    price: float
    quantity: float
    fee: float
    fee_currency: str
    timestamp: datetime

    @classmethod
    def from_ccxt(cls, data: dict[str, Any]) -> Fill:
        """Create Fill from CCXT trade dict."""
        ts = data.get("timestamp", 0)
        return cls(
            price=float(data.get("price", 0)),
            quantity=float(data.get("amount", 0)),
            fee=float(data.get("fee", {}).get("cost", 0) or 0),
            fee_currency=data.get("fee", {}).get("currency", "USDT") or "USDT",
            timestamp=datetime.fromtimestamp(ts / 1000) if ts else datetime.now(),
        )


@dataclass
class Order:
    """
    An order on the exchange.

    Attributes:
        id: Exchange order ID.
        client_order_id: Client-specified order ID.
        symbol: Trading pair (CCXT format, e.g., "BTC/USDT:USDT").
        side: Order side (buy/sell).
        type: Order type (market/limit/stop).
        status: Order status.
        price: Order price (None for market orders).
        amount: Order quantity.
        filled: Filled quantity.
        remaining: Remaining quantity.
        average: Average fill price.
        cost: Total cost (filled * average).
        stop_price: Stop/trigger price for stop orders.
        take_profit: Take profit price.
        stop_loss: Stop loss price.
        reduce_only: Whether order only reduces position.
        timestamp: Order creation timestamp.
        last_update: Last update timestamp.
        fills: List of fills for this order.
        raw: Raw exchange response.
    """

    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    status: OrderStatus
    amount: float
    client_order_id: str | None = None
    price: float | None = None
    filled: float = 0.0
    remaining: float | None = None
    average: float | None = None
    cost: float = 0.0
    stop_price: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    reduce_only: bool = False
    timestamp: datetime | None = None
    last_update: datetime | None = None
    fills: list[Fill] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ccxt(cls, data: dict[str, Any]) -> Order:
        """Create Order from CCXT order dict."""
        ts = data.get("timestamp", 0)
        last_ts = data.get("lastUpdateTimestamp") or data.get("lastTradeTimestamp", 0)

        # Parse order type
        order_type_str = (data.get("type") or "market").lower()
        order_type = OrderType.MARKET
        if "limit" in order_type_str:
            order_type = OrderType.LIMIT
        elif "stop" in order_type_str and "profit" in order_type_str:
            order_type = OrderType.TAKE_PROFIT_MARKET
        elif "stop" in order_type_str:
            order_type = OrderType.STOP_MARKET

        # Parse status
        status_str = (data.get("status") or "open").lower()
        status = OrderStatus.OPEN
        if status_str == "closed":
            status = OrderStatus.CLOSED
        elif status_str == "canceled":
            status = OrderStatus.CANCELED
        elif status_str == "expired":
            status = OrderStatus.EXPIRED
        elif status_str == "rejected":
            status = OrderStatus.REJECTED

        # Parse fills
        fills = [Fill.from_ccxt(f) for f in data.get("trades", [])]

        # Safely extract numeric values (handle None, empty string, etc.)
        def safe_float(val: Any, default: float = 0.0) -> float:
            if val is None or val == "":
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def safe_float_or_none(val: Any) -> float | None:
            if val is None or val == "":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        return cls(
            id=str(data.get("id", "")),
            client_order_id=data.get("clientOrderId"),
            symbol=data.get("symbol", ""),
            side=OrderSide((data.get("side") or "buy").lower()),
            type=order_type,
            status=status,
            price=safe_float_or_none(data.get("price")),
            amount=safe_float(data.get("amount"), 0.0),
            filled=safe_float(data.get("filled"), 0.0),
            remaining=safe_float_or_none(data.get("remaining")),
            average=safe_float_or_none(data.get("average")),
            cost=safe_float(data.get("cost"), 0.0),
            stop_price=safe_float_or_none(data.get("stopPrice")),
            reduce_only=data.get("reduceOnly", False),
            timestamp=datetime.fromtimestamp(ts / 1000) if ts else None,
            last_update=datetime.fromtimestamp(last_ts / 1000) if last_ts else None,
            fills=fills,
            raw=data,
        )

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.CLOSED and self.filled >= self.amount

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status == OrderStatus.OPEN


@dataclass
class Position:
    """
    An open position on the exchange.

    Attributes:
        symbol: Trading pair (CCXT format).
        side: Position side (long/short).
        size: Position size (absolute value).
        entry_price: Average entry price.
        mark_price: Current mark price.
        liquidation_price: Liquidation price.
        unrealized_pnl: Unrealized profit/loss.
        leverage: Position leverage.
        margin_type: Margin type (isolated/cross).
        notional: Position notional value.
        percentage: Unrealized PnL percentage.
        timestamp: Last update timestamp.
        raw: Raw exchange response.
    """

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float = 0.0
    liquidation_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    margin_type: str = "isolated"
    notional: float = 0.0
    percentage: float = 0.0
    timestamp: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ccxt(cls, data: dict[str, Any]) -> Position | None:
        """
        Create Position from CCXT position dict.

        Returns None if position is empty (size = 0).
        """
        # contracts field holds position size
        contracts = float(data.get("contracts", 0) or 0)

        if contracts == 0:
            return None

        # In Hedge Mode, determine side from positionSide field (not contracts sign)
        # positionSide is in the raw 'info' dict from Binance
        info = data.get("info", {})
        position_side = info.get("positionSide", "").upper()

        if position_side == "LONG":
            side = PositionSide.LONG
        elif position_side == "SHORT":
            side = PositionSide.SHORT
        else:
            # Fallback: use CCXT's 'side' field or contracts sign
            ccxt_side = data.get("side", "").lower()
            if ccxt_side == "long":
                side = PositionSide.LONG
            elif ccxt_side == "short":
                side = PositionSide.SHORT
            else:
                # Last resort: use contracts sign (One-Way Mode)
                side = PositionSide.LONG if contracts > 0 else PositionSide.SHORT

        ts = data.get("timestamp", 0)

        # Safely parse leverage (handle non-numeric values)
        try:
            leverage_val = int(float(data.get("leverage", 1) or 1))
        except (ValueError, TypeError):
            leverage_val = 1

        return cls(
            symbol=data.get("symbol", ""),
            side=side,
            size=abs(contracts),
            entry_price=float(data.get("entryPrice", 0) or 0),
            mark_price=float(data.get("markPrice", 0) or 0),
            liquidation_price=float(data.get("liquidationPrice", 0) or 0),
            unrealized_pnl=float(data.get("unrealizedPnl", 0) or 0),
            leverage=leverage_val,
            margin_type=data.get("marginMode", "isolated") or "isolated",
            notional=float(data.get("notional", 0) or 0),
            percentage=float(data.get("percentage", 0) or 0),
            timestamp=datetime.fromtimestamp(ts / 1000) if ts else None,
            raw=data,
        )

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT

    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL percentage based on entry and mark price."""
        if self.entry_price <= 0:
            return 0.0

        if self.is_long:
            return ((self.mark_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.mark_price) / self.entry_price) * 100
