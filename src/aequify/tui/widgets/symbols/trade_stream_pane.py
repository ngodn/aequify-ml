"""
Trade stream pane widget.

Displays real-time trade data from the @trade WebSocket stream
for the selected symbol.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from aequify.tui.widgets.theme_colors import ThemeColorsMixin


@dataclass
class TradeData:
    """
    Trade data for display.

    Simplified trade info for the UI.
    """

    trade_id: int
    symbol: str
    price: float
    quantity: float
    timestamp_ms: int
    is_buyer_maker: bool  # True = sell (taker sold), False = buy (taker bought)

    @property
    def side(self) -> str:
        """Get taker side: 'BUY' or 'SELL'."""
        return "SELL" if self.is_buyer_maker else "BUY"

    def side_color(self, success_color: str, error_color: str) -> str:
        """Get color for side based on theme."""
        return error_color if self.is_buyer_maker else success_color

    @property
    def notional(self) -> float:
        """Get notional value."""
        return self.price * self.quantity

    @property
    def timestamp(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp_ms / 1000)

    @classmethod
    def from_trade_model(cls, trade: Any) -> TradeData:
        """Create from Trade model."""
        return cls(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            price=trade.price,
            quantity=trade.quantity,
            timestamp_ms=trade.timestamp_ms,
            is_buyer_maker=trade.is_buyer_maker,
        )


class TradeStreamPane(VerticalScroll, ThemeColorsMixin):
    """
    Pane displaying real-time trade stream data.

    Shows recent trades with price, quantity, side, and timestamp.
    Limited to last N trades to prevent memory issues.
    """

    DEFAULT_CSS = """
    TradeStreamPane {
        padding: 0 1;
        background: transparent;
    }

    TradeStreamPane #no-trades-message {
        color: $text-muted;
        text-style: italic;
        padding: 1;
    }

    TradeStreamPane #stream-status {
        dock: top;
        height: auto;
        padding: 0 0 1 0;
    }

    TradeStreamPane #trades-content {
        height: auto;
    }
    """

    # Maximum trades to display (keep it reasonable for UI performance)
    MAX_DISPLAY_TRADES = 50

    # Stream state
    is_streaming: reactive[bool] = reactive(False)
    total_trades: reactive[int] = reactive(0)  # Total trades received since start

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._trades: deque[TradeData] = deque(maxlen=self.MAX_DISPLAY_TRADES)
        self._selected_symbol: str | None = None
        self._display_dirty: bool = False

    def compose(self) -> ComposeResult:
        yield Static("", id="stream-status")
        yield Static(
            "Waiting for trade data...",
            id="no-trades-message",
        )
        yield Static("", id="trades-content")

    def on_mount(self) -> None:
        self._update_status()

    def watch_is_streaming(self, streaming: bool) -> None:
        """Update status when streaming state changes."""
        self._update_status()

    def watch_total_trades(self, count: int) -> None:
        """Update display when trade count changes."""
        self._update_status()
        self._update_trades_display()

    def _update_status(self) -> None:
        """Update the status line."""
        status = self.query_one("#stream-status", Static)
        status_text = Text()

        muted = self.muted_style

        if self.is_streaming:
            status_text.append("● LIVE", style=self.bold_success_style)
        else:
            status_text.append("○ IDLE", style=self.bold_warning_style)

        # Show displayed/total trades
        displayed = len(self._trades)
        status_text.append(f"  [{displayed}/{self.total_trades}]", style=muted)

        if self._selected_symbol:
            base = (
                self._selected_symbol.split("/")[0]
                if "/" in self._selected_symbol
                else self._selected_symbol
            )
            status_text.append(f"  {base}", style=muted)

        status.update(status_text)

    def _update_trades_display(self) -> None:
        """Update the trades list display."""
        no_trades_msg = self.query_one("#no-trades-message", Static)
        content = self.query_one("#trades-content", Static)

        if not self._trades:
            no_trades_msg.display = True
            content.display = False
            return

        no_trades_msg.display = False
        content.display = True

        content.update(self._build_trades_display())

    def _build_trades_display(self) -> Text:
        """Build rich text display of trades."""
        text = Text()

        # Get theme colors
        success = self.success_color
        error = self.error_color
        muted = self.muted_style

        # Header
        text.append("Time       ", style=f"bold {muted}")
        text.append("Side  ", style=f"bold {muted}")
        text.append("Price            ", style=f"bold {muted}")
        text.append("Qty              ", style=f"bold {muted}")
        text.append("Value\n", style=f"bold {muted}")

        # Show trades (most recent first)
        for trade in reversed(self._trades):
            # Time
            time_str = trade.timestamp.strftime("%H:%M:%S")
            text.append(f"{time_str}   ", style=muted)

            # Side
            text.append(f"{trade.side:<5} ", style=trade.side_color(success, error))

            # Price
            price_str = self._format_price(trade.price)
            text.append(f"{price_str:<16} ", style="")

            # Quantity
            qty_str = self._format_quantity(trade.quantity)
            text.append(f"{qty_str:<16} ", style="")

            # Notional value
            value_str = f"{trade.notional:,.2f}"
            text.append(f"{value_str}\n", style=muted)

        return text

    @staticmethod
    def _format_price(price: float) -> str:
        """Format price without scientific notation."""
        if price >= 1:
            return f"{price:,.8f}".rstrip("0").rstrip(".")
        return f"{price:.10f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_quantity(qty: float) -> str:
        """Format quantity."""
        if qty >= 1000:
            return f"{qty:,.2f}"
        elif qty >= 1:
            return f"{qty:,.4f}"
        return f"{qty:.8f}".rstrip("0").rstrip(".")

    def add_trade(self, trade: Any) -> None:
        """
        Add a trade to the display.

        Args:
            trade: Trade model or dict with trade data.
        """
        # Filter by selected symbol
        trade_symbol = getattr(trade, "symbol", None) or trade.get("symbol", "")
        if self._selected_symbol and trade_symbol != self._selected_symbol:
            return

        # Convert to TradeData
        if hasattr(trade, "trade_id"):
            trade_data = TradeData.from_trade_model(trade)
        else:
            trade_data = TradeData(
                trade_id=trade.get("trade_id", 0),
                symbol=trade.get("symbol", ""),
                price=trade.get("price", 0),
                quantity=trade.get("quantity", 0),
                timestamp_ms=trade.get("timestamp_ms", 0),
                is_buyer_maker=trade.get("is_buyer_maker", False),
            )

        # deque with maxlen automatically pops oldest when full (FIFO behavior)
        self._trades.append(trade_data)
        self.total_trades += 1

    def set_symbol(self, symbol: str | None) -> None:
        """
        Set the symbol to filter trades.

        Args:
            symbol: Symbol to filter by, or None to show all.
        """
        if symbol != self._selected_symbol:
            self._selected_symbol = symbol
            # Clear trades when symbol changes
            self._trades.clear()
            self.total_trades = 0
            self._update_status()
            self._update_trades_display()

    def set_streaming(self, streaming: bool) -> None:
        """Set streaming status."""
        self.is_streaming = streaming

    def clear(self) -> None:
        """Clear all trades and reset state."""
        self._trades.clear()
        self._selected_symbol = None
        self.total_trades = 0
        self.is_streaming = False
        self._update_status()
        self._update_trades_display()
