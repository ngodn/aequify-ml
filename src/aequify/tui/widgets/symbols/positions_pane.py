"""
Positions pane widget.

Displays active trading positions with PnL, entry price,
mark price, and other position details.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from aequify.tui.widgets.theme_colors import ThemeColorsMixin


@dataclass
class PositionInfo:
    """
    Position information from exchange.

    Contains CCXT unified position fields.
    """

    symbol: str = ""
    side: str = ""  # "long" or "short"
    contracts: float = 0.0
    contract_size: float = 1.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    liquidation_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    notional: float = 0.0
    leverage: int = 0
    margin_mode: str = "cross"  # "cross" or "isolated"
    initial_margin: float = 0.0
    maint_margin: float = 0.0
    margin_ratio: float = 0.0
    timestamp: datetime | None = None

    # Raw info for additional details
    raw_info: dict[str, Any] | None = None

    # Hedge tracking (populated by hedge pair grouping)
    is_hedge_pair: bool = False  # True if this is part of a hedge pair
    hedge_partner: "PositionInfo | None" = None  # The other position in the pair

    @classmethod
    def from_ccxt_position(cls, pos: dict[str, Any]) -> PositionInfo:
        """
        Create PositionInfo from CCXT position structure.

        Args:
            pos: CCXT position dict from fetch_positions()

        Returns:
            PositionInfo with extracted data.
        """
        info = pos.get("info", {})

        # Parse timestamp
        ts = pos.get("timestamp")
        timestamp = None
        if ts:
            try:
                timestamp = datetime.fromtimestamp(ts / 1000)
            except (OSError, ValueError, TypeError):
                pass

        # Get leverage from info if not in main dict
        leverage = pos.get("leverage")
        if leverage is None:
            # Calculate from initial margin percentage
            init_margin_pct = pos.get("initialMarginPercentage", 0)
            if init_margin_pct and init_margin_pct > 0:
                leverage = int(1 / init_margin_pct)

        return cls(
            symbol=pos.get("symbol", ""),
            side=(pos.get("side") or "").lower(),
            contracts=float(pos.get("contracts", 0) or 0),
            contract_size=float(pos.get("contractSize", 1) or 1),
            entry_price=float(pos.get("entryPrice", 0) or 0),
            mark_price=float(pos.get("markPrice", 0) or 0),
            liquidation_price=float(pos.get("liquidationPrice", 0) or 0),
            unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
            unrealized_pnl_pct=float(pos.get("percentage", 0) or 0),
            notional=float(pos.get("notional", 0) or 0),
            leverage=leverage or 0,
            margin_mode=pos.get("marginMode", "cross") or "cross",
            initial_margin=float(pos.get("initialMargin", 0) or 0),
            maint_margin=float(pos.get("maintenanceMargin", 0) or 0),
            margin_ratio=float(pos.get("marginRatio", 0) or 0),
            timestamp=timestamp,
            raw_info=info,
        )

    @property
    def base_symbol(self) -> str:
        """Extract base symbol (e.g., 'SQD' from 'SQD/USDT:USDT')."""
        if "/" in self.symbol:
            return self.symbol.split("/")[0]
        return self.symbol

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.side.lower() == "long"

    @property
    def is_profitable(self) -> bool:
        """Check if position is in profit."""
        return self.unrealized_pnl > 0


class PositionsPane(VerticalScroll, ThemeColorsMixin):
    """
    Pane displaying active trading positions.

    Shows all open positions with entry price, mark price,
    unrealized PnL, and other position details.

    Supports both REST API polling and real-time WebSocket updates.
    """

    class PositionsPnlUpdated(Message):
        """Posted when position PnL data is updated."""

        def __init__(self, pnl_by_symbol: dict[str, float]) -> None:
            super().__init__()
            self.pnl_by_symbol = pnl_by_symbol
            """Dict mapping symbol to aggregated PnL percentage."""

    DEFAULT_CSS = """
    PositionsPane {
        padding: 0 1;
        background: transparent;
    }

    PositionsPane #stream-status {
        dock: top;
        height: auto;
        padding: 0 0 1 0;
    }

    PositionsPane #no-positions-message {
        color: $text-muted;
        text-style: italic;
        padding: 1;
    }
    """

    positions: reactive[list[PositionInfo]] = reactive(list, always_update=True)
    is_streaming: reactive[bool] = reactive(False)
    total_updates: reactive[int] = reactive(0)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._selected_symbol: str | None = None
        self._all_positions: list[PositionInfo] = []
        self._last_update: datetime | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="stream-status")
        yield Static(
            "No active positions",
            id="no-positions-message",
        )
        yield Static("", id="positions-content")

    def on_mount(self) -> None:
        self._update_status()

    def watch_positions(self, positions: list[PositionInfo]) -> None:
        """Update display when positions change."""
        self._update_status()
        self._update_positions_display()

    def watch_is_streaming(self, streaming: bool) -> None:
        """Update status when streaming state changes."""
        self._update_status()

    def watch_total_updates(self, count: int) -> None:
        """Update status when update count changes."""
        self._update_status()

    def _update_status(self) -> None:
        """Update the status line."""
        status = self.query_one("#stream-status", Static)
        status_text = Text()

        muted = self.muted_style

        if self.is_streaming:
            status_text.append("● LIVE", style=self.bold_success_style)
        else:
            status_text.append("○ IDLE", style=self.bold_warning_style)

        # Show position count
        displayed = len(self.positions)
        total = len(self._all_positions)
        if self._selected_symbol:
            status_text.append(f"  [{displayed}/{total}]", style=muted)
        else:
            status_text.append(f"  [{displayed}]", style=muted)

        status_text.append(f"  Updates: {self.total_updates}", style=muted)

        if self._last_update:
            status_text.append(f"  Last: {self._last_update.strftime('%H:%M:%S')}", style=muted)

        status.update(status_text)

    def _update_positions_display(self) -> None:
        """Update the positions list display."""
        no_pos_msg = self.query_one("#no-positions-message", Static)
        content = self.query_one("#positions-content", Static)

        if not self.positions:
            no_pos_msg.display = True
            content.display = False
            return

        no_pos_msg.display = False
        content.display = True

        # Build the positions display
        content.update(self._build_positions_display(self.positions))

    @staticmethod
    def _format_price(value: float, precision: int = 8) -> str:
        """Format price without scientific notation."""
        if value == 0:
            return "0"
        if value >= 1:
            return f"{value:,.{min(precision, 4)}f}".rstrip("0").rstrip(".")
        formatted = f"{value:.{precision}f}".rstrip("0").rstrip(".")
        return formatted

    @staticmethod
    def _format_pnl(value: float) -> str:
        """Format PnL with sign and color indicator."""
        if value >= 0:
            return f"+{value:.4f}"
        return f"{value:.4f}"

    def _build_positions_display(self, positions: list[PositionInfo]) -> Text:
        """Build rich text display of positions."""
        text = Text()

        # Get theme colors
        accent = self.accent_color
        warning = self.warning_color
        muted = self.muted_style

        # Group positions by symbol to detect hedge pairs
        by_symbol: dict[str, list[PositionInfo]] = {}
        for pos in positions:
            symbol = pos.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(pos)

        # Identify hedge pairs (same symbol, both sides)
        hedge_pairs: list[tuple[PositionInfo, PositionInfo]] = []
        single_positions: list[PositionInfo] = []

        for symbol, pos_list in by_symbol.items():
            if len(pos_list) == 2:
                # Check if we have both long and short
                long_pos = next((p for p in pos_list if p.is_long), None)
                short_pos = next((p for p in pos_list if not p.is_long), None)
                if long_pos and short_pos:
                    hedge_pairs.append((short_pos, long_pos))  # stuck first, then hedge
                else:
                    single_positions.extend(pos_list)
            else:
                single_positions.extend(pos_list)

        # Summary
        total_pnl = sum(p.unrealized_pnl for p in positions)
        long_count = sum(1 for p in positions if p.is_long)
        short_count = len(positions) - long_count
        hedged_count = len(hedge_pairs)

        text.append("Summary\n", style=f"bold {accent}")
        text.append(f"  Total Positions  ", style=muted)
        text.append(f"{len(positions)}")
        text.append(f" ({long_count}L / {short_count}S)")
        if hedged_count > 0:
            text.append(f" [{hedged_count} hedged]", style=warning)
        text.append("\n")
        text.append(f"  Total uPnL       ", style=muted)
        pnl_style = self.pnl_style(total_pnl)
        text.append(f"{self._format_pnl(total_pnl)} USDT\n", style=pnl_style)
        text.append("\n")

        # Display hedge pairs first
        for stuck, hedge in hedge_pairs:
            self._append_hedge_pair(text, stuck, hedge)

        # Then display individual positions
        for i, pos in enumerate(single_positions):
            self._append_position(text, pos, i + 1)

        return text

    def _append_hedge_pair(self, text: Text, stuck: PositionInfo, hedge: PositionInfo) -> None:
        """Append a hedge pair display showing combined PnL and recovery."""
        # Get theme colors
        success = self.success_color
        error = self.error_color
        warning = self.warning_color
        muted = self.muted_style

        # Header with HEDGED indicator
        text.append(f"{stuck.base_symbol} ", style="bold")
        text.append("HEDGED\n", style=f"bold {warning}")

        # Stuck position (usually losing)
        stuck_side = "SHORT" if not stuck.is_long else "LONG"
        stuck_style = error if not stuck.is_long else success
        stuck_pnl_style = self.pnl_style(stuck.unrealized_pnl)
        text.append(f"  {stuck_side}:  ", style=f"{muted} {stuck_style}")
        text.append(f"{stuck.unrealized_pnl:+.2f} USDT ", style=stuck_pnl_style)
        text.append(f"({stuck.unrealized_pnl_pct:+.0f}%)\n", style=stuck_pnl_style)

        # Hedge position (usually winning or smaller loss)
        hedge_side = "LONG" if hedge.is_long else "SHORT"
        hedge_style = success if hedge.is_long else error
        hedge_pnl_style = self.pnl_style(hedge.unrealized_pnl)
        text.append(f"  {hedge_side}:  ", style=f"{muted} {hedge_style}")
        text.append(f"{hedge.unrealized_pnl:+.2f} USDT ", style=hedge_pnl_style)
        text.append(f"({hedge.unrealized_pnl_pct:+.0f}%)\n", style=hedge_pnl_style)

        # Combined PnL
        combined_pnl = stuck.unrealized_pnl + hedge.unrealized_pnl
        combined_style = self.bold_pnl_style(combined_pnl)
        text.append(f"  Combined:  ", style=muted)
        text.append(f"{combined_pnl:+.2f} USDT\n", style=combined_style)

        # Recovery percentage (how much of max loss has been recovered)
        # Max loss is the more negative of the two at their worst
        max_loss = min(stuck.unrealized_pnl, 0)  # The stuck position's loss
        if max_loss < 0:
            # Recovery = how much better combined is vs stuck alone
            recovery_pct = ((max_loss - combined_pnl) / abs(max_loss)) * 100
            recovery_pct = max(0, min(100, recovery_pct))  # Clamp 0-100
            recovery_style = success if recovery_pct >= 50 else warning
            text.append(f"  Recovery:  ", style=muted)
            text.append(f"{recovery_pct:.0f}% of max loss\n", style=recovery_style)

        # Size info
        text.append(f"  Notional:  ", style=muted)
        text.append(f"{stuck.notional:,.2f} + {hedge.notional:,.2f} USDT\n")

        text.append("\n")

    def _append_position(self, text: Text, pos: PositionInfo, index: int) -> None:
        """Append a single position to the text display."""
        # Get theme colors
        success = self.success_color
        error = self.error_color
        warning = self.warning_color
        muted = self.muted_style

        # Position header with side indicator
        side_style = success if pos.is_long else error
        side_label = "LONG" if pos.is_long else "SHORT"

        text.append(f"{pos.base_symbol} ", style="bold")
        text.append(f"{side_label}\n", style=f"bold {side_style}")

        # Size and notional
        text.append(f"  Size             ", style=muted)
        text.append(f"{pos.contracts:,.0f} contracts\n")
        text.append(f"  Notional         ", style=muted)
        text.append(f"{pos.notional:,.2f} USDT\n")

        # Prices
        text.append(f"  Entry Price      ", style=muted)
        text.append(f"{self._format_price(pos.entry_price)}\n")
        text.append(f"  Mark Price       ", style=muted)
        text.append(f"{self._format_price(pos.mark_price)}\n")

        # PnL
        pnl_style = self.pnl_style(pos.unrealized_pnl)
        text.append(f"  Unrealized PnL   ", style=muted)
        text.append(f"{self._format_pnl(pos.unrealized_pnl)} USDT ", style=pnl_style)
        text.append(f"({pos.unrealized_pnl_pct:+.2f}%)\n", style=pnl_style)

        # Margin info
        text.append(f"  Margin Mode      ", style=muted)
        text.append(f"{pos.margin_mode.title()}")
        if pos.leverage:
            text.append(f" ({pos.leverage}x)")
        text.append("\n")
        text.append(f"  Initial Margin   ", style=muted)
        text.append(f"{pos.initial_margin:,.4f} USDT\n")

        # Liquidation
        if pos.liquidation_price and pos.liquidation_price < 1e10:
            text.append(f"  Liq. Price       ", style=muted)
            text.append(f"{self._format_price(pos.liquidation_price)}\n", style=warning)

        # Timestamp
        if pos.timestamp:
            text.append(f"  Updated          ", style=muted)
            text.append(f"{pos.timestamp.strftime('%H:%M:%S')}\n")

        text.append("\n")

    def update_positions(self, positions_data: list[dict[str, Any]]) -> None:
        """
        Update positions from raw CCXT data (REST API or WebSocket).

        Args:
            positions_data: List of position dicts from fetch_positions() or watch_positions()
        """
        self._last_update = datetime.now()

        # Filter to only active positions (non-zero contracts)
        active_positions = []
        pnl_by_symbol: dict[str, float] = {}

        for pos_data in positions_data:
            contracts = float(pos_data.get("contracts", 0) or 0)
            if contracts != 0:
                pos_info = PositionInfo.from_ccxt_position(pos_data)
                active_positions.append(pos_info)

                # Aggregate PnL per symbol (for hedged positions)
                symbol = pos_info.symbol
                if symbol not in pnl_by_symbol:
                    pnl_by_symbol[symbol] = 0.0
                pnl_by_symbol[symbol] += pos_info.unrealized_pnl_pct

        self._all_positions = active_positions
        self.total_updates += 1

        # Mark as streaming if we're receiving updates
        if not self.is_streaming:
            self.is_streaming = True

        # Apply symbol filter if set
        self._apply_symbol_filter()

        # Post message with PnL data for symbol browser
        if pnl_by_symbol:
            self.post_message(self.PositionsPnlUpdated(pnl_by_symbol))

    def _apply_symbol_filter(self) -> None:
        """Apply symbol filter to positions."""
        if self._selected_symbol:
            self.positions = [p for p in self._all_positions if p.symbol == self._selected_symbol]
        else:
            self.positions = self._all_positions.copy()

    def set_symbol(self, symbol: str | None) -> None:
        """
        Set the symbol to filter positions.

        Args:
            symbol: Symbol to filter by, or None to show all.
        """
        if symbol != self._selected_symbol:
            self._selected_symbol = symbol
            self._apply_symbol_filter()

    def set_streaming(self, streaming: bool) -> None:
        """Set streaming status."""
        self.is_streaming = streaming

    def clear(self) -> None:
        """Clear all positions and reset state."""
        self._all_positions = []
        self._selected_symbol = None
        self.positions = []
        self.total_updates = 0
        self.is_streaming = False
        self._last_update = None
        self._update_status()
        self._update_positions_display()
