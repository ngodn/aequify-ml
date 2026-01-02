"""
Selected symbol area widget.

Main widget for displaying detailed information and actions
for a selected trading symbol. Uses a tabbed interface similar
to RequestEditor.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static, TabPane

from aequify.tui.widgets.symbols.apex_chart_pane import APEXChartPane
from aequify.tui.widgets.symbols.browser import SymbolData
from aequify.tui.widgets.symbols.positions_pane import PositionsPane
from aequify.tui.widgets.symbols.symbol_info import MarketInfo, SymbolInfoPane
from aequify.tui.widgets.symbols.trade_stream_pane import TradeStreamPane
from aequify.tui.widgets.tabbed_content import AequifyTabbedContent


class SelectedSymbolTabbedContent(AequifyTabbedContent):
    """Tabbed content for selected symbol area."""

    pass


class SelectedSymbolArea(Vertical):
    """
    Area for displaying selected symbol information and actions.

    Shows detailed info about the selected symbol in a tabbed interface:
    - Symbol Info: Contract details, precision, limits
    - Positions: Active trading positions
    """

    DEFAULT_CSS = """
    SelectedSymbolArea {
        height: 2fr;
    }

    SelectedSymbolArea #no-symbol-selected {
        color: $text-muted;
        text-style: italic;
        padding: 1 2;
        height: 1fr;
        content-align: center middle;
        hatch: right $surface-lighten-1 70%;
    }

    SelectedSymbolArea SelectedSymbolTabbedContent {
        height: 1fr;
    }
    """

    # Currently selected symbol
    selected_symbol: reactive[SymbolData | None] = reactive(None)

    # Market data cache (keyed by symbol string)
    _market_cache: dict[str, dict[str, Any]]

    # Positions cache
    _positions_cache: list[dict[str, Any]]

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._market_cache = {}
        self._positions_cache = []

    def compose(self) -> ComposeResult:
        yield Static(
            "Select a symbol from the list",
            id="no-symbol-selected",
        )
        with SelectedSymbolTabbedContent(id="symbol-tabs"):
            with TabPane("Symbol Info", id="symbol-info-pane"):
                yield SymbolInfoPane(id="symbol-info")
            with TabPane("RTDS:@position", id="positions-pane"):
                yield PositionsPane(id="positions")
            with TabPane("RTDS:@trade", id="trade-stream-pane"):
                yield TradeStreamPane(id="trade-stream")
            with TabPane("Engine: APEX", id="apex-chart-pane"):
                yield APEXChartPane(id="apex-chart")

    def on_mount(self) -> None:
        self.border_title = "Selected Symbol"
        self.add_class("section")
        # Initially hide tabs until a symbol is selected
        self._update_display()

    def watch_selected_symbol(self, symbol: SymbolData | None) -> None:
        """Handle symbol selection changes."""
        self._update_display()

        if symbol:
            self.border_title = f"Symbol: {symbol.display_name}"
            self._load_symbol_info(symbol)
            # Update positions filtered by selected symbol
            self.positions_pane.set_symbol(symbol.symbol)
            # Update trade stream to filter by selected symbol
            self.trade_stream_pane.set_symbol(symbol.symbol)
            # Update APEX chart to filter by selected symbol
            self.apex_chart_pane.set_symbol(symbol.symbol)
        else:
            self.border_title = "Selected Symbol"
            self.symbol_info_pane.clear()
            self.positions_pane.set_symbol(None)
            self.trade_stream_pane.set_symbol(None)
            self.apex_chart_pane.set_symbol(None)

    def _update_display(self) -> None:
        """Update visibility of content based on selection."""
        no_symbol = self.query_one("#no-symbol-selected", Static)
        tabs = self.query_one("#symbol-tabs", SelectedSymbolTabbedContent)

        if self.selected_symbol:
            no_symbol.display = False
            tabs.display = True
        else:
            no_symbol.display = True
            tabs.display = False

    def _load_symbol_info(self, symbol: SymbolData) -> None:
        """Load and display symbol info."""
        # Check cache first
        market = self._market_cache.get(symbol.symbol)

        # Update the info pane
        self.symbol_info_pane.update_from_symbol(symbol, market)

    def update_market_data(self, markets: dict[str, dict[str, Any]]) -> None:
        """
        Update the market data cache.

        Args:
            markets: Dict of symbol -> market data from exchange.
        """
        self._market_cache = markets

        # If a symbol is selected, refresh its info
        if self.selected_symbol:
            market = markets.get(self.selected_symbol.symbol)
            self.symbol_info_pane.update_from_symbol(self.selected_symbol, market)

    def update_positions(self, positions: list[dict[str, Any]]) -> None:
        """
        Update the positions display (from REST API or WebSocket).

        Args:
            positions: List of position dicts from fetch_positions() or watch_positions()
        """
        self._positions_cache = positions
        # PositionsPane handles filtering internally via set_symbol
        self.positions_pane.update_positions(positions)

    def select_symbol(self, symbol: SymbolData) -> None:
        """
        Select a symbol to display.

        Args:
            symbol: The symbol data to display.
        """
        self.selected_symbol = symbol

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self.selected_symbol = None

    @property
    def symbol_info_pane(self) -> SymbolInfoPane:
        """Get the symbol info pane."""
        return self.query_one("#symbol-info", SymbolInfoPane)

    @property
    def positions_pane(self) -> PositionsPane:
        """Get the positions pane."""
        return self.query_one("#positions", PositionsPane)

    @property
    def trade_stream_pane(self) -> TradeStreamPane:
        """Get the trade stream pane."""
        return self.query_one("#trade-stream", TradeStreamPane)

    @property
    def apex_chart_pane(self) -> APEXChartPane:
        """Get the APEX chart pane."""
        return self.query_one("#apex-chart", APEXChartPane)

    @property
    def has_selection(self) -> bool:
        """Check if a symbol is selected."""
        return self.selected_symbol is not None
