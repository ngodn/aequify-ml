"""
Symbol browser widget for displaying filtered trading symbols.

Shows symbols from the filter pipeline with:
- Price change percentage
- Active position indicator
- Selection handling for symbol management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.style import Style
from rich.text import Text, TextType
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import Reactive, reactive
from textual.widgets import LoadingIndicator, Static, Tree
from textual.widgets.tree import TreeNode

from aequify.tui.help_data import HelpData
from aequify.tui.widgets.theme_colors import ThemeColorsMixin
from aequify.tui.widgets.tree import AequifyTree


@dataclass
class SymbolData:
    """
    Data for a trading symbol.

    Attributes:
        symbol: CCXT format symbol (e.g., "PIPPIN/USDT:USDT").
        base: Base currency (e.g., "PIPPIN").
        quote: Quote currency (e.g., "USDT").
        price_change_pct: 24h price change percentage.
        has_position: Whether symbol has an active position.
        has_long: Whether symbol has an active long position.
        has_short: Whether symbol has an active short position.
        position_pnl: Unrealized PnL percentage if active.
        init_stage: Initialization stage ("backfilling", "bootstrapping", or empty).
    """

    symbol: str
    base: str = ""
    quote: str = "USDT"
    price_change_pct: float = 0.0
    has_position: bool = False
    has_long: bool = False
    has_short: bool = False
    position_pnl: float = 0.0
    init_stage: str = ""  # "backfilling", "bootstrapping", or empty when done
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.base and "/" in self.symbol:
            self.base = self.symbol.split("/")[0]

    @property
    def display_name(self) -> str:
        """Get display name (base currency)."""
        return self.base or self.symbol


class SymbolTree(AequifyTree[SymbolData], ThemeColorsMixin):
    """
    Tree widget for displaying trading symbols.

    Shows symbols with:
    - Price change indicator (color-coded)
    - Position indicator if active
    """

    help = HelpData(
        title="Symbol Browser",
        description="""\
Shows filtered trading symbols from the filter pipeline.
- Symbols are sorted by absolute price change
- Active positions are highlighted
- Press Enter to select a symbol for management
- `j` and `k` can be used to navigate
- `g` and `G` jumps to top/bottom
""",
    )

    BINDING_GROUP_TITLE = "Symbol Browser"

    BINDINGS = [
        Binding("r", "refresh_symbols", "Refresh", tooltip="Refresh symbol list"),
        Binding("enter", "select_symbol", "Select", tooltip="Select symbol for management"),
    ]

    COMPONENT_CLASSES = {
        "node-selected",
        "node-has-position",
        "price-positive",
        "price-negative",
    }

    def __init__(
        self,
        label: TextType = "Symbols",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            label,
            data=None,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self._symbols: dict[str, SymbolData] = {}

    @dataclass
    class SymbolSelected(Message):
        """Posted when a symbol is selected."""

        symbol: SymbolData
        node: TreeNode[SymbolData]
        tree: "SymbolTree"

        @property
        def control(self) -> "SymbolTree":
            return self.tree

    @dataclass
    class SymbolsUpdated(Message):
        """Posted when symbols are updated."""

        count: int
        tree: "SymbolTree"

        @property
        def control(self) -> "SymbolTree":
            return self.tree

    @dataclass
    class RefreshRequested(Message):
        """Posted when user requests refresh."""

        tree: "SymbolTree"

        @property
        def control(self) -> "SymbolTree":
            return self.tree

    currently_selected: Reactive[TreeNode[SymbolData] | None] = reactive(None)

    def watch_currently_selected(self, node: TreeNode[SymbolData] | None) -> None:
        if node and node.data:
            self.post_message(
                self.SymbolSelected(
                    symbol=node.data,
                    node=node,
                    tree=self,
                )
            )

    def render_label(
        self,
        node: TreeNode[SymbolData],
        base_style: Style,
        style: Style,
    ) -> Text:
        """Render label for a symbol node."""
        if not self.is_mounted or node.data is None:
            return Text(str(node._label))

        symbol_data = node.data

        # Get theme colors
        success = self.success_color
        error = self.error_color
        warning = self.warning_color
        accent = self.accent_color
        secondary = self.secondary_color
        muted = self.muted_style

        # Build the label
        parts = []

        # Position indicator (L for long, S for short, H for hedged)
        if symbol_data.has_position:
            if symbol_data.has_long and symbol_data.has_short:
                # Hedged position - show H in warning color
                parts.append(("H ", f"bold {warning}"))
            elif symbol_data.has_long:
                parts.append(("L ", success))
            elif symbol_data.has_short:
                parts.append(("S ", error))
            else:
                parts.append(("  ", ""))
        else:
            parts.append(("  ", ""))

        # Symbol name
        name_style = ""
        if node is self.currently_selected:
            name_style = self.get_component_rich_style("node-selected")
        parts.append((symbol_data.display_name, name_style))

        # Price change
        pct = symbol_data.price_change_pct
        if pct > 0:
            pct_text = f" +{pct:.1f}%"
            parts.append((pct_text, success))
        elif pct < 0:
            pct_text = f" {pct:.1f}%"
            parts.append((pct_text, error))
        else:
            pct_text = " 0.0%"
            parts.append((pct_text, muted))

        # Init stage (backfilling/bootstrapping) in italic
        if symbol_data.init_stage == "backfilling":
            parts.append((" (backfilling...)", f"italic {muted}"))
        elif symbol_data.init_stage == "bootstrapping":
            parts.append((" (bootstrapping...)", f"italic {muted}"))

        # Position PnL if active (use accent/secondary to differentiate from 24H% colors)
        if symbol_data.has_position and symbol_data.position_pnl != 0:
            pnl = symbol_data.position_pnl
            if pnl > 0:
                pnl_style = f"bold {accent}"
                pnl_text = f" ({pnl:+.1f}%)"
            else:
                pnl_style = f"bold {secondary}"
                pnl_text = f" ({pnl:.1f}%)"
            parts.append((pnl_text, pnl_style))

        text = Text()
        for content, text_style in parts:
            text.append(content, style=text_style)

        text.stylize(style)
        return text

    @on(Tree.NodeSelected)
    def on_node_selected(self, event: Tree.NodeSelected[SymbolData]) -> None:
        event.stop()
        if event.node.data:
            self.currently_selected = event.node
            self._clear_line_cache()
            self.refresh()

    def action_refresh_symbols(self) -> None:
        """Request a refresh of the symbol list."""
        self.post_message(self.RefreshRequested(tree=self))

    def action_select_symbol(self) -> None:
        """Select the current symbol for management."""
        cursor_node = self.cursor_node
        if cursor_node and cursor_node.data:
            self.currently_selected = cursor_node
            self._clear_line_cache()
            self.refresh()

    def update_symbols(self, symbols: list[SymbolData]) -> None:
        """
        Update the symbol list.

        Args:
            symbols: List of SymbolData to display.
        """
        # Clear existing
        self.root.remove_children()
        self._symbols.clear()

        # Deduplicate by symbol - keep the one with position info if available
        unique_symbols: dict[str, SymbolData] = {}
        for symbol_data in symbols:
            key = symbol_data.symbol
            if key not in unique_symbols:
                unique_symbols[key] = symbol_data
            else:
                # Prefer the one with position data
                existing = unique_symbols[key]
                if symbol_data.has_position and not existing.has_position:
                    unique_symbols[key] = symbol_data
                elif symbol_data.has_position and existing.has_position:
                    # Merge position data
                    existing.has_long = existing.has_long or symbol_data.has_long
                    existing.has_short = existing.has_short or symbol_data.has_short
                    if symbol_data.position_pnl != 0:
                        existing.position_pnl = symbol_data.position_pnl

        # Sort by absolute price change (highest first)
        sorted_symbols = sorted(
            unique_symbols.values(),
            key=lambda s: abs(s.price_change_pct),
            reverse=True,
        )

        # Add to tree
        for symbol_data in sorted_symbols:
            self._symbols[symbol_data.symbol] = symbol_data
            self.root.add_leaf(symbol_data.display_name, data=symbol_data)

        self.root.expand()

        # Reset cursor
        if self.root.children:
            self.cursor_line = 0

        self.post_message(self.SymbolsUpdated(count=len(symbols), tree=self))
        self.refresh()

    def select_first(self) -> bool:
        """
        Select the first symbol in the tree programmatically.

        Returns:
            True if a symbol was selected, False if tree is empty.
        """
        if self.root.children:
            first_node = self.root.children[0]
            if first_node.data:
                self.cursor_line = 0
                self.currently_selected = first_node
                self._clear_line_cache()
                self.refresh()
                return True
        return False

    def get_symbol(self, symbol: str) -> SymbolData | None:
        """Get symbol data by symbol string."""
        return self._symbols.get(symbol)

    def update_symbol_pnl(self, symbol: str, pnl: float) -> None:
        """Update PnL for a specific symbol and refresh its display."""
        if symbol in self._symbols:
            self._symbols[symbol].position_pnl = pnl
            # Force tree to re-render by refreshing
            self.refresh()

    def update_symbol_position(
        self,
        symbol: str,
        has_position: bool,
        has_long: bool = False,
        has_short: bool = False,
        pnl: float = 0.0,
    ) -> None:
        """Update position status for a specific symbol."""
        if symbol in self._symbols:
            self._symbols[symbol].has_position = has_position
            self._symbols[symbol].has_long = has_long
            self._symbols[symbol].has_short = has_short
            self._symbols[symbol].position_pnl = pnl
            self._clear_line_cache()
            self.refresh()

    def clear_all_positions(self) -> None:
        """Clear position flags from all symbols."""
        for symbol_data in self._symbols.values():
            symbol_data.has_position = False
            symbol_data.has_long = False
            symbol_data.has_short = False
            symbol_data.position_pnl = 0.0
        self._clear_line_cache()
        self.refresh()

    def update_symbol_init_stage(self, symbol: str, init_stage: str) -> None:
        """Update init stage for a specific symbol (backfilling, bootstrapping, or empty)."""
        if symbol in self._symbols:
            self._symbols[symbol].init_stage = init_stage
            self._clear_line_cache()
            self.refresh()

    @property
    def symbol_count(self) -> int:
        """Get number of symbols."""
        return len(self._symbols)


class SymbolPreview(VerticalScroll, ThemeColorsMixin):
    """Preview panel for selected symbol."""

    symbol: Reactive[SymbolData | None] = reactive(None)

    def compose(self) -> ComposeResult:
        self.can_focus = False
        yield Static("", id="symbol-info")

    def watch_symbol(self, symbol: SymbolData | None) -> None:
        self._render_symbol(symbol)

    def _render_symbol(self, symbol: SymbolData | None) -> None:
        """Render symbol info to the preview."""
        self.set_class(symbol is None, "hidden")
        if symbol:
            info = self.query_one("#symbol-info", Static)

            # Get theme colors
            success = self.success_color
            error = self.error_color
            warning = self.warning_color

            lines = [
                f"[b]{symbol.display_name}[/b]",
                f"24h Change: {symbol.price_change_pct:+.2f}%",
            ]
            if symbol.has_position:
                if symbol.has_long and symbol.has_short:
                    lines.append(f"[{warning} bold]Position: HEDGED[/]")
                    lines.append("  LONG + SHORT active")
                elif symbol.has_long:
                    lines.append(f"[{success}]Position: LONG[/]")
                elif symbol.has_short:
                    lines.append(f"[{error}]Position: SHORT[/]")
                else:
                    lines.append("Position: Active")
                pnl_color = success if symbol.position_pnl >= 0 else error
                lines.append(f"[{pnl_color}]PnL: {symbol.position_pnl:+.2f}%[/]")
            info.update("\n".join(lines))

    def refresh_display(self) -> None:
        """Force refresh the display with current symbol data."""
        self._render_symbol(self.symbol)


class SymbolBrowser(Vertical):
    """
    Browser widget for filtered trading symbols.

    Shows symbols from the filter pipeline with price changes
    and position indicators.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._loading = False

    def compose(self) -> ComposeResult:
        self.border_title = "Side/Symbol/24H%/(PnL)"
        self.add_class("section")

        yield Static(
            "[i]No symbols loaded.[/]\n\nWaiting for filter pipeline...",
            id="empty-symbols-label",
        )
        yield LoadingIndicator(id="symbols-loading")

        tree = SymbolTree(
            label="Symbols",
            id="symbol-tree",
        )
        tree.guide_depth = 1
        tree.show_root = False
        tree.show_guides = False
        yield tree

        yield SymbolPreview(id="symbol-preview")

    def on_mount(self) -> None:
        self.query_one("#symbols-loading", LoadingIndicator).display = False

    @on(SymbolTree.SymbolSelected)
    def on_symbol_selected(self, event: SymbolTree.SymbolSelected) -> None:
        if event.symbol:
            self.symbol_preview.symbol = event.symbol

    @on(SymbolTree.SymbolsUpdated)
    def on_symbols_updated(self, event: SymbolTree.SymbolsUpdated) -> None:
        empty_label = self.query_one("#empty-symbols-label", Static)
        empty_label.display = event.count == 0
        self.border_subtitle = f"{event.count} symbols"
        self._loading = False
        self.query_one("#symbols-loading", LoadingIndicator).display = False

        # Update preview if a symbol is selected (refresh with new data)
        current = self.symbol_preview.symbol
        if current:
            updated = self.symbol_tree.get_symbol(current.symbol)
            if updated:
                self.symbol_preview.symbol = updated

    @on(SymbolTree.RefreshRequested)
    def on_refresh_requested(self, event: SymbolTree.RefreshRequested) -> None:
        # Bubble up to app level for handling
        pass

    def set_loading(self, loading: bool) -> None:
        """Set loading state."""
        self._loading = loading
        self.query_one("#symbols-loading", LoadingIndicator).display = loading
        if loading:
            self.query_one("#empty-symbols-label", Static).display = False

    def update_symbols(self, symbols: list[SymbolData]) -> None:
        """Update the symbol list."""
        self.symbol_tree.update_symbols(symbols)

    def select_first(self) -> bool:
        """Select the first symbol in the tree programmatically."""
        return self.symbol_tree.select_first()

    def update_symbol_pnl(self, symbol: str, pnl: float) -> None:
        """Update PnL for a specific symbol."""
        self.symbol_tree.update_symbol_pnl(symbol, pnl)
        # Refresh preview if this symbol is selected
        if self.symbol_preview.symbol and self.symbol_preview.symbol.symbol == symbol:
            self.symbol_preview.refresh_display()

    def update_symbol_position(
        self,
        symbol: str,
        has_position: bool,
        has_long: bool = False,
        has_short: bool = False,
        pnl: float = 0.0,
    ) -> None:
        """Update position status for a specific symbol."""
        self.symbol_tree.update_symbol_position(symbol, has_position, has_long, has_short, pnl)
        # Refresh preview if this symbol is selected
        if self.symbol_preview.symbol and self.symbol_preview.symbol.symbol == symbol:
            self.symbol_preview.symbol = self.symbol_tree.get_symbol(symbol)

    def clear_all_positions(self) -> None:
        """Clear position flags from all symbols."""
        self.symbol_tree.clear_all_positions()

    def update_symbol_init_stage(self, symbol: str, init_stage: str) -> None:
        """Update init stage for a specific symbol (backfilling, bootstrapping, or empty)."""
        self.symbol_tree.update_symbol_init_stage(symbol, init_stage)

    @property
    def symbol_tree(self) -> SymbolTree:
        return self.query_one(SymbolTree)

    @property
    def symbol_preview(self) -> SymbolPreview:
        return self.query_one(SymbolPreview)

    @property
    def selected_symbol(self) -> SymbolData | None:
        """Get currently selected symbol."""
        node = self.symbol_tree.currently_selected
        return node.data if node else None
