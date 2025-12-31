"""
Symbol information display widget.

Shows detailed market information for a selected trading symbol,
including contract details, trading rules, and precision info.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from aequify.tui.widgets.symbols.browser import SymbolData
from aequify.tui.widgets.theme_colors import ThemeColorsMixin


@dataclass
class MarketInfo:
    """
    Detailed market information from exchange.

    Contains both CCXT unified fields and Binance-specific info.
    """

    # Core identification
    symbol: str = ""  # CCXT format: "BTC/USDT:USDT"
    id: str = ""  # Exchange format: "BTCUSDT"
    base: str = ""  # Base currency: "BTC"
    quote: str = ""  # Quote currency: "USDT"
    settle: str = ""  # Settlement currency: "USDT"

    # Contract info
    contract_type: str = ""  # "perpetual", "future", etc.
    margin_mode: str = ""  # "cross", "isolated"
    is_linear: bool = True  # Linear (USDT-margined) vs Inverse

    # Status
    active: bool = True
    status: str = "TRADING"  # TRADING, SETTLING, etc.

    # Precision
    price_precision: int = 0
    quantity_precision: int = 0
    base_precision: int = 8
    quote_precision: int = 8

    # Limits
    min_quantity: float = 0.0
    max_quantity: float = 0.0
    min_notional: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    tick_size: float = 0.0
    step_size: float = 0.0

    # Margin requirements
    maint_margin_pct: float = 0.0
    required_margin_pct: float = 0.0

    # Risk parameters
    liquidation_fee: float = 0.0
    market_take_bound: float = 0.0
    trigger_protect: float = 0.0

    # Dates
    onboard_date: datetime | None = None
    delivery_date: datetime | None = None

    # Raw exchange info (for anything not covered above)
    raw_info: dict[str, Any] | None = None

    @classmethod
    def from_ccxt_market(cls, market: dict[str, Any]) -> MarketInfo:
        """
        Create MarketInfo from CCXT market structure.

        Args:
            market: CCXT market dict from exchange.markets[symbol]

        Returns:
            MarketInfo with extracted data.
        """
        info = market.get("info", {})
        limits = market.get("limits", {})

        # Parse dates from timestamps (milliseconds)
        onboard_ts = info.get("onboardDate")
        delivery_ts = info.get("deliveryDate")

        onboard_date = None
        delivery_date = None

        # Handle both string and int timestamps
        if onboard_ts:
            try:
                ts = int(onboard_ts) if isinstance(onboard_ts, str) else onboard_ts
                onboard_date = datetime.fromtimestamp(ts / 1000)
            except (OSError, ValueError, TypeError):
                pass

        if delivery_ts:
            try:
                ts = int(delivery_ts) if isinstance(delivery_ts, str) else delivery_ts
                # Skip far-future dates (perpetual contracts have placeholder dates)
                if ts < 4000000000000:  # Before year 2096
                    delivery_date = datetime.fromtimestamp(ts / 1000)
            except (OSError, ValueError, TypeError):
                pass

        # Extract filter values
        filters = {f.get("filterType"): f for f in info.get("filters", [])}

        price_filter = filters.get("PRICE_FILTER", {})
        lot_size = filters.get("LOT_SIZE", {})
        min_notional_filter = filters.get("MIN_NOTIONAL", {})

        # Get precision from raw info (decimal count), not CCXT precision (step value)
        # CCXT precision is stored as the actual step (e.g., 1e-05), not decimal count
        price_precision = int(info.get("pricePrecision", 0) or 0)
        quantity_precision = int(info.get("quantityPrecision", 0) or 0)

        return cls(
            # Core identification
            symbol=market.get("symbol", ""),
            id=market.get("id", ""),
            base=market.get("base", ""),
            quote=market.get("quote", ""),
            settle=market.get("settle", ""),
            # Contract info
            contract_type=(info.get("contractType") or "").lower() or "perpetual",
            margin_mode=info.get("marginAsset", "USDT"),
            is_linear=market.get("linear", True),
            # Status
            active=market.get("active", True),
            status=info.get("status", "TRADING"),
            # Precision (from raw Binance info - these are decimal counts)
            price_precision=price_precision,
            quantity_precision=quantity_precision,
            base_precision=int(info.get("baseAssetPrecision", 8) or 8),
            quote_precision=int(info.get("quotePrecision", 8) or 8),
            # Limits from CCXT
            min_quantity=float(limits.get("amount", {}).get("min", 0) or 0),
            max_quantity=float(limits.get("amount", {}).get("max", 0) or 0),
            min_notional=float(min_notional_filter.get("notional", 0) or 0),
            min_price=float(price_filter.get("minPrice", 0) or 0),
            max_price=float(price_filter.get("maxPrice", 0) or 0),
            tick_size=float(price_filter.get("tickSize", 0) or 0),
            step_size=float(lot_size.get("stepSize", 0) or 0),
            # Margin requirements
            maint_margin_pct=float(info.get("maintMarginPercent", 0) or 0),
            required_margin_pct=float(info.get("requiredMarginPercent", 0) or 0),
            # Risk parameters
            liquidation_fee=float(info.get("liquidationFee", 0) or 0),
            market_take_bound=float(info.get("marketTakeBound", 0) or 0),
            trigger_protect=float(info.get("triggerProtect", 0) or 0),
            # Dates
            onboard_date=onboard_date,
            delivery_date=delivery_date,
            # Raw info
            raw_info=info,
        )


class SymbolInfoPane(VerticalScroll, ThemeColorsMixin):
    """
    Pane displaying detailed symbol information.

    Shows market contract details, trading rules, precision,
    and limits for the selected symbol.
    """

    DEFAULT_CSS = """
    SymbolInfoPane {
        padding: 0 1;
        background: transparent;
    }

    SymbolInfoPane .info-section {
        margin-bottom: 1;
    }

    SymbolInfoPane .section-title {
        text-style: bold;
        color: $text-accent;
        margin-bottom: 0;
    }

    SymbolInfoPane .info-row {
        height: auto;
    }

    SymbolInfoPane .info-label {
        color: $text-muted;
        width: 20;
    }

    SymbolInfoPane .info-value {
        color: $text;
    }

    SymbolInfoPane .status-trading {
        color: $success;
    }

    SymbolInfoPane .status-settling {
        color: $warning;
    }

    SymbolInfoPane .status-inactive {
        color: $error;
    }

    SymbolInfoPane #no-symbol-message {
        color: $text-muted;
        text-style: italic;
        padding: 1;
    }
    """

    market_info: reactive[MarketInfo | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Static(
            "Select a symbol to view details",
            id="no-symbol-message",
        )
        yield Static("", id="symbol-info-content")

    def watch_market_info(self, info: MarketInfo | None) -> None:
        """Update display when market info changes."""
        no_symbol_msg = self.query_one("#no-symbol-message", Static)
        content = self.query_one("#symbol-info-content", Static)

        if info is None:
            no_symbol_msg.display = True
            content.display = False
            return

        no_symbol_msg.display = False
        content.display = True

        # Build the info display
        content.update(self._build_info_display(info))

    @staticmethod
    def _format_decimal(value: float) -> str:
        """Format a decimal value without scientific notation."""
        if value == 0:
            return "0"
        if value >= 1:
            # For values >= 1, use commas for thousands
            if value == int(value):
                return f"{int(value):,}"
            return f"{value:,.8f}".rstrip("0").rstrip(".")
        # For small decimals, format without scientific notation
        # Use enough decimal places to show significant digits
        formatted = f"{value:.10f}".rstrip("0").rstrip(".")
        return formatted

    def _build_info_display(self, info: MarketInfo) -> Text:
        """Build rich text display of market info."""
        text = Text()

        # Get theme colors
        accent = self.accent_color
        success = self.success_color
        warning = self.warning_color
        muted = self.muted_style

        # Header
        text.append(f"{info.base}", style="bold")
        text.append(f"/{info.quote}", style=muted)
        if info.settle and info.settle != info.quote:
            text.append(f":{info.settle}", style=muted)
        text.append("\n\n")

        # Status section
        text.append("Status\n", style=f"bold {accent}")
        status_style = success if info.status == "TRADING" else warning
        text.append(f"  Trading Status   ", style=muted)
        text.append(f"{info.status}\n", style=status_style)
        text.append(f"  Contract Type    ", style=muted)
        text.append(f"{info.contract_type.title()}\n")
        text.append(f"  Margin Asset     ", style=muted)
        text.append(f"{info.margin_mode}\n")
        text.append(f"  Linear           ", style=muted)
        text.append(f"{'Yes' if info.is_linear else 'No (Inverse)'}\n")
        text.append("\n")

        # Precision section
        text.append("Precision\n", style=f"bold {accent}")
        text.append(f"  Price Precision  ", style=muted)
        text.append(f"{info.price_precision} decimals\n")
        text.append(f"  Qty Precision    ", style=muted)
        text.append(f"{info.quantity_precision} decimals\n")
        text.append(f"  Tick Size        ", style=muted)
        text.append(f"{self._format_decimal(info.tick_size)}\n")
        text.append(f"  Step Size        ", style=muted)
        text.append(f"{self._format_decimal(info.step_size)}\n")
        text.append("\n")

        # Limits section
        text.append("Trading Limits\n", style=f"bold {accent}")
        text.append(f"  Min Quantity     ", style=muted)
        text.append(f"{self._format_decimal(info.min_quantity)}\n")
        text.append(f"  Max Quantity     ", style=muted)
        text.append(f"{info.max_quantity:,.0f}\n")
        text.append(f"  Min Notional     ", style=muted)
        text.append(f"{self._format_decimal(info.min_notional)} {info.quote}\n")
        text.append(f"  Min Price        ", style=muted)
        text.append(f"{self._format_decimal(info.min_price)}\n")
        text.append(f"  Max Price        ", style=muted)
        text.append(f"{self._format_decimal(info.max_price)}\n")
        text.append("\n")

        # Margin section
        text.append("Margin Requirements\n", style=f"bold {accent}")
        text.append(f"  Maint Margin     ", style=muted)
        text.append(f"{info.maint_margin_pct}%\n")
        text.append(f"  Required Margin  ", style=muted)
        text.append(f"{info.required_margin_pct}%\n")
        text.append("\n")

        # Risk parameters section
        text.append("Risk Parameters\n", style=f"bold {accent}")
        text.append(f"  Liquidation Fee  ", style=muted)
        text.append(f"{info.liquidation_fee * 100:.2f}%\n")
        text.append(f"  Market Take Bound", style=muted)
        text.append(f" {info.market_take_bound * 100:.1f}%\n")
        text.append(f"  Trigger Protect  ", style=muted)
        text.append(f"{info.trigger_protect * 100:.2f}%\n")
        text.append("\n")

        # Dates section
        text.append("Dates\n", style=f"bold {accent}")
        if info.onboard_date:
            text.append(f"  Listed           ", style=muted)
            text.append(f"{info.onboard_date.strftime('%Y-%m-%d')}\n")
        if info.delivery_date and info.contract_type != "perpetual":
            text.append(f"  Delivery         ", style=muted)
            text.append(f"{info.delivery_date.strftime('%Y-%m-%d')}\n")

        return text

    def update_from_symbol(
        self,
        symbol_data: SymbolData,
        market: dict[str, Any] | None,
    ) -> None:
        """
        Update the pane with symbol and market data.

        Args:
            symbol_data: Basic symbol data from the browser.
            market: Full CCXT market dict, if available.
        """
        if market:
            self.market_info = MarketInfo.from_ccxt_market(market)
        else:
            # Create minimal info from symbol data only
            self.market_info = MarketInfo(
                symbol=symbol_data.symbol,
                base=symbol_data.base,
                quote=symbol_data.quote,
            )

    def clear(self) -> None:
        """Clear the displayed info."""
        self.market_info = None
