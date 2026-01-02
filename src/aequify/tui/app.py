"""Aequify TUI Application with PubSub updates and theming."""

from __future__ import annotations

import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Label

from aequify import __version__
from aequify.logging import get_logger, setup_tui_logging
from aequify.core.engine import Engine, get_engine
from aequify.core.pubsub import Subscriber
from aequify.tui.themes import BUILTIN_THEMES
from aequify.tui.widgets.gpu_info import GPUInfoModal
from aequify.tui.widgets.response import ConsoleArea
from aequify.tui.widgets.search_bar import SearchBar
from aequify.tui.widgets.symbols import SelectedSymbolArea, SymbolBrowser, SymbolData

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass

# Path to CSS file
CSS_PATH = Path(__file__).parent / "aequify.scss"

# Default theme
DEFAULT_THEME = "galaxy"

# Available theme names for cycling
THEME_NAMES = list(BUILTIN_THEMES.keys())


# ============================================================================
# Header Widgets
# ============================================================================


class AppHeader(Horizontal):
    """The header of the app showing title and version."""

    DEFAULT_CSS = """
    AppHeader {
        height: 1;
        padding: 0 1;
    }
    AppHeader #app-title {
        width: 1fr;
    }
    AppHeader #engine-status {
        width: auto;
        padding-right: 2;
    }
    AppHeader #app-user-host {
        width: auto;
        color: $text-muted;
    }
    """

    # Engine status: "disconnected", "connecting", "running", "error", "stopped"
    engine_status: reactive[str] = reactive("disconnected")

    def compose(self) -> ComposeResult:
        yield Label(f"[b]aequify[/] [dim]{__version__}[/]", id="app-title")
        yield Label("[yellow]æ[/]", id="engine-status")
        import getpass
        username = getpass.getuser()
        hostname = platform.node()
        yield Label(f"{username}@{hostname}", id="app-user-host")

    def watch_engine_status(self, value: str) -> None:
        """Update status indicator when engine status changes."""
        try:
            status_label = self.query_one("#engine-status", Label)
            if value == "running":
                status_label.update("[green]æ[/]")
            elif value == "connecting":
                status_label.update("[yellow]æ[/]")
            elif value == "error":
                status_label.update("[red]æ[/]")
            elif value == "stopped":
                status_label.update("[red dim]æ[/]")
            else:
                status_label.update("[yellow dim]æ[/]")
        except Exception:
            pass


class SystemInfo(Horizontal):
    """System information bar displaying CPU, memory, GPU, network, and I/O stats."""

    DEFAULT_CSS = """
    SystemInfo {
        height: 1;
        padding: 0 3;
        color: $text-muted;
    }
    SystemInfo #system-stats {
        width: 100%;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        # Import here to avoid circular imports
        from aequify.core.system import SystemMonitor
        self._monitor = SystemMonitor(collect_interval=2.0)

    def compose(self) -> ComposeResult:
        yield Label("CPU: ... RAM: ... GPU: ... Net: ... I/O: ...", id="system-stats")

    def on_mount(self) -> None:
        """Start polling system stats."""
        self.set_interval(2.0, self._poll_stats)
        self._poll_stats()  # Initial update

    def _poll_stats(self) -> None:
        """Poll system stats and update display."""
        try:
            stats = self._monitor.collect()
            self._update_label(
                stats.cpu_percent,
                stats.mem_used_app,
                stats.mem_available,
                stats.mem_total,
                stats.gpu_available,
                stats.gpu_percent,
                stats.gpu_mem_used,
                stats.gpu_mem_total,
                stats.net_up,
                stats.net_down,
                stats.disk_read,
                stats.disk_write,
            )
        except Exception:
            pass

    def _format_bytes(self, bytes_val: float, suffix: str = "B") -> str:
        """Format bytes to human readable string."""
        for unit in ("", "K", "M", "G", "T"):
            if abs(bytes_val) < 1024.0:
                return f"{bytes_val:.1f}{unit}{suffix}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}P{suffix}"

    def _format_rate(self, bytes_per_sec: float) -> str:
        """Format bytes per second to human readable rate."""
        return self._format_bytes(bytes_per_sec, "/s")

    def _update_label(
        self,
        cpu_percent: float,
        mem_used_app: int,
        mem_available: int,
        mem_total: int,
        gpu_available: bool,
        gpu_percent: float,
        gpu_mem_used: int,
        gpu_mem_total: int,
        net_up: float,
        net_down: float,
        disk_read: float,
        disk_write: float,
    ) -> None:
        """Update the label with current stats."""
        try:
            label = self.query_one("#system-stats", Label)
            cpu_color = "green" if cpu_percent < 70 else ("yellow" if cpu_percent < 90 else "red")

            # Format: CPU: 16.1%  RAM: 709.9MB/47.7GB/62.6GB  GPU: 14% 2.4GB/12.0GB  Net: ↑8.7K/s ↓87.7K/s  I/O: R0.0/s W790.0K/s
            text = f"[dim]CPU:[/] [{cpu_color}]{cpu_percent:.1f}%[/]"

            # RAM: app_used/available/total
            text += f"  [dim]RAM:[/] {self._format_bytes(mem_used_app)}/{self._format_bytes(mem_available)}/{self._format_bytes(mem_total)}"

            # GPU
            if gpu_available and gpu_mem_total > 0:
                gpu_color = "green" if gpu_percent < 70 else ("yellow" if gpu_percent < 90 else "red")
                text += f"  [dim]GPU:[/] [{gpu_color}]{gpu_percent:.0f}%[/] {self._format_bytes(gpu_mem_used)}/{self._format_bytes(gpu_mem_total)}"

            # Network
            text += f"  [dim]Net:[/] ↑{self._format_rate(net_up)} ↓{self._format_rate(net_down)}"

            # Disk I/O
            text += f"  [dim]I/O:[/] R{self._format_rate(disk_read)} W{self._format_rate(disk_write)}"

            label.update(text)
        except Exception:
            pass


class APIInfo(Horizontal):
    """API information bar displaying rate limits and WebSocket status."""

    DEFAULT_CSS = """
    APIInfo {
        height: 1;
        padding: 0 3;
        color: $text-muted;
    }
    APIInfo #ws-status {
        width: auto;
    }
    APIInfo #api-stats {
        width: 1fr;
    }
    APIInfo #binance-uid {
        width: auto;
    }
    """

    # WebSocket stream limits
    WS_TRADE_STREAM_LIMIT = 1024
    WS_POSITION_STREAM_LIMIT = 1

    # Weight limits (IP-based, resets every minute)
    weight_used: reactive[int] = reactive(0)
    weight_limit: reactive[int] = reactive(2400)

    # Order limits
    order_count_10s: reactive[int] = reactive(0)
    order_count_1m: reactive[int] = reactive(0)
    order_limit_10s: reactive[int] = reactive(300)
    order_limit_1m: reactive[int] = reactive(1200)

    # WebSocket connection status
    trade_streams_connected: reactive[int] = reactive(0)
    trade_streams_total: reactive[int] = reactive(0)
    position_stream_connected: reactive[bool] = reactive(False)

    # Binance UID
    binance_uid: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Label("", id="ws-status")
        yield Label("", id="api-stats")
        yield Label("", id="binance-uid")

    def on_mount(self) -> None:
        self._update_ws_status()
        self._update_api_stats()
        self._update_uid()

    def watch_weight_used(self, value: int) -> None:
        self._update_api_stats()

    def watch_order_count_10s(self, value: int) -> None:
        self._update_api_stats()

    def watch_order_count_1m(self, value: int) -> None:
        self._update_api_stats()

    def watch_trade_streams_connected(self, value: int) -> None:
        self._update_ws_status()

    def watch_trade_streams_total(self, value: int) -> None:
        self._update_ws_status()

    def watch_position_stream_connected(self, value: bool) -> None:
        self._update_ws_status()

    def watch_binance_uid(self, value: int) -> None:
        self._update_uid()

    def _update_api_stats(self) -> None:
        """Update the API stats label."""
        try:
            stats_label = self.query_one("#api-stats", Label)

            # Weight color
            weight_pct = (self.weight_used / self.weight_limit) * 100 if self.weight_limit > 0 else 0
            weight_color = "green" if weight_pct < 50 else ("yellow" if weight_pct < 80 else "red")

            # Orders 10s color
            order_10s_pct = (self.order_count_10s / self.order_limit_10s) * 100 if self.order_limit_10s > 0 else 0
            order_10s_color = "green" if order_10s_pct < 50 else ("yellow" if order_10s_pct < 80 else "red")

            # Orders 1m color
            order_1m_pct = (self.order_count_1m / self.order_limit_1m) * 100 if self.order_limit_1m > 0 else 0
            order_1m_color = "green" if order_1m_pct < 50 else ("yellow" if order_1m_pct < 80 else "red")

            stats_label.update(
                f"API: [dim]Weight:[/] [{weight_color}]{self.weight_used}/{self.weight_limit}[/]  "
                f"[dim]Orders:[/] [{order_10s_color}]{self.order_count_10s}/{self.order_limit_10s}[/] [dim](10s)[/]  "
                f"[{order_1m_color}]{self.order_count_1m}/{self.order_limit_1m}[/] [dim](1m)[/]"
            )
        except Exception:
            pass

    def _update_ws_status(self) -> None:
        """Update the WebSocket status label."""
        try:
            ws_label = self.query_one("#ws-status", Label)

            # Trade streams: connected/total/max
            if self.trade_streams_total > 0:
                if self.trade_streams_connected == self.trade_streams_total:
                    trade_color = "green"
                    trade_icon = "●"
                elif self.trade_streams_connected > 0:
                    trade_color = "yellow"
                    trade_icon = "◐"
                else:
                    trade_color = "red"
                    trade_icon = "○"
                trade_str = (
                    f"[{trade_color}]{trade_icon}[/] "
                    f"{self.trade_streams_connected}/{self.trade_streams_total}/{self.WS_TRADE_STREAM_LIMIT}"
                )
            else:
                trade_str = f"[dim]○ 0/0/{self.WS_TRADE_STREAM_LIMIT}[/]"

            # Position stream: connected/total/max
            pos_connected = 1 if self.position_stream_connected else 0
            if self.position_stream_connected:
                pos_str = f"[green]●[/] {pos_connected}/1/{self.WS_POSITION_STREAM_LIMIT}"
            else:
                pos_str = f"[dim]○[/] {pos_connected}/1/{self.WS_POSITION_STREAM_LIMIT}"

            ws_label.update(f"WS: [dim]Trades:[/] {trade_str}  [dim]Pos:[/] {pos_str}")
        except Exception:
            pass

    def _update_uid(self) -> None:
        """Update the UID label."""
        try:
            uid_label = self.query_one("#binance-uid", Label)
            if self.binance_uid > 0:
                uid_label.update(f"[dim]UID:[/] {self.binance_uid}")
            else:
                uid_label.update("")
        except Exception:
            pass


class AppBody(Vertical):
    """The main body of the app containing the symbol browser and content areas."""

    DEFAULT_CSS = """
    AppBody {
        height: 1fr;
    }
    """


# ============================================================================
# Main Application
# ============================================================================


class EngineStateUpdate(Message):
    """Message sent when engine state is updated via PubSub."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class SystemStatsUpdate(Message):
    """Message sent when system stats are updated via PubSub."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class TradingPositionsUpdate(Message):
    """Message sent when positions are updated via PubSub."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class TradingTradeUpdate(Message):
    """Message sent when a trade is received via PubSub."""

    def __init__(self, symbol: str, data: dict[str, Any]) -> None:
        super().__init__()
        self.symbol = symbol
        self.data = data


class TradingSymbolsUpdate(Message):
    """Message sent when symbol list is updated via PubSub."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class TradingRateLimitUpdate(Message):
    """Message sent when rate limit status is updated via PubSub."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class TradingStreamStatusUpdate(Message):
    """Message sent when WebSocket stream status is updated via PubSub."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class TradingMarketsUpdate(Message):
    """Message sent when market data is updated via PubSub."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class TradingApexUpdate(Message):
    """Message sent when APEX bootstrap state is updated via PubSub."""

    def __init__(self, symbol: str, data: dict[str, Any]) -> None:
        super().__init__()
        self.symbol = symbol
        self.data = data


class TradingApexLiveUpdate(Message):
    """Message sent when APEX live metrics are updated via PubSub."""

    def __init__(self, symbol: str, data: dict[str, Any]) -> None:
        super().__init__()
        self.symbol = symbol
        self.data = data


class Aequify(App):
    """Aequify Terminal User Interface with PubSub updates and theming."""

    TITLE = "Aequify"
    CSS_PATH = CSS_PATH
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+f", "focus_search", "Focus Search"),
        # ("s", "toggle_engine", "Start/Stop Engine"),
        # ("T", "cycle_theme_reverse", "Previous Theme"),
        ("ctrl+g", "gpu_info", "GPU Info"),
        ("t", "cycle_theme", "Cycle Theme"),
    ]

    def __init__(
        self,
        engine: Engine | None = None,
        pubsub_host: str | None = None,
        pubsub_port: int | None = None,
        theme: str = DEFAULT_THEME,
        gpu_info: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine or get_engine()
        # Get pubsub address from engine (which reads from config.yaml)
        self.pubsub_host = pubsub_host or self.engine.pubsub_host
        self.pubsub_port = pubsub_port or self.engine.pubsub_port
        self._initial_theme = theme
        self._subscriber: Subscriber | None = None
        self._current_theme_index = THEME_NAMES.index(theme) if theme in THEME_NAMES else 0
        self.gpu_info = gpu_info  # GPU info from Mojo
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aeq-tui")

        # Register all built-in themes
        for theme_name, textual_theme in BUILTIN_THEMES.items():
            self.register_theme(textual_theme)

    def compose(self) -> ComposeResult:
        # Header section
        yield AppHeader()
        yield SystemInfo()
        yield APIInfo(id="api-info")
        yield SearchBar(id="search-bar")

        # Main body
        with AppBody():
            with Horizontal(id="main-content"):
                # Left column: Symbol Browser
                yield SymbolBrowser(id="symbol-browser")

                # Right column: Selected Symbol Area + Console
                with Vertical(id="right-column"):
                    yield SelectedSymbolArea(id="selected-symbol-area")
                    yield ConsoleArea(id="console-area")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize app, set theme, load mock data, and start PubSub subscriber."""
        # Set the initial theme
        self.theme = self._initial_theme

        # Setup TUI logging - route logs to console widget
        def get_console() -> ConsoleArea | None:
            try:
                return self.query_one("#console-area", ConsoleArea)
            except Exception:
                return None

        setup_tui_logging(get_console)

        # Start subscriber in a worker thread (symbols come from filter pipeline via PubSub)
        self.run_worker(self._subscribe_loop, exclusive=True)

    def _load_mock_symbols(self) -> None:
        """Load mock symbol data for TUI development."""
        mock_symbols = [
            SymbolData("BTC/USDT:USDT", "BTC", "USDT", 2.5, True, True, False, 15.3),
            SymbolData("ETH/USDT:USDT", "ETH", "USDT", -1.2, True, False, True, -5.2),
            SymbolData("SOL/USDT:USDT", "SOL", "USDT", 8.7, False),
            SymbolData("DOGE/USDT:USDT", "DOGE", "USDT", -3.4, False),
            SymbolData("XRP/USDT:USDT", "XRP", "USDT", 1.1, False),
            SymbolData("ADA/USDT:USDT", "ADA", "USDT", -0.5, False),
            SymbolData("AVAX/USDT:USDT", "AVAX", "USDT", 4.2, True, True, True, 2.1),
            SymbolData("LINK/USDT:USDT", "LINK", "USDT", 0.8, False),
            SymbolData("DOT/USDT:USDT", "DOT", "USDT", -2.1, False),
            SymbolData("MATIC/USDT:USDT", "MATIC", "USDT", 5.6, False),
        ]

        try:
            symbol_browser = self.query_one("#symbol-browser", SymbolBrowser)
            symbol_browser.update_symbols(mock_symbols)
            symbol_browser.select_first()
        except Exception:
            pass

    async def _subscribe_loop(self) -> None:
        """Background worker that receives PubSub messages."""
        import asyncio

        logger.info(f"Subscribe loop starting, connecting to {self.pubsub_host}:{self.pubsub_port}")

        self._subscriber = Subscriber(
            host=self.pubsub_host,
            port=self.pubsub_port,
            topics=[
                "engine.state",
                "engine.error",
                "system.stats",
                "trading.positions",
                "trading.trade.*",  # Wildcard for all symbol trades
                "trading.symbols",
                "trading.markets",  # Market data for symbol info
                "trading.rate_limit",
                "trading.stream_status",
                "trading.apex.*",  # Bootstrap state
                "trading.apex.live.*",  # Live metrics
            ],
            auto_reconnect=True,
        )

        def update_engine_status(status: str) -> None:
            try:
                header = self.query_one(AppHeader)
                header.engine_status = status
            except Exception as ex:
                logger.debug(f"Failed to update engine status: {ex}")

        try:
            while True:
                try:
                    # Try to connect (may fail if engine not started)
                    if not self._subscriber.is_connected:
                        logger.debug("Not connected, attempting connect...")
                        try:
                            update_engine_status("connecting")
                            # Run blocking connect in thread pool to avoid blocking UI
                            await asyncio.get_running_loop().run_in_executor(
                                self._executor, self._subscriber.connect
                            )
                            logger.info("PubSub connected!")
                            update_engine_status("running")
                        except Exception as e:
                            logger.warning(f"Connect failed: {type(e).__name__}: {e}")
                            update_engine_status("error")
                            await asyncio.sleep(1.0)
                            continue

                    # Check for messages (non-blocking with short timeout)
                    # Run in executor to avoid blocking UI
                    msg = await asyncio.get_running_loop().run_in_executor(
                        self._executor, lambda: self._subscriber.receive(timeout=0.1)
                    )
                    if msg:
                        # logger.debug(f"Got message: {msg.topic}")
                        if msg.topic == "engine.state":
                            self.post_message(EngineStateUpdate(msg.data))
                        elif msg.topic == "system.stats":
                            self.post_message(SystemStatsUpdate(msg.data))
                        elif msg.topic == "trading.positions":
                            self.post_message(TradingPositionsUpdate(msg.data))
                        elif msg.topic.startswith("trading.trade."):
                            # Extract symbol from topic (trading.trade.<symbol>)
                            symbol = msg.data.get("symbol", "")
                            self.post_message(TradingTradeUpdate(symbol, msg.data))
                        elif msg.topic == "trading.symbols":
                            self.post_message(TradingSymbolsUpdate(msg.data))
                        elif msg.topic == "trading.markets":
                            self.post_message(TradingMarketsUpdate(msg.data))
                        elif msg.topic == "trading.rate_limit":
                            self.post_message(TradingRateLimitUpdate(msg.data))
                        elif msg.topic == "trading.stream_status":
                            self.post_message(TradingStreamStatusUpdate(msg.data))
                        elif msg.topic.startswith("trading.apex.live."):
                            # Live metrics: trading.apex.live.<symbol>
                            symbol = msg.data.get("symbol", "")
                            self.post_message(TradingApexLiveUpdate(symbol, msg.data))
                        elif msg.topic.startswith("trading.apex."):
                            # Bootstrap state: trading.apex.<symbol>
                            symbol = msg.data.get("symbol", "")
                            self.post_message(TradingApexUpdate(symbol, msg.data))

                    # Yield to allow cancellation
                    await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    raise  # Re-raise to exit the loop
                except Exception as e:
                    # Log the exception for debugging
                    logger.error(f"Subscribe loop error: {type(e).__name__}: {e}")
                    # Disconnected - check if engine stopped or just connection lost
                    if self._subscriber:
                        try:
                            self._subscriber.disconnect()
                        except Exception:
                            pass

                    update_engine_status("disconnected")
                    await asyncio.sleep(0.5)
        finally:
            # Cleanup on cancellation
            if self._subscriber:
                try:
                    self._subscriber.disconnect()
                except Exception:
                    pass

    def on_engine_state_update(self, message: EngineStateUpdate) -> None:
        """Handle engine state update from PubSub."""
        try:
            # Update engine status in header based on engine state
            header = self.query_one(AppHeader)
            status = message.data.get("status", "unknown")
            if status == "running":
                header.engine_status = "running"
            elif status == "error":
                header.engine_status = "error"
            elif status in ("stopped", "stopping"):
                header.engine_status = "stopped"
            elif status in ("starting",):
                header.engine_status = "connecting"
            # Keep current status for unknown states
        except Exception:
            pass

    def on_system_stats_update(self, message: SystemStatsUpdate) -> None:
        """Handle system stats update from PubSub."""
        try:
            system_info = self.query_one(SystemInfo)
            system_info.update_stats(message.data)
        except Exception:
            pass

    def on_trading_positions_update(self, message: TradingPositionsUpdate) -> None:
        """Handle positions update from PubSub."""
        try:
            positions = message.data.get("positions", [])
            selected_area = self.query_one("#selected-symbol-area", SelectedSymbolArea)
            selected_area.update_positions(positions)

            # Update symbol browser with position info
            symbol_browser = self.query_one("#symbol-browser", SymbolBrowser)
            for pos in positions:
                symbol = pos.get("symbol", "")
                contracts = float(pos.get("contracts", 0) or 0)
                if symbol and contracts != 0:
                    side = (pos.get("side") or "").lower()
                    pnl = float(pos.get("percentage", 0) or 0)
                    symbol_browser.update_symbol_position(
                        symbol,
                        has_position=True,
                        has_long=(side == "long"),
                        has_short=(side == "short"),
                        pnl=pnl,
                    )
        except Exception as e:
            logger.debug(f"Error handling positions update: {e}")

    def on_trading_trade_update(self, message: TradingTradeUpdate) -> None:
        """Handle trade update from PubSub."""
        try:
            selected_area = self.query_one("#selected-symbol-area", SelectedSymbolArea)
            trade_pane = selected_area.trade_stream_pane

            # Handle single trade
            trade = message.data.get("trade")
            if trade:
                # Convert to format expected by TradeStreamPane
                trade_data = {
                    "trade_id": int(trade.get("id", 0)),
                    "symbol": message.symbol,
                    "price": float(trade.get("price", 0)),
                    "quantity": float(trade.get("amount", 0)),
                    "timestamp_ms": int(trade.get("timestamp", 0)),
                    "is_buyer_maker": trade.get("side") == "sell",
                }
                trade_pane.add_trade(trade_data)
                trade_pane.set_streaming(True)

            # Handle batch of trades
            trades = message.data.get("trades", [])
            for trade in trades:
                trade_data = {
                    "trade_id": int(trade.get("id", 0)),
                    "symbol": message.symbol,
                    "price": float(trade.get("price", 0)),
                    "quantity": float(trade.get("amount", 0)),
                    "timestamp_ms": int(trade.get("timestamp", 0)),
                    "is_buyer_maker": trade.get("side") == "sell",
                }
                trade_pane.add_trade(trade_data)
            if trades:
                trade_pane.set_streaming(True)
        except Exception as e:
            logger.debug(f"Error handling trade update: {e}")

    def on_trading_symbols_update(self, message: TradingSymbolsUpdate) -> None:
        """Handle symbols update from PubSub."""
        try:
            symbols_data = message.data.get("symbols", [])

            # Convert to SymbolData format
            symbols = []
            for sym_data in symbols_data:
                symbol = sym_data.get("symbol", "")
                if not symbol:
                    continue

                base = symbol.split("/")[0] if "/" in symbol else symbol
                symbols.append(
                    SymbolData(
                        symbol=symbol,
                        base=base,
                        quote="USDT",
                        price_change_pct=float(sym_data.get("price_change_pct", 0)),
                        has_position=sym_data.get("has_position", False),
                        has_long=sym_data.get("has_long", False),
                        has_short=sym_data.get("has_short", False),
                        position_pnl=float(sym_data.get("position_pnl", 0)),
                    )
                )

            if symbols:
                symbol_browser = self.query_one("#symbol-browser", SymbolBrowser)
                symbol_browser.update_symbols(symbols)
        except Exception as e:
            logger.debug(f"Error handling symbols update: {e}")

    def on_trading_markets_update(self, message: TradingMarketsUpdate) -> None:
        """Handle market data update from PubSub."""
        try:
            markets = message.data.get("markets", {})
            if not markets:
                return

            # Update selected symbol area's market cache
            selected_area = self.query_one("#selected-symbol-area", SelectedSymbolArea)
            selected_area.update_market_data(markets)
            logger.debug(f"Updated market data cache with {len(markets)} symbols")
        except Exception as e:
            logger.debug(f"Error handling markets update: {e}")

    def on_trading_rate_limit_update(self, message: TradingRateLimitUpdate) -> None:
        """Handle rate limit update from PubSub."""
        try:
            api_info = self.query_one("#api-info", APIInfo)
            api_info.weight_used = message.data.get("weight_used", 0)
            api_info.weight_limit = message.data.get("weight_limit", 2400)
            api_info.order_count_10s = message.data.get("order_count_10s", 0)
            api_info.order_limit_10s = message.data.get("order_limit_10s", 300)
            api_info.order_count_1m = message.data.get("order_count_1m", 0)
            api_info.order_limit_1m = message.data.get("order_limit_1m", 1200)
        except Exception as e:
            logger.debug(f"Error handling rate limit update: {e}")

    def on_trading_stream_status_update(self, message: TradingStreamStatusUpdate) -> None:
        """Handle WebSocket stream status update from PubSub."""
        try:
            # Update API info header
            api_info = self.query_one("#api-info", APIInfo)
            api_info.trade_streams_connected = message.data.get("trade_streams_connected", 0)
            api_info.trade_streams_total = message.data.get("trade_streams_total", 0)
            api_info.position_stream_connected = message.data.get("position_stream_connected", False)
            api_info.binance_uid = message.data.get("binance_uid", 0)

            # Update selected symbol area panes with streaming status
            selected_area = self.query_one("#selected-symbol-area", SelectedSymbolArea)
            position_connected = message.data.get("position_stream_connected", False)
            selected_area.positions_pane.set_streaming(position_connected)

            # Trade stream status (at least 1 connected = streaming)
            trade_connected = message.data.get("trade_streams_connected", 0) > 0
            selected_area.trade_stream_pane.set_streaming(trade_connected)
        except Exception as e:
            logger.debug(f"Error handling stream status update: {e}")

    def on_trading_apex_update(self, message: TradingApexUpdate) -> None:
        """Handle APEX bootstrap state update from PubSub."""
        try:
            from aequify.tui.widgets.symbols.apex_chart_pane import APEXTUIState, OptimizedParams

            selected_area = self.query_one("#selected-symbol-area", SelectedSymbolArea)
            apex_pane = selected_area.apex_chart_pane

            # Only update if this symbol is currently selected
            if apex_pane._selected_symbol != message.symbol:
                return

            state_data = message.data.get("state", {})
            if not state_data:
                return

            # Convert state dict to APEXTUIState
            long_params = None
            short_params = None

            if state_data.get("long_params"):
                lp = state_data["long_params"]
                long_params = OptimizedParams(
                    price_move=lp.get("price_move", -2.5),
                    time_window=lp.get("time_window", 30000),
                    delta_threshold=lp.get("delta_threshold", -50.0),
                    dca_distance_pct=lp.get("dca_distance_pct", -3.0),
                    target_profit=lp.get("target_profit", 1.5),
                    stop_loss=lp.get("stop_loss", 5.0),
                    max_hold_time_ms=lp.get("max_hold_time_ms", 3600000),
                    win_rate=lp.get("win_rate", 0.0),
                    entries=lp.get("entries", 0),
                    avg_pnl=lp.get("avg_pnl", 0.0),
                )

            if state_data.get("short_params"):
                sp = state_data["short_params"]
                short_params = OptimizedParams(
                    price_move=sp.get("price_move", 2.5),
                    time_window=sp.get("time_window", 30000),
                    delta_threshold=sp.get("delta_threshold", 50.0),
                    dca_distance_pct=sp.get("dca_distance_pct", 3.0),
                    target_profit=sp.get("target_profit", 1.5),
                    stop_loss=sp.get("stop_loss", 5.0),
                    max_hold_time_ms=sp.get("max_hold_time_ms", 3600000),
                    win_rate=sp.get("win_rate", 0.0),
                    entries=sp.get("entries", 0),
                    avg_pnl=sp.get("avg_pnl", 0.0),
                )

            tui_state = APEXTUIState(
                is_bootstrapped=state_data.get("is_bootstrapped", False),
                source=state_data.get("source", ""),
                long_params=long_params,
                short_params=short_params,
                backfill_status=state_data.get("backfill_status", ""),
                backfill_progress=state_data.get("backfill_progress", 0.0),
                backfill_current_day=state_data.get("backfill_current_day", 0),
                backfill_total_days=state_data.get("backfill_total_days", 0),
                current_price=state_data.get("current_price", 0.0),
                trade_count=state_data.get("trade_count", 0),
            )

            apex_pane.update_state(tui_state)
        except Exception as e:
            logger.debug(f"Error handling APEX update: {e}")

    def on_trading_apex_live_update(self, message: TradingApexLiveUpdate) -> None:
        """Handle APEX live metrics update from PubSub."""
        try:
            from aequify.tui.widgets.symbols.apex_chart_pane import APEXTUIState, OptimizedParams

            selected_area = self.query_one("#selected-symbol-area", SelectedSymbolArea)
            apex_pane = selected_area.apex_chart_pane

            # Only update if this symbol is currently selected
            if apex_pane._selected_symbol != message.symbol:
                return

            metrics = message.data.get("metrics", {})
            if not metrics:
                return

            # Get current state or create new one
            current_state = apex_pane._state
            if current_state is None:
                current_state = APEXTUIState()

            # Update live metrics while preserving bootstrap params
            long_params = current_state.long_params
            short_params = current_state.short_params

            # Update params if provided in metrics (for thresholds)
            if metrics.get("long_params"):
                lp = metrics["long_params"]
                if long_params:
                    long_params.price_move = lp.get("price_move", long_params.price_move)
                    long_params.delta_threshold = lp.get("delta_threshold", long_params.delta_threshold)

            if metrics.get("short_params"):
                sp = metrics["short_params"]
                if short_params:
                    short_params.price_move = sp.get("price_move", short_params.price_move)
                    short_params.delta_threshold = sp.get("delta_threshold", short_params.delta_threshold)

            # Create updated state with live metrics
            updated_state = APEXTUIState(
                is_bootstrapped=metrics.get("is_bootstrapped", current_state.is_bootstrapped),
                source=current_state.source,
                long_params=long_params,
                short_params=short_params,
                # Price data
                current_price=metrics.get("current_price", current_state.current_price),
                rolling_high=metrics.get("rolling_high", current_state.rolling_high),
                rolling_low=metrics.get("rolling_low", current_state.rolling_low),
                price_move_from_high=metrics.get("price_move_from_high", current_state.price_move_from_high),
                price_move_from_low=metrics.get("price_move_from_low", current_state.price_move_from_low),
                price_window=metrics.get("price_window", current_state.price_window),
                # Volume delta
                long_volume_delta=metrics.get("long_volume_delta", current_state.long_volume_delta),
                short_volume_delta=metrics.get("short_volume_delta", current_state.short_volume_delta),
                imbalance_price_pct=metrics.get("imbalance_price_pct", current_state.imbalance_price_pct),
                # Signal state
                long_signal_ready=metrics.get("long_signal_ready", current_state.long_signal_ready),
                short_signal_ready=metrics.get("short_signal_ready", current_state.short_signal_ready),
                long_signal_offset=metrics.get("long_signal_offset", current_state.long_signal_offset),
                short_signal_offset=metrics.get("short_signal_offset", current_state.short_signal_offset),
                # Trade count
                trade_count=metrics.get("trade_count", current_state.trade_count),
            )

            apex_pane.update_state(updated_state)
            apex_pane.set_active(True)
        except Exception as e:
            logger.debug(f"Error handling APEX live update: {e}")

    def action_toggle_engine(self) -> None:
        """Toggle engine start/stop (runs in worker to avoid blocking TUI)."""
        if self.engine.is_running:
            self.run_worker(self._stop_engine, exclusive=False)
        else:
            self.run_worker(self._start_engine, exclusive=False)

    async def _start_engine(self) -> None:
        """Start engine in background worker."""
        try:
            api_info = self.query_one("#api-info", APIInfo)
            api_info.engine_status = "starting"
        except Exception:
            pass
        self.engine.start()
        self._update_engine_status()

    async def _stop_engine(self) -> None:
        """Stop engine in background worker."""
        try:
            api_info = self.query_one("#api-info", APIInfo)
            api_info.engine_status = "stopping"
        except Exception:
            pass
        self.engine.stop()
        self._update_engine_status()

    def action_cycle_theme(self) -> None:
        """Cycle to the next theme."""
        self._current_theme_index = (self._current_theme_index + 1) % len(THEME_NAMES)
        self.theme = THEME_NAMES[self._current_theme_index]
        self.notify(f"Theme: {self.theme}", timeout=1.5)

    def action_cycle_theme_reverse(self) -> None:
        """Cycle to the previous theme."""
        self._current_theme_index = (self._current_theme_index - 1) % len(THEME_NAMES)
        self.theme = THEME_NAMES[self._current_theme_index]
        self.notify(f"Theme: {self.theme}", timeout=1.5)

    def action_focus_search(self) -> None:
        """Focus the search bar."""
        try:
            search_bar = self.query_one("#search-bar", SearchBar)
            search_bar.focus_input()
        except Exception:
            pass

    def action_gpu_info(self) -> None:
        """Show GPU information modal."""
        self.push_screen(GPUInfoModal(self.gpu_info))

    def on_symbol_tree_symbol_selected(self, event: Any) -> None:
        """Handle symbol selection from browser."""
        try:
            selected_area = self.query_one("#selected-symbol-area", SelectedSymbolArea)
            selected_area.selected_symbol = event.symbol
        except Exception:
            pass

    def on_unmount(self) -> None:
        """Cleanup when app closes."""
        # Shutdown thread pool executor
        self._executor.shutdown(wait=False)

        # Subscriber cleanup is handled by the worker's finally block
        # Stop engine in a thread to avoid blocking
        if self.engine.is_running:
            import threading

            threading.Thread(target=self.engine.stop, daemon=True).start()


def run_tui(
    engine: Engine | None = None,
    pubsub_host: str | None = None,
    pubsub_port: int | None = None,
    theme: str = DEFAULT_THEME,
    gpu_info: dict[str, Any] | None = None,
) -> None:
    """
    Run the TUI application.

    Args:
        engine: Engine instance to control. Uses global if not provided.
        pubsub_host: PubSub server host (default: from engine/config.yaml).
        pubsub_port: PubSub server port (default: from engine/config.yaml).
        theme: Initial theme name (default: "galaxy").
        gpu_info: GPU information dict from Mojo (optional).
    """
    app = Aequify(
        engine=engine,
        pubsub_host=pubsub_host,
        pubsub_port=pubsub_port,
        theme=theme,
        gpu_info=gpu_info,
    )
    app.run()
