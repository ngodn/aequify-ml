"""Aequify TUI Application with PubSub updates and theming."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Label, Static

from aequify import __version__
from aequify.core.pubsub import Subscriber
from aequify.tui.themes import BUILTIN_THEMES
from aequify.tui.widgets.gpu_info import GPUInfoModal
from aequify.tui.widgets.response import ConsoleArea
from aequify.tui.widgets.search_bar import SearchBar
from aequify.tui.widgets.symbols import SelectedSymbolArea, SymbolBrowser, SymbolData


# ============================================================================
# Mock Engine class for TUI development - TODO: Replace with real engine import
# ============================================================================

DEFAULT_PUBSUB_PORT = 5555


class Engine:
    """Mock Engine class for TUI development."""

    def __init__(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False


_engine: Engine | None = None


def get_engine() -> Engine:
    """Get global engine instance."""
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


# ============================================================================
# End mock Engine
# ============================================================================

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
    AppHeader #app-user-host {
        width: auto;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label(f"[b]aequify[/] [dim]{__version__}[/]", id="app-title")
        import getpass
        username = getpass.getuser()
        hostname = platform.node()
        yield Label(f"{username}@{hostname}", id="app-user-host")


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
        pubsub_host: str = "127.0.0.1",
        pubsub_port: int = DEFAULT_PUBSUB_PORT,
        theme: str = DEFAULT_THEME,
        gpu_info: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine or get_engine()
        self.pubsub_host = pubsub_host
        self.pubsub_port = pubsub_port
        self._initial_theme = theme
        self._subscriber: Subscriber | None = None
        self._current_theme_index = THEME_NAMES.index(theme) if theme in THEME_NAMES else 0
        self.gpu_info = gpu_info  # GPU info from Mojo

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

        # Load mock symbol data for development
        self._load_mock_symbols()

        # Start subscriber in a worker thread
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

        self._subscriber = Subscriber(
            host=self.pubsub_host,
            port=self.pubsub_port,
            topics=["engine.state", "engine.error", "system.stats"],
            auto_reconnect=True,
        )

        try:
            while True:
                try:
                    # Try to connect (may fail if engine not started)
                    if not self._subscriber.is_connected:
                        try:
                            self._subscriber.connect()
                        except Exception:
                            # Update status based on engine state
                            self._update_engine_status()
                            await asyncio.sleep(1.0)
                            continue

                    # Check for messages (non-blocking with short timeout)
                    msg = self._subscriber.receive(timeout=0.1)
                    if msg:
                        if msg.topic == "engine.state":
                            self.post_message(EngineStateUpdate(msg.data))
                        elif msg.topic == "system.stats":
                            self.post_message(SystemStatsUpdate(msg.data))

                    # Yield to allow cancellation
                    await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    raise  # Re-raise to exit the loop
                except Exception:
                    # Disconnected - check if engine stopped or just connection lost
                    if self._subscriber:
                        try:
                            self._subscriber.disconnect()
                        except Exception:
                            pass

                    self._update_engine_status()
                    await asyncio.sleep(0.5)
        finally:
            # Cleanup on cancellation
            if self._subscriber:
                try:
                    self._subscriber.disconnect()
                except Exception:
                    pass

    def _update_engine_status(self) -> None:
        """Update engine status in API info bar."""
        try:
            api_info = self.query_one("#api-info", APIInfo)
            if self.engine.is_running:
                api_info.engine_status = "running"
            else:
                api_info.engine_status = "stopped"
        except Exception:
            pass

    def on_engine_state_update(self, message: EngineStateUpdate) -> None:
        """Handle engine state update from PubSub."""
        try:
            api_info = self.query_one("#api-info", APIInfo)
            api_info.engine_status = message.data.get("status", "unknown")
        except Exception:
            pass

    def on_system_stats_update(self, message: SystemStatsUpdate) -> None:
        """Handle system stats update from PubSub."""
        try:
            system_info = self.query_one(SystemInfo)
            system_info.update_stats(message.data)
        except Exception:
            pass

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
        # Subscriber cleanup is handled by the worker's finally block
        # Stop engine in a thread to avoid blocking
        if self.engine.is_running:
            import threading

            threading.Thread(target=self.engine.stop, daemon=True).start()


def run_tui(
    engine: Engine | None = None,
    pubsub_host: str = "127.0.0.1",
    pubsub_port: int = DEFAULT_PUBSUB_PORT,
    theme: str = DEFAULT_THEME,
    gpu_info: dict[str, Any] | None = None,
) -> None:
    """
    Run the TUI application.

    Args:
        engine: Engine instance to control. Uses global if not provided.
        pubsub_host: PubSub server host to connect to.
        pubsub_port: PubSub server port to connect to.
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
