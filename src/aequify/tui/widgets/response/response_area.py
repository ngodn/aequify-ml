from datetime import datetime
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import RichLog, Static

from aequify.tui.widgets.theme_colors import ThemeColorsMixin


class ConsoleLog(RichLog):
    """Console log with vim-style keybindings."""

    BINDINGS = [
        Binding("j", "scroll_down", "Scroll down", show=False),
        Binding("k", "scroll_up", "Scroll up", show=False),
        Binding("h", "scroll_left", "Scroll left", show=False),
        Binding("l", "scroll_right", "Scroll right", show=False),
        Binding("g", "scroll_home", "Scroll to top", show=False),
        Binding("G", "scroll_end", "Scroll to bottom", show=False),
    ]


class ConsoleTitleBar(Static):
    """Clickable title bar for the console."""

    DEFAULT_CSS = """
    ConsoleTitleBar {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    ConsoleTitleBar:hover {
        background: $surface-lighten-1;
        color: $text;
    }
    """

    def __init__(self, collapsed: bool = True) -> None:
        super().__init__()
        self._collapsed = collapsed
        self._update_text()

    def _update_text(self) -> None:
        symbol = "▶" if self._collapsed else "▼"
        action = "expand" if self._collapsed else "collapse"
        self.update(f"{symbol} Console (click to {action})")

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._update_text()

    def on_click(self) -> None:
        """Notify parent to toggle collapsed state."""
        self.post_message(ConsoleArea.ToggleCollapsed())


LogLevel = Literal["info", "warn", "error", "debug", "trade", "system"]


class ConsoleArea(Vertical, ThemeColorsMixin):
    """
    Console logging area for the trading application.

    Features a collapsible container:
    - Collapsed (default): Shows only ~3 lines of recent logs
    - Expanded: Shows more logs (12 lines)

    Click the title bar to toggle between states.
    """

    class ToggleCollapsed(Message):
        """Message to toggle collapsed state."""

        pass

    # Heights
    COLLAPSED_HEIGHT = 4  # title bar (1) + 3 lines of logs
    EXPANDED_HEIGHT = 13  # title bar (1) + 12 lines of logs

    collapsed: reactive[bool] = reactive(True)

    DEFAULT_CSS = """
    ConsoleArea {
        height: auto;
    }

    ConsoleArea.-collapsed {
        height: 4;
    }

    ConsoleArea.-expanded {
        height: 13;
    }

    ConsoleArea ConsoleLog {
        scrollbar-gutter: stable;
        height: 1fr;
    }
    """

    def __init__(
        self,
        max_lines: int | None = 1000,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._max_lines = max_lines

    def on_mount(self) -> None:
        self.add_class("section")
        # Set initial collapsed state
        self._apply_collapsed_state()

    def compose(self) -> ComposeResult:
        yield ConsoleTitleBar(collapsed=self.collapsed)
        yield ConsoleLog(
            highlight=True,
            markup=True,
            wrap=True,
            auto_scroll=True,
            max_lines=self._max_lines,
            id="console-log",
        )

    def _apply_collapsed_state(self) -> None:
        """Apply CSS classes based on collapsed state."""
        if self.collapsed:
            self.add_class("-collapsed")
            self.remove_class("-expanded")
        else:
            self.add_class("-expanded")
            self.remove_class("-collapsed")
        # Update title bar
        try:
            title_bar = self.query_one(ConsoleTitleBar)
            title_bar.set_collapsed(self.collapsed)
        except Exception:
            pass

    def watch_collapsed(self, collapsed: bool) -> None:
        """React to collapsed state changes."""
        self._apply_collapsed_state()

    def on_console_area_toggle_collapsed(self, event: ToggleCollapsed) -> None:
        """Handle toggle request from title bar."""
        self.collapsed = not self.collapsed

    @property
    def log_widget(self) -> ConsoleLog:
        """Get the console log widget."""
        return self.query_one("#console-log", ConsoleLog)

    def _get_level_style(self, level: LogLevel) -> str:
        """Get theme-aware style for a log level."""
        level_styles: dict[LogLevel, str] = {
            "info": self.accent_color,
            "warn": self.warning_color,
            "error": f"{self.error_color} bold",
            "debug": self.muted_style,
            "trade": self.success_color,
            "system": self.secondary_color,
        }
        return level_styles.get(level, "white")

    def log(
        self,
        message: str,
        level: LogLevel = "info",
        timestamp: bool = True,
    ) -> None:
        """
        Write a log message to the console.

        Args:
            message: The message to log.
            level: Log level (info, warn, error, debug, trade, system).
            timestamp: Whether to include timestamp.
        """
        style = self._get_level_style(level)
        level_tag = f"[{style}]{level.upper():6}[/]"

        muted = self.muted_style
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            prefix = f"[{muted}]{ts}[/] {level_tag}"
        else:
            prefix = level_tag

        self.log_widget.write(f"{prefix} {message}")

    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, "info")

    def warn(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, "warn")

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, "error")

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, "debug")

    def trade(self, message: str) -> None:
        """Log a trade message."""
        self.log(message, "trade")

    def system(self, message: str) -> None:
        """Log a system message."""
        self.log(message, "system")

    def clear(self) -> None:
        """Clear the console log."""
        self.log_widget.clear()


# Keep ResponseArea as alias for backward compatibility during transition
ResponseArea = ConsoleArea
