"""Aequify TUI - Terminal User Interface."""

from .app import (
    Aequify,
    APIInfo,
    AppBody,
    AppHeader,
    DEFAULT_THEME,
    SystemInfo,
    THEME_NAMES,
    run_tui,
)
from .themes import BUILTIN_THEMES, Theme
from .widgets import SearchBar, ThemeColorsMixin

__all__ = [
    # App
    "Aequify",
    "APIInfo",
    "AppBody",
    "AppHeader",
    "SystemInfo",
    "run_tui",
    # Themes
    "BUILTIN_THEMES",
    "DEFAULT_THEME",
    "THEME_NAMES",
    "Theme",
    # Widgets
    "SearchBar",
    "ThemeColorsMixin",
]
