"""Aequify - ML and trading platform."""

from importlib.metadata import version

from aequify.logging import (
    get_logger,
    setup_logging,
    setup_tui_logging,
    shutdown_logging,
)
from aequify.core.engine import Engine, get_engine, start_engine, stop_engine

__version__ = version("aequify")

__all__ = [
    "__version__",
    "get_logger",
    "setup_logging",
    "setup_tui_logging",
    "shutdown_logging",
    "Engine",
    "get_engine",
    "start_engine",
    "stop_engine",
]