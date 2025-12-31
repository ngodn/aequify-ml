"""
Aequify Logging - Async file logging with rotation.

Provides non-blocking logging using a background thread queue handler.
File rotation: 1MB max per file, 100 files max.

Configuration is read from config/config.yaml:
    developer:
      log_level: DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

Usage:
    from aequify.logging import setup_logging, get_logger

    setup_logging()  # Call once at startup, reads from config
    logger = get_logger(__name__)
    logger.info("Hello world")

For TUI applications (to avoid console interference):
    setup_logging(console_output=False)  # File only, no console output
    # Later, after TUI is ready:
    setup_tui_logging(console_getter)     # Route logs to TUI ConsoleArea
"""

from __future__ import annotations

import atexit
import logging
import queue
import sys
import threading
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable

    from aequify.tui.widgets.response.response_area import ConsoleArea

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "aequify.log"
DEFAULT_MAX_BYTES = 1 * 1024 * 1024  # 1MB
DEFAULT_BACKUP_COUNT = 100
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# =============================================================================
# Global State
# =============================================================================

_log_queue: queue.Queue[logging.LogRecord] | None = None
_queue_listener: QueueListener | None = None
_is_setup = False
_setup_lock = threading.Lock()


# =============================================================================
# Config Loading
# =============================================================================

# Cached log levels from config
_config_log_levels: dict[str, int] | None = None


def _load_log_levels_from_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, int]:
    """
    Load log levels from config file.

    Config format:
        developer:
          log_level:
            file: DEBUG
            tui: INFO

    Returns:
        Dict with 'file' and 'tui' log levels.
    """
    global _config_log_levels
    if _config_log_levels is not None:
        return _config_log_levels

    defaults = {"file": logging.DEBUG, "tui": logging.INFO}

    try:
        import yaml

        if not config_path.exists():
            _config_log_levels = defaults
            return defaults

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            _config_log_levels = defaults
            return defaults

        # Get developer.log_level from config
        developer = config.get("developer", {})
        log_level = developer.get("log_level", {})

        # Handle both old format (string) and new format (dict)
        if isinstance(log_level, str):
            # Old format: log_level: DEBUG
            level = LOG_LEVEL_MAP.get(log_level.upper(), logging.DEBUG)
            _config_log_levels = {"file": level, "tui": level}
        else:
            # New format: log_level: {file: DEBUG, tui: INFO}
            file_str = log_level.get("file", "DEBUG")
            tui_str = log_level.get("tui", "INFO")
            _config_log_levels = {
                "file": LOG_LEVEL_MAP.get(str(file_str).upper(), logging.DEBUG),
                "tui": LOG_LEVEL_MAP.get(str(tui_str).upper(), logging.INFO),
            }

        return _config_log_levels
    except Exception:
        _config_log_levels = defaults
        return defaults


# =============================================================================
# Formatters
# =============================================================================


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI color codes for console output."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        if color:
            return f"{color}{message}{self.RESET}"
        return message


# =============================================================================
# Setup Functions
# =============================================================================


def setup_logging(
    log_dir: Path | str = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    level: int | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console_output: bool = True,
    console_level: int | None = None,
) -> None:
    """
    Setup async logging with file rotation and optional console output.

    Logging is handled by a background thread to avoid blocking.
    Log level is read from config/config.yaml (developer.log_level) if not specified.

    Args:
        log_dir: Directory for log files.
        log_file: Log file name.
        level: Minimum log level for file handler (reads from config if None).
        max_bytes: Maximum bytes per log file (default 1MB).
        backup_count: Number of backup files to keep (default 100).
        console_output: Whether to also log to console.
        console_level: Minimum log level for console (defaults to level if None).
    """
    # Load level from config if not specified
    if level is None:
        level = _load_log_levels_from_config()["file"]
    if console_level is None:
        console_level = level
    global _log_queue, _queue_listener, _is_setup

    with _setup_lock:
        if _is_setup:
            return

        # Ensure log directory exists
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create the log queue for async handling
        _log_queue = queue.Queue(-1)  # Unlimited size

        # Create handlers
        handlers: list[logging.Handler] = []

        # File handler with rotation
        file_path = log_path / log_file
        file_handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        handlers.append(file_handler)

        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(
                ColoredFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
            )
            handlers.append(console_handler)

        # Create queue listener (runs in background thread)
        _queue_listener = QueueListener(
            _log_queue,
            *handlers,
            respect_handler_level=True,
        )
        _queue_listener.start()

        # Configure root logger to use queue handler
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Accept all, handlers filter

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add queue handler (sends to background thread)
        queue_handler = QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)

        # Register cleanup at exit
        atexit.register(shutdown_logging)

        _is_setup = True

        # Log startup
        logger = get_logger("aequify.logging")
        level_name = logging.getLevelName(level)
        logger.info(f"Logging initialized: {file_path} (level={level_name})")
        logger.debug(f"Max file size: {max_bytes / 1024 / 1024:.1f}MB, backups: {backup_count}")


def shutdown_logging() -> None:
    """Shutdown the logging system gracefully."""
    global _queue_listener, _is_setup

    with _setup_lock:
        if _queue_listener is not None:
            _queue_listener.stop()
            _queue_listener = None
        _is_setup = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def set_level(level: int) -> None:
    """
    Set the root logger level.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logging.getLogger().setLevel(level)


# =============================================================================
# Print Redirect (Optional)
# =============================================================================


class LoggingWriter:
    """Redirect stdout/stderr to logger."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if message and message.strip():
            self._logger.log(self._level, message.rstrip())
        return len(message)

    def flush(self) -> None:
        pass


def redirect_prints_to_logger(
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.WARNING,
) -> tuple[TextIO, TextIO]:
    """
    Redirect print statements to logger.

    Args:
        stdout_level: Log level for stdout.
        stderr_level: Log level for stderr.

    Returns:
        Tuple of original (stdout, stderr) for restoration.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    stdout_logger = get_logger("aequify.stdout")
    stderr_logger = get_logger("aequify.stderr")

    sys.stdout = LoggingWriter(stdout_logger, stdout_level)  # type: ignore
    sys.stderr = LoggingWriter(stderr_logger, stderr_level)  # type: ignore

    return original_stdout, original_stderr


def restore_prints(original: tuple[TextIO, TextIO]) -> None:
    """Restore original stdout/stderr."""
    sys.stdout, sys.stderr = original


# =============================================================================
# TUI Logging (Direct Handler)
# =============================================================================

# Global reference to TUI console getter
_tui_console: "Callable[[], ConsoleArea | None] | None" = None


class TUIHandler(logging.Handler):
    """Routes logs to TUI ConsoleArea widget."""

    def emit(self, record: logging.LogRecord) -> None:
        global _tui_console
        if _tui_console is None:
            return

        try:
            console = _tui_console()
            if console is None:
                return

            msg = self.format(record)

            # Route to appropriate console method based on level
            if record.levelno >= logging.ERROR:
                console.error(msg)
            elif record.levelno >= logging.WARNING:
                console.warn(msg)
            elif record.name.startswith("aequify.engine") or record.name.startswith(
                "aequify.executor"
            ):
                console.info(msg)
            else:
                console.system(msg)

        except Exception:
            pass  # Don't crash on logging errors


def setup_tui_logging(console_getter: "Callable[[], ConsoleArea | None]") -> None:
    """
    Setup TUI console logging - call after TUI is ready.

    Args:
        console_getter: Callable that returns the ConsoleArea widget or None.
    """
    global _tui_console, _queue_listener
    _tui_console = console_getter

    if _queue_listener is None:
        return

    # Add TUI handler to QueueListener
    tui_level = _load_log_levels_from_config()["tui"]
    tui_handler = TUIHandler()
    tui_handler.setLevel(tui_level)
    tui_handler.setFormatter(logging.Formatter("%(message)s"))

    # Stop listener, add handler, restart
    _queue_listener.stop()
    _queue_listener.handlers = (*_queue_listener.handlers, tui_handler)
    _queue_listener.start()
