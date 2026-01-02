"""
RTDS Configuration Loader.

Loads stream configuration from config/config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aequify.core.rtds.trade_streams import StreamConfig


def load_rtds_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load RTDS configuration from config.yaml.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml.

    Returns:
        Dictionary with RTDS config values.
    """
    if config_path is None:
        # Find project root (where config/ is)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "config" / "config.yaml").exists():
                config_path = parent / "config" / "config.yaml"
                break

    if config_path is None or not Path(config_path).exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config.get("rtds", {})


def load_stream_config(config_path: str | Path | None = None) -> StreamConfig:
    """
    Load StreamConfig from config.yaml.

    Args:
        config_path: Path to config file.

    Returns:
        StreamConfig instance.
    """
    rtds = load_rtds_config(config_path)

    return StreamConfig(
        demo=rtds.get("demo", False),
        max_retries=rtds.get("max_retries", 0),
    )


def load_symbols(config_path: str | Path | None = None) -> list[str]:
    """
    Load symbols to stream from config.yaml.

    Args:
        config_path: Path to config file.

    Returns:
        List of symbols in CCXT format.
    """
    rtds = load_rtds_config(config_path)
    return rtds.get("symbols", [])


def get_rtds_info(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Get RTDS configuration info for display.

    Args:
        config_path: Path to config file.

    Returns:
        Dictionary with config info.
    """
    rtds = load_rtds_config(config_path)

    return {
        "exchange": rtds.get("exchange", "binanceusdm"),
        "demo": rtds.get("demo", False),
        "max_retries": rtds.get("max_retries", 0),
        "symbols": rtds.get("symbols", []),
        "symbol_count": len(rtds.get("symbols", [])),
    }
