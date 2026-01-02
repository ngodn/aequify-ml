"""
APEX Utilities - Config loading and helper functions.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import APEXConfig


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (values override base)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_apex_config(config_path: str) -> APEXConfig:
    """
    Load APEXConfig from a YAML config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        APEXConfig instance

    Example:
        config = load_apex_config("config/config.yaml")
    """
    import yaml

    from .config import APEXConfig

    with open(config_path) as f:
        data = yaml.safe_load(f)

    apex_data = data.get("engines", {}).get("apex", {})
    return APEXConfig.from_dict(apex_data)


def load_apex_config_for_symbol(
    config_path: str,
    symbol: str,
    overrides_dir: str | None = None,
) -> APEXConfig:
    """
    Load APEXConfig with per-symbol overrides.

    Loads the base config from config_path, then looks for a symbol-specific
    override file in the overrides directory. Override values are deep-merged
    with the base config.

    Args:
        config_path: Path to the main YAML config file
        symbol: Symbol name (e.g., "PIPPIN/USDT:USDT" or "PIPPINUSDT")
        overrides_dir: Path to overrides directory. If None, uses
                       config_path's parent / "overrides"

    Returns:
        APEXConfig instance with overrides applied

    Example:
        # Loads config/config.yaml, then merges config/overrides/PIPPINUSDT.yaml
        config = load_apex_config_for_symbol("config/config.yaml", "PIPPIN/USDT:USDT")
    """
    import yaml

    from .config import APEXConfig

    # Load base config
    with open(config_path) as f:
        base_data = yaml.safe_load(f)

    apex_data = base_data.get("engines", {}).get("apex", {})

    # Determine overrides directory
    if overrides_dir is None:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        overrides_dir = os.path.join(config_dir, "overrides")

    # Normalize symbol name for filename (remove slashes and colons)
    # "PIPPIN/USDT:USDT" -> "PIPPINUSDT"
    symbol_normalized = symbol.replace("/", "").replace(":", "").replace("USDT", "")
    symbol_normalized = f"{symbol_normalized}USDT"

    # Look for override file
    override_path = os.path.join(overrides_dir, f"{symbol_normalized}.yaml")

    if os.path.exists(override_path):
        with open(override_path) as f:
            override_data = yaml.safe_load(f)

        if override_data:
            override_apex = override_data.get("engines", {}).get("apex", {})
            if override_apex:
                apex_data = deep_merge(apex_data, override_apex)

    return APEXConfig.from_dict(apex_data)
