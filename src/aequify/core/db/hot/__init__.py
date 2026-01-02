"""
Hot Store Module - In-memory trade storage with flush to cold store.

The hot store is implemented in Mojo for performance (see __init__.mojo).
This Python module provides the flush manager for persisting data.

Mojo Components (imported via compiled module):
    - HotStoreRegistry: Per-symbol trade buffers
    - Trade: Trade data structure
    - WindowIndices: Query result indices

Python Components:
    - FlushManager: Background flush to cold store at canonical time windows
    - FlushConfig: Flush configuration
    - FlushStats: Flush statistics

Usage:
    # The Mojo hot store is used directly from Mojo code
    # For Python, import the compiled module:
    #   from aequify.core.db import hot
    #   registry = hot.HotStoreRegistry()

    # Flush manager (Python):
    from aequify.core.db.hot import FlushManager, FlushConfig

    manager = FlushManager(registry, cold_store, FlushConfig())
    await manager.start()
"""

from .flush import (
    FlushConfig,
    FlushManager,
    FlushStats,
    create_flush_manager,
    get_current_minute_of_day,
    get_next_canonical_minute,
    get_previous_canonical_minute,
    seconds_until_next_canonical,
)

__all__ = [
    # Flush Manager
    "FlushManager",
    "FlushConfig",
    "FlushStats",
    "create_flush_manager",
    # Time utilities
    "get_current_minute_of_day",
    "get_next_canonical_minute",
    "get_previous_canonical_minute",
    "seconds_until_next_canonical",
]
