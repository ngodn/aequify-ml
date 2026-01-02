"""
QuestDB Cold Storage.

Provides persistent storage for historical trade data, bootstrap results,
parameter bounds, and session levels.

Components:
- ColdStore: Async cold storage interface
- SyncColdStore: Synchronous wrapper for Mojo/sync contexts
- BackfillManager: Non-blocking background trade backfill
- Models: Trade, BootstrapResult, BootstrapBounds, SessionLevels

Usage:
    # Async usage
    async with ColdStore() as store:
        await store.insert_trades(trades)
        trades = await store.get_trades("BTC/USDT:USDT", limit=1000)

    # Sync usage (from Mojo or sync code)
    with SyncColdStore() as store:
        store.insert_trades(trades)
        trades = store.get_trades("BTC/USDT:USDT", limit=1000)

    # Background backfill
    manager = start_background_backfill(["BTC/USDT:USDT", "ETH/USDT:USDT"])
    # ... do other work ...
    progress = manager.get_progress("BTC/USDT:USDT")
"""

from .backfill import (
    BackfillConfig,
    BackfillManager,
    BackfillProgress,
    BackfillStatus,
    backfill_symbol,
    start_background_backfill,
)
from .client import (
    QuestDBClient,
    QuestDBConfig,
    SyncQuestDBClient,
    load_questdb_config,
)
from .models import (
    BootstrapBounds,
    BootstrapResult,
    Direction,
    ImbalanceLevel,
    SessionLevels,
    Trade,
)
from .store import (
    ColdStore,
    SyncColdStore,
)

__all__ = [
    # Client
    "QuestDBClient",
    "QuestDBConfig",
    "SyncQuestDBClient",
    "load_questdb_config",
    # Store
    "ColdStore",
    "SyncColdStore",
    # Models
    "Trade",
    "BootstrapResult",
    "BootstrapBounds",
    "SessionLevels",
    "ImbalanceLevel",
    "Direction",
    # Backfill
    "BackfillManager",
    "BackfillConfig",
    "BackfillProgress",
    "BackfillStatus",
    "start_background_backfill",
    "backfill_symbol",
]
