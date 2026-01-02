"""
Trading module - Complete trading system with modular components.

Components:
- PositionManager: Position lifecycle tracking
- RiskManager: Pre-trade risk checks
- EntryValidator/Executor: Entry logic
- ExitValidator/Executor: Exit logic
- TradingManager: Main orchestrator
- SymbolManager: Per-symbol trading context
"""

from aequify.core.trading.position_manager import (
    PositionCallback,
    PositionManager,
    TrackedPosition,
    TradeStatus,
)
from aequify.core.trading.risk_manager import (
    RiskCheck,
    RiskCheckResult,
    RiskConfig,
    RiskManager,
)
from aequify.core.trading.symbol_manager import (
    SymbolManager,
    SymbolManagerRegistry,
)
from aequify.core.trading.publisher import (
    TradingPublisher,
    get_trading_publisher,
)
from aequify.core.trading.trading_manager import (
    TradingConfig,
    TradingManager,
    TradingSignal,
    create_trading_manager,
)

# Entry submodule
from aequify.core.trading.entry import (
    EntryConfig,
    EntryExecutor,
    EntryResult,
    EntryValidationResult,
    EntryValidator,
)

# Exit submodule
from aequify.core.trading.exit import (
    ExitConfig,
    ExitExecutor,
    ExitResult,
    ExitValidationResult,
    ExitValidator,
)

__all__ = [
    # Position management
    "PositionManager",
    "TrackedPosition",
    "TradeStatus",
    "PositionCallback",
    # Risk management
    "RiskManager",
    "RiskConfig",
    "RiskCheck",
    "RiskCheckResult",
    # Entry
    "EntryValidator",
    "EntryExecutor",
    "EntryConfig",
    "EntryValidationResult",
    "EntryResult",
    # Exit
    "ExitValidator",
    "ExitExecutor",
    "ExitConfig",
    "ExitValidationResult",
    "ExitResult",
    # Trading
    "TradingManager",
    "TradingConfig",
    "TradingSignal",
    "create_trading_manager",
    # Symbol
    "SymbolManager",
    "SymbolManagerRegistry",
    # Publisher
    "TradingPublisher",
    "get_trading_publisher",
]
