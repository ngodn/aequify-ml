"""
APEX Bootstrap - GPU-accelerated parameter optimization.

Per-symbol adaptive parameter optimizer using Mojo GPU kernels.
Integrates with the aequify runtime for non-blocking operations.

Architecture:
    APEX (one per symbol)
        └─ IsolatedLoop (optional, for non-blocking bootstrap)
            └─ Mojo GPU Kernels
                ├─ volume_imbalance
                ├─ rolling_high / rolling_low
                └─ grid_search_long / grid_search_short

Usage:
    from aequify.core.apex import APEX, APEXConfig

    # Create optimizer for a symbol
    apex = APEX(symbol="BTC/USDT:USDT", config=config)

    # Bootstrap from historical trade data
    result = await apex.bootstrap(timestamps, prices, quantities, sides)

    # Get optimized parameters
    long_params = result.long.params
    short_params = result.short.params
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aequify.core.db.cold.models import BootstrapResult as ColdBootstrapResult

import numpy as np

from aequify.core.apex.config import (
    APEXConfig,
    DEFAULT_CONFIG,
    DirectionalParameters,
    VADParameters,
)
from aequify.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# =============================================================================
# Mojo Kernel Loading
# =============================================================================

# Try to import Mojo modules
_MOJO_AVAILABLE = False
_GPU_AVAILABLE = False
_mojo_bootstrap = None
_mojo_get_gpu_info = None
_mojo_import_error: str | None = None

try:
    import mojo.importer  # noqa: F401

    # Add paths for mojo imports
    _core_dir = Path(__file__).parent.parent  # src/aequify/core
    _kernels_dir = Path(__file__).parent / "kernels"  # src/aequify/core/apex/kernels
    if str(_core_dir) not in sys.path:
        sys.path.insert(0, str(_core_dir))
    if str(_kernels_dir) not in sys.path:
        sys.path.insert(0, str(_kernels_dir))

    # Import apex_gpu for GPU info (from src/aequify/core/apex_gpu.mojo)
    import apex_gpu  # type: ignore[import-not-found]

    _mojo_get_gpu_info = apex_gpu.get_all_info
    _GPU_AVAILABLE = apex_gpu.gpu_available()

    # Import bootstrap kernel (from src/aequify/core/apex/kernels/bootstrap.mojo)
    from bootstrap import bootstrap as _mojo_bootstrap  # type: ignore[import-not-found]

    _MOJO_AVAILABLE = True

except ImportError as e:
    _mojo_import_error = f"Mojo kernels import failed: {e}"
except Exception as e:
    _mojo_import_error = f"Mojo kernels init failed: {type(e).__name__}: {e}"


def is_gpu_available() -> bool:
    """Check if Mojo GPU acceleration is available."""
    return _MOJO_AVAILABLE and _GPU_AVAILABLE


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information."""
    if _MOJO_AVAILABLE and _mojo_get_gpu_info:
        try:
            info = dict(_mojo_get_gpu_info())
            info["available"] = _GPU_AVAILABLE
            info["backend"] = "mojo"
            return info
        except Exception as e:
            return {"available": False, "backend": "none", "error": str(e)}
    return {"available": False, "backend": "none", "error": _mojo_import_error}


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class DirectionResult:
    """Result from parameter optimization for a single direction (LONG or SHORT)."""

    direction: str  # "LONG" or "SHORT"
    params: VADParameters
    entries: int
    winners: int
    win_rate: float
    avg_pnl: float
    loss: float
    gpu_time_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"DirectionResult({self.direction}: {self.params}, "
            f"WR={self.win_rate * 100:.1f}%, entries={self.entries}, "
            f"pnl={self.avg_pnl:+.2f}%)"
        )

    @classmethod
    def from_cold_store(cls, cold_result: "ColdBootstrapResult") -> "DirectionResult":
        """
        Create DirectionResult from cold store BootstrapResult.

        Args:
            cold_result: BootstrapResult from cold store (per-direction).

        Returns:
            DirectionResult with params populated.
        """
        return cls(
            direction=cold_result.direction.value.upper(),
            params=VADParameters(
                price_move=cold_result.price_move,
                time_window=cold_result.time_window,
                delta_threshold=cold_result.imbalance_threshold,
                target_profit=cold_result.take_profit,
                stop_loss=cold_result.stop_loss,
                max_hold_time=cold_result.max_hold_time,
                min_dca_distance=cold_result.min_dca_distance,
            ),
            entries=cold_result.entries,
            winners=cold_result.winners,
            win_rate=cold_result.win_rate,
            avg_pnl=cold_result.avg_pnl,
            loss=cold_result.loss,
        )

    def to_cold_store(self, symbol: str) -> "ColdBootstrapResult":
        """
        Convert to cold store BootstrapResult for persistence.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Cold store BootstrapResult for this direction.
        """
        from aequify.core.db.cold.models import BootstrapResult as ColdBootstrapResult
        from aequify.core.db.cold.models import Direction

        return ColdBootstrapResult(
            symbol=symbol,
            direction=Direction(self.direction.lower()),
            price_move=self.params.price_move,
            min_dca_distance=self.params.min_dca_distance,
            imbalance_threshold=self.params.delta_threshold,
            time_window=self.params.time_window,
            take_profit=self.params.target_profit,
            stop_loss=self.params.stop_loss,
            max_hold_time=self.params.max_hold_time,
            entries=self.entries,
            winners=self.winners,
            win_rate=self.win_rate,
            avg_pnl=self.avg_pnl,
            loss=self.loss,
            timestamp_ms=int(time.time() * 1000),
        )


@dataclass
class BootstrapResult:
    """
    Result from GPU bootstrap optimization.

    Contains best parameters for LONG/SHORT and raw arrays for analysis.
    """

    # Best results for each direction
    long: DirectionResult | None = None
    short: DirectionResult | None = None

    # LONG grid search results (for analysis/override generation)
    long_param_combos: NDArray[np.float64] | None = None
    long_entries: NDArray[np.int32] | None = None
    long_winners: NDArray[np.int32] | None = None
    long_pnls: NDArray[np.float64] | None = None

    # SHORT grid search results (for analysis/override generation)
    short_param_combos: NDArray[np.float64] | None = None
    short_entries: NDArray[np.int32] | None = None
    short_winners: NDArray[np.int32] | None = None
    short_pnls: NDArray[np.float64] | None = None

    gpu_time_ms: float = 0.0

    def __repr__(self) -> str:
        parts = []
        if self.long:
            parts.append(f"LONG: WR={self.long.win_rate * 100:.1f}%, pnl={self.long.avg_pnl:+.2f}%")
        if self.short:
            parts.append(f"SHORT: WR={self.short.win_rate * 100:.1f}%, pnl={self.short.avg_pnl:+.2f}%")
        return f"BootstrapResult({', '.join(parts) or 'none'}, gpu={self.gpu_time_ms:.0f}ms)"


# =============================================================================
# APEX - Per-Symbol Optimizer
# =============================================================================


@dataclass
class APEX:
    """
    Adaptive Parameter Exploration for a single symbol.

    GPU-accelerated parameter optimization using Mojo kernels.

    Supports both LONG and SHORT signal optimization:
    - LONG: price drops + selling pressure -> buy expecting bounce up
    - SHORT: price rises + buying pressure -> sell expecting drop down

    Attributes:
        symbol: Trading pair symbol.
        config: Optimization configuration.
        params: Current optimal parameters for both directions.
        is_bootstrapped: Whether initial optimization is complete.
    """

    symbol: str
    config: APEXConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    params: DirectionalParameters = field(default_factory=DirectionalParameters)
    is_bootstrapped: bool = False

    # Internal state
    _last_optimization_ms: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Log initialization status."""
        if _MOJO_AVAILABLE:
            logger.info(f"[{self.symbol}] APEX initialized with Mojo GPU")
        elif _mojo_import_error:
            logger.warning(f"[{self.symbol}] APEX initialized without GPU: {_mojo_import_error}")
        else:
            logger.warning(f"[{self.symbol}] APEX initialized without GPU (not available)")

    async def bootstrap(
        self,
        timestamps: NDArray[np.int64],
        prices: NDArray[np.float64],
        quantities: NDArray[np.float64],
        sides: NDArray[np.int8],
    ) -> BootstrapResult:
        """
        Bootstrap parameters from historical trade data using GPU.

        Runs GPU-accelerated grid search over parameter space for both
        LONG and SHORT directions.

        Args:
            timestamps: Array of timestamps in milliseconds.
            prices: Array of trade prices.
            quantities: Array of trade quantities.
            sides: Array of trade sides (1=buy, -1=sell).

        Returns:
            BootstrapResult with best parameters and grid search arrays.
        """
        n = len(timestamps)
        logger.info(f"[{self.symbol}] Starting GPU bootstrap with {n:,} trades")

        if not _MOJO_AVAILABLE:
            logger.error(f"[{self.symbol}] Mojo GPU not available, cannot bootstrap")
            return self._empty_bootstrap_result()

        try:
            # Generate parameter grids from config
            long_param_grid = np.array(self.config.get_long_param_grid(), dtype=np.float64)
            short_param_grid = np.array(self.config.get_short_param_grid(), dtype=np.float64)
            time_windows_ms = np.array(self.config.time_windows_ms, dtype=np.int64)

            # Compute optimal GPU params based on dataset size
            if n > 1000:
                time_span_seconds = (timestamps[-1] - timestamps[0]) / 1000.0
                trades_per_second = n / time_span_seconds if time_span_seconds > 0 else 30.0
            else:
                trades_per_second = 30.0  # Default estimate

            max_scan, outer_stride = self.config.compute_gpu_params(n, trades_per_second)

            logger.info(
                f"[{self.symbol}] Grid search: {len(long_param_grid)} LONG, "
                f"{len(short_param_grid)} SHORT combos "
                f"(max_scan={max_scan}, stride={outer_stride}, ~{trades_per_second:.1f} tps)"
            )

            # Call Mojo bootstrap kernel in thread (non-blocking)
            # The Mojo code releases GIL during GPU ops, allowing TUI to run
            mojo_params = {
                "timestamps": timestamps,
                "prices": prices,
                "quantities": quantities,
                "sides": sides,
                "time_windows_ms": time_windows_ms,
                "long_param_grid": long_param_grid,
                "short_param_grid": short_param_grid,
                "imbalance_price_tolerance_pct": self.config.imbalance_price_tolerance_pct,
                "max_scan": max_scan,
                "outer_stride": outer_stride,
                "min_entries": self.config.min_entries_for_validity,
            }
            result = await asyncio.to_thread(_mojo_bootstrap, mojo_params)

            gpu_time_ms = float(result["gpu_time_ms"])

            # Parse LONG result
            long_result = self._parse_mojo_result("LONG", result.get("long"))
            if long_result is None:
                default_long = self.config.get_default_long_params()
                long_result = DirectionResult(
                    direction="LONG",
                    params=default_long,
                    entries=0,
                    winners=0,
                    win_rate=0.0,
                    avg_pnl=0.0,
                    loss=float("inf"),
                )
                logger.warning(
                    f"[{self.symbol}] LONG optimization found no valid params, "
                    f"using defaults: {default_long}"
                )

            # Parse SHORT result
            short_result = self._parse_mojo_result("SHORT", result.get("short"))
            if short_result is None:
                default_short = self.config.get_default_short_params()
                short_result = DirectionResult(
                    direction="SHORT",
                    params=default_short,
                    entries=0,
                    winners=0,
                    win_rate=0.0,
                    avg_pnl=0.0,
                    loss=float("inf"),
                )
                logger.warning(
                    f"[{self.symbol}] SHORT optimization found no valid params, "
                    f"using defaults: {default_short}"
                )

            # Update instance state
            self.params = DirectionalParameters(
                long=long_result.params,
                short=short_result.params,
            )
            self.is_bootstrapped = True
            self._last_optimization_ms = int(time.time() * 1000)

            bootstrap_result = BootstrapResult(
                long=long_result,
                short=short_result,
                long_param_combos=np.array(result["long_param_combos"]),
                long_entries=np.array(result["long_entries"]),
                long_winners=np.array(result["long_winners"]),
                long_pnls=np.array(result["long_pnls"]),
                short_param_combos=np.array(result["short_param_combos"]),
                short_entries=np.array(result["short_entries"]),
                short_winners=np.array(result["short_winners"]),
                short_pnls=np.array(result["short_pnls"]),
                gpu_time_ms=gpu_time_ms,
            )

            # Log architecture info
            arch = result.get("arch", {})
            logger.info(
                f"[{self.symbol}] Bootstrap complete: "
                f"LONG={long_result.params}, "
                f"SHORT={short_result.params}, "
                f"GPU time={gpu_time_ms:.0f}ms "
                f"(warp={arch.get('warp_size', '?')}, block={arch.get('block_size', '?')})"
            )

            return bootstrap_result

        except Exception as e:
            logger.error(f"[{self.symbol}] Bootstrap failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_bootstrap_result()

    def bootstrap_sync(
        self,
        timestamps: NDArray[np.int64],
        prices: NDArray[np.float64],
        quantities: NDArray[np.float64],
        sides: NDArray[np.int8],
    ) -> BootstrapResult:
        """
        Synchronous bootstrap (for use outside async context).

        Wraps the async bootstrap method for synchronous use.
        """
        return asyncio.run(self.bootstrap(timestamps, prices, quantities, sides))

    def _parse_mojo_result(
        self, direction: str, mojo_result: dict | None
    ) -> DirectionResult | None:
        """Parse Mojo bootstrap result into DirectionResult."""
        if mojo_result is None:
            return None

        params = mojo_result.get("params", {})
        dca_info = mojo_result.get("dca", {})

        # Get default DCA distance based on direction
        default_dca = -5.0 if direction == "LONG" else 5.0

        result = DirectionResult(
            direction=direction,
            params=VADParameters(
                price_move=float(params.get("price_move", 0)),
                time_window=int(params.get("time_window", 0)),
                delta_threshold=float(params.get("delta_threshold", 0)),
                target_profit=float(params.get("target_profit", 0)),
                stop_loss=float(params.get("stop_loss", 0)),
                max_hold_time=int(params.get("max_hold_time", 0)),
                dca_distance_pct=float(params.get("dca_distance_pct", default_dca)),
            ),
            entries=int(mojo_result.get("entries", 0)),
            winners=int(mojo_result.get("winners", 0)),
            win_rate=float(mojo_result.get("win_rate", 0)),
            avg_pnl=float(mojo_result.get("avg_pnl", 0)),
            loss=float(mojo_result.get("loss", float("inf"))),
        )

        # Log DCA info if available
        if dca_info:
            logger.debug(
                f"[{self.symbol}] {direction} DCA config: mult={dca_info.get('multiplier', '?')}, "
                f"max_pos={dca_info.get('max_position_mult', '?')}"
            )

        return result

    def _empty_bootstrap_result(self) -> BootstrapResult:
        """Return empty bootstrap result."""
        return BootstrapResult(long=None, short=None, gpu_time_ms=0.0)

    def set_bootstrap_result(self, result: BootstrapResult) -> None:
        """
        Set bootstrap result from cache or external source.

        Updates instance state as if bootstrap() had been called.

        Args:
            result: BootstrapResult to apply.
        """
        self.params = DirectionalParameters(
            long=result.long.params if result.long else None,
            short=result.short.params if result.short else None,
        )
        self.is_bootstrapped = True
        self._last_optimization_ms = int(time.time() * 1000)
        logger.info(
            f"[{self.symbol}] Bootstrap result set: "
            f"LONG={result.long is not None}, SHORT={result.short is not None}"
        )

    def get_signal_params(self) -> dict[str, Any]:
        """
        Get current parameters for signal detection.

        Returns:
            Dictionary with current VAD parameters for both directions.
        """
        result: dict[str, Any] = {
            "symbol": self.symbol,
            "max_hold_time": self.config.position.max_hold_time,
            "is_bootstrapped": self.is_bootstrapped,
            "long": None,
            "short": None,
            # DCA position sizing config (shared)
            "dca": {
                "multiplier": self.config.dca_multiplier,
                "initial_position_usdt": self.config.initial_position_usdt,
                "max_position_usdt": self.config.max_position_usdt,
            },
        }

        if self.params.long:
            result["long"] = {
                "price_move": self.params.long.price_move,
                "time_window": self.params.long.time_window,
                "delta_threshold": self.params.long.delta_threshold,
                "target_profit": self.params.long.target_profit,
                "stop_loss": self.params.long.stop_loss,
                "dca_distance_pct": self.params.long.dca_distance_pct,
            }

        if self.params.short:
            result["short"] = {
                "price_move": self.params.short.price_move,
                "time_window": self.params.short.time_window,
                "delta_threshold": self.params.short.delta_threshold,
                "target_profit": self.params.short.target_profit,
                "stop_loss": self.params.short.stop_loss,
                "dca_distance_pct": self.params.short.dca_distance_pct,
            }

        return result

    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU acceleration is available."""
        return _MOJO_AVAILABLE

    @staticmethod
    def get_gpu_info() -> dict[str, Any]:
        """Get GPU information."""
        return get_gpu_info()


# =============================================================================
# Integration with SymbolManager
# =============================================================================


class APEXManager:
    """
    Manager for multiple APEX instances across symbols.

    Provides a registry pattern for managing per-symbol optimizers.
    Can be integrated with SymbolManagerRegistry for coordinated
    symbol management.

    Usage:
        manager = APEXManager(config)

        # Add a symbol
        apex = manager.add("BTC/USDT:USDT")

        # Bootstrap with data
        result = await apex.bootstrap(ts, px, qty, sides)

        # Get existing optimizer
        apex = manager.get("BTC/USDT:USDT")

        # Remove
        manager.remove("BTC/USDT:USDT")
    """

    def __init__(self, default_config: APEXConfig | None = None) -> None:
        """
        Initialize the APEX manager.

        Args:
            default_config: Default config for new APEX instances.
        """
        self._config = default_config or DEFAULT_CONFIG
        self._instances: dict[str, APEX] = {}

    def add(self, symbol: str, config: APEXConfig | None = None) -> APEX:
        """
        Add an APEX instance for a symbol.

        Args:
            symbol: Trading pair symbol.
            config: Optional per-symbol config (uses default if None).

        Returns:
            The APEX instance.

        Raises:
            ValueError: If symbol already exists.
        """
        if symbol in self._instances:
            raise ValueError(f"APEX for {symbol} already exists")

        apex = APEX(symbol=symbol, config=config or self._config)
        self._instances[symbol] = apex
        logger.info(f"Added APEX for {symbol}")
        return apex

    def get(self, symbol: str) -> APEX | None:
        """Get APEX instance for a symbol."""
        return self._instances.get(symbol)

    def get_or_create(self, symbol: str, config: APEXConfig | None = None) -> APEX:
        """Get existing or create new APEX instance."""
        if symbol not in self._instances:
            return self.add(symbol, config)
        return self._instances[symbol]

    def remove(self, symbol: str) -> bool:
        """Remove APEX instance for a symbol."""
        if symbol in self._instances:
            del self._instances[symbol]
            logger.info(f"Removed APEX for {symbol}")
            return True
        return False

    def symbols(self) -> list[str]:
        """Get list of symbols with APEX instances."""
        return list(self._instances.keys())

    def all(self) -> list[APEX]:
        """Get all APEX instances."""
        return list(self._instances.values())

    def stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        bootstrapped = [a for a in self._instances.values() if a.is_bootstrapped]
        return {
            "total_symbols": len(self._instances),
            "bootstrapped": len(bootstrapped),
            "gpu_available": _MOJO_AVAILABLE,
            "symbols": {
                sym: {
                    "is_bootstrapped": apex.is_bootstrapped,
                    "has_long": apex.params.long is not None,
                    "has_short": apex.params.short is not None,
                }
                for sym, apex in self._instances.items()
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def bootstrap_symbol(
    symbol: str,
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    quantities: NDArray[np.float64],
    sides: NDArray[np.int8],
    config: APEXConfig | None = None,
) -> BootstrapResult:
    """
    Convenience function to bootstrap a symbol.

    Creates a temporary APEX instance and runs bootstrap.

    Args:
        symbol: Trading pair symbol.
        timestamps: Array of timestamps in milliseconds.
        prices: Array of trade prices.
        quantities: Array of trade quantities.
        sides: Array of trade sides (1=buy, -1=sell).
        config: Optional APEX configuration.

    Returns:
        BootstrapResult with optimization results.
    """
    apex = APEX(symbol=symbol, config=config or DEFAULT_CONFIG)
    return await apex.bootstrap(timestamps, prices, quantities, sides)
