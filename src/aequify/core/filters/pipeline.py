"""
Filter pipeline orchestrator.

Runs filters in config order with special handling:
- Active positions are ALWAYS fetched first and merged into results
- symbol_filter MUST be last in config for correct blacklist/whitelist behavior
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

from .base import BaseFilter, FilterConfig, FilterResult

if TYPE_CHECKING:
    from aequify.exchange.binance_futures.client import BinanceFuturesClient

logger = get_logger(__name__)


class FilterPipeline:
    """
    Orchestrates symbol filtering through configured filter chain.

    The pipeline:
    1. Fetches active positions (always first, regardless of config order)
    2. Runs filters in config order
    3. Merges active positions into final result

    Note: symbol_filter should be last in config for correct behavior.
    Blacklist removes symbols, whitelist force-adds symbols.
    """

    def __init__(
        self,
        filter_configs: list[dict[str, Any]],
        client: BinanceFuturesClient,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            filter_configs: List of filter configs from config.yaml.
            client: Exchange client for API calls.
        """
        self.client = client
        self.filter_configs = [FilterConfig.from_dict(fc) for fc in filter_configs]
        self._filters: list[BaseFilter] = []
        self._active_position_symbols: set[str] = set()
        self._position_sides: dict[str, set[str]] = {}  # symbol -> {"LONG", "SHORT"}

    async def initialize(self) -> None:
        """
        Initialize all filters.

        Must be called before run().
        """
        from .absolute_price_percentage_change import AbsolutePricePercentageChangeFilter
        from .active_positions import ActivePositionsFilter
        from .symbol_filter import SymbolFilter

        filter_registry: dict[str, type[BaseFilter]] = {
            "active_positions": ActivePositionsFilter,
            "absolute_price_percentage_change": AbsolutePricePercentageChangeFilter,
            "symbol_filter": SymbolFilter,
        }

        self._filters = []

        for fc in self.filter_configs:
            if not fc.enabled:
                logger.debug(f"Filter '{fc.name}' is disabled, skipping")
                continue

            filter_class = filter_registry.get(fc.name)
            if filter_class is None:
                logger.warning(f"Unknown filter '{fc.name}', skipping")
                continue

            filter_instance = filter_class(config=fc.config, client=self.client)
            self._filters.append(filter_instance)
            logger.debug(f"Initialized filter: {fc.name}")

    async def run(self) -> FilterResult:
        """
        Run the filter pipeline.

        Returns:
            FilterResult with final filtered symbols.
        """
        # Step 1: Always fetch active positions first
        await self._fetch_active_positions()

        # Step 2: Run filters in config order
        # Start with empty set - first filter populates it
        current_symbols: set[str] = set()
        combined_metadata: dict[str, Any] = {}
        all_price_changes: dict[str, float] = {}

        for i, filter_instance in enumerate(self._filters):
            try:
                result = await filter_instance.apply(current_symbols)

                # First filter or symbol_filter handles its own logic
                if i == 0 or filter_instance.name == "symbol_filter":
                    current_symbols = result.symbols
                else:
                    # Subsequent filters intersect with previous results
                    # But we merge active positions
                    current_symbols = result.symbols

                # Merge price_changes separately to avoid overwriting
                if "price_changes" in result.metadata:
                    all_price_changes.update(result.metadata["price_changes"])

                # Update other metadata (excluding price_changes to avoid overwrite)
                for key, value in result.metadata.items():
                    if key != "price_changes":
                        combined_metadata[key] = value

                logger.info(f"Filter '{filter_instance.name}': {len(result.symbols)} symbols")

            except Exception as e:
                logger.error(f"Filter '{filter_instance.name}' failed: {e}")
                # Continue with current symbols on filter failure
                continue

        # Add merged price_changes to metadata
        combined_metadata["price_changes"] = all_price_changes

        # Step 3: Always include active positions in final result
        # (they should not be filtered out by any filter except explicit blacklist)
        final_symbols = current_symbols | self._active_position_symbols

        if self._active_position_symbols:
            logger.info(f"Active positions merged: {self._active_position_symbols}")

        # Convert sets to lists for serialization
        position_sides_serialized = {
            symbol: list(sides) for symbol, sides in self._position_sides.items()
        }

        return FilterResult(
            symbols=final_symbols,
            metadata={
                **combined_metadata,
                "active_positions": list(self._active_position_symbols),
                "position_sides": position_sides_serialized,
            },
        )

    async def _fetch_active_positions(self) -> None:
        """
        Fetch symbols with active positions.

        Active positions are identified by positionAmt != 0:
        - Positive positionAmt = LONG position
        - Negative positionAmt = SHORT position

        For hedged positions, a symbol can have both LONG and SHORT.
        """
        try:
            positions = await self.client.fetch_positions()

            self._active_position_symbols = set()
            self._position_sides = {}

            for pos in positions:
                # contracts != 0 means active position
                position_amt = float(pos.get("contracts", 0) or 0)
                if position_amt != 0:
                    symbol = pos.get("symbol", "")
                    if symbol:
                        self._active_position_symbols.add(symbol)

                        # In hedge mode, side comes from 'side' field
                        # In one-way mode, side is determined by sign of contracts
                        pos_side = pos.get("side", "").upper()
                        if pos_side in ("LONG", "SHORT"):
                            side = pos_side
                        else:
                            side = "LONG" if position_amt > 0 else "SHORT"

                        # Track sides per symbol (for hedged positions)
                        if symbol not in self._position_sides:
                            self._position_sides[symbol] = set()
                        self._position_sides[symbol].add(side)

                        logger.debug(f"Active position: {symbol} {side} ({position_amt})")

            logger.info(f"Found {len(self._active_position_symbols)} active positions")

        except Exception as e:
            logger.error(f"Failed to fetch active positions: {e}")
            self._active_position_symbols = set()
            self._position_sides = {}

    @property
    def active_positions(self) -> set[str]:
        """Get symbols with active positions."""
        return self._active_position_symbols.copy()
