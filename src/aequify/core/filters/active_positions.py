"""
Active positions filter.

This filter is handled specially by the pipeline - it always runs first
and its results are always merged into the final symbol set.

Note: This filter is primarily used internally by the pipeline.
You typically don't need to add it to your config - the pipeline
handles active positions automatically.
"""

from __future__ import annotations

from typing import Any

from aequify.logging import get_logger

from .base import BaseFilter, FilterResult

logger = get_logger(__name__)


class ActivePositionsFilter(BaseFilter):
    """
    Filter that returns symbols with active positions.

    Active positions are identified by positionAmt != 0:
    - Positive positionAmt = LONG position
    - Negative positionAmt = SHORT position

    This filter ignores the input symbols and fetches directly from exchange.
    """

    name = "active_positions"

    async def apply(self, symbols: set[str]) -> FilterResult:
        """
        Fetch and return symbols with active positions.

        Args:
            symbols: Ignored - this filter fetches from exchange.

        Returns:
            FilterResult with symbols that have open positions.
        """
        try:
            positions = await self.client.fetch_positions()

            active_symbols: set[str] = set()
            position_details: dict[str, Any] = {}

            for pos in positions:
                # contracts field in CCXT unified format
                position_amt = float(pos.get("contracts", 0) or 0)

                if position_amt != 0:
                    symbol = pos.get("symbol", "")
                    if symbol:
                        active_symbols.add(symbol)
                        position_details[symbol] = {
                            "side": "LONG" if position_amt > 0 else "SHORT",
                            "size": abs(position_amt),
                            "entry_price": float(pos.get("entryPrice", 0) or 0),
                            "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                            "leverage": int(pos.get("leverage", 1) or 1),
                        }

            logger.info(f"Active positions: {len(active_symbols)} symbols")

            return FilterResult(
                symbols=active_symbols,
                metadata={"position_details": position_details},
            )

        except Exception as e:
            logger.error(f"Failed to fetch active positions: {e}")
            return FilterResult(symbols=set(), metadata={"error": str(e)})
