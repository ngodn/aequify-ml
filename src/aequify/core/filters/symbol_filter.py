"""
Symbol filter for blacklist/whitelist filtering.

IMPORTANT: This filter MUST be last in the config for correct behavior.
- blacklist: Remove symbols from the list
- whitelist: Force-add symbols regardless of previous filters (if active)
"""

from __future__ import annotations

from typing import Any

from aequify.logging import get_logger

from .base import BaseFilter, FilterResult

logger = get_logger(__name__)


class SymbolFilter(BaseFilter):
    """
    Filter that applies blacklist and whitelist rules.

    This filter should be LAST in the config order because:
    - blacklist removes symbols from previous filter results
    - whitelist force-adds symbols regardless of previous filters

    Whitelist symbols are validated to ensure they are:
    - Actively trading (status=TRADING)
    - Not already in the list (no duplicates)

    Config:
        pair: Quote currency filter (e.g., "USDT")
        blacklist: List of base symbols to exclude (e.g., ["BTC", "ETH"])
        whitelist: List of base symbols to force-include (e.g., ["PIPPIN"])
    """

    name = "symbol_filter"

    async def apply(self, symbols: set[str]) -> FilterResult:
        """
        Apply blacklist and whitelist to symbol set.

        Args:
            symbols: Input symbols from previous filters.

        Returns:
            FilterResult with filtered symbols.
        """
        pair = self.config.get("pair", "USDT")
        blacklist = set(self.config.get("blacklist") or [])
        whitelist = set(self.config.get("whitelist") or [])

        # Start with input symbols
        result_symbols = symbols.copy()

        # Get active symbols from exchange markets
        active_symbols: set[str] = set()
        if self.client and self.client._exchange:
            markets = self.client._exchange.markets
            for symbol, market in markets.items():
                is_active = market.get("active", True)
                info = market.get("info", {})
                status = info.get("status", "TRADING")
                if is_active and status == "TRADING":
                    active_symbols.add(symbol)

        # Apply pair filter - only keep symbols matching the pair
        if pair:
            pair_suffix = f"/{pair}:{pair}"
            result_symbols = {s for s in result_symbols if s.endswith(pair_suffix)}

        # Filter out non-ASCII symbols (e.g., Chinese characters) to prevent encoding issues
        non_ascii = {s for s in result_symbols if not s.isascii()}
        if non_ascii:
            logger.warning(f"Filtering out {len(non_ascii)} non-ASCII symbols: {non_ascii}")
            result_symbols -= non_ascii

        # Apply blacklist - remove matching symbols
        if blacklist:
            before_count = len(result_symbols)
            result_symbols = {s for s in result_symbols if self._extract_base(s) not in blacklist}
            removed = before_count - len(result_symbols)
            if removed > 0:
                logger.debug(f"Blacklist removed {removed} symbols")

        # Apply whitelist - force add symbols (only if active and not already present)
        whitelist_added: list[str] = []
        whitelist_skipped: list[str] = []
        whitelist_price_changes: dict[str, float] = {}
        symbols_needing_price: list[str] = []

        if whitelist:
            for base in whitelist:
                # Convert base symbol to CCXT format
                symbol = f"{base}/{pair}:{pair}"

                # Skip if already in result (from previous filter - already has price data)
                if symbol in result_symbols:
                    logger.debug(f"Whitelist: {symbol} already in list")
                    continue

                # Check if symbol is actively trading
                if active_symbols and symbol not in active_symbols:
                    logger.warning(f"Whitelist: {symbol} is not active/trading, skipping")
                    whitelist_skipped.append(base)
                    continue

                result_symbols.add(symbol)
                whitelist_added.append(base)
                symbols_needing_price.append(symbol)
                logger.debug(f"Whitelist added: {symbol}")

            # Fetch price data for newly added whitelist symbols
            if symbols_needing_price and self.client:
                try:
                    for symbol in symbols_needing_price:
                        ticker = await self.client.fetch_ticker(symbol)
                        pct = ticker.get("percentage", 0.0)
                        if pct is not None:
                            whitelist_price_changes[symbol] = float(pct)
                            logger.debug(f"Whitelist {symbol} price change: {pct:.2f}%")
                except Exception as e:
                    logger.warning(f"Failed to fetch ticker for whitelist symbols: {e}")

        logger.info(
            f"Symbol filter: {len(symbols)} -> {len(result_symbols)} "
            f"(blacklist: {len(blacklist)}, whitelist added: {len(whitelist_added)}"
            f"{f', skipped inactive: {whitelist_skipped}' if whitelist_skipped else ''})"
        )

        return FilterResult(
            symbols=result_symbols,
            metadata={
                "blacklist_applied": list(blacklist),
                "whitelist_added": whitelist_added,
                "whitelist_skipped": whitelist_skipped,
                "price_changes": whitelist_price_changes,
            },
        )

    @staticmethod
    def _extract_base(symbol: str) -> str:
        """
        Extract base currency from CCXT symbol.

        Example: "PIPPIN/USDT:USDT" -> "PIPPIN"
        """
        if "/" in symbol:
            return symbol.split("/")[0]
        return symbol
