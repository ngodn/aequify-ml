"""
Absolute price percentage change filter.

Selects top N symbols by absolute 24h price change percentage.
Uses Binance GET /fapi/v1/ticker/24hr endpoint via CCXT.
"""

from __future__ import annotations

from typing import Any

from aequify.logging import get_logger

from .base import BaseFilter, FilterResult

logger = get_logger(__name__)


class AbsolutePricePercentageChangeFilter(BaseFilter):
    """
    Filter that selects symbols with highest absolute price change.

    Fetches 24h ticker data and sorts by |priceChangePercent|,
    returning the top N symbols.

    Only includes symbols that are actively trading (status=TRADING).

    Config:
        top: Number of top symbols to return (default: 40)
    """

    name = "absolute_price_percentage_change"

    # Cache last successful result to use on API failures
    _last_result: FilterResult | None = None

    async def apply(self, symbols: set[str]) -> FilterResult:
        """
        Fetch 24h tickers and return top N by absolute price change.

        Args:
            symbols: Ignored - this filter fetches all tickers.

        Returns:
            FilterResult with top N symbols by |priceChangePercent|.
        """
        top_n = self.config.get("top", 40)

        try:
            # Get active/trading symbols from markets
            # Markets are loaded during client.initialize()
            active_symbols: set[str] = set()
            if self.client and self.client._exchange:
                markets = self.client._exchange.markets
                for symbol, market in markets.items():
                    # Check if symbol is actively trading
                    # CCXT stores 'active' field, and Binance has 'status' in info
                    is_active = market.get("active", True)
                    info = market.get("info", {})
                    status = info.get("status", "TRADING")

                    if is_active and status == "TRADING":
                        active_symbols.add(symbol)

                logger.debug(f"Found {len(active_symbols)} active trading symbols")

            # Fetch all 24h tickers
            tickers = await self.client.fetch_tickers()

            # Extract and sort by absolute price change
            ticker_data: list[tuple[str, float, dict[str, Any]]] = []

            for symbol, ticker in tickers.items():
                # Skip non-USDT pairs
                if not symbol.endswith("/USDT:USDT"):
                    continue

                # Skip inactive/delisted symbols
                if active_symbols and symbol not in active_symbols:
                    logger.debug(f"Skipping inactive symbol: {symbol}")
                    continue

                # Get percentage change (CCXT unified format)
                pct_change = ticker.get("percentage")
                if pct_change is None:
                    continue

                abs_change = abs(float(pct_change))
                ticker_data.append((symbol, abs_change, ticker))

            # Store ALL price changes (for symbols added later via active positions/whitelist)
            all_price_changes: dict[str, float] = {}
            for symbol, abs_change, ticker in ticker_data:
                all_price_changes[symbol] = float(ticker.get("percentage", 0))

            # Sort by absolute change descending
            ticker_data.sort(key=lambda x: x[1], reverse=True)

            # Take top N
            top_symbols = ticker_data[:top_n]

            result_symbols: set[str] = set()
            for symbol, abs_change, ticker in top_symbols:
                result_symbols.add(symbol)

            logger.info(
                f"Top {len(result_symbols)} symbols by |priceChange%|: "
                f"range {top_symbols[0][1]:.2f}% to {top_symbols[-1][1]:.2f}%"
                if top_symbols
                else "No symbols found"
            )

            result = FilterResult(
                symbols=result_symbols,
                metadata={"price_changes": all_price_changes},
            )

            # Cache successful result for fallback on future failures
            AbsolutePricePercentageChangeFilter._last_result = result

            return result

        except Exception as e:
            logger.error(f"Failed to fetch 24h tickers: {e}")

            # On failure, use cached result from last successful fetch
            if AbsolutePricePercentageChangeFilter._last_result is not None:
                cached = AbsolutePricePercentageChangeFilter._last_result
                logger.warning(
                    f"Using cached result with {len(cached.symbols)} symbols due to API failure"
                )
                return FilterResult(
                    symbols=cached.symbols.copy(),
                    metadata={**cached.metadata, "from_cache": True, "error": str(e)},
                )

            # No cache available - return empty (first run failure)
            logger.warning("No cached result available, returning empty set")
            return FilterResult(symbols=set(), metadata={"error": str(e)})
