"""
Binance Futures client using CCXT Pro.

Provides async interface for:
- Market data (tickers, orderbook)
- Account data (positions, balance)
- Order management (create, cancel, modify)

Supports demo mode via CCXT's enable_demo_trading().

DEMO ENDPOINTS (as of Dec 2025):
  REST API:  https://demo-fapi.binance.com
  WebSocket: wss://fstream.binancefuture.com
  API Keys:  Generate from https://testnet.binancefuture.com
Use: exchange.enable_demo_trading(True)
Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import ccxt.pro as ccxtpro

from aequify.logging import get_logger

from .models import (
    Order,
    OrderSide,
    OrderType,
    Position,
)
from .rate_limiter import BinanceFuturesRateLimiter, RateLimitCallback, RateLimitState

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    """
    Configuration for Binance Futures client.

    Attributes:
        api_key: Binance API key.
        api_secret: Binance API secret.
        demo: Use demo mode (Binance testnet).
        leverage: Default leverage for new positions.
        margin_type: Default margin type (isolated/cross).
        receive_window: API receive window in ms.
    """

    api_key: str
    api_secret: str
    demo: bool = False
    leverage: int = 10
    margin_type: str = "isolated"
    receive_window: int = 15000  # 15s to handle network latency


class BinanceFuturesClient:
    """
    Async Binance Futures client using CCXT Pro.

    Thread-safe and supports concurrent operations.
    Uses locks to prevent race conditions on shared state.

    Usage:
        async with BinanceFuturesClient(config) as client:
            positions = await client.fetch_positions()
            order = await client.create_market_order("BTC/USDT:USDT", "buy", 0.001)
    """

    def __init__(self, config: ClientConfig) -> None:
        """
        Initialize client.

        Args:
            config: Client configuration.
        """
        self.config = config
        self._exchange: ccxtpro.binanceusdm | None = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._leverage_set: set[str] = set()  # Track symbols with leverage set
        self._rate_limiter = BinanceFuturesRateLimiter()

    async def initialize(self) -> None:
        """
        Initialize exchange connection.

        Must be called before any other method, or use context manager.
        """
        async with self._lock:
            if self._initialized:
                return

            exchange_config = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "recvWindow": self.config.receive_window,
                    "adjustForTimeDifference": True,  # Auto-sync time with Binance servers
                },
            }

            self._exchange = ccxtpro.binanceusdm(exchange_config)

            # For demo mode, use CCXT's built-in enable_demo_trading()
            # =============================================================================
            # BINANCE USDS-M FUTURES DEMO ENDPOINTS (as of Dec 2025):
            #   REST API:  https://demo-fapi.binance.com (NOT testnet.binancefuture.com!)
            #   WebSocket: wss://fstream.binancefuture.com
            # Demo API keys from: https://testnet.binancefuture.com (same keys work for demo-fapi)
            # Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
            # =============================================================================
            if self.config.demo:
                self._exchange.enable_demo_trading(True)
                self._exchange.options["fetchCurrencies"] = False  # Skip currency fetch for demo
                logger.info("Binance Futures client initialized (DEMO)")
            else:
                logger.info("Binance Futures client initialized (LIVE)")

            # Load markets
            await self._exchange.load_markets()
            self._initialized = True

    async def close(self) -> None:
        """Close exchange connection."""
        async with self._lock:
            if self._exchange:
                await self._exchange.close()
                self._exchange = None
                self._initialized = False
                logger.info("Binance Futures client closed")

    async def __aenter__(self) -> BinanceFuturesClient:
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized or not self._exchange:
            raise RuntimeError("Client not initialized. Call initialize() first.")

    def _update_rate_limits(self) -> None:
        """Update rate limits from last response headers."""
        if self._exchange and self._exchange.last_response_headers:
            self._rate_limiter.update_from_headers(self._exchange.last_response_headers)

    @property
    def rate_limiter(self) -> BinanceFuturesRateLimiter:
        """Get the rate limiter instance."""
        return self._rate_limiter

    @property
    def rate_limit_state(self) -> RateLimitState:
        """Get current rate limit state."""
        return self._rate_limiter.state

    def on_rate_limit_update(self, callback: RateLimitCallback) -> None:
        """Register callback for rate limit updates."""
        self._rate_limiter.on_update(callback)

    # =========================================================================
    # Market Data
    # =========================================================================

    async def fetch_tickers(self) -> dict[str, dict[str, Any]]:
        """
        Fetch 24h tickers for all symbols.

        Returns:
            Dict mapping symbol to ticker data.
        """
        self._ensure_initialized()
        result = await self._exchange.fetch_tickers()
        self._update_rate_limits()
        return result

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Fetch 24h ticker for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT:USDT").

        Returns:
            Ticker data dict.
        """
        self._ensure_initialized()
        result = await self._exchange.fetch_ticker(symbol)
        self._update_rate_limits()
        return result

    async def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Fetch orderbook for a symbol.

        Args:
            symbol: Trading pair.
            limit: Depth limit.

        Returns:
            Orderbook dict with bids/asks.
        """
        self._ensure_initialized()
        result = await self._exchange.fetch_order_book(symbol, limit)
        self._update_rate_limits()
        return result

    # =========================================================================
    # Account Data
    # =========================================================================

    async def fetch_balance(self) -> dict[str, Any]:
        """
        Fetch account balance.

        Returns:
            Balance dict with free/used/total per currency.
        """
        self._ensure_initialized()
        result = await self._exchange.fetch_balance()
        self._update_rate_limits()
        return result

    async def fetch_positions(
        self,
        symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch open positions.

        Args:
            symbols: Optional list of symbols to filter.

        Returns:
            List of position dicts.
        """
        self._ensure_initialized()
        result = await self._exchange.fetch_positions(symbols)
        self._update_rate_limits()
        return result

    async def get_positions(
        self,
        symbols: list[str] | None = None,
    ) -> list[Position]:
        """
        Fetch open positions as Position objects.

        Args:
            symbols: Optional list of symbols to filter.

        Returns:
            List of Position objects (excludes empty positions).
        """
        raw_positions = await self.fetch_positions(symbols)
        positions = []

        for raw in raw_positions:
            pos = Position.from_ccxt(raw)
            if pos is not None:  # Excludes empty positions
                positions.append(pos)

        return positions

    # =========================================================================
    # Leverage & Margin
    # =========================================================================

    async def set_leverage(self, symbol: str, leverage: int) -> int:
        """
        Set leverage for a symbol with automatic retry at lower leverage.

        If the requested leverage is not valid for a symbol (e.g., max is 5x),
        automatically retries with half the leverage until successful.

        Args:
            symbol: Trading pair.
            leverage: Leverage value (1-125).

        Returns:
            The actual leverage that was set.
        """
        self._ensure_initialized()

        # Check if we already have a successful leverage for this symbol
        # Use "|" as separator since symbol contains ":" (e.g., "LIGHT/USDT:USDT")
        for key in self._leverage_set:
            if key.startswith(f"{symbol}|"):
                # Already set for this symbol, return the cached leverage
                cached_leverage = int(key.split("|")[1])
                return cached_leverage

        current_leverage = leverage
        min_leverage = 1

        while current_leverage >= min_leverage:
            try:
                await self._exchange.set_leverage(current_leverage, symbol)
                self._update_rate_limits()
                self._leverage_set.add(f"{symbol}|{current_leverage}")
                if current_leverage != leverage:
                    logger.info(
                        f"Set leverage {current_leverage}x for {symbol} (requested {leverage}x)"
                    )
                else:
                    logger.debug(f"Set leverage {current_leverage}x for {symbol}")
                return current_leverage
            except Exception as e:
                self._update_rate_limits()
                error_str = str(e)

                # If leverage already set, we're good
                if "No need to change" in error_str:
                    self._leverage_set.add(f"{symbol}|{current_leverage}")
                    return current_leverage

                # If leverage not valid, try lower
                if "-4028" in error_str or "not valid" in error_str.lower():
                    logger.warning(
                        f"Leverage {current_leverage}x not valid for {symbol}, trying lower..."
                    )
                    current_leverage = current_leverage // 2
                    if current_leverage < min_leverage:
                        current_leverage = min_leverage
                        # Try one last time with minimum leverage
                        try:
                            await self._exchange.set_leverage(current_leverage, symbol)
                            self._update_rate_limits()
                            self._leverage_set.add(f"{symbol}|{current_leverage}")
                            logger.info(f"Set leverage {current_leverage}x for {symbol} (minimum)")
                            return current_leverage
                        except Exception as e2:
                            logger.error(f"Failed to set even minimum leverage for {symbol}: {e2}")
                            raise
                else:
                    # Other error, don't retry
                    logger.warning(f"Failed to set leverage for {symbol}: {e}")
                    raise

        raise ValueError(f"Could not set any valid leverage for {symbol}")

    async def set_margin_type(self, symbol: str, margin_type: str) -> None:
        """
        Set margin type for a symbol.

        Args:
            symbol: Trading pair.
            margin_type: "isolated" or "cross".
        """
        self._ensure_initialized()

        try:
            await self._exchange.set_margin_mode(margin_type.upper(), symbol)
            self._update_rate_limits()
            logger.debug(f"Set margin type {margin_type} for {symbol}")
        except Exception as e:
            self._update_rate_limits()
            error_str = str(e)
            # Ignore common errors:
            # - "No need to change" = already set
            # - "cannot be changed if there exists open orders" = has open orders/position
            if "No need to change" not in error_str and "open orders" not in error_str.lower():
                logger.warning(f"Failed to set margin type for {symbol}: {e}")

    async def prepare_symbol(self, symbol: str) -> None:
        """
        Prepare symbol for trading (set leverage and margin type).

        Args:
            symbol: Trading pair.
        """
        await self.set_leverage(symbol, self.config.leverage)
        await self.set_margin_type(symbol, self.config.margin_type)

    # =========================================================================
    # Order Management
    # =========================================================================

    async def create_market_order(
        self,
        symbol: str,
        side: str | OrderSide,
        amount: float,
        reduce_only: bool = False,
        params: dict[str, Any] | None = None,
    ) -> Order:
        """
        Create a market order.

        Args:
            symbol: Trading pair.
            side: "buy" or "sell".
            amount: Order quantity.
            reduce_only: Only reduce existing position.
            params: Additional CCXT params.

        Returns:
            Created Order object.
        """
        self._ensure_initialized()

        if isinstance(side, OrderSide):
            side = side.value

        order_params = params or {}
        if reduce_only:
            order_params["reduceOnly"] = True

        raw_order = await self._exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=order_params,
        )
        self._update_rate_limits()

        order = Order.from_ccxt(raw_order)
        logger.info(
            f"Market {side.upper()} {amount} {symbol} -> {order.id} "
            f"(filled: {order.filled}, avg: {order.average})"
        )
        return order

    async def create_limit_order(
        self,
        symbol: str,
        side: str | OrderSide,
        amount: float,
        price: float,
        reduce_only: bool = False,
        params: dict[str, Any] | None = None,
    ) -> Order:
        """
        Create a limit order.

        Args:
            symbol: Trading pair.
            side: "buy" or "sell".
            amount: Order quantity.
            price: Limit price.
            reduce_only: Only reduce existing position.
            params: Additional CCXT params.

        Returns:
            Created Order object.
        """
        self._ensure_initialized()

        if isinstance(side, OrderSide):
            side = side.value

        order_params = params or {}
        if reduce_only:
            order_params["reduceOnly"] = True

        raw_order = await self._exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price,
            params=order_params,
        )
        self._update_rate_limits()

        order = Order.from_ccxt(raw_order)
        logger.info(f"Limit {side.upper()} {amount} {symbol} @ {price} -> {order.id}")
        return order

    async def create_stop_market_order(
        self,
        symbol: str,
        side: str | OrderSide,
        amount: float,
        stop_price: float,
        reduce_only: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> Order:
        """
        Create a stop-market order (for stop-loss).

        Args:
            symbol: Trading pair.
            side: "buy" or "sell".
            amount: Order quantity.
            stop_price: Trigger price.
            reduce_only: Only reduce existing position. None = auto (skip in Hedge Mode).
            params: Additional CCXT params.

        Returns:
            Created Order object.
        """
        self._ensure_initialized()

        if isinstance(side, OrderSide):
            side = side.value

        order_params = {
            "stopPrice": stop_price,
            **(params or {}),
        }

        # In Hedge Mode (with positionSide), reduceOnly is not allowed
        # Only add reduceOnly if explicitly set AND no positionSide
        if reduce_only is not None and "positionSide" not in order_params:
            order_params["reduceOnly"] = reduce_only

        raw_order = await self._exchange.create_order(
            symbol=symbol,
            type="STOP_MARKET",
            side=side,
            amount=amount,
            params=order_params,
        )
        self._update_rate_limits()

        order = Order.from_ccxt(raw_order)
        logger.info(f"Stop-Market {side.upper()} {amount} {symbol} @ {stop_price} -> {order.id}")
        return order

    async def create_take_profit_market_order(
        self,
        symbol: str,
        side: str | OrderSide,
        amount: float,
        stop_price: float,
        reduce_only: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> Order:
        """
        Create a take-profit market order.

        Args:
            symbol: Trading pair.
            side: "buy" or "sell".
            amount: Order quantity.
            stop_price: Trigger price.
            reduce_only: Only reduce existing position. None = auto (skip in Hedge Mode).
            params: Additional CCXT params.

        Returns:
            Created Order object.
        """
        self._ensure_initialized()

        if isinstance(side, OrderSide):
            side = side.value

        order_params = {
            "stopPrice": stop_price,
            **(params or {}),
        }

        # In Hedge Mode (with positionSide), reduceOnly is not allowed
        # Only add reduceOnly if explicitly set AND no positionSide
        if reduce_only is not None and "positionSide" not in order_params:
            order_params["reduceOnly"] = reduce_only

        raw_order = await self._exchange.create_order(
            symbol=symbol,
            type="TAKE_PROFIT_MARKET",
            side=side,
            amount=amount,
            params=order_params,
        )
        self._update_rate_limits()

        order = Order.from_ccxt(raw_order)
        logger.info(f"TP-Market {side.upper()} {amount} {symbol} @ {stop_price} -> {order.id}")
        return order

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> Order:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel.
            symbol: Trading pair.

        Returns:
            Canceled Order object.
        """
        self._ensure_initialized()

        raw_order = await self._exchange.cancel_order(order_id, symbol)
        self._update_rate_limits()
        order = Order.from_ccxt(raw_order)
        logger.info(f"Canceled order {order_id} for {symbol}")
        return order

    async def cancel_all_orders(self, symbol: str) -> list[Order]:
        """
        Cancel all open orders for a symbol (both regular and algo/conditional orders).

        Since Dec 2025, Binance migrated conditional orders (TP/SL) to the Algo Service.
        We must call BOTH endpoints:
        - DELETE /fapi/v1/allOpenOrders - for regular limit orders
        - DELETE /fapi/v1/algoOpenOrders - for conditional/algo orders (TP/SL)

        Args:
            symbol: Trading pair.

        Returns:
            List of canceled Order objects.
        """
        self._ensure_initialized()

        cancelled_count = 0
        cancelled_orders: list[Order] = []

        # 1. Cancel regular orders via ccxt
        try:
            raw_orders = await self._exchange.cancel_all_orders(symbol)
            self._update_rate_limits()
            cancelled_orders.extend([Order.from_ccxt(o) for o in raw_orders])
            cancelled_count += len(raw_orders)
        except Exception as e:
            # Ignore if no orders to cancel
            if "-2011" not in str(e):  # Unknown order
                logger.debug(f"Regular orders cancel for {symbol}: {e}")

        # 2. Cancel algo/conditional orders (TP/SL) via direct API call
        # This is required since Dec 2025 Binance migration
        try:
            # Convert symbol format: "ZKP/USDT:USDT" -> "ZKPUSDT"
            binance_symbol = symbol.replace("/", "").replace(":USDT", "")

            # Use CCXT implicit API: DELETE /fapi/v1/algoOpenOrders
            response = await self._exchange.fapiPrivateDeleteAlgoOpenOrders(
                {
                    "symbol": binance_symbol,
                }
            )
            self._update_rate_limits()

            # Response is {"code": 200, "msg": "..."} - no order details returned
            if response.get("code") == 200:
                logger.debug(f"Algo orders cancel successful for {symbol}")
                cancelled_count += 1  # At least mark that we tried
        except Exception as e:
            # Ignore if no algo orders to cancel or endpoint not found
            err_str = str(e)
            if "-2011" not in err_str and "Unknown" not in err_str:
                logger.debug(f"Algo orders cancel for {symbol}: {e}")

        logger.info(f"Canceled {cancelled_count} orders for {symbol}")
        return cancelled_orders

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        """
        Fetch order by ID.

        Args:
            order_id: Order ID.
            symbol: Trading pair.

        Returns:
            Order object.
        """
        self._ensure_initialized()

        raw_order = await self._exchange.fetch_order(order_id, symbol)
        self._update_rate_limits()
        return Order.from_ccxt(raw_order)

    async def fetch_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[Order]:
        """
        Fetch open orders (regular orders only, not algo/conditional).

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of open Order objects.
        """
        self._ensure_initialized()

        raw_orders = await self._exchange.fetch_open_orders(symbol)
        self._update_rate_limits()
        return [Order.from_ccxt(o) for o in raw_orders]

    async def fetch_algo_open_orders(self, symbol: str | None = None) -> list[dict]:
        """
        Fetch algo/conditional open orders (TP/SL).

        Since Dec 2025, Binance migrated conditional orders to Algo Service.
        Regular fetch_open_orders does NOT return these.

        Args:
            symbol: Optional trading pair (CCXT format e.g. "BTC/USDT:USDT").
                    If None, fetches all algo orders across all symbols.

        Returns:
            List of algo order dicts with keys like:
            - algoId, symbol, side, type, triggerPrice, quantity, etc.
        """
        self._ensure_initialized()

        try:
            params = {}
            if symbol:
                # Convert symbol format: "ZKP/USDT:USDT" -> "ZKPUSDT"
                params["symbol"] = symbol.replace("/", "").replace(":USDT", "")

            # Use CCXT implicit API: GET /fapi/v1/openAlgoOrders
            response = await self._exchange.fapiPrivateGetOpenAlgoOrders(params)
            self._update_rate_limits()

            # Response can be:
            # - List of orders directly: [{"algoId": ..., "symbol": ..., ...}, ...]
            # - Dict with code: {"code": 200, "msg": "OK", "data": {"orders": [...]}}
            if isinstance(response, list):
                logger.debug(f"fetch_algo_open_orders({symbol}): got {len(response)} orders")
                return response
            elif isinstance(response, dict):
                if response.get("code") == 200:
                    orders = response.get("data", {}).get("orders", [])
                    logger.debug(f"fetch_algo_open_orders({symbol}): got {len(orders)} orders")
                    return orders
                logger.debug(f"fetch_algo_open_orders({symbol}): code={response.get('code')}")
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch algo orders for {symbol}: {e}")
            return []

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        """
        Cancel a single algo order by algoId.

        Args:
            symbol: Trading pair (CCXT format e.g. "BTC/USDT:USDT").
            algo_id: The algoId of the order to cancel.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        self._ensure_initialized()

        try:
            # Convert symbol format: "ZKP/USDT:USDT" -> "ZKPUSDT"
            binance_symbol = symbol.replace("/", "").replace(":USDT", "")

            # Use CCXT implicit API: DELETE /fapi/v1/algo/order
            response = await self._exchange.fapiPrivateDeleteAlgoOrder(
                {
                    "symbol": binance_symbol,
                    "algoId": algo_id,
                }
            )
            self._update_rate_limits()

            if response.get("code") == 200:
                logger.debug(f"Cancelled algo order {algo_id} for {symbol}")
                return True
            else:
                logger.debug(f"Failed to cancel algo order {algo_id}: {response}")
                return False
        except Exception as e:
            logger.debug(f"Failed to cancel algo order {algo_id} for {symbol}: {e}")
            return False

    async def cancel_all_algo_orders_globally(self) -> int:
        """
        Cancel ALL algo orders across ALL symbols.

        Fetches all open algo orders first, then cancels per symbol.

        Returns:
            Number of symbols where orders were cancelled.
        """
        self._ensure_initialized()

        # Get all open algo orders
        all_orders = await self.fetch_algo_open_orders()

        if not all_orders:
            return 0

        # Get unique symbols
        symbols = set()
        for order in all_orders:
            binance_symbol = order.get("symbol", "")
            if binance_symbol:
                symbols.add(binance_symbol)

        # Cancel per symbol
        cancelled_count = 0
        for binance_symbol in symbols:
            try:
                # Use CCXT implicit API: DELETE /fapi/v1/algoOpenOrders
                response = await self._exchange.fapiPrivateDeleteAlgoOpenOrders(
                    {
                        "symbol": binance_symbol,
                    }
                )
                self._update_rate_limits()
                if response.get("code") == 200:
                    cancelled_count += 1
                    logger.info(f"Cancelled all algo orders for {binance_symbol}")
            except Exception as e:
                logger.debug(f"Failed to cancel algo orders for {binance_symbol}: {e}")

        return cancelled_count

    # =========================================================================
    # Account Information
    # =========================================================================

    async def fetch_account_uid(self) -> int | None:
        """
        Fetch Binance account UID from mainnet spot account endpoint.

        Always uses mainnet API keys (AEQUIFY_API_KEY/SECRET) regardless
        of testnet mode since UID is tied to the account.

        Returns:
            Account UID as integer, or None if not available.
        """
        self._ensure_initialized()

        try:
            import os

            import ccxt.pro as ccxtpro

            # Always use mainnet API keys for UID
            api_key = os.environ.get("AEQUIFY_API_KEY", "")
            api_secret = os.environ.get("AEQUIFY_API_SECRET", "")

            if not api_key or not api_secret:
                logger.debug("Mainnet API keys not configured, skipping UID fetch")
                return None

            # Create a temporary spot exchange instance pointing to mainnet
            spot = ccxtpro.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                }
            )

            try:
                result = await spot.privateGetAccount()
                return int(result.get("uid", 0)) or None
            finally:
                await spot.close()
        except Exception as e:
            logger.debug(f"Failed to fetch account UID: {e}")
            return None

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def open_long(
        self,
        symbol: str,
        amount: float,
        take_profit: float | None = None,
        stop_loss: float | None = None,
    ) -> tuple[Order, Order | None, Order | None]:
        """
        Open a long position with optional TP/SL.

        Args:
            symbol: Trading pair.
            amount: Position size.
            take_profit: Optional TP price.
            stop_loss: Optional SL price.

        Returns:
            Tuple of (entry_order, tp_order, sl_order).
        """
        await self.prepare_symbol(symbol)

        # For Hedge Mode (dual position), we need to specify positionSide
        # This works for both One-way and Hedge mode
        entry_params = {"positionSide": "LONG"}

        # Entry order
        entry = await self.create_market_order(symbol, "buy", amount, params=entry_params)

        tp_order = None
        sl_order = None

        # TP order (sell to close long) - same positionSide
        if take_profit and entry.filled > 0:
            tp_order = await self.create_take_profit_market_order(
                symbol, "sell", entry.filled, take_profit, params={"positionSide": "LONG"}
            )

        # SL order (sell to close long) - same positionSide
        if stop_loss and entry.filled > 0:
            sl_order = await self.create_stop_market_order(
                symbol, "sell", entry.filled, stop_loss, params={"positionSide": "LONG"}
            )

        return entry, tp_order, sl_order

    async def open_short(
        self,
        symbol: str,
        amount: float,
        take_profit: float | None = None,
        stop_loss: float | None = None,
    ) -> tuple[Order, Order | None, Order | None]:
        """
        Open a short position with optional TP/SL.

        Args:
            symbol: Trading pair.
            amount: Position size.
            take_profit: Optional TP price.
            stop_loss: Optional SL price.

        Returns:
            Tuple of (entry_order, tp_order, sl_order).
        """
        await self.prepare_symbol(symbol)

        # For Hedge Mode (dual position), we need to specify positionSide
        # This works for both One-way and Hedge mode
        entry_params = {"positionSide": "SHORT"}

        # Entry order
        entry = await self.create_market_order(symbol, "sell", amount, params=entry_params)

        tp_order = None
        sl_order = None

        # TP order (buy to close short) - same positionSide
        if take_profit and entry.filled > 0:
            tp_order = await self.create_take_profit_market_order(
                symbol, "buy", entry.filled, take_profit, params={"positionSide": "SHORT"}
            )

        # SL order (buy to close short) - same positionSide
        if stop_loss and entry.filled > 0:
            sl_order = await self.create_stop_market_order(
                symbol, "buy", entry.filled, stop_loss, params={"positionSide": "SHORT"}
            )

        return entry, tp_order, sl_order

    async def close_position(self, symbol: str) -> Order | None:
        """
        Close entire position for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            Close order, or None if no position.
        """
        positions = await self.get_positions([symbol])

        if not positions:
            logger.info(f"No position to close for {symbol}")
            return None

        pos = positions[0]

        # Close by reversing position
        side = "sell" if pos.is_long else "buy"
        position_side = "LONG" if pos.is_long else "SHORT"

        # Cancel any existing orders first
        await self.cancel_all_orders(symbol)

        # Market close with positionSide for Hedge Mode
        # Note: reduceOnly is NOT allowed with positionSide in Hedge Mode
        try:
            raw_order = await self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=pos.size,
                params={"positionSide": position_side},
            )
            self._update_rate_limits()

            order = Order.from_ccxt(raw_order)
            logger.info(f"Closed {pos.side.value.upper()} position for {symbol}")
            return order
        except Exception as e:
            # Position may have been closed already (by TP/SL)
            if "-2022" in str(e):
                logger.info(f"Position {symbol} already closed (no position to reduce)")
                return None
            raise
