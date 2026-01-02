"""
Trading PubSub Publisher - Centralized trading data publishing.

Publishes real-time trading data to PubSub topics for TUI consumption.

Topics:
    trading.positions       - All positions data
    trading.position.<sym>  - Single symbol position (normalized: btcusdtusdt)
    trading.trade.<sym>     - Trade data for symbol (normalized: btcusdtusdt)
    trading.symbols         - Symbol list with metadata
    trading.orders          - Order updates
    trading.rate_limit      - API rate limit status

Usage:
    from aequify.core.trading.publisher import TradingPublisher

    # Get or create publisher (uses Engine's PubSub server)
    publisher = TradingPublisher.get_instance()

    # Publish position updates
    publisher.publish_positions(positions_data)

    # Publish trade
    publisher.publish_trade("BTC/USDT:USDT", trade_dict)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.pubsub import PubSubServer

logger = get_logger(__name__)


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol for topic names.

    Removes slashes and colons, converts to lowercase.
    "BTC/USDT:USDT" -> "btcusdtusdt"
    """
    return symbol.replace("/", "").replace(":", "").lower()


class TradingPublisher:
    """
    Centralized publisher for trading data via PubSub.

    Singleton that publishes trading events to standard topics.
    Automatically discovers the Engine's PubSub server.
    """

    _instance: "TradingPublisher | None" = None
    _lock: threading.Lock = threading.Lock()

    # Topic prefixes
    TOPIC_POSITIONS = "trading.positions"
    TOPIC_POSITION_PREFIX = "trading.position."  # + normalized symbol
    TOPIC_TRADE_PREFIX = "trading.trade."  # + normalized symbol
    TOPIC_SYMBOLS = "trading.symbols"
    TOPIC_ORDERS = "trading.orders"
    TOPIC_RATE_LIMIT = "trading.rate_limit"
    TOPIC_STREAM_STATUS = "trading.stream_status"
    TOPIC_MARKETS = "trading.markets"  # Full market data for symbol info
    TOPIC_APEX_PREFIX = "trading.apex."  # + normalized symbol (bootstrap state)
    TOPIC_APEX_LIVE_PREFIX = "trading.apex.live."  # + normalized symbol (real-time metrics)

    def __new__(cls) -> "TradingPublisher":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._pubsub = None
                    cls._instance._stats = {
                        "positions_published": 0,
                        "trades_published": 0,
                        "symbols_published": 0,
                        "orders_published": 0,
                        "rate_limit_published": 0,
                        "last_publish_time": 0.0,
                    }
        return cls._instance

    @classmethod
    def get_instance(cls) -> "TradingPublisher":
        """Get the singleton instance."""
        return cls()

    def set_pubsub_server(self, pubsub: "PubSubServer") -> None:
        """
        Set the PubSub server to use for publishing.

        Args:
            pubsub: PubSubServer instance (typically from Engine).
        """
        self._pubsub = pubsub
        logger.info("TradingPublisher connected to PubSub server")

    @property
    def is_connected(self) -> bool:
        """Check if connected to PubSub server."""
        return self._pubsub is not None

    def _publish(self, topic: str, data: dict[str, Any]) -> int:
        """
        Internal publish method.

        Args:
            topic: Topic to publish to.
            data: Data payload.

        Returns:
            Number of clients message was sent to.
        """
        if not self._pubsub:
            logger.debug(f"TradingPublisher not connected, skipping {topic}")
            return 0

        try:
            count = self._pubsub.publish(topic, data)
            self._stats["last_publish_time"] = time.time()
            return count
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            return 0

    # =========================================================================
    # Position Publishing
    # =========================================================================

    def publish_positions(self, positions: list[dict[str, Any]]) -> int:
        """
        Publish all positions data.

        Args:
            positions: List of position dicts (CCXT format).

        Returns:
            Number of clients message was sent to.
        """
        data = {
            "positions": positions,
            "count": len(positions),
            "timestamp": time.time(),
        }
        self._stats["positions_published"] += 1
        return self._publish(self.TOPIC_POSITIONS, data)

    def publish_position(self, symbol: str, position: dict[str, Any]) -> int:
        """
        Publish single position update.

        Args:
            symbol: Trading pair in CCXT format.
            position: Position dict (CCXT format).

        Returns:
            Number of clients message was sent to.
        """
        topic = f"{self.TOPIC_POSITION_PREFIX}{_normalize_symbol(symbol)}"
        data = {
            "symbol": symbol,
            "position": position,
            "timestamp": time.time(),
        }
        return self._publish(topic, data)

    # =========================================================================
    # Trade Publishing
    # =========================================================================

    def publish_trade(self, symbol: str, trade: dict[str, Any]) -> int:
        """
        Publish a trade for a symbol.

        Args:
            symbol: Trading pair in CCXT format.
            trade: Trade dict with id, price, amount, timestamp, side.

        Returns:
            Number of clients message was sent to.
        """
        topic = f"{self.TOPIC_TRADE_PREFIX}{_normalize_symbol(symbol)}"
        data = {
            "symbol": symbol,
            "trade": trade,
            "timestamp": time.time(),
        }
        self._stats["trades_published"] += 1
        return self._publish(topic, data)

    def publish_trades_batch(self, symbol: str, trades: list[dict[str, Any]]) -> int:
        """
        Publish batch of trades for a symbol.

        Args:
            symbol: Trading pair in CCXT format.
            trades: List of trade dicts.

        Returns:
            Number of clients message was sent to.
        """
        topic = f"{self.TOPIC_TRADE_PREFIX}{_normalize_symbol(symbol)}"
        data = {
            "symbol": symbol,
            "trades": trades,
            "count": len(trades),
            "batch": True,
            "timestamp": time.time(),
        }
        self._stats["trades_published"] += len(trades)
        return self._publish(topic, data)

    # =========================================================================
    # Symbol Publishing
    # =========================================================================

    def publish_symbols(
        self,
        symbols: list[dict[str, Any]],
        source: str = "filter",
    ) -> int:
        """
        Publish symbol list update.

        Args:
            symbols: List of symbol dicts with metadata.
            source: Source of symbols (filter, watchlist, etc.).

        Returns:
            Number of clients message was sent to.
        """
        data = {
            "symbols": symbols,
            "count": len(symbols),
            "source": source,
            "timestamp": time.time(),
        }
        self._stats["symbols_published"] += 1
        return self._publish(self.TOPIC_SYMBOLS, data)

    def publish_markets(
        self,
        markets: dict[str, dict[str, Any]],
        symbols: list[str] | None = None,
    ) -> int:
        """
        Publish market data for symbol info display.

        Args:
            markets: Full markets dict from exchange.markets.
            symbols: Optional list of symbols to filter (publishes all if None).

        Returns:
            Number of clients message was sent to.
        """
        # Filter to relevant symbols if specified
        if symbols:
            filtered_markets = {s: markets[s] for s in symbols if s in markets}
        else:
            filtered_markets = markets

        data = {
            "markets": filtered_markets,
            "count": len(filtered_markets),
            "timestamp": time.time(),
        }
        return self._publish(self.TOPIC_MARKETS, data)

    # =========================================================================
    # Order Publishing
    # =========================================================================

    def publish_orders(self, orders: list[dict[str, Any]]) -> int:
        """
        Publish order updates.

        Args:
            orders: List of order dicts (CCXT format).

        Returns:
            Number of clients message was sent to.
        """
        data = {
            "orders": orders,
            "count": len(orders),
            "timestamp": time.time(),
        }
        self._stats["orders_published"] += 1
        return self._publish(self.TOPIC_ORDERS, data)

    def publish_order_update(
        self,
        symbol: str,
        order: dict[str, Any],
        event: str = "update",
    ) -> int:
        """
        Publish single order update.

        Args:
            symbol: Trading pair.
            order: Order dict.
            event: Event type (created, filled, canceled, update).

        Returns:
            Number of clients message was sent to.
        """
        data = {
            "symbol": symbol,
            "order": order,
            "event": event,
            "timestamp": time.time(),
        }
        return self._publish(self.TOPIC_ORDERS, data)

    # =========================================================================
    # Stream Status Publishing
    # =========================================================================

    def publish_stream_status(
        self,
        trade_streams_connected: int = 0,
        trade_streams_total: int = 0,
        position_stream_connected: bool = False,
        binance_uid: int = 0,
    ) -> int:
        """
        Publish WebSocket stream status.

        Args:
            trade_streams_connected: Number of trade streams connected.
            trade_streams_total: Total number of trade streams.
            position_stream_connected: Whether position stream is connected.
            binance_uid: Binance account UID.

        Returns:
            Number of clients message was sent to.
        """
        data = {
            "trade_streams_connected": trade_streams_connected,
            "trade_streams_total": trade_streams_total,
            "position_stream_connected": position_stream_connected,
            "binance_uid": binance_uid,
            "timestamp": time.time(),
        }
        return self._publish(self.TOPIC_STREAM_STATUS, data)

    # =========================================================================
    # Rate Limit Publishing
    # =========================================================================

    def publish_rate_limit(
        self,
        weight_used: int,
        weight_limit: int,
        order_count_10s: int = 0,
        order_limit_10s: int = 300,
        order_count_1m: int = 0,
        order_limit_1m: int = 1200,
    ) -> int:
        """
        Publish rate limit status.

        Args:
            weight_used: Current API weight used.
            weight_limit: Maximum API weight.
            order_count_10s: Orders in last 10 seconds.
            order_limit_10s: Order limit per 10 seconds.
            order_count_1m: Orders in last minute.
            order_limit_1m: Order limit per minute.

        Returns:
            Number of clients message was sent to.
        """
        data = {
            "weight_used": weight_used,
            "weight_limit": weight_limit,
            "order_count_10s": order_count_10s,
            "order_limit_10s": order_limit_10s,
            "order_count_1m": order_count_1m,
            "order_limit_1m": order_limit_1m,
            "timestamp": time.time(),
        }
        self._stats["rate_limit_published"] += 1
        return self._publish(self.TOPIC_RATE_LIMIT, data)

    # =========================================================================
    # APEX State Publishing
    # =========================================================================

    def publish_apex_state(
        self,
        symbol: str,
        state: dict[str, Any],
    ) -> int:
        """
        Publish APEX bootstrap/analysis state for a symbol.

        Args:
            symbol: Trading pair in CCXT format.
            state: APEX state dict with bootstrap params, signals, etc.

        Returns:
            Number of clients message was sent to.
        """
        topic = f"{self.TOPIC_APEX_PREFIX}{_normalize_symbol(symbol)}"
        data = {
            "symbol": symbol,
            "state": state,
            "timestamp": time.time(),
        }
        return self._publish(topic, data)

    def publish_apex_live_metrics(
        self,
        symbol: str,
        metrics: dict[str, Any],
    ) -> int:
        """
        Publish APEX real-time metrics for a symbol.

        Called periodically with live volume delta, price moves, etc.

        Args:
            symbol: Trading pair in CCXT format.
            metrics: Dict with volume_delta, price_move_from_high/low, etc.

        Returns:
            Number of clients message was sent to.
        """
        topic = f"{self.TOPIC_APEX_LIVE_PREFIX}{_normalize_symbol(symbol)}"
        data = {
            "symbol": symbol,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        return self._publish(topic, data)

    # =========================================================================
    # Callback Factories
    # =========================================================================

    def make_position_callback(self) -> Any:
        """
        Create a position callback that publishes positions via PubSub.

        Returns a callback compatible with BinanceFuturesPositionStream.on_position.

        Usage:
            from aequify.core.trading import get_trading_publisher
            from aequify.core.rtds import BinanceFuturesPositionStream

            publisher = get_trading_publisher()
            stream = BinanceFuturesPositionStream(
                on_position=publisher.make_position_callback(),
                ...
            )
        """

        def callback(positions: list[Any]) -> None:
            """Publish positions via PubSub."""
            try:
                # Convert PositionData to dicts for publishing
                positions_data = []
                for pos in positions:
                    if hasattr(pos, "symbol"):
                        # PositionData object
                        positions_data.append({
                            "symbol": pos.symbol,
                            "side": pos.side,
                            "contracts": pos.contracts,
                            "entryPrice": pos.entry_price,
                            "markPrice": pos.mark_price,
                            "liquidationPrice": pos.liquidation_price,
                            "unrealizedPnl": pos.unrealized_pnl,
                            "percentage": pos.unrealized_pnl_pct,
                            "notional": pos.notional,
                            "leverage": pos.leverage,
                            "marginMode": pos.margin_mode,
                            "initialMargin": pos.initial_margin,
                            "timestamp": pos.timestamp_ms if hasattr(pos, "timestamp_ms") else 0,
                        })
                    else:
                        # Already a dict
                        positions_data.append(pos)

                self.publish_positions(positions_data)
            except Exception as e:
                logger.debug(f"Failed to publish positions: {e}")

        return callback

    def make_trade_callback(self, symbol: str) -> Any:
        """
        Create a trade callback that publishes trades via PubSub.

        Args:
            symbol: Trading pair in CCXT format.

        Returns a callback compatible with on_trade handlers.
        """

        def callback(trade: dict[str, Any]) -> None:
            """Publish trade via PubSub."""
            try:
                self.publish_trade(symbol, trade)
            except Exception as e:
                logger.debug(f"Failed to publish trade: {e}")

        return callback

    def make_rate_limit_callback(self) -> Any:
        """
        Create a rate limit callback that publishes rate limit status via PubSub.

        Returns a callback compatible with BinanceFuturesRateLimiter.on_update.

        Usage:
            from aequify.core.trading import get_trading_publisher
            from aequify.core.exchange.binance_futures import BinanceFuturesRateLimiter

            publisher = get_trading_publisher()
            limiter = BinanceFuturesRateLimiter()
            limiter.on_update(publisher.make_rate_limit_callback())
        """

        def callback(state: Any) -> None:
            """Publish rate limit status via PubSub."""
            try:
                self.publish_rate_limit(
                    weight_used=state.weight_used,
                    weight_limit=state.weight_limit,
                    order_count_10s=state.order_count_10s,
                    order_limit_10s=state.order_limit_10s,
                    order_count_1m=state.order_count_1m,
                    order_limit_1m=state.order_limit_1m,
                )
            except Exception as e:
                logger.debug(f"Failed to publish rate limit: {e}")

        return callback

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get publisher statistics."""
        return {
            **self._stats,
            "connected": self.is_connected,
        }


# Convenience function
def get_trading_publisher() -> TradingPublisher:
    """Get the global TradingPublisher singleton."""
    return TradingPublisher.get_instance()
