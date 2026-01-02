"""Real-time data streams module."""

from aequify.core.rtds.config import StreamConfig
from aequify.core.rtds.trade_streams import BinanceFuturesTradeStream

__all__ = [
    "StreamConfig",
    "BinanceFuturesTradeStream",
]
