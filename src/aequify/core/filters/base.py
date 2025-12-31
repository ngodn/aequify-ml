"""
Base filter class and result types.

All filters inherit from BaseFilter and implement the apply() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.exchange.binance_futures.client import BinanceFuturesClient

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """
    Result from a filter operation.

    Attributes:
        symbols: Set of symbols that passed the filter.
        metadata: Optional metadata about the filtering (e.g., price changes).
    """

    symbols: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self):
        return iter(self.symbols)

    def __contains__(self, item: str) -> bool:
        return item in self.symbols


@dataclass
class FilterConfig:
    """
    Filter configuration from config.yaml.

    Attributes:
        name: Filter name (e.g., "symbol_filter", "absolute_price_percentage_change").
        enabled: Whether filter is active.
        config: Filter-specific configuration dict.
    """

    name: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterConfig:
        """Create FilterConfig from config dict."""
        return cls(
            name=data.get("name", ""),
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
        )


class BaseFilter(ABC):
    """
    Abstract base class for all symbol filters.

    Filters take a set of symbols and return a filtered set based on
    their specific criteria.

    Subclasses must implement:
        - apply(): The filtering logic

    Attributes:
        name: Filter identifier.
        config: Filter-specific configuration.
        client: Exchange client for API calls.
    """

    name: str = "base"

    def __init__(
        self,
        config: dict[str, Any],
        client: BinanceFuturesClient,
    ) -> None:
        """
        Initialize filter.

        Args:
            config: Filter-specific configuration from config.yaml.
            client: Exchange client for API calls.
        """
        self.config = config
        self.client = client

    @abstractmethod
    async def apply(self, symbols: set[str]) -> FilterResult:
        """
        Apply filter to symbol set.

        Args:
            symbols: Input symbols to filter.

        Returns:
            FilterResult with filtered symbols and optional metadata.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
