"""
Entry Validator - Validates trade entry conditions.

Checks DCA distance thresholds, price precision, and position size limits
before allowing an entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.exchange.binance_futures import BinanceFuturesClient, Position, PositionSide

logger = get_logger(__name__)


@dataclass
class EntryValidationResult:
    """
    Result of entry validation.

    Attributes:
        valid: Whether entry is valid.
        reason: Rejection reason if invalid.
        details: Additional validation details.
    """

    valid: bool
    reason: str = ""
    details: dict[str, Any] | None = None


@dataclass
class EntryConfig:
    """
    Entry configuration per direction.

    Attributes:
        initial_entry_size_usdt: Initial position size in USDT.
        maximum_position_size_usdt: Maximum position size in USDT.
        leverage: Trading leverage.
        margin_type: Margin type (isolated/cross).
        dca_multiplier: DCA size multiplier (next entry = current * multiplier).
    """

    initial_entry_size_usdt: float = 6.5
    maximum_position_size_usdt: float = 100.0
    leverage: int = 10
    margin_type: str = "isolated"
    dca_multiplier: float = 4.25

    @classmethod
    def from_config(cls, config: dict[str, Any], side: str = "long") -> EntryConfig:
        """Create EntryConfig from config dict."""
        trading = config.get("trading", {})
        futures = trading.get("futures", {})
        side_config = futures.get(side, {})

        return cls(
            initial_entry_size_usdt=side_config.get("initial_entry_size_usdt", 6.5),
            maximum_position_size_usdt=side_config.get("maximum_position_size_usdt", 100.0),
            leverage=futures.get("leverage", 10),
            margin_type=(futures.get("margin_mode") or "isolated").lower(),
            dca_multiplier=side_config.get("dca_multiplier", 4.25),
        )


class EntryValidator:
    """
    Validates trade entry conditions.

    Checks:
    - DCA distance threshold (price improvement from entry)
    - Position size limits
    - Price precision and minimum quantity

    Usage:
        validator = EntryValidator(client, long_config, short_config)

        result = await validator.validate(
            symbol="BTC/USDT:USDT",
            side=PositionSide.LONG,
            current_price=50000.0,
            dca_distance_pct=-0.5,  # Price must drop 0.5% from entry
        )

        if result.valid:
            # Proceed with entry
            ...
    """

    def __init__(
        self,
        client: "BinanceFuturesClient",
        long_config: EntryConfig,
        short_config: EntryConfig,
    ) -> None:
        """
        Initialize entry validator.

        Args:
            client: Exchange client for market info.
            long_config: Configuration for LONG entries.
            short_config: Configuration for SHORT entries.
        """
        self.client = client
        self.long_config = long_config
        self.short_config = short_config

        # Position cache to reduce API calls
        self._position_cache: dict[str, tuple[list["Position"], float]] = {}
        self._cache_ttl: float = 2.0  # Cache valid for 2 seconds

    def _get_config(self, side: "PositionSide") -> EntryConfig:
        """Get config for position side."""
        from aequify.core.exchange.binance_futures import PositionSide

        return self.long_config if side == PositionSide.LONG else self.short_config

    async def get_cached_positions(self, symbol: str) -> list["Position"]:
        """
        Get positions for a symbol with caching.

        Uses a short TTL cache (2s) to handle burst signals efficiently.

        Args:
            symbol: Trading pair.

        Returns:
            List of position objects from exchange.
        """
        import time

        now = time.time()

        if symbol in self._position_cache:
            cached_positions, cached_time = self._position_cache[symbol]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"Using cached position for {symbol}")
                return cached_positions

        positions = await self.client.get_positions([symbol])
        self._position_cache[symbol] = (positions, now)
        return positions

    def invalidate_cache(self, symbol: str) -> None:
        """Invalidate cached position for a symbol after trade execution."""
        self._position_cache.pop(symbol, None)

    async def validate(
        self,
        symbol: str,
        side: "PositionSide",
        current_price: float,
        current_position_size: float = 0.0,
        dca_distance_pct: float = 0.0,
    ) -> EntryValidationResult:
        """
        Validate entry conditions.

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.
            current_price: Current market price.
            current_position_size: Current position size in USDT (0 if no position).
            dca_distance_pct: DCA distance threshold %.
                LONG: negative (price must drop X% from entry)
                SHORT: positive (price must rise X% from entry)

        Returns:
            EntryValidationResult indicating if entry is valid.
        """
        config = self._get_config(side)
        is_dca = current_position_size > 0

        # Check max position size
        if current_position_size >= config.maximum_position_size_usdt:
            return EntryValidationResult(
                valid=False,
                reason=f"Max position size reached: {current_position_size:.2f}/{config.maximum_position_size_usdt:.2f} USDT",
                details={
                    "current_size": current_position_size,
                    "max_size": config.maximum_position_size_usdt,
                },
            )

        # For DCA: validate price meets distance threshold
        if is_dca:
            dca_result = await self._validate_dca_distance(
                symbol=symbol,
                side=side,
                current_price=current_price,
                dca_distance_pct=dca_distance_pct,
            )
            if not dca_result.valid:
                return dca_result

        # Validate price precision
        precision_result = await self._validate_price_precision(symbol, current_price)
        if not precision_result.valid:
            return precision_result

        return EntryValidationResult(
            valid=True,
            details={
                "is_dca": is_dca,
                "current_size": current_position_size,
                "config_max_size": config.maximum_position_size_usdt,
            },
        )

    async def _validate_dca_distance(
        self,
        symbol: str,
        side: "PositionSide",
        current_price: float,
        dca_distance_pct: float,
    ) -> EntryValidationResult:
        """
        Validate DCA distance threshold.

        For DCA to be valid:
        - LONG: Current price must be below entry by dca_distance_pct
        - SHORT: Current price must be above entry by dca_distance_pct

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.
            current_price: Current market price.
            dca_distance_pct: DCA distance threshold (negative for LONG, positive for SHORT).

        Returns:
            EntryValidationResult.
        """
        from aequify.core.exchange.binance_futures import PositionSide

        # Get existing position to find entry price
        positions = await self.get_cached_positions(symbol)
        existing_pos = None
        for pos in positions:
            if pos.side == side:
                existing_pos = pos
                break

        if not existing_pos or not existing_pos.entry_price:
            return EntryValidationResult(valid=True)  # No existing position

        entry_price = existing_pos.entry_price
        price_from_entry_pct = ((current_price - entry_price) / entry_price) * 100

        if side == PositionSide.LONG:
            # LONG DCA: price must drop by dca_distance_pct (negative) from entry
            if dca_distance_pct != 0.0:
                if price_from_entry_pct > dca_distance_pct:
                    return EntryValidationResult(
                        valid=False,
                        reason=(
                            f"DCA rejected for {symbol} LONG: "
                            f"price from entry {price_from_entry_pct:+.2f}% > threshold {dca_distance_pct:.1f}% "
                            f"(need {abs(dca_distance_pct):.1f}% drop from entry)"
                        ),
                        details={
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "price_from_entry_pct": price_from_entry_pct,
                            "threshold": dca_distance_pct,
                        },
                    )
            elif current_price >= entry_price:
                # No threshold set - require any price improvement
                return EntryValidationResult(
                    valid=False,
                    reason=(
                        f"DCA rejected for {symbol} LONG: "
                        f"current price {current_price:.6f} >= entry {entry_price:.6f} "
                        "(must buy lower to improve average)"
                    ),
                    details={
                        "entry_price": entry_price,
                        "current_price": current_price,
                    },
                )

        elif side == PositionSide.SHORT:
            # SHORT DCA: price must rise by dca_distance_pct (positive) from entry
            if dca_distance_pct != 0.0:
                if price_from_entry_pct < dca_distance_pct:
                    return EntryValidationResult(
                        valid=False,
                        reason=(
                            f"DCA rejected for {symbol} SHORT: "
                            f"price from entry {price_from_entry_pct:+.2f}% < threshold {dca_distance_pct:.1f}% "
                            f"(need {dca_distance_pct:.1f}% rise from entry)"
                        ),
                        details={
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "price_from_entry_pct": price_from_entry_pct,
                            "threshold": dca_distance_pct,
                        },
                    )
            elif current_price <= entry_price:
                # No threshold set - require any price improvement
                return EntryValidationResult(
                    valid=False,
                    reason=(
                        f"DCA rejected for {symbol} SHORT: "
                        f"current price {current_price:.6f} <= entry {entry_price:.6f} "
                        "(must sell higher to improve average)"
                    ),
                    details={
                        "entry_price": entry_price,
                        "current_price": current_price,
                    },
                )

        return EntryValidationResult(valid=True)

    async def _validate_price_precision(
        self,
        symbol: str,
        price: float,
    ) -> EntryValidationResult:
        """
        Validate price precision for the symbol.

        Args:
            symbol: Trading pair.
            price: Price to validate.

        Returns:
            EntryValidationResult.
        """
        if price <= 0:
            return EntryValidationResult(
                valid=False,
                reason=f"Invalid price for {symbol}: {price}",
            )

        return EntryValidationResult(valid=True)

    def calculate_entry_quantity(
        self,
        symbol: str,
        side: "PositionSide",
        price: float,
        size_usdt: float,
    ) -> float:
        """
        Calculate entry quantity from USDT size.

        Uses exchange precision for the quantity.

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.
            price: Entry price.
            size_usdt: Entry size in USDT.

        Returns:
            Quantity with proper precision.
        """
        if price <= 0:
            return 0.0

        quantity = size_usdt / price

        # Use ccxt's built-in precision methods
        exchange = self.client._exchange
        if exchange:
            quantity = float(exchange.amount_to_precision(symbol, quantity))

        return quantity
