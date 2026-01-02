"""
Entry Executor - Executes trade entries with proper order management.

Handles position opening, leverage setup, and TP/SL order creation.
Uses runtime.py for non-blocking async execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.exchange.binance_futures import (
        BinanceFuturesClient,
        Order,
        PositionSide,
    )
    from aequify.core.trading.entry.entry_validator import EntryConfig
    from aequify.core.trading.position_manager import PositionManager, TrackedPosition
    from aequify.core.trading.risk_manager import RiskManager

logger = get_logger(__name__)


@dataclass
class EntryResult:
    """
    Result of an entry execution.

    Attributes:
        success: Whether entry was successful.
        position_id: ID of the created/updated position.
        entry_order: Filled entry order.
        tp_order: Take profit order (if created).
        sl_order: Stop loss order (if created).
        error: Error message if failed.
        details: Additional execution details.
    """

    success: bool
    position_id: str | None = None
    entry_order: "Order | None" = None
    tp_order: "Order | None" = None
    sl_order: "Order | None" = None
    error: str = ""
    details: dict[str, Any] | None = None


class EntryExecutor:
    """
    Executes trade entries.

    Handles:
    - Leverage and margin mode setup
    - Market/limit order placement
    - TP/SL order creation
    - Position tracking updates

    Usage:
        executor = EntryExecutor(
            client=client,
            position_manager=position_manager,
            risk_manager=risk_manager,
            long_config=long_config,
            short_config=short_config,
        )

        result = await executor.execute(
            symbol="BTC/USDT:USDT",
            side=PositionSide.LONG,
            current_price=50000.0,
            tp_pct=1.0,
            sl_pct=0.5,
        )

        if result.success:
            # Entry executed
            ...
    """

    def __init__(
        self,
        client: "BinanceFuturesClient",
        position_manager: "PositionManager",
        risk_manager: "RiskManager",
        long_config: "EntryConfig",
        short_config: "EntryConfig",
    ) -> None:
        """
        Initialize entry executor.

        Args:
            client: Exchange client for order execution.
            position_manager: Position tracking manager.
            risk_manager: Risk manager for position size tracking.
            long_config: Configuration for LONG entries.
            short_config: Configuration for SHORT entries.
        """
        self.client = client
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.long_config = long_config
        self.short_config = short_config

    def _get_config(self, side: "PositionSide") -> "EntryConfig":
        """Get config for position side."""
        from aequify.core.exchange.binance_futures import PositionSide

        return self.long_config if side == PositionSide.LONG else self.short_config

    async def execute(
        self,
        symbol: str,
        side: "PositionSide",
        current_price: float,
        tp_pct: float = 1.0,
        sl_pct: float = 0.5,
        is_hedge: bool = False,
        hedge_of_position_id: str | None = None,
        signal_params: dict[str, Any] | None = None,
        max_hold_time_ms: int = 120000,
    ) -> EntryResult:
        """
        Execute a trade entry.

        Args:
            symbol: Trading pair.
            side: LONG or SHORT.
            current_price: Current market price.
            tp_pct: Take profit percentage from entry.
            sl_pct: Stop loss percentage from entry.
            is_hedge: Whether this is a hedge position.
            hedge_of_position_id: ID of position being hedged.
            signal_params: Signal parameters that triggered this trade.
            max_hold_time_ms: Maximum hold time in milliseconds.

        Returns:
            EntryResult with execution details.
        """
        config = self._get_config(side)

        try:
            # 1. Setup leverage and margin mode
            await self._setup_position_mode(symbol, config)

            # 2. Calculate entry size
            entry_size_usdt = self._calculate_entry_size(symbol, config)
            if entry_size_usdt is None:
                return EntryResult(
                    success=False,
                    error=f"Max position size reached for {symbol}",
                )

            # 3. Calculate quantity
            quantity = self._calculate_quantity(symbol, current_price, entry_size_usdt)
            if quantity <= 0:
                return EntryResult(
                    success=False,
                    error=f"Invalid quantity calculated for {symbol}",
                )

            # 4. Create/get tracked position
            position = await self.position_manager.create_position(
                symbol=symbol,
                side=side,
                signal_params=signal_params,
                max_hold_time_ms=max_hold_time_ms,
                is_hedge=is_hedge,
                hedge_of_position_id=hedge_of_position_id,
            )

            # 5. Place entry order
            entry_order = await self._place_entry_order(symbol, side, quantity)
            if not entry_order or entry_order.status != "closed":
                await self.position_manager.fail_position(
                    position.id, "Entry order not filled"
                )
                return EntryResult(
                    success=False,
                    position_id=position.id,
                    error="Entry order not filled",
                    details={"order": entry_order.to_dict() if entry_order else None},
                )

            # 6. Calculate TP/SL prices
            entry_price = entry_order.average or current_price
            tp_price, sl_price = self._calculate_tp_sl_prices(
                side, entry_price, tp_pct, sl_pct
            )

            # 7. Place TP/SL orders
            tp_order = await self._place_tp_order(symbol, side, quantity, tp_price)
            sl_order = await self._place_sl_order(symbol, side, quantity, sl_price)

            # 8. Update position with entry details
            await self.position_manager.update_entry(
                position_id=position.id,
                entry_order=entry_order,
                tp_order=tp_order,
                sl_order=sl_order,
            )

            # 9. Register with risk manager
            await self.risk_manager.register_position_open(symbol, entry_size_usdt)

            logger.info(
                f"Entry executed for {symbol} {side.value}: "
                f"{quantity} @ {entry_price:.6f} "
                f"(TP: {tp_price:.6f}, SL: {sl_price:.6f})"
            )

            return EntryResult(
                success=True,
                position_id=position.id,
                entry_order=entry_order,
                tp_order=tp_order,
                sl_order=sl_order,
                details={
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "size_usdt": entry_size_usdt,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "is_hedge": is_hedge,
                },
            )

        except Exception as e:
            logger.error(f"Entry execution failed for {symbol} {side.value}: {e}")
            return EntryResult(
                success=False,
                error=str(e),
            )

    async def _setup_position_mode(self, symbol: str, config: "EntryConfig") -> None:
        """Setup leverage and margin mode for the symbol."""
        try:
            # Set leverage
            await self.client.set_leverage(symbol, config.leverage)

            # Set margin mode
            await self.client.set_margin_mode(symbol, config.margin_type)

        except Exception as e:
            # Log but don't fail - might already be set
            logger.debug(f"Position mode setup for {symbol}: {e}")

    def _calculate_entry_size(
        self,
        symbol: str,
        config: "EntryConfig",
    ) -> float | None:
        """
        Calculate entry size in USDT.

        Uses DCA sizing if position exists.
        """
        return self.risk_manager.calculate_dca_entry_size(
            symbol=symbol,
            initial_size=config.initial_entry_size_usdt,
            max_size=config.maximum_position_size_usdt,
            multiplier=config.dca_multiplier,
        )

    def _calculate_quantity(
        self,
        symbol: str,
        price: float,
        size_usdt: float,
    ) -> float:
        """Calculate order quantity from USDT size with precision."""
        if price <= 0:
            return 0.0

        quantity = size_usdt / price

        # Use ccxt's built-in precision methods
        exchange = self.client._exchange
        if exchange:
            quantity = float(exchange.amount_to_precision(symbol, quantity))

        return quantity

    def _calculate_tp_sl_prices(
        self,
        side: "PositionSide",
        entry_price: float,
        tp_pct: float,
        sl_pct: float,
    ) -> tuple[float, float]:
        """
        Calculate TP and SL prices.

        Args:
            side: LONG or SHORT.
            entry_price: Entry price.
            tp_pct: Take profit percentage.
            sl_pct: Stop loss percentage.

        Returns:
            Tuple of (tp_price, sl_price).
        """
        from aequify.core.exchange.binance_futures import PositionSide

        if side == PositionSide.LONG:
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
        else:  # SHORT
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)

        return tp_price, sl_price

    async def _place_entry_order(
        self,
        symbol: str,
        side: "PositionSide",
        quantity: float,
    ) -> "Order | None":
        """Place market entry order."""
        from aequify.core.exchange.binance_futures import PositionSide

        try:
            order_side = "buy" if side == PositionSide.LONG else "sell"

            order = await self.client.create_order(
                symbol=symbol,
                type="market",
                side=order_side,
                amount=quantity,
                params={"positionSide": side.value},
            )

            return order

        except Exception as e:
            logger.error(f"Entry order failed: {e}")
            return None

    async def _place_tp_order(
        self,
        symbol: str,
        side: "PositionSide",
        quantity: float,
        tp_price: float,
    ) -> "Order | None":
        """Place take profit order."""
        from aequify.core.exchange.binance_futures import PositionSide

        try:
            # TP order is opposite side
            order_side = "sell" if side == PositionSide.LONG else "buy"

            # Use exchange precision for price
            exchange = self.client._exchange
            if exchange:
                tp_price = float(exchange.price_to_precision(symbol, tp_price))

            order = await self.client.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=order_side,
                amount=quantity,
                price=None,
                params={
                    "positionSide": side.value,
                    "stopPrice": tp_price,
                    "reduceOnly": True,
                },
            )

            return order

        except Exception as e:
            logger.warning(f"TP order failed for {symbol}: {e}")
            return None

    async def _place_sl_order(
        self,
        symbol: str,
        side: "PositionSide",
        quantity: float,
        sl_price: float,
    ) -> "Order | None":
        """Place stop loss order."""
        from aequify.core.exchange.binance_futures import PositionSide

        try:
            # SL order is opposite side
            order_side = "sell" if side == PositionSide.LONG else "buy"

            # Use exchange precision for price
            exchange = self.client._exchange
            if exchange:
                sl_price = float(exchange.price_to_precision(symbol, sl_price))

            order = await self.client.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=order_side,
                amount=quantity,
                price=None,
                params={
                    "positionSide": side.value,
                    "stopPrice": sl_price,
                    "reduceOnly": True,
                },
            )

            return order

        except Exception as e:
            logger.warning(f"SL order failed for {symbol}: {e}")
            return None

    async def execute_dca(
        self,
        position: "TrackedPosition",
        current_price: float,
        tp_pct: float = 1.0,
        sl_pct: float = 0.5,
    ) -> EntryResult:
        """
        Execute a DCA entry on an existing position.

        Cancels existing TP/SL orders and creates new ones at updated levels.

        Args:
            position: Existing tracked position.
            current_price: Current market price.
            tp_pct: Take profit percentage from new average entry.
            sl_pct: Stop loss percentage from new average entry.

        Returns:
            EntryResult with execution details.
        """
        symbol = position.symbol
        side = position.side
        config = self._get_config(side)

        try:
            # 1. Calculate DCA entry size
            entry_size_usdt = self._calculate_entry_size(symbol, config)
            if entry_size_usdt is None:
                return EntryResult(
                    success=False,
                    position_id=position.id,
                    error="Max position size reached",
                )

            # 2. Calculate quantity
            quantity = self._calculate_quantity(symbol, current_price, entry_size_usdt)
            if quantity <= 0:
                return EntryResult(
                    success=False,
                    position_id=position.id,
                    error="Invalid quantity calculated",
                )

            # 3. Cancel existing TP/SL orders
            await self._cancel_tp_sl_orders(position)

            # 4. Place DCA entry order
            entry_order = await self._place_entry_order(symbol, side, quantity)
            if not entry_order or entry_order.status != "closed":
                return EntryResult(
                    success=False,
                    position_id=position.id,
                    error="DCA entry order not filled",
                )

            # 5. Calculate new average entry price
            old_notional = position.entry_price * position.entry_quantity
            new_notional = (entry_order.average or current_price) * quantity
            new_total_quantity = position.entry_quantity + quantity
            new_avg_price = (old_notional + new_notional) / new_total_quantity

            # 6. Calculate new TP/SL prices from new average
            tp_price, sl_price = self._calculate_tp_sl_prices(
                side, new_avg_price, tp_pct, sl_pct
            )

            # 7. Place new TP/SL orders with total quantity
            tp_order = await self._place_tp_order(symbol, side, new_total_quantity, tp_price)
            sl_order = await self._place_sl_order(symbol, side, new_total_quantity, sl_price)

            # 8. Update position entry details
            position.entry_price = new_avg_price
            position.entry_quantity = new_total_quantity
            position.tp_order = tp_order
            position.sl_order = sl_order
            position.target_profit_price = tp_price
            position.stop_loss_price = sl_price

            # 9. Register additional size with risk manager
            await self.risk_manager.register_position_open(symbol, entry_size_usdt)

            logger.info(
                f"DCA executed for {symbol} {side.value}: "
                f"+{quantity} @ {entry_order.average:.6f} "
                f"(new avg: {new_avg_price:.6f}, total: {new_total_quantity})"
            )

            return EntryResult(
                success=True,
                position_id=position.id,
                entry_order=entry_order,
                tp_order=tp_order,
                sl_order=sl_order,
                details={
                    "dca_price": entry_order.average,
                    "dca_quantity": quantity,
                    "new_avg_price": new_avg_price,
                    "new_total_quantity": new_total_quantity,
                    "size_usdt": entry_size_usdt,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                },
            )

        except Exception as e:
            logger.error(f"DCA execution failed for {symbol}: {e}")
            return EntryResult(
                success=False,
                position_id=position.id,
                error=str(e),
            )

    async def _cancel_tp_sl_orders(self, position: "TrackedPosition") -> None:
        """Cancel existing TP/SL orders for a position."""
        try:
            if position.tp_order and position.tp_order.id:
                await self.client.cancel_order(position.tp_order.id, position.symbol)
                logger.debug(f"Canceled TP order {position.tp_order.id}")
        except Exception as e:
            logger.warning(f"Failed to cancel TP order: {e}")

        try:
            if position.sl_order and position.sl_order.id:
                await self.client.cancel_order(position.sl_order.id, position.symbol)
                logger.debug(f"Canceled SL order {position.sl_order.id}")
        except Exception as e:
            logger.warning(f"Failed to cancel SL order: {e}")
