"""
Exit Executor - Executes position exits with proper order management.

Handles position closing, TP/SL order cancellation, and PnL tracking.
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
    from aequify.core.trading.position_manager import PositionManager, TrackedPosition
    from aequify.core.trading.risk_manager import RiskManager

logger = get_logger(__name__)


@dataclass
class ExitResult:
    """
    Result of an exit execution.

    Attributes:
        success: Whether exit was successful.
        position_id: ID of the closed position.
        exit_order: Filled exit order.
        exit_reason: Reason for exit (TP, SL, MANUAL, TIMEOUT, etc).
        realized_pnl: Realized PnL in USDT.
        realized_pnl_pct: Realized PnL percentage.
        error: Error message if failed.
        details: Additional execution details.
    """

    success: bool
    position_id: str | None = None
    exit_order: "Order | None" = None
    exit_reason: str = ""
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    error: str = ""
    details: dict[str, Any] | None = None


class ExitExecutor:
    """
    Executes position exits.

    Handles:
    - Market close orders
    - TP/SL order cancellation
    - Position tracking updates
    - PnL calculation and reporting

    Usage:
        executor = ExitExecutor(
            client=client,
            position_manager=position_manager,
            risk_manager=risk_manager,
        )

        result = await executor.execute(
            position=position,
            exit_reason="TP",
            current_price=51000.0,
        )

        if result.success:
            # Position closed
            logger.info(f"PnL: {result.realized_pnl_pct:+.2f}%")
    """

    def __init__(
        self,
        client: "BinanceFuturesClient",
        position_manager: "PositionManager",
        risk_manager: "RiskManager",
    ) -> None:
        """
        Initialize exit executor.

        Args:
            client: Exchange client for order execution.
            position_manager: Position tracking manager.
            risk_manager: Risk manager for PnL tracking.
        """
        self.client = client
        self.position_manager = position_manager
        self.risk_manager = risk_manager

    async def execute(
        self,
        position: "TrackedPosition",
        exit_reason: str,
        current_price: float | None = None,
    ) -> ExitResult:
        """
        Execute a position exit.

        Args:
            position: Position to close.
            exit_reason: Reason for exit (TP, SL, MANUAL, TIMEOUT, SMART_CLOSE, etc).
            current_price: Current market price (for market orders).

        Returns:
            ExitResult with execution details.
        """
        symbol = position.symbol
        side = position.side

        try:
            # 1. Cancel existing TP/SL orders first
            await self._cancel_pending_orders(position)

            # 2. Get current position quantity from exchange (to handle partial fills)
            actual_quantity = await self._get_position_quantity(symbol, side)
            if actual_quantity <= 0:
                # Position already closed (by TP/SL hit on exchange)
                return await self._handle_external_close(position, exit_reason)

            # 3. Place market close order
            exit_order = await self._place_close_order(symbol, side, actual_quantity)
            if not exit_order:
                return ExitResult(
                    success=False,
                    position_id=position.id,
                    exit_reason=exit_reason,
                    error="Exit order failed",
                )

            # 4. Calculate realized PnL
            exit_price = exit_order.average or current_price or 0.0
            realized_pnl, realized_pnl_pct = position.calculate_pnl(exit_price)

            # 5. Update position with exit details
            await self.position_manager.update_exit(
                position_id=position.id,
                exit_order=exit_order,
                exit_reason=exit_reason,
            )

            # 6. Register close with risk manager
            await self.risk_manager.register_position_close(symbol, realized_pnl)

            logger.info(
                f"Exit executed for {symbol} {side.value} ({exit_reason}): "
                f"PnL {realized_pnl_pct:+.2f}% ({realized_pnl:+.2f} USDT)"
            )

            return ExitResult(
                success=True,
                position_id=position.id,
                exit_order=exit_order,
                exit_reason=exit_reason,
                realized_pnl=realized_pnl,
                realized_pnl_pct=realized_pnl_pct,
                details={
                    "exit_price": exit_price,
                    "exit_quantity": actual_quantity,
                    "entry_price": position.entry_price,
                },
            )

        except Exception as e:
            logger.error(f"Exit execution failed for {symbol} {side.value}: {e}")
            return ExitResult(
                success=False,
                position_id=position.id,
                exit_reason=exit_reason,
                error=str(e),
            )

    async def execute_by_symbol(
        self,
        symbol: str,
        exit_reason: str = "MANUAL",
    ) -> ExitResult:
        """
        Close position by symbol.

        Looks up the tracked position and closes it.

        Args:
            symbol: Trading pair to close.
            exit_reason: Reason for exit.

        Returns:
            ExitResult with execution details.
        """
        position = await self.position_manager.get_position_by_symbol(symbol)
        if not position:
            return ExitResult(
                success=False,
                error=f"No open position found for {symbol}",
            )

        return await self.execute(position, exit_reason)

    async def execute_close_all(
        self,
        exit_reason: str = "MANUAL",
    ) -> list[ExitResult]:
        """
        Close all open positions.

        Args:
            exit_reason: Reason for exit.

        Returns:
            List of ExitResult for each position.
        """
        positions = await self.position_manager.get_open_positions()
        results = []

        for position in positions:
            result = await self.execute(position, exit_reason)
            results.append(result)

        logger.info(f"Closed {len(results)} positions ({exit_reason})")
        return results

    async def _cancel_pending_orders(self, position: "TrackedPosition") -> None:
        """Cancel all pending orders for a position (TP, SL)."""
        # Cancel TP order
        if position.tp_order and position.tp_order.id:
            try:
                await self.client.cancel_order(position.tp_order.id, position.symbol)
                logger.debug(f"Canceled TP order {position.tp_order.id}")
            except Exception as e:
                # Order might already be filled or canceled
                logger.debug(f"Could not cancel TP order: {e}")

        # Cancel SL order
        if position.sl_order and position.sl_order.id:
            try:
                await self.client.cancel_order(position.sl_order.id, position.symbol)
                logger.debug(f"Canceled SL order {position.sl_order.id}")
            except Exception as e:
                logger.debug(f"Could not cancel SL order: {e}")

    async def _get_position_quantity(
        self,
        symbol: str,
        side: "PositionSide",
    ) -> float:
        """Get current position quantity from exchange."""
        try:
            positions = await self.client.get_positions([symbol])

            for pos in positions:
                if pos.side == side:
                    return abs(pos.contracts) if pos.contracts else 0.0

            return 0.0

        except Exception as e:
            logger.error(f"Failed to get position quantity: {e}")
            return 0.0

    async def _place_close_order(
        self,
        symbol: str,
        side: "PositionSide",
        quantity: float,
    ) -> "Order | None":
        """Place market close order."""
        from aequify.core.exchange.binance_futures import PositionSide

        try:
            # Close order is opposite side
            order_side = "sell" if side == PositionSide.LONG else "buy"

            order = await self.client.create_order(
                symbol=symbol,
                type="market",
                side=order_side,
                amount=quantity,
                params={
                    "positionSide": side.value,
                    "reduceOnly": True,
                },
            )

            return order

        except Exception as e:
            logger.error(f"Close order failed: {e}")
            return None

    async def _handle_external_close(
        self,
        position: "TrackedPosition",
        exit_reason: str,
    ) -> ExitResult:
        """
        Handle position that was closed externally (TP/SL hit on exchange).

        Fetches the actual exit price from recent trades.
        """
        symbol = position.symbol

        try:
            # Try to get last fill price
            exit_price = await self._get_last_fill_price(symbol, position.side)
            if exit_price is None:
                exit_price = position.entry_price  # Fallback

            realized_pnl, realized_pnl_pct = position.calculate_pnl(exit_price)

            # Update position (no exit order since it was external)
            await self.position_manager.update_exit(
                position_id=position.id,
                exit_order=None,
                exit_reason=f"{exit_reason}_EXTERNAL",
            )

            # Register close with risk manager
            await self.risk_manager.register_position_close(symbol, realized_pnl)

            logger.info(
                f"External close detected for {symbol} {position.side.value}: "
                f"PnL {realized_pnl_pct:+.2f}%"
            )

            return ExitResult(
                success=True,
                position_id=position.id,
                exit_order=None,
                exit_reason=f"{exit_reason}_EXTERNAL",
                realized_pnl=realized_pnl,
                realized_pnl_pct=realized_pnl_pct,
                details={
                    "exit_price": exit_price,
                    "external_close": True,
                },
            )

        except Exception as e:
            logger.error(f"Failed to handle external close: {e}")
            return ExitResult(
                success=False,
                position_id=position.id,
                exit_reason=exit_reason,
                error=f"External close handling failed: {e}",
            )

    async def _get_last_fill_price(
        self,
        symbol: str,
        side: "PositionSide",
    ) -> float | None:
        """Get last fill price for a symbol from recent trades."""
        from aequify.core.exchange.binance_futures import PositionSide

        try:
            # Fetch recent trades
            trades = await self.client.fetch_my_trades(symbol, limit=10)
            if not trades:
                return None

            # Find the most recent closing trade
            close_side = "sell" if side == PositionSide.LONG else "buy"
            for trade in reversed(trades):
                if trade.get("side") == close_side:
                    return float(trade.get("price", 0))

            return None

        except Exception as e:
            logger.debug(f"Could not fetch last fill price: {e}")
            return None

    async def close_hedge_pair(
        self,
        symbol: str,
        exit_reason: str = "HEDGE_CLOSE",
    ) -> tuple[ExitResult | None, ExitResult | None]:
        """
        Close both positions of a hedged pair.

        Args:
            symbol: Trading pair.
            exit_reason: Reason for exit.

        Returns:
            Tuple of (original_result, hedge_result).
        """
        original, hedge = await self.position_manager.get_hedged_pair(symbol)

        original_result = None
        hedge_result = None

        if original and original.is_open:
            original_result = await self.execute(original, exit_reason)

        if hedge and hedge.is_open:
            hedge_result = await self.execute(hedge, exit_reason)

        return original_result, hedge_result

    async def partial_close(
        self,
        position: "TrackedPosition",
        close_pct: float,
        exit_reason: str = "PARTIAL",
    ) -> ExitResult:
        """
        Partially close a position.

        Args:
            position: Position to partially close.
            close_pct: Percentage of position to close (0-100).
            exit_reason: Reason for exit.

        Returns:
            ExitResult with execution details.
        """
        symbol = position.symbol
        side = position.side

        if close_pct <= 0 or close_pct > 100:
            return ExitResult(
                success=False,
                position_id=position.id,
                error=f"Invalid close percentage: {close_pct}",
            )

        try:
            # Get current position quantity
            actual_quantity = await self._get_position_quantity(symbol, side)
            if actual_quantity <= 0:
                return ExitResult(
                    success=False,
                    position_id=position.id,
                    error="No position to close",
                )

            # Calculate quantity to close
            close_quantity = actual_quantity * (close_pct / 100)

            # Use exchange precision
            exchange = self.client._exchange
            if exchange:
                close_quantity = float(exchange.amount_to_precision(symbol, close_quantity))

            if close_quantity <= 0:
                return ExitResult(
                    success=False,
                    position_id=position.id,
                    error="Quantity too small after precision",
                )

            # Place partial close order
            exit_order = await self._place_close_order(symbol, side, close_quantity)
            if not exit_order:
                return ExitResult(
                    success=False,
                    position_id=position.id,
                    error="Partial close order failed",
                )

            exit_price = exit_order.average or 0.0
            closed_notional = close_quantity * exit_price

            # Calculate PnL for closed portion
            if position.entry_price > 0:
                from aequify.core.exchange.binance_futures import PositionSide

                if side == PositionSide.LONG:
                    pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
                else:
                    pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100

                realized_pnl = (pnl_pct / 100) * closed_notional
            else:
                realized_pnl = 0.0
                pnl_pct = 0.0

            # Update position quantity (don't fully close it)
            position.entry_quantity = actual_quantity - close_quantity

            # Register partial PnL with risk manager
            await self.risk_manager.register_position_close(symbol, realized_pnl)

            logger.info(
                f"Partial close ({close_pct:.0f}%) for {symbol} {side.value}: "
                f"closed {close_quantity} @ {exit_price:.6f}, "
                f"PnL {pnl_pct:+.2f}%"
            )

            return ExitResult(
                success=True,
                position_id=position.id,
                exit_order=exit_order,
                exit_reason=f"{exit_reason}_{close_pct:.0f}%",
                realized_pnl=realized_pnl,
                realized_pnl_pct=pnl_pct,
                details={
                    "exit_price": exit_price,
                    "close_quantity": close_quantity,
                    "remaining_quantity": position.entry_quantity,
                    "close_pct": close_pct,
                },
            )

        except Exception as e:
            logger.error(f"Partial close failed: {e}")
            return ExitResult(
                success=False,
                position_id=position.id,
                error=str(e),
            )
