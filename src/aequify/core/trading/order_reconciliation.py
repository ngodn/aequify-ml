"""
Order Reconciliation - Sync TP/SL orders with exchange.

Periodically checks positions and ensures TP/SL orders are placed correctly.
Runs on isolated loop to avoid blocking main thread.

TP/SL Priority:
1. Bootstrap result (VADParameters.target_profit, stop_loss) if bootstrapped
2. Config defaults from config.yaml (or symbol overrides)
   - engines.apex.bootstrap.position.target_profit_pct.default
   - engines.apex.bootstrap.position.stop_loss_pct.default

SL Placement:
- Only placed when DCA is exhausted (position at max size)
- While DCA is possible, no SL (allows averaging down)

Usage:
    # In TradingManager, call reconcile_position_orders for each position
    await reconcile_position_orders(
        client=client,
        pos=position,
        config_path="config/config.yaml",
        bootstrap_params=apex.get_params(symbol, side),  # or None
        max_position_size_usdt=100.0,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aequify.logging import get_logger

if TYPE_CHECKING:
    from aequify.core.apex.config import VADParameters
    from aequify.core.exchange.binance_futures import BinanceFuturesClient

logger = get_logger(__name__)


# Default fallback values (last resort if config fails)
DEFAULT_TP_PCT = 1.5
DEFAULT_SL_PCT = -3.0


def get_tp_sl_pct(
    symbol: str,
    config_path: str,
    bootstrap_params: "VADParameters | None" = None,
) -> tuple[float, float]:
    """
    Get TP and SL percentages for a position.

    Priority:
    1. Bootstrap result (if available)
    2. Config defaults (from config.yaml or symbol override)
    3. Hardcoded defaults

    Args:
        symbol: Trading pair.
        config_path: Path to config.yaml.
        bootstrap_params: Bootstrap parameters (if bootstrapped).

    Returns:
        Tuple of (tp_pct, sl_pct).
    """
    # 1. Try bootstrap result first
    if bootstrap_params:
        return bootstrap_params.target_profit, bootstrap_params.stop_loss

    # 2. Try config (with symbol override)
    try:
        from aequify.core.apex.config import load_apex_config_for_symbol

        config = load_apex_config_for_symbol(config_path, symbol)
        tp_pct = config.target_profit_bounds.default
        sl_pct = config.stop_loss_bounds.default
        return tp_pct, sl_pct
    except Exception as e:
        logger.warning(f"Failed to load config for {symbol}: {e}")

    # 3. Hardcoded defaults
    return DEFAULT_TP_PCT, DEFAULT_SL_PCT


async def reconcile_position_orders(
    client: "BinanceFuturesClient",
    pos: Any,
    config_path: str = "config/config.yaml",
    bootstrap_params: "VADParameters | None" = None,
    max_position_size_usdt: float = 100.0,
) -> bool:
    """
    Reconcile TP/SL orders for a single position.

    Checks existing algo orders, cancels mismatched ones, places correct TP/SL.

    TP is always placed.
    SL is only placed when DCA is exhausted (position >= max_position_size_usdt).

    Args:
        client: Binance futures client.
        pos: Position object from exchange.
        config_path: Path to config.yaml.
        bootstrap_params: Bootstrap parameters (if bootstrapped).
        max_position_size_usdt: Max position size for DCA check.

    Returns:
        True if position was closed (already in profit), False otherwise.
    """
    from aequify.core.exchange.binance_futures import PositionSide

    symbol = pos.symbol
    entry_price = pos.entry_price
    total_size = abs(pos.contracts) if hasattr(pos, "contracts") else abs(pos.size)

    if total_size == 0:
        return False

    # 1. Determine direction
    is_long = pos.side == PositionSide.LONG
    position_side = "LONG" if is_long else "SHORT"

    logger.debug(
        f"Reconciliation {symbol}: side={position_side}, "
        f"entry={entry_price}, size={total_size}"
    )

    # 2. Get TP/SL percentages
    tp_pct, sl_pct = get_tp_sl_pct(
        symbol,
        config_path,
        bootstrap_params,
    )

    # 3. Calculate expected TP/SL prices
    if is_long:
        expected_tp = entry_price * (1 + tp_pct / 100)
        expected_sl = entry_price * (1 + sl_pct / 100)  # sl_pct is negative
    else:
        expected_tp = entry_price * (1 - tp_pct / 100)
        expected_sl = entry_price * (1 - sl_pct / 100)  # sl_pct is negative, so this adds

    # Use exchange precision
    exchange = client._exchange
    expected_tp = float(exchange.price_to_precision(symbol, expected_tp))
    expected_sl = float(exchange.price_to_precision(symbol, expected_sl))

    # 4. Check if we can still DCA
    notional = pos.notional if hasattr(pos, "notional") and pos.notional else (entry_price * total_size)
    can_still_dca = abs(notional) < max_position_size_usdt
    need_sl = sl_pct < 0 and not can_still_dca

    # 5. Fetch existing algo orders
    try:
        algo_orders = await client.fetch_algo_open_orders(symbol)
        logger.debug(
            f"Reconciliation {symbol}: found {len(algo_orders)} algo orders, "
            f"TP={expected_tp:.6f}, need_sl={need_sl}"
        )
    except Exception as e:
        logger.warning(f"Failed to fetch algo orders for {symbol}: {e}")
        algo_orders = []

    # 6. Check existing TP/SL orders - filter by positionSide for hedge mode
    has_correct_tp = False
    has_correct_sl = False
    orders_for_this_side = []

    for order in algo_orders:
        # Filter by positionSide - in hedge mode, orders have LONG/SHORT
        order_position_side = order.get("positionSide", "BOTH")
        if order_position_side != position_side and order_position_side != "BOTH":
            continue

        orders_for_this_side.append(order)
        order_type = order.get("type", "")
        trigger_price = float(order.get("triggerPrice", 0))
        quantity = float(order.get("quantity", 0))

        if order_type == "TAKE_PROFIT_MARKET" and trigger_price > 0:
            # Check if TP price matches (within 0.1% tolerance)
            price_diff = abs(trigger_price - expected_tp) / expected_tp
            size_diff = abs(quantity - total_size) / total_size if total_size > 0 else 1
            if price_diff < 0.001 and size_diff < 0.01:
                has_correct_tp = True
                logger.debug(f"Reconciliation {symbol} ({position_side}): TP matches")

        elif order_type == "STOP_MARKET" and trigger_price > 0:
            # Check if SL price matches (within 0.1% tolerance)
            price_diff = abs(trigger_price - expected_sl) / expected_sl
            size_diff = abs(quantity - total_size) / total_size if total_size > 0 else 1
            if price_diff < 0.001 and size_diff < 0.01:
                has_correct_sl = True
                logger.debug(f"Reconciliation {symbol} ({position_side}): SL matches")

    # 7. Determine what needs to be done
    need_tp = not has_correct_tp
    need_new_sl = need_sl and not has_correct_sl

    if not need_tp and not need_new_sl:
        logger.debug(f"Reconciliation {symbol} ({position_side}): orders already correct")
        return False

    # 8. Cancel orders for THIS position side only
    for order in orders_for_this_side:
        algo_id = order.get("algoId")
        if algo_id:
            try:
                await client.cancel_algo_order(symbol, algo_id)
            except Exception as e:
                logger.debug(f"Cancel algo order {algo_id} for {symbol}: {e}")

    close_side = "sell" if is_long else "buy"

    # 9. Place TP (always)
    tp_placed = False
    try:
        await client.create_take_profit_market_order(
            symbol,
            close_side,
            total_size,
            expected_tp,
            params={"positionSide": position_side},
        )
        tp_placed = True
    except Exception as e:
        error_str = str(e)
        if "-2021" in error_str or "immediately trigger" in error_str.lower():
            # TP would immediately trigger = already in profit!
            # Close at market immediately
            logger.info(
                f"Reconciliation {symbol}: TP would trigger immediately, "
                f"closing at market (already in profit)"
            )
            try:
                await client.create_market_order(
                    symbol,
                    close_side,
                    None,
                    params={"positionSide": position_side, "closePosition": True},
                )
                logger.info(f"Reconciliation {symbol}: closed position at market")
                return True  # Position closed
            except Exception as close_err:
                logger.error(f"Reconciliation {symbol}: failed to close at market: {close_err}")
        else:
            logger.error(f"Reconciliation: failed to place TP for {symbol}: {e}")

    # 10. Place SL if DCA exhausted
    if need_sl:
        try:
            await client.create_stop_market_order(
                symbol,
                close_side,
                total_size,
                expected_sl,
                params={"positionSide": position_side},
            )
        except Exception as e:
            logger.error(f"Reconciliation: failed to place SL for {symbol}: {e}")

    if tp_placed:
        logger.info(
            f"Reconciled {symbol} ({position_side}): TP={expected_tp:.6f}, "
            f"SL={'none (DCA possible)' if can_still_dca else f'{expected_sl:.6f}'}"
        )
    else:
        logger.warning(
            f"Reconciliation {symbol} ({position_side}): TP placement failed"
        )

    return False
