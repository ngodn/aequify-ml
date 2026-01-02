"""
Grid Search Kernels - GPU-accelerated parameter optimization for LONG and SHORT strategies.

Both LONG and SHORT kernels use DCA (Dollar Cost Averaging) strategy:
- Warp-level parallelism: 1 warp per parameter combination
- Warp reduction for aggregating results across lanes

Parameter format: [pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist] (9 params)

LONG Strategy:
    - Entry: price drops below rolling high AND selling pressure (negative delta)
    - DCA: price drops further from avg_entry
    - TP: price rises above avg_entry by target_pct
    - SL: price drops below avg_entry by stop_pct (only when position is maxed)

SHORT Strategy:
    - Entry: price rises above rolling low AND buying pressure (positive delta)
    - DCA: price rises further from avg_entry
    - TP: price drops below avg_entry by target_pct
    - SL: price rises above avg_entry by stop_pct (only when position is maxed)

Usage:
    # GPU version - operates on device buffers
    grid_search_long_gpu(ctx, timestamps_dev, prices_dev, deltas_dev, rolling_highs_dev,
                         param_combos_dev, out_entries_dev, out_winners_dev, out_pnl_dev,
                         n_combos, n_windows, n_prices, max_scan, outer_stride)
"""

from math import ceildiv
from sys import has_amd_gpu_accelerator, has_apple_gpu_accelerator, has_nvidia_gpu_accelerator

from gpu import global_idx, block_idx, thread_idx, block_dim, lane_id
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext, DeviceBuffer
from gpu.primitives.warp import shuffle_down, lane_group_sum
from memory import UnsafePointer

from .rolling_low import NO_LOW_SENTINEL


# =============================================================================
# Architecture Constants
# =============================================================================

# Block size - architecture-dependent
@parameter
fn _get_block_size() -> Int:
    @parameter
    if has_apple_gpu_accelerator():
        return 256
    elif has_amd_gpu_accelerator():
        return 256
    else:
        return 256

comptime BLOCK_SIZE: Int = _get_block_size()

# Warps per block - computed from architecture
comptime WARPS_PER_BLOCK: Int = BLOCK_SIZE // WARP_SIZE


# =============================================================================
# Application Constants
# =============================================================================

# Minimum milliseconds between consecutive entries (5 seconds)
comptime MIN_GAP: Int64 = 5000

# Skip first N trades - need enough history for rolling indicators
comptime START_IDX: Int = 1000


# =============================================================================
# Warp Reduction Helpers
# =============================================================================


@always_inline
fn warp_reduce_sum_i32(val: Int32) -> Int32:
    """Warp-level sum reduction using shuffle.

    Uses lane_group_sum which adapts to WARP_SIZE automatically.
    """
    return lane_group_sum[num_lanes=WARP_SIZE](val)


@always_inline
fn warp_reduce_sum_f32(val: Float32) -> Float32:
    """Warp-level sum reduction using shuffle (Float32).

    Uses lane_group_sum which adapts to WARP_SIZE automatically.
    """
    return lane_group_sum[num_lanes=WARP_SIZE](val)


# =============================================================================
# GPU Kernels - Grid Search LONG
# =============================================================================


fn _grid_search_long_gpu_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    deltas: UnsafePointer[Float64, MutAnyOrigin],
    rolling_highs: UnsafePointer[Float64, MutAnyOrigin],
    param_combos: UnsafePointer[Float64, MutAnyOrigin],
    out_entries: UnsafePointer[Int32, MutAnyOrigin],
    out_winners: UnsafePointer[Int32, MutAnyOrigin],
    out_pnl: UnsafePointer[Float64, MutAnyOrigin],
    n_combos: Int32,
    n_windows: Int32,
    n_prices: Int32,
    max_scan: Int32,
    outer_stride: Int32,
):
    """GPU kernel for LONG grid search with DCA.

    LONG Strategy: price drops + selling pressure → buy expecting bounce.

    DCA Logic:
    - Initial entry: pos_size = 1.0, avg_entry = entry_price
    - DCA trigger: price drops by dca_distance_pct from avg_entry
    - If can DCA (pos_size * dca_mult + pos_size <= max_pos_mult):
        new_size = pos_size * dca_mult
        avg_entry = weighted average
        pos_size += new_size
    - SL only active when position is maxed (can't DCA anymore)
    - TP/SL calculated from avg_entry, not original entry

    WARPS_PER_BLOCK warps per block, 1 warp = 1 combo.
    Param format: [pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist].
    """
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())

    var combo_idx = Int(block_idx.x) * WARPS_PER_BLOCK + warp_id
    if combo_idx >= Int(n_combos):
        return

    # Extract parameters (9 params per combo, flattened)
    var param_offset = combo_idx * 9
    var pm = param_combos[param_offset + 0]            # price_move threshold (negative)
    var window_idx = Int(param_combos[param_offset + 1])
    var dt = param_combos[param_offset + 2]            # delta_threshold (negative)
    var target_pct = param_combos[param_offset + 3]    # take profit %
    var stop_pct = param_combos[param_offset + 4]      # stop loss % (negative)
    var max_hold_ms = Int64(Int(param_combos[param_offset + 5]))
    var dca_mult = param_combos[param_offset + 6]      # DCA multiplier (e.g., 4.25)
    var max_pos_mult = param_combos[param_offset + 7]  # max_position / initial_position
    var dca_dist = param_combos[param_offset + 8]      # DCA distance % (negative for LONG)

    # Thread-local accumulators
    var local_entries: Int32 = 0
    var local_winners: Int32 = 0
    var local_pnl: Float32 = 0.0
    var last_entry_ts: Int64 = 0

    # Each lane processes with stride
    var i = START_IDX + lane * Int(outer_stride)
    while i < Int(n_prices):
        if timestamps[i] - last_entry_ts >= MIN_GAP:
            # Pre-computed rolling high for this window
            var high = rolling_highs[window_idx * Int(n_prices) + i]

            if high > 0:
                var drop = (prices[i] - high) / high * 100.0

                # LONG: price dropped AND selling pressure
                if drop <= pm and deltas[i] <= dt:
                    # === ENTRY ===
                    var entry_px = prices[i]
                    var entry_ts = timestamps[i]
                    var max_ts = entry_ts + max_hold_ms

                    # DCA state
                    var avg_entry = entry_px
                    var pos_size: Float64 = 1.0
                    var sl_active = False

                    var exit_px = entry_px
                    var won: Int32 = 0
                    var exited = False

                    # Scan forward for DCA triggers and exit
                    var scan_end = min(Int(n_prices), i + Int(max_scan))
                    for j in range(i + 1, scan_end):
                        var current_px = prices[j]
                        var current_ts = timestamps[j]

                        # Check timeout first
                        if current_ts > max_ts:
                            exit_px = current_px
                            exited = True
                            break

                        # Calculate move from avg_entry
                        var move_from_avg = (current_px - avg_entry) / avg_entry * 100.0

                        # === DCA CHECK ===
                        # LONG DCA: price drops below avg_entry by dca_dist% (negative)
                        if not sl_active and move_from_avg <= dca_dist:
                            # Can we DCA?
                            var new_size = pos_size * dca_mult
                            if pos_size + new_size <= max_pos_mult:
                                # DCA: update weighted average entry
                                avg_entry = (avg_entry * pos_size + current_px * new_size) / (pos_size + new_size)
                                pos_size += new_size
                            else:
                                # Can't DCA anymore - activate stop loss
                                sl_active = True

                        # === EXIT CHECKS (from avg_entry) ===
                        var target_px = avg_entry * (1.0 + target_pct / 100.0)
                        var stop_px = avg_entry * (1.0 + stop_pct / 100.0)  # stop_pct is negative

                        # Check TP
                        if current_px >= target_px:
                            exit_px = current_px
                            won = 1
                            exited = True
                            break

                        # Check SL (only if position is maxed)
                        if sl_active and current_px <= stop_px:
                            exit_px = current_px
                            exited = True
                            break

                    # If we didn't exit in the loop, use last scanned price
                    if not exited:
                        exit_px = prices[min(scan_end - 1, Int(n_prices) - 1)]

                    # LONG PnL: profit when price goes UP, weighted by position size
                    var pnl = Float32((exit_px - avg_entry) / avg_entry * 100.0 * pos_size)

                    local_entries += 1
                    local_winners += won
                    local_pnl += pnl
                    last_entry_ts = timestamps[i]

        i += WARP_SIZE * Int(outer_stride)

    # Warp-level reduction
    var total_entries = warp_reduce_sum_i32(local_entries)
    var total_winners = warp_reduce_sum_i32(local_winners)
    var total_pnl = warp_reduce_sum_f32(local_pnl)

    # Lane 0 writes result
    if lane == 0:
        out_entries[combo_idx] = total_entries
        out_winners[combo_idx] = total_winners
        out_pnl[combo_idx] = Float64(total_pnl)


fn grid_search_long_gpu(
    ctx: DeviceContext,
    timestamps_dev: DeviceBuffer[DType.int64],
    prices_dev: DeviceBuffer[DType.float64],
    deltas_dev: DeviceBuffer[DType.float64],
    rolling_highs_dev: DeviceBuffer[DType.float64],
    param_combos_dev: DeviceBuffer[DType.float64],
    out_entries_dev: DeviceBuffer[DType.int32],
    out_winners_dev: DeviceBuffer[DType.int32],
    out_pnl_dev: DeviceBuffer[DType.float64],
    n_combos: Int,
    n_windows: Int,
    n_prices: Int,
    max_scan: Int,
    outer_stride: Int,
) raises:
    """Launch grid search LONG GPU kernel.

    Args:
        ctx: GPU device context.
        timestamps_dev: Device buffer with timestamps.
        prices_dev: Device buffer with prices.
        deltas_dev: Device buffer with volume imbalance deltas.
        rolling_highs_dev: Device buffer with pre-computed rolling highs
                          (shape: n_windows * n_prices, flattened).
        param_combos_dev: Device buffer with parameter combinations
                         (shape: n_combos * 9, flattened).
        out_entries_dev: Device buffer for output entry counts.
        out_winners_dev: Device buffer for output winner counts.
        out_pnl_dev: Device buffer for output PnL sums.
        n_combos: Number of parameter combinations.
        n_windows: Number of time windows.
        n_prices: Number of prices/trades.
        max_scan: Maximum forward scan distance.
        outer_stride: Stride between lanes for processing.
    """
    var num_blocks = ceildiv(n_combos, WARPS_PER_BLOCK)

    ctx.enqueue_function_checked[_grid_search_long_gpu_kernel, _grid_search_long_gpu_kernel](
        timestamps_dev.unsafe_ptr(),
        prices_dev.unsafe_ptr(),
        deltas_dev.unsafe_ptr(),
        rolling_highs_dev.unsafe_ptr(),
        param_combos_dev.unsafe_ptr(),
        out_entries_dev.unsafe_ptr(),
        out_winners_dev.unsafe_ptr(),
        out_pnl_dev.unsafe_ptr(),
        Int32(n_combos),
        Int32(n_windows),
        Int32(n_prices),
        Int32(max_scan),
        Int32(outer_stride),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


# =============================================================================
# GPU Kernels - Grid Search SHORT
# =============================================================================


fn _grid_search_short_gpu_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    deltas: UnsafePointer[Float64, MutAnyOrigin],
    rolling_lows: UnsafePointer[Float64, MutAnyOrigin],
    param_combos: UnsafePointer[Float64, MutAnyOrigin],
    out_entries: UnsafePointer[Int32, MutAnyOrigin],
    out_winners: UnsafePointer[Int32, MutAnyOrigin],
    out_pnl: UnsafePointer[Float64, MutAnyOrigin],
    n_combos: Int32,
    n_windows: Int32,
    n_prices: Int32,
    max_scan: Int32,
    outer_stride: Int32,
):
    """GPU kernel for SHORT grid search with DCA.

    SHORT Strategy: price rises + buying pressure → sell expecting drop.

    DCA Logic:
    - Initial entry: pos_size = 1.0, avg_entry = entry_price
    - DCA trigger: price RISES by dca_distance_pct from avg_entry (positive)
    - If can DCA (pos_size * dca_mult + pos_size <= max_pos_mult):
        new_size = pos_size * dca_mult
        avg_entry = weighted average
        pos_size += new_size
    - SL only active when position is maxed (can't DCA anymore)
    - TP/SL calculated from avg_entry, not original entry

    WARPS_PER_BLOCK warps per block, 1 warp = 1 combo.
    Param format: [pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist].
    """
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())

    var combo_idx = Int(block_idx.x) * WARPS_PER_BLOCK + warp_id
    if combo_idx >= Int(n_combos):
        return

    # Extract parameters (9 params per combo, flattened)
    var param_offset = combo_idx * 9
    var pm = param_combos[param_offset + 0]            # price_move threshold (positive)
    var window_idx = Int(param_combos[param_offset + 1])
    var dt = param_combos[param_offset + 2]            # delta_threshold (positive)
    var target_pct = param_combos[param_offset + 3]    # take profit %
    var stop_pct = param_combos[param_offset + 4]      # stop loss % (negative)
    var max_hold_ms = Int64(Int(param_combos[param_offset + 5]))
    var dca_mult = param_combos[param_offset + 6]      # DCA multiplier (e.g., 4.25)
    var max_pos_mult = param_combos[param_offset + 7]  # max_position / initial_position
    var dca_dist = param_combos[param_offset + 8]      # DCA distance % (positive for SHORT)

    # Thread-local accumulators
    var local_entries: Int32 = 0
    var local_winners: Int32 = 0
    var local_pnl: Float32 = 0.0
    var last_entry_ts: Int64 = 0

    var i = START_IDX + lane * Int(outer_stride)
    while i < Int(n_prices):
        if timestamps[i] - last_entry_ts >= MIN_GAP:
            var low = rolling_lows[window_idx * Int(n_prices) + i]

            if low > 0 and low < NO_LOW_SENTINEL:
                var rise = (prices[i] - low) / low * 100.0

                # SHORT: price rose AND buying pressure
                if rise >= pm and deltas[i] >= dt:
                    # === ENTRY ===
                    var entry_px = prices[i]
                    var entry_ts = timestamps[i]
                    var max_ts = entry_ts + max_hold_ms

                    # DCA state
                    var avg_entry = entry_px
                    var pos_size: Float64 = 1.0
                    var sl_active = False

                    var exit_px = entry_px
                    var won: Int32 = 0
                    var exited = False

                    # Scan forward for DCA triggers and exit
                    var scan_end = min(Int(n_prices), i + Int(max_scan))
                    for j in range(i + 1, scan_end):
                        var current_px = prices[j]
                        var current_ts = timestamps[j]

                        # Check timeout first
                        if current_ts > max_ts:
                            exit_px = current_px
                            exited = True
                            break

                        # Calculate move from avg_entry
                        var move_from_avg = (current_px - avg_entry) / avg_entry * 100.0

                        # === DCA CHECK ===
                        # SHORT DCA: price rises above avg_entry by dca_dist% (positive)
                        if not sl_active and move_from_avg >= dca_dist:
                            # Can we DCA?
                            var new_size = pos_size * dca_mult
                            if pos_size + new_size <= max_pos_mult:
                                # DCA: update weighted average entry
                                avg_entry = (avg_entry * pos_size + current_px * new_size) / (pos_size + new_size)
                                pos_size += new_size
                            else:
                                # Can't DCA anymore - activate stop loss
                                sl_active = True

                        # === EXIT CHECKS (from avg_entry) ===
                        # SHORT: target is BELOW entry
                        var target_px = avg_entry * (1.0 - target_pct / 100.0)
                        # SHORT: stop is ABOVE entry (stop_pct is negative, so 1 - negative = higher)
                        var stop_px = avg_entry * (1.0 - stop_pct / 100.0)

                        # Check TP (price dropped below target)
                        if current_px <= target_px:
                            exit_px = current_px
                            won = 1
                            exited = True
                            break

                        # Check SL (only if position is maxed, price rose above stop)
                        if sl_active and current_px >= stop_px:
                            exit_px = current_px
                            exited = True
                            break

                    # If we didn't exit in the loop, use last scanned price
                    if not exited:
                        exit_px = prices[min(scan_end - 1, Int(n_prices) - 1)]

                    # SHORT PnL: profit when price goes DOWN, weighted by position size
                    var pnl = Float32((avg_entry - exit_px) / avg_entry * 100.0 * pos_size)

                    local_entries += 1
                    local_winners += won
                    local_pnl += pnl
                    last_entry_ts = timestamps[i]

        i += WARP_SIZE * Int(outer_stride)

    # Warp-level reduction
    var total_entries = warp_reduce_sum_i32(local_entries)
    var total_winners = warp_reduce_sum_i32(local_winners)
    var total_pnl = warp_reduce_sum_f32(local_pnl)

    # Lane 0 writes result
    if lane == 0:
        out_entries[combo_idx] = total_entries
        out_winners[combo_idx] = total_winners
        out_pnl[combo_idx] = Float64(total_pnl)


fn grid_search_short_gpu(
    ctx: DeviceContext,
    timestamps_dev: DeviceBuffer[DType.int64],
    prices_dev: DeviceBuffer[DType.float64],
    deltas_dev: DeviceBuffer[DType.float64],
    rolling_lows_dev: DeviceBuffer[DType.float64],
    param_combos_dev: DeviceBuffer[DType.float64],
    out_entries_dev: DeviceBuffer[DType.int32],
    out_winners_dev: DeviceBuffer[DType.int32],
    out_pnl_dev: DeviceBuffer[DType.float64],
    n_combos: Int,
    n_windows: Int,
    n_prices: Int,
    max_scan: Int,
    outer_stride: Int,
) raises:
    """Launch grid search SHORT GPU kernel.

    Args:
        ctx: GPU device context.
        timestamps_dev: Device buffer with timestamps.
        prices_dev: Device buffer with prices.
        deltas_dev: Device buffer with volume imbalance deltas.
        rolling_lows_dev: Device buffer with pre-computed rolling lows
                         (shape: n_windows * n_prices, flattened).
        param_combos_dev: Device buffer with parameter combinations
                         (shape: n_combos * 9, flattened).
        out_entries_dev: Device buffer for output entry counts.
        out_winners_dev: Device buffer for output winner counts.
        out_pnl_dev: Device buffer for output PnL sums.
        n_combos: Number of parameter combinations.
        n_windows: Number of time windows.
        n_prices: Number of prices/trades.
        max_scan: Maximum forward scan distance.
        outer_stride: Stride between lanes for processing.
    """
    var num_blocks = ceildiv(n_combos, WARPS_PER_BLOCK)

    ctx.enqueue_function_checked[_grid_search_short_gpu_kernel, _grid_search_short_gpu_kernel](
        timestamps_dev.unsafe_ptr(),
        prices_dev.unsafe_ptr(),
        deltas_dev.unsafe_ptr(),
        rolling_lows_dev.unsafe_ptr(),
        param_combos_dev.unsafe_ptr(),
        out_entries_dev.unsafe_ptr(),
        out_winners_dev.unsafe_ptr(),
        out_pnl_dev.unsafe_ptr(),
        Int32(n_combos),
        Int32(n_windows),
        Int32(n_prices),
        Int32(max_scan),
        Int32(outer_stride),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


# =============================================================================
# Exported Constants
# =============================================================================

# Export constants for use in other modules
fn get_block_size() -> Int:
    """Get the block size used for grid search kernels."""
    return BLOCK_SIZE


fn get_warps_per_block() -> Int:
    """Get the number of warps per block."""
    return WARPS_PER_BLOCK


fn get_warp_size() -> Int:
    """Get the warp size for the current architecture."""
    return WARP_SIZE


fn get_start_idx() -> Int:
    """Get the start index for processing (skip initial trades)."""
    return START_IDX


fn get_min_gap() -> Int64:
    """Get the minimum gap between entries in milliseconds."""
    return MIN_GAP
