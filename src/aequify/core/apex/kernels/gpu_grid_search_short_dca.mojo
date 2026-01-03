"""
Grid Search SHORT DCA Kernel - GPU-accelerated parameter optimization with DCA support.

Entry signal for SHORT (shorting the spike):
    1. Price rises from rolling low by >= price_move_pct
    2. Volume delta over the SAME window shows strong buying pressure (>= +delta_threshold)
       This means heavy buying occurred during the price spike period.

DCA sizing:
    - Initial: 6.25 USDT
    - DCA 1: 6.25 * 4.25 = 26.5625 USDT
    - DCA 2+: previous_total * 4.25 (capped at MAX_POSITION)

Exit conditions:
    1. Target profit reached (price dropped by target_pct from avg_entry)
    2. Stop loss hit (price rose by stop_pct from avg_entry)
    3. Max hold time reached AND position is in profit (forced profitable exit only)

Param format: [pm, window_idx, dt, target_pct, stop_pct, max_hold_ms, dca_distance_pct]

Usage:
    grid_search_short_dca_gpu(ctx, timestamps, prices, rolling_lows, volume_deltas,
                              param_combos, out_entries, out_winners, out_pnl,
                              out_dca_count, n_combos, n_windows, n, max_scan, outer_stride)
"""

from math import ceildiv

from gpu import global_idx, block_idx, thread_idx, lane_id
from gpu.host import DeviceContext, DeviceBuffer
from gpu.primitives.warp import lane_group_sum
from memory import UnsafePointer


# =============================================================================
# Architecture-Adaptive Constants
# =============================================================================

comptime HW_INFO = DeviceContext.default_device_info
comptime WARP_SIZE: Int = HW_INFO.warp_size
comptime MAX_BLOCK_SIZE: Int = HW_INFO.max_thread_block_size
comptime BLOCK_SIZE: Int = 256 if MAX_BLOCK_SIZE >= 256 else MAX_BLOCK_SIZE
comptime WARPS_PER_BLOCK: Int = BLOCK_SIZE // WARP_SIZE
comptime SM_COUNT: Int = HW_INFO.sm_count


# =============================================================================
# Application Constants
# =============================================================================

comptime MIN_GAP_MS: Int = 2000
comptime START_IDX: Int = 500
comptime NO_LOW_SENTINEL: Float64 = 999999999.0
comptime INITIAL_SIZE: Float64 = 6.25
comptime DCA_MULTIPLIER: Float64 = 4.25
comptime MAX_POSITION: Float64 = 1000.0
comptime PARAMS_PER_COMBO: Int = 7


# =============================================================================
# Warp Reduction Helpers
# =============================================================================


@always_inline
fn warp_reduce_sum_i32(val: Int32) -> Int32:
    var val_f32 = Float32(Int(val))
    var result = lane_group_sum[num_lanes=WARP_SIZE](val_f32)
    return Int32(Int(result))


@always_inline
fn warp_reduce_sum_f64(val: Float64) -> Float64:
    var val_f32 = Float32(val)
    var result = lane_group_sum[num_lanes=WARP_SIZE](val_f32)
    return Float64(result)


# =============================================================================
# Grid Search SHORT DCA Kernel
# =============================================================================


fn _grid_search_short_dca_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    rolling_lows: UnsafePointer[Float64, MutAnyOrigin],
    volume_deltas: UnsafePointer[Float64, MutAnyOrigin],
    param_combos: UnsafePointer[Float64, MutAnyOrigin],
    out_entries: UnsafePointer[Int32, MutAnyOrigin],
    out_winners: UnsafePointer[Int32, MutAnyOrigin],
    out_pnl: UnsafePointer[Float64, MutAnyOrigin],
    out_dca_count: UnsafePointer[Int32, MutAnyOrigin],
    n_combos: Int32,
    n_windows: Int32,
    n_prices: Int32,
    max_scan: Int32,
    outer_stride: Int32,
):
    """Grid search SHORT DCA kernel.

    Entry Signal (SHORT - shorting the spike with strong buying pressure):
        1. Price rose from rolling low: (price - low) / low * 100 >= pm
        2. Volume delta over SAME window is strongly positive: delta >= +dt
           (heavy buying during the spike = potential overbought reversal)

    Param combo layout (7 params):
        [0] pm: Price move threshold (positive, e.g., 0.5 means 0.5% rise required)
        [1] window_idx: Index into rolling_lows/volume_deltas windows
        [2] dt: Delta threshold (positive, e.g., 20 means delta >= +20% required)
        [3] target_pct: Take profit % (positive)
        [4] stop_pct: Stop loss % (positive)
        [5] max_hold_ms: Maximum hold time in milliseconds
        [6] dca_distance_pct: DCA trigger distance (positive)
    """
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var combo_idx = Int(block_idx.x) * WARPS_PER_BLOCK + warp_id

    if combo_idx >= Int(n_combos):
        return

    # Extract parameters
    var param_base = combo_idx * PARAMS_PER_COMBO
    var pm = param_combos[param_base + 0]
    var window_idx = Int(param_combos[param_base + 1])
    var dt = param_combos[param_base + 2]
    var target_pct = param_combos[param_base + 3]
    var stop_pct = param_combos[param_base + 4]
    var max_hold_ms = Int64(param_combos[param_base + 5])
    var dca_distance_pct = param_combos[param_base + 6]

    var local_entries: Int32 = 0
    var local_winners: Int32 = 0
    var local_pnl: Float64 = 0.0
    var local_dca_count: Int32 = 0

    var last_entry_ts: Int64 = 0
    var n = Int(n_prices)
    var stride = WARP_SIZE * Int(outer_stride)

    var i = START_IDX + lane * Int(outer_stride)
    while i < n:
        if timestamps[i] - last_entry_ts < MIN_GAP_MS:
            i += stride
            continue

        var data_idx = window_idx * n + i
        var low = rolling_lows[data_idx]
        var delta = volume_deltas[data_idx]

        if low <= 0.0 or low >= NO_LOW_SENTINEL:
            i += stride
            continue

        # Price rise from rolling low
        var rise_pct = (prices[i] - low) / low * 100.0

        # SHORT entry:
        # 1. Price rose by at least pm%: rise_pct >= pm
        # 2. Strong buying pressure during spike: delta >= +dt
        if rise_pct >= pm and delta >= dt:
            var entry_px = prices[i]
            var entry_ts = timestamps[i]

            var total_size = INITIAL_SIZE
            var avg_entry = entry_px
            var dca_entries: Int32 = 0
            var last_dca_ts = entry_ts
            var max_ts = entry_ts + max_hold_ms

            var exit_px = entry_px
            var won: Int32 = 0
            var exited = False

            var scan_end = min(n, i + Int(max_scan))
            for j in range(i + 1, scan_end):
                var current_px = prices[j]
                var current_ts = timestamps[j]

                var target_px = avg_entry * (1.0 - target_pct / 100.0)  # Target below entry
                var stop_px = avg_entry * (1.0 + stop_pct / 100.0)      # Stop above entry

                # Target hit (price dropped for SHORT)
                if current_px <= target_px:
                    exit_px = current_px
                    won = 1
                    exited = True
                    break

                # Stop hit (price rose more)
                if current_px >= stop_px:
                    exit_px = current_px
                    exited = True
                    break

                # Max hold - only exit if in profit (SHORT profits when price drops)
                if current_ts > max_ts:
                    if current_px < avg_entry:
                        exit_px = current_px
                        won = 1
                        exited = True
                        break

                # DCA check
                if total_size < MAX_POSITION and current_ts - last_dca_ts >= MIN_GAP_MS:
                    var price_distance = (current_px - avg_entry) / avg_entry * 100.0

                    # SHORT DCA: price must have risen further
                    if price_distance >= dca_distance_pct:
                        var data_idx_j = window_idx * n + j
                        var low_j = rolling_lows[data_idx_j]
                        var delta_j = volume_deltas[data_idx_j]

                        if low_j > 0.0 and low_j < NO_LOW_SENTINEL:
                            var rise_j = (current_px - low_j) / low_j * 100.0

                            if rise_j >= pm and delta_j >= dt:
                                var dca_size: Float64
                                if dca_entries == 0:
                                    dca_size = INITIAL_SIZE * DCA_MULTIPLIER
                                else:
                                    dca_size = total_size * DCA_MULTIPLIER

                                dca_size = min(dca_size, MAX_POSITION - total_size)

                                if dca_size > 0.0:
                                    var new_total = total_size + dca_size
                                    avg_entry = (avg_entry * total_size + current_px * dca_size) / new_total
                                    total_size = new_total
                                    dca_entries += 1
                                    last_dca_ts = current_ts

            if not exited:
                exit_px = prices[min(scan_end - 1, n - 1)]
                if exit_px < avg_entry:
                    won = 1

            # SHORT PnL: profit when exit < entry
            var pnl = (avg_entry - exit_px) / avg_entry * 100.0

            local_entries += 1
            local_winners += won
            local_pnl += pnl
            local_dca_count += dca_entries

            last_entry_ts = timestamps[i]

        i += stride

    # Warp reduction
    var total_entries = warp_reduce_sum_i32(local_entries)
    var total_winners = warp_reduce_sum_i32(local_winners)
    var total_pnl = warp_reduce_sum_f64(local_pnl)
    var total_dca = warp_reduce_sum_i32(local_dca_count)

    if lane == 0:
        out_entries[combo_idx] = total_entries
        out_winners[combo_idx] = total_winners
        out_pnl[combo_idx] = total_pnl
        out_dca_count[combo_idx] = total_dca


# =============================================================================
# Main GPU Function
# =============================================================================


fn grid_search_short_dca_gpu(
    ctx: DeviceContext,
    timestamps_dev: DeviceBuffer[DType.int64],
    prices_dev: DeviceBuffer[DType.float64],
    rolling_lows_dev: DeviceBuffer[DType.float64],
    volume_deltas_dev: DeviceBuffer[DType.float64],
    param_combos_dev: DeviceBuffer[DType.float64],
    out_entries_dev: DeviceBuffer[DType.int32],
    out_winners_dev: DeviceBuffer[DType.int32],
    out_pnl_dev: DeviceBuffer[DType.float64],
    out_dca_count_dev: DeviceBuffer[DType.int32],
    n_combos: Int,
    n_windows: Int,
    n: Int,
    max_scan: Int,
    outer_stride: Int,
) raises:
    """Run SHORT DCA grid search on GPU.

    Args:
        ctx: GPU device context.
        timestamps_dev: Device buffer with timestamps.
        prices_dev: Device buffer with prices.
        rolling_lows_dev: Device buffer with rolling lows (n_windows * n).
        volume_deltas_dev: Device buffer with volume deltas (n_windows * n, SAME windows).
        param_combos_dev: Device buffer with param combos (n_combos * 7).
        out_entries_dev: Output buffer for entry counts.
        out_winners_dev: Output buffer for winner counts.
        out_pnl_dev: Output buffer for PnL sums.
        out_dca_count_dev: Output buffer for DCA entry counts.
        n_combos: Number of parameter combinations.
        n_windows: Number of time windows.
        n: Number of price points.
        max_scan: Maximum trades to scan after entry.
        outer_stride: Stride for entry signal checking.
    """
    var num_blocks = ceildiv(n_combos, WARPS_PER_BLOCK)

    ctx.enqueue_function_checked[_grid_search_short_dca_kernel, _grid_search_short_dca_kernel](
        timestamps_dev.unsafe_ptr(),
        prices_dev.unsafe_ptr(),
        rolling_lows_dev.unsafe_ptr(),
        volume_deltas_dev.unsafe_ptr(),
        param_combos_dev.unsafe_ptr(),
        out_entries_dev.unsafe_ptr(),
        out_winners_dev.unsafe_ptr(),
        out_pnl_dev.unsafe_ptr(),
        out_dca_count_dev.unsafe_ptr(),
        Int32(n_combos),
        Int32(n_windows),
        Int32(n),
        Int32(max_scan),
        Int32(outer_stride),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


# =============================================================================
# Constants Getters
# =============================================================================


fn get_initial_size() -> Float64:
    return INITIAL_SIZE


fn get_dca_multiplier() -> Float64:
    return DCA_MULTIPLIER


fn get_max_position() -> Float64:
    return MAX_POSITION


fn get_params_per_combo() -> Int:
    return PARAMS_PER_COMBO
