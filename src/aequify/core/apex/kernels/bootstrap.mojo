"""
APEX Bootstrap - GPU-accelerated parameter optimization pipeline.

Full GPU pipeline: numpy → GPU → compute → GPU → numpy

Orchestrates:
- volume_imbalance: Price-based with session lookback
- rolling_high/low: O(log n) binary search + SIMD vectorization
- grid_search_long/short: Warp-per-combo with DCA support

Architecture-adaptive constants for NVIDIA/AMD/Apple GPUs.
"""

from math import ceildiv, log2
from sys import has_accelerator

from gpu import thread_idx, block_idx, block_dim, global_idx, barrier
from gpu.host import DeviceContext
from gpu.warp import shuffle_down, WARP_SIZE
from memory import UnsafePointer, memcpy
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from python._cpython import GILReleased


# =============================================================================
# ARCHITECTURE-ADAPTIVE CONSTANTS (from device info)
# =============================================================================

# Get hardware info at compile time
comptime HW_INFO = DeviceContext.default_device_info

# Block size from device - max threads per block (typically 1024)
# We use 256 as optimal balance between occupancy and register pressure
comptime MAX_BLOCK_SIZE: Int = HW_INFO.max_thread_block_size
comptime BLOCK_SIZE: Int = 256 if MAX_BLOCK_SIZE >= 256 else MAX_BLOCK_SIZE

# Warps per block - computed from architecture
comptime WARPS_PER_BLOCK: Int = BLOCK_SIZE // WARP_SIZE

# SM count for grid sizing
comptime SM_COUNT: Int = HW_INFO.sm_count

# Shared memory per SM from architecture (bytes)
comptime SHARED_MEM_PER_SM: Int = HW_INFO.shared_memory_per_multiprocessor

# SIMD vector width (128 bits for modern GPUs)
# Float64: 128 / 64 = 2 elements per vector load
comptime GPU_SIMD_BIT_WIDTH: Int = 128
comptime SIMD_WIDTH_F64: Int = GPU_SIMD_BIT_WIDTH // 64  # = 2
comptime SIMD_WIDTH_F32: Int = GPU_SIMD_BIT_WIDTH // 32  # = 4


# =============================================================================
# APPLICATION CONSTANTS
# =============================================================================

# Minimum trades needed for statistically valid parameter evaluation
comptime MIN_ENTRIES_FOR_VALIDITY: Int = 10

# Minimum milliseconds between consecutive entries (5 seconds)
comptime MIN_GAP: Int = 5000

# Skip first N trades - need enough history for rolling indicators
comptime START_IDX: Int = 1000

# Sentinel value for "no low found"
comptime NO_LOW_SENTINEL: Float64 = 1e18

# Session boundaries in UTC hours
comptime MS_PER_HOUR: Int64 = 3600000
comptime MS_PER_DAY: Int64 = 86400000


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


@always_inline
fn binary_search_lower_bound(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    start: Int,
    end: Int,
    target: Int64,
) -> Int:
    """Binary search to find first index where timestamps[idx] >= target."""
    var lo = start
    var hi = end

    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if timestamps[mid] < target:
            lo = mid + 1
        else:
            hi = mid

    return lo


fn get_previous_session_start_ms(current_ts_ms: Int64) -> Int64:
    """Get the start timestamp of the previous trading session."""
    var ms_since_midnight = current_ts_ms % MS_PER_DAY
    var current_hour = Int32(ms_since_midnight // MS_PER_HOUR)
    var day_start_ms = current_ts_ms - ms_since_midnight

    var prev_session_hour: Int32 = 17
    var prev_day_offset: Int64 = 0

    if current_hour >= 20:
        prev_session_hour = 17
    elif current_hour >= 17:
        prev_session_hour = 13
    elif current_hour >= 13:
        prev_session_hour = 8
    elif current_hour >= 8:
        prev_session_hour = 7
    elif current_hour >= 7:
        prev_session_hour = 3
    elif current_hour >= 3:
        prev_session_hour = 0
    elif current_hour >= 0:
        prev_session_hour = 20
        prev_day_offset = -MS_PER_DAY

    return day_start_ms + prev_day_offset + Int64(prev_session_hour) * MS_PER_HOUR


# =============================================================================
# WARP REDUCTION HELPERS
# =============================================================================


@always_inline
fn warp_reduce_sum_i32(val: Int32) -> Int32:
    """Warp-level sum reduction using shuffle."""
    var result = val

    @parameter
    if WARP_SIZE == 64:
        result = result + shuffle_down(result, 32)
    result = result + shuffle_down(result, 16)
    result = result + shuffle_down(result, 8)
    result = result + shuffle_down(result, 4)
    result = result + shuffle_down(result, 2)
    result = result + shuffle_down(result, 1)
    return result


@always_inline
fn warp_reduce_sum_f32(val: Float32) -> Float32:
    """Warp-level sum reduction using shuffle (Float32)."""
    var result = val

    @parameter
    if WARP_SIZE == 64:
        result = result + shuffle_down(result, 32)
    result = result + shuffle_down(result, 16)
    result = result + shuffle_down(result, 8)
    result = result + shuffle_down(result, 4)
    result = result + shuffle_down(result, 2)
    result = result + shuffle_down(result, 1)
    return result


# =============================================================================
# GPU KERNELS - VOLUME IMBALANCE
# =============================================================================


fn volume_imbalance_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    quantities: UnsafePointer[Float64, MutAnyOrigin],
    sides: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    price_tolerance_pct: Float64,
    n: Int32,
):
    """Compute volume imbalance at price level with session-synchronized lookback."""
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    var current_ts = timestamps[i]
    var current_price = prices[i]
    var tolerance = current_price * (price_tolerance_pct / 100.0)
    var price_low = current_price - tolerance
    var price_high = current_price + tolerance

    var session_start_ms = get_previous_session_start_ms(current_ts)

    # Binary search for session start
    var lo = 0
    var hi = i
    while lo < hi:
        var mid = (lo + hi) // 2
        if timestamps[mid] < session_start_ms:
            lo = mid + 1
        else:
            hi = mid
    var start_idx = lo

    var buy_sum: Float64 = 0.0
    var sell_sum: Float64 = 0.0

    var j = start_idx
    var end_aligned = start_idx + ((i - start_idx) // 2) * 2

    while j < end_aligned:
        var price0 = prices[j]
        var price1 = prices[j + 1]
        var qty0 = quantities[j]
        var qty1 = quantities[j + 1]
        var side0 = sides[j]
        var side1 = sides[j + 1]

        if price0 >= price_low and price0 <= price_high:
            if side0 == 1:
                buy_sum += qty0
            else:
                sell_sum += qty0

        if price1 >= price_low and price1 <= price_high:
            if side1 == 1:
                buy_sum += qty1
            else:
                sell_sum += qty1

        j += 2

    while j < i:
        var price_j = prices[j]
        if price_j >= price_low and price_j <= price_high:
            if sides[j] == 1:
                buy_sum += quantities[j]
            else:
                sell_sum += quantities[j]
        j += 1

    var total = buy_sum + sell_sum
    if total > 0:
        result[i] = (buy_sum - sell_sum) / total * 100.0
    else:
        result[i] = 0.0


# =============================================================================
# GPU KERNELS - ROLLING EXTREMA
# =============================================================================


fn rolling_high_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    lookback_ms: Int64,
    n: Int32,
):
    """Compute rolling maximum price within time window."""
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    var current_ts = timestamps[i]
    var cutoff_ts = current_ts - lookback_ms

    var start_idx = binary_search_lower_bound(timestamps, 0, i, cutoff_ts)

    if start_idx >= i:
        result[i] = 0.0
        return

    var max_val = prices[start_idx]
    var window_size = i - start_idx - 1

    var j = start_idx + 1
    var end_aligned = start_idx + 1 + (window_size // SIMD_WIDTH_F64) * SIMD_WIDTH_F64

    while j < end_aligned:
        var vec = prices.load[width=SIMD_WIDTH_F64](j)
        var local_max = vec.reduce_max()
        if local_max > max_val:
            max_val = local_max
        j += SIMD_WIDTH_F64

    while j < i:
        if prices[j] > max_val:
            max_val = prices[j]
        j += 1

    result[i] = max_val


fn rolling_low_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    lookback_ms: Int64,
    n: Int32,
):
    """Compute rolling minimum price within time window."""
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    var current_ts = timestamps[i]
    var cutoff_ts = current_ts - lookback_ms

    var start_idx = binary_search_lower_bound(timestamps, 0, i, cutoff_ts)

    if start_idx >= i:
        result[i] = NO_LOW_SENTINEL
        return

    var min_val = prices[start_idx]
    var window_size = i - start_idx - 1

    var j = start_idx + 1
    var end_aligned = start_idx + 1 + (window_size // SIMD_WIDTH_F64) * SIMD_WIDTH_F64

    while j < end_aligned:
        var vec = prices.load[width=SIMD_WIDTH_F64](j)
        var local_min = vec.reduce_min()
        if local_min < min_val:
            min_val = local_min
        j += SIMD_WIDTH_F64

    while j < i:
        if prices[j] < min_val:
            min_val = prices[j]
        j += 1

    result[i] = min_val


# =============================================================================
# GPU KERNELS - GRID SEARCH LONG (with DCA)
# =============================================================================


fn grid_search_long_kernel(
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
    """
    LONG grid search with DCA: price drops + selling pressure → buy expecting bounce.

    Param format: [pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist].
    """
    var warp_id = Int(thread_idx.x) // Int(WARP_SIZE)
    var lane_id = Int(thread_idx.x) % Int(WARP_SIZE)

    var combo_idx = Int(block_idx.x) * WARPS_PER_BLOCK + warp_id
    if combo_idx >= Int(n_combos):
        return

    # Extract parameters (9 params per combo)
    var param_offset = combo_idx * 9
    var pm = param_combos[param_offset + 0]
    var window_idx = Int(param_combos[param_offset + 1])
    var dt = param_combos[param_offset + 2]
    var target_pct = param_combos[param_offset + 3]
    var stop_pct = param_combos[param_offset + 4]
    var max_hold_ms = Int64(Int(param_combos[param_offset + 5]))
    var dca_mult = param_combos[param_offset + 6]
    var max_pos_mult = param_combos[param_offset + 7]
    var dca_dist = param_combos[param_offset + 8]

    var local_entries: Int32 = 0
    var local_winners: Int32 = 0
    var local_pnl: Float32 = 0.0
    var last_entry_ts: Int64 = 0

    var i = START_IDX + lane_id * Int(outer_stride)
    while i < Int(n_prices):
        if timestamps[i] - last_entry_ts >= MIN_GAP:
            var high = rolling_highs[window_idx * Int(n_prices) + i]

            if high > 0:
                var drop = (prices[i] - high) / high * 100.0

                if drop <= pm and deltas[i] <= dt:
                    var entry_px = prices[i]
                    var entry_ts = timestamps[i]
                    var max_ts = entry_ts + max_hold_ms

                    var avg_entry = entry_px
                    var pos_size: Float64 = 1.0
                    var sl_active = False

                    var exit_px = entry_px
                    var won: Int32 = 0
                    var exited = False

                    var scan_end = min(Int(n_prices), i + Int(max_scan))
                    for j in range(i + 1, scan_end):
                        var current_px = prices[j]
                        var current_ts = timestamps[j]

                        if current_ts > max_ts:
                            exit_px = current_px
                            exited = True
                            break

                        var move_from_avg = (current_px - avg_entry) / avg_entry * 100.0

                        if not sl_active and move_from_avg <= dca_dist:
                            var new_size = pos_size * dca_mult
                            if pos_size + new_size <= max_pos_mult:
                                avg_entry = (avg_entry * pos_size + current_px * new_size) / (pos_size + new_size)
                                pos_size += new_size
                            else:
                                sl_active = True

                        var target_px = avg_entry * (1.0 + target_pct / 100.0)
                        var stop_px = avg_entry * (1.0 + stop_pct / 100.0)

                        if current_px >= target_px:
                            exit_px = current_px
                            won = 1
                            exited = True
                            break

                        if sl_active and current_px <= stop_px:
                            exit_px = current_px
                            exited = True
                            break

                    if not exited:
                        exit_px = prices[min(scan_end - 1, Int(n_prices) - 1)]

                    var pnl = Float32((exit_px - avg_entry) / avg_entry * 100.0 * pos_size)

                    local_entries += 1
                    local_winners += won
                    local_pnl += pnl
                    last_entry_ts = timestamps[i]

        i += Int(WARP_SIZE) * Int(outer_stride)

    var total_entries = warp_reduce_sum_i32(local_entries)
    var total_winners = warp_reduce_sum_i32(local_winners)
    var total_pnl = warp_reduce_sum_f32(local_pnl)

    if lane_id == 0:
        out_entries[combo_idx] = total_entries
        out_winners[combo_idx] = total_winners
        out_pnl[combo_idx] = Float64(total_pnl)


# =============================================================================
# GPU KERNELS - GRID SEARCH SHORT (with DCA)
# =============================================================================


fn grid_search_short_kernel(
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
    """
    SHORT grid search with DCA: price rises + buying pressure → sell expecting drop.

    Param format: [pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist].
    """
    var warp_id = Int(thread_idx.x) // Int(WARP_SIZE)
    var lane_id = Int(thread_idx.x) % Int(WARP_SIZE)

    var combo_idx = Int(block_idx.x) * WARPS_PER_BLOCK + warp_id
    if combo_idx >= Int(n_combos):
        return

    var param_offset = combo_idx * 9
    var pm = param_combos[param_offset + 0]
    var window_idx = Int(param_combos[param_offset + 1])
    var dt = param_combos[param_offset + 2]
    var target_pct = param_combos[param_offset + 3]
    var stop_pct = param_combos[param_offset + 4]
    var max_hold_ms = Int64(Int(param_combos[param_offset + 5]))
    var dca_mult = param_combos[param_offset + 6]
    var max_pos_mult = param_combos[param_offset + 7]
    var dca_dist = param_combos[param_offset + 8]

    var local_entries: Int32 = 0
    var local_winners: Int32 = 0
    var local_pnl: Float32 = 0.0
    var last_entry_ts: Int64 = 0

    var i = START_IDX + lane_id * Int(outer_stride)
    while i < Int(n_prices):
        if timestamps[i] - last_entry_ts >= MIN_GAP:
            var low = rolling_lows[window_idx * Int(n_prices) + i]

            if low > 0 and low < NO_LOW_SENTINEL:
                var rise = (prices[i] - low) / low * 100.0

                if rise >= pm and deltas[i] >= dt:
                    var entry_px = prices[i]
                    var entry_ts = timestamps[i]
                    var max_ts = entry_ts + max_hold_ms

                    var avg_entry = entry_px
                    var pos_size: Float64 = 1.0
                    var sl_active = False

                    var exit_px = entry_px
                    var won: Int32 = 0
                    var exited = False

                    var scan_end = min(Int(n_prices), i + Int(max_scan))
                    for j in range(i + 1, scan_end):
                        var current_px = prices[j]
                        var current_ts = timestamps[j]

                        if current_ts > max_ts:
                            exit_px = current_px
                            exited = True
                            break

                        var move_from_avg = (current_px - avg_entry) / avg_entry * 100.0

                        if not sl_active and move_from_avg >= dca_dist:
                            var new_size = pos_size * dca_mult
                            if pos_size + new_size <= max_pos_mult:
                                avg_entry = (avg_entry * pos_size + current_px * new_size) / (pos_size + new_size)
                                pos_size += new_size
                            else:
                                sl_active = True

                        var target_px = avg_entry * (1.0 - target_pct / 100.0)
                        var stop_px = avg_entry * (1.0 - stop_pct / 100.0)

                        if current_px <= target_px:
                            exit_px = current_px
                            won = 1
                            exited = True
                            break

                        if sl_active and current_px >= stop_px:
                            exit_px = current_px
                            exited = True
                            break

                    if not exited:
                        exit_px = prices[min(scan_end - 1, Int(n_prices) - 1)]

                    var pnl = Float32((avg_entry - exit_px) / avg_entry * 100.0 * pos_size)

                    local_entries += 1
                    local_winners += won
                    local_pnl += pnl
                    last_entry_ts = timestamps[i]

        i += Int(WARP_SIZE) * Int(outer_stride)

    var total_entries = warp_reduce_sum_i32(local_entries)
    var total_winners = warp_reduce_sum_i32(local_winners)
    var total_pnl = warp_reduce_sum_f32(local_pnl)

    if lane_id == 0:
        out_entries[combo_idx] = total_entries
        out_winners[combo_idx] = total_winners
        out_pnl[combo_idx] = Float64(total_pnl)


# =============================================================================
# RESULT SELECTION
# =============================================================================


fn select_best_params(
    direction: String,
    param_combos: PythonObject,
    entries: PythonObject,
    winners: PythonObject,
    pnl_sums: PythonObject,
    min_entries: Int,
    time_windows_ms: PythonObject,
) raises -> PythonObject:
    """Select best parameters from grid search results."""
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var n_combos = Int(param_combos.shape[0])

    var best_loss: Float64 = 1e30
    var best_result: PythonObject = Python.none()

    for i in range(n_combos):
        var n_entries = Int(entries[i])
        var n_winners = Int(winners[i])

        if n_entries < min_entries:
            continue

        var wr = Float64(n_winners) / Float64(n_entries)
        var avg_pnl = Float64(pnl_sums[i]) / Float64(n_entries)

        var sample_bonus = min(Float64(n_entries) / 50.0, 1.0) * 5.0
        var loss = (1.0 - wr) * 20.0 - avg_pnl * 8.0 - sample_bonus

        if loss < best_loss:
            best_loss = loss

            var pm = param_combos[i, 0]
            var w_idx = Int(param_combos[i, 1])
            var dt = param_combos[i, 2]
            var tp = param_combos[i, 3]
            var sl = param_combos[i, 4]
            var mh = Int(param_combos[i, 5])
            var dca_mult = param_combos[i, 6]
            var max_pos_mult = param_combos[i, 7]
            var dca_dist = param_combos[i, 8]
            var tw_ms = time_windows_ms[w_idx]

            var params_dict = builtins.dict()
            params_dict[PythonObject("price_move")] = pm
            params_dict[PythonObject("time_window")] = tw_ms
            params_dict[PythonObject("delta_threshold")] = dt
            params_dict[PythonObject("target_profit")] = tp
            params_dict[PythonObject("stop_loss")] = sl
            params_dict[PythonObject("max_hold_time")] = PythonObject(mh)
            params_dict[PythonObject("dca_distance_pct")] = dca_dist

            var dca_dict = builtins.dict()
            dca_dict[PythonObject("multiplier")] = dca_mult
            dca_dict[PythonObject("max_position_mult")] = max_pos_mult

            best_result = builtins.dict()
            best_result[PythonObject("direction")] = PythonObject(direction)
            best_result[PythonObject("params")] = params_dict
            best_result[PythonObject("dca")] = dca_dict
            best_result[PythonObject("entries")] = PythonObject(n_entries)
            best_result[PythonObject("winners")] = PythonObject(n_winners)
            best_result[PythonObject("win_rate")] = PythonObject(wr)
            best_result[PythonObject("avg_pnl")] = PythonObject(avg_pnl)
            best_result[PythonObject("loss")] = PythonObject(loss)

    return best_result


# =============================================================================
# BOOTSTRAP INTERNAL - FULL GPU PIPELINE
# =============================================================================


fn bootstrap_internal(
    timestamps: PythonObject,
    prices: PythonObject,
    quantities: PythonObject,
    sides: PythonObject,
    time_windows_ms: PythonObject,
    long_param_grid: PythonObject,
    short_param_grid: PythonObject,
    imbalance_price_tolerance_pct: Float64,
    max_scan: Int,
    outer_stride: Int,
    min_entries: Int,
) raises -> PythonObject:
    """
    Full GPU-accelerated bootstrap pipeline.

    ALL computation runs on GPU. Only data transfer uses CPU.
    """
    var py = Python()
    var np = Python.import_module("numpy")
    var time_module = Python.import_module("time")

    var n = Int(timestamps.shape[0])
    var n_windows = Int(len(time_windows_ms))
    var n_long = Int(long_param_grid.shape[0])
    var n_short = Int(short_param_grid.shape[0])

    var total_start = time_module.perf_counter()

    # Ensure arrays are contiguous and correct dtype
    var ts_arr = np.ascontiguousarray(timestamps, dtype=np.int64)
    var px_arr = np.ascontiguousarray(prices, dtype=np.float64)
    var qty_arr = np.ascontiguousarray(quantities, dtype=np.float64)
    var sd_arr = np.ascontiguousarray(sides, dtype=np.int32)
    var long_params = np.ascontiguousarray(long_param_grid.flatten(), dtype=np.float64)
    var short_params = np.ascontiguousarray(short_param_grid.flatten(), dtype=np.float64)

    var ctx = DeviceContext()

    # STEP 1: Copy input data to GPU
    var ts_host = ctx.enqueue_create_host_buffer[DType.int64](n)
    var px_host = ctx.enqueue_create_host_buffer[DType.float64](n)
    var qty_host = ctx.enqueue_create_host_buffer[DType.float64](n)
    var sd_host = ctx.enqueue_create_host_buffer[DType.int32](n)

    var ts_addr = Int(ts_arr.ctypes.data)
    var px_addr = Int(px_arr.ctypes.data)
    var qty_addr = Int(qty_arr.ctypes.data)
    var sd_addr = Int(sd_arr.ctypes.data)

    var windows = List[Int64]()
    for w_idx in range(n_windows):
        windows.append(Int64(Int(time_windows_ms[w_idx])))

    with GILReleased(py):
        ctx.synchronize()

        memcpy(
            dest=ts_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=ts_addr),
            count=n * 8,
        )
        memcpy(
            dest=px_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=px_addr),
            count=n * 8,
        )
        memcpy(
            dest=qty_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=qty_addr),
            count=n * 8,
        )
        memcpy(
            dest=sd_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=sd_addr),
            count=n * 4,
        )

    var ts_dev = ctx.enqueue_create_buffer[DType.int64](n)
    var px_dev = ctx.enqueue_create_buffer[DType.float64](n)
    var qty_dev = ctx.enqueue_create_buffer[DType.float64](n)
    var sd_dev = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=ts_dev, src_buf=ts_host)
    ctx.enqueue_copy(dst_buf=px_dev, src_buf=px_host)
    ctx.enqueue_copy(dst_buf=qty_dev, src_buf=qty_host)
    ctx.enqueue_copy(dst_buf=sd_dev, src_buf=sd_host)

    # STEP 2: Compute volume imbalance on GPU
    var imbalance_dev = ctx.enqueue_create_buffer[DType.float64](n)
    var imbalance_blocks = ceildiv(n, BLOCK_SIZE)

    ctx.enqueue_function_checked[volume_imbalance_kernel, volume_imbalance_kernel](
        ts_dev.unsafe_ptr(),
        px_dev.unsafe_ptr(),
        qty_dev.unsafe_ptr(),
        sd_dev.unsafe_ptr(),
        imbalance_dev.unsafe_ptr(),
        imbalance_price_tolerance_pct,
        Int32(n),
        grid_dim=imbalance_blocks, block_dim=BLOCK_SIZE
    )

    # STEP 3: Compute rolling highs/lows on GPU
    var rolling_size = n_windows * n
    var rh_dev = ctx.enqueue_create_buffer[DType.float64](rolling_size)
    var rl_dev = ctx.enqueue_create_buffer[DType.float64](rolling_size)

    for w_idx in range(n_windows):
        var window_ms = windows[w_idx]
        var offset = w_idx * n

        ctx.enqueue_function_checked[rolling_high_kernel, rolling_high_kernel](
            ts_dev.unsafe_ptr(),
            px_dev.unsafe_ptr(),
            rh_dev.unsafe_ptr().offset(offset),
            window_ms,
            Int32(n),
            grid_dim=imbalance_blocks, block_dim=BLOCK_SIZE
        )

        ctx.enqueue_function_checked[rolling_low_kernel, rolling_low_kernel](
            ts_dev.unsafe_ptr(),
            px_dev.unsafe_ptr(),
            rl_dev.unsafe_ptr().offset(offset),
            window_ms,
            Int32(n),
            grid_dim=imbalance_blocks, block_dim=BLOCK_SIZE
        )

    with GILReleased(py):
        ctx.synchronize()

    # STEP 4: Upload param grids to GPU
    var long_params_addr = Int(long_params.ctypes.data)
    var short_params_addr = Int(short_params.ctypes.data)

    var long_params_host = ctx.enqueue_create_host_buffer[DType.float64](n_long * 9)
    var short_params_host = ctx.enqueue_create_host_buffer[DType.float64](n_short * 9)
    with GILReleased(py):
        ctx.synchronize()

        memcpy(
            dest=long_params_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=long_params_addr),
            count=n_long * 9 * 8,
        )
        memcpy(
            dest=short_params_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=short_params_addr),
            count=n_short * 9 * 8,
        )

    var long_params_dev = ctx.enqueue_create_buffer[DType.float64](n_long * 9)
    var short_params_dev = ctx.enqueue_create_buffer[DType.float64](n_short * 9)
    ctx.enqueue_copy(dst_buf=long_params_dev, src_buf=long_params_host)
    ctx.enqueue_copy(dst_buf=short_params_dev, src_buf=short_params_host)

    # STEP 5: Run LONG grid search on GPU
    var long_start = time_module.perf_counter()

    var long_entries_dev = ctx.enqueue_create_buffer[DType.int32](n_long)
    var long_winners_dev = ctx.enqueue_create_buffer[DType.int32](n_long)
    var long_pnl_dev = ctx.enqueue_create_buffer[DType.float64](n_long)

    var long_blocks = ceildiv(n_long, WARPS_PER_BLOCK)
    ctx.enqueue_function_checked[grid_search_long_kernel, grid_search_long_kernel](
        ts_dev.unsafe_ptr(),
        px_dev.unsafe_ptr(),
        imbalance_dev.unsafe_ptr(),
        rh_dev.unsafe_ptr(),
        long_params_dev.unsafe_ptr(),
        long_entries_dev.unsafe_ptr(),
        long_winners_dev.unsafe_ptr(),
        long_pnl_dev.unsafe_ptr(),
        Int32(n_long), Int32(n_windows), Int32(n), Int32(max_scan), Int32(outer_stride),
        grid_dim=long_blocks, block_dim=BLOCK_SIZE
    )
    with GILReleased(py):
        ctx.synchronize()

    var long_time_ms = (time_module.perf_counter() - long_start) * 1000

    # Copy LONG results back
    var long_entries_host = ctx.enqueue_create_host_buffer[DType.int32](n_long)
    var long_winners_host = ctx.enqueue_create_host_buffer[DType.int32](n_long)
    var long_pnl_host = ctx.enqueue_create_host_buffer[DType.float64](n_long)
    ctx.enqueue_copy(dst_buf=long_entries_host, src_buf=long_entries_dev)
    ctx.enqueue_copy(dst_buf=long_winners_host, src_buf=long_winners_dev)
    ctx.enqueue_copy(dst_buf=long_pnl_host, src_buf=long_pnl_dev)
    with GILReleased(py):
        ctx.synchronize()

    var long_entries = np.zeros(n_long, dtype=np.int32)
    var long_winners = np.zeros(n_long, dtype=np.int32)
    var long_pnls = np.zeros(n_long, dtype=np.float64)

    var long_entries_addr = Int(long_entries.ctypes.data)
    var long_winners_addr = Int(long_winners.ctypes.data)
    var long_pnls_addr = Int(long_pnls.ctypes.data)

    with GILReleased(py):
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=long_entries_addr),
            src=long_entries_host.unsafe_ptr().bitcast[UInt8](),
            count=n_long * 4,
        )
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=long_winners_addr),
            src=long_winners_host.unsafe_ptr().bitcast[UInt8](),
            count=n_long * 4,
        )
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=long_pnls_addr),
            src=long_pnl_host.unsafe_ptr().bitcast[UInt8](),
            count=n_long * 8,
        )

    var long_result = select_best_params(
        "LONG", long_param_grid, long_entries, long_winners, long_pnls,
        min_entries, time_windows_ms
    )

    # STEP 6: Run SHORT grid search on GPU
    var short_start = time_module.perf_counter()

    var short_entries_dev = ctx.enqueue_create_buffer[DType.int32](n_short)
    var short_winners_dev = ctx.enqueue_create_buffer[DType.int32](n_short)
    var short_pnl_dev = ctx.enqueue_create_buffer[DType.float64](n_short)

    var short_blocks = ceildiv(n_short, WARPS_PER_BLOCK)
    ctx.enqueue_function_checked[grid_search_short_kernel, grid_search_short_kernel](
        ts_dev.unsafe_ptr(),
        px_dev.unsafe_ptr(),
        imbalance_dev.unsafe_ptr(),
        rl_dev.unsafe_ptr(),
        short_params_dev.unsafe_ptr(),
        short_entries_dev.unsafe_ptr(),
        short_winners_dev.unsafe_ptr(),
        short_pnl_dev.unsafe_ptr(),
        Int32(n_short), Int32(n_windows), Int32(n), Int32(max_scan), Int32(outer_stride),
        grid_dim=short_blocks, block_dim=BLOCK_SIZE
    )
    with GILReleased(py):
        ctx.synchronize()

    var short_time_ms = (time_module.perf_counter() - short_start) * 1000

    # Copy SHORT results back
    var short_entries_host = ctx.enqueue_create_host_buffer[DType.int32](n_short)
    var short_winners_host = ctx.enqueue_create_host_buffer[DType.int32](n_short)
    var short_pnl_host = ctx.enqueue_create_host_buffer[DType.float64](n_short)
    ctx.enqueue_copy(dst_buf=short_entries_host, src_buf=short_entries_dev)
    ctx.enqueue_copy(dst_buf=short_winners_host, src_buf=short_winners_dev)
    ctx.enqueue_copy(dst_buf=short_pnl_host, src_buf=short_pnl_dev)
    with GILReleased(py):
        ctx.synchronize()

    var short_entries = np.zeros(n_short, dtype=np.int32)
    var short_winners = np.zeros(n_short, dtype=np.int32)
    var short_pnls = np.zeros(n_short, dtype=np.float64)

    var short_entries_addr = Int(short_entries.ctypes.data)
    var short_winners_addr = Int(short_winners.ctypes.data)
    var short_pnls_addr = Int(short_pnls.ctypes.data)

    with GILReleased(py):
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=short_entries_addr),
            src=short_entries_host.unsafe_ptr().bitcast[UInt8](),
            count=n_short * 4,
        )
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=short_winners_addr),
            src=short_winners_host.unsafe_ptr().bitcast[UInt8](),
            count=n_short * 4,
        )
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=short_pnls_addr),
            src=short_pnl_host.unsafe_ptr().bitcast[UInt8](),
            count=n_short * 8,
        )

    var short_result = select_best_params(
        "SHORT", short_param_grid, short_entries, short_winners, short_pnls,
        min_entries, time_windows_ms
    )

    var total_time_ms = (time_module.perf_counter() - total_start) * 1000

    # STEP 7: Free GPU memory
    _ = ts_dev^
    _ = px_dev^
    _ = qty_dev^
    _ = sd_dev^
    _ = imbalance_dev^
    _ = rh_dev^
    _ = rl_dev^
    _ = long_params_dev^
    _ = short_params_dev^
    _ = long_entries_dev^
    _ = long_winners_dev^
    _ = long_pnl_dev^
    _ = short_entries_dev^
    _ = short_winners_dev^
    _ = short_pnl_dev^

    _ = ts_host^
    _ = px_host^
    _ = qty_host^
    _ = sd_host^
    _ = long_params_host^
    _ = short_params_host^
    _ = long_entries_host^
    _ = long_winners_host^
    _ = long_pnl_host^
    _ = short_entries_host^
    _ = short_winners_host^
    _ = short_pnl_host^

    with GILReleased(py):
        ctx.synchronize()

    # STEP 8: Build result dict
    var builtins = Python.import_module("builtins")

    var arch_dict = builtins.dict()
    arch_dict[PythonObject("warp_size")] = PythonObject(WARP_SIZE)
    arch_dict[PythonObject("block_size")] = PythonObject(BLOCK_SIZE)
    arch_dict[PythonObject("warps_per_block")] = PythonObject(WARPS_PER_BLOCK)
    arch_dict[PythonObject("sm_count")] = PythonObject(SM_COUNT)
    arch_dict[PythonObject("shared_mem_per_sm")] = PythonObject(SHARED_MEM_PER_SM)
    arch_dict[PythonObject("gpu_simd_bit_width")] = PythonObject(GPU_SIMD_BIT_WIDTH)
    arch_dict[PythonObject("simd_width_f64")] = PythonObject(SIMD_WIDTH_F64)
    arch_dict[PythonObject("simd_width_f32")] = PythonObject(SIMD_WIDTH_F32)

    var result = builtins.dict()
    result[PythonObject("long")] = long_result
    result[PythonObject("short")] = short_result
    result[PythonObject("long_param_combos")] = long_param_grid
    result[PythonObject("long_entries")] = long_entries
    result[PythonObject("long_winners")] = long_winners
    result[PythonObject("long_pnls")] = long_pnls
    result[PythonObject("short_param_combos")] = short_param_grid
    result[PythonObject("short_entries")] = short_entries
    result[PythonObject("short_winners")] = short_winners
    result[PythonObject("short_pnls")] = short_pnls
    result[PythonObject("gpu_time_ms")] = total_time_ms
    result[PythonObject("long_time_ms")] = long_time_ms
    result[PythonObject("short_time_ms")] = short_time_ms
    result[PythonObject("n_trades")] = PythonObject(n)
    result[PythonObject("n_long_combos")] = PythonObject(n_long)
    result[PythonObject("n_short_combos")] = PythonObject(n_short)
    result[PythonObject("arch")] = arch_dict

    return result


# =============================================================================
# BOOTSTRAP WRAPPER - Dict-based API for Python
# =============================================================================


@export
fn bootstrap_py(params: PythonObject) raises -> PythonObject:
    """
    Python binding for bootstrap (takes dict as argument).

    Args:
        params: A dict containing:
            timestamps, prices, quantities, sides,
            time_windows_ms, long_param_grid, short_param_grid,
            imbalance_price_tolerance_pct, max_scan, outer_stride, min_entries
    """
    return bootstrap_internal(
        params[PythonObject("timestamps")],
        params[PythonObject("prices")],
        params[PythonObject("quantities")],
        params[PythonObject("sides")],
        params[PythonObject("time_windows_ms")],
        params[PythonObject("long_param_grid")],
        params[PythonObject("short_param_grid")],
        Float64(params[PythonObject("imbalance_price_tolerance_pct")]),
        Int(params[PythonObject("max_scan")]),
        Int(params[PythonObject("outer_stride")]),
        Int(params[PythonObject("min_entries")]),
    )


# =============================================================================
# PYTHON MODULE INIT
# =============================================================================


@export
fn PyInit_bootstrap() -> PythonObject:
    """Initialize the bootstrap Python module."""
    try:
        var m = PythonModuleBuilder("bootstrap")
        m.def_function[bootstrap_py](
            "bootstrap",
            docstring="GPU-accelerated bootstrap optimization"
        )
        return m.finalize()
    except e:
        return PythonObject()
