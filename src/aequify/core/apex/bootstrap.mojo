"""
APEX Bootstrap - GPU-accelerated parameter optimization pipeline.

Full GPU pipeline: numpy → GPU → compute → GPU → numpy

Pipeline:
1. volume_delta_multi: Volume delta over SAME windows as rolling_high/rolling_low
2. rolling_high_multi: Rolling max price over time windows
3. rolling_low_multi: Rolling min price over time windows
4. grid_search_long_dca: LONG position optimization with DCA
5. grid_search_short_dca: SHORT position optimization with DCA

Entry signals:
- LONG: price dropped from rolling_high AND strong selling (delta <= -threshold)
- SHORT: price rose from rolling_low AND strong buying (delta >= +threshold)
"""

from math import ceildiv
from sys import stderr
from time import perf_counter_ns

from gpu.host import DeviceContext
from gpu.globals import WARP_SIZE
from logger import Logger, Level
from memory import memcpy, UnsafePointer
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from python._cpython import GILReleased


# Import from kernels package
from kernels import (
    volume_delta_multi_gpu,
    rolling_high_multi_gpu,
    rolling_low_multi_gpu,
    NO_LOW_SENTINEL,
    grid_search_long_dca_gpu,
    grid_search_short_dca_gpu,
    get_block_size,
    get_warps_per_block,
    get_warp_size,
)


# =============================================================================
# Architecture-Adaptive Constants
# =============================================================================

comptime HW_INFO = DeviceContext.default_device_info
comptime MAX_BLOCK_SIZE: Int = HW_INFO.max_thread_block_size
comptime BLOCK_SIZE: Int = 256 if MAX_BLOCK_SIZE >= 256 else MAX_BLOCK_SIZE
comptime WARPS_PER_BLOCK: Int = BLOCK_SIZE // WARP_SIZE
comptime SM_COUNT: Int = HW_INFO.sm_count
comptime SHARED_MEM_PER_SM: Int = HW_INFO.shared_memory_per_multiprocessor
comptime GPU_SIMD_BIT_WIDTH: Int = 128
comptime SIMD_WIDTH_F64: Int = GPU_SIMD_BIT_WIDTH // 64
comptime SIMD_WIDTH_F32: Int = GPU_SIMD_BIT_WIDTH // 32

comptime MIN_ENTRIES_FOR_VALIDITY: Int = 10


# =============================================================================
# Result Selection
# =============================================================================


fn select_best_params(
    direction: String,
    param_combos: PythonObject,
    entries: PythonObject,
    winners: PythonObject,
    pnl_sums: PythonObject,
    dca_counts: PythonObject,
    min_entries: Int,
    time_windows_ms: PythonObject,
) raises -> PythonObject:
    """Select best parameters from grid search results.

    Param combo layout (7 params per combo):
        [0] pm: Price move threshold
        [1] window_idx: Index into time_windows_ms
        [2] dt: Delta threshold
        [3] target_pct: Take profit percentage
        [4] stop_pct: Stop loss percentage
        [5] max_hold_ms: Maximum hold time in milliseconds
        [6] dca_distance_pct: Distance from avg entry to trigger DCA
    """
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
            var dca_dist = param_combos[i, 6]
            var tw_ms = time_windows_ms[w_idx]
            var n_dca = Int(dca_counts[i])

            var params_dict = builtins.dict()
            params_dict[PythonObject("price_move")] = pm
            params_dict[PythonObject("time_window")] = tw_ms
            params_dict[PythonObject("delta_threshold")] = dt
            params_dict[PythonObject("target_profit")] = tp
            params_dict[PythonObject("stop_loss")] = sl
            params_dict[PythonObject("max_hold_time")] = PythonObject(mh)
            params_dict[PythonObject("dca_distance_pct")] = dca_dist

            best_result = builtins.dict()
            best_result[PythonObject("direction")] = PythonObject(direction)
            best_result[PythonObject("params")] = params_dict
            best_result[PythonObject("entries")] = PythonObject(n_entries)
            best_result[PythonObject("winners")] = PythonObject(n_winners)
            best_result[PythonObject("win_rate")] = PythonObject(wr)
            best_result[PythonObject("avg_pnl")] = PythonObject(avg_pnl)
            best_result[PythonObject("loss")] = PythonObject(loss)
            best_result[PythonObject("dca_count")] = PythonObject(n_dca)

    return best_result


# =============================================================================
# Bootstrap Internal - Full GPU Pipeline
# =============================================================================


fn bootstrap_internal(
    timestamps: PythonObject,
    prices: PythonObject,
    quantities: PythonObject,
    sides: PythonObject,
    time_windows_ms: PythonObject,
    long_param_grid: PythonObject,
    short_param_grid: PythonObject,
    max_scan: Int,
    outer_stride: Int,
    min_entries: Int,
) raises -> PythonObject:
    """
    Full GPU-accelerated bootstrap pipeline.

    Uses optimized GPU kernels:
    - volume_delta_multi_gpu: Volume delta over SAME windows as rolling_high/low
    - rolling_high_multi_gpu / rolling_low_multi_gpu: Multi-window rolling max/min
    - grid_search_long_dca_gpu / grid_search_short_dca_gpu: DCA position simulation
    """
    var py = Python()
    var np = Python.import_module("numpy")
    var log = Logger[Level.DEBUG](stderr, prefix="[MOJO] ")

    var n = Int(timestamps.shape[0])
    var n_windows = Int(len(time_windows_ms))
    var n_long = Int(long_param_grid.shape[0])
    var n_short = Int(short_param_grid.shape[0])

    # Ensure arrays are contiguous
    var ts_arr = np.ascontiguousarray(timestamps, dtype=np.int64)
    var px_arr = np.ascontiguousarray(prices, dtype=np.float64)
    var qty_arr = np.ascontiguousarray(quantities, dtype=np.float64)
    var sd_arr = np.ascontiguousarray(sides, dtype=np.int64)
    var long_params = np.ascontiguousarray(long_param_grid.flatten(), dtype=np.float64)
    var short_params = np.ascontiguousarray(short_param_grid.flatten(), dtype=np.float64)

    # Extract numpy pointers
    var ts_addr = Int(ts_arr.ctypes.data)
    var px_addr = Int(px_arr.ctypes.data)
    var qty_addr = Int(qty_arr.ctypes.data)
    var sd_addr = Int(sd_arr.ctypes.data)
    var long_params_addr = Int(long_params.ctypes.data)
    var short_params_addr = Int(short_params.ctypes.data)

    # Extract time windows
    var windows = List[Int64]()
    for w_idx in range(n_windows):
        windows.append(Int64(Int(time_windows_ms[w_idx])))

    # Pre-allocate output arrays
    var long_entries = np.zeros(n_long, dtype=np.int32)
    var long_winners = np.zeros(n_long, dtype=np.int32)
    var long_pnls = np.zeros(n_long, dtype=np.float64)
    var long_dca_counts = np.zeros(n_long, dtype=np.int32)
    var short_entries = np.zeros(n_short, dtype=np.int32)
    var short_winners = np.zeros(n_short, dtype=np.int32)
    var short_pnls = np.zeros(n_short, dtype=np.float64)
    var short_dca_counts = np.zeros(n_short, dtype=np.int32)

    var long_entries_addr = Int(long_entries.ctypes.data)
    var long_winners_addr = Int(long_winners.ctypes.data)
    var long_pnls_addr = Int(long_pnls.ctypes.data)
    var long_dca_counts_addr = Int(long_dca_counts.ctypes.data)
    var short_entries_addr = Int(short_entries.ctypes.data)
    var short_winners_addr = Int(short_winners.ctypes.data)
    var short_pnls_addr = Int(short_pnls.ctypes.data)
    var short_dca_counts_addr = Int(short_dca_counts.ctypes.data)

    # =========================================================================
    # GPU Computation
    # =========================================================================
    var total_start_ns = perf_counter_ns()
    var long_time_ns: Int = 0
    var short_time_ns: Int = 0

    var ctx = DeviceContext()

    # Create host buffers
    var ts_host = ctx.enqueue_create_host_buffer[DType.int64](n)
    var px_host = ctx.enqueue_create_host_buffer[DType.float64](n)
    var qty_host = ctx.enqueue_create_host_buffer[DType.float64](n)
    var sd_host = ctx.enqueue_create_host_buffer[DType.int64](n)
    var long_params_host = ctx.enqueue_create_host_buffer[DType.float64](n_long * 7)
    var short_params_host = ctx.enqueue_create_host_buffer[DType.float64](n_short * 7)

    with GILReleased(py):
        ctx.synchronize()

        # Copy input data to pinned host memory
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
            count=n * 8,
        )
        memcpy(
            dest=long_params_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=long_params_addr),
            count=n_long * 7 * 8,
        )
        memcpy(
            dest=short_params_host.unsafe_ptr().bitcast[UInt8](),
            src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=short_params_addr),
            count=n_short * 7 * 8,
        )

        # Transfer to GPU
        var ts_dev = ctx.enqueue_create_buffer[DType.int64](n)
        var px_dev = ctx.enqueue_create_buffer[DType.float64](n)
        var qty_dev = ctx.enqueue_create_buffer[DType.float64](n)
        var sd_dev = ctx.enqueue_create_buffer[DType.int64](n)
        ctx.enqueue_copy(dst_buf=ts_dev, src_buf=ts_host)
        ctx.enqueue_copy(dst_buf=px_dev, src_buf=px_host)
        ctx.enqueue_copy(dst_buf=qty_dev, src_buf=qty_host)
        ctx.enqueue_copy(dst_buf=sd_dev, src_buf=sd_host)

        var long_params_dev = ctx.enqueue_create_buffer[DType.float64](n_long * 7)
        var short_params_dev = ctx.enqueue_create_buffer[DType.float64](n_short * 7)
        ctx.enqueue_copy(dst_buf=long_params_dev, src_buf=long_params_host)
        ctx.enqueue_copy(dst_buf=short_params_dev, src_buf=short_params_host)

        # Create windows buffer
        var windows_host = ctx.enqueue_create_host_buffer[DType.int64](n_windows)
        for w_idx in range(n_windows):
            windows_host.unsafe_ptr()[w_idx] = windows[w_idx]
        var windows_dev = ctx.enqueue_create_buffer[DType.int64](n_windows)
        ctx.enqueue_copy(dst_buf=windows_dev, src_buf=windows_host)

        ctx.synchronize()

        # ---------------------------------------------------------------------
        # Compute rolling metrics (all use SAME windows)
        # ---------------------------------------------------------------------
        var rolling_size = n_windows * n

        # Volume delta over same windows as rolling_high/rolling_low
        var vd_dev = ctx.enqueue_create_buffer[DType.float64](rolling_size)
        volume_delta_multi_gpu(ctx, qty_dev, sd_dev, vd_dev, windows_dev, n_windows, n)

        # Rolling high/low
        var rh_dev = ctx.enqueue_create_buffer[DType.float64](rolling_size)
        var rl_dev = ctx.enqueue_create_buffer[DType.float64](rolling_size)
        rolling_high_multi_gpu(ctx, px_dev, rh_dev, windows_dev, n_windows, n)
        rolling_low_multi_gpu(ctx, px_dev, rl_dev, windows_dev, n_windows, n)
        ctx.synchronize()

        # Free windows buffers
        _ = windows_host^
        _ = windows_dev^

        # ---------------------------------------------------------------------
        # LONG grid search
        # ---------------------------------------------------------------------
        var long_start_ns = perf_counter_ns()

        var long_entries_dev = ctx.enqueue_create_buffer[DType.int32](n_long)
        var long_winners_dev = ctx.enqueue_create_buffer[DType.int32](n_long)
        var long_pnl_dev = ctx.enqueue_create_buffer[DType.float64](n_long)
        var long_dca_count_dev = ctx.enqueue_create_buffer[DType.int32](n_long)

        grid_search_long_dca_gpu(
            ctx,
            ts_dev,
            px_dev,
            rh_dev,
            vd_dev,
            long_params_dev,
            long_entries_dev,
            long_winners_dev,
            long_pnl_dev,
            long_dca_count_dev,
            n_long,
            n_windows,
            n,
            max_scan,
            outer_stride,
        )
        ctx.synchronize()

        long_time_ns = Int(perf_counter_ns() - long_start_ns)

        # Copy LONG results back
        var long_entries_host = ctx.enqueue_create_host_buffer[DType.int32](n_long)
        var long_winners_host = ctx.enqueue_create_host_buffer[DType.int32](n_long)
        var long_pnl_host = ctx.enqueue_create_host_buffer[DType.float64](n_long)
        var long_dca_count_host = ctx.enqueue_create_host_buffer[DType.int32](n_long)
        ctx.enqueue_copy(dst_buf=long_entries_host, src_buf=long_entries_dev)
        ctx.enqueue_copy(dst_buf=long_winners_host, src_buf=long_winners_dev)
        ctx.enqueue_copy(dst_buf=long_pnl_host, src_buf=long_pnl_dev)
        ctx.enqueue_copy(dst_buf=long_dca_count_host, src_buf=long_dca_count_dev)
        ctx.synchronize()

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
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=long_dca_counts_addr),
            src=long_dca_count_host.unsafe_ptr().bitcast[UInt8](),
            count=n_long * 4,
        )

        # ---------------------------------------------------------------------
        # SHORT grid search
        # ---------------------------------------------------------------------
        var short_start_ns = perf_counter_ns()

        var short_entries_dev = ctx.enqueue_create_buffer[DType.int32](n_short)
        var short_winners_dev = ctx.enqueue_create_buffer[DType.int32](n_short)
        var short_pnl_dev = ctx.enqueue_create_buffer[DType.float64](n_short)
        var short_dca_count_dev = ctx.enqueue_create_buffer[DType.int32](n_short)

        grid_search_short_dca_gpu(
            ctx,
            ts_dev,
            px_dev,
            rl_dev,
            vd_dev,
            short_params_dev,
            short_entries_dev,
            short_winners_dev,
            short_pnl_dev,
            short_dca_count_dev,
            n_short,
            n_windows,
            n,
            max_scan,
            outer_stride,
        )
        ctx.synchronize()

        short_time_ns = Int(perf_counter_ns() - short_start_ns)

        # Copy SHORT results back
        var short_entries_host = ctx.enqueue_create_host_buffer[DType.int32](n_short)
        var short_winners_host = ctx.enqueue_create_host_buffer[DType.int32](n_short)
        var short_pnl_host = ctx.enqueue_create_host_buffer[DType.float64](n_short)
        var short_dca_count_host = ctx.enqueue_create_host_buffer[DType.int32](n_short)
        ctx.enqueue_copy(dst_buf=short_entries_host, src_buf=short_entries_dev)
        ctx.enqueue_copy(dst_buf=short_winners_host, src_buf=short_winners_dev)
        ctx.enqueue_copy(dst_buf=short_pnl_host, src_buf=short_pnl_dev)
        ctx.enqueue_copy(dst_buf=short_dca_count_host, src_buf=short_dca_count_dev)
        ctx.synchronize()

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
        memcpy(
            dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=short_dca_counts_addr),
            src=short_dca_count_host.unsafe_ptr().bitcast[UInt8](),
            count=n_short * 4,
        )

        # Cleanup GPU memory
        _ = ts_dev^
        _ = px_dev^
        _ = qty_dev^
        _ = sd_dev^
        _ = vd_dev^
        _ = rh_dev^
        _ = rl_dev^
        _ = long_params_dev^
        _ = short_params_dev^
        _ = long_entries_dev^
        _ = long_winners_dev^
        _ = long_pnl_dev^
        _ = long_dca_count_dev^
        _ = short_entries_dev^
        _ = short_winners_dev^
        _ = short_pnl_dev^
        _ = short_dca_count_dev^

        _ = ts_host^
        _ = px_host^
        _ = qty_host^
        _ = sd_host^
        _ = long_params_host^
        _ = short_params_host^
        _ = long_entries_host^
        _ = long_winners_host^
        _ = long_pnl_host^
        _ = long_dca_count_host^
        _ = short_entries_host^
        _ = short_winners_host^
        _ = short_pnl_host^
        _ = short_dca_count_host^

        ctx.synchronize()

    var total_time_ns = perf_counter_ns() - total_start_ns
    var total_time_ms = Float64(total_time_ns) / 1_000_000.0
    var long_time_ms = Float64(long_time_ns) / 1_000_000.0
    var short_time_ms = Float64(short_time_ns) / 1_000_000.0

    # Select best params
    var long_result = select_best_params(
        "LONG", long_param_grid, long_entries, long_winners, long_pnls,
        long_dca_counts, min_entries, time_windows_ms
    )

    var short_result = select_best_params(
        "SHORT", short_param_grid, short_entries, short_winners, short_pnls,
        short_dca_counts, min_entries, time_windows_ms
    )

    # Build result dict
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
    result[PythonObject("long_dca_counts")] = long_dca_counts
    result[PythonObject("short_param_combos")] = short_param_grid
    result[PythonObject("short_entries")] = short_entries
    result[PythonObject("short_winners")] = short_winners
    result[PythonObject("short_pnls")] = short_pnls
    result[PythonObject("short_dca_counts")] = short_dca_counts
    result[PythonObject("gpu_time_ms")] = PythonObject(total_time_ms)
    result[PythonObject("long_time_ms")] = PythonObject(long_time_ms)
    result[PythonObject("short_time_ms")] = PythonObject(short_time_ms)
    result[PythonObject("n_trades")] = PythonObject(n)
    result[PythonObject("n_long_combos")] = PythonObject(n_long)
    result[PythonObject("n_short_combos")] = PythonObject(n_short)
    result[PythonObject("arch")] = arch_dict

    return result


# =============================================================================
# Python Binding
# =============================================================================


@export
fn bootstrap_py(params: PythonObject) raises -> PythonObject:
    """
    Python binding for bootstrap.

    Args:
        params: Dict with `timestamps`, `prices`, `quantities`, `sides`,
            `time_windows_ms`, `long_param_grid`, `short_param_grid`,
            `max_scan`, `outer_stride`, `min_entries`.
    """
    return bootstrap_internal(
        params[PythonObject("timestamps")],
        params[PythonObject("prices")],
        params[PythonObject("quantities")],
        params[PythonObject("sides")],
        params[PythonObject("time_windows_ms")],
        params[PythonObject("long_param_grid")],
        params[PythonObject("short_param_grid")],
        Int(params[PythonObject("max_scan")]),
        Int(params[PythonObject("outer_stride")]),
        Int(params[PythonObject("min_entries")]),
    )


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
