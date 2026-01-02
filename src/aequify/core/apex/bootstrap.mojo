"""
APEX Bootstrap - GPU-accelerated parameter optimization pipeline.

Full GPU pipeline: numpy → GPU → compute → GPU → numpy

Orchestrates:
- volume_imbalance: Prefix-sum based O(n × buckets) implementation
- rolling_high/low: O(log n) binary search + SIMD vectorization
- grid_search_long/short: Warp-per-combo with DCA support

Architecture-adaptive constants for NVIDIA/AMD/Apple GPUs.
"""

from math import ceildiv
from time import perf_counter_ns

from gpu.host import DeviceContext
from gpu.globals import WARP_SIZE
from memory import memcpy, UnsafePointer
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from python._cpython import GILReleased

# Import from kernels package (built separately)
from kernels import (
    volume_imbalance_gpu,
    compute_price_range_cpu,
    rolling_high_gpu,
    rolling_low_gpu,
    NO_LOW_SENTINEL,
    grid_search_long_gpu,
    grid_search_short_gpu,
    get_block_size,
    get_warps_per_block,
    get_warp_size,
)


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

    Uses optimized GPU kernels from kernel modules:
    - volume_imbalance_gpu: O(n × buckets) prefix-sum based
    - rolling_high_gpu / rolling_low_gpu: O(n × window) with SIMD
    - grid_search_long_gpu / grid_search_short_gpu: Warp-per-combo with DCA

    GIL is released during GPU operations to allow other Python threads to run.
    """
    var py = Python()
    var np = Python.import_module("numpy")

    var n = Int(timestamps.shape[0])
    var n_windows = Int(len(time_windows_ms))
    var n_long = Int(long_param_grid.shape[0])
    var n_short = Int(short_param_grid.shape[0])

    # Ensure arrays are contiguous and correct dtype
    var ts_arr = np.ascontiguousarray(timestamps, dtype=np.int64)
    var px_arr = np.ascontiguousarray(prices, dtype=np.float64)
    var qty_arr = np.ascontiguousarray(quantities, dtype=np.float64)
    var sd_arr = np.ascontiguousarray(sides, dtype=np.int32)
    var long_params = np.ascontiguousarray(long_param_grid.flatten(), dtype=np.float64)
    var short_params = np.ascontiguousarray(short_param_grid.flatten(), dtype=np.float64)

    # =========================================================================
    # EXTRACT ALL NUMPY POINTERS BEFORE RELEASING GIL
    # =========================================================================
    var ts_addr = Int(ts_arr.ctypes.data)
    var px_addr = Int(px_arr.ctypes.data)
    var qty_addr = Int(qty_arr.ctypes.data)
    var sd_addr = Int(sd_arr.ctypes.data)
    var long_params_addr = Int(long_params.ctypes.data)
    var short_params_addr = Int(short_params.ctypes.data)

    # Extract time windows to Mojo list (no Python access needed later)
    var windows = List[Int64]()
    for w_idx in range(n_windows):
        windows.append(Int64(Int(time_windows_ms[w_idx])))

    # Compute price range for volume_imbalance_gpu (needs min/max price)
    var min_price = Float64(np.min(px_arr))
    var max_price = Float64(np.max(px_arr))

    # Pre-allocate output numpy arrays (need GIL for this)
    var long_entries = np.zeros(n_long, dtype=np.int32)
    var long_winners = np.zeros(n_long, dtype=np.int32)
    var long_pnls = np.zeros(n_long, dtype=np.float64)
    var short_entries = np.zeros(n_short, dtype=np.int32)
    var short_winners = np.zeros(n_short, dtype=np.int32)
    var short_pnls = np.zeros(n_short, dtype=np.float64)

    # Extract output array addresses
    var long_entries_addr = Int(long_entries.ctypes.data)
    var long_winners_addr = Int(long_winners.ctypes.data)
    var long_pnls_addr = Int(long_pnls.ctypes.data)
    var short_entries_addr = Int(short_entries.ctypes.data)
    var short_winners_addr = Int(short_winners.ctypes.data)
    var short_pnls_addr = Int(short_pnls.ctypes.data)

    # =========================================================================
    # GPU COMPUTATION - ALL IN SINGLE GIL-RELEASED BLOCK
    # =========================================================================
    var total_start_ns = perf_counter_ns()
    var long_time_ns: Int = 0
    var short_time_ns: Int = 0

    var ctx = DeviceContext()

    # Create host buffers (outside GIL block - may need Python internally)
    var ts_host = ctx.enqueue_create_host_buffer[DType.int64](n)
    var px_host = ctx.enqueue_create_host_buffer[DType.float64](n)
    var qty_host = ctx.enqueue_create_host_buffer[DType.float64](n)
    var sd_host = ctx.enqueue_create_host_buffer[DType.int32](n)

    # Create param host buffers
    var long_params_host = ctx.enqueue_create_host_buffer[DType.float64](n_long * 9)
    var short_params_host = ctx.enqueue_create_host_buffer[DType.float64](n_short * 9)

    # =========================================================================
    # SINGLE GIL-RELEASED BLOCK FOR ALL GPU OPERATIONS
    # This allows other Python threads to run while GPU computes
    # =========================================================================
    with GILReleased(py):
        ctx.synchronize()

        # ---------------------------------------------------------------------
        # STEP 1: Copy input data from numpy to pinned host memory
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # STEP 2: Transfer to GPU device memory
        # ---------------------------------------------------------------------
        var ts_dev = ctx.enqueue_create_buffer[DType.int64](n)
        var px_dev = ctx.enqueue_create_buffer[DType.float64](n)
        var qty_dev = ctx.enqueue_create_buffer[DType.float64](n)
        var sd_dev = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_copy(dst_buf=ts_dev, src_buf=ts_host)
        ctx.enqueue_copy(dst_buf=px_dev, src_buf=px_host)
        ctx.enqueue_copy(dst_buf=qty_dev, src_buf=qty_host)
        ctx.enqueue_copy(dst_buf=sd_dev, src_buf=sd_host)

        var long_params_dev = ctx.enqueue_create_buffer[DType.float64](n_long * 9)
        var short_params_dev = ctx.enqueue_create_buffer[DType.float64](n_short * 9)
        ctx.enqueue_copy(dst_buf=long_params_dev, src_buf=long_params_host)
        ctx.enqueue_copy(dst_buf=short_params_dev, src_buf=short_params_host)

        ctx.synchronize()

        # ---------------------------------------------------------------------
        # STEP 3: Compute volume imbalance on GPU (optimized prefix-sum version)
        # ---------------------------------------------------------------------
        var imbalance_dev = ctx.enqueue_create_buffer[DType.float64](n)

        volume_imbalance_gpu(
            ctx,
            ts_dev,
            px_dev,
            qty_dev,
            sd_dev,
            imbalance_dev,
            min_price,
            max_price,
            imbalance_price_tolerance_pct,
            n,
        )
        ctx.synchronize()

        # ---------------------------------------------------------------------
        # STEP 4: Compute rolling highs/lows on GPU
        # ---------------------------------------------------------------------
        var rolling_size = n_windows * n
        var rh_dev = ctx.enqueue_create_buffer[DType.float64](rolling_size)
        var rl_dev = ctx.enqueue_create_buffer[DType.float64](rolling_size)

        for w_idx in range(n_windows):
            var window_ms = windows[w_idx]

            # Create sub-buffers for this window's results
            # Note: We need to create separate buffers and copy, or use offset
            # For simplicity, we'll create temp buffers and copy
            var rh_window_dev = ctx.enqueue_create_buffer[DType.float64](n)
            var rl_window_dev = ctx.enqueue_create_buffer[DType.float64](n)

            rolling_high_gpu(ctx, ts_dev, px_dev, rh_window_dev, window_ms, n)
            rolling_low_gpu(ctx, ts_dev, px_dev, rl_window_dev, window_ms, n)

            # Copy to the correct offset in the combined buffer
            var offset = w_idx * n
            var rh_host_temp = ctx.enqueue_create_host_buffer[DType.float64](n)
            var rl_host_temp = ctx.enqueue_create_host_buffer[DType.float64](n)
            ctx.enqueue_copy(dst_buf=rh_host_temp, src_buf=rh_window_dev)
            ctx.enqueue_copy(dst_buf=rl_host_temp, src_buf=rl_window_dev)
            ctx.synchronize()

            # Copy to combined buffer at offset
            memcpy(
                dest=rh_dev.unsafe_ptr().offset(offset).bitcast[UInt8](),
                src=rh_host_temp.unsafe_ptr().bitcast[UInt8](),
                count=n * 8,
            )
            memcpy(
                dest=rl_dev.unsafe_ptr().offset(offset).bitcast[UInt8](),
                src=rl_host_temp.unsafe_ptr().bitcast[UInt8](),
                count=n * 8,
            )

            _ = rh_window_dev^
            _ = rl_window_dev^
            _ = rh_host_temp^
            _ = rl_host_temp^

        ctx.synchronize()

        # ---------------------------------------------------------------------
        # STEP 5: Run LONG grid search on GPU
        # ---------------------------------------------------------------------
        var long_start_ns = perf_counter_ns()

        var long_entries_dev = ctx.enqueue_create_buffer[DType.int32](n_long)
        var long_winners_dev = ctx.enqueue_create_buffer[DType.int32](n_long)
        var long_pnl_dev = ctx.enqueue_create_buffer[DType.float64](n_long)

        grid_search_long_gpu(
            ctx,
            ts_dev,
            px_dev,
            imbalance_dev,
            rh_dev,
            long_params_dev,
            long_entries_dev,
            long_winners_dev,
            long_pnl_dev,
            n_long,
            n_windows,
            n,
            max_scan,
            outer_stride,
        )
        ctx.synchronize()

        long_time_ns = Int(perf_counter_ns() - long_start_ns)

        # Copy LONG results back to host
        var long_entries_host = ctx.enqueue_create_host_buffer[DType.int32](n_long)
        var long_winners_host = ctx.enqueue_create_host_buffer[DType.int32](n_long)
        var long_pnl_host = ctx.enqueue_create_host_buffer[DType.float64](n_long)
        ctx.enqueue_copy(dst_buf=long_entries_host, src_buf=long_entries_dev)
        ctx.enqueue_copy(dst_buf=long_winners_host, src_buf=long_winners_dev)
        ctx.enqueue_copy(dst_buf=long_pnl_host, src_buf=long_pnl_dev)
        ctx.synchronize()

        # Copy to numpy arrays
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

        # ---------------------------------------------------------------------
        # STEP 6: Run SHORT grid search on GPU
        # ---------------------------------------------------------------------
        var short_start_ns = perf_counter_ns()

        var short_entries_dev = ctx.enqueue_create_buffer[DType.int32](n_short)
        var short_winners_dev = ctx.enqueue_create_buffer[DType.int32](n_short)
        var short_pnl_dev = ctx.enqueue_create_buffer[DType.float64](n_short)

        grid_search_short_gpu(
            ctx,
            ts_dev,
            px_dev,
            imbalance_dev,
            rl_dev,
            short_params_dev,
            short_entries_dev,
            short_winners_dev,
            short_pnl_dev,
            n_short,
            n_windows,
            n,
            max_scan,
            outer_stride,
        )
        ctx.synchronize()

        short_time_ns = Int(perf_counter_ns() - short_start_ns)

        # Copy SHORT results back to host
        var short_entries_host = ctx.enqueue_create_host_buffer[DType.int32](n_short)
        var short_winners_host = ctx.enqueue_create_host_buffer[DType.int32](n_short)
        var short_pnl_host = ctx.enqueue_create_host_buffer[DType.float64](n_short)
        ctx.enqueue_copy(dst_buf=short_entries_host, src_buf=short_entries_dev)
        ctx.enqueue_copy(dst_buf=short_winners_host, src_buf=short_winners_dev)
        ctx.enqueue_copy(dst_buf=short_pnl_host, src_buf=short_pnl_dev)
        ctx.synchronize()

        # Copy to numpy arrays
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

        # ---------------------------------------------------------------------
        # STEP 7: Free GPU memory
        # ---------------------------------------------------------------------
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

        ctx.synchronize()

    # END OF GIL-RELEASED BLOCK
    # =========================================================================

    var total_time_ns = perf_counter_ns() - total_start_ns

    # Convert timing from nanoseconds to milliseconds
    var total_time_ms = Float64(total_time_ns) / 1_000_000.0
    var long_time_ms = Float64(long_time_ns) / 1_000_000.0
    var short_time_ms = Float64(short_time_ns) / 1_000_000.0

    # =========================================================================
    # STEP 8: Select best params (needs GIL for Python object access)
    # =========================================================================
    var long_result = select_best_params(
        "LONG", long_param_grid, long_entries, long_winners, long_pnls,
        min_entries, time_windows_ms
    )

    var short_result = select_best_params(
        "SHORT", short_param_grid, short_entries, short_winners, short_pnls,
        min_entries, time_windows_ms
    )

    # =========================================================================
    # STEP 9: Build result dict (needs GIL for Python object creation)
    # =========================================================================
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
    result[PythonObject("gpu_time_ms")] = PythonObject(total_time_ms)
    result[PythonObject("long_time_ms")] = PythonObject(long_time_ms)
    result[PythonObject("short_time_ms")] = PythonObject(short_time_ms)
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
        params: A dict containing `timestamps`, `prices`, `quantities`, `sides`,
            `time_windows_ms`, `long_param_grid`, `short_param_grid`,
            `imbalance_price_tolerance_pct`, `max_scan`, `outer_stride`, `min_entries`.
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
