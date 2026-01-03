"""
Rolling High Kernel - GPU-accelerated rolling maximum price computation.

This kernel computes the rolling maximum price over a fixed window of trades.
Used by grid_search_long to detect price drops from recent highs.

Architecture:
    1. Simple per-thread kernel (each thread handles one output index)
    2. Scans the window range to find maximum price
    3. For indices < window: output is 0.0 (invalid/no data)

Formula: rolling_high[i] = max(prices[i-window:i])

This is a direct port of the Python/CUDA kernel from apex_29_ultra.py:
    @cuda.jit
    def rolling_high_kernel(prices, window, out, n):
        i = cuda.grid(1)
        if i >= n:
            return
        if i < window:
            out[i] = 0.0
            return
        max_val = prices[i - window]
        for j in range(i - window + 1, i):
            if prices[j] > max_val:
                max_val = prices[j]
        out[i] = max_val

Usage:
    rolling_high_gpu(ctx, prices_dev, result_dev, window, n)
"""

from math import ceildiv

from gpu import global_idx, lane_id
from gpu.host import DeviceContext, DeviceBuffer
from gpu.primitives.warp import lane_group_max
from memory import UnsafePointer


# =============================================================================
# Architecture-Adaptive Constants (from device info at compile time)
# =============================================================================

# Get hardware info at compile time from the target GPU
comptime HW_INFO = DeviceContext.default_device_info

# Warp size from device (32 for NVIDIA, 64 for AMD, 32 for Apple)
comptime WARP_SIZE: Int = HW_INFO.warp_size

# Block size from device - max threads per block
# We use 256 as optimal balance between occupancy and register pressure
comptime MAX_BLOCK_SIZE: Int = HW_INFO.max_thread_block_size
comptime BLOCK_SIZE: Int = 256 if MAX_BLOCK_SIZE >= 256 else MAX_BLOCK_SIZE

# Warps per block - computed from architecture
comptime WARPS_PER_BLOCK: Int = BLOCK_SIZE // WARP_SIZE

# SM/CU count for optimal grid sizing
comptime SM_COUNT: Int = HW_INFO.sm_count


# =============================================================================
# Warp Reduction Helper
# =============================================================================


@always_inline
fn warp_reduce_max_f64(val: Float64) -> Float64:
    """Warp-level max reduction using lane_group_max.

    Uses Float32 internally for compatibility with lane_group_max,
    then converts back to Float64.
    """
    var val_f32 = Float32(val)
    var result = lane_group_max[num_lanes=WARP_SIZE](val_f32)
    return Float64(result)


# =============================================================================
# Simple Per-Thread Kernel (matches Python implementation exactly)
# =============================================================================


fn _rolling_high_simple_kernel(
    prices: UnsafePointer[Float64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    window: Int32,
    n: Int32,
):
    """Simple per-thread rolling high kernel.

    Each thread handles one output index independently.
    This is the direct Mojo translation of the Python kernel.

    Args:
        prices: Trade prices.
        result: Output rolling high array.
        window: Rolling window size (number of trades).
        n: Number of trades.
    """
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    # For indices < window, output is 0.0 (no valid data)
    if i < Int(window):
        result[i] = 0.0
        return

    # Find max in window [i-window, i)
    var max_val = prices[i - Int(window)]
    for j in range(i - Int(window) + 1, i):
        var p = prices[j]
        if p > max_val:
            max_val = p

    result[i] = max_val


# =============================================================================
# Warp-Cooperative Kernel (optimized for large windows)
# =============================================================================


fn _rolling_high_warp_kernel(
    prices: UnsafePointer[Float64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    window: Int32,
    n: Int32,
):
    """Warp-cooperative rolling high kernel.

    One warp/wavefront (WARP_SIZE threads) processes each output index cooperatively.
    Each lane scans a portion of the window range, then warp-level reduction
    finds the maximum.

    Architecture-adaptive: uses WARP_SIZE from device info
    (32 for NVIDIA/Apple, 64 for AMD).

    Args:
        prices: Trade prices.
        result: Output rolling high array.
        window: Rolling window size (number of trades).
        n: Number of trades.
    """
    # Calculate which output index this warp is responsible for
    var warp_idx = Int(global_idx.x) // WARP_SIZE
    var lane = Int(lane_id())

    # Each warp handles one output index
    var i = warp_idx

    if i >= Int(n):
        return

    # For indices < window, output is 0.0 (no valid data)
    if i < Int(window):
        if lane == 0:
            result[i] = 0.0
        return

    # Warp-cooperative scan: each lane handles every WARP_SIZE-th element
    # Use negative infinity as identity for max reduction
    var local_max: Float64 = -1e308

    # Window range: [i - window, i)
    var start_idx = i - Int(window)

    # Lane k scans indices: start_idx + k, start_idx + k + WARP_SIZE, ...
    var j = start_idx + lane
    while j < i:
        var p = prices[j]
        if p > local_max:
            local_max = p
        j += WARP_SIZE

    # Warp-level reduction to find maximum across all lanes
    var global_max = warp_reduce_max_f64(local_max)

    # Lane 0 writes the final result
    if lane == 0:
        result[i] = global_max


# =============================================================================
# Main GPU Functions
# =============================================================================


fn rolling_high_gpu(
    ctx: DeviceContext,
    prices_dev: DeviceBuffer[DType.float64],
    result_dev: DeviceBuffer[DType.float64],
    window: Int,
    n: Int,
) raises:
    """Run rolling high computation on GPU using warp-cooperative scanning.

    Uses one warp per output index for efficient parallel scanning.

    Args:
        ctx: GPU device context.
        prices_dev: Device buffer with trade prices.
        result_dev: Device buffer for output rolling high.
        window: Rolling window size (number of trades).
        n: Number of trades.
    """
    # Launch one warp per output index
    var num_warps = n
    var num_blocks = ceildiv(num_warps, WARPS_PER_BLOCK)

    ctx.enqueue_function_checked[_rolling_high_warp_kernel, _rolling_high_warp_kernel](
        prices_dev.unsafe_ptr(),
        result_dev.unsafe_ptr(),
        Int32(window),
        Int32(n),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


fn rolling_high_gpu_simple(
    ctx: DeviceContext,
    prices_dev: DeviceBuffer[DType.float64],
    result_dev: DeviceBuffer[DType.float64],
    window: Int,
    n: Int,
) raises:
    """Run rolling high computation on GPU using simple per-thread kernel.

    Each thread handles one output index. More efficient for small windows.

    Args:
        ctx: GPU device context.
        prices_dev: Device buffer with trade prices.
        result_dev: Device buffer for output rolling high.
        window: Rolling window size (number of trades).
        n: Number of trades.
    """
    var num_blocks = ceildiv(n, BLOCK_SIZE)

    ctx.enqueue_function_checked[_rolling_high_simple_kernel, _rolling_high_simple_kernel](
        prices_dev.unsafe_ptr(),
        result_dev.unsafe_ptr(),
        Int32(window),
        Int32(n),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


# =============================================================================
# Multi-Window GPU Function (all windows in single kernel launch)
# =============================================================================


fn _rolling_high_multi_kernel(
    prices: UnsafePointer[Float64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    windows: UnsafePointer[Int64, MutAnyOrigin],
    n_windows: Int32,
    n: Int32,
):
    """Multi-window rolling high kernel.

    Computes rolling highs for all windows in a single kernel launch.
    Output layout: result[w_idx * n + i] = rolling_high for window w_idx at index i

    Args:
        prices: Trade prices.
        result: Output buffer (size = n_windows * n).
        windows: Array of window sizes.
        n_windows: Number of windows.
        n: Number of trades.
    """
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    # Process all windows for this index
    for w_idx in range(Int(n_windows)):
        var window = Int(windows[w_idx])
        var out_idx = w_idx * Int(n) + i

        if i < window:
            result[out_idx] = 0.0
        else:
            var max_val = prices[i - window]
            for j in range(i - window + 1, i):
                var p = prices[j]
                if p > max_val:
                    max_val = p
            result[out_idx] = max_val


fn rolling_high_multi_gpu(
    ctx: DeviceContext,
    prices_dev: DeviceBuffer[DType.float64],
    result_dev: DeviceBuffer[DType.float64],
    windows_dev: DeviceBuffer[DType.int64],
    n_windows: Int,
    n: Int,
) raises:
    """Compute rolling highs for multiple windows in single kernel launch.

    Output layout: result[w_idx * n + i] = rolling_high for window w_idx at index i

    Args:
        ctx: GPU device context.
        prices_dev: Device buffer with trade prices.
        result_dev: Device buffer for output (size = n_windows * n).
        windows_dev: Device buffer with window sizes.
        n_windows: Number of windows.
        n: Number of trades.
    """
    var num_blocks = ceildiv(n, BLOCK_SIZE)

    ctx.enqueue_function_checked[_rolling_high_multi_kernel, _rolling_high_multi_kernel](
        prices_dev.unsafe_ptr(),
        result_dev.unsafe_ptr(),
        windows_dev.unsafe_ptr(),
        Int32(n_windows),
        Int32(n),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


# =============================================================================
# CPU Reference Implementation
# =============================================================================


fn rolling_high_cpu[
    origin: MutOrigin,
](
    prices: UnsafePointer[Float64],
    result: UnsafePointer[Float64, origin],
    window: Int,
    n: Int,
):
    """CPU reference implementation of rolling high.

    Uses the same algorithm as GPU for validation.

    Args:
        prices: Trade prices.
        result: Output rolling high array.
        window: Rolling window size.
        n: Number of trades.
    """
    for i in range(n):
        if i < window:
            result[i] = 0.0
            continue

        var max_val = prices[i - window]
        for j in range(i - window + 1, i):
            var p = prices[j]
            if p > max_val:
                max_val = p

        result[i] = max_val


fn rolling_high_cpu_parallel(
    prices: UnsafePointer[Float64],
    result: UnsafePointer[Float64, MutAnyOrigin],
    window: Int,
    n: Int,
):
    """CPU parallel implementation of rolling high.

    Uses Mojo's parallelize for multi-core execution.

    Args:
        prices: Trade prices.
        result: Output rolling high array.
        window: Rolling window size.
        n: Number of trades.
    """
    from algorithm import parallelize

    @parameter
    fn process_index(i: Int):
        if i < window:
            result[i] = 0.0
            return

        var max_val = prices[i - window]
        for j in range(i - window + 1, i):
            var p = prices[j]
            if p > max_val:
                max_val = p

        result[i] = max_val

    parallelize[process_index](n)
