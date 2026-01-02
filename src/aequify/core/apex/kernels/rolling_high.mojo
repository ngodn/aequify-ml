"""
Rolling High Kernel - Compute rolling maximum price within time window.

Both CPU and GPU implementations with SIMD vectorization and binary search
for O(log n) window boundary finding.

Usage:
    # CPU version - operates on pointers directly
    var result = rolling_high_cpu(timestamps, prices, result, lookback_ms, n)

    # GPU version - operates on device buffers
    rolling_high_gpu(ctx, timestamps_dev, prices_dev, result_dev, lookback_ms, n)
"""

from math import ceildiv
from sys import has_amd_gpu_accelerator, has_apple_gpu_accelerator, has_nvidia_gpu_accelerator
from sys.info import simd_width_of

from algorithm import parallelize
from gpu import block_dim, block_idx, thread_idx, global_idx
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host.info import AppleMetalFamily, AMDCDNA3Family, AMDRDNAFamily
from memory import UnsafePointer


# =============================================================================
# Architecture Constants
# =============================================================================

# SIMD width for Float64 on CPU (platform-dependent)
comptime CPU_SIMD_WIDTH_F64 = simd_width_of[DType.float64]()

# GPU SIMD width - architecture-dependent
# NVIDIA: 2 (128-bit loads), AMD CDNA: 4 (256-bit), AMD RDNA: 2, Apple: 2
@parameter
fn _get_gpu_simd_width() -> Int:
    @parameter
    if has_amd_gpu_accelerator():
        # AMD CDNA supports wider vector ops, RDNA uses 2
        return 4
    else:
        # NVIDIA and Apple Metal use 128-bit (2x Float64)
        return 2

comptime GPU_SIMD_WIDTH_F64: Int = _get_gpu_simd_width()

# Block size - architecture-dependent
# Uses max_thread_block_size from GPU family or sensible default
@parameter
fn _get_block_size() -> Int:
    @parameter
    if has_apple_gpu_accelerator():
        # Apple Metal: max is 1024, use 256 for good occupancy
        return 256
    elif has_amd_gpu_accelerator():
        # AMD: CDNA supports 1024, RDNA supports 1024, use 256 for balance
        return 256
    else:
        # NVIDIA: typically 256 or 512 works well
        return 256

comptime BLOCK_SIZE: Int = _get_block_size()


# =============================================================================
# Binary Search Helper
# =============================================================================


@always_inline
fn binary_search_lower_bound(
    timestamps: UnsafePointer[Int64],
    start: Int,
    end: Int,
    target: Int64,
) -> Int:
    """Binary search to find first index where timestamps[idx] >= target.

    Returns the smallest index i in [start, end) such that timestamps[i] >= target.
    If no such index exists, returns end.

    Complexity: O(log n) instead of O(n) linear scan.
    """
    var lo = start
    var hi = end

    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if timestamps[mid] < target:
            lo = mid + 1
        else:
            hi = mid

    return lo


# =============================================================================
# CPU Implementation
# =============================================================================


fn rolling_high_cpu[
    origin: MutOrigin,
](
    timestamps: UnsafePointer[Int64],
    prices: UnsafePointer[Float64],
    result: UnsafePointer[Float64, origin],
    lookback_ms: Int64,
    n: Int,
):
    """Compute rolling maximum price within time window (CPU, single-threaded).

    Uses SIMD vectorization for finding maximum within window.

    Args:
        timestamps: Pointer to timestamps array (sorted ascending).
        prices: Pointer to prices array.
        result: Pointer to output array for rolling highs.
        lookback_ms: Lookback window in milliseconds.
        n: Number of elements.
    """
    for i in range(n):
        var current_ts = timestamps[i]
        var cutoff_ts = current_ts - lookback_ms

        # Binary search to find first index >= cutoff_ts
        var start_idx = binary_search_lower_bound(timestamps, 0, i, cutoff_ts)

        if start_idx >= i:
            result[i] = 0.0
            continue

        # Find max with SIMD vectorized loads
        var max_val = prices[start_idx]
        var window_size = i - start_idx - 1

        var j = start_idx + 1
        var end_aligned = start_idx + 1 + (window_size // CPU_SIMD_WIDTH_F64) * CPU_SIMD_WIDTH_F64

        # SIMD vectorized max finding
        while j < end_aligned:
            var vec = prices.load[width=CPU_SIMD_WIDTH_F64](j)
            var local_max = vec.reduce_max()
            if local_max > max_val:
                max_val = local_max
            j += CPU_SIMD_WIDTH_F64

        # Handle remaining elements (scalar)
        while j < i:
            if prices[j] > max_val:
                max_val = prices[j]
            j += 1

        result[i] = max_val


fn rolling_high_cpu_parallel[
    origin: MutOrigin,
](
    timestamps: UnsafePointer[Int64],
    prices: UnsafePointer[Float64],
    result: UnsafePointer[Float64, origin],
    lookback_ms: Int64,
    n: Int,
):
    """Compute rolling maximum price within time window (CPU, parallel).

    Parallelizes across elements using work-stealing.

    Args:
        timestamps: Pointer to timestamps array (sorted ascending).
        prices: Pointer to prices array.
        result: Pointer to output array for rolling highs.
        lookback_ms: Lookback window in milliseconds.
        n: Number of elements.
    """
    # Capture pointers by value for the closure
    var ts_ptr = timestamps
    var px_ptr = prices
    var res_ptr = result
    var lb_ms = lookback_ms

    @parameter
    fn compute_rolling_high(i: Int):
        var current_ts = ts_ptr[i]
        var cutoff_ts = current_ts - lb_ms

        # Binary search to find first index >= cutoff_ts
        var start_idx = binary_search_lower_bound(ts_ptr, 0, i, cutoff_ts)

        if start_idx >= i:
            res_ptr[i] = 0.0
            return

        # Find max with SIMD vectorized loads
        var max_val = px_ptr[start_idx]
        var window_size = i - start_idx - 1

        var j = start_idx + 1
        var end_aligned = start_idx + 1 + (window_size // CPU_SIMD_WIDTH_F64) * CPU_SIMD_WIDTH_F64

        # SIMD vectorized max finding
        while j < end_aligned:
            var vec = px_ptr.load[width=CPU_SIMD_WIDTH_F64](j)
            var local_max = vec.reduce_max()
            if local_max > max_val:
                max_val = local_max
            j += CPU_SIMD_WIDTH_F64

        # Handle remaining elements (scalar)
        while j < i:
            if px_ptr[j] > max_val:
                max_val = px_ptr[j]
            j += 1

        res_ptr[i] = max_val

    parallelize[compute_rolling_high](n)


# =============================================================================
# GPU Implementation
# =============================================================================


fn _rolling_high_gpu_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    lookback_ms: Int64,
    n: Int32,
):
    """GPU kernel for rolling maximum price within time window.

    Each thread processes one element.
    Uses binary search for O(log n) window boundary finding.
    Uses SIMD vec2 loads for finding maximum.
    """
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    var current_ts = timestamps[i]
    var cutoff_ts = current_ts - lookback_ms

    # Binary search to find first index >= cutoff_ts
    var lo = 0
    var hi = i
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if timestamps[mid] < cutoff_ts:
            lo = mid + 1
        else:
            hi = mid
    var start_idx = lo

    if start_idx >= i:
        result[i] = 0.0
        return

    # Find max with SIMD vectorized loads
    var max_val = prices[start_idx]
    var window_size = i - start_idx - 1

    var j = start_idx + 1
    var end_aligned = start_idx + 1 + (window_size // GPU_SIMD_WIDTH_F64) * GPU_SIMD_WIDTH_F64

    # SIMD vec2 loads with reduce_max
    while j < end_aligned:
        var vec = prices.load[width=GPU_SIMD_WIDTH_F64](j)
        var local_max = vec.reduce_max()
        if local_max > max_val:
            max_val = local_max
        j += GPU_SIMD_WIDTH_F64

    # Handle remaining elements (scalar)
    while j < i:
        if prices[j] > max_val:
            max_val = prices[j]
        j += 1

    result[i] = max_val


fn rolling_high_gpu(
    ctx: DeviceContext,
    timestamps_dev: DeviceBuffer[DType.int64],
    prices_dev: DeviceBuffer[DType.float64],
    result_dev: DeviceBuffer[DType.float64],
    lookback_ms: Int64,
    n: Int,
) raises:
    """Launch rolling high GPU kernel.

    Args:
        ctx: GPU device context.
        timestamps_dev: Device buffer with timestamps.
        prices_dev: Device buffer with prices.
        result_dev: Device buffer for output (must be pre-allocated).
        lookback_ms: Lookback window in milliseconds.
        n: Number of elements.
    """
    var num_blocks = ceildiv(n, BLOCK_SIZE)

    ctx.enqueue_function_checked[_rolling_high_gpu_kernel, _rolling_high_gpu_kernel](
        timestamps_dev.unsafe_ptr(),
        prices_dev.unsafe_ptr(),
        result_dev.unsafe_ptr(),
        lookback_ms,
        Int32(n),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


fn compute_rolling_high[
    origin: MutOrigin,
](
    timestamps: UnsafePointer[Int64],
    prices: UnsafePointer[Float64],
    result: UnsafePointer[Float64, origin],
    lookback_ms: Int64,
    n: Int,
    parallel: Bool = True,
):
    """Compute rolling high on CPU with optional parallelization.

    Args:
        timestamps: Pointer to timestamps array.
        prices: Pointer to prices array.
        result: Pointer to output array.
        lookback_ms: Lookback window in milliseconds.
        n: Number of elements.
        parallel: Whether to use parallel computation (default: True).
    """
    if parallel:
        rolling_high_cpu_parallel(timestamps, prices, result, lookback_ms, n)
    else:
        rolling_high_cpu(timestamps, prices, result, lookback_ms, n)
