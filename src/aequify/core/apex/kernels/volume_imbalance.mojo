"""
Optimized Volume Imbalance Kernel - Using prefix sums for O(1) queries.

This implementation uses price bucketing and prefix sums to achieve
O(n) total work instead of O(n * window_size) of the naive approach.

Architecture:
    1. Detect available VRAM and compute max trades dynamically
    2. Bucket trades by price level
    3. Build prefix sums per bucket (buy/sell volumes)
    4. Query volume imbalance using prefix sums (O(1) per output)

Usage:
    # Get max trades based on available VRAM
    max_trades = get_max_trades_for_vram(tolerance_pct=2.5)

    # Run kernel
    volume_imbalance_gpu(ctx, timestamps, prices, quantities, sides, result, tolerance_pct, n)
"""

from math import ceildiv, floor, ceil
from sys import has_amd_gpu_accelerator, has_apple_gpu_accelerator, has_nvidia_gpu_accelerator, has_accelerator

from algorithm import parallelize
from gpu import global_idx, block_idx, thread_idx, block_dim, warp, block, barrier
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer, stack_allocation, memset_zero, alloc
from memory.pointer import AddressSpace
from python import Python, PythonObject


# =============================================================================
# Architecture Constants
# =============================================================================

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


# =============================================================================
# Session Constants (same as original)
# =============================================================================

comptime MS_PER_HOUR: Int64 = 3600000
comptime MS_PER_DAY: Int64 = 86400000


# =============================================================================
# VRAM Detection and Capacity Planning
# =============================================================================


fn get_vram_info_nvidia() raises -> Tuple[Int, Int]:
    """Get VRAM info using pynvml (nvidia-ml-py).

    Returns:
        Tuple of (free_bytes, total_bytes).
    """
    var pynvml = Python.import_module("pynvml")
    pynvml.nvmlInit()

    var handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    var mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    var free_bytes = Int(mem_info.free)
    var total_bytes = Int(mem_info.total)

    pynvml.nvmlShutdown()

    return (free_bytes, total_bytes)


fn get_vram_info_mojo() raises -> Tuple[Int, Int]:
    """Get VRAM info using Mojo DeviceContext.

    Returns:
        Tuple of (free_bytes, total_bytes).
    """
    var ctx = DeviceContext()
    var mem_info = ctx.get_memory_info()
    return (Int(mem_info[0]), Int(mem_info[1]))


fn get_available_vram_bytes() raises -> Int:
    """Get available VRAM in bytes.

    Tries pynvml first (more accurate), falls back to Mojo DeviceContext.
    """
    try:
        var info = get_vram_info_nvidia()
        return info[0]
    except:
        try:
            var info = get_vram_info_mojo()
            return info[0]
        except:
            # Fallback: assume 8GB available
            return 8 * 1024 * 1024 * 1024


fn compute_memory_per_trade(num_buckets: Int) -> Int:
    """Compute memory required per trade for the prefix sum approach.

    Memory breakdown per trade:
        - timestamp: 8 bytes (Int64)
        - price: 8 bytes (Float64)
        - quantity: 8 bytes (Float64)
        - side: 4 bytes (Int32)
        - result: 8 bytes (Float64)
        - bucket_index: 4 bytes (Int32)
        - prefix_sums: num_buckets * 2 * 8 bytes (buy + sell per bucket)

    Args:
        num_buckets: Number of price buckets.

    Returns:
        Bytes per trade.
    """
    var base_bytes = 8 + 8 + 8 + 4 + 8 + 4  # 40 bytes
    var prefix_bytes = num_buckets * 2 * 8   # 16 * num_buckets bytes
    return base_bytes + prefix_bytes


fn compute_num_buckets(tolerance_pct: Float64) -> Int:
    """Compute optimal number of buckets based on tolerance.

    Rule of thumb: bucket_size = tolerance / 5 for good granularity.
    For 100% price range coverage.

    Args:
        tolerance_pct: Price tolerance percentage (e.g., 2.5 for 2.5%).

    Returns:
        Number of buckets.
    """
    var bucket_size_pct = tolerance_pct / 5.0
    var num_buckets = Int(ceil(100.0 / bucket_size_pct))
    # Clamp between reasonable bounds
    return max(50, min(500, num_buckets))


fn get_max_trades_for_vram(tolerance_pct: Float64 = 2.5, safety_margin: Float64 = 0.85) raises -> Int:
    """Calculate maximum number of trades that fit in available VRAM.

    Args:
        tolerance_pct: Price tolerance percentage.
        safety_margin: Fraction of VRAM to use (0.85 = 85%).

    Returns:
        Maximum number of trades.
    """
    var available_bytes = get_available_vram_bytes()
    var usable_bytes = Int(Float64(available_bytes) * safety_margin)

    var num_buckets = compute_num_buckets(tolerance_pct)
    var bytes_per_trade = compute_memory_per_trade(num_buckets)

    var max_trades = usable_bytes // bytes_per_trade

    return max_trades


fn get_memory_config(n_trades: Int, tolerance_pct: Float64 = 2.5) raises -> PythonObject:
    """Get memory configuration as Python dict for debugging/logging.

    Args:
        n_trades: Number of trades to process.
        tolerance_pct: Price tolerance percentage.

    Returns:
        Python dict with memory configuration.
    """
    var num_buckets = compute_num_buckets(tolerance_pct)
    var bytes_per_trade = compute_memory_per_trade(num_buckets)
    var total_bytes = n_trades * bytes_per_trade

    var available_vram = get_available_vram_bytes()
    var max_trades = get_max_trades_for_vram(tolerance_pct)

    var info = Python.dict()
    info.__setitem__(PythonObject("num_buckets"), value=num_buckets)
    info.__setitem__(PythonObject("bytes_per_trade"), value=bytes_per_trade)
    info.__setitem__(PythonObject("total_bytes_required"), value=total_bytes)
    info.__setitem__(PythonObject("total_gb_required"), value=Float64(total_bytes) / (1024.0 * 1024.0 * 1024.0))
    info.__setitem__(PythonObject("available_vram_bytes"), value=available_vram)
    info.__setitem__(PythonObject("available_vram_gb"), value=Float64(available_vram) / (1024.0 * 1024.0 * 1024.0))
    info.__setitem__(PythonObject("max_trades"), value=max_trades)
    info.__setitem__(PythonObject("fits_in_vram"), value=n_trades <= max_trades)
    info.__setitem__(PythonObject("tolerance_pct"), value=tolerance_pct)

    return info


# =============================================================================
# Session Helpers (same as original)
# =============================================================================


@always_inline
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


# =============================================================================
# Price Bucketing
# =============================================================================


fn compute_price_range_cpu(
    prices: UnsafePointer[Float64],
    n: Int,
) -> Tuple[Float64, Float64]:
    """Compute min and max prices (CPU)."""
    if n == 0:
        return (0.0, 0.0)

    var min_price = prices[0]
    var max_price = prices[0]

    for i in range(1, n):
        var p = prices[i]
        if p < min_price:
            min_price = p
        if p > max_price:
            max_price = p

    return (min_price, max_price)


fn _compute_bucket_indices_kernel(
    prices: UnsafePointer[Float64, MutAnyOrigin],
    bucket_indices: UnsafePointer[Int32, MutAnyOrigin],
    min_price: Float64,
    bucket_size: Float64,
    num_buckets: Int32,
    n: Int32,
):
    """GPU kernel to compute bucket index for each trade."""
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    var price = prices[i]
    var bucket = Int32(floor((price - min_price) / bucket_size))

    # Clamp to valid range
    if bucket < 0:
        bucket = 0
    if bucket >= num_buckets:
        bucket = num_buckets - 1

    bucket_indices[i] = bucket


# =============================================================================
# Prefix Sum Building - Per Bucket
# =============================================================================


fn _build_prefix_sums_kernel(
    quantities: UnsafePointer[Float64, MutAnyOrigin],
    sides: UnsafePointer[Int32, MutAnyOrigin],
    bucket_indices: UnsafePointer[Int32, MutAnyOrigin],
    buy_prefix: UnsafePointer[Float64, MutAnyOrigin],
    sell_prefix: UnsafePointer[Float64, MutAnyOrigin],
    target_bucket: Int32,
    n: Int32,
):
    """GPU kernel to build prefix sums for a single bucket.

    For bucket b, computes:
        buy_prefix[b][i] = sum of buy volumes for bucket b from 0 to i
        sell_prefix[b][i] = sum of sell volumes for bucket b from 0 to i

    This is a simple sequential scan - we launch one block per bucket.
    Uses block-level prefix sum for efficiency.
    """
    var tid = Int(thread_idx.x)
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    # Each thread checks if its trade belongs to target bucket
    var belongs_to_bucket = bucket_indices[i] == target_bucket
    var qty = quantities[i] if belongs_to_bucket else 0.0
    var is_buy = sides[i] == 1 if belongs_to_bucket else False

    var buy_val = qty if is_buy else 0.0
    var sell_val = qty if (belongs_to_bucket and not is_buy) else 0.0

    # Use block prefix sum
    var buy_prefix_val = block.prefix_sum[exclusive=False, block_size=BLOCK_SIZE](buy_val)
    var sell_prefix_val = block.prefix_sum[exclusive=False, block_size=BLOCK_SIZE](sell_val)

    buy_prefix[i] = buy_prefix_val
    sell_prefix[i] = sell_prefix_val


fn _build_prefix_sums_sequential_kernel(
    quantities: UnsafePointer[Float64, MutAnyOrigin],
    sides: UnsafePointer[Int32, MutAnyOrigin],
    bucket_indices: UnsafePointer[Int32, MutAnyOrigin],
    buy_prefix: UnsafePointer[Float64, MutAnyOrigin],
    sell_prefix: UnsafePointer[Float64, MutAnyOrigin],
    num_buckets: Int32,
    n: Int32,
):
    """GPU kernel to build all prefix sums sequentially (one thread per bucket).

    Each thread handles one bucket and scans through all trades.
    This is O(n * num_buckets) total work but highly parallel.
    """
    var bucket = Int(global_idx.x)

    if bucket >= Int(num_buckets):
        return

    var buy_sum: Float64 = 0.0
    var sell_sum: Float64 = 0.0
    var offset = bucket * Int(n)

    for i in range(Int(n)):
        if bucket_indices[i] == Int32(bucket):
            var qty = quantities[i]
            if sides[i] == 1:
                buy_sum += qty
            else:
                sell_sum += qty

        buy_prefix[offset + i] = buy_sum
        sell_prefix[offset + i] = sell_sum


# =============================================================================
# Optimized Volume Imbalance Query Kernel
# =============================================================================


fn _volume_imbalance_query_kernel(
    timestamps: UnsafePointer[Int64, MutAnyOrigin],
    prices: UnsafePointer[Float64, MutAnyOrigin],
    buy_prefix: UnsafePointer[Float64, MutAnyOrigin],
    sell_prefix: UnsafePointer[Float64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    min_price: Float64,
    bucket_size: Float64,
    tolerance_pct: Float64,
    num_buckets: Int32,
    n: Int32,
):
    """GPU kernel to query volume imbalance using prefix sums.

    For each output i:
        1. Binary search to find session start
        2. Compute price range [low, high] from tolerance
        3. Find bucket range that covers [low, high]
        4. Sum prefix differences across relevant buckets
        5. Compute imbalance

    Complexity: O(num_buckets_in_range) per output ≈ O(1)
    """
    var i = Int(global_idx.x)

    if i >= Int(n):
        return

    var current_ts = timestamps[i]
    var current_price = prices[i]
    var tolerance = current_price * (tolerance_pct / 100.0)
    var price_low = current_price - tolerance
    var price_high = current_price + tolerance

    # Get session start
    var session_start_ms = get_previous_session_start_ms(current_ts)

    # Binary search for session start index
    var lo = 0
    var hi = i
    while lo < hi:
        var mid = (lo + hi) // 2
        if timestamps[mid] < session_start_ms:
            lo = mid + 1
        else:
            hi = mid
    var start_idx = lo

    # Convert price range to bucket range
    var bucket_low = Int(floor((price_low - min_price) / bucket_size))
    var bucket_high = Int(floor((price_high - min_price) / bucket_size))

    # Clamp to valid range
    if bucket_low < 0:
        bucket_low = 0
    if bucket_high >= Int(num_buckets):
        bucket_high = Int(num_buckets) - 1
    if bucket_high < bucket_low:
        bucket_high = bucket_low

    # Sum prefix differences across relevant buckets
    var buy_sum: Float64 = 0.0
    var sell_sum: Float64 = 0.0

    for b in range(bucket_low, bucket_high + 1):
        var offset = b * Int(n)

        # prefix[i] - prefix[start_idx - 1] gives sum in range [start_idx, i]
        var buy_at_i = buy_prefix[offset + i]
        var sell_at_i = sell_prefix[offset + i]

        var buy_before: Float64 = 0.0
        var sell_before: Float64 = 0.0
        if start_idx > 0:
            buy_before = buy_prefix[offset + start_idx - 1]
            sell_before = sell_prefix[offset + start_idx - 1]

        buy_sum += buy_at_i - buy_before
        sell_sum += sell_at_i - sell_before

    var total = buy_sum + sell_sum
    if total > 0:
        result[i] = (buy_sum - sell_sum) / total * 100.0
    else:
        result[i] = 0.0


# =============================================================================
# Main GPU Function
# =============================================================================


fn volume_imbalance_gpu(
    ctx: DeviceContext,
    timestamps_dev: DeviceBuffer[DType.int64],
    prices_dev: DeviceBuffer[DType.float64],
    quantities_dev: DeviceBuffer[DType.float64],
    sides_dev: DeviceBuffer[DType.int32],
    result_dev: DeviceBuffer[DType.float64],
    min_price: Float64,
    max_price: Float64,
    tolerance_pct: Float64,
    n: Int,
) raises:
    """Run optimized volume imbalance computation on GPU.

    Uses prefix sums for O(n * num_buckets) total work instead of O(n * window_size).

    Args:
        ctx: GPU device context.
        timestamps_dev: Device buffer with timestamps.
        prices_dev: Device buffer with prices.
        quantities_dev: Device buffer with quantities.
        sides_dev: Device buffer with sides.
        result_dev: Device buffer for output.
        min_price: Minimum price in dataset.
        max_price: Maximum price in dataset.
        tolerance_pct: Price tolerance percentage.
        n: Number of trades.
    """
    var num_buckets = compute_num_buckets(tolerance_pct)
    var price_range = max_price - min_price
    var bucket_size = price_range / Float64(num_buckets)

    # Ensure bucket_size is positive
    if bucket_size <= 0:
        bucket_size = max_price * 0.001  # Fallback: 0.1% of max price

    # Allocate intermediate buffers
    var bucket_indices_dev = ctx.enqueue_create_buffer[DType.int32](n)
    var buy_prefix_dev = ctx.enqueue_create_buffer[DType.float64](n * num_buckets)
    var sell_prefix_dev = ctx.enqueue_create_buffer[DType.float64](n * num_buckets)

    var num_blocks = ceildiv(n, BLOCK_SIZE)

    # Step 1: Compute bucket indices
    ctx.enqueue_function_checked[_compute_bucket_indices_kernel, _compute_bucket_indices_kernel](
        prices_dev.unsafe_ptr(),
        bucket_indices_dev.unsafe_ptr(),
        min_price,
        bucket_size,
        Int32(num_buckets),
        Int32(n),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    # Step 2: Build prefix sums (one thread per bucket, sequential scan)
    var bucket_blocks = ceildiv(num_buckets, BLOCK_SIZE)
    ctx.enqueue_function_checked[_build_prefix_sums_sequential_kernel, _build_prefix_sums_sequential_kernel](
        quantities_dev.unsafe_ptr(),
        sides_dev.unsafe_ptr(),
        bucket_indices_dev.unsafe_ptr(),
        buy_prefix_dev.unsafe_ptr(),
        sell_prefix_dev.unsafe_ptr(),
        Int32(num_buckets),
        Int32(n),
        grid_dim=bucket_blocks,
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    # Step 3: Query volume imbalance using prefix sums
    ctx.enqueue_function_checked[_volume_imbalance_query_kernel, _volume_imbalance_query_kernel](
        timestamps_dev.unsafe_ptr(),
        prices_dev.unsafe_ptr(),
        buy_prefix_dev.unsafe_ptr(),
        sell_prefix_dev.unsafe_ptr(),
        result_dev.unsafe_ptr(),
        min_price,
        bucket_size,
        tolerance_pct,
        Int32(num_buckets),
        Int32(n),
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # Cleanup intermediate buffers
    _ = bucket_indices_dev^
    _ = buy_prefix_dev^
    _ = sell_prefix_dev^


# =============================================================================
# CPU Reference Implementation (for validation)
# =============================================================================


fn volume_imbalance_cpu[
    origin: MutOrigin,
](
    timestamps: UnsafePointer[Int64],
    prices: UnsafePointer[Float64],
    quantities: UnsafePointer[Float64],
    sides: UnsafePointer[Int32],
    result: UnsafePointer[Float64, origin],
    min_price: Float64,
    max_price: Float64,
    tolerance_pct: Float64,
    n: Int,
):
    """CPU reference implementation using prefix sums.

    Uses the same algorithm as GPU but runs sequentially for validation.
    """
    var num_buckets = compute_num_buckets(tolerance_pct)
    var price_range = max_price - min_price
    var bucket_size = price_range / Float64(num_buckets)

    if bucket_size <= 0:
        bucket_size = max_price * 0.001

    # Allocate bucket indices and prefix sums
    var bucket_indices = alloc[Int32](n)
    var buy_prefix = alloc[Float64](n * num_buckets)
    var sell_prefix = alloc[Float64](n * num_buckets)

    # Initialize to zero
    for i in range(n * num_buckets):
        buy_prefix[i] = 0.0
        sell_prefix[i] = 0.0

    # Step 1: Compute bucket indices
    for i in range(n):
        var price = prices[i]
        var bucket = Int32(floor((price - min_price) / bucket_size))
        if bucket < 0:
            bucket = 0
        if bucket >= Int32(num_buckets):
            bucket = Int32(num_buckets - 1)
        bucket_indices[i] = bucket

    # Step 2: Build prefix sums per bucket
    for b in range(num_buckets):
        var offset = b * n
        var buy_sum: Float64 = 0.0
        var sell_sum: Float64 = 0.0

        for i in range(n):
            if bucket_indices[i] == Int32(b):
                var qty = quantities[i]
                if sides[i] == 1:
                    buy_sum += qty
                else:
                    sell_sum += qty

            buy_prefix[offset + i] = buy_sum
            sell_prefix[offset + i] = sell_sum

    # Step 3: Query volume imbalance
    for i in range(n):
        var current_ts = timestamps[i]
        var current_price = prices[i]
        var tolerance = current_price * (tolerance_pct / 100.0)
        var price_low = current_price - tolerance
        var price_high = current_price + tolerance

        var session_start_ms = get_previous_session_start_ms(current_ts)

        # Binary search for start index
        var lo = 0
        var hi = i
        while lo < hi:
            var mid = (lo + hi) // 2
            if timestamps[mid] < session_start_ms:
                lo = mid + 1
            else:
                hi = mid
        var start_idx = lo

        # Bucket range
        var bucket_low = Int(floor((price_low - min_price) / bucket_size))
        var bucket_high = Int(floor((price_high - min_price) / bucket_size))

        if bucket_low < 0:
            bucket_low = 0
        if bucket_high >= num_buckets:
            bucket_high = num_buckets - 1
        if bucket_high < bucket_low:
            bucket_high = bucket_low

        # Sum across buckets
        var buy_sum: Float64 = 0.0
        var sell_sum: Float64 = 0.0

        for b in range(bucket_low, bucket_high + 1):
            var offset = b * n
            var buy_at_i = buy_prefix[offset + i]
            var sell_at_i = sell_prefix[offset + i]

            var buy_before: Float64 = 0.0
            var sell_before: Float64 = 0.0
            if start_idx > 0:
                buy_before = buy_prefix[offset + start_idx - 1]
                sell_before = sell_prefix[offset + start_idx - 1]

            buy_sum += buy_at_i - buy_before
            sell_sum += sell_at_i - sell_before

        var total = buy_sum + sell_sum
        if total > 0:
            result[i] = (buy_sum - sell_sum) / total * 100.0
        else:
            result[i] = 0.0

    # Cleanup
    bucket_indices.free()
    buy_prefix.free()
    sell_prefix.free()
