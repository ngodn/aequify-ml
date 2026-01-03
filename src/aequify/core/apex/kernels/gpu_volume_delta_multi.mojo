"""
Multi-Window Volume Delta Kernel - GPU-accelerated volume delta over multiple windows.

This kernel computes volume delta (buy-sell imbalance) over the SAME windows
as rolling_high/rolling_low. This ensures the volume delta directly corresponds
to the price movement period we're detecting.

For LONG entries:
    - rolling_high detects price drop over window
    - volume_delta over SAME window shows selling pressure during the drop

For SHORT entries:
    - rolling_low detects price spike over window
    - volume_delta over SAME window shows buying pressure during the spike

Formula: delta = (buy_volume - sell_volume) / total_volume * 100

Output layout: result[w_idx * n + i] = volume delta for window w_idx at index i

Usage:
    volume_delta_multi_gpu(ctx, quantities_dev, sides_dev, result_dev,
                           windows_dev, n_windows, n)
"""

from math import ceildiv

from gpu import global_idx, lane_id
from gpu.host import DeviceContext, DeviceBuffer
from gpu.primitives.warp import lane_group_sum
from memory import UnsafePointer


# =============================================================================
# Architecture-Adaptive Constants (from device info at compile time)
# =============================================================================

comptime HW_INFO = DeviceContext.default_device_info
comptime WARP_SIZE: Int = HW_INFO.warp_size
comptime MAX_BLOCK_SIZE: Int = HW_INFO.max_thread_block_size
comptime BLOCK_SIZE: Int = 256 if MAX_BLOCK_SIZE >= 256 else MAX_BLOCK_SIZE
comptime WARPS_PER_BLOCK: Int = BLOCK_SIZE // WARP_SIZE
comptime SM_COUNT: Int = HW_INFO.sm_count


# =============================================================================
# Multi-Window Volume Delta Kernel
# =============================================================================


fn _volume_delta_multi_kernel(
    quantities: UnsafePointer[Float64, MutAnyOrigin],
    sides: UnsafePointer[Int64, MutAnyOrigin],
    result: UnsafePointer[Float64, MutAnyOrigin],
    windows: UnsafePointer[Int64, MutAnyOrigin],
    n_windows: Int32,
    n: Int32,
):
    """Multi-window volume delta kernel.

    Computes volume delta for all windows in a single kernel launch.
    Output layout: result[w_idx * n + i] = volume delta for window w_idx at index i

    For indices < window: output is 0.0 (no valid data)

    Args:
        quantities: Trade quantities (Float64).
        sides: Trade sides (1 = buy, -1 = sell).
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
            # Not enough data for this window
            result[out_idx] = 0.0
        else:
            # Compute buy/sell volumes over window [i-window, i)
            var buy_sum: Float64 = 0.0
            var sell_sum: Float64 = 0.0

            for j in range(i - window, i):
                var qty = quantities[j]
                var side = sides[j]

                if side == 1:
                    buy_sum += qty
                else:
                    sell_sum += qty

            var total = buy_sum + sell_sum
            if total > 0.0:
                result[out_idx] = (buy_sum - sell_sum) / total * 100.0
            else:
                result[out_idx] = 0.0


fn volume_delta_multi_gpu(
    ctx: DeviceContext,
    quantities_dev: DeviceBuffer[DType.float64],
    sides_dev: DeviceBuffer[DType.int64],
    result_dev: DeviceBuffer[DType.float64],
    windows_dev: DeviceBuffer[DType.int64],
    n_windows: Int,
    n: Int,
) raises:
    """Compute volume delta for multiple windows in single kernel launch.

    Uses the SAME windows as rolling_high/rolling_low for consistency.
    Output layout: result[w_idx * n + i] = volume delta for window w_idx at index i

    Args:
        ctx: GPU device context.
        quantities_dev: Device buffer with trade quantities.
        sides_dev: Device buffer with trade sides (1=buy, -1=sell).
        result_dev: Device buffer for output (size = n_windows * n).
        windows_dev: Device buffer with window sizes.
        n_windows: Number of windows.
        n: Number of trades.
    """
    var num_blocks = ceildiv(n, BLOCK_SIZE)

    ctx.enqueue_function_checked[_volume_delta_multi_kernel, _volume_delta_multi_kernel](
        quantities_dev.unsafe_ptr(),
        sides_dev.unsafe_ptr(),
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


fn volume_delta_multi_cpu[
    origin: MutOrigin,
](
    quantities: UnsafePointer[Float64],
    sides: UnsafePointer[Int64],
    result: UnsafePointer[Float64, origin],
    windows: UnsafePointer[Int64],
    n_windows: Int,
    n: Int,
):
    """CPU reference implementation of multi-window volume delta.

    Args:
        quantities: Trade quantities.
        sides: Trade sides (1=buy, -1=sell).
        result: Output buffer (size = n_windows * n).
        windows: Array of window sizes.
        n_windows: Number of windows.
        n: Number of trades.
    """
    for i in range(n):
        for w_idx in range(n_windows):
            var window = Int(windows[w_idx])
            var out_idx = w_idx * n + i

            if i < window:
                result[out_idx] = 0.0
            else:
                var buy_sum: Float64 = 0.0
                var sell_sum: Float64 = 0.0

                for j in range(i - window, i):
                    if sides[j] == 1:
                        buy_sum += quantities[j]
                    else:
                        sell_sum += quantities[j]

                var total = buy_sum + sell_sum
                if total > 0.0:
                    result[out_idx] = (buy_sum - sell_sum) / total * 100.0
                else:
                    result[out_idx] = 0.0


# =============================================================================
# Architecture Info Getters
# =============================================================================


fn get_warp_size() -> Int:
    """Get the warp/wavefront size for the target GPU."""
    return WARP_SIZE


fn get_block_size() -> Int:
    """Get the optimal block size for kernels."""
    return BLOCK_SIZE


fn get_warps_per_block() -> Int:
    """Get the number of warps per block."""
    return WARPS_PER_BLOCK


fn get_sm_count() -> Int:
    """Get the number of streaming multiprocessors."""
    return SM_COUNT
