"""APEX Kernels - GPU-accelerated compute kernels for trading analytics."""

from .rolling_high import (
    rolling_high_cpu,
    rolling_high_cpu_parallel,
    rolling_high_gpu,
    compute_rolling_high,
)
from .rolling_low import (
    rolling_low_cpu,
    rolling_low_cpu_parallel,
    rolling_low_gpu,
    compute_rolling_low,
    NO_LOW_SENTINEL,
)
from .volume_imbalance import (
    volume_imbalance_gpu,
    volume_imbalance_cpu,
    get_max_trades_for_vram,
    get_memory_config,
    compute_num_buckets,
    compute_price_range_cpu,
)
from .grid_search import (
    grid_search_long_gpu,
    grid_search_short_gpu,
    get_block_size,
    get_warps_per_block,
    get_warp_size,
    get_start_idx,
    get_min_gap,
)
# from .benchmark import run_benchmarks
# Note: bootstrap is built separately as a shared library, not part of the package
# from .bootstrap import PyInit_bootstrap
# from .live_price_move import (
#     price_move_from_high_cpu,
#     price_move_from_low_cpu,
#     find_rolling_high_cpu,
#     find_rolling_low_cpu,
#     compute_price_moves_multi_window,
#     price_move_from_high_window,
#     price_move_from_low_window,
# )
# from .live_volume_imbalance import (
#     volume_imbalance_cpu,
#     volume_imbalance_cpu_parallel,
#     volume_imbalance_window,
#     volume_imbalance_near_price,
#     cumulative_delta_from_timestamp,
#     compute_volume_stats,
#     VolumeStats,
# )
