"""APEX Kernels - GPU-accelerated compute kernels for trading analytics.

Kernels:
    - gpu_volume_delta_multi: Multi-window volume delta (same windows as rolling_high/low)
    - gpu_rolling_high: Rolling maximum price over time window
    - gpu_rolling_low: Rolling minimum price over time window
    - gpu_grid_search_long_dca: LONG position grid search with DCA support
    - gpu_grid_search_short_dca: SHORT position grid search with DCA support
"""

# Multi-Window Volume Delta - same windows as rolling_high/rolling_low
from .gpu_volume_delta_multi import (
    volume_delta_multi_gpu,
    volume_delta_multi_cpu,
)

# Rolling High - rolling maximum price over time window
from .gpu_rolling_high import (
    rolling_high_gpu,
    rolling_high_gpu_simple,
    rolling_high_multi_gpu,
    rolling_high_cpu,
    rolling_high_cpu_parallel,
)

# Rolling Low - rolling minimum price over time window
from .gpu_rolling_low import (
    rolling_low_gpu,
    rolling_low_gpu_simple,
    rolling_low_multi_gpu,
    rolling_low_cpu,
    rolling_low_cpu_parallel,
    NO_LOW_SENTINEL,
)

# Grid Search LONG DCA - parameter optimization for LONG positions with DCA
from .gpu_grid_search_long_dca import (
    grid_search_long_dca_gpu,
    get_initial_size,
    get_dca_multiplier,
    get_max_position,
    get_params_per_combo,
)

# Grid Search SHORT DCA - parameter optimization for SHORT positions with DCA
from .gpu_grid_search_short_dca import (
    grid_search_short_dca_gpu,
)

# Architecture info getters
from .gpu_volume_delta_multi import (
    get_warp_size,
    get_block_size,
    get_warps_per_block,
    get_sm_count,
)
