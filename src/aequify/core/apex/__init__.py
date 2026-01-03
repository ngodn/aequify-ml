"""
APEX - Adaptive Parameter Exploration.

GPU-accelerated parameter optimization for trading signals.
"""

from aequify.core.apex.bootstrap import (
    APEX,
    APEXManager,
    BootstrapResult,
    DirectionResult,
    bootstrap_symbol,
    generate_override_for_symbol,
    generate_override_yaml,
    get_gpu_info,
    is_gpu_available,
    write_override_file,
)
from aequify.core.apex.config import (
    APEXConfig,
    DEFAULT_CONFIG,
    DirectionalParameters,
    ParameterBounds,
    VADParameters,
    load_apex_config,
    load_apex_config_for_symbol,
)
from aequify.core.apex.signal_handler import (
    APEXSignalHandler,
    SignalConfig,
    SignalState,
)
from aequify.core.apex.live import (
    APEXLiveDetector,
    APEXLiveDetectorManager,
    LiveDetectorConfig,
    LiveMetrics,
    is_live_kernels_available,
)

__all__ = [
    # Bootstrap
    "APEX",
    "APEXManager",
    "BootstrapResult",
    "DirectionResult",
    "bootstrap_symbol",
    "is_gpu_available",
    "get_gpu_info",
    # Override Generation
    "generate_override_yaml",
    "generate_override_for_symbol",
    "write_override_file",
    # Config
    "APEXConfig",
    "DEFAULT_CONFIG",
    "DirectionalParameters",
    "ParameterBounds",
    "VADParameters",
    "load_apex_config",
    "load_apex_config_for_symbol",
    # Signal Handler
    "APEXSignalHandler",
    "SignalConfig",
    "SignalState",
    # Live Detector
    "APEXLiveDetector",
    "APEXLiveDetectorManager",
    "LiveDetectorConfig",
    "LiveMetrics",
    "is_live_kernels_available",
]
