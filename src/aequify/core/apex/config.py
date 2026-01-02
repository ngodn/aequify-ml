"""
APEX Configuration - Parameters and bounds for optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParameterBounds:
    """Bounds for a single parameter."""

    min: float
    max: float
    default: float
    step: float  # Grid search step size

    def grid_values(self) -> list[float]:
        """Generate grid values for this parameter."""
        values = []
        val = self.min
        while val <= self.max + 1e-9:  # Small epsilon for float comparison
            values.append(round(val, 4))
            val += self.step
        return values

    def clamp(self, value: float) -> float:
        """Clamp value to bounds."""
        return max(self.min, min(self.max, value))


@dataclass
class VADParameters:
    """
    Volume Anomaly Detector parameters for a single direction.

    Signal parameters control entry detection.
    Position parameters control exit (TP from session levels, SL fixed, max hold time).
    DCA parameters control dollar-cost averaging on drawdown.
    """

    # Signal parameters
    price_move: float = -3.0  # Price move % to trigger (neg=drop for LONG, pos=rise for SHORT)
    time_window: int = 300  # Lookback window in ms
    delta_threshold: float = -70.0  # Volume delta % threshold

    # Position parameters
    # Note: TP comes from session levels at runtime, this is fallback only
    target_profit: float = 1.0  # Fallback TP % (used when no session levels)
    stop_loss: float = -0.5  # Stop loss %
    max_hold_time: int = 30000  # Max hold time in ms

    # DCA parameters
    dca_distance_pct: float = -5.0  # Price drop % from avg entry to trigger DCA (neg for LONG, pos for SHORT)

    def to_tuple(self) -> tuple[float, int, float, float, float, int, float]:
        """Convert to tuple for optimization."""
        return (
            self.price_move,
            self.time_window,
            self.delta_threshold,
            self.target_profit,
            self.stop_loss,
            self.max_hold_time,
            self.dca_distance_pct,
        )

    @classmethod
    def from_tuple(cls, t: tuple) -> VADParameters:
        """Create from tuple."""
        if len(t) == 3:
            # Legacy format without position params
            return cls(price_move=t[0], time_window=int(t[1]), delta_threshold=t[2])
        elif len(t) == 5:
            # Format without max_hold_time
            return cls(
                price_move=t[0],
                time_window=int(t[1]),
                delta_threshold=t[2],
                target_profit=t[3],
                stop_loss=t[4],
            )
        elif len(t) == 6:
            # Format without dca_distance_pct
            return cls(
                price_move=t[0],
                time_window=int(t[1]),
                delta_threshold=t[2],
                target_profit=t[3],
                stop_loss=t[4],
                max_hold_time=int(t[5]),
            )
        # Current format: 7 params with DCA
        return cls(
            price_move=t[0],
            time_window=int(t[1]),
            delta_threshold=t[2],
            target_profit=t[3],
            stop_loss=t[4],
            max_hold_time=int(t[5]),
            dca_distance_pct=t[6],
        )

    def __repr__(self) -> str:
        return (
            f"VAD(pm={self.price_move:+.1f}%, tw={self.time_window}ms, "
            f"dt={self.delta_threshold:+.0f}%, tp={self.target_profit:.1f}%, sl={self.stop_loss:.1f}%, "
            f"hold={self.max_hold_time / 1000:.0f}s, dca={self.dca_distance_pct:+.1f}%)"
        )


@dataclass
class DirectionalParameters:
    """Parameters for both LONG and SHORT directions."""

    long: VADParameters | None = None
    short: VADParameters | None = None

    def __repr__(self) -> str:
        parts = []
        if self.long:
            parts.append(f"LONG={self.long}")
        if self.short:
            parts.append(f"SHORT={self.short}")
        return f"DirectionalParams({', '.join(parts) or 'none'})"


@dataclass
class PositionParameters:
    """Virtual position simulation parameters."""

    target_profit: float = 0.3  # Take profit %
    stop_loss: float = -0.5  # Stop loss %
    max_hold_time: int = 120000  # Max hold in ms (120 seconds)


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap optimization."""

    max_trades: int = 0  # 0 = use all available trades, or limit to N trades

    @classmethod
    def from_dict(cls, data: dict) -> BootstrapConfig:
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(max_trades=data.get("max_trades", 0))


@dataclass
class APEXConfig:
    """
    Configuration for APEX optimizer.

    Attributes:
        min_warm_duration_hours: Minimum hours of data required for bootstrap
        max_warm_age_hours: Maximum age of warm data to be considered valid
        max_cache_valid_days: If cache is younger than this, skip backfill entirely
        bootstrap: Bootstrap optimization configuration
        continuous_learning: Continuous learning configuration
        min_entries_for_validity: Minimum entries for statistical significance
        imbalance_price_tolerance_pct: Price tolerance % for volume imbalance calculation
    """

    # Data validation
    min_warm_duration_hours: float = 1.0
    max_warm_age_hours: float = 24.0

    # Cache validity: if cache age < max_cache_valid_days, skip backfill + GPU entirely
    max_cache_valid_days: float = 7.0

    # Bootstrap config
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    # Validity
    min_entries_for_validity: int = 5

    # Volume imbalance price tolerance (footprint/order flow style)
    # Measures buy/sell imbalance at trades within X% of current price
    imbalance_price_tolerance_pct: float = 1.0  # 1% price range

    # Position simulation
    position: PositionParameters = field(default_factory=PositionParameters)

    # ==========================================================================
    # LONG Parameter bounds (price drops + selling pressure)
    # ==========================================================================
    long_price_move_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=-3.0, max=-0.5, default=-2.0, step=0.5)
    )

    long_delta_threshold_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=-70, max=-30, default=-50, step=10)
    )

    # ==========================================================================
    # SHORT Parameter bounds (price rises + buying pressure)
    # ==========================================================================
    short_price_move_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=0.5, max=3.0, default=2.0, step=0.5)
    )

    short_delta_threshold_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=30, max=70, default=50, step=10)
    )

    # ==========================================================================
    # DCA Parameter bounds
    # ==========================================================================
    # LONG DCA: negative values (price must DROP by this % from avg entry)
    long_dca_distance_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=-40.0, max=-2.0, default=-5.0, step=2.0)
    )

    # SHORT DCA: positive values (price must RISE by this % from avg entry)
    short_dca_distance_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=2.0, max=40.0, default=5.0, step=2.0)
    )

    # ==========================================================================
    # DCA Position Sizing (fixed, not grid searched)
    # ==========================================================================
    dca_multiplier: float = 4.25  # Each DCA adds position_size * multiplier
    initial_position_usdt: float = 6.25  # Initial entry size in USDT
    max_position_usdt: float = 180.0  # Maximum total position size in USDT

    # ==========================================================================
    # Signal Entry Offset (triggers slightly before threshold)
    # ==========================================================================
    # LONG offset: triggers at (threshold + offset), e.g., -3.5% + 0.25% = -3.25%
    # SHORT offset: triggers at (threshold - offset), e.g., +6.0% - 1.0% = +5.0%
    long_signal_offset: float = 0.0  # % offset for LONG (0 = exact threshold)
    short_signal_offset: float = 0.0  # % offset for SHORT (0 = exact threshold)

    # ==========================================================================
    # Position Parameter bounds (shared between LONG and SHORT)
    # ==========================================================================
    # Note: TP now comes from session levels at runtime. These bounds are for
    # fallback TP percentage when session levels are not available.

    # Fallback target profit bounds (used when no session levels)
    target_profit_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=0.5, max=2.0, default=1.0, step=0.5)
    )

    stop_loss_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(min=-3.0, max=-1.0, default=-1.0, step=1.0)
    )

    # Max hold time bounds in milliseconds
    max_hold_time_bounds: ParameterBounds = field(
        default_factory=lambda: ParameterBounds(
            min=900000, max=3600000, default=1800000, step=900000
        )
    )

    # ==========================================================================
    # Time windows (shared between LONG and SHORT)
    # ==========================================================================
    # Time windows in milliseconds for rolling high/low computation
    # Used to detect price drops (LONG) or rises (SHORT) within a time period
    # 300ms, 500ms, 1000ms (1s), 2000ms (2s), 3000ms (3s)
    time_windows_ms: list[int] = field(default_factory=lambda: [300, 500, 1000, 2000, 3000])

    def compute_gpu_params(self, n_trades: int, trades_per_second: float = 30.0) -> tuple[int, int]:
        """
        Compute optimal GPU parameters based on dataset characteristics.

        Args:
            n_trades: Number of trades in the dataset
            trades_per_second: Estimated trade frequency (default ~30 for active pairs)

        Returns:
            Tuple of (max_scan, outer_stride)

        Logic:
        - max_scan: Based on max_hold_time_ms. If max hold = 1 hour at 30 trades/sec = 108,000 trades.
          But we cap it reasonably since most exits happen much sooner.
          Formula: min(max_hold_seconds * trades_per_second * 0.5, 2000)

        - outer_stride: Based on dataset size to ensure ~100k-500k entry checks per warp.
          Formula: max(1, n_trades // (WARP_SIZE * target_checks))
          Higher for larger datasets, lower for smaller ones.
        """
        # === max_scan calculation ===
        # Based on max hold time - most exits happen within first 50% of max hold
        max_hold_seconds = self.max_hold_time_bounds.max / 1000.0  # Convert ms to seconds
        estimated_trades_in_hold = int(max_hold_seconds * trades_per_second * 0.5)
        # Clamp between 500 and 2000
        max_scan = max(500, min(2000, estimated_trades_in_hold))

        # === outer_stride calculation ===
        # Target: each warp lane processes ~10k-50k potential entries
        # With WARP_SIZE=32, we want n_trades / (32 * outer_stride) ≈ 20k
        # So outer_stride ≈ n_trades / (32 * 20000) = n_trades / 640000
        WARP_SIZE = 32
        TARGET_ENTRIES_PER_LANE = 20000

        if n_trades < 100000:
            # Small dataset: check more frequently
            outer_stride = max(1, n_trades // (WARP_SIZE * 5000))
        elif n_trades < 1000000:
            # Medium dataset
            outer_stride = max(5, n_trades // (WARP_SIZE * TARGET_ENTRIES_PER_LANE))
        else:
            # Large dataset: can afford to skip more
            outer_stride = max(10, n_trades // (WARP_SIZE * TARGET_ENTRIES_PER_LANE))

        # Clamp between 1 and 50
        outer_stride = max(1, min(50, outer_stride))

        return max_scan, outer_stride

    def get_long_param_grid(self) -> list[list[float]]:
        """
        Get LONG parameter grid for GPU search with DCA.

        LONG: price drops (negative pm) + selling pressure (negative dt)

        Returns:
            List of [pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist] combinations
            - pm: price_move threshold (negative)
            - w_idx: time window index
            - dt: delta_threshold (negative)
            - tp: target_profit %
            - sl: stop_loss %
            - mh: max_hold_time_ms
            - dca_mult: DCA multiplier (4.25)
            - max_pos_mult: max_position / initial_position ratio
            - dca_dist: DCA distance % (negative for LONG)
        """
        pms = self.long_price_move_bounds.grid_values()
        dts = self.long_delta_threshold_bounds.grid_values()
        tps = self.target_profit_bounds.grid_values()
        sls = self.stop_loss_bounds.grid_values()
        mhs = self.max_hold_time_bounds.grid_values()
        dcas = self.long_dca_distance_bounds.grid_values()

        # Position sizing multipliers (fixed)
        dca_mult = self.dca_multiplier
        max_pos_mult = self.max_position_usdt / self.initial_position_usdt

        combos = []
        for pm in pms:
            for w_idx in range(len(self.time_windows_ms)):
                for dt in dts:
                    for tp in tps:
                        for sl in sls:
                            for mh in mhs:
                                for dca_dist in dcas:
                                    combos.append([pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist])

        return combos

    def get_short_param_grid(self) -> list[list[float]]:
        """
        Get SHORT parameter grid for GPU search with DCA.

        SHORT: price rises (positive pm) + buying pressure (positive dt)

        Returns:
            List of [pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist] combinations
            - pm: price_move threshold (positive)
            - w_idx: time window index
            - dt: delta_threshold (positive)
            - tp: target_profit %
            - sl: stop_loss %
            - mh: max_hold_time_ms
            - dca_mult: DCA multiplier (4.25)
            - max_pos_mult: max_position / initial_position ratio
            - dca_dist: DCA distance % (positive for SHORT)
        """
        pms = self.short_price_move_bounds.grid_values()
        dts = self.short_delta_threshold_bounds.grid_values()
        tps = self.target_profit_bounds.grid_values()
        sls = self.stop_loss_bounds.grid_values()
        mhs = self.max_hold_time_bounds.grid_values()
        dcas = self.short_dca_distance_bounds.grid_values()

        # Position sizing multipliers (fixed)
        dca_mult = self.dca_multiplier
        max_pos_mult = self.max_position_usdt / self.initial_position_usdt

        combos = []
        for pm in pms:
            for w_idx in range(len(self.time_windows_ms)):
                for dt in dts:
                    for tp in tps:
                        for sl in sls:
                            for mh in mhs:
                                for dca_dist in dcas:
                                    combos.append([pm, w_idx, dt, tp, sl, mh, dca_mult, max_pos_mult, dca_dist])

        return combos

    def window_idx_to_ms(self, w_idx: int) -> int:
        """Convert window index to milliseconds."""
        return self.time_windows_ms[w_idx]

    # ==========================================================================
    # Default Parameters (fallback when optimization finds no valid params)
    # ==========================================================================

    def get_default_long_params(self) -> VADParameters:
        """
        Get default LONG parameters from config bounds.

        Used as fallback when GPU optimization doesn't find valid params.
        """
        return VADParameters(
            price_move=self.long_price_move_bounds.default,
            time_window=self.time_windows_ms[0] if self.time_windows_ms else 300,
            delta_threshold=self.long_delta_threshold_bounds.default,
            target_profit=self.target_profit_bounds.default,
            stop_loss=self.stop_loss_bounds.default,
            max_hold_time=int(self.max_hold_time_bounds.default),
            dca_distance_pct=self.long_dca_distance_bounds.default,
        )

    def get_default_short_params(self) -> VADParameters:
        """
        Get default SHORT parameters from config bounds.

        Used as fallback when GPU optimization doesn't find valid params.
        """
        return VADParameters(
            price_move=self.short_price_move_bounds.default,
            time_window=self.time_windows_ms[0] if self.time_windows_ms else 300,
            delta_threshold=self.short_delta_threshold_bounds.default,
            target_profit=self.target_profit_bounds.default,
            stop_loss=self.stop_loss_bounds.default,
            max_hold_time=int(self.max_hold_time_bounds.default),
            dca_distance_pct=self.short_dca_distance_bounds.default,
        )

    @classmethod
    def from_dict(cls, data: dict) -> APEXConfig:
        """
        Create APEXConfig from a dictionary (e.g., from YAML config).

        Args:
            data: Dictionary with APEX configuration from engines.apex section

        Returns:
            APEXConfig instance
        """
        # Bootstrap section contains most parameters
        bootstrap_data = data.get("bootstrap", {})

        # Position parameters from bootstrap.position
        pos_data = bootstrap_data.get("position", {})
        tp_data = pos_data.get("target_profit_pct", {})
        sl_bounds_data = pos_data.get("stop_loss_pct", {})
        mh_bounds_data = pos_data.get("max_hold_time_ms", {})

        position = PositionParameters(
            target_profit=tp_data.get("default", 1.0),
            stop_loss=sl_bounds_data.get("default", -1.0),
            max_hold_time=mh_bounds_data.get("default", 1800000),
        )

        # Helper to parse bounds with defaults
        def parse_bounds(bounds_data: dict, defaults: ParameterBounds) -> ParameterBounds:
            if not bounds_data:
                return defaults
            return ParameterBounds(
                min=bounds_data.get("min", defaults.min),
                max=bounds_data.get("max", defaults.max),
                default=bounds_data.get("default", defaults.default),
                step=bounds_data.get("step", defaults.step),
            )

        # LONG bounds from bootstrap.long
        long_data = bootstrap_data.get("long", {})
        long_price_move_bounds = parse_bounds(
            long_data.get("price_move_pct", {}),
            ParameterBounds(min=-7.75, max=-1.75, default=-3.5, step=1.0),
        )
        long_delta_threshold_bounds = parse_bounds(
            long_data.get("volume_imbalance_threshold", {}),
            ParameterBounds(min=-90, max=-70, default=-50, step=10),
        )

        # SHORT bounds from bootstrap.short
        short_data = bootstrap_data.get("short", {})
        short_price_move_bounds = parse_bounds(
            short_data.get("price_move_pct", {}),
            ParameterBounds(min=5.0, max=20.0, default=7.0, step=5.0),
        )
        short_delta_threshold_bounds = parse_bounds(
            short_data.get("volume_imbalance_threshold", {}),
            ParameterBounds(min=70, max=90, default=50, step=10),
        )

        # DCA distance bounds
        long_dca_distance_bounds = parse_bounds(
            long_data.get("dca_distance_pct", {}),
            ParameterBounds(min=-9.75, max=-3.75, default=-3.5, step=2.0),
        )
        short_dca_distance_bounds = parse_bounds(
            short_data.get("dca_distance_pct", {}),
            ParameterBounds(min=10.0, max=40.0, default=7.0, step=10.0),
        )

        # Position bounds (TP/SL/MaxHold) - fallback values when no session levels
        target_profit_bounds = parse_bounds(
            tp_data,
            ParameterBounds(min=1.75, max=5.75, default=1.5, step=2.0),
        )

        stop_loss_bounds = parse_bounds(
            sl_bounds_data,
            ParameterBounds(min=-3.0, max=-1.0, default=-1.0, step=1.0),
        )
        max_hold_time_bounds = parse_bounds(
            mh_bounds_data,
            ParameterBounds(min=900000, max=14400000, default=900000, step=3600000),
        )

        # Time windows in milliseconds from bootstrap.rolling_kernel.lookback_windows_ms
        rolling_kernel_data = bootstrap_data.get("rolling_kernel", {})
        time_windows_ms = rolling_kernel_data.get(
            "lookback_windows_ms", [900000, 14400000]
        )

        # Volume imbalance price tolerance from bootstrap.volume_imbalance_kernel
        volume_imbalance_data = bootstrap_data.get("volume_imbalance_kernel", {})
        imbalance_price_tolerance_pct = volume_imbalance_data.get(
            "price_tolerance_pct", 2.5
        )

        # Bootstrap config
        bootstrap = BootstrapConfig.from_dict(bootstrap_data)

        return cls(
            min_warm_duration_hours=data.get("min_warm_duration_hours", 1.0),
            max_warm_age_hours=data.get("max_warm_age_hours", 24.0),
            max_cache_valid_days=data.get("max_cache_valid_days", 7.0),
            bootstrap=bootstrap,
            min_entries_for_validity=data.get("min_entries_for_validity", 5),
            imbalance_price_tolerance_pct=imbalance_price_tolerance_pct,
            position=position,
            long_price_move_bounds=long_price_move_bounds,
            long_delta_threshold_bounds=long_delta_threshold_bounds,
            short_price_move_bounds=short_price_move_bounds,
            short_delta_threshold_bounds=short_delta_threshold_bounds,
            long_dca_distance_bounds=long_dca_distance_bounds,
            short_dca_distance_bounds=short_dca_distance_bounds,
            long_signal_offset=long_data.get("signal_offset", 0.25),
            short_signal_offset=short_data.get("signal_offset", 0.25),
            target_profit_bounds=target_profit_bounds,
            stop_loss_bounds=stop_loss_bounds,
            max_hold_time_bounds=max_hold_time_bounds,
            time_windows_ms=time_windows_ms,
        )


# Default configuration
DEFAULT_CONFIG = APEXConfig()


# =============================================================================
# Config Loading with Symbol Overrides
# =============================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary.
        override: Override dictionary (values override base).

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_apex_config(config_path: str) -> APEXConfig:
    """
    Load APEXConfig from a YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        APEXConfig instance.

    Example:
        config = load_apex_config("config/config.yaml")
    """
    import yaml

    with open(config_path) as f:
        data = yaml.safe_load(f)

    apex_data = data.get("engines", {}).get("apex", {})
    return APEXConfig.from_dict(apex_data)


def load_apex_config_for_symbol(
    config_path: str,
    symbol: str,
    overrides_dir: str | None = None,
) -> APEXConfig:
    """
    Load APEXConfig with per-symbol overrides.

    Loads the base config from config_path, then looks for a symbol-specific
    override file in the overrides directory. Override values are deep-merged
    with the base config.

    Args:
        config_path: Path to the main YAML config file.
        symbol: Symbol name (e.g., "PIPPIN/USDT:USDT" or "PIPPINUSDT").
        overrides_dir: Path to overrides directory. If None, uses
                       config_path's parent / "overrides".

    Returns:
        APEXConfig instance with overrides applied.

    Example:
        # Loads config/config.yaml, then merges config/overrides/PIPPINUSDT.yaml
        config = load_apex_config_for_symbol("config/config.yaml", "PIPPIN/USDT:USDT")
    """
    import os

    import yaml

    # Load base config
    with open(config_path) as f:
        base_data = yaml.safe_load(f)

    apex_data = base_data.get("engines", {}).get("apex", {})

    # Determine overrides directory
    if overrides_dir is None:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        overrides_dir = os.path.join(config_dir, "overrides")

    # Normalize symbol name for filename (remove slashes and colons)
    # "PIPPIN/USDT:USDT" -> "PIPPINUSDT"
    symbol_normalized = symbol.replace("/", "").replace(":", "").replace("USDT", "")
    symbol_normalized = f"{symbol_normalized}USDT"

    # Look for override file
    override_path = os.path.join(overrides_dir, f"{symbol_normalized}.yaml")

    if os.path.exists(override_path):
        with open(override_path) as f:
            override_data = yaml.safe_load(f)

        if override_data:
            override_apex = override_data.get("engines", {}).get("apex", {})
            if override_apex:
                apex_data = _deep_merge(apex_data, override_apex)

    return APEXConfig.from_dict(apex_data)
