"""
APEX Utilities - Time alignment, session boundaries, and parameter generation.

Provides:
- Canonical time window alignment for standard intervals (1m, 5m, 15m, 1h, 4h)
- Forex trading sessions (Sydney, Tokyo, Mumbai, Frankfurt, London, NY-London, New York)
- Cartesian product generation for parameter grid search

Usage:
    from aequify.core.apex.utils import align_to_window, get_session_bounds, get_current_session

    # Align to 15m canonical window
    var bounds = align_to_window(timestamp_ms, 15)

    # Get session boundaries for a given timestamp
    var session = get_current_session(timestamp_ms)
    var bounds = get_session_bounds(session, day_start_ms)

    # Generate cartesian product of parameter lists
    var n_combos = compute_n_combinations(sizes)
    var params = alloc[Float64](n_combos * n_dims)
    generate_cartesian_product(param_lists, params)
"""

from memory import UnsafePointer


# =============================================================================
# Time Constants
# =============================================================================

comptime MS_PER_SECOND: Int64 = 1_000
comptime MS_PER_MINUTE: Int64 = 60_000
comptime MS_PER_HOUR: Int64 = 3_600_000
comptime MS_PER_DAY: Int64 = 86_400_000


# =============================================================================
# Session Definitions
# =============================================================================

comptime SESSION_COUNT: Int = 7

# Session IDs for indexing
comptime SESSION_SYDNEY: Int = 0
comptime SESSION_TOKYO: Int = 1
comptime SESSION_MUMBAI: Int = 2
comptime SESSION_FRANKFURT: Int = 3
comptime SESSION_LONDON: Int = 4
comptime SESSION_NY_LONDON: Int = 5
comptime SESSION_NEW_YORK: Int = 6

# Session hours (start_hour, end_hour) in UTC
# Note: Sydney crosses midnight (20:00-00:00)
comptime SYDNEY_START: Int = 20
comptime SYDNEY_END: Int = 24  # Midnight represented as 24 for calculation
comptime TOKYO_START: Int = 0
comptime TOKYO_END: Int = 3
comptime MUMBAI_START: Int = 3
comptime MUMBAI_END: Int = 7
comptime FRANKFURT_START: Int = 7
comptime FRANKFURT_END: Int = 8
comptime LONDON_START: Int = 8
comptime LONDON_END: Int = 13
comptime NY_LONDON_START: Int = 13
comptime NY_LONDON_END: Int = 17
comptime NEW_YORK_START: Int = 17
comptime NEW_YORK_END: Int = 20


# =============================================================================
# Window Bounds Result
# =============================================================================

@fieldwise_init
struct WindowBounds(Copyable, Movable, ImplicitlyCopyable, Stringable):
    """
    Time window boundaries in milliseconds.

    Fields:
        start_ms: Start timestamp (inclusive).
        end_ms: End timestamp (exclusive).
    """
    var start_ms: Int64
    var end_ms: Int64

    fn __init__(out self):
        self.start_ms = 0
        self.end_ms = 0

    fn duration_ms(self) -> Int64:
        """Duration in milliseconds."""
        return self.end_ms - self.start_ms

    fn duration_minutes(self) -> Int:
        """Duration in minutes."""
        return Int(self.duration_ms() // MS_PER_MINUTE)

    fn contains(self, timestamp_ms: Int64) -> Bool:
        """Check if timestamp falls within this window."""
        return timestamp_ms >= self.start_ms and timestamp_ms < self.end_ms

    fn __str__(self) -> String:
        return String("WindowBounds(", self.start_ms, ", ", self.end_ms, ")")


# =============================================================================
# Session Info
# =============================================================================

@fieldwise_init
struct SessionInfo(Copyable, Movable, ImplicitlyCopyable, Stringable):
    """
    Forex session information.

    Fields:
        id: Session ID (0-6).
        name: Session name.
        start_hour: Start hour UTC (0-23).
        end_hour: End hour UTC (0-24, 24 = midnight).
    """
    var id: Int
    var name: String
    var start_hour: Int
    var end_hour: Int

    fn duration_hours(self) -> Int:
        """Session duration in hours."""
        if self.end_hour > self.start_hour:
            return self.end_hour - self.start_hour
        # Crosses midnight
        return (24 - self.start_hour) + self.end_hour

    fn __str__(self) -> String:
        return String(
            self.name, " (", self.start_hour, ":00-",
            self.end_hour % 24, ":00 UTC)"
        )


# =============================================================================
# Session Registry
# =============================================================================

fn get_session_info(session_id: Int) -> SessionInfo:
    """
    Get session info by ID.

    Args:
        session_id: Session ID (0-6, use SESSION_* constants).

    Returns:
        SessionInfo for the requested session.
    """
    if session_id == SESSION_SYDNEY:
        return SessionInfo(id=SESSION_SYDNEY, name="Sydney", start_hour=SYDNEY_START, end_hour=SYDNEY_END)
    elif session_id == SESSION_TOKYO:
        return SessionInfo(id=SESSION_TOKYO, name="Tokyo", start_hour=TOKYO_START, end_hour=TOKYO_END)
    elif session_id == SESSION_MUMBAI:
        return SessionInfo(id=SESSION_MUMBAI, name="Mumbai", start_hour=MUMBAI_START, end_hour=MUMBAI_END)
    elif session_id == SESSION_FRANKFURT:
        return SessionInfo(id=SESSION_FRANKFURT, name="Frankfurt", start_hour=FRANKFURT_START, end_hour=FRANKFURT_END)
    elif session_id == SESSION_LONDON:
        return SessionInfo(id=SESSION_LONDON, name="London", start_hour=LONDON_START, end_hour=LONDON_END)
    elif session_id == SESSION_NY_LONDON:
        return SessionInfo(id=SESSION_NY_LONDON, name="NY-London", start_hour=NY_LONDON_START, end_hour=NY_LONDON_END)
    else:
        return SessionInfo(id=SESSION_NEW_YORK, name="New York", start_hour=NEW_YORK_START, end_hour=NEW_YORK_END)


fn get_session_by_name(name: String) -> SessionInfo:
    """
    Get session info by name.

    Args:
        name: Session name (Sydney, Tokyo, Mumbai, Frankfurt, London, NY-London, New York).

    Returns:
        SessionInfo for the requested session. Defaults to New York if not found.
    """
    if name == "Sydney":
        return get_session_info(SESSION_SYDNEY)
    elif name == "Tokyo":
        return get_session_info(SESSION_TOKYO)
    elif name == "Mumbai":
        return get_session_info(SESSION_MUMBAI)
    elif name == "Frankfurt":
        return get_session_info(SESSION_FRANKFURT)
    elif name == "London":
        return get_session_info(SESSION_LONDON)
    elif name == "NY-London":
        return get_session_info(SESSION_NY_LONDON)
    else:
        return get_session_info(SESSION_NEW_YORK)


# =============================================================================
# Time Alignment Functions
# =============================================================================

fn align_to_window(timestamp_ms: Int64, window_minutes: Int) -> WindowBounds:
    """
    Align timestamp to canonical window boundary.

    Returns the window that contains the given timestamp, aligned to clock.
    For example, 15m windows align to :00, :15, :30, :45.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.
        window_minutes: Window size in minutes (1, 5, 15, 60, 240, etc).

    Returns:
        WindowBounds with start_ms and end_ms.
    """
    var window_ms = Int64(window_minutes) * MS_PER_MINUTE
    var start = (timestamp_ms // window_ms) * window_ms
    return WindowBounds(start_ms=start, end_ms=start + window_ms)


fn align_to_1m(timestamp_ms: Int64) -> WindowBounds:
    """Align to 1-minute window."""
    return align_to_window(timestamp_ms, 1)


fn align_to_5m(timestamp_ms: Int64) -> WindowBounds:
    """Align to 5-minute window."""
    return align_to_window(timestamp_ms, 5)


fn align_to_15m(timestamp_ms: Int64) -> WindowBounds:
    """Align to 15-minute window."""
    return align_to_window(timestamp_ms, 15)


fn align_to_1h(timestamp_ms: Int64) -> WindowBounds:
    """Align to 1-hour window."""
    return align_to_window(timestamp_ms, 60)


fn align_to_4h(timestamp_ms: Int64) -> WindowBounds:
    """Align to 4-hour window."""
    return align_to_window(timestamp_ms, 240)


fn get_day_start(timestamp_ms: Int64) -> Int64:
    """
    Get the start of the UTC day containing the timestamp.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        Timestamp of 00:00:00 UTC on that day.
    """
    return (timestamp_ms // MS_PER_DAY) * MS_PER_DAY


fn get_hour_of_day(timestamp_ms: Int64) -> Int:
    """
    Get hour of day (0-23) for a timestamp.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        Hour of day in UTC (0-23).
    """
    return Int((timestamp_ms % MS_PER_DAY) // MS_PER_HOUR)


fn get_minute_of_hour(timestamp_ms: Int64) -> Int:
    """
    Get minute of hour (0-59) for a timestamp.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        Minute of hour (0-59).
    """
    return Int((timestamp_ms % MS_PER_HOUR) // MS_PER_MINUTE)


# =============================================================================
# Session Boundary Functions
# =============================================================================

fn get_session_bounds(session: SessionInfo, day_start_ms: Int64) -> WindowBounds:
    """
    Get session boundaries for a specific day.

    Args:
        session: SessionInfo to get bounds for.
        day_start_ms: Start of the day (00:00 UTC) in milliseconds.

    Returns:
        WindowBounds for the session on that day.

    Note:
        For Sydney (20:00-00:00), the end_ms will be at midnight (next day start).
    """
    var start_ms = day_start_ms + Int64(session.start_hour) * MS_PER_HOUR
    var end_hour = session.end_hour

    # Handle midnight crossing (end_hour = 24 means next day 00:00)
    if end_hour >= 24:
        var end_ms = day_start_ms + MS_PER_DAY  # Next day start
        return WindowBounds(start_ms=start_ms, end_ms=end_ms)

    var end_ms = day_start_ms + Int64(end_hour) * MS_PER_HOUR
    return WindowBounds(start_ms=start_ms, end_ms=end_ms)


fn get_session_bounds_for_timestamp(session: SessionInfo, timestamp_ms: Int64) -> WindowBounds:
    """
    Get session boundaries for the day containing the timestamp.

    Args:
        session: SessionInfo to get bounds for.
        timestamp_ms: Any timestamp within the desired day.

    Returns:
        WindowBounds for the session on that day.
    """
    var day_start = get_day_start(timestamp_ms)
    return get_session_bounds(session, day_start)


fn get_current_session(timestamp_ms: Int64) -> SessionInfo:
    """
    Get the active forex session for a given timestamp.

    Sessions are contiguous and non-overlapping:
        Sydney:     20:00-00:00 UTC
        Tokyo:      00:00-03:00 UTC
        Mumbai:     03:00-07:00 UTC
        Frankfurt:  07:00-08:00 UTC
        London:     08:00-13:00 UTC
        NY-London:  13:00-17:00 UTC
        New York:   17:00-20:00 UTC

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        SessionInfo for the active session.
    """
    var hour = get_hour_of_day(timestamp_ms)

    if hour >= SYDNEY_START:  # 20-23
        return get_session_info(SESSION_SYDNEY)
    elif hour < TOKYO_END:  # 0-2
        return get_session_info(SESSION_TOKYO)
    elif hour < MUMBAI_END:  # 3-6
        return get_session_info(SESSION_MUMBAI)
    elif hour < FRANKFURT_END:  # 7
        return get_session_info(SESSION_FRANKFURT)
    elif hour < LONDON_END:  # 8-12
        return get_session_info(SESSION_LONDON)
    elif hour < NY_LONDON_END:  # 13-16
        return get_session_info(SESSION_NY_LONDON)
    else:  # 17-19
        return get_session_info(SESSION_NEW_YORK)


fn get_current_session_bounds(timestamp_ms: Int64) -> WindowBounds:
    """
    Get the boundaries of the currently active session.

    Convenience function combining get_current_session and get_session_bounds.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        WindowBounds for the currently active session.
    """
    var session = get_current_session(timestamp_ms)
    return get_session_bounds_for_timestamp(session, timestamp_ms)


# =============================================================================
# Lookback Utilities
# =============================================================================

fn get_lookback_bounds(timestamp_ms: Int64, lookback_minutes: Int) -> WindowBounds:
    """
    Get bounds for a lookback window ending at the given timestamp.

    Unlike align_to_window which aligns to clock, this creates a window
    of exactly lookback_minutes ending at timestamp_ms.

    Args:
        timestamp_ms: End of the lookback window (exclusive).
        lookback_minutes: How far back to look.

    Returns:
        WindowBounds from (timestamp - lookback) to timestamp.
    """
    var lookback_ms = Int64(lookback_minutes) * MS_PER_MINUTE
    return WindowBounds(start_ms=timestamp_ms - lookback_ms, end_ms=timestamp_ms)


fn get_previous_session_ids(current_session_id: Int, count: Int) -> List[Int]:
    """
    Get the previous N session IDs (not including current).

    Useful for analyzing recent session behavior.

    Args:
        current_session_id: Current session ID.
        count: Number of previous sessions to return (max 7).

    Returns:
        List of session IDs, most recent first.
    """
    var result = List[Int]()
    var session_id = current_session_id
    var n = count if count <= SESSION_COUNT else SESSION_COUNT

    for _ in range(n):
        # Go to previous session (wrap around)
        session_id = (session_id - 1 + SESSION_COUNT) % SESSION_COUNT
        result.append(session_id)

    return result^


# =============================================================================
# Cartesian Product Generation
# =============================================================================


fn arange(start: Float64, stop: Float64, step: Float64) -> List[Float64]:
    """
    Generate values from start to stop (inclusive) with given step.

    Similar to numpy.arange but inclusive of the endpoint when it falls
    exactly on a step boundary.

    Args:
        start: Starting value (inclusive).
        stop: Ending value (inclusive if on step boundary).
        step: Step size between values.

    Returns:
        List of values from start to stop.

    Examples:
        arange(1.75, 3.75, 1.0) -> [1.75, 2.75, 3.75]
        arange(-5.0, -1.0, 1.0) -> [-5.0, -4.0, -3.0, -2.0, -1.0].
    """
    var result = List[Float64]()
    var val = start
    # Small epsilon for floating point comparison
    var epsilon: Float64 = 1e-9

    if step > 0:
        while val <= stop + epsilon:
            result.append(val)
            val += step
    elif step < 0:
        while val >= stop - epsilon:
            result.append(val)
            val += step
    else:
        # step == 0, just return start
        result.append(start)

    return result^


fn arange_count(start: Float64, stop: Float64, step: Float64) -> Int:
    """
    Compute how many values arange(start, stop, step) would generate.

    Useful for pre-allocating buffers without actually generating values.

    Args:
        start: Starting value.
        stop: Ending value.
        step: Step size.

    Returns:
        Number of values that would be generated.
    """
    if step == 0:
        return 1

    var range_size = stop - start
    if step > 0:
        if range_size < 0:
            return 0
        return Int(range_size / step) + 1
    else:
        if range_size > 0:
            return 0
        return Int(-range_size / -step) + 1


fn compute_n_combinations(sizes: List[Int]) -> Int:
    """
    Compute total number of combinations for a cartesian product.

    Args:
        sizes: List of sizes for each dimension.

    Returns:
        Total number of combinations (product of all sizes).
    """
    var n_combos = 1
    for i in range(len(sizes)):
        n_combos *= sizes[i]
    return n_combos


fn generate_cartesian_product[
    origin: MutOrigin,
](
    param_lists: List[List[Float64]],
    out_params: UnsafePointer[Float64, origin],
) -> Int:
    """
    Generate cartesian product of N parameter lists using index arithmetic.

    This function computes the cartesian product without nested loops by using
    modular arithmetic to determine which element from each list belongs at
    each position. This approach is:
    - Dynamic: works with any number of dimensions and sizes
    - Parallelizable: each combo_idx can be computed independently
    - Memory efficient: O(1) extra memory beyond input/output

    The formula for each dimension d at combo_idx:
        list_index = (combo_idx // stride[d]) % size[d]
    where stride[d] = size[d+1] * size[d+2] * ... * size[N-1]

    Args:
        param_lists: List of parameter lists (N lists of varying sizes).
        out_params: Output buffer for flattened combinations.
                    Must be pre-allocated with size: n_combos * n_dims.

    Returns:
        Number of combinations generated.

    Example:
        Given 2 lists with values [1.0, 2.0] and [10.0, 20.0, 30.0],
        generates 6 combinations in row-major order.
    """
    var n_dims = len(param_lists)
    if n_dims == 0:
        return 0

    # Compute sizes and total combinations
    var sizes = List[Int](capacity=n_dims)
    var n_combos = 1
    for i in range(n_dims):
        var size = len(param_lists[i])
        sizes.append(size)
        n_combos *= size

    if n_combos == 0:
        return 0

    # Precompute strides: stride[d] = product of sizes[d+1..N]
    # stride[d] tells us how many combos before the index in dimension d changes
    var strides = List[Int](capacity=n_dims)
    for d in range(n_dims):
        var stride = 1
        for i in range(d + 1, n_dims):
            stride *= sizes[i]
        strides.append(stride)

    # Generate all combinations using index arithmetic
    # Each combo_idx maps to a unique combination of indices
    for combo_idx in range(n_combos):
        for d in range(n_dims):
            var list_idx = (combo_idx // strides[d]) % sizes[d]
            out_params[combo_idx * n_dims + d] = param_lists[d][list_idx]

    return n_combos


fn generate_cartesian_product_parallel[
    origin: MutOrigin,
](
    param_lists: List[List[Float64]],
    out_params: UnsafePointer[Float64, origin],
) -> Int:
    """
    Generate cartesian product with parallel execution.

    Same as generate_cartesian_product but uses parallelize for the outer loop.
    Useful when generating very large parameter spaces.

    Args:
        param_lists: List of parameter lists (N lists of varying sizes).
        out_params: Output buffer for flattened combinations.

    Returns:
        Number of combinations generated.
    """
    from algorithm import parallelize

    var n_dims = len(param_lists)
    if n_dims == 0:
        return 0

    # Compute sizes and total combinations
    var sizes = List[Int](capacity=n_dims)
    var n_combos = 1
    for i in range(n_dims):
        var size = len(param_lists[i])
        sizes.append(size)
        n_combos *= size

    if n_combos == 0:
        return 0

    # Precompute strides
    var strides = List[Int](capacity=n_dims)
    for d in range(n_dims):
        var stride = 1
        for i in range(d + 1, n_dims):
            stride *= sizes[i]
        strides.append(stride)

    # Parallel generation - each combo_idx is independent
    @parameter
    fn generate_combo(combo_idx: Int):
        for d in range(n_dims):
            var list_idx = (combo_idx // strides[d]) % sizes[d]
            out_params[combo_idx * n_dims + d] = param_lists[d][list_idx]

    parallelize[generate_combo](n_combos)

    return n_combos
