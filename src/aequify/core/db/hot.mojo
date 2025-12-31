"""
Hot Data Store - In-Memory Trade Storage with Index Pointers.

Trades stored ONCE sequentially, minute indices mark time boundaries.
Query ANY timeframe (1m, 5m, 15m, 1h, 4h) from the same data.

Architecture:
    Trades Array (sequential as received from stream):
    [T0, T1, T2, ..., T20, T21, ..., T50, ...]
     ^                ^              ^
     idx=0            idx=20         idx=50

    Minute Index (pointers into trades array):
    minute[60] = 0    # 01:00 → first trade of that minute is at trades[0]
    minute[75] = N    # 01:15 → first trade is at trades[N] (N = however many came before)
    minute[90] = M    # 01:30 → first trade is at trades[M]

Usage:
    var store = HotStore()

    # Add trades from stream
    store.add_trade(trade)

    # Query any timeframe (returns start_idx, end_idx)
    var indices = store.get_window(start_minute, end_minute)
    # or use convenience methods:
    var indices = store.get_15m(hour=1, quarter=1)  # 01:15-01:30
    var indices = store.get_1h(hour=1)              # 01:00-02:00
    var indices = store.get_4h(block=0)             # 00:00-04:00

    # Access trades directly
    for i in range(indices[0], indices[1]):
        var trade = store.trades[i]

    # Clean up old data (called externally)
    store.clear_before(minute=60)  # Remove all before 01:00
    store.clear_range(start_idx=0, end_idx=100)  # Remove specific range
"""

from collections import InlineArray

from .models import Trade, WindowIndices


# ===== Time Constants =====
comptime MS_PER_SECOND: Int64 = 1_000
comptime MS_PER_MINUTE: Int64 = 60_000
comptime MS_PER_HOUR: Int64 = 3_600_000
comptime MS_PER_DAY: Int64 = 86_400_000
comptime MINUTES_PER_DAY: Int = 1440


# ===== Time Utilities =====
fn get_minute_of_day(timestamp_ms: Int64) -> Int:
    """
    Get minute index within day (0-1439).

    Examples:
        00:00:00 -> 0
        01:15:30 -> 75
        23:59:59 -> 1439
    """
    return Int((timestamp_ms % MS_PER_DAY) // MS_PER_MINUTE)


fn get_minute_start(timestamp_ms: Int64) -> Int64:
    """Align timestamp to minute boundary."""
    return (timestamp_ms // MS_PER_MINUTE) * MS_PER_MINUTE


fn minutes_to_ms(minutes: Int) -> Int64:
    """Convert minutes to milliseconds."""
    return Int64(minutes) * MS_PER_MINUTE


# ===== Hot Store =====
struct HotStore:
    """
    In-memory trade store with O(1) time-window queries.

    Trades stored once, minute indices mark boundaries.
    Supports any timeframe query from same data.
    """
    var trades: List[Trade]
    var minute_index: InlineArray[Int, MINUTES_PER_DAY]  # 1440 entries
    var current_minute: Int
    var base_day_ms: Int64
    var total_added: Int
    var total_cleared: Int

    fn __init__(out self):
        """Initialize empty store."""
        self.trades = List[Trade]()
        self.minute_index = InlineArray[Int, MINUTES_PER_DAY](fill=-1)
        self.current_minute = -1
        self.base_day_ms = 0
        self.total_added = 0
        self.total_cleared = 0

    # ----- Add Trades -----

    fn add_trade(mut self, trade: Trade):
        """
        Add trade from stream. O(1) operation.

        Updates minute index when crossing minute boundary.
        """
        var trade_minute = get_minute_of_day(trade.timestamp_ms)

        # First trade initializes the store
        if self.current_minute == -1:
            self.base_day_ms = get_minute_start(trade.timestamp_ms) - minutes_to_ms(trade_minute)
            self.current_minute = trade_minute
            self.minute_index[trade_minute] = 0

        # New minute boundary?
        if trade_minute != self.current_minute:
            self.minute_index[trade_minute] = len(self.trades)
            self.current_minute = trade_minute

        self.trades.append(trade)
        self.total_added += 1

    fn add_trades(mut self, trades: List[Trade]):
        """Add multiple trades."""
        for i in range(len(trades)):
            self.add_trade(trades[i])

    # ----- Query Windows -----

    fn get_window(self, start_minute: Int, end_minute: Int) -> WindowIndices:
        """
        Get trade indices for a time window.

        Args:
            start_minute: Start minute of day (0-1439), inclusive.
            end_minute: End minute of day (0-1439), exclusive.

        Returns:
            WindowIndices with start_idx and end_idx into trades array.
        """
        if self.current_minute < 0 or len(self.trades) == 0:
            return WindowIndices()

        # Clamp to valid range
        var s = start_minute if start_minute >= 0 else 0
        var e = end_minute if end_minute <= MINUTES_PER_DAY else MINUTES_PER_DAY

        # Find start index (first trade in or after start_minute)
        var start_idx = -1
        for m in range(s, e):
            if self.minute_index[m] >= 0:
                start_idx = self.minute_index[m]
                break

        if start_idx < 0:
            return WindowIndices()

        # Find end index (first trade of next window, or end of trades)
        var end_idx = len(self.trades)
        for m in range(e, MINUTES_PER_DAY):
            if self.minute_index[m] >= 0:
                end_idx = self.minute_index[m]
                break

        return WindowIndices(start_idx, end_idx)

    fn get_1m(self, hour: Int, minute: Int) -> WindowIndices:
        """Get 1-minute window."""
        var start = hour * 60 + minute
        return self.get_window(start, start + 1)

    fn get_5m(self, hour: Int, block: Int) -> WindowIndices:
        """
        Get 5-minute window.

        Args:
            hour: Hour of day (0-23).
            block: Which 5-min block (0-11, e.g., 0=:00, 1=:05, 2=:10).
        """
        var start = hour * 60 + block * 5
        return self.get_window(start, start + 5)

    fn get_15m(self, hour: Int, quarter: Int) -> WindowIndices:
        """
        Get 15-minute window.

        Args:
            hour: Hour of day (0-23).
            quarter: Which quarter (0=:00, 1=:15, 2=:30, 3=:45).
        """
        var start = hour * 60 + quarter * 15
        return self.get_window(start, start + 15)

    fn get_1h(self, hour: Int) -> WindowIndices:
        """Get 1-hour window."""
        var start = hour * 60
        return self.get_window(start, start + 60)

    fn get_4h(self, block: Int) -> WindowIndices:
        """
        Get 4-hour window.

        Args:
            block: Which 4h block (0-5, e.g., 0=00:00-04:00, 1=04:00-08:00).
        """
        var start = block * 240
        return self.get_window(start, start + 240)

    # ----- Clear/Cleanup -----

    fn clear_before(mut self, minute: Int):
        """
        Remove all trades before the given minute of day.

        Called externally when trades are no longer needed.
        Updates indices to reflect new positions.

        Args:
            minute: Minute of day (0-1439). Trades before this are removed.
        """
        if minute <= 0 or len(self.trades) == 0:
            return

        # Find the index where 'minute' starts
        var cut_idx = -1
        for m in range(minute):
            if self.minute_index[m] >= 0:
                # Clear this minute's index
                self.minute_index[m] = -1

        # Find first valid minute at or after 'minute'
        for m in range(minute, MINUTES_PER_DAY):
            if self.minute_index[m] >= 0:
                cut_idx = self.minute_index[m]
                break

        if cut_idx <= 0:
            return

        # Remove trades before cut_idx
        var new_trades = List[Trade]()
        for i in range(cut_idx, len(self.trades)):
            new_trades.append(self.trades[i])

        self.total_cleared += cut_idx

        # Update minute indices (shift by cut_idx)
        for m in range(minute, MINUTES_PER_DAY):
            if self.minute_index[m] >= 0:
                self.minute_index[m] -= cut_idx

        self.trades = new_trades^

    fn clear_range(mut self, start_idx: Int, end_idx: Int):
        """
        Remove trades in a specific index range.

        Args:
            start_idx: Start index (inclusive).
            end_idx: End index (exclusive).
        """
        if start_idx >= end_idx or start_idx < 0:
            return

        var s = start_idx if start_idx >= 0 else 0
        var e = end_idx if end_idx <= len(self.trades) else len(self.trades)

        # Build new list excluding the range
        var new_trades = List[Trade]()
        for i in range(len(self.trades)):
            if i < s or i >= e:
                new_trades.append(self.trades[i])

        var removed = e - s
        self.total_cleared += removed

        # Update minute indices
        for m in range(MINUTES_PER_DAY):
            if self.minute_index[m] >= 0:
                if self.minute_index[m] >= e:
                    # After removed range: shift back
                    self.minute_index[m] -= removed
                elif self.minute_index[m] >= s:
                    # Within removed range: invalidate
                    self.minute_index[m] = -1

        self.trades = new_trades^

    fn clear_all(mut self):
        """Clear all trades and reset indices."""
        self.total_cleared += len(self.trades)
        self.trades = List[Trade]()
        self.minute_index = InlineArray[Int, MINUTES_PER_DAY](fill=-1)
        self.current_minute = -1

    # ----- Stats -----

    fn __len__(self) -> Int:
        """Current number of trades in store."""
        return len(self.trades)

    fn active_minutes(self) -> Int:
        """Count of minutes with trade data."""
        var count = 0
        for m in range(MINUTES_PER_DAY):
            if self.minute_index[m] >= 0:
                count += 1
        return count

    fn memory_bytes(self) -> Int:
        """Estimated memory usage in bytes."""
        var trade_bytes = len(self.trades) * 33
        var index_bytes = MINUTES_PER_DAY * 8
        return trade_bytes + index_bytes

    fn stats(self) -> String:
        return String(
            "HotStore(trades=", len(self.trades),
            ", active_minutes=", self.active_minutes(),
            ", total_added=", self.total_added,
            ", total_cleared=", self.total_cleared,
            ", memory=", self.memory_bytes() // 1024, "KB)"
        )
