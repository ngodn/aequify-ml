"""
Time Utilities - Session alignment and window boundary calculations.

Provides canonical time window alignment for:
- Standard intervals (1m, 5m, 15m, 1h, 4h)
- Forex trading sessions (Sydney, Tokyo, Mumbai, Frankfurt, London, NY-London, New York)

Usage:
    from aequify.utils import align_to_window, get_session_bounds, get_current_session

    # Align to 15m canonical window
    bounds = align_to_window(timestamp_ms, 15)

    # Get session boundaries for a given timestamp
    session = get_current_session(timestamp_ms)
    bounds = get_session_bounds(session, day_start_ms)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple


# =============================================================================
# Time Constants
# =============================================================================

MS_PER_SECOND: int = 1_000
MS_PER_MINUTE: int = 60_000
MS_PER_HOUR: int = 3_600_000
MS_PER_DAY: int = 86_400_000


# =============================================================================
# Session Definitions
# =============================================================================


class Session(IntEnum):
    """Forex session identifiers."""

    SYDNEY = 0
    TOKYO = 1
    MUMBAI = 2
    FRANKFURT = 3
    LONDON = 4
    NY_LONDON = 5
    NEW_YORK = 6


# Session hours (start_hour, end_hour) in UTC
# Note: Sydney crosses midnight (20:00-00:00), end_hour=24 represents midnight
SESSION_HOURS: dict[Session, tuple[int, int]] = {
    Session.SYDNEY: (20, 24),  # 20:00-00:00 UTC (crosses midnight)
    Session.TOKYO: (0, 3),  # 00:00-03:00 UTC
    Session.MUMBAI: (3, 7),  # 03:00-07:00 UTC
    Session.FRANKFURT: (7, 8),  # 07:00-08:00 UTC
    Session.LONDON: (8, 13),  # 08:00-13:00 UTC
    Session.NY_LONDON: (13, 17),  # 13:00-17:00 UTC
    Session.NEW_YORK: (17, 20),  # 17:00-20:00 UTC
}

SESSION_NAMES: dict[Session, str] = {
    Session.SYDNEY: "Sydney",
    Session.TOKYO: "Tokyo",
    Session.MUMBAI: "Mumbai",
    Session.FRANKFURT: "Frankfurt",
    Session.LONDON: "London",
    Session.NY_LONDON: "NY-London",
    Session.NEW_YORK: "New York",
}

SESSION_COUNT: int = 7


# =============================================================================
# Window Bounds Result
# =============================================================================


class WindowBounds(NamedTuple):
    """
    Time window boundaries in milliseconds.

    Attributes:
        start_ms: Start timestamp (inclusive).
        end_ms: End timestamp (exclusive).
    """

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def duration_minutes(self) -> int:
        """Duration in minutes."""
        return self.duration_ms // MS_PER_MINUTE

    def contains(self, timestamp_ms: int) -> bool:
        """Check if timestamp falls within this window."""
        return self.start_ms <= timestamp_ms < self.end_ms


# =============================================================================
# Session Info
# =============================================================================


@dataclass(slots=True, frozen=True)
class SessionInfo:
    """
    Forex session information.

    Attributes:
        id: Session ID (0-6).
        name: Session name.
        start_hour: Start hour UTC (0-23).
        end_hour: End hour UTC (0-24, 24 = midnight).
    """

    id: Session
    name: str
    start_hour: int
    end_hour: int

    @property
    def duration_hours(self) -> int:
        """Session duration in hours."""
        if self.end_hour > self.start_hour:
            return self.end_hour - self.start_hour
        # Crosses midnight
        return (24 - self.start_hour) + self.end_hour


# =============================================================================
# Session Registry
# =============================================================================


def get_session_info(session_id: Session | int) -> SessionInfo:
    """
    Get session info by ID.

    Args:
        session_id: Session ID (0-6, use Session enum).

    Returns:
        SessionInfo for the requested session.
    """
    session = Session(session_id)
    start_hour, end_hour = SESSION_HOURS[session]
    return SessionInfo(
        id=session,
        name=SESSION_NAMES[session],
        start_hour=start_hour,
        end_hour=end_hour,
    )


def get_session_by_name(name: str) -> SessionInfo:
    """
    Get session info by name.

    Args:
        name: Session name (Sydney, Tokyo, Mumbai, Frankfurt, London, NY-London, New York).

    Returns:
        SessionInfo for the requested session. Defaults to New York if not found.
    """
    name_to_session = {v: k for k, v in SESSION_NAMES.items()}
    session = name_to_session.get(name, Session.NEW_YORK)
    return get_session_info(session)


# =============================================================================
# Time Alignment Functions
# =============================================================================


def align_to_window(timestamp_ms: int, window_minutes: int) -> WindowBounds:
    """
    Align timestamp to canonical window boundary.

    Returns the window that contains the given timestamp, aligned to clock.
    For example, 15m windows align to :00, :15, :30, :45.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.
        window_minutes: Window size in minutes (1, 5, 15, 60, 240, etc).

    Returns:
        WindowBounds with start_ms and end_ms.

    Examples:
        >>> align_to_window(1704067230000, 15)  # 2024-01-01 00:00:30
        WindowBounds(start_ms=1704067200000, end_ms=1704068100000)  # 00:00-00:15
    """
    window_ms = window_minutes * MS_PER_MINUTE
    start = (timestamp_ms // window_ms) * window_ms
    return WindowBounds(start_ms=start, end_ms=start + window_ms)


def align_to_1m(timestamp_ms: int) -> WindowBounds:
    """Align to 1-minute window."""
    return align_to_window(timestamp_ms, 1)


def align_to_5m(timestamp_ms: int) -> WindowBounds:
    """Align to 5-minute window."""
    return align_to_window(timestamp_ms, 5)


def align_to_15m(timestamp_ms: int) -> WindowBounds:
    """Align to 15-minute window."""
    return align_to_window(timestamp_ms, 15)


def align_to_1h(timestamp_ms: int) -> WindowBounds:
    """Align to 1-hour window."""
    return align_to_window(timestamp_ms, 60)


def align_to_4h(timestamp_ms: int) -> WindowBounds:
    """Align to 4-hour window."""
    return align_to_window(timestamp_ms, 240)


def get_day_start(timestamp_ms: int) -> int:
    """
    Get the start of the UTC day containing the timestamp.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        Timestamp of 00:00:00 UTC on that day.
    """
    return (timestamp_ms // MS_PER_DAY) * MS_PER_DAY


def get_hour_of_day(timestamp_ms: int) -> int:
    """
    Get hour of day (0-23) for a timestamp.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        Hour of day in UTC (0-23).
    """
    return (timestamp_ms % MS_PER_DAY) // MS_PER_HOUR


def get_minute_of_hour(timestamp_ms: int) -> int:
    """
    Get minute of hour (0-59) for a timestamp.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        Minute of hour (0-59).
    """
    return (timestamp_ms % MS_PER_HOUR) // MS_PER_MINUTE


# =============================================================================
# Session Boundary Functions
# =============================================================================


def get_session_bounds(session: SessionInfo, day_start_ms: int) -> WindowBounds:
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
    start_ms = day_start_ms + session.start_hour * MS_PER_HOUR

    # Handle midnight crossing (end_hour = 24 means next day 00:00)
    if session.end_hour >= 24:
        end_ms = day_start_ms + MS_PER_DAY  # Next day start
    else:
        end_ms = day_start_ms + session.end_hour * MS_PER_HOUR

    return WindowBounds(start_ms=start_ms, end_ms=end_ms)


def get_session_bounds_for_timestamp(
    session: SessionInfo, timestamp_ms: int
) -> WindowBounds:
    """
    Get session boundaries for the day containing the timestamp.

    Args:
        session: SessionInfo to get bounds for.
        timestamp_ms: Any timestamp within the desired day.

    Returns:
        WindowBounds for the session on that day.
    """
    day_start = get_day_start(timestamp_ms)
    return get_session_bounds(session, day_start)


def get_current_session(timestamp_ms: int) -> SessionInfo:
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
    hour = get_hour_of_day(timestamp_ms)

    if hour >= 20:  # Sydney: 20-23
        return get_session_info(Session.SYDNEY)
    elif hour < 3:  # Tokyo: 0-2
        return get_session_info(Session.TOKYO)
    elif hour < 7:  # Mumbai: 3-6
        return get_session_info(Session.MUMBAI)
    elif hour < 8:  # Frankfurt: 7
        return get_session_info(Session.FRANKFURT)
    elif hour < 13:  # London: 8-12
        return get_session_info(Session.LONDON)
    elif hour < 17:  # NY-London: 13-16
        return get_session_info(Session.NY_LONDON)
    else:  # New York: 17-19
        return get_session_info(Session.NEW_YORK)


def get_current_session_bounds(timestamp_ms: int) -> WindowBounds:
    """
    Get the boundaries of the currently active session.

    Convenience function combining get_current_session and get_session_bounds.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        WindowBounds for the currently active session.
    """
    session = get_current_session(timestamp_ms)
    return get_session_bounds_for_timestamp(session, timestamp_ms)


# =============================================================================
# Lookback Utilities
# =============================================================================


def get_lookback_bounds(timestamp_ms: int, lookback_minutes: int) -> WindowBounds:
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
    lookback_ms = lookback_minutes * MS_PER_MINUTE
    return WindowBounds(start_ms=timestamp_ms - lookback_ms, end_ms=timestamp_ms)


def get_previous_sessions(
    current_session_id: Session | int, count: int
) -> list[SessionInfo]:
    """
    Get the previous N sessions (not including current).

    Useful for analyzing recent session behavior.

    Args:
        current_session_id: Current session ID.
        count: Number of previous sessions to return (max 7).

    Returns:
        List of SessionInfo, most recent first.
    """
    result: list[SessionInfo] = []
    session_id = int(current_session_id)
    n = min(count, SESSION_COUNT)

    for _ in range(n):
        # Go to previous session (wrap around)
        session_id = (session_id - 1 + SESSION_COUNT) % SESSION_COUNT
        result.append(get_session_info(Session(session_id)))

    return result
