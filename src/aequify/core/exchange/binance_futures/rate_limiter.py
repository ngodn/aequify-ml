"""
Binance Futures Rate Limiter.

Tracks API rate limit usage from Binance response headers.
Integrates with the Aequify runtime for async operations.

Binance Futures default limits:
- IP Weight: 2400 per minute
- Order Count: 300 per 10 seconds, 1200 per minute

Headers returned by Binance:
- X-MBX-USED-WEIGHT-1M: Current used weight for IP (1 minute window)
- X-MBX-ORDER-COUNT-10S: Order count in 10s window
- X-MBX-ORDER-COUNT-1M: Order count in 1m window
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Current rate limit state."""

    # Weight limits (IP-based)
    weight_used: int = 0
    weight_limit: int = 2400  # Default for non-VIP

    # Order limits (account-based)
    order_count_10s: int = 0
    order_count_1m: int = 0
    order_limit_10s: int = 300  # Default
    order_limit_1m: int = 1200  # Default

    # Timestamps
    last_update: float = 0.0
    weight_reset_at: float = 0.0  # When weight resets (approx)

    @property
    def weight_percentage(self) -> float:
        """Get weight usage as percentage."""
        if self.weight_limit <= 0:
            return 0.0
        return (self.weight_used / self.weight_limit) * 100

    @property
    def order_10s_percentage(self) -> float:
        """Get 10s order usage as percentage."""
        if self.order_limit_10s <= 0:
            return 0.0
        return (self.order_count_10s / self.order_limit_10s) * 100

    @property
    def order_1m_percentage(self) -> float:
        """Get 1m order usage as percentage."""
        if self.order_limit_1m <= 0:
            return 0.0
        return (self.order_count_1m / self.order_limit_1m) * 100

    @property
    def is_near_limit(self) -> bool:
        """Check if approaching rate limit (>80%)."""
        return self.weight_percentage > 80

    @property
    def is_critical(self) -> bool:
        """Check if at critical rate limit (>95%)."""
        return self.weight_percentage > 95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weight_used": self.weight_used,
            "weight_limit": self.weight_limit,
            "weight_percentage": round(self.weight_percentage, 1),
            "order_count_10s": self.order_count_10s,
            "order_count_1m": self.order_count_1m,
            "order_limit_10s": self.order_limit_10s,
            "order_limit_1m": self.order_limit_1m,
            "is_near_limit": self.is_near_limit,
            "is_critical": self.is_critical,
            "last_update": self.last_update,
        }


# Callback types
SyncCallback = Callable[[RateLimitState], None]
AsyncCallback = Callable[[RateLimitState], Awaitable[None]]
RateLimitCallback = SyncCallback | AsyncCallback


class BinanceFuturesRateLimiter:
    """
    Rate limiter for Binance Futures API.

    Tracks rate limit usage from response headers and provides
    callbacks for state updates (e.g., to update TUI).

    Integrates with aequify.runtime for async callback handling.

    Usage:
        limiter = BinanceFuturesRateLimiter()

        # Register callback for UI updates
        limiter.on_update(lambda state: print(f"Weight: {state.weight_percentage}%"))

        # After each API call, update from response headers
        limiter.update_from_headers(response.headers)

        # Check before making requests
        delay = limiter.get_recommended_delay()
        if delay > 0:
            await asyncio.sleep(delay)

        # Or use the async context manager
        async with limiter.throttle():
            response = await client.get(...)
    """

    def __init__(
        self,
        weight_limit: int = 2400,
        order_limit_1m: int = 1200,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            weight_limit: IP weight limit per minute.
            order_limit_1m: Order limit per minute.
        """
        self._state = RateLimitState(
            weight_limit=weight_limit,
            order_limit_1m=order_limit_1m,
            order_limit_10s=order_limit_1m // 4,
        )
        self._callbacks: list[RateLimitCallback] = []
        self._lock = asyncio.Lock()
        self._headers_logged = False

    @property
    def state(self) -> RateLimitState:
        """Get current rate limit state (read-only snapshot)."""
        return RateLimitState(
            weight_used=self._state.weight_used,
            weight_limit=self._state.weight_limit,
            order_count_10s=self._state.order_count_10s,
            order_count_1m=self._state.order_count_1m,
            order_limit_10s=self._state.order_limit_10s,
            order_limit_1m=self._state.order_limit_1m,
            last_update=self._state.last_update,
            weight_reset_at=self._state.weight_reset_at,
        )

    def on_update(self, callback: RateLimitCallback) -> None:
        """
        Register callback for rate limit updates.

        Callbacks can be sync or async functions.

        Args:
            callback: Function called when rate limits are updated.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: RateLimitCallback) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _notify_callbacks(self) -> None:
        """Notify all registered callbacks (handles both sync and async)."""
        state = self.state
        for callback in self._callbacks:
            try:
                result = callback(state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Rate limit callback error: {e}")

    def _notify_callbacks_sync(self) -> None:
        """
        Notify callbacks synchronously.

        For async callbacks, schedules them as tasks if there's a running loop.
        """
        state = self.state
        for callback in self._callbacks:
            try:
                result = callback(state)
                if asyncio.iscoroutine(result):
                    # Try to schedule in running loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop, skip async callback
                        pass
            except Exception as e:
                logger.error(f"Rate limit callback error: {e}")

    def update_from_headers(self, headers: dict[str, str]) -> bool:
        """
        Update rate limit state from Binance response headers.

        Can be called from sync or async context.

        Args:
            headers: Response headers from Binance API.

        Returns:
            True if state was updated, False otherwise.
        """
        now = time.time()
        updated = False

        # Parse weight used (IP limit)
        weight_header = self._get_header(
            headers,
            "X-MBX-USED-WEIGHT-1M",
            "x-mbx-used-weight-1m",
            "X-MBX-USED-WEIGHT",
            "x-mbx-used-weight",
        )
        if weight_header is not None:
            try:
                new_weight = int(weight_header)
                if new_weight < 0:
                    logger.debug(f"Ignoring negative weight value: {new_weight}")
                elif new_weight != self._state.weight_used:
                    logger.debug(
                        f"Rate limit weight: {self._state.weight_used} -> {new_weight}"
                    )
                    self._state.weight_used = new_weight
                    updated = True

                    if new_weight > 0:
                        self._state.weight_reset_at = now + 60
            except ValueError:
                logger.warning(f"Failed to parse weight header: {weight_header}")
        elif not self._headers_logged:
            self._headers_logged = True
            logger.debug(f"No weight header found. Headers: {list(headers.keys())}")

        # Parse order count 10s
        order_10s = self._get_header(
            headers, "X-MBX-ORDER-COUNT-10S", "x-mbx-order-count-10s"
        )
        if order_10s is not None:
            try:
                new_count = int(order_10s)
                if new_count != self._state.order_count_10s:
                    self._state.order_count_10s = new_count
                    updated = True
            except ValueError:
                pass

        # Parse order count 1m
        order_1m = self._get_header(
            headers, "X-MBX-ORDER-COUNT-1M", "x-mbx-order-count-1m"
        )
        if order_1m is not None:
            try:
                new_count = int(order_1m)
                if new_count != self._state.order_count_1m:
                    self._state.order_count_1m = new_count
                    updated = True
            except ValueError:
                pass

        if updated:
            self._state.last_update = now
            self._notify_callbacks_sync()

        return updated

    async def update_from_headers_async(self, headers: dict[str, str]) -> bool:
        """
        Update rate limit state from headers (async version).

        Use this when you need proper async callback handling.

        Args:
            headers: Response headers from Binance API.

        Returns:
            True if state was updated.
        """
        async with self._lock:
            updated = self.update_from_headers(headers)
            if updated:
                await self._notify_callbacks()
            return updated

    @staticmethod
    def _get_header(headers: dict[str, str], *keys: str) -> str | None:
        """Get first matching header value."""
        for key in keys:
            if key in headers:
                return headers[key]
        return None

    def reset(self) -> None:
        """Reset rate limit state."""
        self._state = RateLimitState(
            weight_limit=self._state.weight_limit,
            order_limit_1m=self._state.order_limit_1m,
            order_limit_10s=self._state.order_limit_10s,
        )
        self._notify_callbacks_sync()

    def set_limits(
        self,
        weight_limit: int | None = None,
        order_limit_1m: int | None = None,
    ) -> None:
        """
        Update rate limits (e.g., for VIP users with higher limits).

        Args:
            weight_limit: IP weight limit per minute.
            order_limit_1m: Order limit per minute.
        """
        if weight_limit is not None:
            self._state.weight_limit = weight_limit
        if order_limit_1m is not None:
            self._state.order_limit_1m = order_limit_1m
            self._state.order_limit_10s = order_limit_1m // 4
        self._notify_callbacks_sync()

    def get_recommended_delay(self) -> float:
        """
        Get recommended delay before next request.

        Returns:
            Delay in seconds (0 if no delay needed).
        """
        pct = self._state.weight_percentage

        if pct >= 95:
            # Critical - wait longer
            return 5.0
        elif pct >= 80:
            # Near limit - slow down
            return 1.0
        elif pct >= 60:
            # Moderate usage
            return 0.2
        return 0.0

    async def wait_if_needed(self) -> float:
        """
        Wait if rate limit requires throttling.

        Returns:
            Time waited in seconds.
        """
        delay = self.get_recommended_delay()
        if delay > 0:
            logger.debug(f"Rate limit throttle: waiting {delay}s")
            await asyncio.sleep(delay)
        return delay

    async def throttle(self) -> "RateLimitThrottleContext":
        """
        Async context manager for throttled requests.

        Usage:
            async with limiter.throttle() as ctx:
                response = await client.get(...)
                ctx.update_headers(response.headers)
        """
        await self.wait_if_needed()
        return RateLimitThrottleContext(self)


class RateLimitThrottleContext:
    """Context for throttled requests."""

    def __init__(self, limiter: BinanceFuturesRateLimiter) -> None:
        self._limiter = limiter

    async def __aenter__(self) -> "RateLimitThrottleContext":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    def update_headers(self, headers: dict[str, str]) -> None:
        """Update rate limiter from response headers."""
        self._limiter.update_from_headers(headers)
