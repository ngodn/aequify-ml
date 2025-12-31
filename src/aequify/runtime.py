"""
Aequify Runtime - Async infrastructure with uvloop.

Provides event loop management for running async code from Mojo.
Supports isolated loops for independent modules.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

try:
    import uvloop

    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False

if TYPE_CHECKING:
    from collections.abc import Generator

T = TypeVar("T")


# =============================================================================
# Global State
# =============================================================================

_main_loop: asyncio.AbstractEventLoop | None = None
_main_loop_thread: threading.Thread | None = None
_thread_executor: concurrent.futures.ThreadPoolExecutor | None = None
_shutdown_event: threading.Event = threading.Event()


# =============================================================================
# Setup Functions
# =============================================================================


def setup_uvloop() -> bool:
    """
    Set uvloop as the default asyncio event loop policy.

    Returns:
        True if uvloop was set up, False if not available.
    """
    if HAS_UVLOOP:
        uvloop.install()
        return True
    return False


def is_uvloop_available() -> bool:
    """Check if uvloop is available."""
    return HAS_UVLOOP


# =============================================================================
# Main Event Loop
# =============================================================================


def get_main_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create the main event loop.

    The main loop runs in a background thread and is shared across
    all async operations from the main Mojo context.

    Returns:
        The main event loop.
    """
    global _main_loop, _main_loop_thread

    if _main_loop is not None and _main_loop.is_running():
        return _main_loop

    # Create new loop
    if HAS_UVLOOP:
        _main_loop = uvloop.new_event_loop()
    else:
        _main_loop = asyncio.new_event_loop()

    # Start loop in background thread
    _shutdown_event.clear()
    _main_loop_thread = threading.Thread(
        target=_run_loop_forever,
        args=(_main_loop,),
        daemon=True,
        name="aequify-main-loop",
    )
    _main_loop_thread.start()

    return _main_loop


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    """Run the event loop forever (in background thread)."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()


def shutdown_main_loop(timeout: float = 5.0) -> None:
    """
    Shutdown the main event loop.

    Args:
        timeout: Maximum time to wait for shutdown.
    """
    global _main_loop, _main_loop_thread

    if _main_loop is None:
        return

    _shutdown_event.set()

    # Schedule loop stop
    if _main_loop.is_running():
        _main_loop.call_soon_threadsafe(_main_loop.stop)

    # Wait for thread
    if _main_loop_thread is not None and _main_loop_thread.is_alive():
        _main_loop_thread.join(timeout=timeout)

    _main_loop = None
    _main_loop_thread = None


# =============================================================================
# Running Async Code
# =============================================================================


def run_async(coro: Awaitable[T], timeout: float | None = None) -> T:
    """
    Run an async coroutine and wait for the result.

    Runs in the main event loop's thread.

    Args:
        coro: The coroutine to run.
        timeout: Maximum time to wait (None for no timeout).

    Returns:
        The coroutine's result.

    Raises:
        asyncio.TimeoutError: If timeout is exceeded.
        Exception: Any exception raised by the coroutine.
    """
    loop = get_main_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def run_async_fire_and_forget(coro: Awaitable[Any]) -> asyncio.Future[Any]:
    """
    Schedule a coroutine to run without waiting for result.

    Args:
        coro: The coroutine to run.

    Returns:
        Future that can be used to check status later.
    """
    loop = get_main_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)


# =============================================================================
# Running in Thread Pool
# =============================================================================


def get_thread_executor(max_workers: int = 4) -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _thread_executor

    if _thread_executor is None:
        _thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="aequify-worker",
        )

    return _thread_executor


def run_in_thread(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Run a blocking function in the thread pool.

    Args:
        func: The function to run.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        The function's result.
    """
    executor = get_thread_executor()
    future = executor.submit(func, *args, **kwargs)
    return future.result()


def run_in_thread_async(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> concurrent.futures.Future[T]:
    """
    Run a blocking function in the thread pool (non-blocking).

    Args:
        func: The function to run.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Future for the result.
    """
    executor = get_thread_executor()
    return executor.submit(func, *args, **kwargs)


# =============================================================================
# Isolated Event Loops
# =============================================================================


class IsolatedLoop:
    """
    An isolated event loop running in its own thread.

    Use this for modules that need their own independent async context.

    Example:
        loop = IsolatedLoop("my-module")
        loop.start()

        result = loop.run(some_async_function())

        loop.stop()
    """

    def __init__(self, name: str = "isolated") -> None:
        """
        Create an isolated event loop.

        Args:
            name: Name for the loop's thread.
        """
        self.name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._stopped = threading.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the underlying event loop."""
        return self._loop

    @property
    def is_running(self) -> bool:
        """Check if the loop is running."""
        return self._loop is not None and self._loop.is_running()

    def start(self) -> None:
        """Start the isolated event loop in a background thread."""
        if self._loop is not None:
            raise RuntimeError("Loop already started")

        # Create loop
        if HAS_UVLOOP:
            self._loop = uvloop.new_event_loop()
        else:
            self._loop = asyncio.new_event_loop()

        # Start thread
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"aeq-{self.name}",
        )
        self._thread.start()

        # Wait for loop to be ready
        self._started.wait(timeout=5.0)

    def _run(self) -> None:
        """Run the event loop (in background thread)."""
        if self._loop is None:
            return

        asyncio.set_event_loop(self._loop)
        self._started.set()

        try:
            self._loop.run_forever()
        finally:
            # Clean up pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()

            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
            self._loop.close()
            self._stopped.set()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the isolated event loop.

        Args:
            timeout: Maximum time to wait for shutdown.
        """
        if self._loop is None:
            return

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._loop = None
        self._thread = None

    def run(self, coro: Awaitable[T], timeout: float | None = None) -> T:
        """
        Run a coroutine in this isolated loop.

        Args:
            coro: The coroutine to run.
            timeout: Maximum time to wait.

        Returns:
            The coroutine's result.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("Loop not running")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def schedule(self, coro: Awaitable[Any]) -> asyncio.Future[Any]:
        """
        Schedule a coroutine without waiting (fire-and-forget).

        Args:
            coro: The coroutine to schedule.

        Returns:
            Future for checking status.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("Loop not running")

        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    @contextmanager
    def running(self) -> Generator[IsolatedLoop, None, None]:
        """
        Context manager for loop lifecycle.

        Example:
            with IsolatedLoop("test").running() as loop:
                loop.run(async_func())
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()


# =============================================================================
# Cleanup
# =============================================================================


def shutdown_all(timeout: float = 5.0) -> None:
    """
    Shutdown all runtime resources.

    Args:
        timeout: Maximum time to wait for each resource.
    """
    global _thread_executor

    # Shutdown main loop
    shutdown_main_loop(timeout)

    # Shutdown thread executor
    if _thread_executor is not None:
        _thread_executor.shutdown(wait=True)
        _thread_executor = None
