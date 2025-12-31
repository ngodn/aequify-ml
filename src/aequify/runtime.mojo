"""
Aequify Runtime - Mojo wrapper for Python async runtime.

Provides event loop management, async execution, and GPU context from Mojo.
"""

from python import Python, PythonObject

# Re-export GPU from core
from .core.gpu import gpu_available, GPUContext, device_count


# =============================================================================
# Logging
# =============================================================================


fn _get_logger() raises -> PythonObject:
    """Get the logger for this module."""
    var logging_mod = Python.import_module("aequify.logging")
    return logging_mod.get_logger("aequify.runtime")


# =============================================================================
# Internal Helpers
# =============================================================================


fn _get_runtime_module() raises -> PythonObject:
    """Load the aequify.runtime Python module."""
    var sys = Python.import_module("sys")
    var os = Python.import_module("os")
    var pathlib = Python.import_module("pathlib")

    var cwd = pathlib.Path(os.getcwd())
    var src_path = cwd / "src"
    var src_path_str = String(src_path.__str__())

    if not Bool(sys.path.__contains__(src_path_str)):
        sys.path.insert(0, src_path_str)

    return Python.import_module("aequify.runtime")


# =============================================================================
# Version
# =============================================================================


fn get_version() raises -> String:
    """Get the aequify package version from pyproject.toml."""
    var metadata = Python.import_module("importlib.metadata")
    return String(metadata.version("aequify"))


# =============================================================================
# Setup
# =============================================================================


fn setup_uvloop() raises -> Bool:
    """
    Set uvloop as the default asyncio event loop policy.

    Returns:
        True if uvloop was set up, False if not available.
    """
    var logger = _get_logger()
    var runtime = _get_runtime_module()
    var result = Bool(runtime.setup_uvloop())
    if result:
        logger.debug("uvloop installed as default event loop policy")
    else:
        logger.debug("uvloop not available, using default asyncio")
    return result


fn is_uvloop_available() raises -> Bool:
    """Check if uvloop is available."""
    var runtime = _get_runtime_module()
    return Bool(runtime.is_uvloop_available())


# =============================================================================
# Main Event Loop
# =============================================================================


fn get_main_loop() raises -> PythonObject:
    """
    Get or create the main event loop.

    The main loop runs in a background thread.

    Returns:
        The asyncio event loop.
    """
    var runtime = _get_runtime_module()
    return runtime.get_main_loop()


fn shutdown_main_loop(timeout: Float64 = 5.0) raises:
    """
    Shutdown the main event loop.

    Args:
        timeout: Maximum time to wait for shutdown.
    """
    var logger = _get_logger()
    logger.debug("Shutting down main event loop (timeout=" + String(timeout) + "s)")
    var runtime = _get_runtime_module()
    runtime.shutdown_main_loop(PythonObject(timeout))
    logger.debug("Main event loop shutdown complete")


# =============================================================================
# Running Async Code
# =============================================================================


fn run_async(coro: PythonObject, timeout: Float64 = 30.0) raises -> PythonObject:
    """
    Run an async coroutine and wait for the result.

    Args:
        coro: Python coroutine to run.
        timeout: Maximum time to wait.

    Returns:
        The coroutine's result.
    """
    var runtime = _get_runtime_module()
    return runtime.run_async(coro, PythonObject(timeout))


fn run_async_fire_and_forget(coro: PythonObject) raises -> PythonObject:
    """
    Schedule a coroutine to run without waiting.

    Args:
        coro: Python coroutine to run.

    Returns:
        Future that can be used to check status.
    """
    var runtime = _get_runtime_module()
    return runtime.run_async_fire_and_forget(coro)


# =============================================================================
# Running in Thread Pool
# =============================================================================


fn run_in_thread(func: PythonObject) raises -> PythonObject:
    """
    Run a blocking function in the thread pool.

    Args:
        func: Python callable (no args).

    Returns:
        The function's result.
    """
    var runtime = _get_runtime_module()
    return runtime.run_in_thread(func)


fn run_in_thread_async(func: PythonObject) raises -> PythonObject:
    """
    Run a blocking function in the thread pool (non-blocking).

    Args:
        func: Python callable (no args).

    Returns:
        Future for the result.
    """
    var runtime = _get_runtime_module()
    return runtime.run_in_thread_async(func)


# =============================================================================
# Isolated Event Loops
# =============================================================================


struct IsolatedLoop:
    """
    An isolated event loop running in its own thread.

    Use this for modules that need their own independent async context.

    Example:
        var loop = IsolatedLoop("my-module")
        loop.start()
        var result = loop.run(some_coro)
        loop.stop()
    """

    var _py_loop: PythonObject
    var name: String

    fn __init__(out self, name: String = "isolated") raises:
        """
        Create an isolated event loop.

        Args:
            name: Name for the loop's thread.
        """
        self.name = name
        var runtime = _get_runtime_module()
        self._py_loop = runtime.IsolatedLoop(PythonObject(name))
        var logger = _get_logger()
        logger.debug("IsolatedLoop created: " + name)

    fn start(mut self) raises:
        """Start the isolated event loop in a background thread."""
        var logger = _get_logger()
        logger.debug("IsolatedLoop starting: " + self.name)
        self._py_loop.start()
        logger.info("IsolatedLoop started: " + self.name)

    fn stop(mut self, timeout: Float64 = 5.0) raises:
        """
        Stop the isolated event loop.

        Args:
            timeout: Maximum time to wait for shutdown.
        """
        var logger = _get_logger()
        logger.debug("IsolatedLoop stopping: " + self.name + " (timeout=" + String(timeout) + "s)")
        self._py_loop.stop(PythonObject(timeout))
        logger.info("IsolatedLoop stopped: " + self.name)

    fn is_running(self) raises -> Bool:
        """Check if the loop is running."""
        return Bool(self._py_loop.is_running)

    fn run(self, coro: PythonObject, timeout: Float64 = 30.0) raises -> PythonObject:
        """
        Run a coroutine in this isolated loop.

        Args:
            coro: Python coroutine to run.
            timeout: Maximum time to wait.

        Returns:
            The coroutine's result.
        """
        return self._py_loop.run(coro, PythonObject(timeout))

    fn schedule(self, coro: PythonObject) raises -> PythonObject:
        """
        Schedule a coroutine without waiting.

        Args:
            coro: Python coroutine to schedule.

        Returns:
            Future for checking status.
        """
        return self._py_loop.schedule(coro)


# =============================================================================
# Cleanup
# =============================================================================


fn shutdown_all(timeout: Float64 = 5.0) raises:
    """
    Shutdown all runtime resources.

    Args:
        timeout: Maximum time to wait for each resource.
    """
    var logger = _get_logger()
    logger.info("Shutting down all runtime resources (timeout=" + String(timeout) + "s)")
    var runtime = _get_runtime_module()
    runtime.shutdown_all(PythonObject(timeout))
    logger.info("All runtime resources shutdown complete")
