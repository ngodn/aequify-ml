"""Aequify - ML framework with Python interop."""

# Socket wrappers
from .core import SocketClient, QueuedClient, SocketServer

# Runtime (async/threading/gpu)
from .runtime import (
    get_version,
    setup_uvloop,
    is_uvloop_available,
    get_main_loop,
    shutdown_main_loop,
    run_async,
    run_async_fire_and_forget,
    run_in_thread,
    run_in_thread_async,
    IsolatedLoop,
    shutdown_all,
    gpu_available,
    GPUContext,
    device_count,
)

# Demo
from .core import SocketTestResult, run_socket_test
