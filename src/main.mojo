"""
Aequify - Mojo Entrypoint

Main entrypoint for the Aequify ML application.
Starts the engine in a background thread and runs TUI on main thread.
"""

from python import Python, PythonObject
from sys import has_apple_gpu_accelerator, has_amd_gpu_accelerator
from aequify import (
    get_version,
    setup_uvloop,
    shutdown_all,
    gpu_available,
    gpu_vendor,
    GPUContext,
)


# =============================================================================
# Logging
# =============================================================================


fn _get_logger() raises -> PythonObject:
    """Get the logger for this module."""
    var logging_mod = Python.import_module("aequify.logging")
    return logging_mod.get_logger("aequify.main")


# =============================================================================
# Setup
# =============================================================================


fn _setup_python_path() raises:
    """Add src to Python path for imports."""
    var sys = Python.import_module("sys")
    var os = Python.import_module("os")
    var pathlib = Python.import_module("pathlib")

    var cwd = pathlib.Path(os.getcwd())
    var src_path = cwd / "src"
    var src_path_str = String(src_path.__str__())

    if not Bool(sys.path.__contains__(src_path_str)):
        sys.path.insert(0, src_path_str)


fn print_startup_info() raises:
    """Print startup information."""
    print("=" * 60)
    print("aequify v" + get_version())
    print("=" * 60)
    print("")

    # Setup uvloop
    if setup_uvloop():
        print("uvloop: enabled")
    else:
        print("uvloop: not available")

    # GPU info
    var vendor = gpu_vendor()
    if gpu_available():
        print("GPU: " + vendor)
        try:
            var gpu = GPUContext()
            print("  Name: " + gpu.device_name())
            var arch = gpu.arch_name()
            print("  Arch: " + arch)
            # Compute units: Apple/AMD use compile-time values, NVIDIA uses DeviceAttribute
            @parameter
            if has_apple_gpu_accelerator():
                var gpu_cores = 8  # Default for M1
                if arch == "apple-m2" or arch == "apple-m3" or arch == "apple-m4" or arch == "apple-m5":
                    gpu_cores = 10
                print("  Cores: " + String(gpu_cores))
            elif has_amd_gpu_accelerator():
                # AMD CU count from arch_name
                var cu_count = 0
                if arch == "gfx942":
                    cu_count = 304  # MI300X
                elif arch == "gfx950":
                    cu_count = 256  # MI355X
                elif arch == "gfx1030":
                    cu_count = 60   # Radeon 6900
                elif arch == "gfx1100":
                    cu_count = 96   # Radeon 7900
                elif arch == "gfx1101":
                    cu_count = 60   # Radeon 7800/7700
                elif arch == "gfx1102":
                    cu_count = 32   # Radeon 7600
                elif arch == "gfx1103":
                    cu_count = 12   # Radeon 780M
                elif arch == "gfx1150":
                    cu_count = 12   # Radeon 880M
                elif arch == "gfx1151":
                    cu_count = 40   # Radeon 8060S
                elif arch == "gfx1152":
                    cu_count = 8    # Radeon 860M
                elif arch == "gfx1200":
                    cu_count = 32   # Radeon 9060
                elif arch == "gfx1201":
                    cu_count = 64   # Radeon 9070
                print("  CUs: " + String(cu_count))
            else:
                print("  SMs: " + String(gpu.multiprocessor_count()))
            print("  Memory: " + String(Int(gpu.total_memory_gb() * 10) / 10.0) + " GB")
        except:
            print("  (could not get GPU info)")
    else:
        print("GPU: none")
    print("")


fn main() raises:
    """Main entrypoint for the Mojo application."""
    _setup_python_path()

    # Setup logging (before anything else, disable console to avoid TUI interference)
    var aequify_mod = Python.import_module("aequify")
    aequify_mod.setup_logging(console_output=PythonObject(False))

    var logger = _get_logger()
    logger.info("Aequify starting...")

    print_startup_info()

    # Import Python modules
    logger.debug("Importing engine and TUI modules")
    var engine_mod = Python.import_module("aequify.core.engine")
    var tui_mod = Python.import_module("aequify.tui")

    # Get GPU info to pass to TUI
    var gpu_info = PythonObject(None)
    if gpu_available():
        try:
            var gpu = GPUContext()
            gpu_info = gpu.to_dict()
            logger.info("GPU context created: " + gpu.device_name())
        except e:
            logger.warning("Failed to get GPU info: " + String(e))

    # Create and start engine in background thread
    print("Starting engine...")
    logger.info("Creating and starting engine")
    var engine = engine_mod.Engine()
    engine.start()
    logger.info("Engine started in background thread")
    print("Engine started in background thread")
    print("")

    # Run TUI on main thread (blocks until TUI exits)
    print("Starting TUI on main thread...")
    logger.info("Starting TUI on main thread")
    tui_mod.run_tui(engine, gpu_info=gpu_info)

    # Cleanup after TUI exits
    print("")
    print("TUI exited, shutting down...")
    logger.info("TUI exited, initiating shutdown")
    engine.stop()
    logger.debug("Engine stopped")
    shutdown_all(5.0)
    logger.debug("Runtime shutdown complete")
    logger.info("Aequify shutdown complete")
    aequify_mod.shutdown_logging()
    print("Goodbye!")
