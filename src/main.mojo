"""
Aequify - Mojo Entrypoint

Main entrypoint for the Aequify ML application.
Starts the engine in a background thread and runs TUI on main thread.
"""

from python import Python, PythonObject
from aequify import (
    get_version,
    setup_uvloop,
    shutdown_all,
    gpu_available,
    GPUContext,
)


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
    if gpu_available():
        print("GPU: available")
        try:
            var gpu = GPUContext()
            print("  Name: " + gpu.device_name())
            print("  Arch: " + gpu.arch_name())
            print("  SMs: " + String(gpu.multiprocessor_count()))
            print("  Memory: " + String(Int(gpu.total_memory_gb() * 10) / 10.0) + " GB")
        except:
            print("  (could not get GPU info)")
    else:
        print("GPU: not available")
    print("")


fn main() raises:
    """Main entrypoint for the Mojo application."""
    _setup_python_path()
    print_startup_info()

    # Import Python modules
    var engine_mod = Python.import_module("aequify.engine")
    var tui_mod = Python.import_module("aequify.tui")

    # Create and start engine in background thread
    print("Starting engine...")
    var engine = engine_mod.Engine()
    engine.start()
    print("Engine started in background thread")
    print("")

    # Run TUI on main thread (blocks until TUI exits)
    print("Starting TUI on main thread...")
    tui_mod.run_tui(engine)

    # Cleanup after TUI exits
    print("")
    print("TUI exited, shutting down...")
    engine.stop()
    shutdown_all(5.0)
    print("Goodbye!")
