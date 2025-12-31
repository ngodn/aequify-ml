"""
Aequify GPU - GPU context and Python bindings.

Provides GPUContext struct for Mojo and Python-callable functions.
"""

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from sys.info import has_accelerator
from gpu.host import DeviceContext, DeviceStream, DeviceAttribute


# =============================================================================
# GPU Functions
# =============================================================================


fn gpu_available() -> Bool:
    """Check if GPU acceleration is available."""
    return has_accelerator()


fn device_count() raises -> Int:
    """Get the number of available GPU devices."""
    return DeviceContext.number_of_devices()


# =============================================================================
# GPU Context Struct
# =============================================================================


struct GPUContext:
    """
    GPU context for compute operations.

    Wraps DeviceContext with convenient accessors for GPU info.
    """

    var ctx: DeviceContext
    var stream: DeviceStream

    fn __init__(out self, device_id: Int = 0) raises:
        """Create a GPU context for the specified device."""
        self.ctx = DeviceContext(device_id)
        self.stream = DeviceStream(self.ctx)

    fn synchronize(self) raises:
        """Wait for all GPU operations to complete."""
        self.stream.synchronize()

    # -------------------------------------------------------------------------
    # Device Identification
    # -------------------------------------------------------------------------

    fn device_name(self) raises -> String:
        """Get the GPU device name."""
        return self.ctx.name()

    fn device_id(self) raises -> Int64:
        """Get the GPU device ID."""
        return self.ctx.id()

    fn api(self) raises -> String:
        """Get the GPU API type (cuda, hip, or cpu)."""
        return self.ctx.api()

    fn arch_name(self) raises -> String:
        """Get the GPU architecture name (e.g., sm_90, gfx942)."""
        return self.ctx.arch_name()

    # -------------------------------------------------------------------------
    # Compute Capability
    # -------------------------------------------------------------------------

    fn compute_capability_major(self) raises -> Int:
        """Get compute capability major version (NVIDIA)."""
        return self.ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR)

    fn compute_capability_minor(self) raises -> Int:
        """Get compute capability minor version (NVIDIA)."""
        return self.ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR)

    fn compute_capability(self) raises -> String:
        """Get compute capability as string (e.g., '8.9')."""
        var major = self.compute_capability_major()
        var minor = self.compute_capability_minor()
        return String(major) + "." + String(minor)

    # -------------------------------------------------------------------------
    # Multiprocessor Info
    # -------------------------------------------------------------------------

    fn multiprocessor_count(self) raises -> Int:
        """Get number of streaming multiprocessors (SMs)."""
        return self.ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    fn clock_rate_khz(self) raises -> Int:
        """Get GPU clock rate in kHz."""
        return self.ctx.get_attribute(DeviceAttribute.CLOCK_RATE)

    fn clock_rate_mhz(self) raises -> Int:
        """Get GPU clock rate in MHz."""
        return self.clock_rate_khz() // 1000

    # -------------------------------------------------------------------------
    # Thread/Block Configuration
    # -------------------------------------------------------------------------

    fn warp_size(self) raises -> Int:
        """Get warp size (typically 32 for NVIDIA, 64 for AMD)."""
        return self.ctx.get_attribute(DeviceAttribute.WARP_SIZE)

    fn max_threads_per_block(self) raises -> Int:
        """Get maximum threads per block."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK)

    fn max_threads_per_multiprocessor(self) raises -> Int:
        """Get maximum resident threads per SM."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR)

    fn max_blocks_per_multiprocessor(self) raises -> Int:
        """Get maximum resident blocks per SM."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR)

    fn max_block_dim_x(self) raises -> Int:
        """Get maximum block dimension X."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_X)

    fn max_block_dim_y(self) raises -> Int:
        """Get maximum block dimension Y."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Y)

    fn max_block_dim_z(self) raises -> Int:
        """Get maximum block dimension Z."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Z)

    fn max_grid_dim_x(self) raises -> Int:
        """Get maximum grid dimension X."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)

    fn max_grid_dim_y(self) raises -> Int:
        """Get maximum grid dimension Y."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Y)

    fn max_grid_dim_z(self) raises -> Int:
        """Get maximum grid dimension Z."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Z)

    # -------------------------------------------------------------------------
    # Memory Info
    # -------------------------------------------------------------------------

    fn max_shared_memory_per_block(self) raises -> Int:
        """Get maximum shared memory per block (bytes)."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)

    fn max_shared_memory_per_multiprocessor(self) raises -> Int:
        """Get maximum shared memory per SM (bytes)."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)

    fn max_registers_per_block(self) raises -> Int:
        """Get maximum 32-bit registers per block."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_REGISTERS_PER_BLOCK)

    fn max_registers_per_multiprocessor(self) raises -> Int:
        """Get maximum 32-bit registers per SM."""
        return self.ctx.get_attribute(DeviceAttribute.MAX_REGISTERS_PER_MULTIPROCESSOR)

    fn memory_info(self) raises -> Tuple[UInt, UInt]:
        """Get memory info as (free_bytes, total_bytes)."""
        return self.ctx.get_memory_info()

    fn total_memory_gb(self) raises -> Float64:
        """Get total GPU memory in GB."""
        var info = self.memory_info()
        return Float64(info[1]) / (1024.0 * 1024.0 * 1024.0)

    fn free_memory_gb(self) raises -> Float64:
        """Get free GPU memory in GB."""
        var info = self.memory_info()
        return Float64(info[0]) / (1024.0 * 1024.0 * 1024.0)

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    fn supports_cooperative_launch(self) raises -> Bool:
        """Check if cooperative kernel launches are supported."""
        return self.ctx.get_attribute(DeviceAttribute.COOPERATIVE_LAUNCH) != 0

    fn is_compatible(self) raises -> Bool:
        """Check if device is compatible with MAX runtime."""
        return self.ctx.is_compatible()

    fn api_version(self) raises -> Int:
        """Get the driver/API version."""
        return self.ctx.get_api_version()

    # -------------------------------------------------------------------------
    # Python Dict Export
    # -------------------------------------------------------------------------

    fn to_dict(self) raises -> PythonObject:
        """Get all GPU info as a Python dict."""
        var mem_info = self.memory_info()

        var info = Python.dict()
        # Device identification
        info.__setitem__("name", value=self.ctx.name())
        info.__setitem__("device_id", value=Int(self.ctx.id()))
        info.__setitem__("api", value=self.ctx.api())
        info.__setitem__("arch", value=self.ctx.arch_name())
        info.__setitem__("api_version", value=self.api_version())
        info.__setitem__("is_compatible", value=self.is_compatible())

        # Compute capability
        info.__setitem__("compute_capability_major", value=self.compute_capability_major())
        info.__setitem__("compute_capability_minor", value=self.compute_capability_minor())

        # Compute units
        info.__setitem__("multiprocessor_count", value=self.multiprocessor_count())
        info.__setitem__("clock_rate_mhz", value=self.clock_rate_mhz())
        info.__setitem__("warp_size", value=self.warp_size())
        info.__setitem__("supports_cooperative_launch", value=self.supports_cooperative_launch())

        # Thread limits
        info.__setitem__("max_threads_per_block", value=self.max_threads_per_block())
        info.__setitem__("max_threads_per_sm", value=self.max_threads_per_multiprocessor())
        info.__setitem__("max_blocks_per_sm", value=self.max_blocks_per_multiprocessor())

        # Block dimensions
        info.__setitem__("max_block_dim_x", value=self.max_block_dim_x())
        info.__setitem__("max_block_dim_y", value=self.max_block_dim_y())
        info.__setitem__("max_block_dim_z", value=self.max_block_dim_z())

        # Grid dimensions
        info.__setitem__("max_grid_dim_x", value=self.max_grid_dim_x())
        info.__setitem__("max_grid_dim_y", value=self.max_grid_dim_y())
        info.__setitem__("max_grid_dim_z", value=self.max_grid_dim_z())

        # Memory
        info.__setitem__("memory_free", value=Int(mem_info[0]))
        info.__setitem__("memory_total", value=Int(mem_info[1]))
        info.__setitem__("max_shared_memory_per_block", value=self.max_shared_memory_per_block())
        info.__setitem__("max_shared_memory_per_sm", value=self.max_shared_memory_per_multiprocessor())
        info.__setitem__("max_registers_per_block", value=self.max_registers_per_block())
        info.__setitem__("max_registers_per_sm", value=self.max_registers_per_multiprocessor())

        return info


# =============================================================================
# Python Bindings
# =============================================================================


fn gpu_available_py(args: PythonObject) raises -> PythonObject:
    return PythonObject(gpu_available())


fn device_count_py(args: PythonObject) raises -> PythonObject:
    return PythonObject(device_count())


fn get_all_info_py(args: PythonObject) raises -> PythonObject:
    var device_id = Int(args[0]) if len(args) > 0 else 0
    var gpu = GPUContext(device_id)
    return gpu.to_dict()


@export
fn PyInit_gpu() -> PythonObject:
    """Initialize the gpu Python module."""
    try:
        var m = PythonModuleBuilder("gpu")
        m.def_function[gpu_available_py]("gpu_available", docstring="Check if GPU is available")
        m.def_function[device_count_py]("device_count", docstring="Get number of GPU devices")
        m.def_function[get_all_info_py]("get_all_info", docstring="Get all GPU info as dict")
        return m.finalize()
    except e:
        return PythonObject()
