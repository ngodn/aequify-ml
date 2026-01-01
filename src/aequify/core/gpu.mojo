"""
Aequify GPU - GPU context and Python bindings.

Provides GPUContext struct for Mojo and Python-callable functions.
"""

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from sys import (
    has_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)
from gpu.host import DeviceContext, DeviceStream, DeviceAttribute
from gpu.host.info import (
    AppleMetalFamily,
    AMDCDNA3Family,
    AMDCDNA4Family,
    AMDRDNAFamily,
)


# =============================================================================
# Logging
# =============================================================================


fn _get_logger() raises -> PythonObject:
    """Get the logger for this module."""
    var logging_mod = Python.import_module("aequify.logging")
    return logging_mod.get_logger("aequify.core.gpu")


# =============================================================================
# GPU Detection Functions
# =============================================================================


fn gpu_available() -> Bool:
    """Check if any GPU acceleration is available."""
    return has_accelerator()


fn has_nvidia_gpu() -> Bool:
    """Check if NVIDIA GPU (CUDA) is available."""
    return has_nvidia_gpu_accelerator()


fn has_amd_gpu() -> Bool:
    """Check if AMD GPU (ROCm/HIP) is available."""
    return has_amd_gpu_accelerator()


fn has_apple_gpu() -> Bool:
    """Check if Apple GPU (Metal) is available."""
    return has_apple_gpu_accelerator()


fn gpu_vendor() -> String:
    """
    Get the GPU vendor type.

    Returns:
        "nvidia" for NVIDIA GPUs (CUDA).
        "amd" for AMD GPUs (ROCm/HIP).
        "apple" for Apple GPUs (Metal).
        "none" if no GPU is available.
    """
    @parameter
    if has_nvidia_gpu_accelerator():
        return "nvidia"
    elif has_amd_gpu_accelerator():
        return "amd"
    elif has_apple_gpu_accelerator():
        return "apple"
    else:
        return "none"


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
        var logger = _get_logger()
        logger.debug("Creating GPU context for device " + String(device_id))
        self.ctx = DeviceContext(device_id)
        self.stream = DeviceStream(self.ctx)
        logger.info("GPU context created: " + self.ctx.name() + " (device " + String(device_id) + ")")

    fn synchronize(self) raises:
        """Wait for all GPU operations to complete."""
        var logger = _get_logger()
        logger.debug("Synchronizing GPU stream")
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
        # Vendor detection (compile-time)
        info.__setitem__("vendor", value=gpu_vendor())
        info.__setitem__("is_nvidia", value=has_nvidia_gpu())
        info.__setitem__("is_amd", value=has_amd_gpu())
        info.__setitem__("is_apple", value=has_apple_gpu())

        # Device identification
        info.__setitem__("name", value=self.ctx.name())
        info.__setitem__("device_id", value=Int(self.ctx.id()))
        info.__setitem__("api", value=self.ctx.api())
        info.__setitem__("arch", value=self.ctx.arch_name())
        info.__setitem__("api_version", value=self.api_version())
        info.__setitem__("is_compatible", value=self.is_compatible())

        # All GPU attributes - Apple/AMD use compile-time constants, NVIDIA uses DeviceAttribute
        var arch = self.ctx.arch_name()
        @parameter
        if has_apple_gpu_accelerator():
            # Apple Metal: ALL values from compile-time constants (DeviceAttribute doesn't work)
            # GPU core count is device-specific, derive from arch_name
            var gpu_cores = 8  # Default for M1
            if arch == "apple-m2" or arch == "apple-m3" or arch == "apple-m4" or arch == "apple-m5":
                gpu_cores = 10

            # Compute capability (not applicable to Metal)
            info.__setitem__("compute_capability_major", value=0)
            info.__setitem__("compute_capability_minor", value=0)

            # Compute units
            info.__setitem__("multiprocessor_count", value=gpu_cores)
            info.__setitem__("warp_size", value=Int(AppleMetalFamily.warp_size))
            info.__setitem__("max_threads_per_sm", value=Int(AppleMetalFamily.threads_per_multiprocessor))
            info.__setitem__("max_registers_per_block", value=Int(AppleMetalFamily.max_registers_per_block))
            info.__setitem__("max_shared_memory_per_sm", value=Int(AppleMetalFamily.shared_memory_per_multiprocessor))

            # Clock rate not queryable on Metal
            info.__setitem__("clock_rate_mhz", value=0)
            info.__setitem__("supports_cooperative_launch", value=True)

            # Thread limits (from AppleMetalFamily.max_thread_block_size)
            info.__setitem__("max_threads_per_block", value=Int(AppleMetalFamily.max_thread_block_size))

            # Block dimensions (Metal supports 1024 per dimension, limited by total threads)
            info.__setitem__("max_block_dim_x", value=1024)
            info.__setitem__("max_block_dim_y", value=1024)
            info.__setitem__("max_block_dim_z", value=1024)

            # Grid dimensions (Metal supports very large grids)
            info.__setitem__("max_grid_dim_x", value=2147483647)
            info.__setitem__("max_grid_dim_y", value=65535)
            info.__setitem__("max_grid_dim_z", value=65535)

            # Memory
            info.__setitem__("memory_free", value=Int(mem_info[0]))
            info.__setitem__("memory_total", value=Int(mem_info[1]))
            info.__setitem__("max_shared_memory_per_block", value=Int(AppleMetalFamily.shared_memory_per_multiprocessor))

        elif has_amd_gpu_accelerator():
            # AMD ROCm/HIP: use compile-time constants from CDNA/RDNA families
            var cu_count = 0
            var is_cdna = False

            # CDNA datacenter GPUs (gfx94x, gfx95x)
            if arch == "gfx942":
                cu_count = 304  # MI300X
                is_cdna = True
            elif arch == "gfx950":
                cu_count = 256  # MI355X
                is_cdna = True
            # RDNA consumer GPUs (gfx10xx, gfx11xx, gfx12xx)
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

            # Compute capability (not applicable to AMD)
            info.__setitem__("compute_capability_major", value=0)
            info.__setitem__("compute_capability_minor", value=0)

            info.__setitem__("multiprocessor_count", value=cu_count)

            # Use appropriate family constants
            if is_cdna:
                if arch == "gfx950":
                    info.__setitem__("warp_size", value=Int(AMDCDNA4Family.warp_size))
                    info.__setitem__("max_threads_per_sm", value=Int(AMDCDNA4Family.threads_per_multiprocessor))
                    info.__setitem__("max_registers_per_block", value=Int(AMDCDNA4Family.max_registers_per_block))
                    info.__setitem__("max_shared_memory_per_sm", value=Int(AMDCDNA4Family.shared_memory_per_multiprocessor))
                    info.__setitem__("max_threads_per_block", value=Int(AMDCDNA4Family.max_thread_block_size))
                    info.__setitem__("max_shared_memory_per_block", value=Int(AMDCDNA4Family.shared_memory_per_multiprocessor))
                else:
                    info.__setitem__("warp_size", value=Int(AMDCDNA3Family.warp_size))
                    info.__setitem__("max_threads_per_sm", value=Int(AMDCDNA3Family.threads_per_multiprocessor))
                    info.__setitem__("max_registers_per_block", value=Int(AMDCDNA3Family.max_registers_per_block))
                    info.__setitem__("max_shared_memory_per_sm", value=Int(AMDCDNA3Family.shared_memory_per_multiprocessor))
                    info.__setitem__("max_threads_per_block", value=Int(AMDCDNA3Family.max_thread_block_size))
                    info.__setitem__("max_shared_memory_per_block", value=Int(AMDCDNA3Family.shared_memory_per_multiprocessor))
            else:
                info.__setitem__("warp_size", value=Int(AMDRDNAFamily.warp_size))
                info.__setitem__("max_threads_per_sm", value=Int(AMDRDNAFamily.threads_per_multiprocessor))
                info.__setitem__("max_registers_per_block", value=Int(AMDRDNAFamily.max_registers_per_block))
                info.__setitem__("max_shared_memory_per_sm", value=Int(AMDRDNAFamily.shared_memory_per_multiprocessor))
                info.__setitem__("max_threads_per_block", value=Int(AMDRDNAFamily.max_thread_block_size))
                info.__setitem__("max_shared_memory_per_block", value=Int(AMDRDNAFamily.shared_memory_per_multiprocessor))

            # Clock rate and cooperative launch - query via HIP
            info.__setitem__("clock_rate_mhz", value=self.clock_rate_mhz())
            info.__setitem__("supports_cooperative_launch", value=self.supports_cooperative_launch())

            # Block dimensions (AMD supports 1024 per dimension)
            info.__setitem__("max_block_dim_x", value=1024)
            info.__setitem__("max_block_dim_y", value=1024)
            info.__setitem__("max_block_dim_z", value=1024)

            # Grid dimensions
            info.__setitem__("max_grid_dim_x", value=2147483647)
            info.__setitem__("max_grid_dim_y", value=65535)
            info.__setitem__("max_grid_dim_z", value=65535)

            # Memory
            info.__setitem__("memory_free", value=Int(mem_info[0]))
            info.__setitem__("memory_total", value=Int(mem_info[1]))

        else:
            # NVIDIA: query runtime DeviceAttribute
            info.__setitem__("compute_capability_major", value=self.compute_capability_major())
            info.__setitem__("compute_capability_minor", value=self.compute_capability_minor())

            info.__setitem__("multiprocessor_count", value=self.multiprocessor_count())
            info.__setitem__("warp_size", value=self.warp_size())
            info.__setitem__("max_threads_per_sm", value=self.max_threads_per_multiprocessor())
            info.__setitem__("max_registers_per_block", value=self.max_registers_per_block())
            info.__setitem__("max_registers_per_sm", value=self.max_registers_per_multiprocessor())

            info.__setitem__("clock_rate_mhz", value=self.clock_rate_mhz())
            info.__setitem__("supports_cooperative_launch", value=self.supports_cooperative_launch())

            info.__setitem__("max_threads_per_block", value=self.max_threads_per_block())
            info.__setitem__("max_blocks_per_sm", value=self.max_blocks_per_multiprocessor())

            info.__setitem__("max_block_dim_x", value=self.max_block_dim_x())
            info.__setitem__("max_block_dim_y", value=self.max_block_dim_y())
            info.__setitem__("max_block_dim_z", value=self.max_block_dim_z())

            info.__setitem__("max_grid_dim_x", value=self.max_grid_dim_x())
            info.__setitem__("max_grid_dim_y", value=self.max_grid_dim_y())
            info.__setitem__("max_grid_dim_z", value=self.max_grid_dim_z())

            info.__setitem__("memory_free", value=Int(mem_info[0]))
            info.__setitem__("memory_total", value=Int(mem_info[1]))
            info.__setitem__("max_shared_memory_per_block", value=self.max_shared_memory_per_block())

        return info


# =============================================================================
# Python Bindings
# =============================================================================


fn gpu_available_py(args: PythonObject) raises -> PythonObject:
    var logger = _get_logger()
    var available = gpu_available()
    logger.debug("gpu_available() called, result: " + String(available))
    return PythonObject(available)


fn has_nvidia_gpu_py(args: PythonObject) raises -> PythonObject:
    var logger = _get_logger()
    var result = has_nvidia_gpu()
    logger.debug("has_nvidia_gpu() called, result: " + String(result))
    return PythonObject(result)


fn has_amd_gpu_py(args: PythonObject) raises -> PythonObject:
    var logger = _get_logger()
    var result = has_amd_gpu()
    logger.debug("has_amd_gpu() called, result: " + String(result))
    return PythonObject(result)


fn has_apple_gpu_py(args: PythonObject) raises -> PythonObject:
    var logger = _get_logger()
    var result = has_apple_gpu()
    logger.debug("has_apple_gpu() called, result: " + String(result))
    return PythonObject(result)


fn gpu_vendor_py(args: PythonObject) raises -> PythonObject:
    var logger = _get_logger()
    var vendor = gpu_vendor()
    logger.debug("gpu_vendor() called, result: " + vendor)
    return PythonObject(vendor)


fn device_count_py(args: PythonObject) raises -> PythonObject:
    var logger = _get_logger()
    var count = device_count()
    logger.debug("device_count() called, result: " + String(count))
    return PythonObject(count)


fn get_all_info_py(args: PythonObject) raises -> PythonObject:
    var logger = _get_logger()
    var device_id = Int(args[0]) if len(args) > 0 else 0
    logger.debug("get_all_info() called for device " + String(device_id))
    var gpu = GPUContext(device_id)
    return gpu.to_dict()


@export
fn PyInit_gpu() -> PythonObject:
    """Initialize the gpu Python module."""
    try:
        var m = PythonModuleBuilder("gpu")
        m.def_function[gpu_available_py]("gpu_available", docstring="Check if any GPU is available")
        m.def_function[has_nvidia_gpu_py]("has_nvidia_gpu", docstring="Check if NVIDIA GPU is available")
        m.def_function[has_amd_gpu_py]("has_amd_gpu", docstring="Check if AMD GPU is available")
        m.def_function[has_apple_gpu_py]("has_apple_gpu", docstring="Check if Apple GPU is available")
        m.def_function[gpu_vendor_py]("gpu_vendor", docstring="Get GPU vendor (nvidia/amd/apple/none)")
        m.def_function[device_count_py]("device_count", docstring="Get number of GPU devices")
        m.def_function[get_all_info_py]("get_all_info", docstring="Get all GPU info as dict")
        return m.finalize()
    except e:
        return PythonObject()
