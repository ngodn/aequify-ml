"""
System monitoring module - collects CPU, RAM, GPU, Network, and Disk I/O stats.

Used by Engine to publish system stats via PubSub for TUI consumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemStats:
    """System statistics snapshot."""

    # CPU
    cpu_percent: float = 0.0

    # Memory
    mem_used_app: int = 0  # Process RSS
    mem_available: int = 0
    mem_total: int = 0

    # GPU (NVIDIA)
    gpu_available: bool = False
    gpu_percent: float = 0.0
    gpu_mem_used: int = 0
    gpu_mem_total: int = 0

    # Network I/O (rates in bytes/sec)
    net_up: float = 0.0
    net_down: float = 0.0

    # Disk I/O (rates in bytes/sec)
    disk_read: float = 0.0
    disk_write: float = 0.0

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for PubSub serialization."""
        return {
            "cpu_percent": self.cpu_percent,
            "mem_used_app": self.mem_used_app,
            "mem_available": self.mem_available,
            "mem_total": self.mem_total,
            "gpu_available": self.gpu_available,
            "gpu_percent": self.gpu_percent,
            "gpu_mem_used": self.gpu_mem_used,
            "gpu_mem_total": self.gpu_mem_total,
            "net_up": self.net_up,
            "net_down": self.net_down,
            "disk_read": self.disk_read,
            "disk_write": self.disk_write,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemStats":
        """Create from dict (PubSub deserialization)."""
        return cls(
            cpu_percent=data.get("cpu_percent", 0.0),
            mem_used_app=data.get("mem_used_app", 0),
            mem_available=data.get("mem_available", 0),
            mem_total=data.get("mem_total", 0),
            gpu_available=data.get("gpu_available", False),
            gpu_percent=data.get("gpu_percent", 0.0),
            gpu_mem_used=data.get("gpu_mem_used", 0),
            gpu_mem_total=data.get("gpu_mem_total", 0),
            net_up=data.get("net_up", 0.0),
            net_down=data.get("net_down", 0.0),
            disk_read=data.get("disk_read", 0.0),
            disk_write=data.get("disk_write", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


class SystemMonitor:
    """
    Collects system statistics.

    Usage:
        monitor = SystemMonitor()
        stats = monitor.collect()  # Returns SystemStats
    """

    def __init__(self, collect_interval: float = 1.0) -> None:
        """
        Initialize the system monitor.

        Args:
            collect_interval: Expected interval between collect() calls (for rate calculation).
        """
        self._interval = collect_interval
        self._gpu_available = False
        self._gpu_handle: Any = None
        self._last_net_io: tuple[int, int] | None = None
        self._last_disk_io: tuple[int, int] | None = None
        self._last_collect_time: float = 0.0

        self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize GPU monitoring if available."""
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._gpu_available = True
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._gpu_available = False
            self._gpu_handle = None

    def collect(self) -> SystemStats:
        """
        Collect current system statistics.

        Returns:
            SystemStats with current values.
        """
        import psutil

        now = time.time()
        elapsed = now - self._last_collect_time if self._last_collect_time > 0 else self._interval
        self._last_collect_time = now

        stats = SystemStats(timestamp=now)

        # CPU
        try:
            stats.cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # Memory
        try:
            mem = psutil.virtual_memory()
            process = psutil.Process()
            stats.mem_used_app = process.memory_info().rss
            stats.mem_available = mem.available
            stats.mem_total = mem.total
        except Exception:
            pass

        # GPU
        if self._gpu_available and self._gpu_handle:
            try:
                import pynvml

                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                stats.gpu_available = True
                stats.gpu_percent = util.gpu
                stats.gpu_mem_used = mem_info.used
                stats.gpu_mem_total = mem_info.total
            except Exception:
                pass

        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            if self._last_net_io and elapsed > 0:
                stats.net_up = (net_io.bytes_sent - self._last_net_io[0]) / elapsed
                stats.net_down = (net_io.bytes_recv - self._last_net_io[1]) / elapsed
            self._last_net_io = (net_io.bytes_sent, net_io.bytes_recv)
        except Exception:
            pass

        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                if self._last_disk_io and elapsed > 0:
                    stats.disk_read = (disk_io.read_bytes - self._last_disk_io[0]) / elapsed
                    stats.disk_write = (disk_io.write_bytes - self._last_disk_io[1]) / elapsed
                self._last_disk_io = (disk_io.read_bytes, disk_io.write_bytes)
        except Exception:
            pass

        return stats
