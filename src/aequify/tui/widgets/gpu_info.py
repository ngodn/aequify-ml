"""GPU Information Modal - Displays detailed GPU specs from Mojo gpu module."""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static


def _format_bytes(bytes_val: int, suffix: str = "B") -> str:
    """Format bytes to human readable string."""
    val = float(bytes_val)
    for unit in ("", "K", "M", "G", "T"):
        if abs(val) < 1024.0:
            return f"{val:.1f} {unit}{suffix}"
        val /= 1024.0
    return f"{val:.1f} P{suffix}"


class GPUInfoModal(ModalScreen[None]):
    """Modal screen displaying GPU information."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    GPUInfoModal {
        align: center middle;
    }

    GPUInfoModal > Container {
        width: 76;
        height: auto;
        max-height: 32;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    GPUInfoModal .modal-title {
        text-align: center;
        text-style: bold;
        color: $text;
        padding-bottom: 1;
    }

    GPUInfoModal .device-header {
        text-align: center;
        padding-bottom: 1;
    }

    GPUInfoModal .section-grid {
        grid-size: 2;
        grid-gutter: 1 2;
        height: auto;
    }

    GPUInfoModal .section-panel {
        height: auto;
        padding: 0 1;
    }

    GPUInfoModal .section-title {
        text-style: bold;
        color: $primary;
    }

    GPUInfoModal .info-row {
        height: 1;
    }

    GPUInfoModal .info-label {
        color: $text-muted;
        width: 20;
    }

    GPUInfoModal .info-value {
        width: 1fr;
    }

    GPUInfoModal .memory-bar {
        padding-top: 1;
    }

    GPUInfoModal .footer-hint {
        text-align: center;
        color: $text-muted;
        padding-top: 1;
    }

    GPUInfoModal .no-gpu {
        text-align: center;
        color: $warning;
        padding: 2;
    }
    """

    def __init__(self, gpu_info: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._gpu_info = dict(gpu_info) if gpu_info else None

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("GPU Information", classes="modal-title")

            if self._gpu_info is None:
                yield Static("No GPU available or Mojo module not loaded", classes="no-gpu")
            else:
                yield self._build_content()

            yield Static("Press ESC to close", classes="footer-hint")

    def _build_content(self) -> Static:
        """Build the GPU info content as Rich renderable."""
        info = self._gpu_info
        if info is None:
            return Static("")

        # Header
        name = info.get("name", "Unknown GPU")
        vendor = str(info.get("vendor", "unknown")).upper()
        api = str(info.get("api", "")).upper()
        arch = info.get("arch", "")
        cc_major = info.get("compute_capability_major", 0)
        cc_minor = info.get("compute_capability_minor", 0)
        is_compat = info.get("is_compatible", False)

        # Vendor color
        vendor_style = {"NVIDIA": "green", "AMD": "red", "APPLE": "magenta"}.get(vendor, "dim")

        header_text = Text()
        header_text.append(f"{name} ", style="bold")
        header_text.append(f"[{vendor}]\n", style=vendor_style)
        header_text.append("API: ", style="dim")
        header_text.append(f"{api}    ", style="cyan")
        header_text.append("Arch: ", style="dim")
        header_text.append(f"{arch}    ", style="cyan")
        # Compute capability only meaningful for NVIDIA
        if vendor == "NVIDIA":
            header_text.append("Compute: ", style="dim")
            header_text.append(f"{cc_major}.{cc_minor}    ", style="cyan")
        header_text.append("Compatible: ", style="dim")
        header_text.append("Yes" if is_compat else "No", style="green" if is_compat else "red")

        # Create 2x2 grid tables
        # Top-left: Compute
        compute_table = Table(box=None, show_header=False, padding=(0, 1))
        compute_table.add_column("Label", style="dim", width=18)
        compute_table.add_column("Value", width=14)

        # Compute unit count - vendor-specific terminology
        is_apple = info.get("is_apple", False)
        is_amd = info.get("is_amd", False)
        sm_count = info.get("multiprocessor_count")
        if is_apple:
            compute_table.add_row("GPU Cores", f"{sm_count}" if sm_count else "[dim]N/A[/]")
        elif is_amd:
            compute_table.add_row("Compute Units", f"{sm_count} CUs" if sm_count else "[dim]N/A[/]")
        else:
            compute_table.add_row("Multiprocessors", f"{sm_count} SMs" if sm_count else "[dim]N/A[/]")
        compute_table.add_row("Clock Rate", f"{info.get('clock_rate_mhz', 0)} MHz")
        warp = info.get("warp_size")
        # Vendor-specific terminology: Apple=SIMD, AMD=Wavefront, NVIDIA=Warp
        if is_apple:
            warp_label = "SIMD Width"
        elif is_amd:
            warp_label = "Wavefront"
        else:
            warp_label = "Warp Size"
        compute_table.add_row(warp_label, f"{warp} threads" if warp else "[dim]N/A[/]")
        compute_table.add_row("API Version", str(info.get("api_version", 0)))
        coop = info.get("supports_cooperative_launch", False)
        compute_table.add_row("Cooperative", "[green]Supported[/]" if coop else "[dim]No[/]")
        compute_table.add_row("Device Count", "1")
        compute_table.add_row("Device ID", str(info.get("device_id", 0)))

        # Top-right: Memory
        # Apple GPUs use unified memory architecture (shared with system RAM)
        mem_label = "Unified" if is_apple else "VRAM"
        mem_total = info.get("memory_total", 0)
        mem_free = info.get("memory_free", 0)
        mem_used = mem_total - mem_free
        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

        memory_table = Table(box=None, show_header=False, padding=(0, 1))
        memory_table.add_column("Label", style="dim", width=18)
        memory_table.add_column("Value", width=14)

        memory_table.add_row(f"{mem_label} Total", _format_bytes(mem_total))
        memory_table.add_row(f"{mem_label} Free", _format_bytes(mem_free))
        memory_table.add_row(f"{mem_label} Used", _format_bytes(mem_used))
        memory_table.add_row("", "")  # spacer

        # Memory bar
        bar_width = 28
        filled = int(bar_width * mem_pct / 100)
        bar_color = "green" if mem_pct < 70 else ("yellow" if mem_pct < 90 else "red")
        bar_str = f"[{bar_color}]{'█' * filled}[/][dim]{'░' * (bar_width - filled)}[/] {mem_pct:.0f}%"
        memory_table.add_row("", bar_str)

        # Vendor-specific unit label: NVIDIA=SM, AMD=CU, Apple=Core
        unit_label = "CU" if is_amd else ("Core" if is_apple else "SM")

        memory_table.add_row("Shared/Block", _format_bytes(info.get("max_shared_memory_per_block", 0)))
        shared_sm = info.get("max_shared_memory_per_sm")
        memory_table.add_row(f"Shared/{unit_label}", _format_bytes(shared_sm) if shared_sm else "[dim]N/A[/]")
        regs_block = info.get("max_registers_per_block")
        memory_table.add_row("Registers/Block", str(regs_block) if regs_block else "[dim]N/A[/]")
        regs_sm = info.get("max_registers_per_sm")
        memory_table.add_row(f"Registers/{unit_label}", str(regs_sm) if regs_sm else "[dim]N/A[/]")

        # Bottom-left: Threads
        threads_table = Table(box=None, show_header=False, padding=(0, 1))
        threads_table.add_column("Label", style="dim", width=18)
        threads_table.add_column("Value", width=14)

        threads_table.add_row("Max Threads/Block", str(info.get("max_threads_per_block", 0)))
        threads_sm = info.get("max_threads_per_sm")
        threads_table.add_row(f"Max Threads/{unit_label}", str(threads_sm) if threads_sm else "[dim]N/A[/]")
        blocks_sm = info.get("max_blocks_per_sm")
        threads_table.add_row(f"Max Blocks/{unit_label}", str(blocks_sm) if blocks_sm else "[dim]N/A[/]")

        # Bottom-right: Grid
        grid_table = Table(box=None, show_header=False, padding=(0, 1))
        grid_table.add_column("Label", style="dim", width=18)
        grid_table.add_column("Value", width=14)

        grid_table.add_row("Max Block Dim X", str(info.get("max_block_dim_x", 0)))
        grid_table.add_row("Max Block Dim Y", str(info.get("max_block_dim_y", 0)))
        grid_table.add_row("Max Block Dim Z", str(info.get("max_block_dim_z", 0)))
        grid_table.add_row("Max Grid Dim X", str(info.get("max_grid_dim_x", 0)))
        grid_table.add_row("Max Grid Dim Y", str(info.get("max_grid_dim_y", 0)))
        grid_table.add_row("Max Grid Dim Z", str(info.get("max_grid_dim_z", 0)))

        # Create panels
        compute_panel = Panel(compute_table, title="[bold]Compute[/]", border_style="dim")
        memory_panel = Panel(memory_table, title="[bold]Memory[/]", border_style="dim")
        threads_panel = Panel(threads_table, title="[bold]Threads[/]", border_style="dim")
        grid_panel = Panel(grid_table, title="[bold]Grid[/]", border_style="dim")

        # Create main grid layout
        main_grid = Table.grid(padding=(0, 1))
        main_grid.add_column(width=35)
        main_grid.add_column(width=35)

        main_grid.add_row(compute_panel, memory_panel)
        main_grid.add_row(threads_panel, grid_panel)

        # Combine header and grid
        content = Group(
            Panel(header_text, border_style="cyan", padding=(0, 1)),
            main_grid,
        )

        return Static(content)
