"""
System Benchmarking & Computational Profiling Package (Milestone M10).
"""

from src.benchmarking.profiler import (
    StageBenchmarkResult,
    SystemProfiler,
    get_hardware_specs,
    plot_memory_waterfall,
    plot_rtf_scaling_curves,
    plot_stage_latency_breakdown,
)

__all__ = [
    "SystemProfiler",
    "StageBenchmarkResult",
    "get_hardware_specs",
    "plot_stage_latency_breakdown",
    "plot_rtf_scaling_curves",
    "plot_memory_waterfall",
]

