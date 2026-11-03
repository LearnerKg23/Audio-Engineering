"""
System Benchmark & Computational Profiler (Milestone M10).

Provides high-resolution measurement of:
1. Wall-Clock Latency & Processing Time (T_proc) via time.perf_counter()
2. Real-Time Factor (RTF = T_proc / T_audio)
3. Audio Processing Throughput (T_audio / T_proc)
4. Peak System Memory (RAM) via psutil & tracemalloc
5. Peak GPU VRAM (CUDA & Apple Silicon MPS)
6. Algorithmic Scaling Curves across audio durations (15s, 30s, 60s, 120s)
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import json
import logging
from pathlib import Path
import platform
import time
import tracemalloc
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("system_profiler")


def get_hardware_specs() -> Dict[str, Any]:
    """Inspects and returns system hardware, host architecture, and accelerator capabilities."""
    vm = psutil.virtual_memory()
    specs = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_architecture": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "available_ram_gb": round(vm.available / (1024 ** 3), 2),
        "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_device_count": 0,
        "mps_available": False,
    }

    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            specs["cuda_available"] = True
            specs["cuda_device_count"] = torch.cuda.device_count()
            specs["cuda_device_name"] = torch.cuda.get_device_name(0)
            specs["cuda_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            specs["mps_available"] = True

    return specs


@dataclass
class StageBenchmarkResult:
    """Stores computational performance metrics for a single stage execution."""
    stage_name: str
    audio_duration_seconds: float
    elapsed_seconds: float
    rtf: float
    throughput_ratio: float
    ram_initial_mb: float
    ram_peak_mb: float
    ram_delta_mb: float
    vram_peak_mb: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemProfiler:
    """
    High-precision computational profiler tracking latency, memory, and RTF.
    """

    def __init__(self, warmup: bool = True):
        self.warmup = warmup
        self.process = psutil.Process()
        self.records: List[StageBenchmarkResult] = []

    def _get_current_ram_mb(self) -> float:
        """Returns current process Resident Set Size (RSS) in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def _get_peak_vram_mb(self) -> float:
        """Returns peak allocated GPU VRAM in MB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def _reset_vram_tracker(self) -> None:
        """Resets PyTorch CUDA peak memory tracker if active."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def profile_scope(
        self,
        stage_name: str,
        audio_duration_seconds: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager measuring execution time, peak RAM, and peak VRAM of a code block.
        """
        gc.collect()
        self._reset_vram_tracker()
        tracemalloc.start()

        ram_start = self._get_current_ram_mb()
        t0 = time.perf_counter()

        metrics_ctx: Dict[str, Any] = {}
        try:
            yield metrics_ctx
        finally:
            t_elapsed = time.perf_counter() - t0
            _, tm_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            ram_end = self._get_current_ram_mb()
            ram_delta = round(max(0.0, ram_end - ram_start), 2)
            ram_peak = round(ram_start + (tm_peak / (1024 * 1024)), 2)
            vram_peak = round(self._get_peak_vram_mb(), 2)

            rtf = round(t_elapsed / audio_duration_seconds if audio_duration_seconds > 0 else 0.0, 4)
            throughput = round(audio_duration_seconds / t_elapsed if t_elapsed > 0 else 0.0, 2)

            rec = StageBenchmarkResult(
                stage_name=stage_name,
                audio_duration_seconds=round(audio_duration_seconds, 2),
                elapsed_seconds=round(t_elapsed, 4),
                rtf=rtf,
                throughput_ratio=throughput,
                ram_initial_mb=round(ram_start, 2),
                ram_peak_mb=ram_peak,
                ram_delta_mb=ram_delta,
                vram_peak_mb=vram_peak,
                details=details or {},
            )
            self.records.append(rec)
            logger.info(
                f"[Benchmark] {stage_name:<24} | Time: {t_elapsed:.3f}s | "
                f"RTF: {rtf:<6.4f} ({throughput}x) | Peak RAM: {ram_peak:.1f} MB"
            )

    def benchmark_callable(
        self,
        func: Callable[[], Any],
        stage_name: str,
        audio_duration_seconds: float,
        num_runs: int = 3,
    ) -> StageBenchmarkResult:
        """
        Executes an optional warmup run followed by multi-pass average benchmarking.
        """
        if self.warmup:
            try:
                func()
            except Exception as e:
                logger.warning(f"Warmup pass encountered exception: {e}")

        times, ram_deltas, ram_peaks = [], [], []

        for _ in range(num_runs):
            with self.profile_scope(stage_name, audio_duration_seconds) as _:
                func()
            last_rec = self.records[-1]
            times.append(last_rec.elapsed_seconds)
            ram_deltas.append(last_rec.ram_delta_mb)
            ram_peaks.append(last_rec.ram_peak_mb)

        avg_time = float(np.mean(times))
        avg_rtf = round(avg_time / audio_duration_seconds if audio_duration_seconds > 0 else 0.0, 4)
        avg_throughput = round(audio_duration_seconds / avg_time if avg_time > 0 else 0.0, 2)

        return StageBenchmarkResult(
            stage_name=stage_name,
            audio_duration_seconds=audio_duration_seconds,
            elapsed_seconds=round(avg_time, 4),
            rtf=avg_rtf,
            throughput_ratio=avg_throughput,
            ram_initial_mb=self.records[-1].ram_initial_mb,
            ram_peak_mb=round(float(np.mean(ram_peaks)), 2),
            ram_delta_mb=round(float(np.mean(ram_deltas)), 2),
            vram_peak_mb=self.records[-1].vram_peak_mb,
            details={"num_passes": num_runs},
        )


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def plot_stage_latency_breakdown(
    stage_results: Dict[str, float],
    output_png_path: Path,
    title: str = "Stage-by-Stage Latency Breakdown",
) -> None:
    """Plots horizontal stacked bar chart of stage-wise processing time."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(stage_results.keys())
    times = list(stage_results.values())
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"][:len(labels)]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, times, color=colors, edgecolor="black", alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Processing Latency (Seconds)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Add text labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(0.02, width * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}s",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()


def plot_rtf_scaling_curves(
    durations: List[float],
    stage_rtfs: Dict[str, List[float]],
    output_png_path: Path,
) -> None:
    """Plots RTF scaling across input audio durations (15s, 30s, 60s, 120s)."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (stage, rtfs) in enumerate(stage_rtfs.items()):
        ax.plot(
            durations,
            rtfs,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2.2,
            markersize=7,
            label=stage,
        )

    # Parity threshold
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5, label="Realtime Parity (RTF = 1.0)")

    ax.set_xlabel("Input Audio Duration (Seconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Real-Time Factor (RTF)", fontsize=11, fontweight="bold")
    ax.set_title("Computational Scaling & RTF Across Audio Durations", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(durations)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()


def plot_memory_waterfall(
    stages: List[str],
    ram_mb: List[float],
    output_png_path: Path,
) -> None:
    """Plots memory allocation waterfall profile across the pipeline lifecycle."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

    x_pos = np.arange(len(stages))
    bars = ax.bar(x_pos, ram_mb, color="#4a7bb0", edgecolor="black", width=0.5, alpha=0.9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, fontsize=9, fontweight="bold", rotation=15)
    ax.set_ylabel("Peak RAM RSS (MB)", fontsize=11, fontweight="bold")
    ax.set_title("Memory Profile Across Pipeline Lifecycle", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 5,
            f"{int(h)} MB",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()

