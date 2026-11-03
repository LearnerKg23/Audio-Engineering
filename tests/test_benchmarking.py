"""
Unit & Integration Tests for Milestone M10:
System Benchmarking & Computational Profiling Suite.
"""

import json
from pathlib import Path
import tempfile
import time
import pytest

from scripts.run_system_benchmark import prepare_benchmark_audio_fixtures
from src.benchmarking.profiler import (
    StageBenchmarkResult,
    SystemProfiler,
    get_hardware_specs,
    plot_memory_waterfall,
    plot_rtf_scaling_curves,
    plot_stage_latency_breakdown,
)
from src.core.audio_io import compute_audio_metrics, load_audio


def test_hardware_specs_detection():
    """Verifies system hardware and accelerator detection."""
    specs = get_hardware_specs()
    assert "os_platform" in specs
    assert "cpu_architecture" in specs
    assert "cpu_count_logical" in specs
    assert "total_ram_gb" in specs
    assert specs["total_ram_gb"] > 0.0
    assert "pytorch_version" in specs


def test_profiler_scope_context_manager():
    """Verifies that SystemProfiler accurately profiles elapsed time, memory, and RTF."""
    profiler = SystemProfiler(warmup=False)

    with profiler.profile_scope("Unit_Test_Stage", audio_duration_seconds=10.0):
        time.sleep(0.05)
        # Allocate small array to test memory tracking
        _ = [i ** 2 for i in range(100000)]

    assert len(profiler.records) == 1
    rec = profiler.records[0]
    assert rec.stage_name == "Unit_Test_Stage"
    assert rec.audio_duration_seconds == 10.0
    assert rec.elapsed_seconds >= 0.04
    assert rec.rtf > 0.0
    assert rec.throughput_ratio > 0.0
    assert rec.ram_peak_mb > 0.0


def test_stage_benchmark_result_serialization():
    """Verifies StageBenchmarkResult serializes cleanly to dictionary and JSON."""
    result = StageBenchmarkResult(
        stage_name="BSS_Stage",
        audio_duration_seconds=30.0,
        elapsed_seconds=1.5,
        rtf=0.05,
        throughput_ratio=20.0,
        ram_initial_mb=250.0,
        ram_peak_mb=320.0,
        ram_delta_mb=70.0,
        vram_peak_mb=0.0,
        details={"model": "conv_tasnet"},
    )
    d = result.to_dict()
    assert d["stage_name"] == "BSS_Stage"
    assert d["rtf"] == 0.05
    json_str = json.dumps(d)
    assert "BSS_Stage" in json_str


def test_prepare_benchmark_audio_fixtures():
    """Verifies preparation of standardized duration audio clips."""
    base_dir = Path(__file__).resolve().parent.parent
    fixtures = prepare_benchmark_audio_fixtures(base_dir)

    assert 15 in fixtures
    assert 30 in fixtures
    assert 60 in fixtures
    assert 120 in fixtures

    for dur, p in fixtures.items():
        assert p.exists()
        waveform, sr = load_audio(p)
        meta = compute_audio_metrics(waveform, sr)
        assert abs(meta["duration_seconds"] - dur) < 0.2
        assert sr == 16000


def test_plot_generation_integrity():
    """Verifies that benchmarking visualization plotting functions produce valid image files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        # 1. Latency breakdown plot
        lat_png = tmp_p / "lat.png"
        plot_stage_latency_breakdown({"Stage 0": 0.05, "Stage 1": 1.2, "Stage 2": 0.3}, lat_png)
        assert lat_png.exists()
        assert lat_png.stat().st_size > 1000

        # 2. RTF scaling curves
        rtf_png = tmp_p / "rtf.png"
        plot_rtf_scaling_curves([15, 30, 60], {"Separation": [0.08, 0.07, 0.07]}, rtf_png)
        assert rtf_png.exists()
        assert rtf_png.stat().st_size > 1000

        # 3. Memory waterfall plot
        mem_png = tmp_p / "mem.png"
        plot_memory_waterfall(["Stage 1", "Stage 2"], [300.0, 350.0], mem_png)
        assert mem_png.exists()
        assert mem_png.stat().st_size > 1000


def test_system_benchmark_notebook_structure():
    """Verifies that notebooks/01_system_benchmark.ipynb exists, is valid JSON, and adheres to nbformat 4."""
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "01_system_benchmark.ipynb"

    assert nb_path.exists(), f"Missing benchmark notebook: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert nb_data.get("nbformat") == 4
    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 6

    # Verify that cells cover the required sections
    all_code_text = " ".join(
        "".join(cell.get("source", []))
        for cell in nb_data["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "get_hardware_specs" in all_code_text
    assert "SystemProfiler" in all_code_text
    assert "run_isolated_stage_benchmarks" in all_code_text
    assert "run_duration_scaling_benchmark" in all_code_text
    assert "benchmark_summary.json" in all_code_text
