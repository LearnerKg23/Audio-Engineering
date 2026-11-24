"""Unit tests for scripts/export_benchmark_tables.py."""

import csv
import json
from pathlib import Path

from scripts.export_benchmark_tables import (
    TABLE_1_HEADERS,
    ExportConfig,
    export_benchmark_tables,
    main,
)


def test_export_benchmark_tables_clean_execution(tmp_path: Path):
    """Verifies clean Markdown and CSV export merging summary and related JSONs."""
    bench_dir = tmp_path / "benchmarks"
    out_dir = tmp_path / "exported"
    bench_dir.mkdir(parents=True)

    summary_payload = {
        "systems": [
            {
                "system": "Stage1-3 Pipeline (Indic Whisper + Diarization)",
                "si_sdr": 11.456,
                "der": 4.82,
                "wer": 14.25,
                "cer": 6.10,
                "hallucination_rate": 1.35,
                "rtf": 0.425,
            },
            {
                "system": "Baseline ASR-Only",
                "si_sdr": 7.90,
                "der": 12.40,
            },
        ]
    }
    summary_path = bench_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    # Supplementary evaluation JSON filling missing metrics for Baseline ASR-Only
    extra_asr_payload = {
        "Baseline ASR-Only": {
            "wer": 19.80,
            "cer": 9.45,
            "hallucination_rate": 3.20,
            "rtf": 0.210,
        }
    }
    (bench_dir / "asr_eval.json").write_text(
        json.dumps(extra_asr_payload), encoding="utf-8"
    )

    config = ExportConfig(
        summary_path=summary_path,
        metrics_dir=bench_dir,
        output_dir=out_dir,
    )
    md_path, csv_path = export_benchmark_tables(config)

    assert md_path.exists()
    assert csv_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    for header in TABLE_1_HEADERS:
        assert header in md_text
    assert "11.46" in md_text
    assert "0.425" in md_text
    assert "19.80" in md_text

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == TABLE_1_HEADERS
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["System"] == "Stage1-3 Pipeline (Indic Whisper + Diarization)"
    assert rows[0]["SI-SDR"] == "11.46"
    assert rows[0]["DER"] == "4.82"
    assert rows[0]["WER"] == "14.25"
    assert rows[0]["CER"] == "6.10"
    assert rows[0]["Hallucination Rate"] == "1.35"
    assert rows[0]["RTF"] == "0.425"

    assert rows[1]["System"] == "Baseline ASR-Only"
    assert rows[1]["WER"] == "19.80"
    assert rows[1]["Hallucination Rate"] == "3.20"
    assert rows[1]["RTF"] == "0.210"


def test_export_cli_main_returns_zero(tmp_path: Path):
    """Verifies CLI entry point runs cleanly and exits with status code 0."""
    summary_path = tmp_path / "benchmark_summary.json"
    out_dir = tmp_path / "out"
    summary_path.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "system": "TestModel",
                        "si_sdr": 9.5,
                        "der": 5.0,
                        "wer": 12.0,
                        "cer": 4.5,
                        "hallucination_rate": 0.8,
                        "rtf": 0.35,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--summary",
            str(summary_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "table1_benchmark.md").exists()
    assert (out_dir / "table1_benchmark.csv").exists()
