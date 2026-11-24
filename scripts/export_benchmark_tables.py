#!/usr/bin/env python3
"""Export benchmark evaluation JSONs into publication-ready Markdown and CSV tables.

Reads ``outputs/benchmarks/benchmark_summary.json`` alongside optional related
evaluation JSONs in the benchmark directory and renders Table 1 in both
Markdown and CSV formats with columns:
``System``, ``SI-SDR``, ``DER``, ``WER``, ``CER``, ``Hallucination Rate``, ``RTF``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TABLE_1_HEADERS: List[str] = [
    "System",
    "SI-SDR",
    "DER",
    "WER",
    "CER",
    "Hallucination Rate",
    "RTF",
]

_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "si_sdr": (
        "SI-SDR",
        "si_sdr",
        "sisdr",
        "si_sdr_db",
        "sisdr_db",
        "mean_si_sdr",
    ),
    "der": (
        "DER",
        "der",
        "der_percent",
        "diarization_error_rate",
        "mean_der",
    ),
    "wer": (
        "WER",
        "wer",
        "wer_percent",
        "macro_wer_percent",
        "micro_wer_percent",
        "word_error_rate",
    ),
    "cer": (
        "CER",
        "cer",
        "cer_percent",
        "macro_cer_percent",
        "micro_cer_percent",
        "character_error_rate",
    ),
    "hallucination_rate": (
        "Hallucination Rate",
        "hallucination_rate",
        "hallucination_rate_percent",
        "hallucination",
        "hallucination_pct",
    ),
    "rtf": (
        "RTF",
        "rtf",
        "real_time_factor",
        "mean_rtf",
    ),
}


@dataclass
class ExportConfig:
    """CLI configuration for exporting benchmark tables."""

    summary_path: Path = Path("outputs/benchmarks/benchmark_summary.json")
    metrics_dir: Optional[Path] = None
    output_dir: Path = Path("outputs/benchmarks")
    markdown_name: str = "table1_benchmark.md"
    csv_name: str = "table1_benchmark.csv"


@dataclass
class BenchmarkRow:
    """Normalized metrics for a single system or pipeline configuration."""

    system: str
    metrics: Dict[str, Optional[float]] = field(
        default_factory=lambda: {
            "si_sdr": None,
            "der": None,
            "wer": None,
            "cer": None,
            "hallucination_rate": None,
            "rtf": None,
        }
    )

    def merge_from(self, payload: Dict[str, Any], overwrite: bool = False) -> None:
        """Populate missing (or all, if overwrite=True) metrics from a dict."""
        extracted = _extract_metrics_dict(payload)
        for key, value in extracted.items():
            if value is None:
                continue
            if overwrite or self.metrics.get(key) is None:
                self.metrics[key] = value


def _to_float(value: Any) -> Optional[float]:
    """Convert numeric values safely to float, ignoring booleans and non-numeric strings."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        if not cleaned or cleaned in {"-", "N/A", "n/a", "null", "None"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_metrics_dict(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract normalized metric floats from a flat or nested dictionary."""
    result: Dict[str, Optional[float]] = {
        "si_sdr": None,
        "der": None,
        "wer": None,
        "cer": None,
        "hallucination_rate": None,
        "rtf": None,
    }

    # Check nested sub-dictionaries first as fallback containers
    nested_containers: List[Dict[str, Any]] = []
    for sub_key in ("metrics", "results", "summary", "asr", "diarization", "separation", "latency"):
        sub_val = data.get(sub_key)
        if isinstance(sub_val, dict):
            nested_containers.append(sub_val)

    for metric_key, aliases in _KEY_ALIASES.items():
        for alias in aliases:
            if alias in data:
                parsed = _to_float(data[alias])
                if parsed is not None:
                    result[metric_key] = parsed
                    break
        if result[metric_key] is None:
            for container in nested_containers:
                for alias in aliases:
                    if alias in container:
                        parsed = _to_float(container[alias])
                        if parsed is not None:
                            result[metric_key] = parsed
                            break
                if result[metric_key] is not None:
                    break

    return result


def _load_json_safe(path: Path) -> Optional[Any]:
    """Load JSON from disk, returning None if file is missing or malformed."""
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_system_name(entry: Dict[str, Any], fallback: str) -> str:
    """Resolve a human-readable system identifier from a JSON record."""
    for field_name in ("system", "system_name", "model", "model_name", "pipeline", "name", "id"):
        val = entry.get(field_name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return fallback


def _entries_from_payload(payload: Any, source_stem: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalize diverse JSON shapes into ``(system_name, record_dict)`` pairs."""
    pairs: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                name = _extract_system_name(item, f"{source_stem}_{idx + 1}")
                pairs.append((name, item))
        return pairs

    if not isinstance(payload, dict):
        return pairs

    # Case 1: Container holds a list under "systems", "models", "entries", or "benchmarks"
    for list_key in ("systems", "models", "entries", "benchmarks", "runs"):
        container = payload.get(list_key)
        if isinstance(container, list):
            for idx, item in enumerate(container):
                if isinstance(item, dict):
                    name = _extract_system_name(item, f"{source_stem}_{idx + 1}")
                    pairs.append((name, item))
            if pairs:
                return pairs
        elif isinstance(container, dict):
            for sys_name, item in container.items():
                if isinstance(item, dict):
                    resolved = _extract_system_name(item, str(sys_name))
                    pairs.append((resolved, item))
            if pairs:
                return pairs

    # Case 2: Top-level dict mapping system names -> metric dicts
    dict_values = {
        k: v for k, v in payload.items() if isinstance(v, dict)
    }
    if dict_values and not any(k in payload for k in ("si_sdr", "der", "wer", "cer", "rtf")):
        for sys_name, item in dict_values.items():
            resolved = _extract_system_name(item, str(sys_name))
            pairs.append((resolved, item))
        return pairs

    # Case 3: Single flat or nested metrics dictionary
    name = _extract_system_name(payload, source_stem)
    pairs.append((name, payload))
    return pairs


def load_benchmark_rows(
    summary_path: Path,
    metrics_dir: Optional[Path] = None,
) -> List[BenchmarkRow]:
    """Load and merge metrics from ``benchmark_summary.json`` and related JSON files."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Benchmark summary JSON not found: {summary_path}")

    summary_data = _load_json_safe(summary_path)
    rows_by_system: Dict[str, BenchmarkRow] = {}

    def _ingest_pairs(pairs: Iterable[Tuple[str, Dict[str, Any]]], overwrite: bool) -> None:
        for system_name, record in pairs:
            if system_name not in rows_by_system:
                rows_by_system[system_name] = BenchmarkRow(system=system_name)
            rows_by_system[system_name].merge_from(record, overwrite=overwrite)

    _ingest_pairs(_entries_from_payload(summary_data, summary_path.stem), overwrite=True)

    search_dir = metrics_dir if metrics_dir is not None else summary_path.parent
    if search_dir.exists() and search_dir.is_dir():
        for json_file in sorted(search_dir.glob("*.json")):
            if json_file.resolve() == summary_path.resolve():
                continue
            extra_data = _load_json_safe(json_file)
            if extra_data is not None:
                _ingest_pairs(_entries_from_payload(extra_data, json_file.stem), overwrite=False)

    return list(rows_by_system.values())


def _format_cell(value: Optional[float], decimals: int = 2) -> str:
    """Format a float metric for publication tables or fallback to '-'."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def rows_to_table_records(rows: Sequence[BenchmarkRow]) -> List[Dict[str, str]]:
    """Convert normalized ``BenchmarkRow`` items into string rows keyed by Table 1 headers."""
    records: List[Dict[str, str]] = []
    for row in rows:
        records.append(
            {
                "System": row.system,
                "SI-SDR": _format_cell(row.metrics.get("si_sdr"), decimals=2),
                "DER": _format_cell(row.metrics.get("der"), decimals=2),
                "WER": _format_cell(row.metrics.get("wer"), decimals=2),
                "CER": _format_cell(row.metrics.get("cer"), decimals=2),
                "Hallucination Rate": _format_cell(
                    row.metrics.get("hallucination_rate"), decimals=2
                ),
                "RTF": _format_cell(row.metrics.get("rtf"), decimals=3),
            }
        )
    return records


def render_markdown_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render Table 1 as a publication-ready GitHub-flavored Markdown table."""
    records = rows_to_table_records(rows)
    header_line = "| " + " | ".join(TABLE_1_HEADERS) + " |"
    separator_line = "| " + " | ".join("---" for _ in TABLE_1_HEADERS) + " |"
    body_lines = [
        "| " + " | ".join(rec[col] for col in TABLE_1_HEADERS) + " |"
        for rec in records
    ]
    return "\n".join(["### Table 1: Benchmark Evaluation Summary", "", header_line, separator_line, *body_lines]) + "\n"


def write_csv_table(rows: Sequence[BenchmarkRow], csv_path: Path) -> None:
    """Write Table 1 to a CSV file using exact Table 1 column headers."""
    records = rows_to_table_records(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_1_HEADERS)
        writer.writeheader()
        writer.writerows(records)


def export_benchmark_tables(config: ExportConfig) -> Tuple[Path, Path]:
    """Generate both Markdown and CSV Table 1 exports and return their file paths."""
    rows = load_benchmark_rows(
        summary_path=config.summary_path,
        metrics_dir=config.metrics_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    md_path = config.output_dir / config.markdown_name
    csv_path = config.output_dir / config.csv_name

    md_content = render_markdown_table(rows)
    md_path.write_text(md_content, encoding="utf-8")
    write_csv_table(rows, csv_path)

    return md_path, csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> ExportConfig:
    """Parse command-line arguments for the benchmark table exporter."""
    parser = argparse.ArgumentParser(
        description="Export benchmark evaluation JSONs into Markdown and CSV Table 1."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/benchmarks/benchmark_summary.json"),
        help="Path to benchmark_summary.json.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Directory containing supplementary evaluation JSONs (defaults to summary parent).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks"),
        help="Directory where Markdown and CSV tables will be written.",
    )
    parser.add_argument(
        "--md-name",
        type=str,
        default="table1_benchmark.md",
        help="Output Markdown filename.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="table1_benchmark.csv",
        help="Output CSV filename.",
    )

    args = parser.parse_args(argv)
    return ExportConfig(
        summary_path=args.summary,
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        markdown_name=args.md_name,
        csv_name=args.csv_name,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for exporting benchmark tables."""
    config = parse_args(argv)
    try:
        md_path, csv_path = export_benchmark_tables(config)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        print(f"Error exporting benchmark tables: {exc}")
        return 1

    print(f"Exported Markdown table to: {md_path}")
    print(f"Exported CSV table to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
