#!/usr/bin/env python3
"""Standalone audio health inspection utility for directories of WAV files.

Audits sample rates, bit depths, peak dBFS, DC offset magnitude, and estimated
SNR across directories such as ``assets/`` and ``data/``. Handles corrupted RIFF
headers, truncated payloads, IEEE float WAVs, and mono/stereo channel mismatches
without aborting batch runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import List, Optional, Sequence, Tuple

import numpy as np


_EPS = 1e-12
_MIN_DB = -100.0
_MAX_DB = 120.0
_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_IEEE_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


@dataclass(frozen=True)
class AuditConfig:
    """Typed CLI configuration options."""

    target_dir: Path
    recursive: bool
    frame_ms: float
    fail_on_error: bool


@dataclass(frozen=True)
class AudioHealthReport:
    """Health metrics extracted from a single WAV file."""

    filepath: Path
    sample_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    channels: Optional[int] = None
    peak_dbfs: Optional[float] = None
    dc_offset: Optional[float] = None
    estimated_snr_db: Optional[float] = None
    status: str = "OK"


def parse_args(argv: Optional[Sequence[str]] = None) -> AuditConfig:
    """Parse and validate command-line arguments into a typed config object."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a directory of .wav files and generate a summary table of "
            "sample rates, bit depths, peak dBFS, DC offset, and estimated SNR."
        )
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Directory containing .wav files (e.g., assets/ or data/).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan subdirectories for .wav files.",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=25.0,
        help="Analysis frame duration in milliseconds for SNR estimation (default: 25.0).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Return exit code 1 if any corrupted or unreadable WAV files are detected.",
    )

    args = parser.parse_args(argv)
    if args.frame_ms <= 0.0:
        parser.error("--frame-ms must be positive.")

    return AuditConfig(
        target_dir=args.target_dir,
        recursive=args.recursive,
        frame_ms=args.frame_ms,
        fail_on_error=args.fail_on_error,
    )


def _read_riff_wav(filepath: Path) -> Tuple[int, int, int, int, bytes]:
    """Parse RIFF/WAVE chunks safely for both PCM and IEEE float WAV files."""
    with filepath.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("Invalid RIFF/WAVE header.")

        format_tag: Optional[int] = None
        channels: Optional[int] = None
        sample_rate: Optional[int] = None
        bit_depth: Optional[int] = None
        data_bytes: Optional[bytes] = None

        while True:
            chunk_hdr = handle.read(8)
            if not chunk_hdr:
                break
            if len(chunk_hdr) < 8:
                raise ValueError("Truncated RIFF chunk header.")

            chunk_id, chunk_size = struct.unpack("<4sI", chunk_hdr)
            if chunk_id == b"fmt ":
                fmt_data = handle.read(chunk_size)
                if len(fmt_data) < 16:
                    raise ValueError("Truncated fmt chunk.")
                format_tag, channels, sample_rate, _, _, bit_depth = struct.unpack(
                    "<HHIIHH", fmt_data[:16]
                )
                if format_tag == _WAVE_FORMAT_EXTENSIBLE and len(fmt_data) >= 26:
                    (format_tag,) = struct.unpack("<H", fmt_data[24:26])
            elif chunk_id == b"data":
                data_bytes = handle.read(chunk_size)
            else:
                handle.seek(chunk_size, 1)

            # RIFF chunks are word-aligned to even byte boundaries
            if chunk_size % 2 == 1:
                handle.seek(1, 1)

    if format_tag is None or channels is None or sample_rate is None or bit_depth is None:
        raise ValueError("Missing fmt chunk in WAV header.")
    if data_bytes is None:
        raise ValueError("Missing data chunk in WAV file.")
    if channels <= 0 or sample_rate <= 0 or bit_depth <= 0 or bit_depth % 8 != 0:
        raise ValueError("Invalid channel, sample rate, or bit depth metadata.")

    return format_tag, channels, sample_rate, bit_depth, data_bytes


def _decode_samples(
    raw_bytes: bytes,
    bit_depth: int,
    channels: int,
    format_tag: int,
) -> np.ndarray:
    """Decode raw WAV payload bytes into a finite float64 array of shape (N, C)."""
    sample_width = bit_depth // 8

    if format_tag == _WAVE_FORMAT_IEEE_FLOAT:
        if sample_width == 4:
            if len(raw_bytes) % 4 != 0:
                raise ValueError("Truncated 32-bit float payload.")
            normalized = np.frombuffer(raw_bytes, dtype="<f4").astype(np.float64)
        elif sample_width == 8:
            if len(raw_bytes) % 8 != 0:
                raise ValueError("Truncated 64-bit float payload.")
            normalized = np.frombuffer(raw_bytes, dtype="<f8").astype(np.float64)
        else:
            raise ValueError(f"Unsupported IEEE float bit depth: {bit_depth}-bit.")
    elif format_tag == _WAVE_FORMAT_PCM:
        if sample_width == 1:
            data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64)
            normalized = (data - 128.0) / 128.0
        elif sample_width == 2:
            if len(raw_bytes) % 2 != 0:
                raise ValueError("Truncated 16-bit PCM payload.")
            data = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64)
            normalized = data / 32768.0
        elif sample_width == 3:
            raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
            if raw_u8.size % 3 != 0:
                raise ValueError("Truncated 24-bit PCM payload.")
            triplets = raw_u8.reshape(-1, 3)
            signed = (
                triplets[:, 0].astype(np.int32)
                | (triplets[:, 1].astype(np.int32) << 8)
                | (triplets[:, 2].astype(np.int32) << 16)
            )
            signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
            normalized = signed.astype(np.float64) / 8388608.0
        elif sample_width == 4:
            if len(raw_bytes) % 4 != 0:
                raise ValueError("Truncated 32-bit PCM payload.")
            data = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float64)
            normalized = data / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bit_depth}-bit.")
    else:
        raise ValueError(f"Unsupported WAV format tag: {format_tag}.")

    if normalized.size == 0:
        raise ValueError("Empty audio payload.")
    if normalized.size % channels != 0:
        raise ValueError(
            f"Sample count ({normalized.size}) is not divisible by "
            f"channel count ({channels})."
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Audio payload contains non-finite (NaN/Inf) samples.")

    return normalized.reshape(-1, channels)


def _estimate_snr_db(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float,
) -> float:
    """Estimate SNR (dB) from 95th-percentile active vs. 10th-percentile noise RMS."""
    mono = np.mean(samples, axis=1) if samples.ndim > 1 else samples
    frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))

    if mono.size < frame_len:
        rms = float(np.sqrt(np.mean(np.square(mono))))
        return _MIN_DB if rms <= _EPS else 60.0

    num_frames = mono.size // frame_len
    trimmed = mono[: num_frames * frame_len].reshape(num_frames, frame_len)
    frame_rms = np.sqrt(np.mean(np.square(trimmed), axis=1))

    noise_floor = float(np.percentile(frame_rms, 10))
    signal_level = float(np.percentile(frame_rms, 95))

    if signal_level <= _EPS:
        return _MIN_DB
    snr = 20.0 * np.log10(max(signal_level, _EPS) / max(noise_floor, _EPS))
    return float(np.clip(snr, _MIN_DB, _MAX_DB))


def inspect_wav_file(filepath: Path, frame_ms: float = 25.0) -> AudioHealthReport:
    """Safely inspect a single WAV file and compute audio health metrics.

    Args:
        filepath: Path to the target ``.wav`` file.
        frame_ms: Frame window length in milliseconds for SNR estimation.

    Returns:
        An ``AudioHealthReport`` containing extracted metrics or diagnostic status.
    """
    try:
        if filepath.stat().st_size == 0:
            return AudioHealthReport(filepath=filepath, status="ERR: Empty file")

        format_tag, channels, sample_rate, bit_depth, raw_bytes = _read_riff_wav(filepath)
        samples = _decode_samples(raw_bytes, bit_depth, channels, format_tag)

        peak_linear = float(np.max(np.abs(samples)))
        peak_dbfs = float(
            np.clip(20.0 * np.log10(max(peak_linear, _EPS)), _MIN_DB, _MAX_DB)
        )

        # Evaluate DC offset per channel so mono/stereo discrepancies are preserved
        channel_dc = np.abs(np.mean(samples, axis=0))
        dc_offset = float(np.max(channel_dc))

        snr_db = _estimate_snr_db(samples, sample_rate, frame_ms)

        status = "OK"
        if channels > 1:
            channel_peaks = np.max(np.abs(samples), axis=0)
            if np.any(channel_peaks <= _EPS) and np.any(channel_peaks > _EPS):
                status = "WARN: Silent channel in multi-ch"

        return AudioHealthReport(
            filepath=filepath,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            peak_dbfs=round(peak_dbfs, 2),
            dc_offset=round(dc_offset, 6),
            estimated_snr_db=round(snr_db, 2),
            status=status,
        )
    except (struct.error, EOFError, ValueError, OSError) as exc:
        return AudioHealthReport(
            filepath=filepath,
            status=f"ERR: {exc}",
        )


def collect_wav_files(target_dir: Path, recursive: bool) -> List[Path]:
    """Collect .wav files case-insensitively in sorted order."""
    candidates = target_dir.rglob("*") if recursive else target_dir.glob("*")
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def format_summary_table(
    reports: Sequence[AudioHealthReport],
    base_dir: Path,
) -> str:
    """Format inspection reports into an aligned ASCII table."""
    headers = [
        "File",
        "SR (Hz)",
        "Bit",
        "Ch",
        "Peak (dBFS)",
        "DC Offset",
        "Est. SNR (dB)",
        "Status",
    ]
    rows: List[List[str]] = []
    for report in reports:
        try:
            display_name = str(report.filepath.relative_to(base_dir))
        except ValueError:
            display_name = report.filepath.name
        rows.append(
            [
                display_name,
                str(report.sample_rate) if report.sample_rate is not None else "-",
                f"{report.bit_depth}-bit" if report.bit_depth is not None else "-",
                str(report.channels) if report.channels is not None else "-",
                f"{report.peak_dbfs:.2f}" if report.peak_dbfs is not None else "-",
                f"{report.dc_offset:.6f}" if report.dc_offset is not None else "-",
                (
                    f"{report.estimated_snr_db:.2f}"
                    if report.estimated_snr_db is not None
                    else "-"
                ),
                report.status,
            ]
        )

    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def _render_row(cells: Sequence[str]) -> str:
        aligned = [
            cell.ljust(col_widths[i]) if i in (0, len(cells) - 1) else cell.rjust(col_widths[i])
            for i, cell in enumerate(cells)
        ]
        return " | ".join(aligned)

    divider = "-+-".join("-" * width for width in col_widths)
    lines = [_render_row(headers), divider]
    lines.extend(_render_row(row) for row in rows)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the WAV health audit CLI."""
    config = parse_args(argv)
    if not config.target_dir.exists() or not config.target_dir.is_dir():
        print(f"Error: Target directory does not exist: {config.target_dir}")
        return 1

    wav_files = collect_wav_files(config.target_dir, config.recursive)
    if not wav_files:
        print(f"No .wav files found in '{config.target_dir}'.")
        return 0

    reports = [inspect_wav_file(path, config.frame_ms) for path in wav_files]
    print(format_summary_table(reports, config.target_dir))

    error_count = sum(1 for rep in reports if rep.status.startswith("ERR:"))
    warn_count = sum(1 for rep in reports if rep.status.startswith("WARN:"))
    ok_count = len(reports) - error_count - warn_count

    print(
        f"\nAudited {len(reports)} file(s): "
        f"{ok_count} OK, {warn_count} warning(s), {error_count} error(s)."
    )
    if config.fail_on_error and error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
