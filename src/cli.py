"""
Command-Line Interface (CLI) Orchestrator for Advanced Speech Engineering Pipeline (Milestone M8).

Enables full pipeline invocation, configuration overrides, selective stage execution,
and export artifact management from the terminal.

Usage:
    python -m src --input data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav
    python src/cli.py -i input.wav -o outputs/my_run --mock-mode
    python -m src -i input.wav --bypass-separation --asr-engine whisper --llm-provider mock
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import List, Optional

from src.pipeline import SpeechEngineeringPipeline


def setup_logger(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configures console and module logging."""
    logger = logging.getLogger("speech_cli")
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Also configure speech_pipeline logger
    pipe_logger = logging.getLogger("speech_pipeline")
    pipe_logger.setLevel(level)
    if not pipe_logger.handlers:
        pipe_logger.addHandler(handler)

    return logger


def build_parser() -> argparse.ArgumentParser:
    """Builds and configures the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="speech-pipeline",
        description="Advanced Computational Speech Engineering Pipeline (Milestone M8 Orchestrator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Full pipeline execution on overlapping mixture (Mock/Fast mode):
     python -m src -i data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav --mock-mode

  2. Single-speaker monologue (Bypassing Stage 1 Separation):
     python -m src -i input_mono.wav --bypass-separation -o outputs/monologue_run

  3. Custom Separation and ASR models:
     python -m src -i audio.wav --separation-model demucs --asr-engine indic_wav2vec --llm-provider mock

  4. Force recomputation ignoring previous cached intermediates:
     python -m src -i audio.wav -o outputs/run_01 --no-cache
        """,
    )

    # Core I/O
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to the input audio file (.wav, .mp3, .flac, etc.).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Destination directory for output artifacts, stems, and reports. Defaults to outputs/runs/<session_id>.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file overriding default settings.",
    )
    parser.add_argument(
        "-s", "--session-id",
        type=str,
        default=None,
        help="Custom session identifier for tracking and telemetry.",
    )

    # Hardware & Model Engine Overrides
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for deep learning inference (default: auto).",
    )
    parser.add_argument(
        "--separation-model",
        type=str,
        choices=["conv_tasnet", "demucs"],
        default=None,
        help="Blind Source Separation model architecture for Stage 1.",
    )
    parser.add_argument(
        "--asr-engine",
        type=str,
        choices=["indic_wav2vec", "whisper"],
        default=None,
        help="Speech-to-text acoustic model engine for Stage 3.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["mock", "groq", "openai", "sarvam", "local"],
        default=None,
        help="LLM provider backend for Stage 4 semantic post-processing.",
    )

    # Stage Bypass Flags (Graceful Degradation / Ablations)
    parser.add_argument(
        "--bypass-separation",
        action="store_true",
        help="Bypass Stage 1 Blind Source Separation (ideal for single-speaker audio).",
    )
    parser.add_argument(
        "--bypass-diarization",
        action="store_true",
        help="Bypass Stage 2 Speaker Diarization (assumes single speaker).",
    )
    parser.add_argument(
        "--bypass-asr",
        action="store_true",
        help="Bypass Stage 3 Automatic Speech Recognition.",
    )
    parser.add_argument(
        "--bypass-llm",
        action="store_true",
        help="Bypass Stage 4 Semantic Post-Processing & Guardrails.",
    )

    # Performance, Caching & Mock Mode
    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Run pipeline using lightweight mock inference (rapid CI verification).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation of all stages, ignoring intermediate cached files.",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Do not serialize Contract 1-5 JSON artifacts or Markdown report to disk.",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress informational stdout output except error messages.",
    )

    return parser


def format_summary_banner(state, output_dir: Path) -> str:
    """Formats an executive summary banner from PipelineState."""
    meta = state.audio_metadata
    dur = meta.duration_seconds if meta else 0.0
    telemetry = state.execution_telemetry
    tot_time = telemetry.get("total_pipeline_time_seconds", 0.0)
    rtf = telemetry.get("cumulative_rtf", 0.0)
    speed = telemetry.get("throughput_ratio", 0.0)

    # Summary counts
    num_sources = (
        state.separation_result.num_sources_detected
        if state.separation_result
        else ("Bypassed" if telemetry.get("stage1_separation", {}).get("bypassed") else 1)
    )
    num_speakers = (
        state.diarization_timeline.total_speakers_detected
        if state.diarization_timeline
        else ("Bypassed" if telemetry.get("stage2_diarization", {}).get("bypassed") else 1)
    )
    num_utterances = (
        len(state.raw_transcript.utterances)
        if state.raw_transcript
        else ("Bypassed" if telemetry.get("stage3_asr", {}).get("bypassed") else 0)
    )
    report_findings = (
        len(state.final_report.key_findings)
        if state.final_report
        else ("Bypassed" if telemetry.get("stage4_llm", {}).get("bypassed") else 0)
    )

    banner = [
        "",
        "=" * 76,
        "   ADVANCED COMPUTATIONAL SPEECH ENGINEERING PIPELINE (M8 ORCHESTRATOR)",
        "=" * 76,
        f" Session ID          : {state.session_id}",
        f" Input Audio         : {state.input_audio_path}",
        f" Audio Duration      : {dur:.2f} seconds ({dur/60.0:.2f} minutes)",
        f" Processing Time     : {tot_time:.2f}s  |  RTF: {rtf:.4f}  |  Speed: {speed}x Realtime",
        "-" * 76,
        " STAGE BREAKDOWN:",
        f"   - Stage 0 Preprocessing : {telemetry.get('stage0_preprocessing', {}).get('elapsed_seconds', 0.0):.3f}s [OK]",
        f"   - Stage 1 Separation    : {telemetry.get('stage1_separation', {}).get('elapsed_seconds', 0.0):.3f}s (Stems: {num_sources})",
        f"   - Stage 2 Diarization   : {telemetry.get('stage2_diarization', {}).get('elapsed_seconds', 0.0):.3f}s (Speakers: {num_speakers})",
        f"   - Stage 3 ASR Engine    : {telemetry.get('stage3_asr', {}).get('elapsed_seconds', 0.0):.3f}s (Utterances: {num_utterances})",
        f"   - Stage 4 LLM Guardrail : {telemetry.get('stage4_llm', {}).get('elapsed_seconds', 0.0):.3f}s (Key Findings: {report_findings})",
        "-" * 76,
        " GENERATED ARTIFACTS:",
        f"   * Output Directory  : {output_dir.resolve()}",
        f"   * Metadata (C1)     : {output_dir / 'metadata.json'}",
        f"   * Separation (C2)   : {output_dir / 'separation_manifest.json' if state.separation_result else '[Bypassed]'}",
        f"   * Diarization (C3)  : {output_dir / 'diarization_timeline.json' if state.diarization_timeline else '[Bypassed]'}",
        f"   * Transcript (C4)   : {output_dir / 'raw_transcript.json' if state.raw_transcript else '[Bypassed]'}",
        f"   * Final Report (C5) : {output_dir / 'final_report.json' if state.final_report else '[Bypassed]'}",
        f"   * Markdown Report   : {output_dir / 'final_report.md' if state.final_report else '[Bypassed]'}",
        "=" * 76,
        "",
    ]
    return "\n".join(banner)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI orchestrator main execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logger(verbose=args.verbose, quiet=args.quiet)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Build config overrides from CLI
    config_overrides = {}
    if args.separation_model:
        config_overrides.setdefault("separation", {})["model_name"] = args.separation_model
    if args.asr_engine:
        config_overrides.setdefault("asr", {})["engine_type"] = args.asr_engine
    if args.llm_provider:
        config_overrides.setdefault("llm", {})["provider"] = args.llm_provider
    if args.mock_mode:
        config_overrides.setdefault("asr", {})["mock_mode"] = True
        config_overrides.setdefault("llm", {})["provider"] = "mock"

    try:
        if not args.quiet:
            logger.info("Initializing Speech Engineering Pipeline...")

        pipeline = SpeechEngineeringPipeline(
            config_path=args.config,
            config_dict=config_overrides if config_overrides else None,
            device=args.device,
            enable_caching=not args.no_cache,
            mock_mode=args.mock_mode,
        )

        state = pipeline.run(
            input_audio_path=input_path,
            output_dir=args.output_dir,
            session_id=args.session_id,
            bypass_separation=args.bypass_separation,
            bypass_diarization=args.bypass_diarization,
            bypass_asr=args.bypass_asr,
            bypass_llm=args.bypass_llm,
            force_recompute=args.no_cache,
            save_artifacts=not args.no_save_artifacts,
        )

        out_dir = Path(state.execution_telemetry.get("output_dir", args.output_dir or "outputs/runs"))
        if not args.quiet:
            print(format_summary_banner(state, out_dir))

        return 0

    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
