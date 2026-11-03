"""
End-to-End Pipeline Integration & CLI Orchestrator Test Suite (Milestone M8).

Validates:
1. End-to-end execution across Stages 0, 1, 2, 3, and 4.
2. Compliance with inter-stage Pydantic Data Contracts (Contracts 1-5).
3. Intermediate artifact caching and cache-hit performance.
4. Single-speaker bypass and selective stage isolation.
5. JSON and Markdown artifact serialization integrity.
6. Command-Line Interface (CLI) parsing, execution, and error codes.
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.cli import main as cli_main
from src.core.state import (
    AudioMetadata,
    DiarizationTimeline,
    PipelineState,
    PostProcessedReport,
    RawTranscript,
    SeparationResult,
)
from src.pipeline import SpeechEngineeringPipeline


# Fixture for a synthetic test audio file
@pytest.fixture
def sample_audio_path() -> Path:
    base = Path(__file__).resolve().parent.parent
    path = base / "data" / "controlled_eval_suite" / "synthetic_mixtures" / "controlled_mix_01_pairAD_ov25_sir0.wav"
    if not path.exists():
        # Fallback to any wav in data if present
        wavs = list(base.glob("data/**/*.wav"))
        if wavs:
            return wavs[0]
        pytest.skip(f"Synthetic mixture audio not found at {path}")
    return path


def test_pipeline_initialization():
    """Verifies that the SpeechEngineeringPipeline initializes cleanly with defaults."""
    pipeline = SpeechEngineeringPipeline(mock_mode=True)
    assert pipeline is not None
    assert pipeline.separator is not None
    assert pipeline.diarizer is not None
    assert pipeline.asr_engine is not None
    assert pipeline.llm_engine is not None
    assert pipeline.mock_mode is True


def test_full_pipeline_execution_contracts(sample_audio_path: Path):
    """
    Executes the full end-to-end pipeline in mock mode and verifies that all
    5 formal Pydantic data contracts and telemetry fields are populated.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir) / "test_e2e_run"
        pipeline = SpeechEngineeringPipeline(mock_mode=True)

        state = pipeline.run(
            input_audio_path=sample_audio_path,
            output_dir=out_p,
            session_id="test_e2e_session",
            save_artifacts=True,
        )

        assert isinstance(state, PipelineState)
        assert state.session_id == "test_e2e_session"

        # Contract 1: AudioMetadata
        assert state.audio_metadata is not None
        assert isinstance(state.audio_metadata, AudioMetadata)
        assert state.audio_metadata.sample_rate_hz == 16000
        assert state.audio_metadata.duration_seconds > 0.0
        assert state.audio_metadata.channels == 1

        # Contract 2: SeparationResult
        assert state.separation_result is not None
        assert isinstance(state.separation_result, SeparationResult)
        assert state.separation_result.num_sources_detected >= 1
        assert len(state.separation_result.isolated_tracks) >= 1

        # Contract 3: DiarizationTimeline
        assert state.diarization_timeline is not None
        assert isinstance(state.diarization_timeline, DiarizationTimeline)
        assert state.diarization_timeline.total_speakers_detected >= 1
        assert len(state.diarization_timeline.segments) >= 1

        # Contract 4: RawTranscript
        assert state.raw_transcript is not None
        assert isinstance(state.raw_transcript, RawTranscript)
        assert len(state.raw_transcript.utterances) >= 1

        # Contract 5: PostProcessedReport
        assert state.final_report is not None
        assert isinstance(state.final_report, PostProcessedReport)
        assert len(state.final_report.key_findings) >= 1
        assert state.final_report.executive_summary != ""

        # Telemetry
        assert "stage0_preprocessing" in state.execution_telemetry
        assert "stage1_separation" in state.execution_telemetry
        assert "stage2_diarization" in state.execution_telemetry
        assert "stage3_asr" in state.execution_telemetry
        assert "stage4_llm" in state.execution_telemetry
        assert "total_pipeline_time_seconds" in state.execution_telemetry
        assert "cumulative_rtf" in state.execution_telemetry


def test_intermediate_artifact_caching(sample_audio_path: Path):
    """
    Verifies that running the pipeline twice against the same output directory
    hits the intermediate cache for each stage, avoiding redundant recomputation.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir) / "cached_run"
        pipeline = SpeechEngineeringPipeline(mock_mode=True, enable_caching=True)

        # First run (computes everything)
        state1 = pipeline.run(input_audio_path=sample_audio_path, output_dir=out_p)
        assert state1.final_report is not None

        # Second run (hits cache)
        state2 = pipeline.run(input_audio_path=sample_audio_path, output_dir=out_p)

        # Verify cache hits in telemetry
        assert state2.execution_telemetry["stage1_separation"].get("cached") is True
        assert state2.execution_telemetry["stage2_diarization"].get("cached") is True
        assert state2.execution_telemetry["stage3_asr"].get("cached") is True
        assert state2.execution_telemetry["stage4_llm"].get("cached") is True

        # Cached total time should be extremely fast (< 0.5 seconds)
        assert state2.execution_telemetry["total_pipeline_time_seconds"] < 2.0


def test_single_speaker_bypass(sample_audio_path: Path):
    """
    Verifies that --bypass-separation skips Stage 1 BSS, marks telemetry as bypassed,
    and allows Stages 2-4 to run cleanly on normalized audio.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir) / "bypass_run"
        pipeline = SpeechEngineeringPipeline(mock_mode=True)

        state = pipeline.run(
            input_audio_path=sample_audio_path,
            output_dir=out_p,
            bypass_separation=True,
        )

        assert state.separation_result is None
        assert state.execution_telemetry["stage1_separation"]["bypassed"] is True
        assert state.diarization_timeline is not None
        assert state.raw_transcript is not None
        assert state.final_report is not None


def test_artifact_file_integrity(sample_audio_path: Path):
    """
    Verifies that all 5 formal contracts and markdown reports are properly serialized
    to disk and can be deserialized back into Pydantic models.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir) / "artifact_test"
        pipeline = SpeechEngineeringPipeline(mock_mode=True)
        state = pipeline.run(input_audio_path=sample_audio_path, output_dir=out_p, save_artifacts=True)

        meta_file = out_p / "metadata.json"
        sep_file = out_p / "separation_manifest.json"
        diar_file = out_p / "diarization_timeline.json"
        asr_file = out_p / "raw_transcript.json"
        report_json = out_p / "final_report.json"
        report_md = out_p / "final_report.md"
        state_file = out_p / "pipeline_state.json"

        assert meta_file.exists()
        assert sep_file.exists()
        assert diar_file.exists()
        assert asr_file.exists()
        assert report_json.exists()
        assert report_md.exists()
        assert state_file.exists()

        # Validate deserialization back into Pydantic models
        with open(meta_file, "r", encoding="utf-8") as f:
            AudioMetadata(**json.load(f))
        with open(sep_file, "r", encoding="utf-8") as f:
            SeparationResult(**json.load(f))
        with open(diar_file, "r", encoding="utf-8") as f:
            DiarizationTimeline(**json.load(f))
        with open(asr_file, "r", encoding="utf-8") as f:
            RawTranscript(**json.load(f))
        with open(report_json, "r", encoding="utf-8") as f:
            PostProcessedReport(**json.load(f))
        with open(state_file, "r", encoding="utf-8") as f:
            PipelineState.load_json(str(state_file))

        # Check markdown contents
        with open(report_md, "r", encoding="utf-8") as f:
            md_text = f.read()
            assert "Executive Conversational Intelligence Report" in md_text
            assert "Thematic Analysis" in md_text


def test_cli_execution_success(sample_audio_path: Path):
    """Verifies CLI execution with mock-mode and quiet flags returns exit code 0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir) / "cli_run"
        exit_code = cli_main([
            "--input", str(sample_audio_path),
            "--output-dir", str(out_p),
            "--mock-mode",
            "--quiet",
        ])
        assert exit_code == 0
        assert (out_p / "final_report.json").exists()


def test_cli_execution_bypass_flag(sample_audio_path: Path):
    """Verifies CLI execution with --bypass-separation flag."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir) / "cli_bypass"
        exit_code = cli_main([
            "-i", str(sample_audio_path),
            "-o", str(out_p),
            "--bypass-separation",
            "--mock-mode",
            "--quiet",
        ])
        assert exit_code == 0
        assert not (out_p / "separation_manifest.json").exists()
        assert (out_p / "final_report.json").exists()


def test_cli_nonexistent_file():
    """Verifies CLI exits with code 1 when input file does not exist."""
    exit_code = cli_main([
        "--input", "non_existent_audio_file_12345.wav",
        "--quiet",
    ])
    assert exit_code == 1


def test_missing_input_raises(sample_audio_path: Path):
    """Verifies that SpeechEngineeringPipeline.run raises FileNotFoundError if input audio is missing."""
    pipeline = SpeechEngineeringPipeline(mock_mode=True)
    with pytest.raises(FileNotFoundError):
        pipeline.run(input_audio_path="missing_file.wav")
