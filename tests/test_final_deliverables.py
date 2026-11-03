"""
Unit & Integration Tests for Milestone M12:
Final Deliverables, Academic Report & Technical Interview Portfolio.
"""

import json
from pathlib import Path
import pytest

from src.core.state import (
    AudioMetadata,
    DiarizationSegment,
    DiarizationTimeline,
    KeyFinding,
    PipelineState,
    PostProcessedReport,
    RawTranscript,
    SeparatedTrack,
    SeparationResult,
    SpeakerDialogue,
    Utterance,
)


def test_readme_completeness():
    """Validates that README.md exists and contains all required public documentation sections."""
    base_dir = Path(__file__).resolve().parent.parent
    readme_path = base_dir / "README.md"
    assert readme_path.exists(), "README.md is missing from repository root"

    content = readme_path.read_text(encoding="utf-8")
    assert "# Advanced Computational Speech Engineering" in content
    assert "## 1. Project Overview & Architectural Vision" in content
    assert "## 2. Key Empirical Findings & Project Highlights" in content
    assert "## 3. Repository Directory Layout" in content
    assert "## 4. Getting Started & Installation" in content
    assert "## 5. Usage & Quickstart" in content
    assert "## 6. Comprehensive Milestones Summary" in content
    assert "## 7. Interactive Showcase & Web Audio Demonstrator" in content
    assert "## 8. License & Academic Attribution" in content


def test_final_academic_report_completeness():
    """Validates that the master academic report exists and contains all 10 mandated sections."""
    base_dir = Path(__file__).resolve().parent.parent
    report_path = base_dir / "docs" / "FINAL_ACADEMIC_REPORT.md"
    assert report_path.exists(), "docs/FINAL_ACADEMIC_REPORT.md is missing"

    content = report_path.read_text(encoding="utf-8")
    assert "## Abstract" in content
    assert "## 1. Introduction & Problem Formulation" in content
    assert "## 2. Multi-Stage Architectural Design & Mathematical Formulations" in content
    assert "## 3. Data Strategy & Corpora Analysis" in content
    assert "## 4. Multi-Stage Empirical Results Summary" in content
    assert "## 5. Cross-Stage Error Propagation & Progressive Ablation Study" in content
    assert "## 6. System Benchmarking & Computational Scalability" in content
    assert "## 7. Real-World In-The-Wild Validation (Track B)" in content
    assert "## 8. Failure Case Taxonomy & Root-Cause Mitigation" in content
    assert "## 9. Conclusion, Limitations & Future Research" in content
    assert "## References" in content

    # Check key scientific findings
    assert "+4.2" in content  # Critical Breakdown Threshold
    assert "SI-SDR" in content
    assert "Eigengap" in content


def test_technical_interview_portfolio_completeness():
    """Validates that the technical interview defense portfolio exists and covers all core ML questions."""
    base_dir = Path(__file__).resolve().parent.parent
    portfolio_path = base_dir / "docs" / "TECHNICAL_INTERVIEW_PORTFOLIO.md"
    assert portfolio_path.exists(), "docs/TECHNICAL_INTERVIEW_PORTFOLIO.md is missing"

    content = portfolio_path.read_text(encoding="utf-8")
    assert "## 1. Executive Resume Bullets (STAR Format)" in content
    assert "## 2. System Design & Architectural Tradeoffs" in content
    assert "## 3. Deep-Dive Mathematical Formulations" in content
    assert "## 4. Cross-Stage Error Propagation Theory" in content
    assert "## 5. Production Systems & Latency Engineering" in content
    assert "## 6. Hallucination Elimination & Guardrails" in content
    assert "## 7. Real-World Audio Challenges (Field Conditions)" in content

    # Check key technical answers
    assert "Scale-Invariant SDR" in content
    assert "Eigengap" in content
    assert "Phantom Voice Channel" in content
    assert "Real-Time Factor" in content


def test_interactive_showcase_artifacts():
    """Validates that interactive web showcase and benchmark notebook exist."""
    base_dir = Path(__file__).resolve().parent.parent

    # Web demo
    demo_path = base_dir / "demo.html"
    docs_index_path = base_dir / "docs" / "index.html"
    assert demo_path.exists() or docs_index_path.exists(), "Interactive showcase demo is missing"

    # Notebook
    nb_path = base_dir / "notebooks" / "01_system_benchmark.ipynb"
    assert nb_path.exists(), "Benchmark notebook is missing"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    assert len(nb_json.get("cells", [])) >= 6


def test_pydantic_contracts_roundtrip_integrity(tmp_path: Path):
    """Verifies that all inter-stage data contracts (Contracts 1-5) serialize and deserialize cleanly."""
    state = PipelineState(
        session_id="m12_validation_session",
        input_audio_path="data/controlled_eval_suite/synthetic_mixtures/mix_snr0_overlap25.wav",
        audio_metadata=AudioMetadata(
            source_file="test.wav",
            sample_rate_hz=16000,
            channels=1,
            duration_seconds=5.0,
            rms_energy_db=-20.0,
            snr_estimate_db=35.0,
        ),
        separation_result=SeparationResult(
            input_audio_path="test.wav",
            num_sources_detected=2,
            separation_model="htdemucs",
            isolated_tracks=[
                SeparatedTrack(source_index=0, track_path="stem0.wav", energy_ratio=0.6, peak_amplitude=0.8),
                SeparatedTrack(source_index=1, track_path="stem1.wav", energy_ratio=0.4, peak_amplitude=0.6),
            ],
        ),
        diarization_timeline=DiarizationTimeline(
            audio_file="test.wav",
            algorithm="pyannote_spectral",
            total_speakers_detected=2,
            segments=[
                DiarizationSegment(segment_id=0, speaker_id="SPEAKER_00", start_time=0.0, end_time=2.5),
                DiarizationSegment(segment_id=1, speaker_id="SPEAKER_01", start_time=2.5, end_time=5.0),
            ],
        ),
        raw_transcript=RawTranscript(
            pipeline_stage="ASR",
            model_name="indic_wav2vec",
            utterances=[
                Utterance(utterance_id=0, speaker_id="SPEAKER_00", start_time=0.0, end_time=2.5, raw_text="नमस्कार किसान भाइयों"),
                Utterance(utterance_id=1, speaker_id="SPEAKER_01", start_time=2.5, end_time=5.0, raw_text="फसल की सुरक्षा के लिए जानकारी दें"),
            ],
        ),
        final_report=PostProcessedReport(
            metadata={"session_id": "m12_validation_session", "total_speakers": 2},
            executive_summary="कृषि विशेषज्ञों ने फसलों की सुरक्षा के लिए महत्वपूर्ण मार्गदर्शन प्रदान किया।",
            primary_topics=["कृषि मार्गदर्शन", "फसल सुरक्षा"],
            keyword_tags=["फसल", "कृषि", "सुरक्षा"],
            key_findings=[
                KeyFinding(topic="फसल सुरक्षा", detail="समय पर कीटनाशक और जल प्रबंधन आवश्यक है", speaker_attribution="SPEAKER_00"),
            ],
            speaker_separated_dialogue=[
                SpeakerDialogue(speaker="SPEAKER_00", timestamp="00:00 - 00:02", cleaned_text="नमस्कार किसान भाइयों।"),
                SpeakerDialogue(speaker="SPEAKER_01", timestamp="00:02 - 00:05", cleaned_text="फसल की सुरक्षा के लिए जानकारी दें।"),
            ],
        ),
    )

    out_file = tmp_path / "test_state.json"
    state.save_json(str(out_file))
    assert out_file.exists()

    loaded_state = PipelineState.load_json(str(out_file))
    assert loaded_state.session_id == state.session_id
    assert loaded_state.audio_metadata.duration_seconds == 5.0
    assert loaded_state.separation_result.num_sources_detected == 2
    assert loaded_state.diarization_timeline.total_speakers_detected == 2
    assert len(loaded_state.raw_transcript.utterances) == 2
    assert len(loaded_state.final_report.key_findings) == 1

