"""
Unit & Integration Tests for Milestone M9:
Cross-Stage Error Propagation & Scientific Ablation Study.
"""

from pathlib import Path
import tempfile
import pytest

from evaluation.ablation_runner import (
    AblationExperimentRunner,
    AblationMetrics,
    PipelineVariant,
)
from evaluation.error_propagation import (
    ErrorCascadeDataPoint,
    ErrorCascadingAnalyzer,
)
from evaluation.failure_taxonomy import (
    CANONICAL_FAILURE_TAXONOMY,
    FailureCategory,
    FailureDiagnosticEngine,
)
from src.core.state import (
    DiarizationSegment,
    DiarizationTimeline,
    PipelineState,
    RawTranscript,
    SeparatedTrack,
    SeparationResult,
    Utterance,
)


@pytest.fixture
def test_audio_file() -> Path:
    base = Path(__file__).resolve().parent.parent
    path = base / "data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav"
    if not path.exists():
        wavs = list(base.glob("data/**/*.wav"))
        if wavs:
            return wavs[0]
        pytest.skip("No audio files available for testing.")
    return path


@pytest.fixture
def mock_reference_transcripts() -> dict:
    return {
        "SPEAKER_00": "उन्हें दस दिन तक rehab करना होगा",
        "SPEAKER_01": "यह एक वॉइस ऑडियो डेटासेट है",
    }


def test_pipeline_variant_enum():
    """Verifies that all 4 canonical ablation variants exist."""
    assert PipelineVariant.PIPELINE_A.value == "Pipeline A (Naive Baseline)"
    assert PipelineVariant.PIPELINE_B.value == "Pipeline B (Separation Only)"
    assert PipelineVariant.PIPELINE_C.value == "Pipeline C (Traditional Diarized)"
    assert PipelineVariant.PIPELINE_D.value == "Pipeline D (Full Proposed Cascaded)"


def test_ablation_variant_a_execution(test_audio_file: Path, mock_reference_transcripts: dict):
    """Verifies Pipeline A (Naive baseline) execution and metric computation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = AblationExperimentRunner(mock_mode=True)
        metrics, state = runner.run_variant(
            variant=PipelineVariant.PIPELINE_A,
            audio_path=test_audio_file,
            reference_transcripts=mock_reference_transcripts,
            output_dir=Path(tmp_dir) / "pipeline_a",
        )

        assert isinstance(metrics, AblationMetrics)
        assert metrics.pipeline_variant == PipelineVariant.PIPELINE_A.value
        # In Pipeline A, BSS and Diarization are bypassed
        assert metrics.stage1_si_sdr_db is None
        assert metrics.stage2_der_percent is None
        assert metrics.stage4_hallucination_rate is None
        assert metrics.stage3_wer_percent >= 0.0
        assert metrics.rtf >= 0.0


def test_ablation_variant_d_execution(test_audio_file: Path, mock_reference_transcripts: dict):
    """Verifies Pipeline D (Full proposed cascaded system) execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = AblationExperimentRunner(mock_mode=True)
        metrics, state = runner.run_variant(
            variant=PipelineVariant.PIPELINE_D,
            audio_path=test_audio_file,
            reference_transcripts=mock_reference_transcripts,
            output_dir=Path(tmp_dir) / "pipeline_d",
        )

        assert isinstance(metrics, AblationMetrics)
        assert metrics.pipeline_variant == PipelineVariant.PIPELINE_D.value
        assert state.separation_result is not None
        assert state.diarization_timeline is not None
        assert state.raw_transcript is not None
        assert state.final_report is not None
        assert metrics.stage4_hallucination_rate is not None
        assert metrics.stage4_attribution_integrity is not None


def test_error_cascading_correlations():
    """Verifies statistical cross-stage correlation calculations."""
    points = [
        ErrorCascadeDataPoint(
            condition_id="P1",
            scenario_description="Clean",
            overlap_ratio=0.25,
            snr_noise_db=None,
            si_sdr_db=12.0,
            der_percent=8.0,
            wer_percent=12.0,
            hallucination_rate_percent=0.0,
            attribution_integrity_percent=100.0,
        ),
        ErrorCascadeDataPoint(
            condition_id="P2",
            scenario_description="Moderate",
            overlap_ratio=0.50,
            snr_noise_db=15.0,
            si_sdr_db=6.0,
            der_percent=20.0,
            wer_percent=30.0,
            hallucination_rate_percent=2.0,
            attribution_integrity_percent=85.0,
        ),
        ErrorCascadeDataPoint(
            condition_id="P3",
            scenario_description="Severe",
            overlap_ratio=0.75,
            snr_noise_db=5.0,
            si_sdr_db=1.0,
            der_percent=45.0,
            wer_percent=60.0,
            hallucination_rate_percent=15.0,
            attribution_integrity_percent=55.0,
        ),
    ]

    analyzer = ErrorCascadingAnalyzer(points)
    correlations = analyzer.compute_cross_stage_correlations()

    # SI-SDR vs DER should be strongly negative
    assert correlations["si_sdr_vs_der"].pearson_r < -0.9
    # DER vs WER should be strongly positive
    assert correlations["der_vs_wer"].pearson_r > 0.9
    # WER vs Hallucination should be strongly positive
    assert correlations["wer_vs_hallucination"].pearson_r > 0.8
    # DER vs Attribution should be strongly negative
    assert correlations["der_vs_attribution"].pearson_r < -0.9


def test_critical_acoustic_threshold():
    """Verifies computation of critical breakdown threshold."""
    points = [
        ErrorCascadeDataPoint("P1", "Clean", 0.25, None, 12.0, 8.0, 15.0, 0.0, 100.0),
        ErrorCascadeDataPoint("P2", "Moderate", 0.50, 15.0, 7.0, 18.0, 25.0, 1.0, 90.0),
        ErrorCascadeDataPoint("P3", "Breakdown", 0.50, 10.0, 4.0, 30.0, 45.0, 6.0, 75.0),
        ErrorCascadeDataPoint("P4", "Severe", 0.50, 5.0, 1.5, 45.0, 65.0, 16.0, 50.0),
    ]

    analyzer = ErrorCascadingAnalyzer(points)
    thresh = analyzer.compute_critical_acoustic_threshold(wer_failure_ceiling=40.0, hallucination_failure_ceiling=5.0)

    assert thresh["si_sdr_crit_db"] == 4.0
    assert thresh["wer_at_threshold"] == 45.0
    assert thresh["hallucination_at_threshold"] == 6.0
    assert thresh["recommended_operational_margin_db"] == 7.0


def test_canonical_failure_taxonomy_integrity():
    """Verifies structure and categories of the canonical failure taxonomy."""
    assert len(CANONICAL_FAILURE_TAXONOMY) >= 8

    categories = {item.category for item in CANONICAL_FAILURE_TAXONOMY}
    assert FailureCategory.BSS_FAILURE in categories
    assert FailureCategory.DIARIZATION_FAILURE in categories
    assert FailureCategory.ASR_FAILURE in categories
    assert FailureCategory.LLM_FAILURE in categories

    # Verify fields
    for item in CANONICAL_FAILURE_TAXONOMY:
        assert item.failure_code.startswith("ERR_")
        assert item.name != ""
        assert item.algorithmic_root_cause != ""
        assert item.mitigation_strategy != ""
        assert item.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_failure_diagnostic_engine_detection():
    """Verifies that FailureDiagnosticEngine detects known acoustic/semantic anomalies."""
    # 1. State with Phantom Voice channel (one stem has 0.001 energy ratio)
    state_phantom = PipelineState(
        input_audio_path="test.wav",
        separation_result=SeparationResult(
            input_audio_path="test.wav",
            num_sources_detected=2,
            isolated_tracks=[
                SeparatedTrack(source_index=0, track_path="stem0.wav", energy_ratio=0.999),
                SeparatedTrack(source_index=1, track_path="stem1.wav", energy_ratio=0.001),
            ],
        ),
    )
    warnings = FailureDiagnosticEngine.inspect_state(state_phantom)
    assert any(w["failure_code"] == "ERR_BSS_02" for w in warnings)

    # 2. State with rapid speaker ping-ponging
    state_pingpong = PipelineState(
        input_audio_path="test.wav",
        diarization_timeline=DiarizationTimeline(
            audio_file="test.wav",
            total_speakers_detected=2,
            segments=[
                DiarizationSegment(segment_id=0, speaker_id="SPEAKER_00", start_time=0.0, end_time=0.3),
                DiarizationSegment(segment_id=1, speaker_id="SPEAKER_01", start_time=0.3, end_time=0.6),
                DiarizationSegment(segment_id=2, speaker_id="SPEAKER_00", start_time=0.6, end_time=0.9),
                DiarizationSegment(segment_id=3, speaker_id="SPEAKER_01", start_time=0.9, end_time=1.2),
                DiarizationSegment(segment_id=4, speaker_id="SPEAKER_00", start_time=1.2, end_time=1.5),
            ],
        ),
    )
    warnings_pp = FailureDiagnosticEngine.inspect_state(state_pingpong)
    assert any(w["failure_code"] == "ERR_DIAR_01" for w in warnings_pp)

    # 3. State with ASR repetition loop
    state_loop = PipelineState(
        input_audio_path="test.wav",
        raw_transcript=RawTranscript(
            pipeline_stage="ASR",
            model_name="indic_wav2vec",
            utterances=[
                Utterance(
                    utterance_id=0,
                    speaker_id="SPEAKER_00",
                    start_time=0.0,
                    end_time=3.0,
                    raw_text="है है है है है है है",
                )
            ],
        ),
    )
    warnings_loop = FailureDiagnosticEngine.inspect_state(state_loop)
    assert any(w["failure_code"] == "ERR_ASR_01" for w in warnings_loop)

