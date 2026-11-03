"""
Unit Tests for Stage 4: Semantic Post-Processing & LLM Guardrails (Milestone M7).
"""

import json
from pathlib import Path
import pytest

from src.core.state import (
    KeyFinding,
    PipelineState,
    PostProcessedReport,
    RawTranscript,
    SpeakerDialogue,
    Utterance,
)
from src.llm.llm_engine import LLMEngine
from src.llm.provider import (
    BaseLLMProvider,
    GroqProvider,
    MockLLMProvider,
    OpenAIProvider,
    get_llm_provider,
)
from evaluation.llm_metrics import LLMMetricsEvaluator


@pytest.fixture
def sample_raw_transcript() -> RawTranscript:
    """Provides a realistic bilingual Hinglish transcript matching Stage 3 output."""
    return RawTranscript(
        pipeline_stage="ASR",
        model_name="indic_wav2vec",
        utterances=[
            Utterance(
                utterance_id=0,
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=4.71,
                detected_language="hi",
                raw_text="s उन् s ह s ें s s द s ि s न s s त s क s s र s ी s ह s ा s ब s s क s र s न s ा s s ह s ो s ग s ा s s औ s र s s उ s स s क s े s s ब s ा s द s s उ s न s क s ा s s फ s ि s ट s न s े s स s s फ s ै s स s s लि s य s ा s s ज s ा s ए s ग s ा s s",
                confidence_score=0.92,
            ),
            Utterance(
                utterance_id=1,
                speaker_id="SPEAKER_01",
                start_time=3.54,
                end_time=9.54,
                detected_language="hi",
                raw_text="s य s े s s ए s क s s व s ॉ s इ s स s s ऑ s ड s ि s य s ो s ड s े s ट s ा s s स s े s ट s ह s ै s s ज s ि s स s क s ा s s उ s प s य s ो s ग s s आ s प s s आ s र s ् s ट s ि s फ s ि s शिय s ल s s इ s ं s ट s े s ल s ि s ज s े s ं s स s s ए s आ s ई s s क s े s s प s ्र s श s ि s क् s ष s ण के s",
                confidence_score=0.88,
            ),
        ],
    )


def test_post_processed_report_schema():
    """Validates Pydantic schema contract for PostProcessedReport."""
    report = PostProcessedReport(
        metadata={"session_id": "test_01", "total_speakers": 2, "duration_seconds": 9.54},
        executive_summary="Two speakers discuss medical rehabilitation and speech AI datasets.",
        primary_topics=["Rehabilitation", "AI Datasets"],
        keyword_tags=["rehab", "dataset"],
        key_findings=[
            KeyFinding(
                topic="Rehab Duration",
                detail="Ten-day rehab required before fitness test.",
                speaker_attribution="SPEAKER_00",
            )
        ],
        speaker_separated_dialogue=[
            SpeakerDialogue(
                speaker="SPEAKER_00",
                timestamp="00:00.00 - 00:04.71",
                cleaned_text="उन्हें दस दिन तक rehab करना होगा।",
            )
        ],
    )

    json_str = report.model_dump_json()
    assert "Two speakers discuss" in json_str
    reconstructed = PostProcessedReport.model_validate_json(json_str)
    assert reconstructed.primary_topics == ["Rehabilitation", "AI Datasets"]
    assert reconstructed.key_findings[0].speaker_attribution == "SPEAKER_00"


def test_prompt_assembly_and_guardrails(sample_raw_transcript):
    """Verifies Jinja2 prompt compilation and presence of 4 negative guardrails."""
    engine = LLMEngine(provider_type="mock")
    engine.initialize()

    prompt = engine.assemble_prompt(sample_raw_transcript, session_id="test_session_123")
    assert "[INPUT TRANSCRIPT]" in prompt
    assert "test_session_123" in prompt
    assert "STRICT TRUTHFULNESS & SOURCE GROUNDING" in prompt
    assert "PRESERVE SPEAKER ROLES & ATTRIBUTION INTEGRITY" in prompt
    assert "PHONETIC REPAIR WITHOUT CONTENT INVENTION" in prompt
    assert "DETERMINISTIC JSON SCHEMA ENFORCEMENT" in prompt


def test_mock_llm_provider_generation(sample_raw_transcript):
    """Tests deterministic generation from MockLLMProvider."""
    provider = MockLLMProvider()
    provider.initialize()

    engine = LLMEngine(provider_type="mock")
    engine.initialize()
    prompt = engine.assemble_prompt(sample_raw_transcript)

    raw_json = provider.generate_structured_report(prompt)
    report_dict = json.loads(raw_json)
    report = PostProcessedReport(**report_dict)

    assert len(report.key_findings) >= 2
    assert len(report.speaker_separated_dialogue) == 2
    assert "SPEAKER_00" in [d.speaker for d in report.speaker_separated_dialogue]
    assert "SPEAKER_01" in [d.speaker for d in report.speaker_separated_dialogue]
    assert "rehab" in report.executive_summary.lower() or "rehabilitation" in report.executive_summary.lower()


def test_llm_provider_factory():
    """Verifies get_llm_provider handles valid types and graceful defaults."""
    mock_p = get_llm_provider("mock")
    assert isinstance(mock_p, MockLLMProvider)

    groq_p = get_llm_provider("groq", offline_fallback=True)
    assert isinstance(groq_p, GroqProvider)

    openai_p = get_llm_provider("openai", offline_fallback=True)
    assert isinstance(openai_p, OpenAIProvider)

    fallback_p = get_llm_provider("unknown_provider")
    assert isinstance(fallback_p, MockLLMProvider)


def test_llm_engine_lifecycle(sample_raw_transcript):
    """Verifies LLMEngine conforms to BaseModule and updates PipelineState."""
    engine = LLMEngine(provider_type="mock")
    engine.initialize()

    state = PipelineState(
        session_id="session_lifecycle_test",
        input_audio_path="data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav",
        raw_transcript=sample_raw_transcript,
    )

    updated_state = engine.process(state)
    assert updated_state.final_report is not None
    assert updated_state.final_report.metadata["total_speakers"] == 2
    assert "stage4_llm" in updated_state.execution_telemetry
    assert updated_state.execution_telemetry["stage4_llm"]["status"] == "SUCCESS"


def test_hallucination_rate_evaluator(sample_raw_transcript):
    """Verifies that hallucination rate is 0.0% on grounded reports and detects hallucinations."""
    evaluator = LLMMetricsEvaluator()
    engine = LLMEngine(provider_type="mock")
    report = engine.post_process_transcript(sample_raw_transcript)

    raw_text = " ".join(u.raw_text for u in sample_raw_transcript.utterances)
    metrics = evaluator.compute_hallucination_rate(report, raw_text)

    # Clean report should have 0.0% hallucination rate
    assert metrics["hallucination_rate_percent"] == 0.0
    assert metrics["is_hallucination_free"] is True

    # Inject an invented claim / number (hallucination)
    hallucinated_summary = "SPEAKER_00 claimed the patient suffered from severe diabetes and required $95000 in cardiac surgery."
    hallucinated_metrics = evaluator.compute_hallucination_rate(hallucinated_summary, raw_text)
    assert hallucinated_metrics["hallucination_rate_percent"] > 0.0
    assert hallucinated_metrics["is_hallucination_free"] is False


def test_attribution_integrity_evaluator(sample_raw_transcript):
    """Verifies detection of invalid speaker attribution."""
    evaluator = LLMMetricsEvaluator()
    engine = LLMEngine(provider_type="mock")
    report = engine.post_process_transcript(sample_raw_transcript)

    result = evaluator.compute_attribution_integrity(report, sample_raw_transcript)
    assert result["attribution_integrity_percent"] == 100.0
    assert result["is_attribution_valid"] is True

    # Corrupt speaker attribution with non-existent speaker
    corrupted_report = report.model_copy(deep=True)
    corrupted_report.key_findings.append(
        KeyFinding(topic="False Attribution", detail="Random claim", speaker_attribution="SPEAKER_99")
    )
    corrupted_result = evaluator.compute_attribution_integrity(corrupted_report, sample_raw_transcript)
    assert corrupted_result["attribution_integrity_percent"] < 100.0
    assert corrupted_result["is_attribution_valid"] is False
    assert "SPEAKER_99" in corrupted_result["invalid_speaker_labels"]


def test_rouge_metrics_calculation():
    """Verifies mathematical correctness of ROUGE-1, ROUGE-2, and ROUGE-L."""
    evaluator = LLMMetricsEvaluator()
    text1 = "उन्हें दस दिन तक rehab करना होगा"
    text2 = "उन्हें दस दिन तक rehab करना होगा"

    # Identical texts -> F1 should be 1.0
    scores = evaluator.compute_rouge(text1, text2)
    assert scores["rouge_1"]["f1"] == 1.0
    assert scores["rouge_2"]["f1"] == 1.0
    assert scores["rouge_l"]["f1"] == 1.0

    # Non-identical texts
    text3 = "उन्हें बीस दिन तक अस्पताल में रुकना होगा"
    partial_scores = evaluator.compute_rouge(text1, text3)
    assert 0.0 < partial_scores["rouge_1"]["f1"] < 1.0
    assert partial_scores["rouge_2"]["f1"] < partial_scores["rouge_1"]["f1"]


def test_markdown_report_rendering(sample_raw_transcript):
    """Verifies rendering of the publication-quality Markdown report."""
    engine = LLMEngine(provider_type="mock")
    report = engine.post_process_transcript(sample_raw_transcript)

    md = engine.render_markdown_report(report)
    assert "# 📋 Executive Conversational Intelligence Report" in md
    assert "## 1. Executive Summary" in md
    assert "## 2. Thematic Analysis & Tags" in md
    assert "## 3. Key Findings & Speaker Attributions" in md
    assert "## 4. Polished Speaker Dialogue Timeline" in md
    assert "SPEAKER_00" in md
    assert "SPEAKER_01" in md


def test_save_report_artifacts(tmp_path, sample_raw_transcript):
    """Verifies that final_report.json and final_report.md are created correctly."""
    engine = LLMEngine(provider_type="mock")
    report = engine.post_process_transcript(sample_raw_transcript)

    json_path, md_path = engine.save_report_artifacts(report, tmp_path, file_prefix="test_report")
    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["executive_summary"] == report.executive_summary

