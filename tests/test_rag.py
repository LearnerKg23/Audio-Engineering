"""
Unit & Integration Tests for Stage 4 Retrieval-Augmented Generation (RAG) Engine.
"""

from pathlib import Path
import pytest

from src.core.state import (
    AudioMetadata,
    DiarizationSegment,
    DiarizationTimeline,
    PipelineState,
    RAGAnswer,
    RawTranscript,
    Utterance,
)
from src.llm.rag_engine import ConversationRAGEngine
from src.pipeline import SpeechEngineeringPipeline


@pytest.fixture
def sample_transcript() -> RawTranscript:
    return RawTranscript(
        pipeline_stage="ASR",
        model_name="indic_wav2vec",
        utterances=[
            Utterance(
                utterance_id=0,
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=3.8,
                detected_language="hi-en",
                raw_text="उन्हें दस दिन तक rehab करना होगा और उसके बाद उनका fitness test लिया जाएगा",
                confidence_score=0.95,
            ),
            Utterance(
                utterance_id=1,
                speaker_id="SPEAKER_01",
                start_time=3.8,
                end_time=8.4,
                detected_language="hi",
                raw_text="इस योजना के अंतर्गत सभी महिलाओं को विशेष वित्तीय सहायता दी जाएगी",
                confidence_score=0.92,
            ),
            Utterance(
                utterance_id=2,
                speaker_id="SPEAKER_02",
                start_time=8.4,
                end_time=12.5,
                detected_language="hi-en",
                raw_text="smartphone में विशेष तौर से इसी काम के लिए बने hardware और software का इस्तेमाल किया गया है",
                confidence_score=0.94,
            ),
        ],
    )


def test_rag_grounded_query_success(sample_transcript: RawTranscript):
    """Verifies that queries matching transcript content return grounded answers with citations."""
    engine = ConversationRAGEngine()
    engine.index_transcript(sample_transcript)

    # Query about rehab
    ans: RAGAnswer = engine.query("What was said about rehab and fitness?")
    assert ans.is_grounded is True
    assert "SPEAKER_00" in ans.cited_speakers
    assert "rehab" in ans.answer.lower() or "fitness" in ans.answer.lower()
    assert ans.guardrail_status == "PASSED"
    assert ans.confidence > 0.5


def test_rag_speaker_attribution(sample_transcript: RawTranscript):
    """Verifies that asking about a specific speaker correctly attributes their statements."""
    engine = ConversationRAGEngine()
    engine.index_transcript(sample_transcript)

    ans: RAGAnswer = engine.query("What hardware and software was mentioned?")
    assert ans.is_grounded is True
    assert "SPEAKER_02" in ans.cited_speakers
    assert "smartphone" in ans.answer.lower() or "hardware" in ans.answer.lower()


def test_rag_negative_guardrail_rejection(sample_transcript: RawTranscript):
    """Verifies that asking about ungrounded external entities triggers the negative guardrail."""
    engine = ConversationRAGEngine()
    engine.index_transcript(sample_transcript)

    # Completely unmentioned topic
    ans: RAGAnswer = engine.query("What is the current temperature and weather forecast in Mumbai?")
    assert ans.is_grounded is False
    assert ans.confidence == 0.0
    assert ans.guardrail_status == "NEGATIVE_CONSTRAINT_TRIGGERED"
    assert "Grounding Guardrail" in ans.answer
    assert "Speculation prevented" in ans.answer


def test_pipeline_answer_question_integration(sample_transcript: RawTranscript):
    """Verifies that SpeechEngineeringPipeline.answer_question executes and appends to PipelineState."""
    pipeline = SpeechEngineeringPipeline(mock_mode=True)
    state = PipelineState(
        session_id="rag_test_session",
        input_audio_path="test.wav",
        raw_transcript=sample_transcript,
    )

    ans = pipeline.answer_question("Tell me about financial assistance for women", state)
    assert ans.is_grounded is True
    assert "SPEAKER_01" in ans.cited_speakers
    assert len(state.rag_answers) == 1
    assert state.rag_answers[0].is_grounded is True
