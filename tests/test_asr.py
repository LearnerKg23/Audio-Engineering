"""
Unit Tests for Stage 3: Regional & Code-Switched ASR.

Validates:
1. IndicTextNormalizer (Unicode NFKC, Devanagari nukta harmonization, punctuation, lowercasing).
2. WER and CER calculation with Wagner-Fischer edit distance decomposition.
3. ASREngine BaseModule lifecycle and waveform transcription.
4. Diarization segment slicing with 50ms plosive safety padding.
5. End-to-end PipelineState contract enrichment.
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from evaluation.asr_metrics import (
    compute_cer,
    compute_corpus_metrics,
    compute_levenshtein_alignment,
    compute_wer,
)
from src.asr.asr_engine import ASREngine
from src.asr.text_normalizer import (
    IndicTextNormalizer,
    normalize_indic_text,
    tokenize_indic_characters,
    tokenize_indic_words,
)
from src.core.state import (
    AudioMetadata,
    DiarizationSegment,
    DiarizationTimeline,
    PipelineState,
    RawTranscript,
    SeparatedTrack,
    SeparationResult,
    Utterance,
)


# ==============================================================================
# 1. Indic Text Normalizer Tests
# ==============================================================================


def test_indic_normalizer_nfkc_and_nuktas():
    """Validates Unicode NFKC and Devanagari nukta harmonization."""
    normalizer = IndicTextNormalizer(compose_nuktas=True)

    # Decomposed nukta: क (U+0915) + nukta (U+093C)
    decomposed = "\u0915\u093c"
    # Composed nukta: क़ (U+0958)
    composed = "\u0958"

    assert normalizer.normalize(decomposed) == composed
    assert normalizer.normalize(composed) == composed


def test_indic_normalizer_punctuation_and_case():
    """Validates punctuation stripping and lowercase standardization for Hinglish."""
    normalizer = IndicTextNormalizer()

    raw_text = "नमस्कार! किसान भाइयों, आज MSP का Rate क्या है? (2026)"
    expected = "नमस्कार किसान भाइयों आज msp का rate क्या है 2026"
    normalized = normalizer.normalize(raw_text)

    assert normalized == expected


def test_indic_tokenization():
    """Validates word and character token extraction."""
    text = "आज market में wheat का price ₹2500 है।"
    words = tokenize_indic_words(text)
    chars = tokenize_indic_characters(text)

    assert "market" in words
    assert "wheat" in words
    assert "price" in words
    assert "2500" in words
    assert len(words) == 8
    assert len(chars) > len(words)
    assert not any(c.isspace() for c in chars)


# ==============================================================================
# 2. WER & CER Metric Tests
# ==============================================================================


def test_wer_identical_strings():
    """Identical reference and hypothesis must return 0% WER and CER."""
    ref = "उन्हें दस दिन तक rehab करना होगा"
    hyp = "उन्हें दस दिन तक rehab करना होगा"

    res_w = compute_wer(ref, hyp)
    res_c = compute_cer(ref, hyp)

    assert res_w["wer"] == 0.0
    assert res_w["wer_percent"] == 0.0
    assert res_w["substitutions"] == 0
    assert res_w["deletions"] == 0
    assert res_w["insertions"] == 0
    assert res_w["hits"] == 7

    assert res_c["cer"] == 0.0
    assert res_c["cer_percent"] == 0.0


def test_wer_error_decomposition():
    """Validates substitution, deletion, and insertion counting."""
    # 1. Pure substitution
    ref_sub = "यह एक वॉइस डेटासेट है"
    hyp_sub = "यह एक सुंदर डेटासेट है"
    res_sub = compute_wer(ref_sub, hyp_sub)
    assert res_sub["substitutions"] == 1
    assert res_sub["deletions"] == 0
    assert res_sub["insertions"] == 0

    # 2. Pure deletion
    ref_del = "यह एक वॉइस डेटासेट है"
    hyp_del = "यह एक वॉइस है"
    res_del = compute_wer(ref_del, hyp_del)
    assert res_del["substitutions"] == 0
    assert res_del["deletions"] == 1
    assert res_del["insertions"] == 0

    # 3. Pure insertion
    ref_ins = "यह एक वॉइस डेटासेट है"
    hyp_ins = "यह एक नया वॉइस डेटासेट है"
    res_ins = compute_wer(ref_ins, hyp_ins)
    assert res_ins["substitutions"] == 0
    assert res_ins["deletions"] == 0
    assert res_ins["insertions"] == 1

    # 4. Mixed: 1 sub, 1 del
    ref_mix = "यह एक वॉइस अच्छा डेटासेट"
    hyp_mix = "यह एक सुंदर डेटासेट"
    res_mix = compute_wer(ref_mix, hyp_mix)
    assert res_mix["substitutions"] == 1
    assert res_mix["deletions"] == 1
    assert res_mix["insertions"] == 0
    assert res_mix["ref_words"] == 5
    assert res_mix["wer"] == 0.4


def test_wer_cer_empty_edge_cases():
    """Validates edge cases when reference or hypothesis is empty."""
    res_empty_ref = compute_wer("", "extra words")
    assert res_empty_ref["wer"] == 1.0
    assert res_empty_ref["insertions"] == 2

    res_empty_hyp = compute_wer("missing words", "")
    assert res_empty_hyp["wer"] == 1.0
    assert res_empty_hyp["deletions"] == 2

    res_both_empty = compute_wer("", "")
    assert res_both_empty["wer"] == 0.0


def test_corpus_metrics():
    """Validates macro and micro corpus-level metric aggregation."""
    refs = [
        "उन्हें दस दिन तक rehab करना होगा",
        "यह एक वॉइस ऑडियो डेटासेट है",
    ]
    hyps = [
        "उन्हें दस दिन तक rehab करना होगा",
        "यह एक सुंदर ऑडियो डेटासेट",
    ]

    corpus = compute_corpus_metrics(refs, hyps)
    assert corpus["total_samples"] == 2
    assert corpus["macro_wer_percent"] > 0.0
    assert corpus["micro_wer_percent"] > 0.0
    assert corpus["total_ref_words"] == 7 + 6


# ==============================================================================
# 3. ASREngine Lifecycle & Slicing Tests
# ==============================================================================


def test_asr_engine_mock_mode_lifecycle():
    """Validates ASREngine lifecycle in mock mode."""
    engine = ASREngine(mock_mode=True)
    assert not engine._is_initialized

    engine.initialize()
    assert engine._is_initialized
    assert "mock" in engine.active_model_name

    # Transcribe synthetic waveform
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    synthetic_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    text, conf = engine.transcribe_waveform(synthetic_audio, sample_rate=sr)
    assert len(text) > 0
    assert conf > 0.8


def test_asr_engine_silent_slice():
    """Validates that silent slices return empty transcript."""
    engine = ASREngine(mock_mode=True)
    engine.initialize()

    silent_audio = np.zeros(16000, dtype=np.float32)
    text, conf = engine.transcribe_waveform(silent_audio, sample_rate=16000)
    assert text == ""
    assert conf == 0.0


def test_asr_engine_pipeline_state_integration(tmp_path: Path):
    """
    Validates complete pipeline state ingestion:
    DiarizationTimeline + SeparationResult -> RawTranscript with Utterance attribution.
    """
    # 1. Create temporary synthetic audio files
    sr = 16000
    dur = 4.0
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    audio_data = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    mix_path = tmp_path / "test_mix.wav"
    stem0_path = tmp_path / "test_stem0.wav"
    stem1_path = tmp_path / "test_stem1.wav"

    sf.write(str(mix_path), audio_data, sr)
    sf.write(str(stem0_path), audio_data, sr)
    sf.write(str(stem1_path), audio_data, sr)

    # 2. Build PipelineState with Stage 1 and Stage 2 artifacts
    state = PipelineState(
        input_audio_path=str(mix_path),
        audio_metadata=AudioMetadata(
            source_file=str(mix_path),
            duration_seconds=dur,
            sample_rate_hz=sr,
        ),
        separation_result=SeparationResult(
            input_audio_path=str(mix_path),
            num_sources_detected=2,
            isolated_tracks=[
                SeparatedTrack(source_index=0, track_path=str(stem0_path), energy_ratio=0.5),
                SeparatedTrack(source_index=1, track_path=str(stem1_path), energy_ratio=0.5),
            ],
        ),
        diarization_timeline=DiarizationTimeline(
            audio_file=str(mix_path),
            total_speakers_detected=2,
            segments=[
                DiarizationSegment(segment_id=0, speaker_id="SPEAKER_00", start_time=0.0, end_time=1.8),
                DiarizationSegment(segment_id=1, speaker_id="SPEAKER_01", start_time=2.0, end_time=3.8),
            ],
        ),
    )

    # 3. Process with ASREngine
    asr_engine = ASREngine(mock_mode=True, safety_padding_ms=50.0)
    updated_state = asr_engine(state)

    # 4. Assert Contract 4 compliance
    assert updated_state.raw_transcript is not None
    assert updated_state.raw_transcript.pipeline_stage == "ASR"
    assert len(updated_state.raw_transcript.utterances) == 2

    u0 = updated_state.raw_transcript.utterances[0]
    u1 = updated_state.raw_transcript.utterances[1]

    assert u0.speaker_id == "SPEAKER_00"
    assert u0.start_time == 0.0
    assert u0.end_time == 1.8
    assert len(u0.raw_text) > 0

    assert u1.speaker_id == "SPEAKER_01"
    assert u1.start_time == 2.0
    assert u1.end_time == 3.8
    assert len(u1.raw_text) > 0

    # Assert telemetry
    assert "ASREngine" in updated_state.execution_telemetry
    assert "ASREngine_metrics" in updated_state.execution_telemetry
    assert updated_state.execution_telemetry["ASREngine_metrics"]["utterances_count"] == 2
