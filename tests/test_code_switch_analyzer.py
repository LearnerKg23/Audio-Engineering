"""Unit tests for src/asr/code_switch_analyzer.py."""

import pytest

from src.asr.code_switch_analyzer import (
    analyze_code_switching,
    analyze_transcript_corpus,
)
from src.core.state import RawTranscript, Utterance


def test_code_switch_transitions_and_ratios():
    """Verifies Devanagari/Latin token counts, switch boundaries, and ratios."""
    sample = "आज market में wheat का price देखना है"
    stats = analyze_code_switching(sample)

    assert stats.total_tokens == 8
    assert stats.devanagari_tokens == 5
    assert stats.latin_tokens == 3
    assert stats.code_switch_transitions == 6
    assert 0.0 < stats.latin_ratio < 1.0
    assert len(stats.sample_switch_pairs) <= stats.code_switch_transitions


def test_hyphenated_and_zwj_detection():
    """Verifies detection of intra-word hyphens and ZWJ/ZWNJ joiners."""
    sample = "e-rickshaw और प्र\u200dयोग day-to-day\u200ccheck"
    stats = analyze_code_switching(sample)

    assert stats.hyphenated_tokens == 2
    assert stats.zwj_zwnj_count == 2
    assert stats.mixed_script_tokens == 0


def test_empty_and_invalid_inputs():
    """Verifies empty string handling and strict input validation."""
    empty_stats = analyze_code_switching("   ")
    assert empty_stats.total_tokens == 0
    assert empty_stats.code_switch_transitions == 0
    assert empty_stats.latin_ratio == 0.0

    with pytest.raises(TypeError):
        analyze_code_switching(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        analyze_code_switching("test", max_samples=-1)


def test_corpus_aggregation_over_raw_transcript():
    """Verifies corpus-level aggregation across multiple utterances."""
    transcript = RawTranscript(
        utterances=[
            Utterance(
                utterance_id=0,
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=2.0,
                raw_text="किसान को subsidy मिली",
            ),
            Utterance(
                utterance_id=1,
                speaker_id="SPEAKER_01",
                start_time=2.2,
                end_time=4.5,
                raw_text="bank account में transfer हुआ",
            ),
        ]
    )

    stats = analyze_transcript_corpus(transcript)
    assert stats.total_tokens == 9
    assert stats.latin_tokens == 4
    assert stats.devanagari_tokens == 5
