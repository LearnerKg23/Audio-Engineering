"""
Unit & Integration Tests for Milestone M11:
Real-World Indian Audio Domain Robustness & Stress-Testing Suite (Track B).
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf

from evaluation.real_world_evaluator import (
    AcousticStressMetrics,
    DomainAggregateMetrics,
    RealWorldClipMetadata,
    RealWorldClipResult,
    RealWorldEvaluator,
    RealWorldValidationReport,
    SpeakerTurnReference,
    StageEvaluationMetrics,
)
from scripts.run_real_world_validation import (
    plot_acoustic_stress_radar,
    plot_code_switching_accuracy,
    plot_diarization_and_snr_robustness,
    plot_domain_wer_cer_comparison,
)
from src.core.audio_io import load_audio
from src.pipeline import SpeechEngineeringPipeline


def test_real_world_manifest_loading():
    """Validates loading and schema validation of Track B manifest."""
    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = base_dir / "data" / "manifests" / "real_world_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    evaluator = RealWorldEvaluator()
    clips = evaluator.load_manifest(manifest_path)

    assert len(clips) >= 3
    clip_ids = {c.clip_id for c in clips}
    assert "air_formal_news_01" in clip_ids
    assert "air_spontaneous_panel_01" in clip_ids
    assert "vaani_rural_call_01" in clip_ids

    for c in clips:
        assert c.duration_seconds > 0.0
        assert len(c.speaker_turns) >= 1
        audio_p = base_dir / c.file_path
        assert audio_p.exists(), f"Audio file does not exist: {audio_p}"


def test_acoustic_stress_metrics_extraction():
    """Validates acoustic stress extraction including cellular telephone bandpass ratio."""
    evaluator = RealWorldEvaluator()

    # 1. Synthetic clean tone at 1 kHz (inside telephone band [300, 3400] Hz)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio_1khz = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    stress_1k = evaluator.compute_acoustic_stress_features(audio_1khz, sr)
    assert stress_1k.cellular_bandpass_ratio > 0.80  # Most power is at 1 kHz
    assert stress_1k.clipping_ratio == 0.0
    assert abs(stress_1k.duration_seconds - 1.0) < 0.01

    # 2. Synthetic high-frequency tone at 6 kHz (outside telephone band)
    audio_6khz = (0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    stress_6k = evaluator.compute_acoustic_stress_features(audio_6khz, sr)
    assert stress_6k.cellular_bandpass_ratio < 0.20  # Power is outside telephone band


def test_script_token_classes_separation():
    """Validates decomposition of bilingual text into Devanagari Hindi and Latin Hinglish tokens."""
    evaluator = RealWorldEvaluator()

    sample_text = "smartphone में विशेष तौर से hardware और software का इस्तेमाल किया गया है।"
    hi_tokens, en_tokens = evaluator.extract_script_token_classes(sample_text)

    # Devanagari tokens
    assert "विशेष" in hi_tokens
    assert "तौर" in hi_tokens
    assert "इस्तेमाल" in hi_tokens

    # Latin tokens
    assert "smartphone" in en_tokens
    assert "hardware" in en_tokens
    assert "software" in en_tokens


def test_bilingual_wer_calculation():
    """Validates bilingual WER calculation decomposing Hindi and Latin error rates."""
    evaluator = RealWorldEvaluator()

    ref = "smartphone में विशेष तौर से hardware का इस्तेमाल किया गया है"
    # Hyp has one Latin error (softwares instead of hardware) and identical Hindi tokens
    hyp = "smartphone में विशेष तौर से softwares का इस्तेमाल किया गया है"

    res = evaluator.compute_bilingual_wer(ref, hyp)
    assert res["overall"]["wer_percent"] > 0.0
    assert res["hindi_tokens"]["wer_percent"] == 0.0
    assert res["hinglish_tokens"]["wer_percent"] > 0.0


def test_domain_entity_extraction():
    """Validates domain entity extraction across medical, technical, and regulatory keywords."""
    evaluator = RealWorldEvaluator()

    text_1 = "उन्हें दस दिन तक rehab करना होगा और उसके बाद उनका fitness test लिया जाएगा।"
    entities_1 = evaluator.extract_key_domain_entities(text_1)
    assert "rehab" in entities_1
    assert "fitness test" in entities_1

    text_2 = "non-stop flight के लिए carry bag में कोई भी liquid पदार्थ ले जाने से मना किया गया।"
    entities_2 = evaluator.extract_key_domain_entities(text_2)
    assert "non-stop flight" in entities_2
    assert "carry bag" in entities_2
    assert "liquid" in entities_2


def test_real_world_evaluator_clip_execution():
    """Tests evaluating a real-world clip (AIR studio news) with pipeline integration."""
    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = base_dir / "data" / "manifests" / "real_world_manifest.json"
    evaluator = RealWorldEvaluator()
    clips = evaluator.load_manifest(manifest_path)

    # Pick the clean studio news clip for quick lifecycle test
    news_clip = next(c for c in clips if c.clip_id == "air_formal_news_01")

    pipeline = SpeechEngineeringPipeline(mock_mode=True)
    result = evaluator.evaluate_clip(news_clip, pipeline, run_baseline=False)

    assert result.clip_id == "air_formal_news_01"
    assert result.proposed_pipeline.acoustic_stress.snr_estimate_db > 20.0
    assert result.proposed_pipeline.ref_speaker_count == 1
    assert result.proposed_pipeline.wer_percent >= 0.0
    assert result.proposed_pipeline.real_time_factor > 0.0


def test_real_world_plot_generation():
    """Verifies that all 4 domain robustness visualization figures are generated properly."""
    evaluator = RealWorldEvaluator()

    mock_stress = AcousticStressMetrics(
        duration_seconds=6.0,
        sample_rate_hz=16000,
        rms_energy_db=-18.0,
        peak_amplitude=0.8,
        snr_estimate_db=45.0,
        dynamic_range_db=30.0,
        clipping_ratio=0.0,
        dc_offset=0.0,
        cellular_bandpass_ratio=0.45,
        crest_factor_db=12.0,
    )

    mock_metrics = StageEvaluationMetrics(
        acoustic_stress=mock_stress,
        num_sources_detected=1,
        separation_executed=True,
        stem_energy_distribution=[1.0],
        ref_speaker_count=1,
        hyp_speaker_count=1,
        speaker_count_error=0,
        der_percent=4.2,
        jer_percent=5.1,
        missed_speech_seconds=0.1,
        false_alarm_seconds=0.0,
        confusion_seconds=0.0,
        wer_percent=12.5,
        cer_percent=6.2,
        substitutions=2,
        deletions=0,
        insertions=1,
        hindi_token_wer_percent=10.0,
        hinglish_token_wer_percent=15.0,
        hallucination_rate_percent=0.0,
        rouge_l_f1=0.85,
        total_elapsed_seconds=2.5,
        real_time_factor=0.416,
        throughput_x_realtime=2.4,
    )

    clip_res = RealWorldClipResult(
        clip_id="air_formal_news_01",
        domain="formal_studio_news",
        speech_style="formal_read_speech",
        primary_language="Hindi",
        proposed_pipeline=mock_metrics,
    )

    domain_agg = DomainAggregateMetrics(
        domain="formal_studio_news",
        total_duration_seconds=6.0,
        mean_snr_db=45.0,
        mean_der_percent=4.2,
        mean_wer_percent=12.5,
        mean_cer_percent=6.2,
        mean_hallucination_rate=0.0,
        mean_rtf=0.416,
        clip_count=1,
    )

    report = RealWorldValidationReport(
        total_clips_evaluated=1,
        total_audio_duration_seconds=6.0,
        domain_aggregates={"formal_studio_news": domain_agg},
        clip_results=[clip_res],
        code_switching_summary={
            "mean_hindi_devanagari_wer_percent": 10.0,
            "mean_hinglish_latin_wer_percent": 15.0,
            "code_switching_wer_disparity_delta": 5.0,
        },
        cellular_degradation_boundary={
            "telephone_bandpass_cutoff_hz": [300, 3400],
            "vaani_cellular_spectral_energy_ratio": 0.884,
            "air_studio_spectral_energy_ratio": 0.412,
            "observed_acoustic_attenuation_db": -6.3,
            "minimum_viable_snr_threshold_db": 21.0,
            "verdict": "Cellular bandpass does not trigger CTC collapse.",
        },
        failure_mode_occurrences={},
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        p1 = tmp_p / "wer_cer.png"
        plot_domain_wer_cer_comparison(report, p1)
        assert p1.exists()

        p2 = tmp_p / "diar_snr.png"
        plot_diarization_and_snr_robustness(report, p2)
        assert p2.exists()

        p3 = tmp_p / "code_switch.png"
        plot_code_switching_accuracy(report, p3)
        assert p3.exists()

        p4 = tmp_p / "radar.png"
        plot_acoustic_stress_radar(report, p4)
        assert p4.exists()
