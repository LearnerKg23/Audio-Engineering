"""
Unit Tests for Stage 2: Acoustic Speaker Diarization Module.
"""

from pathlib import Path
import numpy as np
import pytest

from evaluation.diarization_metrics import DiarizationEvaluator
from src.core.audio_io import save_audio
from src.core.state import DiarizationSegment, DiarizationTimeline, PipelineState, SeparatedTrack, SeparationResult
from src.diarization.clusterer import SpeakerClusterer
from src.diarization.diarizer import SpeakerDiarizer
from src.diarization.embedder import SpeakerEmbeddingExtractor


@pytest.fixture
def synthetic_dialogue_audio(tmp_path: Path):
    """
    Creates a 6.0 second synthetic two-speaker alternating dialogue:
    - 0.0s - 2.5s: Speaker 0 (low pitch harmonic series ~180 Hz)
    - 2.5s - 3.5s: Short transition / overlap
    - 3.5s - 6.0s: Speaker 1 (high pitch harmonic series ~420 Hz)
    """
    sr = 16000
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t, dtype=np.float32)

    # Speaker 0 (0.0 to 3.0s)
    spk0_mask = (t >= 0.0) & (t <= 3.0)
    audio[spk0_mask] = (
        0.5 * np.sin(2 * np.pi * 180.0 * t[spk0_mask])
        + 0.3 * np.sin(2 * np.pi * 360.0 * t[spk0_mask])
        + 0.2 * np.sin(2 * np.pi * 540.0 * t[spk0_mask])
    )

    # Speaker 1 (2.5 to 6.0s - 0.5s overlap between 2.5s and 3.0s)
    spk1_mask = (t >= 2.5) & (t <= 6.0)
    audio[spk1_mask] += (
        0.5 * np.sin(2 * np.pi * 420.0 * t[spk1_mask])
        + 0.3 * np.sin(2 * np.pi * 840.0 * t[spk1_mask])
        + 0.2 * np.sin(2 * np.pi * 1260.0 * t[spk1_mask])
    )

    # Peak normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9

    audio_file = tmp_path / "synthetic_dialogue.wav"
    save_audio(audio_file, audio, sample_rate=sr)
    return audio_file


def test_embedding_extractor_properties(synthetic_dialogue_audio):
    extractor = SpeakerEmbeddingExtractor(
        embedding_dim=192,
        window_duration=1.5,
        step_duration=0.5,
        sample_rate=16000,
    )
    timestamps, embeddings = extractor.extract_sliding_embeddings(synthetic_dialogue_audio)

    assert len(timestamps) > 0
    assert len(embeddings) == len(timestamps)
    assert embeddings.shape[1] == 192

    # Check unit-norm normalization
    norms = np.linalg.norm(embeddings, axis=1)
    for n in norms:
        assert np.isclose(n, 1.0, atol=1e-3)


def test_affinity_matrix_and_eigengap():
    clusterer = SpeakerClusterer(min_speakers=1, max_speakers=4)

    # Create two distinctly separated embedding clusters in 192-d space
    np.random.seed(42)
    cluster_a = np.random.randn(8, 192).astype(np.float32) + 5.0
    cluster_b = np.random.randn(8, 192).astype(np.float32) - 5.0
    embs = np.vstack([cluster_a, cluster_b])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    affinity = clusterer.compute_affinity_matrix(embs)
    assert affinity.shape == (16, 16)
    assert np.all(affinity >= 0.0)
    assert np.allclose(np.diag(affinity), 1.0)

    # Eigengap heuristic should clearly detect K=2
    estimated_k = clusterer.estimate_speaker_count_eigengap(affinity)
    assert estimated_k == 2


def test_spectral_and_gmm_clustering():
    clusterer_spectral = SpeakerClusterer(clustering_method="spectral")
    clusterer_gmm = SpeakerClusterer(clustering_method="gmm")

    np.random.seed(42)
    cluster_a = np.random.randn(10, 192).astype(np.float32) + 8.0
    cluster_b = np.random.randn(10, 192).astype(np.float32) - 8.0
    embs = np.vstack([cluster_a, cluster_b])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    labels_spec, k_spec = clusterer_spectral.cluster_spectral(embs, num_speakers=2)
    assert k_spec == 2
    assert len(set(labels_spec[:10])) == 1
    assert len(set(labels_spec[10:])) == 1
    assert labels_spec[0] != labels_spec[10]

    labels_gmm, k_gmm = clusterer_gmm.cluster_gmm(embs, num_speakers=2)
    assert k_gmm == 2
    assert len(set(labels_gmm[:10])) == 1
    assert len(set(labels_gmm[10:])) == 1
    assert labels_gmm[0] != labels_gmm[10]


def test_der_and_jer_metrics():
    evaluator = DiarizationEvaluator(collar_seconds=0.25)

    ref = [
        ("SPEAKER_00", 0.0, 3.0),
        ("SPEAKER_01", 3.0, 6.0),
    ]

    # Perfect hypothesis
    hyp_perfect = [
        ("SPEAKER_00", 0.0, 3.0),
        ("SPEAKER_01", 3.0, 6.0),
    ]
    res_perfect = evaluator.compute_der(ref, hyp_perfect)
    assert res_perfect["der_percent"] == 0.0
    assert res_perfect["missed_seconds"] == 0.0
    assert res_perfect["false_alarm_seconds"] == 0.0
    assert res_perfect["confusion_seconds"] == 0.0

    jer_perfect = evaluator.compute_jer(ref, hyp_perfect)
    assert jer_perfect["jer_percent"] == 0.0

    # Swapped labels (Hungarian algorithm should resolve permutation)
    hyp_swapped = [
        ("CLUSTER_B", 0.0, 3.0),
        ("CLUSTER_A", 3.0, 6.0),
    ]
    res_swapped = evaluator.compute_der(ref, hyp_swapped)
    assert res_swapped["der_percent"] == 0.0

    # Definite error hypothesis (Speaker 0 only throughout)
    hyp_err = [
        ("SPEAKER_00", 0.0, 6.0),
    ]
    res_err = evaluator.compute_der(ref, hyp_err)
    assert res_err["der_percent"] > 20.0
    assert res_err["confusion_seconds"] > 0.0


def test_speaker_diarizer_pipeline_state(synthetic_dialogue_audio):
    diarizer = SpeakerDiarizer(clustering_method="spectral")
    diarizer.initialize()

    state = PipelineState(input_audio_path=str(synthetic_dialogue_audio))
    updated_state = diarizer(state)

    assert updated_state.diarization_timeline is not None
    timeline = updated_state.diarization_timeline
    assert timeline.total_speakers_detected in [1, 2]
    assert len(timeline.segments) > 0
    assert "SpeakerDiarizer" in updated_state.execution_telemetry


def test_speaker_diarizer_separation_aware_mode(tmp_path: Path):
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    stem0 = (0.5 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    stem1 = (0.5 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)

    p0 = tmp_path / "stem_0.wav"
    p1 = tmp_path / "stem_1.wav"
    save_audio(p0, stem0, sr)
    save_audio(p1, stem1, sr)

    state = PipelineState(
        input_audio_path=str(tmp_path / "mix.wav"),
        separation_result=SeparationResult(
            input_audio_path=str(tmp_path / "mix.wav"),
            num_sources_detected=2,
            isolated_tracks=[
                SeparatedTrack(source_index=0, track_path=str(p0), energy_ratio=0.5),
                SeparatedTrack(source_index=1, track_path=str(p1), energy_ratio=0.5),
            ],
        ),
    )

    diarizer = SpeakerDiarizer(clustering_method="spectral")
    updated_state = diarizer(state)

    timeline = updated_state.diarization_timeline
    assert timeline is not None
    assert timeline.total_speakers_detected == 2
    speakers = set(s.speaker_id for s in timeline.segments)
    assert "SPEAKER_00" in speakers
    assert "SPEAKER_01" in speakers
