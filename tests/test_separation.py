"""
Unit tests for Speech Separation Modules (Demucs and Conv-TasNet).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.core.audio_io import save_audio
from src.core.state import PipelineState
from src.separation.conv_tasnet_wrapper import ConvTasNetSeparator
from src.separation.demucs_wrapper import DemucsSeparator


def test_compute_channel_energy_distribution():
    separator = ConvTasNetSeparator(device="cpu")
    active_track = np.full(16000, 0.1, dtype=np.float32)
    silent_track = np.zeros(16000, dtype=np.float32)

    distribution = separator.compute_channel_energy_distribution(
        [active_track, torch.from_numpy(silent_track)]
    )

    assert distribution[0]["channel_index"] == 0
    assert distribution[0]["rms_energy"] == pytest.approx(0.1)
    assert distribution[0]["active_speech_ratio"] == 1.0
    assert distribution[1]["rms_energy"] == 0.0
    assert distribution[1]["active_speech_ratio"] == 0.0


def test_compute_channel_energy_distribution_rejects_empty_track():
    separator = ConvTasNetSeparator(device="cpu")
    with pytest.raises(ValueError, match="must not be empty"):
        separator.compute_channel_energy_distribution([np.array([])])


@pytest.fixture
def short_mixture_file(tmp_path: Path):
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 2 distinct frequency tones representing concurrent talkers
    mix = (0.4 * np.sin(2 * np.pi * 300.0 * t) + 0.4 * np.sin(2 * np.pi * 900.0 * t)).astype(np.float32)

    mix_file = tmp_path / "test_mix_short.wav"
    save_audio(mix_file, mix, sample_rate=sr)
    return mix_file


def test_conv_tasnet_separation(tmp_path: Path, short_mixture_file):
    separator = ConvTasNetSeparator(device="cpu")
    separator.initialize()

    out_dir = tmp_path / "convtasnet_out"
    stems, meta = separator.separate_file(short_mixture_file, output_dir=out_dir)

    assert len(stems) == 2
    assert meta["num_stems"] == 2
    assert meta["model_name"] == "conv_tasnet_base_libri2mix"
    assert meta["inference_time_seconds"] > 0.0

    for s in meta["stems"]:
        stem_path = Path(s["file_path"])
        assert stem_path.exists()
        assert s["energy_ratio"] >= 0.0


def test_demucs_separation(tmp_path: Path, short_mixture_file):
    separator = DemucsSeparator(device="cpu", model_name="htdemucs")
    separator.initialize()

    out_dir = tmp_path / "demucs_out"
    stems, meta = separator.separate_file(short_mixture_file, output_dir=out_dir)

    assert len(stems) == 2
    assert meta["num_stems"] == 2
    assert meta["model_name"] == "htdemucs"
    assert meta["inference_time_seconds"] > 0.0

    for s in meta["stems"]:
        stem_path = Path(s["file_path"])
        assert stem_path.exists()
        assert s["energy_ratio"] >= 0.0


def test_separation_pipeline_state_integration(tmp_path: Path, short_mixture_file):
    separator = ConvTasNetSeparator(device="cpu")
    state = PipelineState(input_audio_path=str(short_mixture_file))

    updated_state = separator(state)
    assert updated_state.separation_result is not None
    assert updated_state.separation_result.num_sources_detected == 2
    assert len(updated_state.separation_result.isolated_tracks) == 2
    assert "ConvTasNetSeparator" in updated_state.execution_telemetry
