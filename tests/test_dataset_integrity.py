"""Integrity checks for the Track A and Track B audio datasets."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import soundfile as sf

from src.utils.audio import inspect_audio_metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRACK_A_MANIFEST = REPOSITORY_ROOT / "data/manifests/controlled_ground_truth.json"
TRACK_B_MANIFEST = REPOSITORY_ROOT / "data/manifests/real_world_manifest.json"
EXPECTED_SAMPLE_RATE_HZ = 16000


def _read_utf8_manifest(path: Path) -> Dict[str, Any]:
    """Read a JSON manifest with strict UTF-8 decoding."""
    with path.open("r", encoding="utf-8", errors="strict") as manifest_file:
        return json.load(manifest_file)


def _assert_audio_file_is_valid(relative_path: str) -> None:
    """Assert a manifest audio path exists and has valid 16 kHz metadata."""
    audio_path = REPOSITORY_ROOT / relative_path
    assert audio_path.is_file(), f"Manifest audio file is missing: {audio_path}"

    metadata = inspect_audio_metadata(audio_path)
    assert metadata["duration_seconds"] > 0.0, f"Audio duration is not positive: {audio_path}"
    assert metadata["sample_rate_hz"] == EXPECTED_SAMPLE_RATE_HZ, (
        f"Expected {EXPECTED_SAMPLE_RATE_HZ} Hz audio, got "
        f"{metadata['sample_rate_hz']} Hz: {audio_path}"
    )


def test_track_a_manifest_audio_integrity():
    manifest = _read_utf8_manifest(TRACK_A_MANIFEST)

    assert manifest["sample_rate_hz"] == EXPECTED_SAMPLE_RATE_HZ
    mixtures = manifest["mixtures"]
    assert len(mixtures) == manifest["total_mixtures"]
    assert mixtures

    for mixture in mixtures:
        assert mixture["sample_rate_hz"] == EXPECTED_SAMPLE_RATE_HZ
        assert mixture["total_duration_seconds"] > 0.0
        _assert_audio_file_is_valid(mixture["mixture_file_path"])

        for source in mixture["sources"]:
            _assert_audio_file_is_valid(source["original_stem_path"])
            _assert_audio_file_is_valid(source["aligned_ground_truth_path"])


def test_track_b_audio_and_ground_truth_transcript_integrity():
    manifest = _read_utf8_manifest(TRACK_B_MANIFEST)

    assert manifest["sample_rate_hz"] == EXPECTED_SAMPLE_RATE_HZ
    clips = manifest["clips"]
    assert len(clips) == manifest["total_clips"]
    assert clips

    observed_non_ascii_transcript = False
    for clip in clips:
        assert clip["duration_seconds"] > 0.0
        _assert_audio_file_is_valid(clip["file_path"])
        assert clip["speaker_turns"], f"Missing ground-truth turns for {clip['clip_id']}"

        for turn in clip["speaker_turns"]:
            transcript = turn["transcript"]
            assert isinstance(transcript, str) and transcript.strip()
            encoded_transcript = transcript.encode("utf-8", errors="strict")
            assert encoded_transcript.decode("utf-8", errors="strict") == transcript
            assert "\ufffd" not in transcript
            observed_non_ascii_transcript |= any(ord(character) > 127 for character in transcript)

    assert observed_non_ascii_transcript, "Expected UTF-8 ground-truth text in Track B."


def test_audio_metadata_inspection_does_not_read_waveform(tmp_path: Path, monkeypatch):
    audio_path = tmp_path / "header_only.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), EXPECTED_SAMPLE_RATE_HZ)

    def fail_if_waveform_read(*args, **kwargs):
        pytest.fail("Metadata inspection should not decode the waveform.")

    monkeypatch.setattr(sf, "read", fail_if_waveform_read)
    metadata = inspect_audio_metadata(audio_path)

    assert metadata["sample_rate_hz"] == EXPECTED_SAMPLE_RATE_HZ
    assert metadata["frames"] == 1600
    assert metadata["duration_seconds"] == pytest.approx(0.1)


def test_audio_metadata_inspection_rejects_invalid_file(tmp_path: Path):
    invalid_audio_path = tmp_path / "invalid.wav"
    invalid_audio_path.write_bytes(b"not an audio file")

    with pytest.raises(ValueError, match="Could not inspect audio metadata"):
        inspect_audio_metadata(invalid_audio_path)
