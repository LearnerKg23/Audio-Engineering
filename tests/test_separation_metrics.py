"""Tests for source-separation evaluation metrics."""

import numpy as np
import pytest
import torch

from src.separation.metrics import (
    calculate_sar,
    calculate_sdr,
    calculate_sir,
    calculate_sisdr,
)


@pytest.fixture
def orthogonal_sources():
    time = np.arange(128, dtype=np.float64)
    target = np.sin(2 * np.pi * time / 16)
    interference = np.cos(2 * np.pi * time / 16)
    artifact = np.sin(2 * np.pi * 13 * time / 128)
    return target, interference, artifact


def test_source_projection_components_are_orthogonal(orthogonal_sources):
    target, interference, artifact = orthogonal_sources
    estimate = 2.0 * target + 0.5 * interference + 0.25 * artifact

    # Orthogonal clean sources make the least-squares decomposition explicit.
    assert np.dot(target, interference) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(target, artifact) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(interference, artifact) == pytest.approx(0.0, abs=1e-12)
    assert calculate_sdr(
        target, estimate, interference_references=[interference]
    ) == pytest.approx(
        10
        * np.log10(
            (2.0**2 * np.dot(target, target))
            / (
                (0.5**2 * np.dot(interference, interference))
                + (0.25**2 * np.dot(artifact, artifact))
            )
        ),
        abs=1e-10,
    )
    assert calculate_sir(target, estimate, [interference]) == pytest.approx(
        10
        * np.log10(
            (2.0**2 * np.dot(target, target))
            / (0.5**2 * np.dot(interference, interference))
        ),
        abs=1e-10,
    )
    assert calculate_sar(target, estimate, [interference]) == pytest.approx(
        10
        * np.log10(
            (
                (2.0**2 * np.dot(target, target))
                + (0.5**2 * np.dot(interference, interference))
            )
            / (0.25**2 * np.dot(artifact, artifact))
        ),
        abs=1e-10,
    )


def test_zero_interference_and_artifact_scores_are_finite(orthogonal_sources):
    target, interference, _ = orthogonal_sources
    clean_estimate = 3.0 * target

    assert calculate_sir(target, clean_estimate, [interference]) == 100.0
    assert calculate_sdr(target, clean_estimate, interference_references=[interference]) == 100.0
    assert calculate_sar(target, clean_estimate, [interference]) == 100.0


def test_sisdr_is_scale_invariant_and_accepts_torch_tensors(orthogonal_sources):
    target, interference, _ = orthogonal_sources
    estimate = target + 0.1 * interference

    numpy_score = calculate_sisdr(target, estimate)
    torch_score = calculate_sisdr(
        torch.tensor(target), torch.tensor(estimate * 5.0)
    )
    assert torch_score == pytest.approx(numpy_score, abs=1e-10)


def test_metrics_reject_mismatched_signal_lengths():
    with pytest.raises(ValueError, match="same number of samples"):
        calculate_sdr(np.ones(4), np.ones(3))


def test_sisdr_zero_reference_uses_lower_score_limit():
    assert calculate_sisdr(np.zeros(16), np.ones(16)) == -100.0
